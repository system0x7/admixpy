"""Single-pulse admixture dating from within-individual ancestry covariance.

The ancestry score follows the two-reference construction in DATES v4010
(Chintalapati, Patterson & Moorjani, 2022, doi:10.7554/eLife.77625).
This is an independent implementation, not a wrapper or a compatibility claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
import warnings

import numpy as np
import pandas as pd
from scipy.fft import irfft, next_fast_len, rfft
from scipy.optimize import minimize_scalar

from . import genotypes as _geno


_REFERENCE = "https://github.com/MoorjaniLab/DATES_v4010"
_PARAMETERS = ["generations", "decay_rate", "amplitude", "offset"]


@dataclass
class AdmixtureDateResult:
    summary: pd.DataFrame
    parameters: pd.DataFrame
    curve: pd.DataFrame
    diagnostics: pd.DataFrame
    settings: dict
    jackknife: pd.DataFrame | None = None

    def __repr__(self) -> str:
        row = self.summary.iloc[0]
        return (
            f"AdmixtureDateResult(target={row['target']!r})\n\n"
            + self.summary[["est", "se", "n_individuals", "n_snps", "n_chromosomes",
                            "fit_status", "identifiability"]].to_string(
                index=False, float_format=lambda x: f"{x:.4g}",
            )
        )


def _names(value, name):
    values = [value] if isinstance(value, str) else list(value)
    if not values or any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique names")
    return values


def _read_input(data, target, sources, target_iids, format):
    if not isinstance(data, (str, Path)) or Path(data).is_dir():
        raise ValueError("admixture_date requires a genotype dataset prefix; f2 caches and AfData are unsupported")
    pref = str(data)
    if format is None:
        format = "plink" if all(Path(pref + e).exists() for e in (".bed", ".bim", ".fam")) else "auto"
    if format not in {"auto", "plink", "eigenstrat", "packedancestrymap", "tgeno"}:
        raise ValueError("Unsupported genotype format")
    if format == "plink":
        ind = _geno._read_table(pref + ".fam", ["population", "iid", "p1", "p2", "sex", "pheno"])
        snp = _geno.read_snp(pref + ".bim", plink=True)
        path = pref + ".bed"
    else:
        ind, snp = _geno.read_ind(pref + ".ind"), _geno.read_snp(pref + ".snp")
        path = _geno._tgeno_path(pref)
        detected = _geno.detect_geno_format(path, nind=len(ind), nsnp=len(snp))
        if format != "auto" and format != detected:
            raise ValueError(f"Requested {format} but detected {detected}")
        format = detected
    missing = set([target, *sources]) - set(ind.population)
    if missing:
        raise ValueError(f"Populations missing in ind/fam file: {sorted(missing)}")
    take_target = ind.population.eq(target).to_numpy(copy=True)
    if target_iids is not None:
        target_iids = _names(target_iids, "target_iids")
        missing = set(target_iids) - set(ind.loc[take_target, "iid"])
        if missing:
            raise ValueError(f"Individuals missing from target population: {sorted(missing)}")
        take_target &= ind.iid.isin(target_iids).to_numpy()
    nind = len(ind)
    indices = np.flatnonzero(take_target | ind.population.isin(sources).to_numpy())
    ind = ind.iloc[indices].reset_index(drop=True)
    if ind.iid.duplicated().any():
        raise ValueError("Selected individual IDs must be unique")
    if format == "plink":
        g = _geno._read_plink_bed(path, len(snp), nind, sample_indices=indices)
    else:
        if format == "eigenstrat":
            g = _geno._read_eigenstrat_geno(path, nind, last=len(snp), sample_indices=indices)
        elif format == "packedancestrymap":
            g = _geno._read_packed_geno(path, len(snp), nind, sample_indices=indices)
        else:
            g = _geno._read_tgeno(path, len(snp), nind, sample_indices=indices)
    if g.shape != (len(snp), len(ind)):
        raise ValueError("Genotype dimensions do not match SNP and individual metadata")
    if np.any(np.isfinite(g) & ~np.isin(g, [0, 1, 2])):
        raise ValueError("Genotypes must be encoded as 0, 1, 2, or missing")
    if snp.SNP.duplicated().any():
        raise ValueError("SNP IDs must be unique")
    return g, ind, snp, format


class _InsufficientInformativeSNPs(ValueError):
    pass


def _ancestry_scores(g, pa, pb, min_snps):
    """Regression mixture coefficient and sqrt(alpha*beta)-scaled score.

    A heterozygote averages the scores of its two alleles, rather than the
    likelihoods. This also accepts pseudo-haploid calls encoded as 0/2.
    """
    valid = np.isfinite(g) & np.isfinite(pa) & np.isfinite(pb) & (pa != pb)
    if valid.sum() < min_snps:
        raise _InsufficientInformativeSNPs(
            f"Fewer than min_snps={min_snps} observed ancestry-informative SNPs (observed {valid.sum()})"
        )
    delta = pa[valid] - pb[valid]
    alpha = float(np.dot(g[valid] / 2 - pb[valid], delta) / np.dot(delta, delta))
    if not 0 < alpha < 1:
        raise ValueError(f"Ancestry regression is outside (0, 1): {alpha:.6g}; target must be admixed between the references")
    p = alpha * pa[valid] + (1 - alpha) * pb[valid]
    score = np.full(len(g), np.nan)
    score[valid] = np.sqrt(alpha * (1 - alpha)) * delta * (
        (g[valid] / 2) / p - (1 - g[valid] / 2) / (1 - p)
    )
    return score, alpha


def _pair_stats(position, score, bin_size, max_distance, method, mesh_subdivisions):
    """Return additive [pair count, endpoint sum, product sum, distance sum].

    Endpoints are symmetrized, so reversing a chromosome cannot change the
    covariance. Pairs always come from one chromosome of one individual.
    """
    nbins = int(np.ceil(max_distance / bin_size))
    result = np.zeros((nbins, 4))
    valid = np.isfinite(score)
    x, y = np.asarray(position)[valid], np.asarray(score)[valid]
    if len(y) < 2:
        return result
    if method == "direct":
        for i in range(len(y) - 1):
            tolerance = 16 * np.finfo(float).eps * max(1., abs(x[i]), max_distance)
            stop = np.searchsorted(x, x[i] + max_distance + tolerance, side="left")
            d = x[i + 1:stop] - x[i]
            ratio = d / bin_size
            nearest = np.rint(ratio)
            # Keep mathematically equal bin boundaries together despite float subtraction.
            ratio = np.where(np.abs(ratio - nearest) <= tolerance / bin_size, nearest, ratio)
            b = np.floor(ratio).astype(int)
            good = (b < nbins) & (d < max_distance - tolerance)
            b, d, other = b[good], d[good], y[i + 1:stop][good]
            for col, value in enumerate((np.ones(len(b)), y[i] + other, y[i] * other, d)):
                result[:, col] += np.bincount(b, weights=value, minlength=nbins)
        return result

    mesh = bin_size / mesh_subdivisions
    origin = position[0]
    span = (x[-1] - origin) / mesh
    if span > 5_000_000:
        raise ValueError("FFT mesh exceeds 5 million cells; increase bin_size, decrease mesh_subdivisions, or use method='direct'")
    cell = np.floor((x - origin) / mesh + .5).astype(np.int64)
    length = int(cell[-1]) + 1
    count = np.bincount(cell, minlength=length).astype(float)
    total = np.bincount(cell, weights=y, minlength=length)
    squares = np.bincount(cell, weights=y * y, minlength=length)
    size = next_fast_len(2 * length - 1)
    fc, ft = rfft(count, size), rfft(total, size)
    lag = np.arange(min(length, int(np.ceil(max_distance / mesh))))
    pairs = np.rint(irfft(fc * fc.conj(), size)[:len(lag)])
    endpoints = irfft(ft * fc.conj() + fc * ft.conj(), size)[:len(lag)]
    products = irfft(ft * ft.conj(), size)[:len(lag)]
    # Zero lag contains self-pairs and both orders. Retain each distinct pair once.
    pairs[0] = np.sum(count * (count - 1)) / 2
    endpoints[0] = np.sum((count - 1) * total)
    products[0] = np.sum(total * total - squares) / 2
    empty = pairs <= 0
    pairs[empty], endpoints[empty], products[empty] = 0, 0, 0
    b = lag // mesh_subdivisions
    good = (b < nbins) & (lag * mesh < max_distance)
    for col, value in enumerate((pairs, endpoints, products, pairs * lag * mesh)):
        result[:, col] = np.bincount(b[good], weights=value[good], minlength=nbins)
    return result


def _covariance(stats):
    count = stats[:, 0]
    covariance, distance = np.full(len(count), np.nan), np.full(len(count), np.nan)
    good = count > 0
    covariance[good] = stats[good, 2] / count[good] - (stats[good, 1] / (2 * count[good])) ** 2
    distance[good] = stats[good, 3] / count[good]
    return covariance, distance


def _fit_decay(distance, covariance, fit_intercept, rate_bounds):
    """Profile out nonnegative amplitude and an unconstrained intercept."""
    good = np.isfinite(distance) & np.isfinite(covariance)
    x, y = distance[good], covariance[good]
    nparam = 3 if fit_intercept else 2
    if len(y) <= nparam or np.ptp(x) <= 0:
        return dict(status="insufficient_bins", values=np.full(4, np.nan), sse=np.nan,
                    nrmsd=np.nan, boundary=False, condition=np.inf)
    scale = max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    yn = y / scale
    x0 = float(x.min())
    xx = x - x0

    def profile(log_rate):
        rate = np.exp(log_rate)
        basis = np.exp(-rate * xx)
        if fit_intercept:
            centered = basis - basis.mean()
            denom = np.dot(centered, centered)
            amp = max(0., float(np.dot(centered, yn - yn.mean()) / denom)) if denom else 0.
            offset = float(yn.mean() - amp * basis.mean())
        else:
            amp = max(0., float(np.dot(basis, yn) / np.dot(basis, basis)))
            offset = 0.
        residual = yn - (amp * basis + offset)
        return float(np.dot(residual, residual)), amp * scale, offset * scale

    grid = np.linspace(*np.log(rate_bounds), 65)
    costs = np.array([profile(z)[0] for z in grid])
    candidates = [(costs[0], grid[0]), (costs[-1], grid[-1])]
    for i in range(1, len(grid) - 1):
        if costs[i] <= costs[i - 1] and costs[i] <= costs[i + 1]:
            opt = minimize_scalar(lambda z: profile(z)[0], bounds=(grid[i - 1], grid[i + 1]),
                                  method="bounded", options={"xatol": 1e-10})
            if opt.success:
                candidates.append((opt.fun, opt.x))
    _, log_rate = min(candidates)
    rate = float(np.exp(log_rate))
    cost, amp_at_start, offset = profile(log_rate)
    flat = np.ptp(yn) < 1e-12 or amp_at_start <= scale * 1e-12
    if flat:
        offset = float(y.mean()) if fit_intercept else 0.
        return dict(status="no_signal", values=np.array([np.nan, np.nan, 0., offset]),
                    sse=float(np.sum((y - offset) ** 2)), nrmsd=np.nan,
                    boundary=False, condition=np.inf)
    with np.errstate(over="ignore"):
        amplitude = float(amp_at_start * np.exp(rate * x0))
    fitted = amp_at_start * np.exp(-rate * xx) + offset
    dynamic_range = np.ptp(fitted)
    nrmsd = float(np.sqrt(cost / len(y)) * scale / dynamic_range) if dynamic_range else np.nan
    basis = np.exp(-rate * xx)
    columns = [basis, xx * basis]
    if fit_intercept:
        columns.append(np.ones(len(x)))
    jac = np.column_stack(columns)
    jac /= np.maximum(np.linalg.norm(jac, axis=0), np.finfo(float).tiny)
    condition = float(np.linalg.cond(jac))
    boundary = bool(np.isclose(rate, rate_bounds[0], rtol=1e-4) or np.isclose(rate, rate_bounds[1], rtol=1e-4))
    # Match the v4010 executable (dates_expfit.c), which adds one generation.
    return dict(status="converged", values=np.array([rate + 1, rate, amplitude, offset]),
                sse=cost * scale ** 2, nrmsd=nrmsd, boundary=boundary, condition=condition)


def _jackknife_se(full, replicates, weights):
    """Unequal-delete-group jackknife; never silently discard failed fits."""
    reps, weights = np.asarray(replicates), np.asarray(weights, float)
    if len(weights) < 2 or not np.isfinite(reps).all() or not np.isfinite(full).all():
        return np.full(len(full), np.nan)
    h = weights.sum() / weights
    center = np.sum(full - reps, axis=0) + np.average(reps, axis=0, weights=weights)
    pseudo = h[:, None] * full - (h[:, None] - 1) * reps
    return np.sqrt(np.mean((pseudo - center) ** 2 / (h[:, None] - 1), axis=0))


def admixture_date(
    data,
    target: str,
    sources,
    *,
    target_iids=None,
    min_distance: float = 0.0045,
    max_distance: float = 1.0,
    bin_size: float = 0.001,
    map_unit: str | None = None,
    fit_intercept: bool = True,
    jackknife: str | None = "chromosome",
    confidence: float = 0.95,
    return_jackknife: bool = False,
    method: str = "fft",
    mesh_subdivisions: int = 10,
    rate_bounds=(0.001, 10000.0),
    min_snps: int = 100,
    autosomes_only: bool = True,
    adjust_pseudohaploid: bool = True,
    format: str | None = None,
    verbose: bool = True,
) -> AdmixtureDateResult:
    """Estimate a single admixture date from two reference populations.

    ``data`` is an EIGENSTRAT, packed AncestryMap, TGENO, or PLINK prefix.
    Population allele-frequency tables and f2 caches cannot be used. Target
    genotypes remain separate, only within-individual SNP pairs are accumulated.
    Selected genotypes are loaded into memory.

    ``target_iids`` selects individual IDs within the target population. Omit it
    to select all target individuals. Targets with fewer than ``min_snps``
    observed ancestry-informative SNPs are skipped with a warning and recorded
    in ``settings['skipped_individuals']`` (ID to reason). Counts and
    ``settings['target_iids']`` include only retained targets. If none remain,
    a ValueError is raised.

    All distance arguments are Morgans. Input maps default to Morgans for
    EIGENSTRAT-family files and centiMorgans for PLINK; ``map_unit`` overrides
    this. No physical-distance fallback is performed. Only whole bins starting
    at or above ``min_distance`` are fitted, using their mean pair distances.

    The curve is endpoint-symmetrized ancestry-score covariance and is fitted
    by unweighted binwise least squares to ``A * exp(-decay_rate*d) + offset``.
    Amplitude is nonnegative. ``generations = decay_rate + 1`` follows the DATES
    v4010 executable, and refers to generations before the samples lived.

    FFT uses a mesh of ``bin_size / mesh_subdivisions`` Morgans. Direct mode
    enumerates exact pairs and is useful for small-data verification. Counts
    refer to SNP pairs per target individual, not independent observations.

    Chromosome jackknife weights use observed informative target-SNP counts.
    Genome-wide ancestry regression coefficients are held fixed across deletions.
    Failed, boundary, or ill-conditioned deletion fits suppress uncertainty;
    confidence intervals are approximate normal jackknife intervals. No formal
    admixture-detection or goodness-of-fit p-value is supplied.
    """
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a nonempty population name")
    if sources is None:
        raise ValueError("sources must contain exactly two reference populations")
    sources = _names(sources, "sources")
    if len(sources) != 2 or target in sources:
        raise ValueError("sources must contain exactly two distinct populations different from target")
    distances = np.asarray([min_distance, max_distance, bin_size], float)
    if not np.isfinite(distances).all() or not 0 <= min_distance < max_distance or bin_size <= 0:
        raise ValueError("Require 0 <= min_distance < max_distance and bin_size > 0 (Morgans)")
    if np.ceil(max_distance / bin_size) > 100_000:
        raise ValueError("More than 100,000 distance bins requested; increase bin_size")
    if map_unit not in {None, "M", "cM"}:
        raise ValueError("map_unit must be 'M' or 'cM'")
    if method not in {"fft", "direct"}:
        raise ValueError("method must be 'fft' or 'direct'")
    if jackknife not in {None, "chromosome"}:
        raise ValueError("jackknife must be 'chromosome' or None")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    for value, name, minimum in [(mesh_subdivisions, "mesh_subdivisions", 1), (min_snps, "min_snps", 2)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    bounds = np.asarray(rate_bounds, float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
        raise ValueError("rate_bounds must be two finite, increasing positive decay rates")
    for value, name in [(fit_intercept, "fit_intercept"), (return_jackknife, "return_jackknife"),
                        (autosomes_only, "autosomes_only"), (adjust_pseudohaploid, "adjust_pseudohaploid")]:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")

    g, ind, snp, detected_format = _read_input(data, target, sources, target_iids, format)
    input_unit = map_unit or ("cM" if detected_format == "plink" else "M")
    position = snp.cm.to_numpy(float) / (100 if input_unit == "cM" else 1)
    chromosome = snp.CHR.astype(str).str.replace(r"^chr", "", regex=True, case=False).to_numpy(copy=True)
    numeric_chrom = pd.to_numeric(pd.Series(chromosome), errors="coerce").to_numpy()
    keep = np.isin(numeric_chrom, np.arange(1, 23)) if autosomes_only else np.ones(len(snp), bool)
    if not keep.any():
        raise ValueError("No SNPs remain after chromosome selection")
    g, position, chromosome = g[keep], position[keep], chromosome[keep]
    # Normalize numeric chromosome labels (e.g. chr1 and 1.0) to the same group.
    numeric_chrom = numeric_chrom[keep]
    for i in np.flatnonzero(np.isfinite(numeric_chrom)):
        chromosome[i] = f"{numeric_chrom[i]:g}"
    if not np.isfinite(position).all():
        raise ValueError("A finite genetic map is required for every selected SNP")
    chroms = list(dict.fromkeys(chromosome))
    chrom_indices = [np.flatnonzero(chromosome == ch) for ch in chroms]
    map_origins = {}
    for ch, indices in zip(chroms, chrom_indices):
        p = position[indices]
        if np.any(np.diff(p) < 0):
            raise ValueError(f"Genetic map decreases on chromosome {ch}; sort SNPs and genotypes together")
        if len(p) > 1 and np.ptp(p) == 0:
            raise ValueError(f"No usable genetic map on chromosome {ch}; physical positions cannot substitute")
        # Genetic-map origins are arbitrary; only within-chromosome differences
        # enter the estimator. AADR includes valid negative starting coordinates.
        map_origins[ch] = float(p[0])
        position[indices] -= p[0]
    if verbose:
        print(f"Estimating admixture date: {target}, sources={sources}, format={detected_format}, map={input_unit}", flush=True)
    labels = ind.population.to_numpy()
    source_index = np.array([sources.index(p) if p in sources else -1 for p in labels])
    ploidy = _geno._detect_pseudohaploid(g, source_index, True) if adjust_pseudohaploid else np.full(len(ind), 2.)
    af, _ = _geno._geno_to_af_arrays(g, source_index, sources, ploidy)
    pa, pb = af.T
    nbins = int(np.ceil(max_distance / bin_size))
    by_chrom = np.zeros((len(chroms), nbins, 4))
    observed = np.zeros(len(chroms), dtype=int)
    used_snps = np.zeros(len(g), bool)
    coefficients = {}
    skipped_individuals = {}
    for col in np.flatnonzero(labels == target):
        iid = ind.iid.iloc[col]
        try:
            score, alpha = _ancestry_scores(g[:, col], pa, pb, min_snps)
        except _InsufficientInformativeSNPs as exc:
            skipped_individuals[iid] = str(exc)
            warnings.warn(f"Skipping target individual {iid}: {exc}", RuntimeWarning, stacklevel=2)
            continue
        except ValueError as exc:
            raise ValueError(f"Target individual {iid}: {exc}") from exc
        coefficients[iid] = alpha
        used_snps |= np.isfinite(score)
        for k, indices in enumerate(chrom_indices):
            observed[k] += int(np.isfinite(score[indices]).sum())
            by_chrom[k] += _pair_stats(position[indices], score[indices], bin_size,
                                       max_distance, method, mesh_subdivisions)
        if verbose:
            print(f"  Processed {iid}: {np.isfinite(score).sum()} informative SNPs", flush=True)

    if not coefficients:
        raise ValueError(
            f"No usable target individuals remain: all {len(skipped_individuals)} targets have "
            f"fewer than min_snps={min_snps} observed ancestry-informative SNPs"
        )

    total = by_chrom.sum(axis=0)
    covariance, distance = _covariance(total)
    left = np.arange(nbins) * bin_size
    right = np.minimum(left + bin_size, max_distance)
    selected = (left >= min_distance - bin_size * 1e-10) & (total[:, 0] > 0)
    fit = _fit_decay(distance[selected], covariance[selected], fit_intercept, bounds)
    full = fit["values"]
    flags = []
    if fit["status"] != "converged":
        flags.append(fit["status"])
    if fit["boundary"]:
        flags.append("rate_at_boundary")
    if fit["condition"] > 1e8:
        flags.append("ill_conditioned_fit")
    if fit["status"] == "converged" and not np.isfinite(full).all():
        flags.append("amplitude_overflow")
    active = by_chrom[:, selected, 0].sum(axis=1) > 0
    replicas, rows, failed = [], [], 0
    se = np.full(4, np.nan)
    stable_full = fit["status"] == "converged" and not flags
    if jackknife is not None:
        for k in np.flatnonzero(active):
            loo = total - by_chrom[k]
            cy, dx = _covariance(loo)
            same_bins = selected & (loo[:, 0] > 0)
            # Replicates must fit the same bin panel to define one statistic.
            missing_bins = not np.array_equal(same_bins, selected)
            rf = _fit_decay(dx[selected], cy[selected], fit_intercept, bounds)
            stable = (not missing_bins and rf["status"] == "converged" and not rf["boundary"]
                      and rf["condition"] <= 1e8 and np.isfinite(rf["values"]).all())
            failed += int(not stable)
            replicas.append(rf["values"] if stable else np.full(4, np.nan))
            rows.append(dict(chromosome=chroms[k], n_observations=int(observed[k]),
                             weight=float(observed[k] / observed[active].sum()),
                             **dict(zip(_PARAMETERS, rf["values"])),
                             fit_status=rf["status"], rate_at_boundary=rf["boundary"],
                             same_bins=not missing_bins, usable=stable))
        if active.sum() < 2:
            flags.append("insufficient_jackknife_chromosomes")
        elif failed:
            flags.append("unstable_jackknife")
        elif stable_full:
            se = _jackknife_se(full, replicas, observed[active])
    if not fit_intercept:
        se[3] = 0.
    if np.isfinite(se[0]) and se[0] >= .5 * full[0]:
        flags.append("imprecise_date")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    parameters = pd.DataFrame(dict(parameter=_PARAMETERS, est=full, se=se,
                                   ci_low=full - z * se, ci_high=full + z * se,
                                   unit=["generations", "1/M", "score^2", "score^2"]))
    fitted = np.full(nbins, np.nan)
    if fit["status"] == "no_signal":
        fitted[:] = full[3]
    elif fit["status"] == "converged" and np.isfinite(full[2]):
        fitted = full[2] * np.exp(-full[1] * distance) + full[3]
    summary = pd.DataFrame([dict(
        target=target, source1=sources[0], source2=sources[1], est=full[0], se=se[0],
        ci_low=parameters.ci_low.iloc[0], ci_high=parameters.ci_high.iloc[0],
        n_individuals=len(coefficients), n_snps=int(used_snps.sum()), n_chromosomes=int(active.sum()),
        n_bins=int(selected.sum()), min_distance=min_distance, max_distance=max_distance,
        fit_status=fit["status"], identifiability=("unresolved" if flags else
                                                "not_assessed" if jackknife is None else "resolved"),
        status=";".join(flags) if flags else "ok",
    )])
    diagnostics = pd.DataFrame([dict(sse=fit["sse"], nrmsd=fit["nrmsd"],
        condition=fit["condition"], rate_at_boundary=fit["boundary"],
        n_jackknife=len(rows), n_failed_jackknife=failed,
        n_skipped_individuals=len(skipped_individuals),
        date_relative_se=float(se[0] / full[0]) if np.isfinite(se[0]) else np.nan,
        min_ancestry_coefficient=min(coefficients.values()),
        max_ancestry_coefficient=max(coefficients.values()))])
    curve = pd.DataFrame(dict(bin_left=left, bin_right=right, distance=distance,
                              covariance=covariance, fitted=fitted, residual=covariance - fitted,
                              n_pairs=np.rint(total[:, 0]).astype(np.int64), included=selected))
    from . import __version__
    settings = dict(version=__version__, estimator="dates_v4010_score_symmetric_covariance_v1",
                    reference=_REFERENCE, reference_version="4010", method=method,
                    mesh_subdivisions=mesh_subdivisions, input_format=detected_format,
                    input_map_unit=input_unit, map_origins_morgans=map_origins,
                    distance_unit="M", bin_size=bin_size,
                    min_distance=min_distance, max_distance=max_distance,
                    fit_intercept=fit_intercept, fit_weights="equal_bins",
                    rate_bounds=tuple(map(float, bounds)), generation_offset=1,
                    generation_convention="generations = decay_rate + 1; before sampling",
                    jackknife=jackknife, jackknife_weights="observed_informative_target_snp_counts",
                    ancestry_coefficients=coefficients, jackknife_coefficients="fixed_full_data",
                    confidence=confidence, min_snps=min_snps, autosomes_only=autosomes_only,
                    adjust_pseudohaploid=adjust_pseudohaploid, target_iids=list(coefficients),
                    skipped_individuals=skipped_individuals,
                    covariance_centering="pooled_symmetric_pair_endpoints",
                    reference_compatibility="not established")
    if flags:
        warnings.warn("Admixture date diagnostics: " + ", ".join(flags) + ". Inspect curve and diagnostics.",
                      RuntimeWarning, stacklevel=2)
    return AdmixtureDateResult(summary, parameters, curve, diagnostics, settings,
                               pd.DataFrame(rows) if return_jackknife else None)
