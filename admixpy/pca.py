"""Reference genotype PCA and observed-SNP least-squares projection.

Genotypes count A1 and remain on the 0/1/2 dosage scale, including 0/2
pseudo-haploid calls. Fitting uses reference-only HWE standardization and
mean imputation. Projection never estimates new allele frequencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from . import genotypes as _geno


class _Unset:
    def __repr__(self):
        return "<default>"


_UNSET = _Unset()
_FIT_DEFAULTS = dict(n_components=10, keepsnps=None, max_snps=None, seed=0,
                     maxmiss=0.05, minmaf=0.01, auto_only=True,
                     transitions=True, transversions=True, shrinkage="none")
_SNP_COLUMNS = ["SNP", "CHR", "cm", "POS", "A1", "A2"]
_FORMAT_VERSION = 1


@dataclass
class PCAModel:
    snps: pd.DataFrame
    frequencies: np.ndarray
    scales: np.ndarray
    loadings: np.ndarray
    eigenvalues: np.ndarray
    total_variance: float
    shrinkage_factors: np.ndarray
    settings: dict

    @property
    def n_components(self):
        return len(self.eigenvalues)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if not isinstance(self.snps, pd.DataFrame) or not set(_SNP_COLUMNS) <= set(self.snps):
            raise ValueError("Model SNP metadata must contain SNP, CHR, cm, POS, A1, A2")
        if not isinstance(self.eigenvalues, np.ndarray) or self.eigenvalues.ndim != 1:
            raise ValueError("Model eigenvalues must be a one-dimensional array")
        m, k = len(self.snps), self.n_components
        if not m or not k or self.snps.SNP.isna().any() or self.snps.SNP.duplicated().any():
            raise ValueError("Model requires unique SNP IDs and at least one SNP and component")
        for name, shape in [("frequencies", (m,)), ("scales", (m,)),
                            ("loadings", (m, k)), ("eigenvalues", (k,)),
                            ("shrinkage_factors", (k,))]:
            a = getattr(self, name)
            if not isinstance(a, np.ndarray) or a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"Invalid model {name}; expected finite array of shape {shape}")
        if np.any((self.frequencies <= 0) | (self.frequencies >= 1)):
            raise ValueError("Model allele frequencies must be between zero and one")
        if np.any(self.scales <= 0) or np.any(self.eigenvalues <= 0):
            raise ValueError("Model scales and eigenvalues must be positive")
        if np.any((self.shrinkage_factors <= 0) | (self.shrinkage_factors > 1)):
            raise ValueError("Model shrinkage factors must be in (0, 1]")
        if (not np.isfinite(self.total_variance) or self.total_variance <= 0
                or self.total_variance < self.eigenvalues.sum() * (1 - 1e-10)):
            raise ValueError("Model total variance is inconsistent with its eigenvalues")
        if not isinstance(self.settings, dict):
            raise ValueError("Model settings must be a dictionary")


@dataclass
class PCAResult:
    scores: pd.DataFrame
    eigenvalues: pd.DataFrame
    diagnostics: pd.DataFrame
    settings: dict
    model: PCAModel

    def __repr__(self):
        counts = self.scores.role.value_counts()
        return (f"PCAResult(n_reference={counts.get('reference', 0)}, "
                f"n_projected={counts.get('projected', 0)}, "
                f"n_snps={len(self.model.snps)}, n_components={self.model.n_components})\n\n"
                + self.eigenvalues.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


def _names(value, name):
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value)
    if not values or any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique names")
    return values


def _select(ind, pops, iids):
    pops, iids = _names(pops, "pops"), _names(iids, "iids")
    keep = np.ones(len(ind), dtype=bool)
    for values, column, name in [(pops, "population", "Populations"), (iids, "iid", "Individuals")]:
        if values is not None:
            missing = set(values) - set(ind[column])
            if missing:
                raise ValueError(f"{name} missing in ind/fam file: {sorted(missing)}")
            keep &= ind[column].isin(values).to_numpy()
    if not keep.any():
        raise ValueError("Sample selection is empty")
    if ind.loc[keep, "iid"].duplicated().any():
        raise ValueError("Selected individual IDs must be unique")
    return np.flatnonzero(keep)


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_genotypes(g):
    if np.any(~(np.isnan(g) | np.isin(g, [0, 1, 2]))):
        raise ValueError("Genotypes must be encoded as 0, 1, 2, or missing")


def _metadata_mask(snp, options):
    keep = np.ones(len(snp), dtype=bool)
    if options["keepsnps"] is not None:
        ids = _names(options["keepsnps"], "keepsnps")
        missing = set(ids) - set(snp.SNP)
        if missing:
            warnings.warn(f"{len(missing)} keepsnps IDs are absent from the dataset", stacklevel=3)
        keep &= snp.SNP.isin(ids).to_numpy()
    if options["auto_only"]:
        chrom = pd.to_numeric(snp.CHR.astype(str).str.replace(r"^chr", "", regex=True, case=False), errors="coerce")
        keep &= chrom.between(1, 22).to_numpy() & (chrom % 1 == 0).to_numpy()
    a1, a2 = snp.A1.astype(str).str.upper(), snp.A2.astype(str).str.upper()
    valid = a1.isin(list("ACGT")) & a2.isin(list("ACGT")) & a1.ne(a2)
    keep &= valid.to_numpy()
    pairs = a1 + a2
    transition = pairs.isin(["AG", "GA", "CT", "TC"]).to_numpy()
    if not options["transitions"]:
        keep &= ~transition
    if not options["transversions"]:
        keep &= transition
    return keep


def _selected_blocks(dataset, indices, sources, chunk_size):
    """Read only chunks containing model SNPs, preserving their fitted order."""
    for start in range(0, len(dataset.snp), chunk_size):
        stop = min(start + chunk_size, len(dataset.snp))
        lo, hi = np.searchsorted(sources, [start, stop])
        if lo < hi:
            g = dataset.read(start + 1, stop, indices)
            _validate_genotypes(g)
            yield np.arange(lo, hi), g[sources[lo:hi] - start]


def _fit_reference(dataset, indices, options, chunk_size):
    n = len(indices)
    k = options["n_components"]
    if k > n - 1:
        raise ValueError(f"{n} reference individuals support at most {n - 1} components")
    candidates = _metadata_mask(dataset.snp, options)
    frequencies = np.full(len(dataset.snp), np.nan)
    eligible = np.zeros(len(dataset.snp), dtype=bool)
    for start, g in dataset.iter_chunks(indices, chunk_size):
        _validate_genotypes(g)
        count = np.isfinite(g).sum(axis=1)
        p = np.divide(np.nansum(g, axis=1), 2 * count,
                      out=np.full(len(g), np.nan), where=count > 0)
        stop = start + len(g)
        frequencies[start:stop] = p
        eligible[start:stop] = (candidates[start:stop] & (p > 0) & (p < 1)
                                & (np.minimum(p, 1 - p) >= options["minmaf"])
                                & (((n - count) / n) <= options["maxmiss"]))
    sources = np.flatnonzero(eligible)
    n_eligible = len(sources)
    cap = options["max_snps"]
    if cap is not None and cap < len(sources):
        sources = np.sort(np.random.default_rng(options["seed"]).choice(sources, cap, replace=False))
    m = len(sources)
    if m == 0:
        raise ValueError("No polymorphic reference SNPs remain after filtering")
    if k > min(n - 1, m):
        raise ValueError("Requested components exceed reference matrix dimensions")
    p = frequencies[sources]
    scales = np.sqrt(2 * p * (1 - p))
    gram = np.zeros((n, n))
    counts = np.zeros(n, dtype=int)
    for model_indices, g in _selected_blocks(dataset, indices, sources, chunk_size):
        counts += np.isfinite(g).sum(axis=0)
        z = (g - 2 * p[model_indices, None]) / scales[model_indices, None]
        z = np.nan_to_num(z, nan=0.0)
        gram += z.T @ z
    if np.any(counts == 0):
        missing = dataset.ind.iloc[indices[counts == 0]].iid.tolist()
        raise ValueError(f"Reference individuals have no observed retained SNPs: {missing}")
    gram /= m
    total_variance = float(np.trace(gram))
    eigenvalues, u = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, u = eigenvalues[order], u[:, order]
    tolerance = np.finfo(float).eps * n * max(float(eigenvalues[0]), 1.0)
    positive = eigenvalues > tolerance
    if positive.sum() < k:
        raise ValueError(f"Requested {k} components but reference matrix has rank {positive.sum()}")
    factors = (_autoshrink_factors(eigenvalues[positive], n_components=k, n_samples=n)
               if options["shrinkage"] == "autoshrink" else np.ones(k))
    eigenvalues, u = eigenvalues[:k], u[:, :k]
    singular = np.sqrt(m * eigenvalues)
    loadings = np.empty((m, k))
    for model_indices, g in _selected_blocks(dataset, indices, sources, chunk_size):
        z = np.nan_to_num((g - 2 * p[model_indices, None]) / scales[model_indices, None], nan=0.0)
        loadings[model_indices] = (z @ u) / singular
    signs = np.sign(loadings[np.argmax(np.abs(loadings), axis=0), np.arange(k)])
    loadings *= signs
    scores = u * signs * singular * factors
    from . import __version__
    settings = {key: value for key, value in options.items() if key != "keepsnps"}
    settings.update(software="admixpy", software_version=__version__,
                    normalization="diploid_hwe", counted_allele="A1",
                    coordinate_convention="X_times_loading; reference_times_shrinkage_factor",
                    missing_reference="mean_imputation", n_input_snps=len(dataset.snp),
                    n_eligible_snps=n_eligible, n_snps=m, n_reference=n,
                    reference_iids=dataset.ind.iloc[indices].iid.tolist(),
                    reference_populations=dataset.ind.iloc[indices].population.tolist(),
                    keepsnps_supplied=options["keepsnps"] is not None,
                    format=dataset.format, input_prefix=str(dataset.path))
    snps = dataset.snp.iloc[sources][_SNP_COLUMNS].reset_index(drop=True).copy()
    snps[["A1", "A2"]] = snps[["A1", "A2"]].apply(lambda col: col.str.upper())
    model = PCAModel(snps, p, scales, loadings, eigenvalues, total_variance, factors, settings)
    diag = dataset.ind.iloc[indices][["iid", "population"]].reset_index(drop=True).copy()
    diag["role"] = "reference"
    diag["n_snps"] = counts
    diag["frac_observed"] = counts / m
    diag["status"] = "ok"
    return model, scores, diag


def _harmonize(model, snp, mismatch):
    by_id = {sid: i for i, sid in enumerate(model.snps.SNP)}
    reference_a1 = model.snps.A1.astype(str).str.upper().to_numpy()
    reference_a2 = model.snps.A2.astype(str).str.upper().to_numpy()
    sources, model_indices, flips, incompatible = [], [], [], []
    for j, (sid, a1, a2) in enumerate(zip(snp.SNP, snp.A1, snp.A2)):
        i = by_id.get(sid)
        if i is None:
            continue
        pair = (str(a1).upper(), str(a2).upper())
        ref = (reference_a1[i], reference_a2[i])
        if pair == ref:
            flip = False
        elif pair == ref[::-1]:
            flip = True
        else:
            incompatible.append(sid)
            continue
        sources.append(j)
        model_indices.append(i)
        flips.append(flip)
    if incompatible and mismatch == "error":
        raise ValueError(f"Incompatible alleles for matched SNP IDs: {incompatible[:10]}")
    if incompatible:
        warnings.warn(f"Dropping {len(incompatible)} allele-incompatible model SNPs", stacklevel=3)
    return (np.asarray(sources, dtype=int), np.asarray(model_indices, dtype=int),
            np.asarray(flips, dtype=bool), len(incompatible))


def _project_samples(dataset, indices, model, chunk_size, mismatch, min_snps):
    sources, model_indices, flips, n_mismatched = _harmonize(model, dataset.snp, mismatch)
    n, k = len(indices), model.n_components
    r = np.zeros((n, k, k))
    rhs = np.zeros((n, k))
    residual_squared = np.zeros(n)
    counts = np.zeros(n, dtype=int)
    for matched, g in _selected_blocks(dataset, indices, sources, chunk_size):
        mi = model_indices[matched]
        g = np.where(flips[matched, None], 2 - g, g)
        z = ((g - 2 * model.frequencies[mi, None]) / model.scales[mi, None]).T
        present = np.isfinite(z)
        counts += present.sum(axis=1)
        loading = model.loadings[mi]
        for sample in range(n):
            if not present[sample].any():
                continue
            a = np.vstack((r[sample], loading[present[sample]]))
            b = np.concatenate((rhs[sample], z[sample, present[sample]]))
            q, new_r = np.linalg.qr(a, mode="reduced")
            new_rhs = q.T @ b
            residual_squared[sample] += max(float(b @ b - new_rhs @ new_rhs), 0.0)
            r[sample], rhs[sample] = new_r, new_rhs
    coords = np.full((n, k), np.nan)
    ranks = np.zeros(n, dtype=int)
    conditions = np.full(n, np.inf)
    residuals = np.full(n, np.nan)
    statuses = []
    for sample in range(n):
        if counts[sample] == 0:
            statuses.append("no_observed_snps")
            continue
        rcond = np.finfo(float).eps * max(int(counts[sample]), k)
        score, _, rank, singular = np.linalg.lstsq(r[sample], rhs[sample], rcond=rcond)
        ranks[sample] = rank
        if rank == k:
            conditions[sample] = singular[0] / singular[-1]
        residuals[sample] = np.sqrt(residual_squared[sample] + np.linalg.norm(r[sample] @ score - rhs[sample]) ** 2)
        if counts[sample] < min_snps:
            statuses.append("insufficient_snps")
        elif rank < k:
            statuses.append("rank_deficient")
        else:
            coords[sample] = score
            statuses.append("ok")
    diag = dataset.ind.iloc[indices][["iid", "population"]].reset_index(drop=True).copy()
    diag["role"] = "projected"
    diag["n_snps"] = counts
    diag["frac_observed"] = counts / len(model.snps)
    diag["rank"] = ranks
    diag["condition_number"] = conditions
    diag["residual_norm"] = residuals
    diag["status"] = statuses
    matching = dict(n_query_snps=len(dataset.snp), n_model_snps=len(model.snps),
                    n_matched_snps=len(sources), n_allele_flips=int(flips.sum()),
                    n_mismatched_snps=n_mismatched)
    if any(status != "ok" for status in statuses):
        warnings.warn("Some projected individuals failed SNP-count or rank checks; see diagnostics", stacklevel=3)
    return coords, diag, matching


def _score_table(ind, indices, scores, role):
    out = ind.iloc[indices][["iid", "population"]].reset_index(drop=True).copy()
    out["role"] = role
    return pd.concat([out, pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(scores.shape[1])])], axis=1)


def pca(
    data, *, model=None, pops=None, iids=None, project_pops=None, project_iids=None,
    n_components=_UNSET, keepsnps=_UNSET, max_snps=_UNSET, seed=_UNSET,
    maxmiss=_UNSET, minmaf=_UNSET, auto_only=_UNSET,
    transitions=_UNSET, transversions=_UNSET, shrinkage=_UNSET,
    mismatch="error", min_snps=100, chunk_size=4096, format=None, verbose=True,
) -> PCAResult:
    """Fit reference PCA, optionally project samples, or apply an existing model.

    ``pops``/``iids`` select fitting individuals; ``project_pops``/``project_iids``
    select projected individuals. Each pair intersects when both are supplied.
    Labels and dataset order are preserved. Projection requires an explicit
    fitting selection or ``model``. The two selections must not overlap.

    Fitting defaults: n_components=10, keepsnps=None, max_snps=None, seed=0,
    maxmiss=0.05, minmaf=0.01, auto_only=True, transitions=True,
    transversions=True, shrinkage="none". All fitting options, including
    explicitly supplied defaults, are rejected when applying ``model``.

    ``keepsnps`` is a candidate SNP-ID panel intersected with reference QC.
    ``maxmiss`` measures missing individual calls among fitting individuals.
    ``max_snps`` uniformly subsamples eligible SNPs without replacement using
    ``seed``; this is a computational cap, not LD pruning. Fixed model SNPs
    and normalization are reused during projection, with allele-order flips
    handled explicitly and no automatic strand-complement inference.

    Projection uses observed-SNP least squares. Failed rows remain with NaN
    coordinates and a diagnostic status. No model is refitted. The reference
    Gram matrix requires O(n_reference**2) memory despite chunked SNP reads.
    """
    supplied = dict(n_components=n_components, keepsnps=keepsnps, max_snps=max_snps,
                    seed=seed, maxmiss=maxmiss, minmaf=minmaf, auto_only=auto_only,
                    transitions=transitions, transversions=transversions, shrinkage=shrinkage)
    project = project_pops is not None or project_iids is not None
    _positive_int(chunk_size, "chunk_size")
    _positive_int(min_snps, "min_snps")
    if mismatch not in {"error", "drop"}:
        raise ValueError("mismatch must be 'error' or 'drop'")
    if model is not None:
        if not isinstance(model, PCAModel):
            raise ValueError("model must be a PCAModel")
        model._validate()
        invalid = [key for key, value in supplied.items() if value is not _UNSET]
        if pops is not None or iids is not None or invalid:
            raise ValueError("Fitting selectors and options cannot be supplied with model: " + ", ".join(invalid))
        if not project:
            raise ValueError("model requires project_pops or project_iids")
    elif project and pops is None and iids is None:
        raise ValueError("Projection requires explicit fitting pops/iids or an existing model")
    options = {key: (_FIT_DEFAULTS[key] if value is _UNSET else value) for key, value in supplied.items()}
    if model is None:
        _positive_int(options["n_components"], "n_components")
        if options["max_snps"] is not None:
            _positive_int(options["max_snps"], "max_snps")
        if isinstance(options["seed"], (bool, np.bool_)) or not isinstance(options["seed"], (int, np.integer)) or options["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        for key, upper in [("maxmiss", 1), ("minmaf", 0.5)]:
            value = options[key]
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or not 0 <= value <= upper:
                raise ValueError(f"{key} must be in [0, {upper}]")
        for key in ["auto_only", "transitions", "transversions"]:
            if not isinstance(options[key], (bool, np.bool_)):
                raise ValueError(f"{key} must be boolean")
        if options["shrinkage"] not in {"none", "autoshrink"}:
            raise ValueError("shrinkage must be 'none' or 'autoshrink'")
        # Store plain Python values so settings are portable JSON.
        options = {key: value.item() if isinstance(value, np.generic) else value for key, value in options.items()}
    dataset = _geno._open_genotypes(data, format)
    if dataset.snp.SNP.isna().any() or dataset.snp.SNP.duplicated().any():
        raise ValueError("SNP IDs must be nonmissing and unique")
    projected = _select(dataset.ind, project_pops, project_iids) if project else None
    fitting = _select(dataset.ind, pops, iids) if model is None else None
    if fitting is not None and projected is not None and np.intersect1d(fitting, projected).size:
        raise ValueError("Fitting and projected sample selections overlap")
    score_tables, diagnostic_tables = [], []
    mode = "fit" if model is None else "project"
    if model is None:
        _geno._log(f"Fitting PCA with {len(fitting)} reference individuals", verbose)
        model, scores, diag = _fit_reference(dataset, fitting, options, chunk_size)
        score_tables.append(_score_table(dataset.ind, fitting, scores, "reference"))
        diagnostic_tables.append(diag)
    matching = {}
    if projected is not None:
        _geno._log(f"Projecting {len(projected)} individuals onto {model.n_components} PCs", verbose)
        scores, diag, matching = _project_samples(dataset, projected, model, chunk_size, mismatch, min_snps)
        score_tables.append(_score_table(dataset.ind, projected, scores, "projected"))
        diagnostic_tables.append(diag)
    eigenvalues = pd.DataFrame(dict(component=[f"PC{i + 1}" for i in range(model.n_components)],
                                   eigenvalue=model.eigenvalues,
                                   explained_variance_ratio=model.eigenvalues / model.total_variance,
                                   shrinkage_factor=model.shrinkage_factors))
    settings = dict(mode=mode, format=dataset.format, input_prefix=str(data),
                    chunk_size=int(chunk_size), min_snps=int(min_snps), mismatch=mismatch,
                    fitting_iids=[] if fitting is None else dataset.ind.iloc[fitting].iid.tolist(),
                    projected_iids=[] if projected is None else dataset.ind.iloc[projected].iid.tolist(),
                    model_settings=dict(model.settings), **matching)
    return PCAResult(pd.concat(score_tables, ignore_index=True), eigenvalues,
                     pd.concat(diagnostic_tables, ignore_index=True), settings, model)


def write_pca(path, model: PCAModel):
    """Write a fitted model to a new directory (model.npz and metadata.json)."""
    if not isinstance(model, PCAModel):
        raise ValueError("model must be a PCAModel")
    model._validate()
    metadata = json.dumps(dict(format_version=_FORMAT_VERSION, settings=model.settings), indent=2, allow_nan=False)
    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    snp = model.snps
    np.savez_compressed(output / "model.npz",
                        SNP=snp.SNP.to_numpy(dtype=str), CHR=snp.CHR.to_numpy(dtype=str),
                        cm=snp.cm.to_numpy(dtype=float), POS=snp.POS.to_numpy(dtype=float),
                        A1=snp.A1.to_numpy(dtype=str), A2=snp.A2.to_numpy(dtype=str),
                        frequencies=model.frequencies, scales=model.scales,
                        loadings=model.loadings, eigenvalues=model.eigenvalues,
                        total_variance=np.asarray(model.total_variance),
                        shrinkage_factors=model.shrinkage_factors)
    (output / "metadata.json").write_text(metadata + "\n", encoding="utf-8")


def read_pca(path) -> PCAModel:
    """Read and validate a model written by write_pca; never loads pickle data."""
    source = Path(path)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != _FORMAT_VERSION:
        raise ValueError("Unsupported PCA model format version")
    with np.load(source / "model.npz", allow_pickle=False) as arrays:
        required = {*_SNP_COLUMNS, "frequencies", "scales", "loadings", "eigenvalues", "total_variance", "shrinkage_factors"}
        if not required <= set(arrays.files):
            raise ValueError("PCA model is missing required arrays")
        snps = pd.DataFrame({col: arrays[col] for col in _SNP_COLUMNS})
        return PCAModel(snps, arrays["frequencies"], arrays["scales"], arrays["loadings"],
                        arrays["eigenvalues"], float(arrays["total_variance"]),
                        arrays["shrinkage_factors"], metadata["settings"])


_TW1_99_PERCENTILE = 2.023449
_MIN_NOISE_EIGENVALUES = 10


def _autoshrink_factors(
    eigenvalues: np.ndarray,
    *,
    n_components: int,
    n_samples: int,
) -> np.ndarray:
    """Estimate smartpca-autoshrink-style factors for reference PC scores.

    This follows smartpca's inexpensive ``fastshrink``/``autoshrink`` path:
    significant spikes are selected with a Tracy-Widom test, the effective
    marker count is estimated from the residual spectrum, and the spectral
    shrinker is evaluated for each selected spike.  smartpca uses a lookup
    table for the Tracy-Widom tail; only its 1% cutoff is needed here.
    """

    factors = np.ones(n_components, dtype=np.float64)
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    # smartpca disables its fast shrinker unless the requested axes leave a
    # residual spectrum large enough for the moment/TW estimates.
    if eigenvalues.size - n_components < _MIN_NOISE_EIGENVALUES:
        return factors

    max_spikes = n_components
    n_spikes = max_spikes
    for index in range(max_spikes):
        statistic, _effective_markers = _tracy_widom_statistic(eigenvalues[index:])
        if statistic < _TW1_99_PERCENTILE:
            n_spikes = index
            break
    if n_spikes == 0:
        return factors

    noise = eigenvalues[n_spikes:]
    effective_markers = _effective_marker_count(noise)
    if not np.isfinite(effective_markers) or effective_markers <= 0:
        return factors
    gamma = effective_markers / float(n_samples)

    normalized = eigenvalues / np.sum(eigenvalues)
    spectrum = normalized[n_spikes:]
    for index, spike in enumerate(normalized[:n_spikes]):
        inverse_gap = 1.0 / (spectrum - spike)
        m_hat = float(np.mean(inverse_gap))
        v_hat = gamma * m_hat - (1.0 - gamma) / spike
        d_hat = spike * m_hat * v_hat
        if d_hat == 0.0:
            continue
        ell = 1.0 / d_hat
        mp_hat = float(np.mean(inverse_gap * inverse_gap))
        vp_hat = gamma * mp_hat + (1.0 - gamma) / (spike * spike)
        dp_hat = (
            m_hat * v_hat
            + spike * mp_hat * v_hat
            + spike * m_hat * vp_hat
        )
        denominator = dp_hat * ell
        if denominator == 0.0:
            continue
        squared_factor = m_hat / denominator
        if not np.isfinite(squared_factor) or squared_factor <= 0.0:
            continue
        factor = float(np.sqrt(squared_factor))
        # A factor outside this range is not a shrinkage correction and is a
        # sign that the selected spike is not separated from the noise bulk.
        if 0.0 < factor <= 1.0:
            factors[index] = factor
    return factors


def _effective_marker_count(eigenvalues: np.ndarray) -> float:
    """Moment estimate used by EIGENSOFT's ``dotwcalc`` routine."""

    count = eigenvalues.size
    total = float(np.sum(eigenvalues))
    if count == 0 or total <= 0.0:
        return float("nan")
    scaled = eigenvalues * (count / total)
    denominator = float(np.sum(scaled * scaled) - count)
    if denominator <= 0.0:
        return float("inf")
    return float(count * (count + 2) / denominator)


def _tracy_widom_statistic(eigenvalues: np.ndarray) -> tuple[float, float]:
    count = eigenvalues.size
    effective_markers = _effective_marker_count(eigenvalues)
    total = float(np.sum(eigenvalues))
    if not np.isfinite(effective_markers) or effective_markers <= 0.0 or total <= 0.0:
        return float("-inf"), effective_markers
    largest = float(eigenvalues[0]) * (count / total) * effective_markers
    larger = max(float(count), effective_markers)
    smaller = min(float(count), effective_markers)
    root_sum = np.sqrt(larger - 1.0) + np.sqrt(smaller)
    center = root_sum * root_sum
    scale = root_sum * (
        1.0 / np.sqrt(larger - 1.0) + 1.0 / np.sqrt(smaller)
    ) ** (1.0 / 3.0)
    return (largest - center) / scale, effective_markers


