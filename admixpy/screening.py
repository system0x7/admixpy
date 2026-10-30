"""Fast, descriptive screening of convex mixtures in f4 coordinates."""

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import fstats as _fstats


@dataclass
class ScreeningResult:
    models: pd.DataFrame
    weights: pd.DataFrame
    coordinates: pd.DataFrame
    settings: dict

    def format(self, top=5) -> str:
        """Display at most ``top`` models per source count, with a source key."""
        if isinstance(top, bool) or not isinstance(top, Integral) or top < 1:
            raise ValueError("top must be a positive integer")
        aliases = {pop: f"S{i}" for i, pop in enumerate(self.settings["candidates"], 1)}
        lines = [f"ScreeningResult(target={self.settings['target']!r})",
                 "Lower score = closer fit. Exploratory scores, no model-fit p-values.",
                 "Source key:"]
        lines.extend(f"  {alias} = {pop}" for pop, alias in aliases.items())
        by_model = {model: frame for model, frame in self.weights.groupby("model", sort=False)}
        for size, frame in self.models.groupby("n_sources", sort=True):
            shown = frame.head(top)
            display = shown[["model", "score", "max_abs_z", "status"]].copy()
            mixtures = []
            for model in shown.model:
                weights = by_model.get(model)
                mixtures.append(" ".join(
                    f"{aliases[row.source]}:{row.weight:.1%}"
                    for row in weights.itertuples(index=False)
                ) if weights is not None else "-")
            display.insert(1, "mixture", mixtures)
            for col in ["score", "max_abs_z"]:
                display[col] = display[col].map(lambda x: f"{x:.3g}" if np.isfinite(x) else "-")
            lines.extend(["", f"{size}-source models ({len(shown)} of {len(frame)}):",
                          display.to_string(index=False)])
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.format()


def _populations(value, name):
    values = [value] if isinstance(value, str) else list(value)
    if not values or any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f"{name} must contain nonempty population names")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique population names")
    return values


def _residual_stats(weights, estimates, covariance):
    """Weighted source-target contrasts, including shared target covariance."""
    residual = weights @ estimates
    cross = np.einsum("aibj,b->aij", covariance, weights)
    cov = np.einsum("a,aij->ij", weights, cross)
    return residual, (cov + cov.T) / 2, cross


def _fit_mixture(estimates, covariance, ridge):
    """Minimize a descriptive quadratic distance, without a chi-square claim."""
    n_sources, n_coordinates = estimates.shape
    design = (estimates[:-1] - estimates[-1]).T
    if n_sources > 1 and np.linalg.matrix_rank(design) < n_sources - 1:
        return None, "indistinguishable_sources", "Source coordinates are affinely dependent"
    # Scale both coordinates and covariance into typical SE units for SLSQP.
    scale = float(np.mean(np.diag(covariance.reshape(estimates.size, estimates.size))))
    if scale <= 0:
        return None, "no_uncertainty", "No positive target/source variance"
    estimates = estimates / np.sqrt(scale)
    covariance = covariance / scale
    raw_covariance = covariance.copy()
    matrix = covariance.reshape(estimates.size, estimates.size).copy()
    matrix.flat[::len(matrix) + 1] += ridge
    covariance = matrix.reshape(covariance.shape)

    def objective(weights):
        residual, cov, cross = _residual_stats(weights, estimates, covariance)
        solved = np.linalg.solve(cov, residual)
        value = float(residual @ solved)
        gradient = 2 * (estimates @ solved) - 2 * np.einsum(
            "i,aij,j->a", solved, cross, solved
        )
        return value, gradient

    uniform = np.full(n_sources, 1 / n_sources)
    if n_sources == 1:
        weights = uniform
    else:
        vertices = np.eye(n_sources)
        best_vertex = min(vertices, key=lambda w: objective(w)[0])
        normalizer = max(objective(uniform)[0], 1.)

        def scaled_objective(weights):
            value, gradient = objective(weights)
            return value / normalizer, gradient / normalizer

        fits = [minimize(
            scaled_objective, start, jac=True, method="SLSQP",
            bounds=[(0., 1.)] * n_sources,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                         "jac": lambda w: np.ones(n_sources)},
            options={"ftol": 1e-10, "maxiter": 300},
        ) for start in (uniform, best_vertex)]
        valid = [fit for fit in fits if fit.success and np.isfinite(fit.fun) and np.isfinite(fit.x).all()
                 and abs(fit.x.sum() - 1) < 1e-7
                 and np.all((fit.x >= -1e-7) & (fit.x <= 1 + 1e-7))]
        if not valid:
            return None, "optimizer_failed", "; ".join(str(fit.message) for fit in fits)
        weights = np.clip(min(valid, key=lambda fit: fit.fun).x, 0, 1)
        weights /= weights.sum()
    residual, raw_cov, _ = _residual_stats(weights, estimates, raw_covariance)
    _, cov, _ = _residual_stats(weights, estimates, covariance)
    variance = np.diag(raw_cov)
    z = np.divide(residual, np.sqrt(np.maximum(variance, 0)),
                  out=np.full(n_coordinates, np.nan), where=variance > 0)
    eigenvalues = np.linalg.eigvalsh(raw_cov)
    rank = int(np.sum(eigenvalues > max(eigenvalues[-1], 0) * 1e-10))
    flags = []
    if rank < n_coordinates:
        flags.append("singular_covariance")
    if np.any(weights <= 1e-6):
        flags.append("boundary")
    return dict(weights=weights,
                score=float(np.sqrt(max(objective(weights)[0], 0) / n_coordinates)),
                rms_f4=float(np.sqrt(np.mean(residual ** 2) * scale)),
                max_abs_z=float(np.max(np.abs(z))) if np.isfinite(z).all() else np.nan,
                covariance_rank=rank,
                covariance_condition=float(np.linalg.cond(cov))), ";".join(flags) or "ok", ""


def screening(
    data,
    target: str,
    candidates,
    refs,
    *,
    outgroup: str | None = None,
    source_sizes=None,
    allsnps: bool = False,
    covariance_ridge: float = 1e-6,
    verbose: bool = True,
    **kwargs,
) -> ScreeningResult:
    """Quickly explore source mixtures using one reusable joint f4 basis.

    Fit nonnegative weights summing to one to qpAdm's exact contrast basis,
    ``f4(source, target; reference, outgroup)``, using every non-outgroup
    reference. The shared ``qpwave_f4stats`` builder loads all candidate
    coordinates once. The default ``allsnps=False`` uses a common
    SNP panel across all populations for genotype input. ``allsnps=True`` uses
    each contrast's available SNPs and requires genotype input or a matching
    f4 cache; coordinates may then describe different SNP panels. Precomputed
    f2 input supports only ``allsnps=False``. If ``outgroup`` is
    omitted, the first entry of ``refs`` is used and removed from the reference
    panel. An explicit outgroup is also removed from ``refs`` if present.
    At least one other reference is required, and the default two-source
    screen needs at least two other references.

    By default explore 2 through 4 sources, limited by the candidate count and
    number of reference contrasts. Explicit ``source_sizes`` may include 1;
    each size must leave at least one residual coordinate.

    The score is sqrt(r.T @ inv(V) @ r / n_contrasts), where r is the weighted
    sum of source-target contrasts and V is their joint covariance propagated
    at w. Target uncertainty is already included in these contrasts. A small ridge
    (``covariance_ridge`` times the mean joint variance for that model) is
    added to the joint covariance before propagation. SLSQP uses two starts;
    this is a local exploratory fit, not a guarantee of a global optimum.
    Smaller scores indicate closer fits, conditional on uncertainty and the
    selected panel. No p-values, weight SEs, or ancestry-validity claims are
    returned. More sources and noisier candidates can improve scores; compare
    within source count and validate selected models with qpAdm.

    ``models`` retains source lists, numeric scores, covariance diagnostics,
    and fit status. ``weights`` has one numeric row per model/source.
    ``coordinates`` retains the f4 estimates and marginal SEs. The display
    shows five models per source count with a source key; ``format(top=10)``
    changes the display limit. Failures are recorded, never silently dropped.
    Other keyword arguments are forwarded to ``f4_stats`` (e.g. blgsize,
    keepsnps, format, stream). No qpAdm fits or resampling refits are performed.
    """
    refs = _populations(refs, "refs")
    if outgroup is None:
        outgroup = refs[0]
    for value, name in [(target, "target"), (outgroup, "outgroup")]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty population name")
    candidates = _populations(candidates, "candidates")
    refs = [p for p in refs if p != outgroup]
    if not refs:
        raise ValueError("Need at least one reference apart from the outgroup")
    populations = [target, outgroup, *candidates, *refs]
    if len(set(populations)) != len(populations):
        raise ValueError("target, outgroup, candidates, and references must be distinct")
    limit = min(len(candidates), len(refs))
    if source_sizes is None:
        sizes = list(range(2, min(4, limit) + 1))
    else:
        sizes = list(source_sizes)
    if not sizes or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral)
                        or not 1 <= n <= limit for n in sizes):
        raise ValueError(f"source_sizes must contain integers between 1 and {limit}; add references or candidates")
    sizes = sorted(set(sizes))
    if (isinstance(covariance_ridge, (bool, np.bool_)) or not np.isfinite(covariance_ridge)
            or not 0 < covariance_ridge < 1):
        raise ValueError("covariance_ridge must be finite and between 0 and 1")
    if not isinstance(allsnps, (bool, np.bool_)):
        raise ValueError("allsnps must be boolean")
    allsnps = bool(allsnps)
    reserved = {"comb", "unique_only", "covariance", "keep_blocks", "keep_loo",
                "left_base", "right_base"} & kwargs.keys()
    if reserved:
        raise TypeError("screening controls these f4_stats options: " + ", ".join(sorted(reserved)))
    if kwargs.get("afprod", False):
        raise ValueError("screening requires f4 coordinates, not afprod=True")
    is_cache = isinstance(data, _fstats.F4BlockCache)
    if is_cache and bool(data.allsnps) != allsnps:
        raise ValueError("Requested allsnps does not match the f4 cache; rebuild it or use the matching setting")
    # Request explicit row order so covariance coordinates cannot be misaligned.
    rows = pd.DataFrame([
        (source, target, ref, outgroup)
        for source in candidates for ref in refs
    ], columns=["pop1", "pop2", "pop3", "pop4"])
    # Cached statistics already have their SNP panels; f4_stats accepts
    # allsnps=True only when calculating new statistics from genotype input.
    basis = _fstats.qpwave_f4stats(
        data, left=[target, *candidates], right=[outgroup, *refs],
        left_base=target, right_base=outgroup,
        allsnps=False if is_cache else allsnps, covariance=True,
        keep_blocks=False, keep_loo=False, verbose=verbose, **kwargs,
    )
    stats = basis.f4
    estimate = np.asarray(stats.est, float)
    covariance = np.asarray(stats.cov, float)
    if (estimate.shape != (len(rows),) or covariance.shape != (len(rows), len(rows))
            or not np.isfinite(estimate).all() or not np.isfinite(covariance).all()):
        raise ValueError("screening requires finite estimates and a full joint covariance for every coordinate")
    if not stats.rows[rows.columns].reset_index(drop=True).equals(rows):
        raise ValueError("f4 coordinate order does not match the requested screening basis")
    covariance = (covariance + covariance.T) / 2
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] < -1e-8 * max(abs(eigenvalues[-1]), np.finfo(float).tiny):
        raise ValueError("Joint f4 covariance is not positive semidefinite")
    # Remove only numerical negative eigenvalues, without changing a valid PSD matrix.
    if eigenvalues[0] < 0:
        values, vectors = np.linalg.eigh(covariance)
        covariance = (vectors * np.maximum(values, 0)) @ vectors.T
    if np.trace(covariance) <= 0:
        raise ValueError("screening requires positive jackknife variance; inspect usable blocks")
    n_coordinates = len(refs)
    estimates = estimate.reshape(len(candidates), n_coordinates)
    model_rows, weight_rows = [], []
    for size in sizes:
        for indices in combinations(range(len(candidates)), size):
            model = len(model_rows) + 1
            sources = [candidates[i] for i in indices]
            take = np.array([i * n_coordinates + j for i in indices
                             for j in range(n_coordinates)])
            selected_cov = covariance[np.ix_(take, take)].reshape(
                size, n_coordinates, size, n_coordinates)
            try:
                result, status, message = _fit_mixture(
                    estimates[list(indices)], selected_cov, covariance_ridge)
            except np.linalg.LinAlgError as exc:
                result, status, message = None, "numerical_failure", str(exc)
            summary = dict(model=model, n_sources=size, sources=sources, score=np.nan,
                           rms_f4=np.nan, max_abs_z=np.nan, n_contrasts=n_coordinates,
                           covariance_rank=pd.NA, covariance_condition=np.nan,
                           status=status, message=message)
            if result is not None:
                weights = result.pop("weights")
                summary.update(result)
                weight_rows.extend(dict(model=model, source=source, weight=float(weight))
                                   for source, weight in zip(sources, weights))
            model_rows.append(summary)
    models = pd.DataFrame(model_rows).sort_values(
        ["n_sources", "score"], kind="stable", ignore_index=True, na_position="last")
    models["covariance_rank"] = models.covariance_rank.astype("Int64")
    coordinates = rows.copy()
    coordinates["est"] = estimate
    coordinates["se"] = np.sqrt(np.maximum(np.diag(covariance), 0))
    snp_counts = getattr(stats, "snp_counts", None)
    if snp_counts is not None:
        coordinates["n"] = np.sum(snp_counts, axis=1).astype(int)
    from . import __version__
    settings = dict(kwargs, admixpy_version=__version__, target=target, candidates=candidates,
                    refs=refs, outgroup=outgroup, source_sizes=sizes,
                    covariance_ridge=covariance_ridge, allsnps=allsnps,
                    left_base=target, right_base=outgroup,
                    estimator="qpadm_basis_convex_f4_distance", n_blocks=int(np.sum(stats.block_lengths > 0)),
                    score_is_pvalue=False)
    if verbose:
        print(f"Screened {len(models)} source combinations; inspect scores and fit status.", flush=True)
    return ScreeningResult(models, pd.DataFrame(weight_rows, columns=["model", "source", "weight"]),
                           coordinates, settings)
