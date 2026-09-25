from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import linalg, optimize

from . import fstats as _fstats
from .fstats import (
    BlockStats, F4BlockCache, _as_pop_list, _chi2_sf, _contrast_pops,
    _default_genotype_allsnps, _examples_heading, _format_f4_contrasts,
    _model_left_with_target, _models_frame, _require_unique_pops,
    _select_f4_block_cache_model, _validate_resampling,
)


@dataclass
class QpWaveStats:
    f4: BlockStats
    left: list[str]
    right: list[str]
    left_base: str
    right_base: str
    row_pops: list[str]
    col_pops: list[str]

    @property
    def matrix(self) -> np.ndarray:
        return self.f4.est.reshape(len(self.row_pops), len(self.col_pops))

    @property
    def cov(self) -> np.ndarray:
        return self.f4.cov

    @property
    def blocks(self) -> np.ndarray | None:
        if self.f4.blocks is None:
            return None
        return self.f4.blocks.reshape(len(self.row_pops), len(self.col_pops), -1)

    @property
    def loo(self) -> np.ndarray | None:
        if self.f4.loo is None:
            return None
        return self.f4.loo.reshape(len(self.row_pops), len(self.col_pops), -1)


def qpwave_f4stats(
    data,
    left: Sequence[str],
    right: Sequence[str],
    left_base: str | None = None,
    right_base: str | None = None,
    verbose: bool = True,
    **kwargs,
) -> QpWaveStats:
    if not kwargs.get("covariance", True):
        raise ValueError(
            "qpWave and qpAdm require covariance=True; covariance=False is only "
            "supported for standalone f4 statistics"
        )
    if "allsnps" not in kwargs:
        kwargs["allsnps"] = _default_genotype_allsnps(data)
    left = list(left)
    right = list(right)
    left_base, row_pops = _contrast_pops(left, left_base, "left")
    right_base, col_pops = _contrast_pops(right, right_base, "right")
    combos = pd.DataFrame(
        [
            {
                "pop1": row_pop,
                "pop2": left_base,
                "pop3": col_pop,
                "pop4": right_base,
                "left": row_pop,
                "right": col_pop,
            }
            for row_pop in row_pops
            for col_pop in col_pops
        ]
    )
    stats = _fstats.f4_stats(data, combos[["pop1", "pop2", "pop3", "pop4"]], unique_only=False, verbose=verbose, **kwargs)
    stats.rows = combos
    return QpWaveStats(stats, left, right, left_base, right_base, row_pops, col_pops)


def _rank_approx(mat: np.ndarray, rank: int) -> np.ndarray:
    if rank < 0:
        raise ValueError("rank must be non-negative")
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    if rank == 0:
        return np.zeros_like(mat)
    rank = min(rank, len(s))
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def _cov_whitener(cov: np.ndarray, rcond: float) -> np.ndarray:
    vals, vecs = linalg.eigh(cov)
    thresh = max(float(vals[-1]) * rcond, 0.0) if vals.size else 0.0
    keep = vals > thresh
    if not np.any(keep):
        raise ValueError("Covariance matrix has no positive eigenvalues")
    return (vecs[:, keep] / np.sqrt(vals[keep])).T


def _weighted_rank_fit(
    mat: np.ndarray,
    cov: np.ndarray,
    rank: int,
    rcond: float,
    max_nfev: int | None,
    keep: np.ndarray | None = None,
) -> tuple[np.ndarray, bool, float]:
    if rank == 0:
        return np.zeros_like(mat), True, float("nan")
    nrow, ncol = mat.shape
    u, s, vt = np.linalg.svd(mat, full_matrices=False)
    rank = min(rank, len(s))
    root = np.sqrt(s[:rank])
    a0 = u[:, :rank] * root[None, :]
    b0 = root[:, None] * vt[:rank, :]
    x0 = np.concatenate([a0.reshape(-1), b0.reshape(-1)])
    obs = mat.reshape(-1)
    keep = np.isfinite(obs) if keep is None else keep
    whitener = _cov_whitener(cov[np.ix_(keep, keep)], rcond)

    def unpack(x):
        split = nrow * rank
        a = x[:split].reshape(nrow, rank)
        b = x[split:].reshape(rank, ncol)
        return a @ b

    def residuals(x):
        return whitener @ (obs - unpack(x).reshape(-1))[keep]

    res = optimize.least_squares(residuals, x0, method="trf", max_nfev=max_nfev)
    return unpack(res.x), bool(res.success), float(2 * res.cost)


def qpwave_ranktest(
    qpw: QpWaveStats,
    rank: int,
    rcond: float = 1e-10,
    diag: float = 0.0,
    max_nfev: int | None = None,
) -> dict:
    mat = qpw.matrix
    if rank < 0:
        raise ValueError("rank must be non-negative")
    if rank >= min(mat.shape):
        raise ValueError("rank must be smaller than min(number of left contrasts, number of right contrasts)")
    cov = np.asarray(qpw.cov, float)
    _validate_covariance_psd(cov, "qpWave")
    if diag:
        cov = cov + np.eye(cov.shape[0]) * diag
    resid0 = mat.reshape(-1)
    keep = np.isfinite(resid0) & np.isfinite(cov).all(axis=0) & np.isfinite(cov).all(axis=1)
    fitted, converged, opt_cost = _weighted_rank_fit(mat, cov, rank, rcond, max_nfev, keep)
    resid = (mat - fitted).reshape(-1)
    keep = np.isfinite(resid) & np.isfinite(cov).all(axis=0) & np.isfinite(cov).all(axis=1)
    resid_keep = resid[keep]
    cov_keep = cov[np.ix_(keep, keep)]
    invcov = linalg.pinv(cov_keep, rtol=rcond)
    chisq = float(resid_keep @ invcov @ resid_keep)
    dof = (mat.shape[0] - rank) * (mat.shape[1] - rank)
    return {
        "rank": rank,
        "dof": dof,
        "chisq": chisq,
        "p": _chi2_sf(chisq, dof),
        "fitted": fitted,
        "residual": mat - fitted,
        "n_stats": int(keep.sum()),
        "converged": converged,
        "optimizer_cost": opt_cost,
    }


def qpwave(
    data,
    left: Sequence[str],
    right: Sequence[str],
    ranks: Sequence[int] | None = None,
    left_base: str | None = None,
    right_base: str | None = None,
    rcond: float = 1e-10,
    diag: float = 0.0,
    max_nfev: int | None = None,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    if "allsnps" not in kwargs:
        kwargs["allsnps"] = _default_genotype_allsnps(data)
    qpw = _fstats.qpwave_f4stats(data, left, right, left_base=left_base, right_base=right_base, verbose=verbose, **kwargs)
    _validate_f4_inputs(
        qpw,
        allsnps=bool(kwargs["allsnps"]),
        apply_corr=bool(kwargs.get("apply_corr", True)),
        caller="qpWave",
        action="test ranks",
    )
    max_rank = min(qpw.matrix.shape) - 1
    ranks = range(max_rank + 1) if ranks is None else ranks
    rows = []
    for rank in ranks:
        res = qpwave_ranktest(qpw, int(rank), rcond=rcond, diag=diag, max_nfev=max_nfev)
        rows.append({k: res[k] for k in ("rank", "dof", "chisq", "p", "n_stats", "converged", "optimizer_cost")})
    return pd.DataFrame(rows)


def qpwave_multi(
    data,
    models,
    ranks: Sequence[int] | None = None,
    left_base: str | None = None,
    right_base: str | None = None,
    max_nfev: int | None = None,
    use_cache: bool = True,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    models = _models_frame(models)
    if "allsnps" not in kwargs:
        kwargs["allsnps"] = _default_genotype_allsnps(data)
    resampling = _validate_resampling(kwargs.pop("resampling", "pairwise_counts"))
    source = (
        _fstats.f4_model_cache(
            data,
            models,
            left_base=left_base,
            right_base=right_base,
            resampling=resampling,
            verbose=verbose,
            **kwargs,
        )
        if use_cache
        else data
    )
    qp_kwargs = (
        {"resampling": resampling}
        if use_cache
        else {**kwargs, "resampling": resampling}
    )
    rows = []
    for model_i, row in enumerate(models.itertuples(index=False), start=1):
        left = _model_left_with_target(row)
        right = _as_pop_list(row.right)
        model_source = (
            _select_f4_block_cache_model(source, model_i)
            if isinstance(source, F4BlockCache)
            else source
        )
        out = qpwave(
            model_source,
            left=left,
            right=right,
            ranks=ranks,
            left_base=left_base,
            right_base=right_base,
            max_nfev=max_nfev,
            verbose=False,
            **qp_kwargs,
        )
        out.insert(0, "model", model_i)
        out.insert(1, "left", [left] * len(out))
        out.insert(2, "right", [right] * len(out))
        if "target" in models.columns:
            out.insert(1, "target", getattr(row, "target", None))
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _validate_covariance_psd(cov: np.ndarray, caller: str = "qpAdm") -> None:
    """Reject indefinite covariance, allowing eigensolver roundoff at zero."""
    if not cov.size:
        return
    scale = float(np.linalg.norm(cov, ord=np.inf))
    tolerance = 100 * np.finfo(float).eps * len(cov) * scale
    if not np.allclose(cov, cov.T, rtol=0, atol=tolerance):
        raise ValueError(f"{caller} covariance matrix must be symmetric")
    try:
        linalg.cholesky(cov, check_finite=False)
        return
    except linalg.LinAlgError:
        smallest = float(linalg.eigvalsh(cov, subset_by_index=[0, 0], check_finite=False)[0])
    if smallest < -tolerance:
        raise ValueError(
            f"{caller} covariance matrix is not positive semidefinite "
            f"(minimum eigenvalue {smallest:.6g}); cannot compute a valid chi-square test. "
            "Different missing-block patterns can cause this with pairwise resampling. "
            "Inspect population coverage and usable blocks, or recompute from genotype "
            "input with allsnps=False to use a common SNP panel."
        )


def _validate_f4_inputs(
    qpw: QpWaveStats,
    *,
    allsnps: bool,
    apply_corr: bool,
    caller: str,
    action: str,
) -> None:
    estimates = np.asarray(qpw.matrix, float).reshape(-1)
    cov = np.asarray(qpw.cov, float)
    expected_shape = (len(estimates), len(estimates))
    if cov.shape != expected_shape:
        raise ValueError(
            f"{caller} expected an f4 covariance matrix with shape {expected_shape}, "
            f"but received {cov.shape}"
        )

    bad_est = ~np.isfinite(estimates)
    bad_cov = ~np.isfinite(cov)
    if not np.any(bad_est) and not np.any(bad_cov):
        _validate_covariance_psd(cov, caller)
        return

    affected = bad_est.copy()
    if np.any(bad_cov):
        affected |= np.any(bad_cov, axis=0) | np.any(bad_cov, axis=1)
    affected_i = np.flatnonzero(affected)
    rows = qpw.f4.rows.reset_index(drop=True)
    contrast_columns = {"pop1", "pop2", "pop3", "pop4"}
    if contrast_columns.issubset(rows.columns) and np.all(affected_i < len(rows)):
        affected_rows = rows.iloc[affected_i]
        labels = _format_f4_contrasts(affected_rows)
    else:
        limit = 4
        shown = affected_i[:limit]
        labels = "\n".join(f"  - f4 contrast {stat_i}" for stat_i in shown)
        if len(affected_i) > limit:
            labels += f"\n  - ... and {len(affected_i) - limit} more"

    problems = []
    if np.any(bad_est):
        nbad = int(bad_est.sum())
        problems.append(f"{nbad} non-finite f4 estimate{'s' if nbad != 1 else ''}")
    if np.any(bad_cov):
        nbad = int(bad_cov.sum())
        problems.append(f"{nbad} non-finite covariance entr{'ies' if nbad != 1 else 'y'}")
    hint = "Likely causes: too few usable jackknife blocks or insufficient population coverage."
    if not allsnps:
        if apply_corr:
            hint += (
                " With allsnps=False, pseudohaploid singletons can also make the f2 "
                "correction unavailable. For contrasts with four distinct populations, "
                "set apply_corr=False. Alternatives: inspect the listed populations or "
                "use allsnps=True with genotype input."
            )
        else:
            hint += (
                " Inspect the listed populations or use allsnps=True with genotype input."
            )
    raise ValueError(
        f"{caller} cannot {action} because its f4 inputs contain "
        + " and ".join(problems)
        + ".\n\n"
        + _examples_heading("Affected contrasts", len(affected_i))
        + "\n"
        + labels
        + "\n\n"
        + hint
    )
