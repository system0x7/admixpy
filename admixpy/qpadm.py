from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from collections import Counter
from hashlib import sha256
import json
from numbers import Integral
import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import linalg

from . import fstats as _fstats
from .fstats import (
    F4BlockCache, _as_pop_list, _chi2_sf, _default_genotype_allsnps,
    _format_number, _format_pvalue, _format_significant, _models_frame,
    _require_unique_pops, _select_f4_block_cache_model, _validate_resampling,
)
from .qpwave import QpWaveStats, _validate_covariance_psd, _validate_f4_inputs


@dataclass
class QpAdmResult:
    target: str
    left: list[str]
    right: list[str]
    weights: pd.DataFrame
    rankdrop: pd.DataFrame
    popdrop: pd.DataFrame | None = None
    f4: pd.DataFrame | None = None
    qpwave: QpWaveStats | None = None
    weight_cov: np.ndarray | None = None

    @staticmethod
    def _display_frame(
        df: pd.DataFrame,
        cols: Sequence[str],
        max_colwidth: int | None = None,
    ) -> str:
        out = df[[c for c in cols if c in df.columns]].copy()
        for col in out.select_dtypes(include=[np.number]).columns:
            if col in {"p", "p_nested"}:
                out[col] = out[col].map(_format_pvalue)
            elif col == "se":
                out[col] = out[col].map(lambda x: _format_significant(x, 2))
            else:
                decimals = 3 if col == "weight" else 2
                out[col] = out[col].map(lambda x, d=decimals: _format_number(x, d))
        return out.to_string(index=False, max_colwidth=max_colwidth)

    @staticmethod
    def _format_number(x, decimals: int) -> str:
        return _format_number(x, decimals)

    @staticmethod
    def _format_pvalue(x) -> str:
        return _format_pvalue(x)

    def __repr__(self) -> str:
        lines = [f"QpAdmResult(target={self.target!r})", "", "weights:"]
        lines.append(self._display_frame(self.weights, ["left", "weight", "se", "z"]))
        lines.extend(["", "rankdrop:"])
        lines.append(self._display_frame(self.rankdrop, ["f4rank", "dof", "chisq", "p", "p_nested"]))
        if self.popdrop is not None:
            lines.extend(["", "popdrop:"])
            lines.append(
                self._display_frame(
                    self.popdrop,
                    ["pat", "dropped", "f4rank", "dof", "chisq", "p", "feasible", "status"],
                    max_colwidth=72,
                )
            )
        extras = []
        if self.f4 is not None:
            extras.append("f4")
        if self.qpwave is not None:
            extras.append("qpwave")
        if self.weight_cov is not None:
            extras.append("weight_cov")
        if extras:
            lines.extend(["", "extras: " + ", ".join(extras)])
        return "\n".join(lines)

    def to_dict(self) -> dict:
        out = {"weights": self.weights, "rankdrop": self.rankdrop}
        if self.popdrop is not None:
            out["popdrop"] = self.popdrop
        if self.f4 is not None:
            out["f4"] = self.f4
        if self.qpwave is not None:
            out["qpwave"] = self.qpwave
        if self.weight_cov is not None:
            out["weight_cov"] = self.weight_cov
        return out


def _regularized_solve(coeffs: np.ndarray, rhs: np.ndarray, fudge: float) -> np.ndarray:
    mat = np.asarray(coeffs, float).copy()
    mat[np.diag_indices_from(mat)] += fudge * np.trace(mat)
    try:
        return linalg.solve(mat, rhs, assume_a="gen")
    except linalg.LinAlgError:
        return linalg.pinv(mat) @ rhs


def _qpadm_opt_a(b: np.ndarray, xmat: np.ndarray, qinv: np.ndarray, fudge: float) -> np.ndarray:
    nr = xmat.shape[0]
    design = np.kron(np.eye(nr), b)
    xvec = xmat.reshape(-1)
    coeffs = design @ qinv @ design.T
    rhs = design @ qinv @ xvec
    return _regularized_solve(coeffs, rhs, fudge).reshape(nr, b.shape[0])


def _qpadm_opt_b(a: np.ndarray, xmat: np.ndarray, qinv: np.ndarray, fudge: float) -> np.ndarray:
    nc = xmat.shape[1]
    design = np.kron(a, np.eye(nc))
    xvec = xmat.reshape(-1)
    coeffs = design.T @ qinv @ design
    rhs = design.T @ qinv @ xvec
    return _regularized_solve(coeffs, rhs, fudge).reshape(a.shape[1], nc)


def qpadm_weights(
    xmat: np.ndarray,
    qinv: np.ndarray,
    rank: int,
    fudge: float = 0.0001,
    iterations: int = 20,
) -> dict:
    xmat = np.asarray(xmat, float)
    if xmat.ndim != 2:
        raise ValueError("qpAdm requires a two-dimensional f4 matrix")
    if rank < 0 or rank > min(xmat.shape):
        raise ValueError(
            f"qpAdm rank {rank} is incompatible with f4 matrix shape {xmat.shape}; "
            f"rank must be between 0 and {min(xmat.shape)}. "
            "Check the number of source/left and right/reference populations "
            "and the argument order: qpadm(data, target, left, right)."
        )
    if rank == 0:
        return {"weights": np.ones(1), "A": np.zeros((xmat.shape[0], 0)), "B": np.zeros((0, xmat.shape[1]))}
    _, _, vt = np.linalg.svd(xmat, full_matrices=False)
    b = vt[:rank, :]
    a = xmat @ b.T
    for _ in range(iterations):
        a = _qpadm_opt_a(b, xmat, qinv, fudge)
        b = _qpadm_opt_b(a, xmat, qinv, fudge)
    x = np.column_stack([a, np.ones(a.shape[0])]).T
    y = np.concatenate([np.zeros(rank), [1.0]])
    rhs = x.T @ x
    lhs = x.T @ y
    try:
        w = linalg.solve(rhs, lhs, assume_a="sym")
    except linalg.LinAlgError:
        w = linalg.pinv(rhs) @ lhs
    return {"weights": w / np.sum(w), "A": a, "B": b}


def _qpadm_dof(nrow: int, ncol: int, rank: int) -> int:
    return (nrow - rank) * (ncol - rank)


def qpadm_fit(xmat: np.ndarray, qinv: np.ndarray, rank: int, fudge: float = 0.0001, iterations: int = 20) -> dict:
    xmat = np.asarray(xmat, float)
    fit = qpadm_weights(xmat, qinv, rank, fudge=fudge, iterations=iterations)
    fitted = fit["A"] @ fit["B"]
    resid = (xmat - fitted).reshape(-1)
    chisq = float(resid @ qinv @ resid)
    dof = _qpadm_dof(xmat.shape[0], xmat.shape[1], rank)
    return {
        "f4rank": rank,
        "dof": dof,
        "chisq": chisq,
        "p": _chi2_sf(chisq, dof),
        "fitted": fitted,
        "residual": xmat - fitted,
        **fit,
    }


def qpadm_rankdrop(xmat: np.ndarray, qinv: np.ndarray, fudge: float = 0.0001, iterations: int = 20) -> pd.DataFrame:
    max_rank = xmat.shape[0] - 1
    rows = []
    for rank in range(max_rank, -1, -1):
        fit = _fstats.qpadm_fit(xmat, qinv, rank, fudge=fudge, iterations=iterations)
        rows.append({k: fit[k] for k in ("f4rank", "dof", "chisq", "p")})
    out = pd.DataFrame(rows)
    out["dofdiff"] = out["dof"].shift(-1) - out["dof"]
    out["chisqdiff"] = out["chisq"].shift(-1) - out["chisq"]
    out["p_nested"] = [_chi2_sf(c, int(d)) if np.isfinite(c) and np.isfinite(d) else float("nan") for c, d in zip(out["chisqdiff"], out["dofdiff"])]
    return out


def qpadm_popdrop(
    xmat: np.ndarray,
    qinv: np.ndarray,
    sources: Sequence[str],
    fudge: float = 0.0001,
    iterations: int = 20,
    *,
    cov: np.ndarray | None = None,
    fudge_twice: bool = False,
) -> pd.DataFrame:
    """Fit source subsets on the existing SNP panel.

    With raw ``cov``, each subset uses its own regularization, matching an
    independent qpAdm fit. Nested comparisons use the full model's regularized
    covariance for both models. If only ``qinv`` is supplied, its inverse is
    treated as an already regularized covariance and is not regularized again.
    """
    xmat, qinv = np.asarray(xmat, float), np.asarray(qinv, float)
    sources = list(sources)
    nsrc = len(sources)
    _require_unique_pops(sources, "left")
    if xmat.ndim != 2 or len(xmat) != nsrc or not nsrc:
        raise ValueError("qpadm_popdrop requires one f4 matrix row per source")
    expected_shape = (xmat.size, xmat.size)
    if qinv.shape != expected_shape:
        raise ValueError(f"qpadm_popdrop expected qinv shape {expected_shape}, received {qinv.shape}")
    if cov is None:
        if not np.isfinite(qinv).all():
            raise ValueError("qpadm_popdrop qinv must contain only finite values")
        _validate_covariance_psd(qinv)
        try:
            common_cov = linalg.inv(qinv)
        except linalg.LinAlgError:
            raise ValueError("qpadm_popdrop requires raw cov when qinv is singular") from None
    else:
        cov = np.asarray(cov, float)
        if cov.shape != expected_shape:
            raise ValueError(f"qpadm_popdrop expected cov shape {expected_shape}, received {cov.shape}")
        # Raw covariance is authoritative, including its regularization.
        qinv = _qinv_from_cov(cov, fudge, fudge_twice)
        common_cov = _regularize_covariance(cov, fudge, fudge_twice)
    ncol = xmat.shape[1]
    rows = []
    subsets = []
    for nkeep in range(nsrc, 0, -1):
        for keep_tuple in combinations(range(nsrc), nkeep):
            keep = list(keep_tuple)
            flat = np.concatenate([np.arange(i * ncol, (i + 1) * ncol) for i in keep])
            submat = xmat[keep, :]
            if nkeep == nsrc:
                subqinv = qinv
            elif cov is None:
                subqinv = _qinv_from_cov(common_cov[np.ix_(flat, flat)], fudge=0)
            else:
                subqinv = _qinv_from_cov(cov[np.ix_(flat, flat)], fudge, fudge_twice)
            rank = len(keep) - 1
            fit = _fstats.qpadm_fit(submat, subqinv, rank, fudge=fudge, iterations=iterations)
            weights = np.full(nsrc, np.nan)
            weights[keep] = fit["weights"]
            pat = "".join("0" if i in keep else "1" for i in range(nsrc))
            dropped = [sources[i] for i in range(nsrc) if i not in keep]
            row = {
                "pat": pat,
                "dropped": ",".join(dropped),
                "wt": len(dropped),
                "f4rank": rank,
                "dof": fit["dof"],
                "chisq": fit["chisq"],
                "p": fit["p"],
                "feasible": bool(np.all((weights[keep] >= 0) & (weights[keep] <= 1))),
            }
            for src, weight in zip(sources, weights):
                row[src] = weight
            rows.append(row)
            subsets.append((keep, flat))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    best, parents = _popdrop_nested_parents(out, nsrc)
    nested_chisq = np.full(len(out), np.nan)
    for i in np.flatnonzero(best):
        keep, flat = subsets[i]
        if len(keep) == nsrc or cov is None or fudge == 0:
            nested_chisq[i] = out.iloc[i]["chisq"]
        else:
            shared_qinv = _qinv_from_cov(common_cov[np.ix_(flat, flat)], fudge=0)
            nested_fit = _fstats.qpadm_fit(
                xmat[keep], shared_qinv, len(keep) - 1,
                fudge=fudge, iterations=iterations,
            )
            nested_chisq[i] = nested_fit["chisq"]
    dofdiff = np.full(len(out), np.nan)
    chisqdiff = np.full(len(out), np.nan)
    p_nested = np.full(len(out), np.nan)
    parent_patterns = [None] * len(out)
    for child, parent in enumerate(parents):
        if parent < 0:
            continue
        parent_patterns[child] = out.iloc[parent]["pat"]
        dd = out.iloc[child]["dof"] - out.iloc[parent]["dof"]
        cd = nested_chisq[child] - nested_chisq[parent]
        # Finite iteration fits can fail to be monotonic. Do not report a
        # negative likelihood-ratio statistic as a passing nested test.
        tol = 1e-10 * max(1.0, abs(nested_chisq[child]), abs(nested_chisq[parent]))
        if np.isfinite(cd) and cd >= -tol and dd > 0:
            cd = max(0.0, cd)
            p_nested[child] = _chi2_sf(cd, int(dd))
        dofdiff[child], chisqdiff[child] = dd, cd
    out["best"] = best
    out["nested_parent"] = parent_patterns
    out["nested_chisq"] = nested_chisq
    out["dofdiff"] = dofdiff
    out["chisqdiff"] = chisqdiff
    out["p_nested"] = p_nested
    out["status"] = np.where(~np.isfinite(out["p"]), "NA", np.where(out["p"] > 0.05, "PASS", "FAIL"))
    return out.sort_values(["dof", "pat"]).reset_index(drop=True)


def _popdrop_nested_parents(out: pd.DataFrame, nsources: int):
    # Compare every one-source drop to the full model. At each smaller size,
    # select the feasible child with lowest chi-square and a genuine selected
    # parent. Resolve ties by pattern rather than input row order.
    n = len(out)
    best = np.zeros(n, dtype=bool)
    parents = np.full(n, -1, dtype=int)
    pat_arr = out["pat"].to_numpy()
    feasible_arr = out["feasible"].to_numpy()
    chisq_arr = out["chisq"].to_numpy()
    index = {pat: i for i, pat in enumerate(pat_arr)}
    full = index.get("0" * nsources)
    if full is None or not np.isfinite(chisq_arr[full]):
        return best, parents
    best[full] = True
    previous = [full]
    for ndrop in range(1, nsources):
        candidates: dict[int, list[int]] = {}
        for parent in previous:
            pat = pat_arr[parent]
            for k, bit in enumerate(pat):
                if bit != "0":
                    continue
                child = index.get(pat[:k] + "1" + pat[k + 1:])
                if child is not None and np.isfinite(chisq_arr[child]):
                    if ndrop == 1 or feasible_arr[child]:
                        candidates.setdefault(child, []).append(parent)
        if not candidates:
            break
        key = lambda i: (chisq_arr[i], pat_arr[i])
        chosen = list(candidates) if ndrop == 1 else [min(candidates, key=key)]
        for child in chosen:
            best[child] = True
            parents[child] = min(candidates[child], key=key)
        previous = chosen
    return best, parents


def _regularize_covariance(cov: np.ndarray, fudge: float, fudge_twice: bool = False) -> np.ndarray:
    cov = np.asarray(cov, float).copy()
    cov[np.diag_indices_from(cov)] += fudge * np.trace(cov)
    if fudge_twice:
        cov[np.diag_indices_from(cov)] += fudge * np.trace(cov)
    return cov


def _qinv_from_cov(cov: np.ndarray, fudge: float, fudge_twice: bool = False) -> np.ndarray:
    cov = np.asarray(cov, float).copy()
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square")
    nonfinite = int(np.size(cov) - np.isfinite(cov).sum())
    if nonfinite:
        suffix = "entry" if nonfinite == 1 else "entries"
        raise ValueError(
            f"Covariance matrix contains {nonfinite} non-finite {suffix}; "
            "cannot compute its inverse"
        )
    _validate_covariance_psd(cov)
    cov = _regularize_covariance(cov, fudge, fudge_twice)
    try:
        return linalg.inv(cov)
    except linalg.LinAlgError:
        return linalg.pinv(cov)


def _validate_qpadm_f4(qpw: QpWaveStats, *, allsnps: bool, apply_corr: bool) -> None:
    _validate_f4_inputs(
        qpw,
        allsnps=allsnps,
        apply_corr=apply_corr,
        caller="qpAdm",
        action="fit the model",
    )


def _weights_covariance(qpw: QpWaveStats, qinv: np.ndarray, rank: int, fudge: float, iterations: int) -> np.ndarray:
    loo = qpw.loo
    if loo is None:
        return np.full((len(qpw.row_pops), len(qpw.row_pops)), np.nan)
    if len(qpw.row_pops) == 1:
        return np.full((1, 1), np.nan)
    wmat = []
    for i in range(loo.shape[2]):
        try:
            wmat.append(qpadm_weights(loo[:, :, i], qinv, rank, fudge=fudge, iterations=iterations)["weights"])
        except Exception:
            wmat.append(np.full(len(qpw.row_pops), np.nan))
    wmat = np.asarray(wmat, float)
    keep = np.isfinite(wmat).all(axis=1)
    wmat = wmat[keep]
    if len(wmat) < 2:
        return np.full((wmat.shape[1], wmat.shape[1]), np.nan)
    num_replicates = len(wmat)
    scale = (num_replicates - 1) / math.sqrt(num_replicates)
    return np.cov(wmat * scale, rowvar=False)


def _validate_qpadm_populations(target, sources, right, left_base=None, right_base=None):
    if not isinstance(target, str):
        raise TypeError("target must be a population name string; positional qpadm order is qpadm(data, target, left, right)")
    _require_unique_pops(sources, "left")
    _require_unique_pops(right, "right")
    if not sources:
        raise ValueError("At least one source/left population is required")
    if len(right) < 2:
        raise ValueError("At least two right/reference populations are required")
    if target in sources:
        raise ValueError(f"target {target!r} should not also appear among source/left populations")
    if left_base is not None and left_base != target:
        raise ValueError("qpAdm left_base must be the target population; changing it changes the meaning of the source weights")
    if right_base is not None and right_base not in right:
        raise ValueError("right_base must be included in right")
    if len(sources) > len(right):
        raise ValueError(
            f"qpAdm has {len(sources)} source/left populations but only {len(right)} "
            "right/reference populations; the requested rank exceeds the number "
            "of right contrasts. Add right populations or reduce the sources."
        )


def qpadm(
    data,
    target: str,
    left: Sequence[str] | None = None,
    right: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
    fudge: float = 0.0001,
    fudge_twice: bool = False,
    iterations: int = 20,
    getcov: bool = True,
    return_f4: bool = False,
    return_stats: bool = False,
    return_cov: bool = False,
    verbose: bool = True,
    *,
    popdrop: bool = True,
    **kwargs,
) -> QpAdmResult:
    if not isinstance(target, str):
        raise TypeError("target must be a population name string; positional qpadm order is qpadm(data, target, left, right)")
    if left is not None and sources is not None:
        raise ValueError(
            "Specify only one of left or sources; sources is an alias for left. "
            "The positional qpadm order is qpadm(data, target, left, right); "
            "there is no separate positional outgroup argument."
        )
    sources = _as_pop_list(left if sources is None else sources)
    if right is None:
        raise ValueError("right populations are required")
    right = _as_pop_list(right)
    _validate_qpadm_populations(target, sources, right, kwargs.get("left_base"), kwargs.get("right_base"))
    if "allsnps" not in kwargs:
        kwargs["allsnps"] = _default_genotype_allsnps(data)
    left_full = [target] + [p for p in sources if p != target]
    qpw = _fstats.qpwave_f4stats(data, left=left_full, right=right, verbose=verbose, **kwargs)
    _validate_qpadm_f4(
        qpw,
        allsnps=bool(kwargs["allsnps"]),
        apply_corr=bool(kwargs.get("apply_corr", True)),
    )
    xmat = qpw.matrix
    qinv = _qinv_from_cov(qpw.cov, fudge=fudge, fudge_twice=fudge_twice)
    rank = len(sources) - 1
    fit = _fstats.qpadm_fit(xmat, qinv, rank, fudge=fudge, iterations=iterations)
    wcov = _fstats._weights_covariance(qpw, qinv, rank, fudge, iterations) if getcov else np.full((len(sources), len(sources)), np.nan)
    se = np.sqrt(np.diag(wcov)) if wcov.ndim == 2 and wcov.shape[0] == len(sources) else np.full(len(sources), np.nan)
    weights = pd.DataFrame({"target": target, "left": sources, "weight": fit["weights"], "se": se})
    weights["z"] = weights["weight"] / weights["se"]
    rankdrop = qpadm_rankdrop(xmat, qinv, fudge=fudge, iterations=iterations)
    popdrop_result = _fstats.qpadm_popdrop(
        xmat, qinv, sources, fudge=fudge, iterations=iterations,
        cov=qpw.cov, fudge_twice=fudge_twice,
    ) if popdrop else None
    f4 = None
    qpw_out = qpw if return_stats else None
    weight_cov = wcov if return_cov else None
    if return_f4:
        f4 = qpw.f4.rows.copy()
        f4["est"] = xmat.reshape(-1)
        f4["fit"] = fit["fitted"].reshape(-1)
        f4["diff"] = f4["est"] - f4["fit"]
    return QpAdmResult(
        target=target,
        left=sources,
        right=list(right),
        weights=weights,
        rankdrop=rankdrop,
        popdrop=popdrop_result,
        f4=f4,
        qpwave=qpw_out,
        weight_cov=weight_cov,
    )


def qpadm_multi(
    data,
    models,
    use_cache: bool = True,
    full_results: bool = True,
    verbose: bool = True,
    **kwargs,
):
    models = _models_frame(models)
    if "target" not in models.columns:
        raise ValueError("models must contain a 'target' column for qpadm_multi")
    for model_i, row in enumerate(models.itertuples(index=False), start=1):
        try:
            _validate_qpadm_populations(
                row.target, _as_pop_list(row.left), _as_pop_list(row.right),
                kwargs.get("left_base"), kwargs.get("right_base"),
            )
        except (ValueError, TypeError) as err:
            raise ValueError(f"Model {model_i}: {err}") from None
    if models.empty:
        return pd.DataFrame()
    qpadm_keys = {"fudge", "fudge_twice", "iterations", "getcov", "return_f4", "return_stats", "return_cov", "popdrop"}
    if "allsnps" not in kwargs:
        kwargs["allsnps"] = _default_genotype_allsnps(data)
    resampling = _validate_resampling(kwargs.pop("resampling", "pairwise_counts"))
    qpadm_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in qpadm_keys}
    if not full_results:
        qpadm_kwargs.update(getcov=False, popdrop=False, return_f4=False, return_stats=False, return_cov=False)
    source = (
        _fstats.f4_model_cache(data, models, resampling=resampling, verbose=verbose, **kwargs)
        if use_cache
        else data
    )
    qp_kwargs = (
        {**qpadm_kwargs, "resampling": resampling,
         **{key: kwargs[key] for key in ("left_base", "right_base") if key in kwargs}}
        if use_cache
        else {**kwargs, **qpadm_kwargs, "resampling": resampling}
    )
    rows = []
    for model_i, row in enumerate(models.itertuples(index=False), start=1):
        model_source = (
            _select_f4_block_cache_model(source, model_i)
            if isinstance(source, F4BlockCache)
            else source
        )
        res = qpadm(
            model_source,
            target=row.target,
            left=_as_pop_list(row.left),
            right=_as_pop_list(row.right),
            verbose=False,
            **qp_kwargs,
        )
        if full_results:
            rows.append({"model": model_i, "target": row.target, "left": _as_pop_list(row.left), "right": _as_pop_list(row.right), "result": res})
        else:
            rankdrop = res.rankdrop.copy()
            rankdrop.insert(0, "model", model_i)
            rankdrop.insert(1, "target", row.target)
            rankdrop.insert(2, "left", [_as_pop_list(row.left)] * len(rankdrop))
            rankdrop.insert(3, "right", [_as_pop_list(row.right)] * len(rankdrop))
            rows.append(rankdrop)
    return pd.DataFrame(rows) if full_results else pd.concat(rows, ignore_index=True)



def _population_list(pops, name):
    if isinstance(pops, str):
        raise TypeError(f"{name} must be a sequence of population names, not a string")
    values = list(pops)
    if any(not isinstance(pop, str) or not pop.strip() for pop in values):
        raise ValueError(f"{name} must contain nonempty population names")
    return values


def _validate_rotation_inputs(leftright, target, rightfix):
    """Validate and normalize inputs, including generators."""
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be a nonempty population name")
    leftright = _population_list(leftright, "leftright")
    rightfix = [] if rightfix is None else _population_list(rightfix, "rightfix")
    if not leftright:
        raise ValueError("leftright must contain at least one population")
    counts = Counter(leftright + [target] + rightfix)
    duplicates = sorted(pop for pop, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate populations across inputs: {duplicates}")
    return leftright, target, rightfix


def _rotation_splits(pops, rightfix_count=0, *, source_sizes=None):
    """Split candidates into sources and rotating references."""
    pops = _population_list(pops, "pops")
    if len(set(pops)) != len(pops):
        raise ValueError("pops must contain unique population names")
    if isinstance(rightfix_count, bool) or not isinstance(rightfix_count, Integral) or rightfix_count < 0:
        raise ValueError("rightfix_count must be a nonnegative integer")
    max_left = min(len(pops), (len(pops) + rightfix_count - 1) // 2)
    if max_left < 1:
        raise ValueError("No testable rotations: add candidates or fixed right populations")
    sizes = list(range(1, max_left + 1)) if source_sizes is None else list(source_sizes)
    if not sizes or any(isinstance(n, bool) or not isinstance(n, Integral) or not 1 <= n <= max_left for n in sizes):
        raise ValueError(f"source_sizes must contain integers between 1 and {max_left}")
    rows = []
    for left_size in sorted(set(sizes)):
        for left_tuple in combinations(pops, left_size):
            left = list(left_tuple)
            rows.append({"left": left, "right": [pop for pop in pops if pop not in left]})
    return rows


def qpadm_rotate_models(leftright, target, rightfix=None, *, source_sizes=None):
    """Build a model table for admixpy.qpadm_multi, without reading genotypes."""
    leftright, target, rightfix = _validate_rotation_inputs(leftright, target, rightfix)
    return pd.DataFrame([
        {"target": target, "left": split["left"], "right": rightfix + split["right"]}
        for split in _rotation_splits(leftright, len(rightfix), source_sizes=source_sizes)
    ], columns=["target", "left", "right"])



@dataclass
class QpAdmRotationResult:
    models: pd.DataFrame
    weights: pd.DataFrame
    errors: pd.DataFrame
    settings: dict

    def __repr__(self):
        completed = int((self.models["status"] == "ok").sum())
        return (f"QpAdmRotationResult(models={len(self.models)}, "
                f"completed={completed}, failed={len(self.errors)}, "
                f"weights={len(self.weights)})")


def qpadm_rotate(
    data, leftright, target, rightfix=None, *, source_sizes=None,
    full_results=False, getcov=False, use_cache=True, on_error="raise",
    verbose=True, **kwargs,
) -> QpAdmRotationResult:
    """Rotate candidate sources onto the reference panel and test each model.

    ``leftright`` contains candidate sources; unused candidates move to the
    right after ``rightfix``. Only source counts with positive fit degrees of
    freedom are generated. ``source_sizes`` restricts these counts.

    Returns models, weights, errors, and settings. Models retain population
    lists and generation order; p-values are not used to rank models.
    ``full_results=True`` retains weights and feasibility; additionally set
    ``getcov=True`` for standard errors. Otherwise feasibility is missing.
    Population-drop fits are skipped. Other genotype and fitting options are
    forwarded as in qpadm_multi. Cached input must cover the generated models
    in their generation order.

    ``on_error='record'`` records model-level ValueError/linear algebra errors
    and continues. Invalid arguments and cache construction failures still
    raise. The default ``on_error='raise'`` stops on the first failed model.
    Export individual tables with pandas, e.g. result.weights.to_csv(...).
    """
    if on_error not in {"raise", "record"}:
        raise ValueError("on_error must be 'raise' or 'record'")
    if getcov and not full_results:
        raise ValueError("getcov=True requires full_results=True")
    unsupported = {"popdrop", "return_f4", "return_stats", "return_cov", "sources", "left", "right"} & kwargs.keys()
    if unsupported:
        raise TypeError(f"Unsupported rotation options: {', '.join(sorted(unsupported))}")
    models = qpadm_rotate_models(leftright, target, rightfix, source_sizes=source_sizes)
    for row in models.itertuples(index=False):
        _validate_qpadm_populations(target, row.left, row.right, kwargs.get("left_base"), kwargs.get("right_base"))
    kwargs.setdefault("allsnps", _default_genotype_allsnps(data))
    kwargs["resampling"] = _validate_resampling(kwargs.get("resampling", "pairwise_counts"))
    fit_options = {key: kwargs.pop(key, default) for key, default in
                   (("fudge", .0001), ("fudge_twice", False), ("iterations", 20))}
    from . import __version__
    settings = dict(kwargs, **fit_options, admixpy_version=__version__,
                    source_sizes=sorted(set(models.left.map(len))),
                    full_results=full_results, getcov=getcov, use_cache=use_cache,
                    on_error=on_error)
    if verbose:
        print(f"Evaluating {len(models)} qpAdm rotation models...")
    source = _fstats.f4_model_cache(data, models, verbose=verbose, **kwargs) if use_cache else data
    qp_kwargs = ({key: kwargs[key] for key in ("resampling", "left_base", "right_base") if key in kwargs}
                 if use_cache else kwargs)
    model_rows, weight_rows, errors = [], [], []
    for model_i, row in enumerate(models.itertuples(index=False), start=1):
        identity = [target, row.left, row.right, kwargs.get("left_base") or target,
                    kwargs.get("right_base") or row.right[0]]
        model_id = sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:20]
        summary = dict(model=model_id, target=target, left=row.left, right=row.right,
                       n_sources=len(row.left), f4rank=len(row.left)-1,
                       dof=len(row.right)-len(row.left), chisq=np.nan, p=np.nan,
                       feasible=pd.NA, status="ok")
        model_source = (_select_f4_block_cache_model(source, model_i)
                        if isinstance(source, F4BlockCache) else source)
        try:
            result = qpadm(model_source, target, row.left, row.right, popdrop=False,
                           getcov=getcov, verbose=False, **fit_options, **qp_kwargs)
        except (ValueError, np.linalg.LinAlgError) as error:
            if on_error == "raise":
                raise
            summary["status"] = "error"
            errors.append(dict(model=model_id, error_type=type(error).__name__, message=str(error)))
        else:
            fit = result.rankdrop.loc[result.rankdrop.f4rank == len(row.left)-1].iloc[0]
            summary.update(chisq=fit.chisq, p=fit.p)
            if full_results:
                summary["feasible"] = bool(result.weights.weight.between(0, 1).all())
                for weight in result.weights.itertuples(index=False):
                    weight_rows.append(dict(model=model_id, source=weight.left,
                                            weight=weight.weight, se=weight.se, z=weight.z))
        model_rows.append(summary)
    output = pd.DataFrame(model_rows)
    output["feasible"] = output["feasible"].astype("boolean")
    return QpAdmRotationResult(
        models=output,
        weights=pd.DataFrame(weight_rows, columns=["model", "source", "weight", "se", "z"]),
        errors=pd.DataFrame(errors, columns=["model", "error_type", "message"]),
        settings=settings,
    )
