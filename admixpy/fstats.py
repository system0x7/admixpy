from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement, product
import math
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats as scipy_stats

from .genotypes import AfData, anygeno_to_afs, discard_from_aftable, get_block_lengths, is_polymorphic, iter_geno_to_afs


def _log(message: str, verbose: bool):
    if verbose:
        print(message, flush=True)


def _log_block(stat: str, b: int, total: int, start: int, stop: int, verbose: bool) -> None:
    # Single carriage-returned status line, updated ~20 times across the run.
    if not verbose or total <= 0:
        return
    step = max(1, total // 20)
    is_last = b == total - 1
    if b == 0 or is_last or (b + 1) % step == 0:
        end = "\n" if is_last else ""
        print(
            f"\rComputing {stat} block {b + 1}/{total}: SNP rows {start + 1}-{stop}      ",
            end=end,
            flush=True,
        )


def _chi2_sf(x: float, df: int) -> float:
    if not np.isfinite(x) or df <= 0:
        return float("nan")
    return float(scipy_stats.chi2.sf(x, df))


def _format_number(x, decimals: int) -> str:
    if not np.isfinite(x):
        return "NaN"
    rounded = float(np.round(x, decimals))
    return np.format_float_positional(rounded, precision=decimals, fractional=True, trim="-")


def _format_significant(x, digits: int = 6) -> str:
    if not np.isfinite(x):
        return "NaN"
    return f"{x:.{digits}g}"


def _format_pvalue(x) -> str:
    if not np.isfinite(x):
        return "NaN"
    if x == 0:
        return "0"
    if abs(x) < 0.001:
        return f"{x:.3g}"
    return _format_number(x, 3)


def _nanmean(arr: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def format_fstats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-formatted copy of an f-statistics result frame."""
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        if col == "p":
            out[col] = out[col].map(_format_pvalue)
        elif col == "z":
            out[col] = out[col].map(lambda x: _format_number(x, 2))
        elif col in {"est", "se"}:
            out[col] = out[col].map(_format_significant)
    return out


class FStatsFrame(pd.DataFrame):
    """DataFrame that keeps raw numeric values but displays f-stats compactly."""

    @property
    def _constructor(self):
        return FStatsFrame

    def __repr__(self) -> str:
        return format_fstats(pd.DataFrame(self)).to_string()

    def _repr_html_(self):
        return None


@dataclass
class F2Blocks:
    data: np.ndarray
    pops1: list[str]
    pops2: list[str]
    block_lengths: np.ndarray
    stat: str = "f2"
    snp_counts: np.ndarray | None = None
    fst_num: np.ndarray | None = None
    fst_den: np.ndarray | None = None

    def subset(self, pops=None, pops2=None) -> "F2Blocks":
        pops = self.pops1 if pops is None else list(pops)
        pops2 = pops if pops2 is None else list(pops2)
        i = [self.pops1.index(p) for p in pops]
        j = [self.pops2.index(p) for p in pops2]
        ix = np.ix_(i, j, range(self.data.shape[2]))

        def take(arr):
            return None if arr is None else arr[ix]

        return F2Blocks(
            self.data[ix],
            pops,
            pops2,
            self.block_lengths.copy(),
            self.stat,
            take(self.snp_counts),
            take(self.fst_num),
            take(self.fst_den),
        )

    def pair(self, pop1: str, pop2: str) -> np.ndarray:
        return self.data[self.pops1.index(pop1), self.pops2.index(pop2), :]

    def pair_counts(self, pop1: str, pop2: str) -> np.ndarray | None:
        if self.snp_counts is None:
            return None
        return self.snp_counts[self.pops1.index(pop1), self.pops2.index(pop2), :]

    def pair_fst_components(self, pop1: str, pop2: str) -> tuple[np.ndarray, np.ndarray] | None:
        if self.fst_num is None or self.fst_den is None:
            return None
        i, j = self.pops1.index(pop1), self.pops2.index(pop2)
        num, den = self.fst_num[i, j, :], self.fst_den[i, j, :]
        if not np.any(np.isfinite(num) & np.isfinite(den)):
            return None
        return num, den

    def select_blocks(self, keep: Sequence[bool]) -> "F2Blocks":
        keep = np.asarray(keep, bool)

        def take(arr):
            return None if arr is None else arr[:, :, keep]

        return F2Blocks(
            self.data[:, :, keep],
            self.pops1,
            self.pops2,
            self.block_lengths[keep],
            self.stat,
            take(self.snp_counts),
            take(self.fst_num),
            take(self.fst_den),
        )


@dataclass
class BlockStats:
    rows: pd.DataFrame
    blocks: np.ndarray | None
    block_lengths: np.ndarray
    stat: str
    loo: np.ndarray | None = None
    est: np.ndarray | None = None
    cov: np.ndarray | None = None
    variances: np.ndarray | None = None
    resampling: str = "pairwise_counts"

    @property
    def se(self) -> np.ndarray:
        if self.cov is None:
            if self.variances is None:
                return np.full(len(self.rows), np.nan)
            return np.sqrt(np.asarray(self.variances, float))
        return np.sqrt(np.diag(self.cov))

    @property
    def z(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.est / self.se

    @property
    def p(self) -> np.ndarray:
        return np.array([math.erfc(abs(z) / math.sqrt(2)) if np.isfinite(z) else float("nan") for z in self.z])

    def to_frame(self, round_z: int | None = None, round_p: int | None = None) -> pd.DataFrame:
        out = self.rows.copy()
        ratio_num = getattr(self, "ratio_num", None)
        # In allsnps mode the per-stat per-block SNP counts are attached as
        # `snp_counts`; report each statistic using its own effective block sizes.
        snp_counts = getattr(self, "snp_counts", None)
        if ratio_num is not None:
            est_vec = np.asarray(self.est, float)
            se_vec = self.se
            with np.errstate(invalid="ignore", divide="ignore"):
                z = est_vec / se_vec
        elif snp_counts is not None and self.blocks is not None:
            n = len(self.rows)
            est_vec = np.empty(n, dtype=float)
            se_vec = np.empty(n, dtype=float)
            for i in range(n):
                e, v = _jack_stats_per_stat(self.blocks[i], snp_counts[i])
                est_vec[i] = e
                se_vec[i] = math.sqrt(v) if np.isfinite(v) else float("nan")
            with np.errstate(invalid="ignore", divide="ignore"):
                z = est_vec / se_vec
        else:
            est_vec = np.asarray(self.est, float) if self.est is not None else np.full(len(self.rows), np.nan)
            se_vec = self.se
            z = self.z
        out["est"] = est_vec
        out["se"] = se_vec
        out["z"] = np.round(z, round_z) if round_z is not None else z
        p = np.array([math.erfc(abs(zz) / math.sqrt(2)) if np.isfinite(zz) else float("nan") for zz in z])
        out["p"] = np.round(p, round_p) if round_p is not None else p
        return FStatsFrame(out)


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


@dataclass
class F4ModelCache:
    blocks: F2Blocks
    models: pd.DataFrame | None = None
    left_pops: list[str] | None = None
    right_pops: list[str] | None = None


@dataclass
class F4BlockCache:
    stats: BlockStats
    models: pd.DataFrame | None = None
    allsnps: bool = False
    deferred_covariance: bool = False


def _select_f4_block_cache_model(cache: F4BlockCache, model: int) -> F4BlockCache:
    if "model" not in cache.stats.rows.columns:
        return cache
    take = np.flatnonzero(cache.stats.rows["model"].to_numpy() == model)
    if take.size == 0:
        raise ValueError(f"F4 cache does not contain model {model}")
    source = cache.stats
    stats = BlockStats(
        rows=source.rows.iloc[take].reset_index(drop=True),
        blocks=None if source.blocks is None else np.asarray(source.blocks)[take],
        block_lengths=np.asarray(source.block_lengths, float).copy(),
        stat=source.stat,
        loo=None if source.loo is None else np.asarray(source.loo)[take],
        est=None if source.est is None else np.asarray(source.est)[take],
        cov=None if source.cov is None else np.asarray(source.cov)[np.ix_(take, take)],
        variances=None if source.variances is None else np.asarray(source.variances)[take],
        resampling=source.resampling,
    )
    for name in ("influence", "contributes", "snp_counts"):
        value = getattr(source, name, None)
        if value is not None:
            setattr(stats, name, np.asarray(value)[take])
    if hasattr(source, "nominal_block_lengths"):
        stats.nominal_block_lengths = np.asarray(source.nominal_block_lengths, float).copy()
    models = None if cache.models is None else cache.models.iloc[[model - 1]].reset_index(drop=True)
    return F4BlockCache(
        stats=stats, models=models, allsnps=cache.allsnps,
        deferred_covariance=cache.deferred_covariance,
    )


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


def _block_mean(arr: np.ndarray, block_lengths: Sequence[int]) -> np.ndarray:
    out = np.empty(arr.shape[:2] + (len(block_lengths),), dtype=float)
    start = 0
    for b, n in enumerate(block_lengths):
        stop = start + int(n)
        with np.errstate(invalid="ignore"):
            out[:, :, b] = _nanmean(arr[:, :, start:stop], axis=2)
        start = stop
    return out


def _outer_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.T[:, None, :] * b.T[None, :, :]


def _singleton_observation_rows(*pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    affected = None
    for afs, counts in pairs:
        values = np.isfinite(afs) & np.isfinite(counts) & (counts < 2)
        rows = values if values.ndim == 1 else np.any(values, axis=tuple(range(1, values.ndim)))
        affected = rows.copy() if affected is None else affected | rows
    return np.asarray([], dtype=bool) if affected is None else affected


def _count_affected_blocks(affected_rows: np.ndarray, block_lengths: Sequence[int]) -> int:
    affected_rows = np.asarray(affected_rows, bool)
    start = 0
    affected_blocks = 0
    for n in block_lengths:
        stop = start + int(n)
        affected_blocks += int(np.any(affected_rows[start:stop]))
        start = stop
    if start != len(affected_rows):
        raise ValueError("Block lengths must sum to the number of SNP rows")
    return affected_blocks


def _singleton_warning_message(
    stat: str,
    apply_corr: bool,
    affected_blocks: int,
    total_blocks: int,
) -> str:
    impact = "in 1 block" if total_blocks == 1 else f"in {affected_blocks} of {total_blocks} blocks"
    if apply_corr:
        return (
            f"{stat} bias correction requires at least two independent allele observations; "
            f"excluding affected SNP values with count < 2 {impact}"
        )
    return (
        f"{stat} used values with fewer than two independent allele observations {impact}; "
        "estimates may be sampling-biased because apply_corr=False"
    )


@dataclass
class _SingletonWarningSummary:
    stat: str
    apply_corr: bool
    affected_blocks: int = 0

    def observe(
        self,
        block_lengths: Sequence[int],
        *pairs: tuple[np.ndarray, np.ndarray],
        population_labels: Sequence[Sequence[str]] | None = None,
    ) -> None:
        affected_rows = _singleton_observation_rows(*pairs)
        self.affected_blocks += _count_affected_blocks(affected_rows, block_lengths)

    def warn(self, total_blocks: int, *, stacklevel: int) -> None:
        if not self.affected_blocks:
            return
        warnings.warn(
            _singleton_warning_message(
                self.stat,
                self.apply_corr,
                self.affected_blocks,
                total_blocks,
            ),
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def _warn_singleton_observations(
    stat: str,
    apply_corr: bool,
    *pairs: tuple[np.ndarray, np.ndarray],
    block_lengths: Sequence[int] | None = None,
    population_labels: Sequence[Sequence[str]] | None = None,
) -> None:
    affected_rows = _singleton_observation_rows(*pairs)
    if not np.any(affected_rows):
        return
    if block_lengths is None:
        block_lengths = [len(affected_rows)]
    total_blocks = len(block_lengths)
    affected_blocks = _count_affected_blocks(affected_rows, block_lengths)
    message = _singleton_warning_message(stat, apply_corr, affected_blocks, total_blocks)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _sample_bias_correction(afs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    correction = np.full_like(afs, np.nan, dtype=float)
    valid = np.isfinite(afs) & np.isfinite(counts) & (counts > 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(afs * (1 - afs), counts - 1, out=correction, where=valid)
    return correction


def mats_to_f2arr(
    afmat1: pd.DataFrame | np.ndarray,
    afmat2: pd.DataFrame | np.ndarray,
    countmat1: pd.DataFrame | np.ndarray,
    countmat2: pd.DataFrame | np.ndarray,
    block_lengths: Sequence[int],
    snpwt: np.ndarray | None = None,
    apply_corr: bool = True,
    verbose: bool = False,
    population_labels: Sequence[Sequence[str]] | None = None,
) -> np.ndarray:
    a1, a2 = np.asarray(afmat1, float), np.asarray(afmat2, float)
    c1, c2 = np.asarray(countmat1, float), np.asarray(countmat2, float)
    out = np.empty((a1.shape[1], a2.shape[1], len(block_lengths)), dtype=float)
    snpwt = None if snpwt is None else np.asarray(snpwt, float)
    _warn_singleton_observations(
        "f2",
        apply_corr,
        (a1, c1),
        (a2, c2),
        block_lengths=block_lengths,
        population_labels=population_labels,
    )
    start = 0
    for b, n in enumerate(block_lengths):
        stop = start + int(n)
        _log_block("f2", b, len(block_lengths), start, stop, verbose)
        vals = (a1[start:stop].T[:, None, :] - a2[start:stop].T[None, :, :]) ** 2
        if apply_corr:
            corr1 = _sample_bias_correction(a1[start:stop], c1[start:stop])
            corr2 = _sample_bias_correction(a2[start:stop], c2[start:stop])
            vals = vals - (corr1.T[:, None, :] + corr2.T[None, :, :])
        if snpwt is not None:
            vals = vals * snpwt[start:stop][None, None, :]
        with np.errstate(invalid="ignore"):
            out[:, :, b] = _nanmean(vals, axis=2)
        start = stop
    return out


def mats_to_aparr(afmat1, afmat2, countmat1, countmat2, block_lengths, snpwt=None, apply_corr=True, verbose: bool = False) -> np.ndarray:
    a1, a2 = np.asarray(afmat1, float), np.asarray(afmat2, float)
    out = np.empty((a1.shape[1], a2.shape[1], len(block_lengths)), dtype=float)
    snpwt = None if snpwt is None else np.asarray(snpwt, float)
    start = 0
    for b, n in enumerate(block_lengths):
        stop = start + int(n)
        _log_block("allele-product", b, len(block_lengths), start, stop, verbose)
        vals = (_outer_pair(a1[start:stop], a2[start:stop]) + _outer_pair(1 - a1[start:stop], 1 - a2[start:stop])) / 2
        if snpwt is not None:
            vals = vals * snpwt[start:stop][None, None, :]
        with np.errstate(invalid="ignore"):
            out[:, :, b] = _nanmean(vals, axis=2)
        start = stop
    return out


def mats_to_ctarr(afmat1, afmat2, block_lengths, verbose: bool = False) -> np.ndarray:
    # Per-block finite-pair availability fraction for each pop1 x pop2 cell.
    # Multiply by the corresponding block length to obtain a raw SNP count.
    a1 = np.isfinite(np.asarray(afmat1, float)).astype(float)
    a2 = np.isfinite(np.asarray(afmat2, float)).astype(float)
    out = np.empty((a1.shape[1], a2.shape[1], len(block_lengths)), dtype=float)
    start = 0
    for b, n in enumerate(block_lengths):
        stop = start + int(n)
        _log_block("count", b, len(block_lengths), start, stop, verbose)
        out[:, :, b] = (a1[start:stop].T @ a2[start:stop]) / int(n)
        start = stop
    return out


def _hudson_fst_components(
    afmat1,
    afmat2,
    countmat1,
    countmat2,
    block_lengths,
    snpwt=None,
    apply_corr=True,
    verbose: bool = False,
    population_labels: Sequence[Sequence[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a1, a2 = np.asarray(afmat1, float), np.asarray(afmat2, float)
    c1, c2 = np.asarray(countmat1, float), np.asarray(countmat2, float)
    out = np.empty((a1.shape[1], a2.shape[1], len(block_lengths)), dtype=float)
    snp_counts = np.zeros_like(out)
    num_sums = np.full_like(out, np.nan)
    den_sums = np.full_like(out, np.nan)
    snpwt = None if snpwt is None else np.asarray(snpwt, float)
    _warn_singleton_observations(
        "FST",
        apply_corr,
        (a1, c1),
        (a2, c2),
        block_lengths=block_lengths,
        population_labels=population_labels,
    )
    start = 0
    for b, n in enumerate(block_lengths):
        stop = start + int(n)
        _log_block("fst", b, len(block_lengths), start, stop, verbose)
        h1 = a1[start:stop] * (1 - a1[start:stop])
        h2 = a2[start:stop] * (1 - a2[start:stop])
        raw_num = (a1[start:stop].T[:, None, :] - a2[start:stop].T[None, :, :]) ** 2
        denom = raw_num + h1.T[:, None, :] + h2.T[None, :, :]
        if apply_corr:
            corr1 = _sample_bias_correction(a1[start:stop], c1[start:stop])
            corr2 = _sample_bias_correction(a2[start:stop], c2[start:stop])
            num = raw_num - (corr1.T[:, None, :] + corr2.T[None, :, :])
        else:
            num = raw_num
        if snpwt is not None:
            weight = snpwt[start:stop][None, None, :]
            num = num * weight
            denom = denom * weight
        valid = np.isfinite(num) & np.isfinite(denom)
        counts = valid.sum(axis=2).astype(float)
        num_sum = np.sum(np.where(valid, num, 0.0), axis=2)
        den_sum = np.sum(np.where(valid, denom, 0.0), axis=2)
        num_sum[counts == 0] = np.nan
        den_sum[counts == 0] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, :, b] = num_sum / den_sum
        snp_counts[:, :, b] = counts
        num_sums[:, :, b] = num_sum
        den_sums[:, :, b] = den_sum
        start = stop
    return out, snp_counts, num_sums, den_sums


def _hudson_fst(afmat1, afmat2, countmat1, countmat2, block_lengths, snpwt=None, apply_corr=True, verbose: bool = False) -> np.ndarray:
    out, _, _, _ = _hudson_fst_components(
        afmat1,
        afmat2,
        countmat1,
        countmat2,
        block_lengths,
        snpwt=snpwt,
        apply_corr=apply_corr,
        verbose=verbose,
    )
    return out


def afs_to_f2_blocks(
    afdat: AfData,
    blgsize: float = 0.05,
    pops1: Sequence[str] | None = None,
    pops2: Sequence[str] | None = None,
    outpop: str | None = None,
    afprod: bool = False,
    fst: bool = False,
    poly_only: Sequence[str] | bool = ("f2",),
    apply_corr: bool = True,
    stats: Sequence[str] | None = None,
    verbose: bool = True,
):
    pops = list(afdat.afs.columns)
    pops1 = pops if pops1 is None else list(pops1)
    pops2 = pops1 if pops2 is None else list(pops2)
    poly = is_polymorphic(afdat.afs)
    if poly_only is True:
        poly_only = ("f2", "ap", "fst")
    elif poly_only is False:
        poly_only = ()
    snpwt_all = None
    if outpop is not None:
        p = afdat.afs[outpop].to_numpy(float)
        snpwt_all = 1 / (p * (1 - p))

    def build(stat: str) -> F2Blocks:
        use = poly if stat in poly_only else np.ones(len(poly), dtype=bool)
        snp = afdat.snpfile.loc[use].reset_index(drop=True)
        bl = get_block_lengths(snp, blgsize)
        a1, a2 = afdat.afs.loc[use, pops1], afdat.afs.loc[use, pops2]
        c1, c2 = afdat.counts.loc[use, pops1], afdat.counts.loc[use, pops2]
        snpwt = snpwt_all[use] if snpwt_all is not None else None
        fst_num = fst_den = None
        if stat == "f2":
            arr = mats_to_f2arr(
                a1,
                a2,
                c1,
                c2,
                bl,
                snpwt,
                apply_corr,
                verbose=verbose,
                population_labels=(pops1, pops2),
            )
            count_a1, count_a2 = a1.copy(), a2.copy()
            if apply_corr:
                count_a1 = count_a1.mask(c1 <= 1)
                count_a2 = count_a2.mask(c2 <= 1)
            snp_counts = np.rint(
                mats_to_ctarr(count_a1, count_a2, bl, verbose=False) * np.asarray(bl)[None, None, :]
            )
        elif stat == "ap":
            arr = mats_to_aparr(a1, a2, c1, c2, bl, snpwt, apply_corr, verbose=verbose)
            snp_counts = np.rint(mats_to_ctarr(a1, a2, bl, verbose=False) * np.asarray(bl)[None, None, :])
        elif stat == "fst":
            arr, snp_counts, fst_num, fst_den = _hudson_fst_components(
                a1,
                a2,
                c1,
                c2,
                bl,
                snpwt,
                apply_corr,
                verbose=verbose,
                population_labels=(pops1, pops2),
            )
        else:
            raise ValueError(stat)
        if pops1 == pops2:
            for i in range(len(pops1)):
                arr[i, i, :] = 0
        return F2Blocks(arr, pops1, pops2, bl, stat, snp_counts, fst_num, fst_den)

    if stats is None:
        stats = ["f2"]
        if afprod:
            stats.append("ap")
        if fst:
            stats.append("fst")
    elif isinstance(stats, str):
        stats = [stats]
    out = {}
    for stat in dict.fromkeys(stats):
        out[f"{stat}_blocks"] = build(stat)
    return out


def f2_from_geno(
    pref: str | Path,
    inds=None,
    pops=None,
    blgsize: float = 0.05,
    maxmiss: float = 0,
    minmaf: float = 0,
    maxmaf: float = 0.5,
    pops2=None,
    outpop: str | None = None,
    outpop_scale: bool = True,
    transitions: bool = True,
    transversions: bool = True,
    auto_only: bool = True,
    keepsnps=None,
    afprod: bool = False,
    fst: bool = False,
    poly_only: Sequence[str] | bool = ("f2",),
    minac2: bool | int = False,
    format: str | None = None,
    adjust_pseudohaploid=True,
    chunk_size: int = 10_000,
    tgeno_chunked: bool = False,
    remove_na: bool = False,
    apply_corr: bool = True,
    verbose: bool = True,
) -> F2Blocks:
    if inds is None and pops is not None and pops2 is not None:
        pops = list(dict.fromkeys(list(pops) + list(pops2)))
    afdat = anygeno_to_afs(
        pref,
        inds=inds,
        pops=pops,
        format=format,
        adjust_pseudohaploid=adjust_pseudohaploid,
        chunk_size=chunk_size,
        verbose=verbose,
        tgeno_chunked=tgeno_chunked,
    )
    _log("Filtering SNPs", verbose)
    afdat = discard_from_aftable(
        afdat,
        maxmiss=maxmiss,
        minmaf=minmaf,
        maxmaf=maxmaf,
        minac2=minac2,
        outpop=outpop,
        transitions=transitions,
        transversions=transversions,
        auto_only=auto_only,
        keepsnps=keepsnps,
        poly_only=False,
    )
    arrs = afs_to_f2_blocks(
        afdat,
        blgsize=blgsize,
        pops1=pops,
        pops2=pops2,
        outpop=outpop if outpop_scale else None,
        afprod=afprod,
        fst=fst,
        poly_only=poly_only,
        apply_corr=apply_corr,
        stats=("ap",) if afprod else ("fst",) if fst else ("f2",),
        verbose=verbose,
    )
    blocks = arrs["ap_blocks"] if afprod else arrs["fst_blocks"] if fst else arrs["f2_blocks"]
    if afprod:
        blocks = F2Blocks(
            _scale_ap_blocks(blocks.data),
            blocks.pops1,
            blocks.pops2,
            blocks.block_lengths,
            "ap",
            blocks.snp_counts,
        )
    if remove_na:
        keep = np.isfinite(blocks.data).all(axis=(0, 1))
        if not np.any(keep):
            bad_pairs = np.any(~np.isfinite(blocks.data), axis=2)
            affected = []
            seen = set()
            for i, j in np.argwhere(bad_pairs):
                pair = (blocks.pops1[i], blocks.pops2[j])
                key = tuple(sorted(pair))
                if key not in seen:
                    seen.add(key)
                    affected.append(pair)
            stat = "FST" if blocks.stat == "fst" else blocks.stat
            limit = 4
            labels = [
                f"  - {stat}({pop1}, {pop2})"
                for pop1, pop2 in affected[:limit]
            ]
            if len(affected) > limit:
                labels.append(f"  - ... and {len(affected) - limit} more")
            pair_word = "pair" if len(affected) == 1 else "pairs"
            heading = _examples_heading("Affected pairs", len(affected), limit)
            raise ValueError(
                f"No blocks remain after remove_na=True: non-finite {stat} values affect "
                f"{len(affected)} population {pair_word}.\n\n"
                + heading
                + "\n"
                + "\n".join(labels)
                + "\n\nInspect population coverage, or use remove_na=False."
            )
        blocks = blocks.select_blocks(keep)
    return blocks


def _scale_ap_blocks(arr: np.ndarray, from_: float = 0) -> np.ndarray:
    out = -2 * arr.copy()
    out = out - np.nanmin(out) + from_
    if out.shape[0] == out.shape[1]:
        for i in range(out.shape[0]):
            out[i, i, :] = 0
    return out


def _as_pop_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _default_genotype_allsnps(data) -> bool:
    return isinstance(data, (str, Path)) and not Path(data).is_dir()


_ALLSNPS_DIRECT_ERROR = "allsnps=True is only available when reading directly from genotype files"


def _require_unique_pops(pops: Sequence[str], name: str) -> None:
    seen = set()
    duplicates = []
    for pop in pops:
        if pop in seen and pop not in duplicates:
            duplicates.append(pop)
        seen.add(pop)
    if duplicates:
        raise ValueError(f"Duplicate {name} populations are not allowed: {', '.join(duplicates)}")


def _models_frame(models) -> pd.DataFrame:
    if isinstance(models, pd.DataFrame):
        out = models.copy()
    else:
        out = pd.DataFrame(models)
    if "left" not in out.columns or "right" not in out.columns:
        raise ValueError("models must contain 'left' and 'right' columns")
    return out.reset_index(drop=True)


def _model_left_with_target(row) -> list[str]:
    left = _as_pop_list(row.left)
    target = getattr(row, "target", None)
    if target is not None and not (isinstance(target, float) and np.isnan(target)):
        left = [target] + [p for p in left if p != target]
    return left


def f4_model_cache(
    data,
    models,
    resampling: str = "pairwise_counts",
    verbose: bool = True,
    **kwargs,
) -> F4ModelCache | F4BlockCache:
    covariance = kwargs.pop("covariance", True)
    if not covariance:
        raise ValueError(
            "f4_model_cache requires covariance=True because qpWave and qpAdm "
            "require the full f4 covariance matrix"
        )
    resampling = _validate_resampling(resampling)
    allsnps = bool(kwargs.pop("allsnps", False))
    left_base = kwargs.pop("left_base", None)
    right_base = kwargs.pop("right_base", None)
    if isinstance(data, F4ModelCache):
        if allsnps:
            raise ValueError(_ALLSNPS_DIRECT_ERROR)
        return data
    if isinstance(data, F4BlockCache):
        if allsnps:
            raise ValueError(_ALLSNPS_DIRECT_ERROR)
        _validate_cache_resampling(data, resampling)
        return data
    models = _models_frame(models)
    left_pops = []
    right_pops = []
    for row in models.itertuples(index=False):
        left_pops.extend(_model_left_with_target(row))
        right_pops.extend(_as_pop_list(row.right))
    left_pops = list(dict.fromkeys(left_pops))
    right_pops = list(dict.fromkeys(right_pops))
    if _default_genotype_allsnps(data):
        combos = []
        for model_i, row in enumerate(models.itertuples(index=False), start=1):
            left = _model_left_with_target(row)
            right = _as_pop_list(row.right)
            if len(left) < 2 or len(right) < 2:
                continue
            model_left_base, row_pops = _contrast_pops(left, left_base, "left")
            model_right_base, col_pops = _contrast_pops(right, right_base, "right")
            for row_pop in row_pops:
                for col_pop in col_pops:
                    combos.append(
                        {
                            "model": model_i,
                            "pop1": row_pop,
                            "pop2": model_left_base,
                            "pop3": col_pop,
                            "pop4": model_right_base,
                        }
                    )
        _log(f"Loading reusable f4 cache for {len(combos)} population quadruples", verbose)
        # Keep the block information needed to form covariance on demand.
        # Avoid the dense covariance between every contrast in every model.
        kwargs["keep_loo"] = True
        stats = f4_stats(
            data,
            pd.DataFrame(combos),
            unique_only=False,
            allsnps=allsnps,
            resampling=resampling,
            covariance=False,
            verbose=verbose,
            **kwargs,
        )
        return F4BlockCache(stats=stats, models=models, allsnps=allsnps, deferred_covariance=True)
    _log(f"Loading reusable f2 cache for {len(left_pops) * len(right_pops)} population pairs", verbose)
    if resampling == "pairwise_counts":
        kwargs.setdefault("remove_na", False)
    blocks = get_f2(data, pops=left_pops, pops2=right_pops, **kwargs)
    return F4ModelCache(blocks=blocks, models=models, left_pops=left_pops, right_pops=right_pops)


def write_f2(blocks: F2Blocks, outdir: str | Path, overwrite: bool = False):
    if blocks.snp_counts is None:
        raise ValueError("write_f2() requires real per-pair SNP counts")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = outdir / f"block_lengths_{blocks.stat}.npy"
    if meta.exists() and not overwrite:
        raise FileExistsError(meta)
    np.save(meta, blocks.block_lengths)
    for i, p1 in enumerate(blocks.pops1):
        for j, p2 in enumerate(blocks.pops2):
            a, b = sorted((p1, p2))
            d = outdir / a
            d.mkdir(exist_ok=True)
            path = d / f"{b}_{blocks.stat}.npz"
            if path.exists() and not overwrite:
                continue
            payload = {
                "schema_version": np.asarray(2, dtype=np.int64),
                "est": blocks.data[i, j, :],
            }
            payload["n_finite"] = blocks.snp_counts[i, j, :]
            if blocks.fst_num is not None and blocks.fst_den is not None:
                payload["fst_num"] = blocks.fst_num[i, j, :]
                payload["fst_den"] = blocks.fst_den[i, j, :]
            np.savez_compressed(path, **payload)


def read_f2(f2_dir: str | Path, pops: Sequence[str] | None = None, pops2: Sequence[str] | None = None, type: str = "f2", remove_na: bool = False) -> F2Blocks:
    f2_dir = Path(f2_dir)
    if pops is None:
        pops = sorted(p.name for p in f2_dir.iterdir() if p.is_dir())
    pops = list(pops)
    pops2 = pops if pops2 is None else list(pops2)
    bl = np.load(f2_dir / f"block_lengths_{type}.npy")
    arr = np.full((len(pops), len(pops2), len(bl)), np.nan)
    snp_counts = np.full_like(arr, np.nan)
    fst_num = np.full_like(arr, np.nan) if type == "fst" else None
    fst_den = np.full_like(arr, np.nan) if type == "fst" else None
    have_fst_components = False
    for i, p1 in enumerate(pops):
        for j, p2 in enumerate(pops2):
            a, b = sorted((p1, p2))
            path = f2_dir / a / f"{b}_{type}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path) as dat:
                arr[i, j, :] = dat["est"]
                if "n_finite" in dat.files:
                    snp_counts[i, j, :] = dat["n_finite"]
                else:
                    raise ValueError(
                        f"Cache file {path} is missing required per-pair SNP counts; "
                        "rebuild the cache"
                    )
                if type == "fst" and "fst_num" in dat.files and "fst_den" in dat.files:
                    fst_num[i, j, :] = dat["fst_num"]
                    fst_den[i, j, :] = dat["fst_den"]
                    have_fst_components = True
    if remove_na:
        keep = np.isfinite(arr).all(axis=(0, 1))
        arr, snp_counts, bl = arr[:, :, keep], snp_counts[:, :, keep], bl[keep]
        if fst_num is not None and fst_den is not None:
            fst_num, fst_den = fst_num[:, :, keep], fst_den[:, :, keep]
    if not have_fst_components:
        fst_num = fst_den = None
    return F2Blocks(arr, pops, pops2, bl, type, snp_counts, fst_num, fst_den)


def get_f2(data, pops=None, pops2=None, **kwargs) -> F2Blocks:
    allsnps = kwargs.pop("allsnps", False)
    if allsnps:
        raise ValueError("get_f2() cannot cache allsnps=True statistics; batch direct f4 calls with allsnps=True instead")
    if isinstance(data, F4ModelCache):
        return data.blocks.subset(pops, pops2)
    if isinstance(data, F2Blocks):
        return data.subset(pops, pops2)
    if isinstance(data, (str, Path)) and Path(data).is_dir():
        type_ = kwargs.get("type")
        if type_ is None:
            if kwargs.get("fst"):
                type_ = "fst"
            elif kwargs.get("afprod"):
                type_ = "ap"
            else:
                type_ = "f2"
        return read_f2(data, pops, pops2, type=type_, remove_na=kwargs.get("remove_na", False))
    return f2_from_geno(data, pops=pops, pops2=pops2, **kwargs)


def est_to_loo(blocks: F2Blocks | np.ndarray, block_lengths: Sequence[int] | None = None):
    arr = blocks.data if isinstance(blocks, F2Blocks) else np.asarray(blocks, float)
    bl = blocks.block_lengths if isinstance(blocks, F2Blocks) else np.asarray(block_lengths, float)
    finite = np.isfinite(arr)
    weights = np.where(finite, bl[None, None, :], 0)
    numer = np.nansum(arr * bl[None, None, :], axis=2)
    denom = np.sum(weights, axis=2)
    tot = np.full(numer.shape, np.nan, dtype=float)
    np.divide(numer, denom, out=tot, where=denom != 0)
    rel = np.zeros_like(arr, dtype=float)
    np.divide(weights, denom[:, :, None], out=rel, where=denom[:, :, None] != 0)
    out = np.full_like(arr, np.nan, dtype=float)
    present = finite & (rel < 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[present] = ((tot[:, :, None] - arr * rel) / (1 - rel))[present]
    if isinstance(blocks, F2Blocks):
        return F2Blocks(out, blocks.pops1, blocks.pops2, blocks.block_lengths, blocks.stat)
    return out


def stats_to_loo(blocks: np.ndarray, block_lengths: Sequence[int]) -> np.ndarray:
    arr = np.asarray(blocks, float)
    if arr.ndim == 1:
        arr = arr[None, :]
    bl = np.asarray(block_lengths, float)
    if arr.shape[1] != len(bl):
        raise ValueError("blocks must have one column per block")
    weights = np.where(np.isfinite(arr), bl[None, :], 0)
    numer = np.nansum(arr * bl[None, :], axis=1)
    denom = np.sum(weights, axis=1)
    tot = np.full(numer.shape, np.nan, dtype=float)
    np.divide(numer, denom, out=tot, where=denom != 0)
    # Each statistic can be absent from a different set of physical blocks.
    # Normalize nominal weights over its finite blocks, and treat deleting an
    # absent block as deleting no observations (LOO equals the full estimate).
    rel = np.zeros_like(arr, dtype=float)
    np.divide(weights, denom[:, None], out=rel, where=denom[:, None] != 0)
    out = np.broadcast_to(tot[:, None], arr.shape).copy()
    present = np.isfinite(arr) & (rel < 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[present] = ((tot[:, None] - arr * rel) / (1 - rel))[present]
    out[np.isfinite(arr) & (rel >= 1)] = np.nan
    return out


def jack_vec_stats(loo_vec: Sequence[float], block_lengths: Sequence[int]) -> tuple[float, float]:
    loo = np.asarray(loo_vec, float)
    bl = np.asarray(block_lengths, float)
    keep = np.isfinite(loo)
    loo, bl = loo[keep], bl[keep]
    if loo.size < 2:
        return (float(loo[0]), float("nan")) if loo.size == 1 else (float("nan"), float("nan"))
    n = bl.sum()
    h = n / bl
    est = np.sum(loo * (1 - 1 / h)) / np.sum(1 - 1 / h)
    var = np.mean((est - loo) ** 2 * (h - 1))
    return float(est), float(var)


@dataclass
class _CountJackknife:
    total: float
    loo: np.ndarray
    influence: np.ndarray
    contributes: np.ndarray
    n: float


def _validate_resampling(resampling: str) -> str:
    if resampling not in {"pairwise_counts", "nominal_blocks"}:
        raise ValueError("resampling must be 'pairwise_counts' or 'nominal_blocks'")
    return resampling


def _require_pair_counts(blocks: F2Blocks, pop1: str, pop2: str) -> np.ndarray:
    counts = blocks.pair_counts(pop1, pop2)
    if counts is None:
        raise ValueError(
            "resampling='pairwise_counts' requires real per-pair SNP counts; "
            "provide counts or use resampling='nominal_blocks'"
        )
    counts = np.asarray(counts, float)
    vals = blocks.pair(pop1, pop2)
    if np.any(np.isfinite(vals) & ~np.isfinite(counts)):
        raise ValueError(f"Pair-specific SNP counts are incomplete for {pop1!r}, {pop2!r}")
    return counts


def _count_jackknife(block_ests: np.ndarray, n_per_block: np.ndarray) -> _CountJackknife:
    """Count-weighted total, physical-block LOO values, and jackknife influence."""
    ests = np.asarray(block_ests, float)
    counts = np.asarray(n_per_block, float)
    if ests.shape != counts.shape:
        raise ValueError("block estimates and per-block counts must have the same shape")
    valid = np.isfinite(ests) & np.isfinite(counts) & (counts > 0)
    loo = np.full(ests.shape, np.nan, dtype=float)
    influence = np.zeros(ests.shape, dtype=float)
    if not np.any(valid):
        return _CountJackknife(float("nan"), loo, np.full_like(influence, np.nan), valid, 0.0)

    total_n = float(np.sum(counts[valid]))
    total = float(np.sum(ests[valid] * counts[valid]) / total_n)
    # A block with n=0 deletes no observations, so its LOO equals the full
    # estimate and its influence is exactly zero.
    loo[~valid] = total
    remaining = total_n - counts[valid]
    can_delete = remaining > 0
    valid_i = np.flatnonzero(valid)
    delete_i = valid_i[can_delete]
    loo[delete_i] = (
        total * total_n - ests[delete_i] * counts[delete_i]
    ) / remaining[can_delete]
    if len(delete_i) >= 2:
        h = total_n / counts[delete_i]
        tau = h * total - (h - 1.0) * loo[delete_i]
        influence[delete_i] = (tau - total) / np.sqrt(h - 1.0)
    else:
        influence[valid] = np.nan
    return _CountJackknife(total, loo, influence, valid, total_n)


def _jack_stats_per_stat(block_ests: np.ndarray, n_per_block: np.ndarray) -> tuple[float, float]:
    # Per-stat block jackknife where each stat has its own per-block SNP count
    # (allsnps mode). Blocks with no contributing SNPs or a non-finite estimate
    # are excluded before computing unequal-delete-block pseudovalues.
    jack = _count_jackknife(block_ests, n_per_block)
    keep = jack.contributes & np.isfinite(jack.influence)
    if int(np.sum(keep)) < 2:
        return jack.total, float("nan")
    return jack.total, float(np.mean(jack.influence[keep] ** 2))


def _jack_ratio_stats(
    block_num: np.ndarray,
    block_den: np.ndarray,
    n_per_block: np.ndarray,
) -> tuple[float, float]:
    """Jackknife a ratio using leave-one-block-out numerator/denominator sums."""
    num = np.asarray(block_num, float)
    den = np.asarray(block_den, float)
    n = np.asarray(n_per_block, float)
    keep = np.isfinite(num) & np.isfinite(den) & (n > 0)
    num, den, n = num[keep], den[keep], n[keep]
    if num.size == 0:
        return float("nan"), float("nan")
    total_num, total_den = float(num.sum()), float(den.sum())
    total = float(np.divide(total_num, total_den)) if total_den != 0 else float("nan")
    if num.size < 2 or not np.isfinite(total):
        return float(total), float("nan")
    total_n = float(n.sum())
    loo_num = total_num - num
    loo_den = total_den - den
    loo = np.full(num.shape, np.nan, dtype=float)
    np.divide(loo_num, loo_den, out=loo, where=loo_den != 0)
    h = total_n / n
    valid_loo = np.isfinite(loo) & np.isfinite(h) & (h > 1)
    if not np.all(valid_loo):
        return float(total), float("nan")
    # Recompute the nonlinear ratio for each unequal-delete block. Report the
    # full-data ratio as the point estimate and use the bias-corrected
    # jackknife center only for the variance calculation.
    jack_center = float(np.sum(total - loo) + np.sum(loo * n) / total_n)
    tau = h * total - (h - 1.0) * loo
    var = float(np.mean((tau - jack_center) ** 2 / (h - 1.0)))
    return float(total), var


def _ratio_block_jackknife(
    block_num_means: np.ndarray,
    block_den_means: np.ndarray,
    block_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Jackknife ratios of pooled block components.

    Inputs are statistic-by-block matrices of component means and their
    effective block weights, normally contributing SNP counts. Components are
    pooled before full-data and leave-one-block-out ratios are formed.
    """
    num = np.asarray(block_num_means, float)
    den = np.asarray(block_den_means, float)
    weights = np.asarray(block_weights, float)
    if num.ndim == 1:
        num, den, weights = num[None, :], den[None, :], weights[None, :]
    if num.shape != den.shape or num.shape != weights.shape:
        raise ValueError("ratio numerator, denominator, and weights must have matching shapes")

    nstats, nblocks = num.shape
    estimates = np.full(nstats, np.nan, dtype=float)
    loo = np.full((nstats, nblocks), np.nan, dtype=float)
    influence = np.full((nstats, nblocks), np.nan, dtype=float)
    contributes = np.zeros((nstats, nblocks), dtype=bool)

    for stat_i in range(nstats):
        valid = (
            np.isfinite(num[stat_i])
            & np.isfinite(den[stat_i])
            & np.isfinite(weights[stat_i])
            & (weights[stat_i] > 0)
        )
        contributes[stat_i] = valid
        if not np.any(valid):
            continue
        w = weights[stat_i, valid]
        nsum = float(w.sum())
        num_sums = num[stat_i, valid] * w
        den_sums = den[stat_i, valid] * w
        total_num = float(num_sums.sum())
        total_den = float(den_sums.sum())
        full = float(np.divide(total_num, total_den)) if total_den != 0 else float("nan")

        valid_idx = np.flatnonzero(valid)
        remaining_w = nsum - w
        remaining_den = total_den - den_sums
        can_delete = (remaining_w > 0) & (remaining_den != 0)
        delete_idx = valid_idx[can_delete]
        loo[stat_i, delete_idx] = (
            total_num - num_sums[can_delete]
        ) / remaining_den[can_delete]

        finite = np.isfinite(loo[stat_i])
        if int(finite.sum()) < 2:
            estimates[stat_i] = full
            continue
        lw = weights[stat_i, finite]
        lv = loo[stat_i, finite]
        finite_n = float(lw.sum())
        delete_weight = 1.0 - lw / finite_n
        jack_total = float(np.average(lv, weights=delete_weight))
        jack_est = float(np.sum(jack_total - lv) + np.average(lv, weights=lw))
        estimates[stat_i] = jack_est
        h = finite_n / lw
        tau = h * jack_total - (h - 1.0) * lv
        influence[stat_i, finite] = (tau - jack_est) / np.sqrt(h - 1.0)

    cov = _influence_covariance(influence, contributes)
    return estimates, loo, influence, cov


def jackknife_cov(
    loo_mat: np.ndarray,
    block_lengths: Sequence[int],
    est: Sequence[float] | None = None,
    contributes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    loo = np.asarray(loo_mat, float)
    if loo.ndim == 1:
        loo = loo[None, :]
    bl = np.asarray(block_lengths, float)
    if loo.shape[1] != len(bl):
        if loo.shape[0] == len(bl):
            loo = loo.T
        else:
            raise ValueError("loo_mat must have one column per block")
    if contributes is None:
        contributes = np.isfinite(loo)
    else:
        contributes = np.asarray(contributes, bool)
        if contributes.ndim == 1:
            contributes = contributes[None, :]
        if contributes.shape != loo.shape:
            raise ValueError("contributes must have the same shape as loo_mat")
    if est is None:
        est_vec = np.array(
            [jack_vec_stats(row[keep], bl[keep])[0] for row, keep in zip(loo, contributes)]
        )
    else:
        est_vec = np.asarray(est, float)
    valid = contributes & np.isfinite(loo)
    mask = valid.astype(float)
    pair_counts = mask @ mask.T
    pair_lengths = (mask * bl[None, :]) @ mask.T
    finite_est = np.isfinite(est_vec)
    deltas = np.where(valid & finite_est[:, None], est_vec[:, None] - loo, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted_deltas = deltas / np.sqrt(bl)[None, :]
        weighted_products = weighted_deltas @ weighted_deltas.T
        products = deltas @ deltas.T
        numer = pair_lengths * weighted_products - products
    cov = np.full((loo.shape[0], loo.shape[0]), np.nan, dtype=float)
    usable = (pair_counts >= 2) & finite_est[:, None] & finite_est[None, :]
    np.divide(numer, pair_counts, out=cov, where=usable)
    return cov, est_vec


def block_covariance(stats: BlockStats | np.ndarray, block_lengths: Sequence[int] | None = None) -> np.ndarray:
    if isinstance(stats, BlockStats):
        influence = getattr(stats, "influence", None)
        contributes = getattr(stats, "contributes", None)
        if influence is not None and contributes is not None:
            cov = _influence_covariance(influence, contributes)
            stats.cov = cov
            return cov
        loo = stats.loo
        contributes = None if stats.blocks is None else np.isfinite(stats.blocks)
        if loo is None:
            if stats.blocks is None:
                raise ValueError("BlockStats must contain loo or blocks to estimate covariance")
            loo = stats_to_loo(stats.blocks, stats.block_lengths)
        cov, est = jackknife_cov(
            loo,
            stats.block_lengths,
            stats.est,
            contributes=contributes,
        )
        stats.cov = cov
        stats.est = est
        return cov
    if block_lengths is None:
        raise ValueError("block_lengths is required when passing an array")
    cov, _ = jackknife_cov(stats, block_lengths)
    return cov


def f2(
    data,
    pop1=None,
    pop2=None,
    *,
    unique_only: bool = True,
    resampling: str = "pairwise_counts",
    **kwargs,
) -> pd.DataFrame:
    """Pairwise f2 estimates with pair-count or nominal-block resampling.

    By default, blocks are weighted by the number of finite SNPs for each
    population pair and the result includes an ``n`` column. Set
    ``resampling='nominal_blocks'`` to weight every pair by nominal block size.
    """
    resampling = _validate_resampling(resampling)
    kwargs = dict(kwargs)
    if resampling == "pairwise_counts":
        kwargs.setdefault("remove_na", False)
    requested_pops = None
    requested_pops2 = None
    if pop1 is not None:
        requested_pops = [pop1] if isinstance(pop1, str) else list(pop1)
    if pop2 is not None:
        requested_pops2 = [pop2] if isinstance(pop2, str) else list(pop2)
    blocks = get_f2(data, pops=requested_pops, pops2=requested_pops2, **kwargs)
    p1s = blocks.pops1 if pop1 is None else [pop1] if isinstance(pop1, str) else list(pop1)
    p2s = blocks.pops2 if pop2 is None else [pop2] if isinstance(pop2, str) else list(pop2)
    pairs = list(combinations_with_replacement(p1s, 2)) if pop2 is None and unique_only else list(product(p1s, p2s))
    loo = est_to_loo(blocks) if resampling == "nominal_blocks" else None
    rows = []
    for a, b in pairs:
        pair_blocks = blocks.pair(a, b)
        if resampling == "nominal_blocks":
            est, var = jack_vec_stats(loo.pair(a, b), blocks.block_lengths)
            row = {"pop1": a, "pop2": b, "est": est, "se": float(np.sqrt(var))}
        else:
            pair_counts = _require_pair_counts(blocks, a, b)
            est, var = _jack_stats_per_stat(pair_blocks, pair_counts)
            n = int(np.sum(pair_counts[np.isfinite(pair_blocks) & (pair_counts > 0)]))
            row = {"pop1": a, "pop2": b, "est": est, "se": float(np.sqrt(var)), "n": n}
        rows.append(row)
    return FStatsFrame(pd.DataFrame(rows))


def fst(
    data,
    pop1=None,
    pop2=None,
    *,
    unique_only: bool = True,
    resampling: str = "pairwise_counts",
    fst_aggregation: str = "block_ratios",
    **kwargs,
) -> pd.DataFrame:
    """Hudson FST per population pair, with LOO-jackknife standard errors

    raw_num = (p1-p2)^2; denom = raw_num + p1(1-p1) + p2(1-p2).
    With apply_corr=True, num subtracts the two finite-sample corrections.
    ``fst_aggregation='block_ratios'`` preserves the default behavior.
    ``'pooled_components'`` combines cached numerator and denominator sums.
    `data` may be a genotype prefix, an fst-typed F2Blocks, or a precomputed-blocks
    directory. Pair-count resampling adds an ``n`` column.
    """
    resampling = _validate_resampling(resampling)
    if fst_aggregation not in {"block_ratios", "pooled_components"}:
        raise ValueError("fst_aggregation must be 'block_ratios' or 'pooled_components'")
    kwargs = dict(kwargs)
    if resampling == "pairwise_counts":
        kwargs.setdefault("remove_na", False)
    if isinstance(data, F2Blocks) and data.stat != "fst":
        raise ValueError(f"fst() requires F2Blocks with stat='fst' (got {data.stat!r}); recompute with fst=True")
    if isinstance(data, F4ModelCache) and data.blocks.stat != "fst":
        raise ValueError("fst() does not work with F4ModelCache caches built without fst=True")
    requested_pops = None
    requested_pops2 = None
    if pop1 is not None:
        requested_pops = [pop1] if isinstance(pop1, str) else list(pop1)
    if pop2 is not None:
        requested_pops2 = [pop2] if isinstance(pop2, str) else list(pop2)
    blocks = get_f2(data, pops=requested_pops, pops2=requested_pops2, fst=True, **kwargs)
    p1s = blocks.pops1 if pop1 is None else [pop1] if isinstance(pop1, str) else list(pop1)
    p2s = blocks.pops2 if pop2 is None else [pop2] if isinstance(pop2, str) else list(pop2)
    pairs = list(combinations_with_replacement(p1s, 2)) if pop2 is None and unique_only else list(product(p1s, p2s))
    loo = est_to_loo(blocks) if resampling == "nominal_blocks" and fst_aggregation == "block_ratios" else None
    rows = []
    for a, b in pairs:
        pair_blocks = blocks.pair(a, b)
        if a == b:
            est = 0.0
            var = 0.0 if np.isfinite(pair_blocks).sum() >= 2 else float("nan")
        elif fst_aggregation == "pooled_components":
            components = blocks.pair_fst_components(a, b)
            if components is None:
                raise ValueError(
                    "fst_aggregation='pooled_components' requires numerator and denominator "
                    "components; rebuild the FST cache"
                )
            weights = (
                _require_pair_counts(blocks, a, b)
                if resampling == "pairwise_counts"
                else np.asarray(blocks.block_lengths, float)
            )
            est, var = _jack_ratio_stats(components[0], components[1], weights)
        elif resampling == "pairwise_counts":
            pair_counts = _require_pair_counts(blocks, a, b)
            est, var = _jack_stats_per_stat(pair_blocks, pair_counts)
        else:
            est, var = jack_vec_stats(loo.pair(a, b), blocks.block_lengths)
        row = {"pop1": a, "pop2": b, "est": est, "se": float(np.sqrt(var))}
        if resampling == "pairwise_counts":
            pair_counts = _require_pair_counts(blocks, a, b)
            row["n"] = int(np.sum(pair_counts[np.isfinite(pair_blocks) & (pair_counts > 0)]))
        rows.append(row)
    return FStatsFrame(pd.DataFrame(rows))


def f3_from_f2(blocks: F2Blocks, pop1: str, pop2: str, pop3: str) -> np.ndarray:
    return (blocks.pair(pop1, pop2) + blocks.pair(pop1, pop3) - blocks.pair(pop2, pop3)) / 2


def _influence_covariance(influence: np.ndarray, contributes: np.ndarray) -> np.ndarray:
    influence = np.asarray(influence, float)
    contributes = np.asarray(contributes, bool)
    valid = contributes & np.isfinite(influence)
    mask = valid.astype(float)
    pair_counts = mask @ mask.T
    values = np.where(valid, influence, 0.0)
    products = values @ values.T
    cov = np.full(pair_counts.shape, np.nan, dtype=float)
    np.divide(products, pair_counts, out=cov, where=pair_counts >= 2)
    return cov


def _influence_variances(influence: np.ndarray, contributes: np.ndarray) -> np.ndarray:
    # Return only covariance diagonal entries (for custom usage)
    influence = np.asarray(influence, float)
    contributes = np.asarray(contributes, bool)
    valid = contributes & np.isfinite(influence)
    counts = valid.sum(axis=1)
    sums = np.sum(np.where(valid, influence * influence, 0.0), axis=1)
    variances = np.full(influence.shape[0], np.nan, dtype=float)
    np.divide(sums, counts, out=variances, where=counts >= 2)
    return variances


def _pairwise_composite_jackknife(
    blocks: F2Blocks,
    specs: Sequence[Sequence[tuple[float, str, str]]],
    covariance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Combine count-weighted pair estimates into linear f3/f4 statistics.

    Each physical block is deleted from every required population pair. Pair
    blocks with no observations have zero influence rather than an undefined
    ``0 * inf`` pseudovalue.
    """
    npairs: dict[tuple[str, str], _CountJackknife] = {}
    nblocks = blocks.data.shape[2]
    nstats = len(specs)
    totals = np.full(nstats, np.nan, dtype=float)
    loo = np.full((nstats, nblocks), np.nan, dtype=float)
    influence = np.full((nstats, nblocks), np.nan, dtype=float)
    contributes = np.zeros((nstats, nblocks), dtype=bool)

    for stat_i, spec in enumerate(specs):
        combined: dict[tuple[str, str], float] = {}
        for coefficient, pop1, pop2 in spec:
            key = (pop1, pop2)
            combined[key] = combined.get(key, 0.0) + float(coefficient)
        combined = {key: coefficient for key, coefficient in combined.items() if coefficient != 0}

        pair_jacks = []
        failed = False
        for key, coefficient in combined.items():
            if key not in npairs:
                counts = _require_pair_counts(blocks, *key)
                npairs[key] = _count_jackknife(blocks.pair(*key), counts)
            jack = npairs[key]
            if not np.isfinite(jack.total):
                failed = True
                break
            pair_jacks.append((coefficient, jack))
        if failed or not pair_jacks:
            continue

        totals[stat_i] = sum(coefficient * jack.total for coefficient, jack in pair_jacks)
        loo[stat_i] = sum(coefficient * jack.loo for coefficient, jack in pair_jacks)
        influence[stat_i] = sum(coefficient * jack.influence for coefficient, jack in pair_jacks)
        for coefficient, jack in pair_jacks:
            if coefficient != 0:
                contributes[stat_i] |= jack.contributes

    cov = _influence_covariance(influence, contributes) if covariance else None
    return totals, loo, influence, cov, contributes


def _all_block_pops(blocks: F2Blocks) -> list[str]:
    return list(dict.fromkeys(blocks.pops1 + blocks.pops2))


def _f3_combinations(data, pop1, pop2, pop3, unique_only: bool) -> pd.DataFrame:
    if isinstance(pop1, pd.DataFrame):
        cols = ["pop1", "pop2", "pop3"]
        if not all(c in pop1.columns for c in cols):
            raise ValueError("Population-combination data frame must have pop1, pop2, pop3 columns")
        return pop1[cols].copy()

    if pop2 is not None or pop3 is not None:
        p1, p2, p3 = map(_as_list, (pop1, pop2, pop3))
        if p1 is None or p2 is None or p3 is None:
            raise ValueError("pop2 and pop3 must both be provided for f3")
        return pd.DataFrame(product(p1, p2, p3), columns=["pop1", "pop2", "pop3"])

    pops = _as_list(pop1)
    if pops is None:
        if isinstance(data, F2Blocks):
            pops = _all_block_pops(data)
        elif isinstance(data, (str, Path)) and Path(data).is_dir():
            pops = sorted(p.name for p in Path(data).iterdir() if p.is_dir())
        else:
            raise ValueError("pop1 is required when computing f3 directly from genotype data")

    if unique_only:
        if len(pops) < 3:
            raise ValueError("Not enough populations for f3")
        combos = []
        for a, b, c in combinations(pops, 3):
            combos.extend(((a, b, c), (b, c, a), (c, a, b)))
    else:
        combos = product(pops, pops, pops)
    return pd.DataFrame(combos, columns=["pop1", "pop2", "pop3"])


def qp3pop(
    data,
    pop1=None,
    pop2=None,
    pop3=None,
    *,
    unique_only: bool = True,
    resampling: str = "pairwise_counts",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Compute f3(A; B, C) statistics.

    Genotype prefixes use a direct per-SNP estimator, which corrects only
    populations repeated across both contrasts. By default, that numerator is
    normalized by unbiased target heterozygosity; pass ``outgroupmode=True``
    for raw f3 units. F2Blocks and cache directories retain the unnormalized,
    f2-derived estimator.
    """
    resampling = _validate_resampling(resampling)
    kwargs = dict(kwargs)
    combos = _f3_combinations(data, pop1, pop2, pop3, unique_only)
    if unique_only:
        combos = combos.drop_duplicates().reset_index(drop=True)
    is_genotype = _default_genotype_allsnps(data)
    requested_allsnps = kwargs.pop("allsnps", None)
    allsnps = is_genotype if requested_allsnps is None else bool(requested_allsnps)
    if is_genotype:
        # Direct f3 handles missingness per combination, so the f2-cache-only
        # remove_na option has no role on this path.
        kwargs.pop("remove_na", None)
        stats = f3_stats_from_geno(
            data,
            combos,
            allsnps=allsnps,
            verbose=verbose,
            **kwargs,
        )
        if resampling == "nominal_blocks":
            nominal_lengths = np.asarray(stats.nominal_block_lengths, float)
            if getattr(stats, "ratio_num", None) is not None:
                nominal_weights = np.broadcast_to(nominal_lengths, stats.ratio_num.shape).copy()
                nominal_weights[stats.snp_counts <= 0] = 0
                est, loo, _, cov = _ratio_block_jackknife(
                    stats.ratio_num,
                    stats.ratio_den,
                    nominal_weights,
                )
            else:
                loo = stats_to_loo(stats.blocks, nominal_lengths)
                cov, est = jackknife_cov(
                    loo,
                    nominal_lengths,
                    contributes=np.isfinite(stats.blocks),
                )
            stats.block_lengths = nominal_lengths
            stats.loo = loo
            stats.est = est
            stats.cov = cov
            stats.snp_counts = None
        out = stats.to_frame()
        if resampling == "pairwise_counts":
            out["n"] = np.sum(stats.snp_counts, axis=1).astype(int)
        return FStatsFrame(out)
    if allsnps:
        raise ValueError(_ALLSNPS_DIRECT_ERROR)
    if resampling == "pairwise_counts":
        kwargs.setdefault("remove_na", False)
    pops = list(dict.fromkeys(list(combos["pop1"]) + list(combos["pop2"]) + list(combos["pop3"])))
    _log(f"Loading f2 data for {len(pops) * len(pops)} population pairs", verbose)
    blocks = get_f2(data, pops=pops, **kwargs)
    if resampling == "nominal_blocks":
        loo = est_to_loo(blocks)
        pairwise_est = pairwise_cov = None
    else:
        specs = [
            [
                (0.5, row.pop1, row.pop2),
                (0.5, row.pop1, row.pop3),
                (-0.5, row.pop2, row.pop3),
            ]
            for row in combos.itertuples(index=False)
        ]
        pairwise_est, _, _, pairwise_cov, _ = _pairwise_composite_jackknife(
            blocks,
            specs,
        )
    rows = []
    _log(f"Computing f3 for {len(combos)} population combinations", verbose)
    for stat_i, row in enumerate(combos.itertuples(index=False)):
        if resampling == "nominal_blocks":
            vals = f3_from_f2(loo, row.pop1, row.pop2, row.pop3)
            est, var = jack_vec_stats(vals, blocks.block_lengths)
        else:
            est = float(pairwise_est[stat_i])
            var = float(pairwise_cov[stat_i, stat_i])
        se = float(np.sqrt(var))
        z = est / se if np.isfinite(se) and se != 0 else float("nan")
        p = math.erfc(abs(z) / math.sqrt(2)) if np.isfinite(z) else float("nan")
        rows.append(
            {
                "pop1": row.pop1,
                "pop2": row.pop2,
                "pop3": row.pop3,
                "est": est,
                "se": se,
                "z": z,
                "p": p,
            }
        )
    return FStatsFrame(pd.DataFrame(rows))


def f4_from_f2(blocks: F2Blocks, pop1: str, pop2: str, pop3: str, pop4: str) -> np.ndarray:
    return (
        blocks.pair(pop1, pop4)
        + blocks.pair(pop2, pop3)
        - blocks.pair(pop1, pop3)
        - blocks.pair(pop2, pop4)
    ) / 2


def _as_list(x):
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


def _f4_combinations(pop1, pop2, pop3, pop4, comb: bool) -> pd.DataFrame:
    if isinstance(pop1, pd.DataFrame):
        cols = ["pop1", "pop2", "pop3", "pop4"]
        if not all(c in pop1.columns for c in cols):
            raise ValueError("Population-combination data frame must have pop1, pop2, pop3, pop4 columns")
        keep = cols + (["model"] if "model" in pop1.columns else [])
        return pop1[keep].copy()
    p1, p2, p3, p4 = map(_as_list, (pop1, pop2, pop3, pop4))
    if p1 is None or p2 is None or p3 is None or p4 is None:
        raise ValueError("pop1, pop2, pop3, and pop4 are required for f4")
    if comb:
        return pd.DataFrame(product(p1, p2, p3, p4), columns=["pop1", "pop2", "pop3", "pop4"])
    lengths = {len(p1), len(p2), len(p3), len(p4)}
    if len(lengths) != 1:
        raise ValueError("With comb=False, pop1/pop2/pop3/pop4 must have equal lengths")
    return pd.DataFrame({"pop1": p1, "pop2": p2, "pop3": p3, "pop4": p4})


def _format_f4_contrasts(rows: pd.DataFrame, limit: int = 4) -> str:
    rows = rows.reset_index(drop=True)
    shown = rows.head(limit)
    labels = [
        f"  - f4({row.pop1}, {row.pop2}; {row.pop3}, {row.pop4})"
        for row in shown.itertuples(index=False)
    ]
    if len(rows) > len(shown):
        labels.append(f"  - ... and {len(rows) - len(shown)} more")
    return "\n".join(labels)


def _examples_heading(label: str, count: int, limit: int = 4) -> str:
    if count > limit:
        return f"{label} (first {limit} of {count}):"
    return f"{label}:"


def _f4_matmul_groups(idx: np.ndarray, models: np.ndarray, eligible: np.ndarray):
    """Plan small Cartesian products of allele-frequency differences.

    Sparse, arbitrary quadruple lists retain the per-contrast path so that
    batching cannot create a much larger matrix than the requested output.
    """
    groups = []
    for model in pd.unique(models):
        take = np.flatnonzero((models == model) & eligible)
        if len(take) < 4:
            continue
        left, left_i = np.unique(idx[take, :2], axis=0, return_inverse=True)
        right, right_i = np.unique(idx[take, 2:], axis=0, return_inverse=True)
        if len(left) * len(right) <= 2 * len(take):
            groups.append((model, take, left, right, left_i, right_i))
    return groups


def _f4_matmul_block(block, left, right, left_i, right_i, snpwt=None):
    a = block[:, left[:, 0]] - block[:, left[:, 1]]
    b = block[:, right[:, 0]] - block[:, right[:, 1]]
    valid_a, valid_b = np.isfinite(a), np.isfinite(b)
    counts = valid_a.astype(float).T @ valid_b.astype(float)
    a, b = np.where(valid_a, a, 0.0), np.where(valid_b, b, 0.0)
    if snpwt is not None:
        a *= snpwt[:, None]
    sums = a.T @ b
    counts = counts[left_i, right_i]
    estimates = np.full(len(left_i), np.nan)
    np.divide(sums[left_i, right_i], counts, out=estimates, where=counts > 0)
    return estimates, counts


def _f4_direct_blocks_from_afs(
    afdat: AfData,
    combos: pd.DataFrame,
    blgsize: float = 0.05,
    allsnps: bool = False,
    poly_only: bool = False,
    snpwt: Sequence[float] | None = None,
    apply_corr: bool = True,
    stat_name: str = "f4",
    normalize_by_target_het: bool = False,
    covariance: bool = True,
    verbose: bool = True,
    singleton_warning: _SingletonWarningSummary | None = None,
) -> BlockStats:
    cols = ["pop1", "pop2", "pop3", "pop4"]
    combos = combos.reset_index(drop=True).copy()
    missing_cols = [c for c in cols if c not in combos.columns]
    if missing_cols:
        raise ValueError(f"f4 combinations are missing columns: {missing_cols}")
    if "model" not in combos.columns:
        combos["model"] = 1

    pops = list(afdat.afs.columns)
    missing = sorted(set(combos[cols].to_numpy().reshape(-1)) - set(pops))
    if missing:
        raise ValueError(f"Populations missing from allele-frequency table: {missing}")

    arr = afdat.afs.to_numpy(float)
    count_arr = afdat.counts.to_numpy(float)
    pop_i = {p: i for i, p in enumerate(pops)}
    combo_rows = tuple(combos.itertuples(index=False))
    model_values = combos["model"].to_numpy()
    idx = np.asarray([[pop_i[getattr(row, c)] for c in cols] for row in combo_rows], dtype=int)
    correction_coefficients = np.zeros((len(combos), len(pops)), dtype=float)
    for stat_i, row in enumerate(combo_rows):
        first: dict[str, float] = {}
        second: dict[str, float] = {}
        first[row.pop1] = first.get(row.pop1, 0.0) + 1.0
        first[row.pop2] = first.get(row.pop2, 0.0) - 1.0
        second[row.pop3] = second.get(row.pop3, 0.0) + 1.0
        second[row.pop4] = second.get(row.pop4, 0.0) - 1.0
        for pop in set(first) | set(second):
            correction_coefficients[stat_i, pop_i[pop]] = first.get(pop, 0.0) * second.get(pop, 0.0)
    required_by_stat = tuple(
        np.flatnonzero(correction_coefficients[stat_i] != 0)
        for stat_i in range(len(combos))
    )
    correction_pops = np.flatnonzero(np.any(correction_coefficients != 0, axis=0))
    correction_labels = [pops[i] for i in correction_pops]
    block_lengths = get_block_lengths(afdat.snpfile, blgsize)
    if correction_pops.size:
        pairs = ((arr[:, correction_pops], count_arr[:, correction_pops]),)
        if singleton_warning is None:
            _warn_singleton_observations(
                stat_name,
                apply_corr,
                *pairs,
                block_lengths=block_lengths,
                population_labels=(correction_labels,),
            )
        else:
            singleton_warning.observe(
                block_lengths,
                *pairs,
                population_labels=(correction_labels,),
            )
    nstats = len(combos)
    out = np.full((nstats, len(block_lengths)), np.nan, dtype=float)
    denominator = np.full_like(out, np.nan) if normalize_by_target_het else None
    snp_counts = np.zeros((nstats, len(block_lengths)), dtype=float)
    snpwt = None if snpwt is None else np.asarray(snpwt, float)
    if snpwt is not None and len(snpwt) != arr.shape[0]:
        raise ValueError("snpwt must have one value per retained SNP")

    use_by_model: dict[object, np.ndarray] = {}
    if not allsnps:
        for model, sub in combos.groupby("model", sort=False):
            model_pops = sorted(set(sub[cols].to_numpy().reshape(-1)))
            model_idx = [pop_i[p] for p in model_pops]
            use = np.isfinite(arr[:, model_idx]).all(axis=1)
            if poly_only:
                vals = arr[:, model_idx]
                # Direct ADMIXTOOLS statistics retain segregating sites even
                # when every population has the same non-boundary frequency.
                # Only sites fixed at 0 or fixed at 1 are monomorphic here.
                use &= (np.nanmax(vals, axis=1) > 0) & (np.nanmin(vals, axis=1) < 1)
            use_by_model[model] = use

    groups = []
    if (
        not normalize_by_target_het
        and not (allsnps and poly_only)
        and (snpwt is None or np.isfinite(snpwt).all())
    ):
        eligible = ~np.any(correction_coefficients != 0, axis=1)
        groups = _f4_matmul_groups(idx, model_values, eligible)
    scalar_stats = np.ones(nstats, dtype=bool)
    for _, take, *_ in groups:
        scalar_stats[take] = False
    scalar_indices = np.flatnonzero(scalar_stats)

    start = 0
    for b, n in enumerate(block_lengths):
        stop = start + int(n)
        _log_block(f"direct {stat_name}", b, len(block_lengths), start, stop, verbose)
        block = arr[start:stop]
        count_block = count_arr[start:stop]
        block_snpwt = None if snpwt is None else snpwt[start:stop]
        for model, take, left, right, left_i, right_i in groups:
            use = slice(None) if allsnps else use_by_model[model][start:stop]
            weights = None if block_snpwt is None else block_snpwt[use]
            out[take, b], snp_counts[take, b] = _f4_matmul_block(
                block[use], left, right, left_i, right_i, weights,
            )
        for stat_i in scalar_indices:
            p = idx[stat_i]
            vals = block[:, p]
            use = (
                np.isfinite(vals).all(axis=1)
                if allsnps
                else use_by_model[model_values[stat_i]][start:stop].copy()
            )
            if allsnps and poly_only:
                finite_vals = vals[use]
                if finite_vals.size:
                    use_idx = np.where(use)[0]
                    use[use_idx] &= (
                        (np.max(finite_vals, axis=1) > 0)
                        & (np.min(finite_vals, axis=1) < 1)
                    )
            correction = correction_coefficients[stat_i]
            required = required_by_stat[stat_i]
            if apply_corr and required.size:
                use &= np.isfinite(count_block[:, required]).all(axis=1)
                use &= (count_block[:, required] > 1).all(axis=1)
            if normalize_by_target_het:
                target_idx = p[0]
                use &= np.isfinite(count_block[:, target_idx])
                use &= count_block[:, target_idx] > 1
            if not np.any(use):
                continue
            f4vals = (vals[use, 0] - vals[use, 1]) * (vals[use, 2] - vals[use, 3])
            if apply_corr:
                for pop_idx in required:
                    corr = _sample_bias_correction(
                        block[use, pop_idx],
                        count_block[use, pop_idx],
                    )
                    f4vals = f4vals - correction[pop_idx] * corr
            if block_snpwt is not None:
                f4vals = f4vals * block_snpwt[use]
            out[stat_i, b] = float(np.mean(f4vals))
            if normalize_by_target_het:
                target_p = block[use, target_idx]
                target_n = count_block[use, target_idx]
                target_het = 2.0 * target_p * (1.0 - target_p) * target_n / (target_n - 1.0)
                if block_snpwt is not None:
                    # Normalized f3 is a ratio of two SNP-weighted sums.  Apply
                    # the same optional outgroup weight to both components.
                    target_het = target_het * block_snpwt[use]
                denominator[stat_i, b] = float(np.mean(target_het))
            snp_counts[stat_i, b] = int(use.sum())
        start = stop

    effective_lengths = np.nanmax(snp_counts, axis=0)
    effective_lengths = np.where(effective_lengths > 0, effective_lengths, block_lengths).astype(float)
    if normalize_by_target_het:
        ratio_blocks = np.full_like(out, np.nan)
        np.divide(out, denominator, out=ratio_blocks, where=denominator != 0)
        est, loo, _, cov = _ratio_block_jackknife(out, denominator, snp_counts)
        result_blocks = ratio_blocks
    else:
        jacks = [_count_jackknife(out[i], snp_counts[i]) for i in range(nstats)]
        est = np.asarray([jack.total for jack in jacks], float)
        loo = np.asarray([jack.loo for jack in jacks], float)
        influence = np.asarray([jack.influence for jack in jacks], float)
        contributes = np.asarray([jack.contributes for jack in jacks], bool)
        cov = _influence_covariance(influence, contributes) if covariance else None
        result_blocks = out
    rows = combos.drop(columns=["model"]) if set(combos["model"]) == {1} else combos
    stats = BlockStats(rows=rows.reset_index(drop=True), blocks=result_blocks, block_lengths=effective_lengths, stat=stat_name, loo=loo, est=est, cov=cov)
    if not covariance and not normalize_by_target_het:
        stats.variances = _influence_variances(influence, contributes)
    if not normalize_by_target_het:
        stats.influence = influence
        stats.contributes = contributes
    stats.snp_counts = snp_counts
    stats.nominal_block_lengths = np.asarray(block_lengths, float)
    if normalize_by_target_het:
        stats.ratio_num = out
        stats.ratio_den = denominator
    return stats


def f4_stats_from_geno(
    pref: str | Path,
    popcombs: pd.DataFrame,
    *,
    blgsize: float = 0.05,
    maxmiss: float | None = None,
    minmaf: float = 0,
    maxmaf: float = 0.5,
    minac2: bool | int = False,
    outpop: str | None = None,
    outpop_scale: bool = True,
    transitions: bool = True,
    transversions: bool = True,
    auto_only: bool = True,
    keepsnps=None,
    allsnps: bool = False,
    poly_only: bool = False,
    apply_corr: bool = True,
    format: str | None = None,
    adjust_pseudohaploid=True,
    chunk_size: int = 250_000,
    tgeno_chunked: bool = False,
    verbose: bool = True,
    stream: bool = False,
    covariance: bool = True,
) -> BlockStats:
    cols = ["pop1", "pop2", "pop3", "pop4"]
    pops = list(dict.fromkeys(popcombs[cols].to_numpy().reshape(-1)))
    if outpop is not None and outpop not in pops:
        pops.append(outpop)
    if maxmiss is None:
        maxmiss = 1 if allsnps or "model" in popcombs.columns else 0
    if stream:
        return _f4_stats_from_geno_stream(
            pref,
            popcombs,
            blgsize=blgsize,
            maxmiss=maxmiss,
            minmaf=minmaf,
            maxmaf=maxmaf,
            minac2=minac2,
            outpop=outpop,
            outpop_scale=outpop_scale,
            transitions=transitions,
            transversions=transversions,
            auto_only=auto_only,
            keepsnps=keepsnps,
            allsnps=allsnps,
            poly_only=poly_only,
            apply_corr=apply_corr,
            format=format,
            adjust_pseudohaploid=adjust_pseudohaploid,
            chunk_size=chunk_size,
            verbose=verbose,
            covariance=covariance,
        )
    afdat = anygeno_to_afs(
        pref,
        pops=pops,
        format=format,
        adjust_pseudohaploid=adjust_pseudohaploid,
        chunk_size=chunk_size,
        verbose=verbose,
        tgeno_chunked=tgeno_chunked,
    )
    _log("Filtering SNPs", verbose)
    afdat = discard_from_aftable(
        afdat,
        maxmiss=maxmiss,
        minmaf=minmaf,
        maxmaf=maxmaf,
        minac2=minac2,
        outpop=outpop,
        transitions=transitions,
        transversions=transversions,
        auto_only=auto_only,
        keepsnps=keepsnps,
        poly_only=False,
    )
    snpwt = None
    if outpop is not None and outpop_scale:
        outgroup_af = afdat.afs[outpop].to_numpy(float)
        snpwt = 1 / (outgroup_af * (1 - outgroup_af))
    return _f4_direct_blocks_from_afs(
        afdat,
        popcombs,
        blgsize=blgsize,
        allsnps=allsnps,
        poly_only=poly_only,
        snpwt=snpwt,
        apply_corr=apply_corr,
        covariance=covariance,
        verbose=verbose,
    )


def _f3_direct_blocks_from_afs(
    afdat: AfData,
    combos: pd.DataFrame,
    blgsize: float = 0.05,
    allsnps: bool = True,
    poly_only: bool = False,
    snpwt: Sequence[float] | None = None,
    apply_corr: bool = True,
    outgroupmode: bool = False,
    verbose: bool = True,
    singleton_warning: _SingletonWarningSummary | None = None,
) -> BlockStats:
    cols = ["pop1", "pop2", "pop3"]
    missing_cols = [col for col in cols if col not in combos.columns]
    if missing_cols:
        raise ValueError(f"f3 combinations are missing columns: {missing_cols}")
    combos = combos.reset_index(drop=True).copy()
    if singleton_warning is None:
        target_has_two = {}
        for pop in combos["pop1"].unique():
            af = afdat.afs[pop].to_numpy(float)
            count = afdat.counts[pop].to_numpy(float)
            target_has_two[pop] = bool(
                np.any(np.isfinite(af) & np.isfinite(count) & (count > 1))
            )
        _validate_f3_targets(
            combos,
            target_has_two,
            apply_corr=apply_corr,
            outgroupmode=outgroupmode,
        )
    mapped = pd.DataFrame(
        {
            "pop1": combos["pop1"],
            "pop2": combos["pop2"],
            "pop3": combos["pop1"],
            "pop4": combos["pop3"],
        }
    )
    if "model" in combos.columns:
        mapped["model"] = combos["model"]
    stats = _f4_direct_blocks_from_afs(
        afdat,
        mapped,
        blgsize=blgsize,
        allsnps=allsnps,
        poly_only=poly_only,
        snpwt=snpwt,
        apply_corr=apply_corr,
        stat_name="f3",
        normalize_by_target_het=not outgroupmode,
        verbose=verbose,
        singleton_warning=singleton_warning,
    )
    stats.rows = combos[cols].copy()
    stats.stat = "f3"
    return stats


def _validate_f3_targets(
    combos: pd.DataFrame,
    has_two_observations: dict[str, bool],
    *,
    apply_corr: bool,
    outgroupmode: bool,
) -> None:
    target = combos["pop1"]
    requires_two = np.full(len(combos), not outgroupmode, dtype=bool)
    if apply_corr:
        requires_two |= (target != combos["pop2"]) & (target != combos["pop3"])
    required_targets = dict.fromkeys(target[requires_two])
    unavailable = [
        pop for pop in required_targets if not has_two_observations.get(pop, False)
    ]
    if unavailable:
        limit = 4
        shown = unavailable[:limit]
        names = ", ".join(repr(pop) for pop in shown)
        if len(unavailable) > limit:
            names += f", ... and {len(unavailable) - limit} more"
        target_word = "target population" if len(unavailable) == 1 else "target populations"
        heading = _examples_heading("Affected targets", len(unavailable), limit)
        raise ValueError(
            f"f3 cannot estimate {len(unavailable)} {target_word}: no retained SNP has "
            "at least two independent allele observations.\n\n"
            + heading
            + " "
            + names
            + "\n\nFor a biased estimate, use outgroupmode=True and apply_corr=False."
        )


def _concat_afdata(parts: Sequence[AfData]) -> AfData:
    if not parts:
        raise ValueError("Cannot concatenate an empty allele-frequency block")
    snp = pd.concat([part.snpfile for part in parts], ignore_index=True)
    afs = pd.concat([part.afs.reset_index(drop=True) for part in parts], ignore_index=True)
    counts = pd.concat([part.counts.reset_index(drop=True) for part in parts], ignore_index=True)
    afs.index = snp["SNP"]
    counts.index = snp["SNP"]
    return AfData(afs, counts, snp)


def _f4_stats_from_geno_stream(
    pref: str | Path,
    popcombs: pd.DataFrame,
    *,
    blgsize: float,
    maxmiss: float,
    minmaf: float,
    maxmaf: float,
    minac2: bool | int,
    outpop: str | None,
    outpop_scale: bool,
    transitions: bool,
    transversions: bool,
    auto_only: bool,
    keepsnps,
    allsnps: bool,
    poly_only: bool,
    apply_corr: bool,
    format: str | None,
    adjust_pseudohaploid,
    chunk_size: int,
    verbose: bool,
    covariance: bool,
) -> BlockStats:
    """Two-pass, bounded-memory direct f4 computation."""
    cols = ["pop1", "pop2", "pop3", "pop4"]
    combos = popcombs.reset_index(drop=True).copy()
    pops = list(dict.fromkeys(combos[cols].to_numpy().reshape(-1)))
    if outpop is not None and outpop not in pops:
        pops.append(outpop)

    minac2_pops: list[str] | None = None
    if minac2 == 2 and keepsnps is None:
        max_counts = {pop: 0.0 for pop in pops}
        for chunk in iter_geno_to_afs(
            pref,
            pops=pops,
            format=format,
            adjust_pseudohaploid=adjust_pseudohaploid,
            chunk_size=chunk_size,
            verbose=False,
        ):
            for pop in pops:
                values = chunk.counts[pop].to_numpy(float)
                if values.size:
                    max_counts[pop] = max(max_counts[pop], float(np.nanmax(values)))
        minac2_pops = [pop for pop, maximum in max_counts.items() if maximum > 1]

    keep_chunks: list[np.ndarray] = []
    retained_snp_parts: list[pd.DataFrame] = []
    _log("Filtering SNPs (streaming pass 1/2)", verbose)
    for chunk in iter_geno_to_afs(
        pref,
        pops=pops,
        format=format,
        adjust_pseudohaploid=adjust_pseudohaploid,
        chunk_size=chunk_size,
        verbose=verbose,
    ):
        try:
            filter_snp = chunk.snpfile.copy()
            filter_snp["_stream_row"] = np.arange(len(filter_snp))
            filtered = discard_from_aftable(
                AfData(chunk.afs, chunk.counts, filter_snp),
                maxmiss=maxmiss,
                minmaf=minmaf,
                maxmaf=maxmaf,
                minac2=False if minac2_pops is not None else minac2,
                outpop=outpop,
                transitions=transitions,
                transversions=transversions,
                auto_only=auto_only,
                keepsnps=keepsnps,
                poly_only=False,
            )
            keep = np.zeros(len(chunk.snpfile), dtype=bool)
            keep[filtered.snpfile["_stream_row"].to_numpy(int)] = True
            if minac2_pops:
                keep &= (chunk.counts[minac2_pops].to_numpy(float) > 1).all(axis=1)
            if np.any(keep):
                retained_snp_parts.append(chunk.snpfile.loc[keep].reset_index(drop=True))
        except ValueError as err:
            if str(err) != "No SNPs remain after filtering":
                raise
            keep = np.zeros(len(chunk.snpfile), dtype=bool)
        keep_chunks.append(keep)
    if not retained_snp_parts:
        raise ValueError("No SNPs remain after filtering")

    retained_snp = pd.concat(retained_snp_parts, ignore_index=True)
    block_lengths = get_block_lengths(retained_snp, blgsize)
    nstats, nblocks = len(combos), len(block_lengths)
    block_estimates = np.full((nstats, nblocks), np.nan, dtype=float)
    snp_counts = np.zeros((nstats, nblocks), dtype=float)

    _log("Computing direct f4 (streaming pass 2/2)", verbose)
    second_stream = iter_geno_to_afs(
        pref,
        pops=pops,
        format=format,
        adjust_pseudohaploid=adjust_pseudohaploid,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    block_i = 0
    block_remaining = int(block_lengths[0]) if nblocks else 0
    block_parts: list[AfData] = []
    singleton_warning = _SingletonWarningSummary("f4", apply_corr)

    def finish_block(parts: list[AfData], index: int) -> None:
        block_afdat = _concat_afdata(parts)
        snpwt = None
        if outpop is not None and outpop_scale:
            outgroup_af = block_afdat.afs[outpop].to_numpy(float)
            snpwt = 1 / (outgroup_af * (1 - outgroup_af))
        block_stats = _f4_direct_blocks_from_afs(
            block_afdat,
            combos,
            blgsize=float("inf"),
            allsnps=allsnps,
            poly_only=poly_only,
            snpwt=snpwt,
            apply_corr=apply_corr,
            covariance=False,
            verbose=False,
            singleton_warning=singleton_warning,
        )
        block_estimates[:, index] = block_stats.blocks[:, 0]
        snp_counts[:, index] = block_stats.snp_counts[:, 0]

    for chunk_i, chunk in enumerate(second_stream):
        keep = keep_chunks[chunk_i]
        if not np.any(keep):
            continue
        kept_snp = chunk.snpfile.loc[keep].reset_index(drop=True)
        kept_afs = chunk.afs.iloc[keep].copy()
        kept_counts = chunk.counts.iloc[keep].copy()
        kept_afs.index = kept_snp["SNP"]
        kept_counts.index = kept_snp["SNP"]
        offset = 0
        while offset < len(kept_snp):
            take = min(block_remaining, len(kept_snp) - offset)
            stop = offset + take
            part_snp = kept_snp.iloc[offset:stop].reset_index(drop=True)
            part_afs = kept_afs.iloc[offset:stop].copy()
            part_counts = kept_counts.iloc[offset:stop].copy()
            part_afs.index = part_snp["SNP"]
            part_counts.index = part_snp["SNP"]
            block_parts.append(AfData(part_afs, part_counts, part_snp))
            block_remaining -= take
            offset = stop
            if block_remaining == 0:
                finish_block(block_parts, block_i)
                block_i += 1
                block_parts = []
                if block_i < nblocks:
                    block_remaining = int(block_lengths[block_i])
    if block_parts or block_i != nblocks:
        raise RuntimeError("Streaming f4 block assembly did not consume the retained SNP layout")
    singleton_warning.warn(nblocks, stacklevel=3)

    effective_lengths = np.nanmax(snp_counts, axis=0)
    effective_lengths = np.where(effective_lengths > 0, effective_lengths, block_lengths).astype(float)
    jacks = [_count_jackknife(block_estimates[i], snp_counts[i]) for i in range(nstats)]
    est = np.asarray([jack.total for jack in jacks], float)
    loo = np.asarray([jack.loo for jack in jacks], float)
    influence = np.asarray([jack.influence for jack in jacks], float)
    contributes = np.asarray([jack.contributes for jack in jacks], bool)
    cov = _influence_covariance(influence, contributes) if covariance else None
    rows = combos.drop(columns=["model"]) if "model" in combos and set(combos["model"]) == {1} else combos
    stats = BlockStats(rows.reset_index(drop=True), block_estimates, effective_lengths, "f4", loo, est, cov)
    if not covariance:
        stats.variances = _influence_variances(influence, contributes)
    stats.influence = influence
    stats.contributes = contributes
    stats.snp_counts = snp_counts
    stats.nominal_block_lengths = np.asarray(block_lengths, float)
    return stats


def _f3_stats_from_geno_stream(
    pref: str | Path,
    popcombs: pd.DataFrame,
    *,
    blgsize: float,
    maxmiss: float,
    minmaf: float,
    maxmaf: float,
    minac2: bool | int,
    outpop: str | None,
    outpop_scale: bool,
    transitions: bool,
    transversions: bool,
    auto_only: bool,
    keepsnps,
    allsnps: bool,
    poly_only: bool,
    apply_corr: bool,
    outgroupmode: bool,
    format: str | None,
    adjust_pseudohaploid,
    chunk_size: int,
    verbose: bool,
) -> BlockStats:
    """Two-pass, bounded-memory direct f3 computation.

    The first pass stores only a retained-SNP mask and metadata so block
    boundaries remain identical to the materialized implementation. The
    second pass computes one physical block at a time.
    """
    cols = ["pop1", "pop2", "pop3"]
    pops = list(dict.fromkeys(popcombs[cols].to_numpy().reshape(-1)))
    if outpop is not None and outpop not in pops:
        pops.append(outpop)

    keep_chunks: list[np.ndarray] = []
    retained_snp_parts: list[pd.DataFrame] = []
    target_has_two = {pop: False for pop in popcombs["pop1"].unique()}
    first_stream = iter_geno_to_afs(
        pref,
        pops=pops,
        format=format,
        adjust_pseudohaploid=adjust_pseudohaploid,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    _log("Filtering SNPs (streaming pass 1/2)", verbose)
    for chunk in first_stream:
        try:
            filter_snp = chunk.snpfile.copy()
            filter_snp["_stream_row"] = np.arange(len(filter_snp))
            filtered = discard_from_aftable(
                AfData(chunk.afs, chunk.counts, filter_snp),
                maxmiss=maxmiss,
                minmaf=minmaf,
                maxmaf=maxmaf,
                minac2=minac2,
                outpop=outpop,
                transitions=transitions,
                transversions=transversions,
                auto_only=auto_only,
                keepsnps=keepsnps,
                poly_only=False,
            )
            keep = np.zeros(len(chunk.snpfile), dtype=bool)
            keep[filtered.snpfile["_stream_row"].to_numpy(int)] = True
            retained_snp_parts.append(chunk.snpfile.loc[keep].reset_index(drop=True))
        except ValueError as err:
            if str(err) != "No SNPs remain after filtering":
                raise
            keep = np.zeros(len(chunk.snpfile), dtype=bool)
        keep_chunks.append(keep)
        for pop, already_usable in target_has_two.items():
            if already_usable or not np.any(keep):
                continue
            af = chunk.afs[pop].to_numpy(float)
            count = chunk.counts[pop].to_numpy(float)
            target_has_two[pop] = bool(
                np.any(keep & np.isfinite(af) & np.isfinite(count) & (count > 1))
            )
    if not retained_snp_parts:
        raise ValueError("No SNPs remain after filtering")
    _validate_f3_targets(
        popcombs,
        target_has_two,
        apply_corr=apply_corr,
        outgroupmode=outgroupmode,
    )

    retained_snp = pd.concat(retained_snp_parts, ignore_index=True)
    block_lengths = get_block_lengths(retained_snp, blgsize)
    nstats = len(popcombs)
    nblocks = len(block_lengths)
    numerator = np.full((nstats, nblocks), np.nan, dtype=float)
    denominator = np.full_like(numerator, np.nan) if not outgroupmode else None
    snp_counts = np.zeros((nstats, nblocks), dtype=float)

    _log("Computing direct f3 (streaming pass 2/2)", verbose)
    second_stream = iter_geno_to_afs(
        pref,
        pops=pops,
        format=format,
        adjust_pseudohaploid=adjust_pseudohaploid,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    block_i = 0
    block_remaining = int(block_lengths[0]) if nblocks else 0
    block_parts: list[AfData] = []
    singleton_warning = _SingletonWarningSummary("f3", apply_corr)

    def finish_block(parts: list[AfData], index: int) -> None:
        block_afdat = _concat_afdata(parts)
        snpwt = None
        if outpop is not None and outpop_scale:
            outgroup_af = block_afdat.afs[outpop].to_numpy(float)
            snpwt = 1 / (outgroup_af * (1 - outgroup_af))
        block_stats = _f3_direct_blocks_from_afs(
            block_afdat,
            popcombs,
            blgsize=float("inf"),
            allsnps=allsnps,
            poly_only=poly_only,
            snpwt=snpwt,
            apply_corr=apply_corr,
            outgroupmode=outgroupmode,
            verbose=False,
            singleton_warning=singleton_warning,
        )
        snp_counts[:, index] = block_stats.snp_counts[:, 0]
        if outgroupmode:
            numerator[:, index] = block_stats.blocks[:, 0]
        else:
            numerator[:, index] = block_stats.ratio_num[:, 0]
            denominator[:, index] = block_stats.ratio_den[:, 0]

    for chunk_i, chunk in enumerate(second_stream):
        keep = keep_chunks[chunk_i]
        if not np.any(keep):
            continue
        kept = AfData(
            chunk.afs.iloc[keep].copy(),
            chunk.counts.iloc[keep].copy(),
            chunk.snpfile.loc[keep].reset_index(drop=True),
        )
        kept.afs.index = kept.snpfile["SNP"]
        kept.counts.index = kept.snpfile["SNP"]
        offset = 0
        while offset < len(kept.snpfile):
            take = min(block_remaining, len(kept.snpfile) - offset)
            stop = offset + take
            part_snp = kept.snpfile.iloc[offset:stop].reset_index(drop=True)
            part_afs = kept.afs.iloc[offset:stop].copy()
            part_counts = kept.counts.iloc[offset:stop].copy()
            part_afs.index = part_snp["SNP"]
            part_counts.index = part_snp["SNP"]
            block_parts.append(AfData(part_afs, part_counts, part_snp))
            block_remaining -= take
            offset = stop
            if block_remaining == 0:
                finish_block(block_parts, block_i)
                block_i += 1
                block_parts = []
                if block_i < nblocks:
                    block_remaining = int(block_lengths[block_i])
    if block_parts or block_i != nblocks:
        raise RuntimeError("Streaming f3 block assembly did not consume the retained SNP layout")
    singleton_warning.warn(nblocks, stacklevel=3)

    effective_lengths = np.nanmax(snp_counts, axis=0)
    effective_lengths = np.where(effective_lengths > 0, effective_lengths, block_lengths).astype(float)
    if outgroupmode:
        jacks = [_count_jackknife(numerator[i], snp_counts[i]) for i in range(nstats)]
        est = np.asarray([jack.total for jack in jacks], float)
        loo = np.asarray([jack.loo for jack in jacks], float)
        influence = np.asarray([jack.influence for jack in jacks], float)
        contributes = np.asarray([jack.contributes for jack in jacks], bool)
        cov = _influence_covariance(influence, contributes)
        blocks = numerator
    else:
        blocks = np.full_like(numerator, np.nan)
        np.divide(numerator, denominator, out=blocks, where=denominator != 0)
        est, loo, _, cov = _ratio_block_jackknife(numerator, denominator, snp_counts)

    stats = BlockStats(
        rows=popcombs[cols].reset_index(drop=True),
        blocks=blocks,
        block_lengths=effective_lengths,
        stat="f3",
        loo=loo,
        est=est,
        cov=cov,
    )
    stats.snp_counts = snp_counts
    stats.nominal_block_lengths = np.asarray(block_lengths, float)
    if not outgroupmode:
        stats.ratio_num = numerator
        stats.ratio_den = denominator
    return stats


def f3_stats_from_geno(
    pref: str | Path,
    popcombs: pd.DataFrame,
    *,
    blgsize: float = 0.05,
    maxmiss: float | None = None,
    minmaf: float = 0,
    maxmaf: float = 0.5,
    minac2: bool | int = False,
    outpop: str | None = None,
    outpop_scale: bool = True,
    transitions: bool = True,
    transversions: bool = True,
    auto_only: bool = True,
    keepsnps=None,
    allsnps: bool = True,
    poly_only: bool = False,
    apply_corr: bool = True,
    outgroupmode: bool = False,
    format: str | None = None,
    adjust_pseudohaploid=True,
    chunk_size: int = 250_000,
    tgeno_chunked: bool = False,
    verbose: bool = True,
    stream: bool = False,
) -> BlockStats:
    """Compute corrected f3 blocks directly from genotype data.

    For f3(A; B, C), finite-sample correction is required only for
    populations repeated across the two contrasts. With distinct A, B, and C,
    this is A, so a singleton B or C remains estimable. Unless
    ``outgroupmode=True``, the corrected numerator is normalized by unbiased
    target heterozygosity.
    """
    cols = ["pop1", "pop2", "pop3"]
    missing_cols = [col for col in cols if col not in popcombs.columns]
    if missing_cols:
        raise ValueError(f"f3 combinations are missing columns: {missing_cols}")
    if maxmiss is None:
        maxmiss = 1 if allsnps else 0
    if stream:
        return _f3_stats_from_geno_stream(
            pref,
            popcombs,
            blgsize=blgsize,
            maxmiss=maxmiss,
            minmaf=minmaf,
            maxmaf=maxmaf,
            minac2=minac2,
            outpop=outpop,
            outpop_scale=outpop_scale,
            transitions=transitions,
            transversions=transversions,
            auto_only=auto_only,
            keepsnps=keepsnps,
            allsnps=allsnps,
            poly_only=poly_only,
            apply_corr=apply_corr,
            outgroupmode=outgroupmode,
            format=format,
            adjust_pseudohaploid=adjust_pseudohaploid,
            chunk_size=chunk_size,
            verbose=verbose,
        )
    pops = list(dict.fromkeys(popcombs[cols].to_numpy().reshape(-1)))
    if outpop is not None and outpop not in pops:
        pops.append(outpop)
    afdat = anygeno_to_afs(
        pref,
        pops=pops,
        format=format,
        adjust_pseudohaploid=adjust_pseudohaploid,
        chunk_size=chunk_size,
        verbose=verbose,
        tgeno_chunked=tgeno_chunked,
    )
    _log("Filtering SNPs", verbose)
    afdat = discard_from_aftable(
        afdat,
        maxmiss=maxmiss,
        minmaf=minmaf,
        maxmaf=maxmaf,
        minac2=minac2,
        outpop=outpop,
        transitions=transitions,
        transversions=transversions,
        auto_only=auto_only,
        keepsnps=keepsnps,
        poly_only=False,
    )
    snpwt = None
    if outpop is not None and outpop_scale:
        outgroup_af = afdat.afs[outpop].to_numpy(float)
        snpwt = 1 / (outgroup_af * (1 - outgroup_af))
    return _f3_direct_blocks_from_afs(
        afdat,
        popcombs,
        blgsize=blgsize,
        allsnps=allsnps,
        poly_only=poly_only,
        snpwt=snpwt,
        apply_corr=apply_corr,
        outgroupmode=outgroupmode,
        verbose=verbose,
    )


def _set_direct_resampling(
    stats: BlockStats,
    resampling: str,
    *,
    covariance: bool,
) -> BlockStats:
    stats.resampling = resampling
    if resampling == "pairwise_counts":
        return stats
    nominal_lengths = np.asarray(stats.nominal_block_lengths, float)
    if stats.blocks is None:
        raise ValueError("Direct nominal-block resampling requires retained block estimates")
    loo = stats_to_loo(stats.blocks, nominal_lengths)
    contributes = np.isfinite(stats.blocks)
    stats.block_lengths = nominal_lengths
    stats.loo = loo
    if covariance:
        stats.cov, stats.est = jackknife_cov(
            loo,
            nominal_lengths,
            contributes=contributes,
        )
        stats.variances = None
    else:
        results = [
            jack_vec_stats(loo[i, contributes[i]], nominal_lengths[contributes[i]])
            for i in range(len(loo))
        ]
        stats.est = np.asarray([result[0] for result in results], float)
        stats.variances = np.asarray([result[1] for result in results], float)
        stats.cov = None
    stats.snp_counts = None
    stats.influence = None
    stats.contributes = contributes
    return stats


def _validate_cache_resampling(cache: F4BlockCache, resampling: str) -> None:
    if cache.stats.resampling != resampling:
        raise ValueError(
            f"F4 cache uses resampling={cache.stats.resampling!r}, but "
            f"resampling={resampling!r} was requested; rebuild the cache "
            "with the requested resampling method."
        )


def f4_stats(
    data,
    pop1,
    pop2=None,
    pop3=None,
    pop4=None,
    *,
    comb: bool = True,
    unique_only: bool = True,
    afprod: bool = False,
    keep_blocks: bool = True,
    keep_loo: bool = True,
    covariance: bool = True,
    allsnps: bool = False,
    resampling: str = "pairwise_counts",
    verbose: bool = True,
    **kwargs,
) -> BlockStats:
    resampling = _validate_resampling(resampling)
    kwargs = dict(kwargs)
    combos = _f4_combinations(pop1, pop2, pop3, pop4, comb)
    if unique_only:
        combos = combos.drop_duplicates().reset_index(drop=True)
    is_genotype = _default_genotype_allsnps(data)
    if allsnps and isinstance(data, F4BlockCache):
        raise ValueError(_ALLSNPS_DIRECT_ERROR)
    if isinstance(data, F4BlockCache):
        _validate_cache_resampling(data, resampling)
        key = ["pop1", "pop2", "pop3", "pop4"]
        cached = data.stats
        if "model" in combos.columns and "model" in cached.rows.columns:
            key.append("model")
        elif cached.rows.duplicated(key).any():
            raise ValueError(
                "F4 cache contains the same contrast under multiple model-specific "
                "SNP panels; request contrasts with a model column"
            )
        if covariance and cached.cov is None and not data.deferred_covariance:
            raise ValueError(
                "F4BlockCache does not contain a full covariance matrix; "
                "rebuild the cache with covariance=True"
            )
        left = combos.reset_index().rename(columns={"index": "_order"})
        cached_rows = cached.rows.reset_index().rename(columns={"index": "_cache_i"}).drop_duplicates(key)
        merged = left.merge(
            cached_rows,
            on=key,
            how="left",
            suffixes=("", "_cached"),
        ).sort_values("_order")
        if merged["_cache_i"].isna().any():
            missing = combos.loc[merged["_cache_i"].isna().to_numpy(), key]
            heading = _examples_heading("Missing contrasts", len(missing))
            raise ValueError(
                f"F4 cache is missing {len(missing)} requested "
                f"contrast{'s' if len(missing) != 1 else ''}.\n\n"
                + heading
                + "\n"
                + _format_f4_contrasts(missing)
                + "\n\nRebuild the cache with the required models or populations."
            )
        take = merged["_cache_i"].to_numpy(int)
        blocks = cached.blocks[take] if cached.blocks is not None and keep_blocks else None
        loo = cached.loo[take] if cached.loo is not None and keep_loo else None
        est = cached.est[take] if cached.est is not None else None
        cov = cached.cov[np.ix_(take, take)] if covariance and cached.cov is not None else None
        if covariance and cov is None and data.deferred_covariance:
            if resampling == "pairwise_counts":
                cov = _influence_covariance(cached.influence[take], cached.contributes[take])
            else:
                cov, _ = jackknife_cov(
                    cached.loo[take], cached.block_lengths,
                    est=est, contributes=cached.contributes[take],
                )
        variances = (
            np.asarray(cached.variances, float)[take]
            if cached.variances is not None
            else None
        )
        if not covariance and variances is None and cached.cov is not None:
            variances = np.diag(cached.cov)[take]
        stats = BlockStats(
            rows=combos,
            blocks=blocks,
            block_lengths=cached.block_lengths.copy(),
            stat="f4",
            loo=loo,
            est=est,
            cov=cov,
            variances=variances,
            resampling=resampling,
        )
        for name in ("influence", "contributes"):
            value = getattr(cached, name, None)
            if value is not None:
                setattr(stats, name, np.asarray(value)[take])
        if hasattr(cached, "snp_counts") and cached.snp_counts is not None:
            stats.snp_counts = np.asarray(cached.snp_counts, float)[take]
        if hasattr(cached, "nominal_block_lengths"):
            stats.nominal_block_lengths = np.asarray(cached.nominal_block_lengths, float).copy()
        return stats
    if is_genotype:
        if afprod:
            raise ValueError(
                "afprod=True is only available for precomputed f2 data; "
                "direct genotype f4 uses the direct per-SNP estimator"
            )
        kwargs.pop("remove_na", None)
        stats = f4_stats_from_geno(
            data,
            combos,
            allsnps=allsnps,
            verbose=verbose,
            covariance=covariance,
            **kwargs,
        )
        stats = _set_direct_resampling(stats, resampling, covariance=covariance)
        if not keep_blocks:
            stats.blocks = None
        if not keep_loo:
            stats.loo = None
        if not covariance:
            stats.cov = None
        return stats
    if allsnps:
        raise ValueError(_ALLSNPS_DIRECT_ERROR)
    if resampling == "pairwise_counts":
        kwargs.setdefault("remove_na", False)
    pops1 = list(dict.fromkeys(list(combos["pop1"]) + list(combos["pop2"])))
    pops2 = list(dict.fromkeys(list(combos["pop3"]) + list(combos["pop4"])))
    _log(f"Loading {'ap' if afprod else 'f2'} data for {len(pops1) * len(pops2)} population pairs", verbose)
    blocks = get_f2(data, pops=pops1, pops2=pops2, afprod=afprod, **kwargs)
    stat_blocks = []
    _log(f"Computing f4 for {len(combos)} population combinations", verbose)
    for row in combos.itertuples(index=False):
        stat_blocks.append(f4_from_f2(blocks, row.pop1, row.pop2, row.pop3, row.pop4))
    stat_blocks = np.asarray(stat_blocks, float)
    if resampling == "nominal_blocks":
        loo_blocks = est_to_loo(blocks)
        stat_loo = np.asarray(
            [
                f4_from_f2(loo_blocks, row.pop1, row.pop2, row.pop3, row.pop4)
                for row in combos.itertuples(index=False)
            ],
            float,
        )
        if covariance:
            cov, est = jackknife_cov(stat_loo, blocks.block_lengths)
            variances = None
        else:
            stat_results = [
                jack_vec_stats(row, blocks.block_lengths)
                for row in stat_loo
            ]
            est = np.asarray([result[0] for result in stat_results], float)
            variances = np.asarray([result[1] for result in stat_results], float)
            cov = None
    else:
        specs = [
            [
                (0.5, row.pop1, row.pop4),
                (0.5, row.pop2, row.pop3),
                (-0.5, row.pop1, row.pop3),
                (-0.5, row.pop2, row.pop4),
            ]
            for row in combos.itertuples(index=False)
        ]
        est, stat_loo, influence, cov, contributes = _pairwise_composite_jackknife(
            blocks,
            specs,
            covariance=covariance,
        )
    stats = BlockStats(
        rows=combos,
        blocks=stat_blocks if keep_blocks else None,
        block_lengths=blocks.block_lengths.copy(),
        stat="f4",
        loo=stat_loo if keep_loo else None,
        est=est,
        cov=cov if covariance else None,
        resampling=resampling,
    )
    if resampling == "pairwise_counts":
        stats.influence = influence
        stats.contributes = contributes
        if not covariance:
            stats.variances = _influence_variances(influence, contributes)
    elif not covariance:
        stats.variances = variances
    return stats


def qpdstat(
    data,
    pop1,
    pop2=None,
    pop3=None,
    pop4=None,
    *,
    comb: bool = True,
    unique_only: bool = True,
    afprod: bool = False,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    if "allsnps" not in kwargs:
        kwargs["allsnps"] = _default_genotype_allsnps(data)
    stats = f4_stats(
        data,
        pop1,
        pop2,
        pop3,
        pop4,
        comb=comb,
        unique_only=unique_only,
        afprod=afprod,
        verbose=verbose,
        **kwargs,
    )
    out = stats.to_frame()
    bad_est = ~np.isfinite(out["est"].to_numpy(float))
    if np.any(bad_est):
        rows = out.loc[bad_est, ["pop1", "pop2", "pop3", "pop4"]]
        count = int(bad_est.sum())
        heading = _examples_heading("Affected contrasts", count)
        hint = "Inspect population coverage and usable jackknife blocks."
        if bool(kwargs.get("apply_corr", True)):
            if not bool(kwargs["allsnps"]):
                hint = (
                    "With allsnps=False, pseudohaploid singletons may make the f2 correction "
                    "unavailable. For contrasts with four distinct populations, use "
                    "apply_corr=False. Alternatives: inspect population coverage and usable "
                    "jackknife blocks, or use allsnps=True with genotype input."
                )
            else:
                hint = (
                    "With apply_corr=True, a repeated population in an f4 contrast can "
                    "require at least two independent allele observations. A pseudohaploid "
                    "singleton cannot supply that finite-sample correction."
                )
        raise ValueError(
            f"f4 produced {count} non-finite estimate{'s' if count != 1 else ''}.\n\n"
            + heading
            + "\n"
            + _format_f4_contrasts(rows)
            + "\n\n"
            + hint
        )
    snp_counts = getattr(stats, "snp_counts", None)
    if snp_counts is not None:
        out["n"] = np.sum(snp_counts, axis=1).astype(int)
    return FStatsFrame(out)


def f4(
    data,
    pop1,
    pop2=None,
    pop3=None,
    pop4=None,
    *,
    comb: bool = True,
    unique_only: bool = True,
    afprod: bool = False,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Compute f4 statistics and return a result data frame."""
    return qpdstat(
        data,
        pop1,
        pop2,
        pop3,
        pop4,
        comb=comb,
        unique_only=unique_only,
        afprod=afprod,
        verbose=verbose,
        **kwargs,
    )


def _contrast_pops(pops: Sequence[str], base: str | None, name: str) -> tuple[str, list[str]]:
    pops = list(pops)
    _require_unique_pops(pops, name)
    if len(pops) < 2:
        raise ValueError(f"{name} must contain at least two populations")
    base = pops[0] if base is None else base
    if base not in pops:
        raise ValueError(f"{name}_base must be included in {name}")
    return base, [p for p in pops if p != base]


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
    stats = f4_stats(data, combos[["pop1", "pop2", "pop3", "pop4"]], unique_only=False, verbose=verbose, **kwargs)
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
    qpw = qpwave_f4stats(data, left, right, left_base=left_base, right_base=right_base, verbose=verbose, **kwargs)
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
        f4_model_cache(
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
        fit = qpadm_fit(xmat, qinv, rank, fudge=fudge, iterations=iterations)
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
            fit = qpadm_fit(submat, subqinv, rank, fudge=fudge, iterations=iterations)
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
            nested_fit = qpadm_fit(
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
    qpw = qpwave_f4stats(data, left=left_full, right=right, verbose=verbose, **kwargs)
    _validate_qpadm_f4(
        qpw,
        allsnps=bool(kwargs["allsnps"]),
        apply_corr=bool(kwargs.get("apply_corr", True)),
    )
    xmat = qpw.matrix
    qinv = _qinv_from_cov(qpw.cov, fudge=fudge, fudge_twice=fudge_twice)
    rank = len(sources) - 1
    fit = qpadm_fit(xmat, qinv, rank, fudge=fudge, iterations=iterations)
    wcov = _weights_covariance(qpw, qinv, rank, fudge, iterations) if getcov else np.full((len(sources), len(sources)), np.nan)
    se = np.sqrt(np.diag(wcov)) if wcov.ndim == 2 and wcov.shape[0] == len(sources) else np.full(len(sources), np.nan)
    weights = pd.DataFrame({"target": target, "left": sources, "weight": fit["weights"], "se": se})
    weights["z"] = weights["weight"] / weights["se"]
    rankdrop = qpadm_rankdrop(xmat, qinv, fudge=fudge, iterations=iterations)
    popdrop_result = qpadm_popdrop(
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
        f4_model_cache(data, models, resampling=resampling, verbose=verbose, **kwargs)
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


f3 = qp3pop
