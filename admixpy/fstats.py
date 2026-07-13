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

from .genotypes import AfData, anygeno_to_afs, discard_from_aftable, get_block_lengths, is_polymorphic


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

    @property
    def se(self) -> np.ndarray:
        if self.cov is None:
            return np.full(len(self.rows), np.nan)
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
        # In allsnps mode the per-stat per-block SNP counts are attached as
        # `snp_counts`; in that case report each stat with its own block sizes
        # (matches admixtools::jack_dat_stats, the formula used by qpdstat-allsnps).
        snp_counts = getattr(self, "snp_counts", None)
        if snp_counts is not None and self.blocks is not None:
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


def _has_singleton_observations(afs: np.ndarray, counts: np.ndarray) -> bool:
    return bool(np.any(np.isfinite(afs) & np.isfinite(counts) & (counts < 2)))


def _warn_singleton_observations(stat: str, apply_corr: bool, *pairs: tuple[np.ndarray, np.ndarray]) -> None:
    if not any(_has_singleton_observations(afs, counts) for afs, counts in pairs):
        return
    if apply_corr:
        message = (
            f"{stat} bias correction requires at least two independent allele observations; "
            "excluding population-pair SNP values with count < 2"
        )
    else:
        message = (
            f"{stat} includes population-pair SNP values with count < 2 because apply_corr=False; "
            "those values cannot be estimated without sampling bias"
        )
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
) -> np.ndarray:
    a1, a2 = np.asarray(afmat1, float), np.asarray(afmat2, float)
    c1, c2 = np.asarray(countmat1, float), np.asarray(countmat2, float)
    out = np.empty((a1.shape[1], a2.shape[1], len(block_lengths)), dtype=float)
    snpwt = None if snpwt is None else np.asarray(snpwt, float)
    _warn_singleton_observations("f2", apply_corr, (a1, c1), (a2, c2))
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
        vals = _outer_pair(a1[start:stop], a2[start:stop])
        out[:, :, b] = np.nanmean(vals, axis=2)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a1, a2 = np.asarray(afmat1, float), np.asarray(afmat2, float)
    c1, c2 = np.asarray(countmat1, float), np.asarray(countmat2, float)
    out = np.empty((a1.shape[1], a2.shape[1], len(block_lengths)), dtype=float)
    snp_counts = np.zeros_like(out)
    num_sums = np.full_like(out, np.nan)
    den_sums = np.full_like(out, np.nan)
    snpwt = None if snpwt is None else np.asarray(snpwt, float)
    _warn_singleton_observations("FST", apply_corr, (a1, c1), (a2, c2))
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
            arr = mats_to_f2arr(a1, a2, c1, c2, bl, snpwt, apply_corr, verbose=verbose)
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
    remove_na: bool = True,
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
            raise ValueError("No blocks remain after discarding blocks with missing values")
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
    resampling = _validate_resampling(resampling)
    allsnps = bool(kwargs.pop("allsnps", False))
    if isinstance(data, F4ModelCache):
        if allsnps:
            raise ValueError(_ALLSNPS_DIRECT_ERROR)
        return data
    if isinstance(data, F4BlockCache):
        if allsnps:
            raise ValueError(_ALLSNPS_DIRECT_ERROR)
        return data
    models = _models_frame(models)
    left_pops = []
    right_pops = []
    for row in models.itertuples(index=False):
        left_pops.extend(_model_left_with_target(row))
        right_pops.extend(_as_pop_list(row.right))
    left_pops = list(dict.fromkeys(left_pops))
    right_pops = list(dict.fromkeys(right_pops))
    if allsnps:
        combos = []
        for model_i, row in enumerate(models.itertuples(index=False), start=1):
            left = _model_left_with_target(row)
            right = _as_pop_list(row.right)
            if len(left) < 2 or len(right) < 2:
                continue
            left_base, row_pops = _contrast_pops(left, None, "left")
            right_base, col_pops = _contrast_pops(right, None, "right")
            for row_pop in row_pops:
                for col_pop in col_pops:
                    combos.append(
                        {
                            "model": model_i,
                            "pop1": row_pop,
                            "pop2": left_base,
                            "pop3": col_pop,
                            "pop4": right_base,
                        }
                    )
        _log(f"Loading reusable f4 cache for {len(combos)} population quadruples", verbose)
        stats = f4_stats(
            data,
            pd.DataFrame(combos),
            unique_only=False,
            allsnps=True,
            resampling=resampling,
            verbose=verbose,
            **kwargs,
        )
        return F4BlockCache(stats=stats, models=models, allsnps=True)
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


def read_f2(f2_dir: str | Path, pops: Sequence[str] | None = None, pops2: Sequence[str] | None = None, type: str = "f2", remove_na: bool = True) -> F2Blocks:
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
        return read_f2(data, pops, pops2, type=type_, remove_na=kwargs.get("remove_na", True))
    return f2_from_geno(data, pops=pops, pops2=pops2, **kwargs)


def est_to_loo(blocks: F2Blocks | np.ndarray, block_lengths: Sequence[int] | None = None):
    arr = blocks.data if isinstance(blocks, F2Blocks) else np.asarray(blocks, float)
    bl = blocks.block_lengths if isinstance(blocks, F2Blocks) else np.asarray(block_lengths, float)
    numer = np.nansum(arr * bl[None, None, :], axis=2)
    denom = np.nansum(np.where(np.isfinite(arr), bl[None, None, :], 0), axis=2)
    tot = np.full(numer.shape, np.nan, dtype=float)
    np.divide(numer, denom, out=tot, where=denom != 0)
    rel = bl / bl.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (tot[:, :, None] - arr * rel[None, None, :]) / (1 - rel[None, None, :])
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
    rel = bl / bl.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (tot[:, None] - arr * rel[None, :]) / (1 - rel[None, :])


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
    # (allsnps mode). Mirrors admixtools::est_to_loo_dat + jack_dat_stats, which
    # use the 'tot' form of cpp_jack_vec_stats. Blocks with n=0 or non-finite
    # block estimate are dropped.
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
    # Unequal-delete-block jackknife, matching the `tot` form used by
    # admixtools::jack_dat_stats while recomputing the nonlinear ratio.
    # Report the full-data ratio as the point estimate and use the
    # bias-corrected jackknife center only for the variance calculation.
    jack_center = float(np.sum(total - loo) + np.sum(loo * n) / total_n)
    tau = h * total - (h - 1.0) * loo
    var = float(np.mean((tau - jack_center) ** 2 / (h - 1.0)))
    return float(total), var


def jackknife_cov(loo_mat: np.ndarray, block_lengths: Sequence[int], est: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    loo = np.asarray(loo_mat, float)
    if loo.ndim == 1:
        loo = loo[None, :]
    bl = np.asarray(block_lengths, float)
    if loo.shape[1] != len(bl):
        if loo.shape[0] == len(bl):
            loo = loo.T
        else:
            raise ValueError("loo_mat must have one column per block")
    h = bl.sum() / bl
    if est is None:
        est_vec = np.array([jack_vec_stats(row, bl)[0] for row in loo])
    else:
        est_vec = np.asarray(est, float)
    cov = np.full((loo.shape[0], loo.shape[0]), np.nan)
    for i in range(loo.shape[0]):
        for j in range(i, loo.shape[0]):
            keep = np.isfinite(loo[i]) & np.isfinite(loo[j])
            if np.any(keep):
                val = np.mean((est_vec[i] - loo[i, keep]) * (est_vec[j] - loo[j, keep]) * (h[keep] - 1))
                cov[i, j] = cov[j, i] = val
    return cov, est_vec


def block_covariance(stats: BlockStats | np.ndarray, block_lengths: Sequence[int] | None = None) -> np.ndarray:
    if isinstance(stats, BlockStats):
        loo = stats.loo
        if loo is None:
            if stats.blocks is None:
                raise ValueError("BlockStats must contain loo or blocks to estimate covariance")
            loo = stats_to_loo(stats.blocks, stats.block_lengths)
        cov, est = jackknife_cov(loo, stats.block_lengths, stats.est)
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
    nstats = influence.shape[0]
    cov = np.full((nstats, nstats), np.nan, dtype=float)
    for i in range(nstats):
        for j in range(i, nstats):
            keep = (
                contributes[i]
                & contributes[j]
                & np.isfinite(influence[i])
                & np.isfinite(influence[j])
            )
            if int(np.sum(keep)) >= 2:
                value = float(np.mean(influence[i, keep] * influence[j, keep]))
                cov[i, j] = cov[j, i] = value
    return cov


def _pairwise_composite_jackknife(
    blocks: F2Blocks,
    specs: Sequence[Sequence[tuple[float, str, str]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    return totals, loo, influence, _influence_covariance(influence, contributes)


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
    unique_only: bool = True,
    resampling: str = "pairwise_counts",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    resampling = _validate_resampling(resampling)
    kwargs = dict(kwargs)
    if resampling == "pairwise_counts":
        kwargs.setdefault("remove_na", False)
    combos = _f3_combinations(data, pop1, pop2, pop3, unique_only)
    if unique_only:
        combos = combos.drop_duplicates().reset_index(drop=True)
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
        pairwise_est, _, _, pairwise_cov = _pairwise_composite_jackknife(blocks, specs)
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
        return pop1[cols].copy()
    p1, p2, p3, p4 = map(_as_list, (pop1, pop2, pop3, pop4))
    if p1 is None or p2 is None or p3 is None or p4 is None:
        raise ValueError("pop1, pop2, pop3, and pop4 are required for f4")
    if comb:
        return pd.DataFrame(product(p1, p2, p3, p4), columns=["pop1", "pop2", "pop3", "pop4"])
    lengths = {len(p1), len(p2), len(p3), len(p4)}
    if len(lengths) != 1:
        raise ValueError("With comb=False, pop1/pop2/pop3/pop4 must have equal lengths")
    return pd.DataFrame({"pop1": p1, "pop2": p2, "pop3": p3, "pop4": p4})


def _f4_direct_blocks_from_afs(
    afdat: AfData,
    combos: pd.DataFrame,
    blgsize: float = 0.05,
    allsnps: bool = False,
    poly_only: bool = False,
    snpwt: Sequence[float] | None = None,
    verbose: bool = True,
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
    pop_i = {p: i for i, p in enumerate(pops)}
    idx = np.asarray([[pop_i[getattr(row, c)] for c in cols] for row in combos.itertuples(index=False)], dtype=int)
    block_lengths = get_block_lengths(afdat.snpfile, blgsize)
    nstats = len(combos)
    out = np.full((nstats, len(block_lengths)), np.nan, dtype=float)
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
                use &= np.nanmax(vals, axis=1) != np.nanmin(vals, axis=1)
            use_by_model[model] = use

    start = 0
    for b, n in enumerate(block_lengths):
        stop = start + int(n)
        _log_block("direct f4", b, len(block_lengths), start, stop, verbose)
        block = arr[start:stop]
        for stat_i, row in enumerate(combos.itertuples(index=False)):
            p = idx[stat_i]
            vals = block[:, p]
            use = np.isfinite(vals).all(axis=1) if allsnps else use_by_model[getattr(row, "model")][start:stop]
            if allsnps and poly_only:
                finite_vals = vals[use]
                if finite_vals.size:
                    use_idx = np.where(use)[0]
                    use[use_idx] &= np.max(finite_vals, axis=1) != np.min(finite_vals, axis=1)
            if not np.any(use):
                continue
            f4vals = (vals[use, 0] - vals[use, 1]) * (vals[use, 2] - vals[use, 3])
            if snpwt is not None:
                f4vals = f4vals * snpwt[start:stop][use]
            out[stat_i, b] = float(np.mean(f4vals))
            snp_counts[stat_i, b] = int(use.sum())
        start = stop

    effective_lengths = np.nanmax(snp_counts, axis=0)
    effective_lengths = np.where(effective_lengths > 0, effective_lengths, block_lengths).astype(float)
    jacks = [_count_jackknife(out[i], snp_counts[i]) for i in range(nstats)]
    est = np.asarray([jack.total for jack in jacks], float)
    loo = np.asarray([jack.loo for jack in jacks], float)
    influence = np.asarray([jack.influence for jack in jacks], float)
    contributes = np.asarray([jack.contributes for jack in jacks], bool)
    cov = _influence_covariance(influence, contributes)
    rows = combos.drop(columns=["model"]) if set(combos["model"]) == {1} else combos
    stats = BlockStats(rows=rows.reset_index(drop=True), blocks=out, block_lengths=effective_lengths, stat="f4", loo=loo, est=est, cov=cov)
    stats.snp_counts = snp_counts
    return stats


def f4_stats_from_geno(
    pref: str | Path,
    popcombs: pd.DataFrame,
    blgsize: float = 0.05,
    maxmiss: float | None = None,
    minmaf: float = 0,
    maxmaf: float = 0.5,
    outpop: str | None = None,
    transitions: bool = True,
    transversions: bool = True,
    auto_only: bool = True,
    keepsnps=None,
    allsnps: bool = False,
    poly_only: bool = False,
    format: str | None = None,
    adjust_pseudohaploid=True,
    chunk_size: int = 10_000,
    tgeno_chunked: bool = False,
    verbose: bool = True,
) -> BlockStats:
    cols = ["pop1", "pop2", "pop3", "pop4"]
    pops = list(dict.fromkeys(popcombs[cols].to_numpy().reshape(-1)))
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
    if maxmiss is None:
        maxmiss = 1 if allsnps else 0
    afdat = discard_from_aftable(
        afdat,
        maxmiss=maxmiss,
        minmaf=minmaf,
        maxmaf=maxmaf,
        outpop=outpop,
        transitions=transitions,
        transversions=transversions,
        auto_only=auto_only,
        keepsnps=keepsnps,
        poly_only=False,
    )
    return _f4_direct_blocks_from_afs(afdat, popcombs, blgsize=blgsize, allsnps=allsnps, poly_only=poly_only, verbose=verbose)


def f4_stats(
    data,
    pop1,
    pop2=None,
    pop3=None,
    pop4=None,
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
    if afprod and allsnps:
        raise ValueError("afprod=True and allsnps=True together are not supported")
    combos = _f4_combinations(pop1, pop2, pop3, pop4, comb)
    if unique_only:
        combos = combos.drop_duplicates().reset_index(drop=True)
    if allsnps and isinstance(data, F4BlockCache):
        raise ValueError(_ALLSNPS_DIRECT_ERROR)
    if isinstance(data, F4BlockCache):
        key = ["pop1", "pop2", "pop3", "pop4"]
        cached = data.stats
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
            raise ValueError(f"F4 cache does not contain requested combinations:\n{missing}")
        take = merged["_cache_i"].to_numpy(int)
        blocks = cached.blocks[take] if cached.blocks is not None and keep_blocks else None
        loo = cached.loo[take] if cached.loo is not None and keep_loo else None
        est = cached.est[take] if cached.est is not None else None
        cov = cached.cov[np.ix_(take, take)] if covariance and cached.cov is not None else None
        return BlockStats(rows=combos, blocks=blocks, block_lengths=cached.block_lengths.copy(), stat="f4", loo=loo, est=est, cov=cov)
    if allsnps:
        if isinstance(data, (F2Blocks, F4ModelCache)) or (isinstance(data, (str, Path)) and Path(data).is_dir()):
            raise ValueError(_ALLSNPS_DIRECT_ERROR)
        stats = f4_stats_from_geno(data, combos, allsnps=True, verbose=verbose, **kwargs)
        if not keep_blocks:
            stats.blocks = None
        if not keep_loo:
            stats.loo = None
        if not covariance:
            stats.cov = None
        return stats
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
        cov, est = jackknife_cov(stat_loo, blocks.block_lengths)
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
        est, stat_loo, _, cov = _pairwise_composite_jackknife(blocks, specs)
    return BlockStats(
        rows=combos,
        blocks=stat_blocks if keep_blocks else None,
        block_lengths=blocks.block_lengths.copy(),
        stat="f4",
        loo=stat_loo if keep_loo else None,
        est=est,
        cov=cov if covariance else None,
    )


def qpdstat(
    data,
    pop1,
    pop2=None,
    pop3=None,
    pop4=None,
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
    return stats.to_frame()


def f4(
    data,
    pop1,
    pop2=None,
    pop3=None,
    pop4=None,
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
    if rank >= min(mat.shape):
        raise ValueError("rank must be smaller than min(number of left contrasts, number of right contrasts)")
    cov = np.asarray(qpw.cov, float)
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
    qpw = qpwave_f4stats(data, left, right, left_base=left_base, right_base=right_base, verbose=verbose, **kwargs)
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
    source = f4_model_cache(data, models, verbose=verbose, **kwargs) if use_cache else data
    qp_kwargs = {} if use_cache else kwargs
    rows = []
    for model_i, row in enumerate(models.itertuples(index=False), start=1):
        left = _model_left_with_target(row)
        right = _as_pop_list(row.right)
        out = qpwave(
            source,
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
    if rank == 0:
        fit = {"weights": np.ones(1), "A": np.zeros((xmat.shape[0], 0)), "B": np.zeros((0, xmat.shape[1]))}
    else:
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
) -> pd.DataFrame:
    sources = list(sources)
    nsrc = len(sources)
    ncol = xmat.shape[1]
    rows = []
    for nkeep in range(nsrc, 0, -1):
        for keep_tuple in combinations(range(nsrc), nkeep):
            keep = list(keep_tuple)
            flat = np.concatenate([np.arange(i * ncol, (i + 1) * ncol) for i in keep])
            submat = xmat[keep, :]
            subqinv = qinv[np.ix_(flat, flat)]
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
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    best, dofdiff, chisqdiff, p_nested = _popdrop_nested_chain(out, nsrc)
    out["best"] = best
    out["dofdiff"] = dofdiff
    out["chisqdiff"] = chisqdiff
    out["p_nested"] = p_nested
    out["status"] = np.where(~np.isfinite(out["p"]), "NA", np.where(out["p"] > 0.05, "PASS", "FAIL"))
    return out.sort_values(["dof", "pat"]).reset_index(drop=True)


def _popdrop_nested_chain(out: pd.DataFrame, nsources: int):
    # Mirrors admixtools::qpadm_popdrop: 'best' is the seed of all (rnk-1)-rank
    # rows plus, for each lower rank r in (rnk-2..1), the lowest-chisq feasible
    # child of any pattern already in the chain. dofdiff/chisqdiff/p_nested are
    # filled only on chain rows, sequentially along descending dof.
    n = len(out)
    best = np.zeros(n, dtype=bool)
    dofdiff = np.full(n, np.nan)
    chisqdiff = np.full(n, np.nan)
    p_nested = np.full(n, np.nan)
    rnk = nsources - 1
    if rnk < 1:
        return best, dofdiff, chisqdiff, p_nested

    f4rank_arr = out["f4rank"].to_numpy()
    pat_arr = out["pat"].to_numpy()
    feasible_arr = out["feasible"].to_numpy()
    chisq_arr = out["chisq"].to_numpy()
    dof_arr = out["dof"].to_numpy()

    seed_idx = np.where(f4rank_arr == rnk - 1)[0]
    if seed_idx.size == 0:
        return best, dofdiff, chisqdiff, p_nested
    best[seed_idx] = True
    chain_idx = list(seed_idx)
    chain_pats = set(pat_arr[seed_idx])

    def child_patterns(pat: str) -> list[str]:
        return [pat[:k] + "1" + pat[k + 1 :] for k, c in enumerate(pat) if c == "0"]

    for level in range(rnk - 2, 0, -1):
        children: set[str] = set()
        for p in chain_pats:
            children.update(child_patterns(p))
        if not children:
            break
        mask = (f4rank_arr == level) & np.isin(pat_arr, list(children)) & feasible_arr
        if not mask.any():
            continue
        cand_idx = np.where(mask)[0]
        winner = cand_idx[int(np.nanargmin(chisq_arr[cand_idx]))]
        best[winner] = True
        chain_idx.append(winner)
        chain_pats.add(pat_arr[winner])

    chain_sorted = sorted(chain_idx, key=lambda i: -dof_arr[i])
    for k in range(len(chain_sorted) - 1):
        i, j = chain_sorted[k], chain_sorted[k + 1]
        dd = dof_arr[i] - dof_arr[j]
        cd = chisq_arr[i] - chisq_arr[j]
        dofdiff[i] = dd
        chisqdiff[i] = cd
        if np.isfinite(cd) and np.isfinite(dd) and dd > 0:
            p_nested[i] = _chi2_sf(cd, int(dd))
    return best, dofdiff, chisqdiff, p_nested


def _qinv_from_cov(cov: np.ndarray, fudge: float, fudge_twice: bool = False) -> np.ndarray:
    cov = np.asarray(cov, float).copy()
    cov[np.diag_indices_from(cov)] += fudge * np.trace(cov)
    if fudge_twice:
        cov[np.diag_indices_from(cov)] += fudge * np.trace(cov)
    try:
        return linalg.inv(cov)
    except linalg.LinAlgError:
        return linalg.pinv(cov)


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
    return np.cov(wmat * math.sqrt(len(wmat) - 1), rowvar=False)


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
    **kwargs,
) -> QpAdmResult:
    if not isinstance(target, str):
        raise TypeError("target must be a population name string; positional qpadm order is qpadm(data, target, left, right)")
    sources = _as_pop_list(left if sources is None else sources)
    if right is None:
        raise ValueError("right populations are required")
    right = _as_pop_list(right)
    _require_unique_pops(sources, "left")
    if len(sources) < 1:
        raise ValueError("At least one source/left population is required")
    if len(right) < 1:
        raise ValueError("At least one right/reference population is required")
    if "allsnps" not in kwargs:
        kwargs["allsnps"] = _default_genotype_allsnps(data)
    left_full = [target] + [p for p in sources if p != target]
    qpw = qpwave_f4stats(data, left=left_full, right=right, verbose=verbose, **kwargs)
    xmat = qpw.matrix
    qinv = _qinv_from_cov(qpw.cov, fudge=fudge, fudge_twice=fudge_twice)
    rank = len(sources) - 1
    fit = qpadm_fit(xmat, qinv, rank, fudge=fudge, iterations=iterations)
    wcov = _weights_covariance(qpw, qinv, rank, fudge, iterations) if getcov else np.full((len(sources), len(sources)), np.nan)
    se = np.sqrt(np.diag(wcov)) if wcov.ndim == 2 and wcov.shape[0] == len(sources) else np.full(len(sources), np.nan)
    weights = pd.DataFrame({"target": target, "left": sources, "weight": fit["weights"], "se": se})
    weights["z"] = weights["weight"] / weights["se"]
    rankdrop = qpadm_rankdrop(xmat, qinv, fudge=fudge, iterations=iterations)
    popdrop = qpadm_popdrop(xmat, qinv, sources, fudge=fudge, iterations=iterations)
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
        popdrop=popdrop,
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
            _require_unique_pops(_as_pop_list(row.left), "left")
        except ValueError as err:
            raise ValueError(f"Model {model_i}: {err}") from None
    qpadm_keys = {"fudge", "fudge_twice", "iterations", "getcov", "return_f4", "return_stats", "return_cov"}
    if "allsnps" not in kwargs:
        kwargs["allsnps"] = _default_genotype_allsnps(data)
    resampling = _validate_resampling(kwargs.pop("resampling", "pairwise_counts"))
    qpadm_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in qpadm_keys}
    source = (
        f4_model_cache(data, models, resampling=resampling, verbose=verbose, **kwargs)
        if use_cache
        else data
    )
    qp_kwargs = (
        {**qpadm_kwargs, "resampling": resampling}
        if use_cache
        else {**kwargs, **qpadm_kwargs, "resampling": resampling}
    )
    rows = []
    for model_i, row in enumerate(models.itertuples(index=False), start=1):
        res = qpadm(
            source,
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
