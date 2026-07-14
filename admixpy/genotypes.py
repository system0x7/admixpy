from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
import math
import re
import warnings

import numpy as np
import pandas as pd


def _packed_ancestrymap_lut() -> np.ndarray:
    lut = np.empty((256, 4), dtype=np.uint8)
    vals = np.arange(256, dtype=np.uint16)
    lut[:, 0] = (vals & 192) >> 6
    lut[:, 1] = (vals & 48) >> 4
    lut[:, 2] = (vals & 12) >> 2
    lut[:, 3] = vals & 3
    return lut


def _plink_lut() -> np.ndarray:
    lut = np.empty((256, 4), dtype=np.uint8)
    vals = np.arange(256, dtype=np.uint16)
    for j in range(4):
        code = (vals >> (2 * j)) & 3
        lut[:, j] = np.where(code < 2, code + 2, 3 - code)
    return lut


_PACKED_ANCESTRYMAP_LUT = _packed_ancestrymap_lut()
_PLINK_LUT = _plink_lut()


def _log(message: str, verbose: bool):
    if verbose:
        print(message, flush=True)


def _log_chunk(label: str, i: int, total: int, start: int, stop: int, verbose: bool) -> None:
    # Single carriage-returned status line, ~20 updates plus first/last.
    if not verbose or total <= 0:
        return
    step = max(1, total // 20)
    is_last = i == total
    if i == 1 or is_last or i % step == 0:
        end = "\n" if is_last else ""
        print(
            f"\rReading {label} chunk {i}/{total}: SNPs {start}-{stop}      ",
            end=end,
            flush=True,
        )


def _ntest(adjust, n: int) -> int:
    if isinstance(adjust, bool) or not isinstance(adjust, int):
        return min(1000, n)
    return min(int(adjust), n)


def _packed_record_len(n: int) -> int:
    return max(48, math.ceil(n / 4))


def _parse_geno_header_with_kind(path: str | Path) -> tuple[str, int, int] | None:
    head = Path(path).read_bytes()[:48]
    try:
        text = head.decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return None
    match = re.match(r"^(GENO|TGENO)\s+(\d+)\s+(\d+)\s+[0-9a-fA-F]+\s+[0-9a-fA-F]+", text)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), int(match.group(3))


def _parse_geno_header(path: str | Path) -> tuple[int, int] | None:
    header = _parse_geno_header_with_kind(path)
    if header is None:
        return None
    _, nind, nsnp = header
    return nind, nsnp


def _tgeno_path(pref: str | Path) -> Path:
    pref = str(pref)
    geno = Path(pref + ".geno")
    if geno.exists():
        return geno
    return Path(pref + ".tgeno")


def _tgeno_layout(path: str | Path, nsnp: int, nind: int) -> tuple[int, int]:
    path = Path(path)
    size = path.stat().st_size
    min_record_len = math.ceil(nsnp / 4)
    header = _parse_geno_header_with_kind(path)
    if header is not None:
        data_size = size - 48
        if data_size >= 0 and nind > 0 and data_size % nind == 0:
            record_len = data_size // nind
            if record_len >= min_record_len:
                return 48, record_len
    if nind > 0 and size % nind == 0:
        record_len = size // nind
        if record_len >= min_record_len:
            return 0, record_len
    return _packed_record_len(nsnp), _packed_record_len(nsnp)


def detect_geno_format(
    path: str | Path,
    nind: int | None = None,
    nsnp: int | None = None,
    validate: bool = False,
) -> str:
    """Return 'packedancestrymap', 'tgeno', or 'eigenstrat'.

    Without a GENO binary header, only the first row is checked for EIGENSTRAT
    shape (length nind, characters in {0,1,2,9}). Pass validate=True to scan every
    row; this requires nsnp.
    """
    path = Path(path)
    header = _parse_geno_header_with_kind(path)
    if header is not None:
        header_kind, header_nind, header_nsnp = header
        if nind is not None and header_nind != nind:
            raise ValueError(f"GENO header nind={header_nind} does not match ind file nind={nind}")
        if nsnp is not None and header_nsnp != nsnp:
            raise ValueError(f"GENO header nsnp={header_nsnp} does not match snp file nsnp={nsnp}")
        nind = header_nind
        nsnp = header_nsnp
        if header_kind == "TGENO":
            _tgeno_layout(path, nsnp, nind)
            return "tgeno"
        size = path.stat().st_size
        packed_size = (nsnp + 1) * _packed_record_len(nind)
        tgeno_size = (nind + 1) * _packed_record_len(nsnp)
        if size == packed_size:
            return "packedancestrymap"
        if size == tgeno_size:
            return "tgeno"
        raise ValueError(
            f"GENO header found, but file size {size} matches neither PACKEDANCESTRYMAP "
            f"({packed_size}) nor TGENO ({tgeno_size})"
        )

    if nind is None:
        raise ValueError("nind is required to validate EIGENSTRAT text genotype files")
    with path.open("rb") as fh:
        first = fh.readline()
        if not first:
            raise ValueError("Genotype file is empty")
        row = first.rstrip(b"\r\n")
        if len(row) != nind or any(c not in b"0129" for c in row):
            raise ValueError("No GENO binary header and file is not valid EIGENSTRAT text")
        if validate:
            if nsnp is None:
                raise ValueError("nsnp is required when validate=True")
            for i in range(1, nsnp):
                line = fh.readline()
                if not line:
                    raise ValueError(f"EIGENSTRAT genotype file ended before SNP {i + 1}")
                row = line.rstrip(b"\r\n")
                if len(row) != nind or any(c not in b"0129" for c in row):
                    raise ValueError(f"Invalid EIGENSTRAT row at SNP {i + 1}")
            if fh.readline():
                raise ValueError("EIGENSTRAT genotype file has more rows than the SNP file")
    return "eigenstrat"


@dataclass
class AfData:
    afs: pd.DataFrame
    counts: pd.DataFrame
    snpfile: pd.DataFrame


def _read_table(path: str | Path, names: Sequence[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=list(names), comment="#")


def read_ind(path: str | Path) -> pd.DataFrame:
    return _read_table(path, ["iid", "sex", "population"])


def read_snp(path: str | Path, plink: bool = False) -> pd.DataFrame:
    names = ["CHR", "SNP", "cm", "POS", "A1", "A2"] if plink else ["SNP", "CHR", "cm", "POS", "A1", "A2"]
    out = _read_table(path, names)
    out["cm"] = pd.to_numeric(out["cm"], errors="coerce")
    out["POS"] = pd.to_numeric(out["POS"], errors="coerce")
    return out


def _match_samples(haveinds: Sequence[str], havepops: Sequence[str], inds=None, pops=None):
    haveinds = np.asarray(haveinds, dtype=object)
    havepops = np.asarray(havepops, dtype=object)
    if inds is not None:
        inds = list(inds)
        missing = sorted(set(inds) - set(haveinds))
        if missing:
            raise ValueError(f"Individuals missing in ind/fam file: {missing}")
    if pops is not None:
        pops = list(pops)
        if inds is None:
            missing = sorted(set(pops) - set(havepops))
            if missing:
                raise ValueError(f"Populations missing in ind/fam file: {missing}")

    labels = np.full(len(haveinds), None, dtype=object)
    if inds is None and pops is None:
        labels[:] = havepops
        upops = list(dict.fromkeys(havepops))
    elif inds is not None and pops is not None:
        if len(inds) != len(pops):
            raise ValueError("'inds' and 'pops' must have the same length")
        mapping = dict(zip(inds, pops))
        for i, ind in enumerate(haveinds):
            labels[i] = mapping.get(ind)
        upops = list(dict.fromkeys(pops))
    elif pops is not None:
        wanted = set(pops)
        labels = np.array([p if p in wanted else None for p in havepops], dtype=object)
        upops = list(dict.fromkeys(pops))
    else:
        wanted = set(inds)
        labels = np.array([ind if ind in wanted else None for ind in haveinds], dtype=object)
        upops = [ind for ind in haveinds if ind in wanted]

    pop_index = {p: i for i, p in enumerate(upops)}
    indvec = np.array([pop_index[x] if x in pop_index else -1 for x in labels], dtype=np.int64)
    return indvec, upops


def _detect_pseudohaploid(geno: np.ndarray, indvec: np.ndarray, ntest: int | bool) -> np.ndarray:
    ploidy = np.full(len(indvec), 2, dtype=float)
    if not ntest:
        return ploidy
    n = _ntest(ntest, geno.shape[0])
    for i in np.where(indvec >= 0)[0]:
        vals = geno[:n, i]
        vals = vals[np.isfinite(vals)]
        if vals.size and not np.any(vals == 1):
            ploidy[i] = 1
    return ploidy


def _geno_to_af_arrays(geno: np.ndarray, indvec: np.ndarray, popnames: Sequence[str], ploidy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nsnp = geno.shape[0]
    npop = len(popnames)
    indvec = np.asarray(indvec)
    keep = indvec >= 0
    if not np.any(keep):
        return np.full((nsnp, npop), np.nan, dtype=float), np.zeros((nsnp, npop), dtype=float)
    g = geno[:, keep].astype(float)
    p = ploidy[keep]
    pops = indvec[keep].astype(np.int64)
    present = np.isfinite(g)
    denom = present * p
    alt = np.where(present, g / (3.0 - p), 0.0)
    onehot = np.zeros((len(pops), npop), dtype=float)
    onehot[np.arange(len(pops)), pops] = 1.0
    counts = denom @ onehot
    alt_sum = alt @ onehot
    with np.errstate(invalid="ignore", divide="ignore"):
        afs = alt_sum / counts
    afs[counts == 0] = np.nan
    return afs, counts


def _geno_to_afs(geno: np.ndarray, indvec: np.ndarray, popnames: Sequence[str], ploidy: np.ndarray, snp_ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    afs, counts = _geno_to_af_arrays(geno, indvec, popnames, ploidy)
    return (
        pd.DataFrame(afs, index=snp_ids, columns=popnames),
        pd.DataFrame(counts, index=snp_ids, columns=popnames),
    )


def _read_eigenstrat_geno(
    path: str | Path,
    nind: int,
    first: int = 1,
    last: int | None = None,
    sample_indices: np.ndarray | None = None,
) -> np.ndarray:
    # Spec-compliant EIGENSTRAT: each row is `nind` digit chars + a fixed
    # terminator ('\n' or '\r\n'); the file is a regular grid we seek into.
    path = Path(path)
    if last is not None and last < first:
        n_cols = nind if sample_indices is None else len(np.asarray(sample_indices))
        return np.empty((0, n_cols), dtype=float)
    with open(path, "rb") as fh:
        head = fh.readline()
    line_bytes = len(head)
    if line_bytes - nind not in (1, 2):
        raise ValueError(
            f"Unexpected EIGENSTRAT line stride: {line_bytes} bytes for nind={nind} "
            f"(expected nind+1 for LF or nind+2 for CRLF terminators)"
        )
    file_size = path.stat().st_size
    if file_size % line_bytes == 0:
        total_rows = file_size // line_bytes
    elif (file_size - nind) > 0 and (file_size - nind) % line_bytes == 0:
        total_rows = (file_size - nind) // line_bytes + 1
    else:
        raise ValueError(
            f"EIGENSTRAT file size {file_size} is not a multiple of line stride {line_bytes}; "
            f"file may have inconsistent line endings or trailing junk"
        )
    if last is None or last > total_rows:
        last = total_rows
    if last < first:
        n_cols = nind if sample_indices is None else len(np.asarray(sample_indices))
        return np.empty((0, n_cols), dtype=float)

    n_rows = last - first + 1
    start_byte = (first - 1) * line_bytes
    expected = n_rows * line_bytes
    with open(path, "rb") as fh:
        fh.seek(start_byte)
        raw = fh.read(expected)
    buf = np.frombuffer(raw, dtype=np.uint8)
    if buf.size < expected:
        # Final row may be missing its terminator; pad so reshape works (pad is sliced off).
        buf = np.concatenate([buf, np.zeros(expected - buf.size, dtype=np.uint8)])
    grid = buf.reshape(n_rows, line_bytes)
    if sample_indices is None:
        cols = grid[:, :nind]
    else:
        cols = grid[:, np.asarray(sample_indices, dtype=np.int64)]
    digits = cols.astype(np.int16) - 48
    out = digits.astype(float)
    out[digits == 9] = np.nan
    return out


def read_eigenstrat(pref: str | Path, inds=None, pops=None, first: int = 1, last: int | None = None) -> dict:
    pref = str(pref)
    ind = read_ind(pref + ".ind")
    snp = read_snp(pref + ".snp")
    last = len(snp) if last is None else min(last, len(snp))
    indvec, _ = _match_samples(ind.iid, ind.population, inds, pops)
    keep = np.where(indvec >= 0)[0]
    geno = _read_eigenstrat_geno(pref + ".geno", len(ind), first, last, sample_indices=keep)
    return {"geno": geno, "ind": ind.loc[keep].reset_index(drop=True), "snp": snp.iloc[first - 1:last].reset_index(drop=True)}


def eigenstrat_to_afs(
    pref: str | Path,
    inds=None,
    pops=None,
    adjust_pseudohaploid=True,
    chunk_size: int = 10_000,
    verbose: bool = True,
) -> AfData:
    pref = str(pref)
    ind = read_ind(pref + ".ind")
    snp = read_snp(pref + ".snp")
    indvec, popnames = _match_samples(ind.iid, ind.population, inds, pops)
    keep_inds = np.where(indvec >= 0)[0]
    indvec_sub = indvec[keep_inds]
    nsnp = len(snp)
    _log(f"Reading EIGENSTRAT data: {nsnp} SNPs, {len(ind)} samples, {len(keep_inds)} selected samples, {len(popnames)} populations", verbose)
    ntest = _ntest(adjust_pseudohaploid, nsnp)
    if adjust_pseudohaploid:
        _log(f"Detecting pseudohaploid samples from first {ntest} SNPs", verbose)
        test_geno = _read_eigenstrat_geno(pref + ".geno", len(ind), 1, ntest, sample_indices=keep_inds)
    else:
        test_geno = np.empty((0, len(keep_inds)), dtype=float)
    ploidy = _detect_pseudohaploid(test_geno, indvec_sub, adjust_pseudohaploid) if adjust_pseudohaploid else np.full(len(indvec_sub), 2.0)
    afs = np.full((nsnp, len(popnames)), np.nan, dtype=float)
    counts = np.zeros((nsnp, len(popnames)), dtype=float)
    nchunks = math.ceil(nsnp / chunk_size) if nsnp else 0
    for chunk_i, start in enumerate(range(1, nsnp + 1, chunk_size), start=1):
        stop = min(start + chunk_size - 1, nsnp)
        _log_chunk("EIGENSTRAT", chunk_i, nchunks, start, stop, verbose)
        geno = _read_eigenstrat_geno(pref + ".geno", len(ind), start, stop, sample_indices=keep_inds)
        a, c = _geno_to_af_arrays(geno, indvec_sub, popnames, ploidy)
        afs[start - 1:stop, :] = a
        counts[start - 1:stop, :] = c
    return AfData(
        pd.DataFrame(afs, index=snp.SNP, columns=popnames),
        pd.DataFrame(counts, index=snp.SNP, columns=popnames),
        snp,
    )


def _decode_packed_ancestrymap(raw: bytes, nind: int) -> np.ndarray:
    vals = _PACKED_ANCESTRYMAP_LUT[np.frombuffer(raw, dtype=np.uint8)].reshape(-1)[:nind]
    out = vals.astype(float)
    out[out == 3] = np.nan
    return out


def _read_packed_geno(
    path: str | Path,
    nsnp: int,
    nind: int,
    first: int = 1,
    last: int | None = None,
    sample_indices: np.ndarray | None = None,
) -> np.ndarray:
    last = nsnp if last is None else min(last, nsnp)
    bytes_per_snp = _packed_record_len(nind)
    with open(path, "rb") as fh:
        fh.seek(first * bytes_per_snp)
        raw = np.frombuffer(fh.read((last - first + 1) * bytes_per_snp), dtype=np.uint8)
    if raw.size == 0:
        return np.empty((0, nind), dtype=float)
    raw = raw.reshape(-1, bytes_per_snp)
    if sample_indices is None:
        decoded = _PACKED_ANCESTRYMAP_LUT[raw].reshape(-1, bytes_per_snp * 4)[:, :nind]
    else:
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        byte_indices = sample_indices // 4
        offsets = sample_indices % 4
        decoded = _PACKED_ANCESTRYMAP_LUT[raw[:, byte_indices]][:, np.arange(len(sample_indices)), offsets]
    out = decoded.astype(float)
    out[out == 3] = np.nan
    return out


def _read_tgeno(
    path: str | Path,
    nsnp: int,
    nind: int,
    first: int = 1,
    last: int | None = None,
    sample_indices: np.ndarray | None = None,
) -> np.ndarray:
    last = nsnp if last is None else min(last, nsnp)
    if last < first:
        n_cols = nind if sample_indices is None else len(np.asarray(sample_indices))
        return np.empty((0, n_cols), dtype=float)
    header_len, bytes_per_ind = _tgeno_layout(path, nsnp, nind)
    if sample_indices is None:
        sample_indices = np.arange(nind, dtype=np.int64)
    else:
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
    out = np.empty((last - first + 1, len(sample_indices)), dtype=float)
    snp_indices = np.arange(first - 1, last, dtype=np.int64)
    byte_min = int(snp_indices[0]) // 4
    byte_max = int(snp_indices[-1]) // 4
    slice_len = byte_max - byte_min + 1
    local_byte_idx = (snp_indices // 4) - byte_min
    offsets = snp_indices % 4
    with open(path, "rb") as fh:
        for col, sample_i in enumerate(sample_indices):
            fh.seek(header_len + sample_i * bytes_per_ind + byte_min)
            raw = np.frombuffer(fh.read(slice_len), dtype=np.uint8)
            vals = _PACKED_ANCESTRYMAP_LUT[raw[local_byte_idx], offsets]
            out[:, col] = vals
    out[out == 3] = np.nan
    return out


def read_packedancestrymap(pref: str | Path, inds=None, pops=None, first: int = 1, last: int | None = None) -> dict:
    pref = str(pref)
    ind = read_ind(pref + ".ind")
    snp = read_snp(pref + ".snp")
    last = len(snp) if last is None else min(last, len(snp))
    indvec, _ = _match_samples(ind.iid, ind.population, inds, pops)
    keep = indvec >= 0
    geno = _read_packed_geno(pref + ".geno", len(snp), len(ind), first, last, np.where(keep)[0])
    return {"geno": geno, "ind": ind.loc[keep].reset_index(drop=True), "snp": snp.iloc[first - 1:last].reset_index(drop=True)}


def packedancestrymap_to_afs(pref: str | Path, inds=None, pops=None, adjust_pseudohaploid=True, chunk_size: int = 10_000, verbose: bool = True) -> AfData:
    pref = str(pref)
    ind = read_ind(pref + ".ind")
    snp = read_snp(pref + ".snp")
    indvec, popnames = _match_samples(ind.iid, ind.population, inds, pops)
    keep_inds = np.where(indvec >= 0)[0]
    indvec_sub = indvec[keep_inds]
    _log(f"Reading packed AncestryMap data: {len(snp)} SNPs, {len(ind)} samples, {len(keep_inds)} selected samples, {len(popnames)} populations", verbose)
    ntest = _ntest(adjust_pseudohaploid, len(snp))
    if adjust_pseudohaploid:
        _log(f"Detecting pseudohaploid samples from first {ntest} SNPs", verbose)
    test_geno = _read_packed_geno(pref + ".geno", len(snp), len(ind), 1, ntest, keep_inds) if adjust_pseudohaploid else np.empty((0, len(keep_inds)))
    ploidy = _detect_pseudohaploid(test_geno, indvec_sub, adjust_pseudohaploid) if adjust_pseudohaploid else np.full(len(indvec_sub), 2.0)
    afs = np.full((len(snp), len(popnames)), np.nan, dtype=float)
    counts = np.zeros((len(snp), len(popnames)), dtype=float)
    nchunks = math.ceil(len(snp) / chunk_size)
    for chunk_i, start in enumerate(range(1, len(snp) + 1, chunk_size), start=1):
        stop = min(start + chunk_size - 1, len(snp))
        _log_chunk("packed AncestryMap", chunk_i, nchunks, start, stop, verbose)
        geno = _read_packed_geno(pref + ".geno", len(snp), len(ind), start, stop, keep_inds)
        a, c = _geno_to_af_arrays(geno, indvec_sub, popnames, ploidy)
        afs[start - 1:stop, :] = a
        counts[start - 1:stop, :] = c
    return AfData(
        pd.DataFrame(afs, index=snp.SNP, columns=popnames),
        pd.DataFrame(counts, index=snp.SNP, columns=popnames),
        snp,
    )


def tgeno_to_afs(
    pref: str | Path,
    inds=None,
    pops=None,
    adjust_pseudohaploid=True,
    chunk_size: int = 10_000,
    verbose: bool = True,
    chunked: bool = False,
) -> AfData:
    pref = str(pref)
    geno_path = _tgeno_path(pref)
    ind = read_ind(pref + ".ind")
    snp = read_snp(pref + ".snp")
    indvec, popnames = _match_samples(ind.iid, ind.population, inds, pops)
    keep_inds = np.where(indvec >= 0)[0]
    indvec_sub = indvec[keep_inds]
    nsnp = len(snp)
    npop = len(popnames)
    _log(f"Reading TGENO data: {nsnp} SNPs, {len(ind)} samples, {len(keep_inds)} selected samples, {npop} populations", verbose)
    ntest = _ntest(adjust_pseudohaploid, nsnp)
    if adjust_pseudohaploid and ntest > 0:
        _log(f"Detecting pseudohaploid samples from first {ntest} SNPs", verbose)
        test_geno = _read_tgeno(geno_path, nsnp, len(ind), 1, ntest, keep_inds)
        ploidy = _detect_pseudohaploid(test_geno, indvec_sub, adjust_pseudohaploid)
    else:
        ploidy = np.full(len(indvec_sub), 2.0)

    counts = np.zeros((nsnp, npop), dtype=float)
    if chunked:
        afs = np.full((nsnp, npop), np.nan, dtype=float)
        nchunks = math.ceil(nsnp / chunk_size) if nsnp else 0
        for chunk_i, start in enumerate(range(1, nsnp + 1, chunk_size), start=1):
            stop = min(start + chunk_size - 1, nsnp)
            _log_chunk("TGENO", chunk_i, nchunks, start, stop, verbose)
            geno = _read_tgeno(geno_path, nsnp, len(ind), start, stop, keep_inds)
            a, c = _geno_to_af_arrays(geno, indvec_sub, popnames, ploidy)
            afs[start - 1:stop, :] = a
            counts[start - 1:stop, :] = c
    else:
        header_len, bytes_per_ind = _tgeno_layout(geno_path, nsnp, len(ind))
        snp_byte_idx = (np.arange(nsnp, dtype=np.int64) // 4)
        snp_off = (np.arange(nsnp, dtype=np.int64) % 4)
        alt_sum = np.zeros((nsnp, npop), dtype=float)
        nkept = len(keep_inds)
        log_step = max(1, nkept // 20)
        with open(geno_path, "rb") as fh:
            for col, sample_i in enumerate(keep_inds):
                if verbose and (col == 0 or col == nkept - 1 or (col + 1) % log_step == 0):
                    end = "\n" if col == nkept - 1 else ""
                    print(f"\rReading TGENO sample {col + 1}/{nkept}      ", end=end, flush=True)
                fh.seek(header_len + int(sample_i) * bytes_per_ind)
                raw = np.frombuffer(fh.read(bytes_per_ind), dtype=np.uint8)
                vals = _PACKED_ANCESTRYMAP_LUT[raw[snp_byte_idx], snp_off].astype(float)
                vals[vals == 3] = np.nan
                p = ploidy[col]
                pop_i = indvec_sub[col]
                present = np.isfinite(vals)
                counts[:, pop_i] += present.astype(float) * p
                alt_sum[:, pop_i] += np.where(present, vals / (3.0 - p), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            afs = alt_sum / counts
        afs[counts == 0] = np.nan
    return AfData(
        pd.DataFrame(afs, index=snp.SNP, columns=popnames),
        pd.DataFrame(counts, index=snp.SNP, columns=popnames),
        snp,
    )


def _read_plink_bed(
    path: str | Path,
    nsnp: int,
    nind: int,
    first: int = 1,
    last: int | None = None,
    sample_indices: np.ndarray | None = None,
) -> np.ndarray:
    last = nsnp if last is None else min(last, nsnp)
    bytes_per_snp = math.ceil(nind / 4)
    with open(path, "rb") as fh:
        magic = fh.read(3)
        if magic[:2] != b"\x6c\x1b" or magic[2] != 1:
            raise ValueError("Only SNP-major PLINK .bed files are supported")
        fh.seek(3 + (first - 1) * bytes_per_snp)
        raw = np.frombuffer(fh.read((last - first + 1) * bytes_per_snp), dtype=np.uint8)
    raw = raw.reshape(-1, bytes_per_snp)
    if sample_indices is None:
        decoded = _PLINK_LUT[raw].reshape(-1, bytes_per_snp * 4)[:, :nind]
    else:
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        byte_indices = sample_indices // 4
        offsets = sample_indices % 4
        decoded = _PLINK_LUT[raw[:, byte_indices]][:, np.arange(len(sample_indices)), offsets]
    out = decoded.astype(float)
    out[out == 3] = np.nan
    return out


def read_plink(pref: str | Path, inds=None, pops=None) -> dict:
    pref = str(pref)
    fam = _read_table(pref + ".fam", ["population", "iid", "p1", "p2", "sex", "pheno"])
    bim = read_snp(pref + ".bim", plink=True)
    indvec, _ = _match_samples(fam.iid, fam.population, inds, pops)
    keep = indvec >= 0
    geno = _read_plink_bed(pref + ".bed", len(bim), len(fam), sample_indices=np.where(keep)[0])
    return {"geno": geno, "fam": fam.loc[keep].reset_index(drop=True), "bim": bim}


def plink_to_afs(pref: str | Path, inds=None, pops=None, adjust_pseudohaploid=True, chunk_size: int = 10_000, verbose: bool = True) -> AfData:
    pref = str(pref)
    fam = _read_table(pref + ".fam", ["population", "iid", "p1", "p2", "sex", "pheno"])
    bim = read_snp(pref + ".bim", plink=True)
    indvec, popnames = _match_samples(fam.iid, fam.population, inds, pops)
    keep_inds = np.where(indvec >= 0)[0]
    indvec_sub = indvec[keep_inds]
    _log(f"Reading PLINK data: {len(bim)} SNPs, {len(fam)} samples, {len(keep_inds)} selected samples, {len(popnames)} populations", verbose)
    ntest = _ntest(adjust_pseudohaploid, len(bim))
    if adjust_pseudohaploid:
        _log(f"Detecting pseudohaploid samples from first {ntest} SNPs", verbose)
    test_geno = _read_plink_bed(pref + ".bed", len(bim), len(fam), 1, ntest, keep_inds) if adjust_pseudohaploid else np.empty((0, len(keep_inds)))
    ploidy = _detect_pseudohaploid(test_geno, indvec_sub, adjust_pseudohaploid) if adjust_pseudohaploid else np.full(len(indvec_sub), 2.0)
    afs = np.full((len(bim), len(popnames)), np.nan, dtype=float)
    counts = np.zeros((len(bim), len(popnames)), dtype=float)
    nchunks = math.ceil(len(bim) / chunk_size)
    for chunk_i, start in enumerate(range(1, len(bim) + 1, chunk_size), start=1):
        stop = min(start + chunk_size - 1, len(bim))
        _log_chunk("PLINK", chunk_i, nchunks, start, stop, verbose)
        geno = _read_plink_bed(pref + ".bed", len(bim), len(fam), start, stop, keep_inds)
        a, c = _geno_to_af_arrays(geno, indvec_sub, popnames, ploidy)
        afs[start - 1:stop, :] = a
        counts[start - 1:stop, :] = c
    return AfData(
        pd.DataFrame(afs, index=bim.SNP, columns=popnames),
        pd.DataFrame(counts, index=bim.SNP, columns=popnames),
        bim,
    )


def anygeno_to_afs(
    pref: str | Path,
    inds=None,
    pops=None,
    format: str | None = None,
    adjust_pseudohaploid=True,
    chunk_size: int = 10_000,
    verbose: bool = True,
    tgeno_chunked: bool = False,
) -> AfData:
    pref = str(pref)
    if format is None:
        if all(Path(pref + ext).exists() for ext in (".bed", ".bim", ".fam")):
            format = "plink"
        elif all(Path(pref + ext).exists() for ext in (".geno", ".snp", ".ind")):
            ind = read_ind(pref + ".ind")
            snp = read_snp(pref + ".snp")
            format = detect_geno_format(pref + ".geno", nind=len(ind), nsnp=len(snp))
        elif Path(pref + ".tgeno").exists() and all(Path(pref + ext).exists() for ext in (".snp", ".ind")):
            ind = read_ind(pref + ".ind")
            snp = read_snp(pref + ".snp")
            format = detect_geno_format(pref + ".tgeno", nind=len(ind), nsnp=len(snp))
            if format != "tgeno":
                raise ValueError(f"{pref}.tgeno is not a valid TGENO file")
        else:
            raise FileNotFoundError("Genotype files not found")
    format = format.lower()
    if format == "plink":
        return plink_to_afs(pref, inds, pops, adjust_pseudohaploid, chunk_size=chunk_size, verbose=verbose)
    if format == "packedancestrymap":
        return packedancestrymap_to_afs(pref, inds, pops, adjust_pseudohaploid, chunk_size=chunk_size, verbose=verbose)
    if format == "tgeno":
        return tgeno_to_afs(pref, inds, pops, adjust_pseudohaploid, chunk_size=chunk_size, verbose=verbose, chunked=tgeno_chunked)
    if format == "eigenstrat":
        return eigenstrat_to_afs(pref, inds, pops, adjust_pseudohaploid, chunk_size=chunk_size, verbose=verbose)
    raise ValueError("format must be 'plink', 'eigenstrat', 'packedancestrymap', or 'tgeno'")


def iter_geno_to_afs(
    pref: str | Path,
    inds=None,
    pops=None,
    format: str | None = None,
    adjust_pseudohaploid=True,
    chunk_size: int = 10_000,
    verbose: bool = True,
) -> Iterator[AfData]:
    """Yield population allele frequencies in bounded SNP chunks.

    Unlike :func:`anygeno_to_afs`, this does not materialize full SNP-by-
    population matrices. All supported formats use the same sample matching,
    pseudohaploid detection, and allele-count conversion as their full readers.
    """
    pref = str(pref)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if format is None:
        if all(Path(pref + ext).exists() for ext in (".bed", ".bim", ".fam")):
            format = "plink"
        elif all(Path(pref + ext).exists() for ext in (".geno", ".snp", ".ind")):
            ind0 = read_ind(pref + ".ind")
            snp0 = read_snp(pref + ".snp")
            format = detect_geno_format(pref + ".geno", nind=len(ind0), nsnp=len(snp0))
        elif Path(pref + ".tgeno").exists() and all(Path(pref + ext).exists() for ext in (".snp", ".ind")):
            ind0 = read_ind(pref + ".ind")
            snp0 = read_snp(pref + ".snp")
            format = detect_geno_format(pref + ".tgeno", nind=len(ind0), nsnp=len(snp0))
        else:
            raise FileNotFoundError("Genotype files not found")
    format = format.lower()

    if format == "plink":
        individuals = _read_table(pref + ".fam", ["population", "iid", "p1", "p2", "sex", "pheno"])
        snp = read_snp(pref + ".bim", plink=True)
        geno_path = pref + ".bed"

        def read_range(first, last, keep):
            return _read_plink_bed(geno_path, len(snp), len(individuals), first, last, keep)

    elif format in {"eigenstrat", "packedancestrymap", "tgeno"}:
        individuals = read_ind(pref + ".ind")
        snp = read_snp(pref + ".snp")
        geno_path = _tgeno_path(pref) if format == "tgeno" else pref + ".geno"
        if format == "eigenstrat":
            def read_range(first, last, keep):
                return _read_eigenstrat_geno(geno_path, len(individuals), first, last, keep)
        elif format == "packedancestrymap":
            def read_range(first, last, keep):
                return _read_packed_geno(geno_path, len(snp), len(individuals), first, last, keep)
        else:
            def read_range(first, last, keep):
                return _read_tgeno(geno_path, len(snp), len(individuals), first, last, keep)
    else:
        raise ValueError("format must be 'plink', 'eigenstrat', 'packedancestrymap', or 'tgeno'")

    indvec, popnames = _match_samples(individuals.iid, individuals.population, inds, pops)
    keep_inds = np.where(indvec >= 0)[0]
    indvec_sub = indvec[keep_inds]
    nsnp = len(snp)
    _log(
        f"Streaming {format} data: {nsnp} SNPs, {len(individuals)} samples, "
        f"{len(keep_inds)} selected samples, {len(popnames)} populations",
        verbose,
    )
    ntest = _ntest(adjust_pseudohaploid, nsnp)
    if adjust_pseudohaploid and ntest > 0:
        _log(f"Detecting pseudohaploid samples from first {ntest} SNPs", verbose)
        test_geno = read_range(1, ntest, keep_inds)
        ploidy = _detect_pseudohaploid(test_geno, indvec_sub, adjust_pseudohaploid)
    else:
        ploidy = np.full(len(indvec_sub), 2.0)

    nchunks = math.ceil(nsnp / chunk_size) if nsnp else 0
    for chunk_i, start in enumerate(range(1, nsnp + 1, chunk_size), start=1):
        stop = min(start + chunk_size - 1, nsnp)
        _log_chunk(format, chunk_i, nchunks, start, stop, verbose)
        geno = read_range(start, stop, keep_inds)
        afs, counts = _geno_to_af_arrays(geno, indvec_sub, popnames, ploidy)
        chunk_snp = snp.iloc[start - 1:stop].reset_index(drop=True)
        yield AfData(
            pd.DataFrame(afs, index=chunk_snp.SNP, columns=popnames),
            pd.DataFrame(counts, index=chunk_snp.SNP, columns=popnames),
            chunk_snp,
        )


def weighted_row_means(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(values * weights, axis=1) / np.nansum(np.where(np.isfinite(values), weights, 0), axis=1)


def is_polymorphic(afs: pd.DataFrame) -> np.ndarray:
    """True where a row has at least two finite values and they are not all equal."""
    arr = afs.to_numpy(float)
    finite_count = np.isfinite(arr).sum(axis=1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        rng = np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)
    return (finite_count > 1) & np.isfinite(rng) & (rng > 0)


def discard_from_aftable(
    afdat: AfData,
    maxmiss: float = 0,
    minmaf: float = 0,
    maxmaf: float = 0.5,
    minac2: bool | int = False,
    outpop: str | None = None,
    auto_only: bool = True,
    poly_only: bool = False,
    transitions: bool = True,
    transversions: bool = True,
    keepsnps: Iterable[str] | None = None,
) -> AfData:
    snp = afdat.snpfile.copy().reset_index(drop=True)
    afs = afdat.afs.reset_index(drop=True)
    counts = afdat.counts.reset_index(drop=True)
    if keepsnps is not None:
        keep = snp["SNP"].isin(set(keepsnps)).to_numpy()
    else:
        miss = (counts.to_numpy(float) == 0).mean(axis=1) if maxmiss < 1 else np.zeros(len(snp))
        maf = np.full(len(snp), 0.2)
        if minmaf > 0 or maxmaf < 0.5:
            af = weighted_row_means(afs.to_numpy(float), counts.to_numpy(float))
            maf = np.minimum(af, 1 - af)
        minac = np.full(len(snp), 2.0)
        if minac2:
            cm = counts.to_numpy(float)
            popmask = np.ones(cm.shape[1], dtype=bool)
            if minac2 == 2:
                popmask = np.nanmax(cm, axis=0) > 1
            minac = np.nanmin(cm[:, popmask], axis=1)
        poly = is_polymorphic(afs) if poly_only else np.ones(len(snp), dtype=bool)
        outgroupaf = afs[outpop].to_numpy(float) if outpop is not None else np.full(len(snp), 0.5)
        chrom = pd.to_numeric(snp["CHR"].astype(str).str.replace(r"[A-Za-z]+", "", regex=True), errors="coerce")
        chrom_num = chrom.to_numpy()
        if auto_only:
            unparsable = chrom.isna().to_numpy()
            if unparsable.any():
                # Match admixtools: warn and drop, rather than refuse to load.
                bad = sorted(set(snp.loc[unparsable, "CHR"].astype(str)))[:6]
                warnings.warn(
                    f"Dropping {int(unparsable.sum())} SNPs on non-numeric chromosomes "
                    f"(e.g. {bad}). Set auto_only=False to keep them.",
                    stacklevel=2,
                )
            chrom_num = np.where(unparsable, 99, chrom_num)  # 99 fails the CHR <= 22 filter
        mut = np.full(len(snp), "", dtype=object)
        if not transitions or not transversions:
            a1 = snp["A1"].astype(str).str.upper()
            a2 = snp["A2"].astype(str).str.upper()
            pairs = ["".join(sorted(x)) for x in zip(a1, a2)]
            mut = np.array(["transition" if p in {"AG", "CT"} else "transversion" if p in {"AC", "AT", "CG", "GT"} else "" for p in pairs])
        keep = (
            (miss <= maxmiss)
            & (maf >= minmaf)
            & (maf <= maxmaf)
            & (outgroupaf > 0)
            & (outgroupaf < 1)
            & ((not minac2) | (minac > 1))
            & ((not auto_only) | (chrom_num <= 22))
            & ((not poly_only) | poly)
            & (transitions | (mut != "transition"))
            & (transversions | (mut != "transversion"))
        )
    if not np.any(keep):
        raise ValueError("No SNPs remain after filtering")
    snp = snp.loc[keep].reset_index(drop=True)
    afs = afs.loc[keep].set_index(snp["SNP"])
    counts = counts.loc[keep].set_index(snp["SNP"])
    return AfData(afs, counts, snp)


def get_block_lengths(dat: pd.DataFrame, blgsize: float = 0.05) -> np.ndarray:
    if len(dat) == 0:
        return np.array([], dtype=int)
    distcol = "cm"
    if blgsize >= 100:
        distcol = "POS"
    elif dat["cm"].nunique(dropna=True) < 2:
        if dat["POS"].nunique(dropna=True) > 2:
            blgsize = 2_000_000
            distcol = "POS"
            warnings.warn(f"No genetic linkage map found; defining blocks by {blgsize:g} bp")
        else:
            warnings.warn("No usable map or base positions found; each chromosome is one block")
    chrom = pd.to_numeric(dat["CHR"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    if chrom.isna().any():
        chrom = pd.factorize(dat["CHR"])[0] + 1
    else:
        chrom = chrom.to_numpy()
    pos = dat[distcol].to_numpy(float)
    lengths = []
    last_chr = None
    first_pos = -1e20
    size = 0
    for c, p in zip(chrom, pos):
        if c != last_chr or p - first_pos >= blgsize:
            if size:
                lengths.append(size)
            last_chr = c
            first_pos = p
            size = 0
        size += 1
    if size:
        lengths.append(size)
    return np.asarray(lengths, dtype=int)
