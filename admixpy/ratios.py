from dataclasses import dataclass
from statistics import NormalDist
import warnings

import numpy as np
import pandas as pd

from .fstats import (
    F4BlockCache, _default_genotype_allsnps, _set_direct_resampling,
    _validate_resampling, f4_stats,
)


_POP_COLUMNS = ["target", "source1", "source2", "reference", "outgroup"]


@dataclass
class F4RatioResult:
    summary: pd.DataFrame
    components: pd.DataFrame
    settings: dict
    blocks: pd.DataFrame | None = None

    def __repr__(self) -> str:
        columns = [
            "target", "source1", "source2", "reference", "outgroup",
            "est", "se", "n", "status",
        ]
        return self.summary[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.3g}",
        )


def _ratio_models(target, source1, source2, reference, outgroup, models):
    if models is not None:
        if any(value is not None for value in (target, source1, source2, reference, outgroup)):
            raise ValueError("Supply either models or individual population arguments")
        frame = pd.DataFrame(models).copy()
        missing = [col for col in _POP_COLUMNS if col not in frame]
        if missing:
            raise ValueError(f"Ratio models are missing columns: {missing}")
        frame = frame[_POP_COLUMNS].reset_index(drop=True)
    else:
        values = dict(
            target=target,
            source1=source1,
            source2=source2,
            reference=reference,
            outgroup=outgroup,
        )
        missing = [name for name, value in values.items() if value is None]
        if missing:
            noun = "argument" if len(missing) == 1 else "arguments"
            raise ValueError(
                f"f4_ratio() missing required population {noun}: {', '.join(missing)}"
            )
        refs = [reference] if isinstance(reference, str) else list(reference)
        if not refs:
            raise ValueError("reference must contain at least one population")
        frame = pd.DataFrame(
            [[target, source1, source2, ref, outgroup] for ref in refs],
            columns=_POP_COLUMNS,
        )
    if frame.empty:
        raise ValueError("At least one ratio model/reference is required")
    for i, values in enumerate(frame.itertuples(index=False, name=None), 1):
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"Model {i}: all five population names must be nonempty strings")
        if len(set(values)) != 5:
            raise ValueError(f"Model {i}: target, sources, reference, and outgroup must be distinct")
    frame.insert(0, "model", np.arange(1, len(frame) + 1))
    return frame


def _ratio_uncertainty(full, loo, weights):
    """Return the unequal-delete-block jackknife SE for a ratio."""
    if len(weights) < 2 or not np.isfinite(full) or not np.isfinite(loo).all():
        return float("nan")
    h = weights.sum() / weights
    center = np.sum(full - loo) + np.average(loo, weights=weights)
    pseudo = h * full - (h - 1) * loo
    return float(np.sqrt(np.mean((pseudo - center) ** 2 / (h - 1))))


def f4_ratio(
    data,
    target: str | None = None,
    source1: str | None = None,
    source2: str | None = None,
    reference=None,
    outgroup: str | None = None,
    *,
    models=None,
    allsnps: bool = False,
    resampling: str = "pairwise_counts",
    confidence: float = 0.95,
    denominator_z_min: float = 3.0,
    return_blocks: bool = False,
    verbose: bool = True,
    **kwargs,
) -> F4RatioResult:
    """Estimate source1 ancestry as
    f4(target, source2; reference, outgroup) /
    f4(source1, source2; reference, outgroup).

    Assumes a valid two-source topology and does not test model fit.
    """
    model_table = _ratio_models(target, source1, source2, reference, outgroup, models)
    resampling = _validate_resampling(resampling)
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if not np.isfinite(denominator_z_min) or denominator_z_min <= 0:
        raise ValueError("denominator_z_min must be finite and positive")
    reserved = {"keep_blocks", "keep_loo", "covariance", "unique_only", "comb"} & kwargs.keys()
    if reserved:
        raise ValueError(f"f4_ratio manages these f4 options internally: {sorted(reserved)}")
    if isinstance(data, F4BlockCache):
        raise ValueError("f4_ratio requires genotype or precomputed f2 input, not F4BlockCache")
    direct = _default_genotype_allsnps(data)
    if allsnps and not direct:
        raise ValueError("allsnps=True requires direct genotype input")

    # Identical models share a panel and both components. With allsnps or f2
    # input, contrasts can additionally be shared across different models.
    unique_models = model_table.drop_duplicates(_POP_COLUMNS)
    combo_rows, lookup, model_indices = [], {}, {}
    for row in unique_models.itertuples(index=False):
        indices = []
        panel = row.model if direct and not allsnps else None
        for first in (row.target, row.source1):
            pops = (first, row.source2, row.reference, row.outgroup)
            key = (panel, *pops)
            if key not in lookup:
                lookup[key] = len(combo_rows)
                entry = dict(zip(("pop1", "pop2", "pop3", "pop4"), pops))
                if panel is not None:
                    entry["model"] = panel
                combo_rows.append(entry)
            indices.append(lookup[key])
        model_indices[tuple(getattr(row, c) for c in _POP_COLUMNS)] = indices

    stats = f4_stats(
        data, pd.DataFrame(combo_rows), unique_only=False,
        allsnps=allsnps, resampling="pairwise_counts" if direct else resampling,
        covariance=False, keep_blocks=True, keep_loo=True, verbose=verbose, **kwargs,
    )
    counts = getattr(stats, "snp_counts", None)
    nominal = np.asarray(getattr(stats, "nominal_block_lengths", stats.block_lengths), float)
    if direct and resampling == "nominal_blocks":
        stats = _set_direct_resampling(stats, resampling, covariance=False)
    component_se = stats.se
    panel_name = "shared" if direct and not allsnps else "per_statistic" if direct else "cached_pairwise"
    critical = NormalDist().inv_cdf(0.5 + confidence / 2)
    summaries, components, block_tables = [], [], []
    for row in model_table.to_dict("records"):
        ni, di = model_indices[tuple(row[c] for c in _POP_COLUMNS)]
        num, den = stats.est[[ni, di]]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            est = float(num / den) if den != 0 else float("nan")
            loo = stats.loo[ni] / stats.loo[di]
        shared = direct and not allsnps
        if shared:
            if not np.array_equal(counts[ni], counts[di]):
                raise ValueError("Ratio components do not have matching shared-panel SNP counts")
            active = counts[ni] > 0
        else:
            contributes = getattr(stats, "contributes", None)
            active = (contributes[ni] | contributes[di]) if contributes is not None else (
                np.isfinite(stats.blocks[ni]) | np.isfinite(stats.blocks[di])
            )
        weights = counts[ni] if shared and resampling == "pairwise_counts" else nominal
        active &= np.isfinite(weights) & (weights > 0)
        se = _ratio_uncertainty(est, loo[active], weights[active])
        den_se = component_se[di]
        with np.errstate(divide="ignore", invalid="ignore"):
            den_z = float(den / den_se)
        flags = []
        if not np.isfinite(est):
            flags.append("undefined")
        if np.isfinite(den) and den != 0:
            if np.isnan(den_z) or abs(den_z) < denominator_z_min:
                flags.append("weak_denominator")
            den_loo = stats.loo[di, active]
            if np.any(np.isfinite(den_loo) & (np.sign(den_loo) != np.sign(den))):
                flags.append("unstable_denominator")
        if active.sum() < 2:
            flags.append("insufficient_blocks")
        elif not np.isfinite(se):
            flags.append("undefined_jackknife")
        reliable = not flags
        summaries.append({
            **row, "est": est, "se": se,
            "ci_low": est - critical * se if reliable else np.nan,
            "ci_high": est + critical * se if reliable else np.nan,
            "n": int(counts[ni].sum()) if shared else pd.NA,
            "n_blocks": int(active.sum()), "denominator_z": den_z,
            "snp_panel": panel_name, "status": ";".join(flags) if flags else "ok",
        })
        for name, i in (("numerator", ni), ("denominator", di)):
            with np.errstate(divide="ignore", invalid="ignore"):
                z = float(stats.est[i] / component_se[i])
            components.append({
                "model": row["model"], "component": name,
                **stats.rows.iloc[i][["pop1", "pop2", "pop3", "pop4"]].to_dict(),
                "est": stats.est[i], "se": component_se[i], "z": z,
                "n": int(counts[i].sum()) if counts is not None else pd.NA,
            })
        if return_blocks:
            block_tables.append(pd.DataFrame({
                "model": row["model"], "block": np.arange(1, len(nominal) + 1),
                "block_length": nominal, "jackknife_weight": weights,
                "contributes": active, "numerator": stats.blocks[ni],
                "denominator": stats.blocks[di], "numerator_loo": stats.loo[ni],
                "denominator_loo": stats.loo[di], "ratio_loo": loo,
                "numerator_n": counts[ni] if counts is not None else pd.NA,
                "denominator_n": counts[di] if counts is not None else pd.NA,
            }))
    summary = pd.DataFrame(summaries).astype({"n": "Int64"})
    component_table = pd.DataFrame(components).astype({"n": "Int64"})
    bad = summary.loc[summary.status != "ok", ["model", "status"]]
    if not bad.empty:
        details = ", ".join(f"{r.model}: {r.status}" for r in bad.head(8).itertuples())
        warnings.warn(
            f"f4_ratio numerical diagnostics ({len(bad)} model(s)): {details}. "
            "Confidence intervals are suppressed; inspect components and block diagnostics.",
            RuntimeWarning, stacklevel=2,
        )
    from . import __version__
    settings = {
        "version": __version__, "allsnps": allsnps, "resampling": resampling,
        "confidence": confidence, "denominator_z_min": denominator_z_min,
        "snp_panel": panel_name,
        "jackknife_weights": "shared_snp_counts" if direct and not allsnps and resampling == "pairwise_counts" else "nominal_blocks",
        "input_options": dict(kwargs), "blgsize": kwargs.get("blgsize", 0.05) if direct else None,
        "apply_corr": kwargs.get("apply_corr", True) if direct else None,
    }
    return F4RatioResult(
        summary, component_table, settings,
        pd.concat(block_tables, ignore_index=True) if return_blocks else None,
    )
