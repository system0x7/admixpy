import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import admixpy
from admixpy.fstats import BlockStats, F4BlockCache, afs_to_f2_blocks
from admixpy.genotypes import AfData


POPS = dict(target="T", source1="A", source2="B", reference="R", outgroup="O")


def allele_data(alpha=0.25, noisy=False):
    rng = np.random.default_rng(24)
    n = 24
    a = rng.uniform(0.6, 0.8, n)
    b = rng.uniform(0.1, 0.3, n)
    t = alpha * a + (1 - alpha) * b
    if noisy:
        t += np.repeat([0.03, -0.02, 0.04, -0.01, 0.02, -0.04], 4)
    af = pd.DataFrame(dict(T=t, A=a, B=b, R=rng.uniform(0.6, 0.8, n),
                           O=rng.uniform(0.1, 0.2, n), R2=rng.uniform(0.4, 0.8, n)))
    snp = pd.DataFrame(dict(SNP=[f"s{i}" for i in range(n)], CHR=np.repeat(range(1, 7), 4),
                            cm=np.tile([0, .01, .02, .03], 6), POS=np.arange(n) * 100,
                            A1="A", A2="G"))
    af.index = snp.SNP
    return AfData(af, pd.DataFrame(20., index=af.index, columns=af.columns), snp)


def direct(af, **kwargs):
    options = {**POPS, "verbose": False, **kwargs}
    with patch("admixpy.fstats.anygeno_to_afs", return_value=af):
        return admixpy.f4_ratio("unused", **options)


class F4RatioTests(unittest.TestCase):
    def test_recovers_known_mixture_and_complement_without_clipping(self):
        af = allele_data()
        result = direct(af, return_blocks=True)
        row = result.summary.iloc[0]
        self.assertIsInstance(result, admixpy.F4RatioResult)
        self.assertAlmostEqual(row.est, .25)
        self.assertAlmostEqual(row.se, 0, places=12)
        self.assertEqual(row.status, "ok")
        self.assertEqual(row.n, 24)
        self.assertEqual(row.n_blocks, 6)
        np.testing.assert_allclose(result.blocks.ratio_loo, .25)
        self.assertAlmostEqual(direct(af, source1="B", source2="A").summary.est[0], .75)
        self.assertAlmostEqual(direct(allele_data(alpha=1.2)).summary.est[0], 1.2)

    def test_equal_block_error_matches_manual_delete_block_ratios(self):
        af = allele_data(noisy=True)
        result = direct(af)
        x = af.afs
        numerator = (x["T"] - x.B) * (x.R - x.O)
        denominator = (x.A - x.B) * (x.R - x.O)
        loo = []
        for block in range(6):
            keep = np.arange(24) // 4 != block
            loo.append(numerator[keep].mean() / denominator[keep].mean())
        loo = np.asarray(loo)
        expected_se = np.sqrt(5 / 6 * ((loo - loo.mean()) ** 2).sum())
        self.assertAlmostEqual(result.summary.est[0], numerator.mean() / denominator.mean())
        self.assertAlmostEqual(result.summary.se[0], expected_se)
        self.assertIsNone(result.blocks)
        self.assertGreater(result.summary.ci_high[0], result.summary.est[0])

    def test_shared_panel_unequal_count_jackknife_and_model_scope(self):
        af = allele_data(noisy=True)
        # Different omissions per model: R's absence must not remove R2's SNPs.
        af.afs.loc[af.afs.index[[0, 1, 4, 12]], "R"] = np.nan
        with patch("admixpy.fstats.anygeno_to_afs", return_value=af) as reader:
            result = admixpy.f4_ratio("unused", **{**POPS, "reference": ["R", "R2"]},
                                      verbose=False, return_blocks=True)
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(result.summary.n.tolist(), [20, 24])
        for ref in ["R", "R2"]:
            one = direct(af, reference=ref, return_blocks=True)
            many = result.summary[result.summary.reference == ref].iloc[0]
            self.assertAlmostEqual(one.summary.est[0], many.est)
            self.assertAlmostEqual(one.summary.se[0], many.se)
        x = af.afs.dropna(subset=list(POPS.values()))
        u, v = (x["T"] - x.B) * (x.R - x.O), (x.A - x.B) * (x.R - x.O)
        full = u.sum() / v.sum()
        weights, reps = [], []
        for chrom in range(1, 7):
            take = af.snpfile.set_index("SNP").loc[x.index, "CHR"] == chrom
            weights.append(take.sum())
            reps.append(u[~take].sum() / v[~take].sum())
        w, reps = np.asarray(weights), np.asarray(reps)
        h = w.sum() / w
        pseudo = h * full - (h - 1) * reps
        center = full + np.sum((1 - w / w.sum()) * (full - reps))
        se = np.sqrt(np.mean((pseudo - center) ** 2 / (h - 1)))
        self.assertAlmostEqual(result.summary.est[0], full)
        self.assertAlmostEqual(result.summary.se[0], se)

    def test_allsnps_preserves_component_counts_without_claiming_shared_n(self):
        af = allele_data(noisy=True)
        af.afs.loc[af.afs.index[:3], "T"] = np.nan
        result = direct(af, allsnps=True, return_blocks=True)
        self.assertTrue(pd.isna(result.summary.n[0]))
        self.assertEqual(result.components.n.tolist(), [21, 24])
        self.assertEqual(result.settings["jackknife_weights"], "nominal_blocks")
        self.assertEqual(result.summary.snp_panel[0], "per_statistic")
        shared = direct(af)
        self.assertEqual(shared.components.n.tolist(), [21, 21])

    def test_nominal_resampling_retains_direct_snp_counts(self):
        af = allele_data(noisy=True)
        af.afs.loc[af.afs.index[:3], "T"] = np.nan
        result = direct(af, resampling="nominal_blocks", return_blocks=True)
        self.assertEqual(result.summary.n[0], 21)
        self.assertEqual(result.components.n.tolist(), [21, 21])
        self.assertEqual(result.settings["jackknife_weights"], "nominal_blocks")

    def test_complete_f2_cache_matches_direct_and_round_trip(self):
        af = allele_data(noisy=True)
        blocks = afs_to_f2_blocks(af, poly_only=False, verbose=False)["f2_blocks"]
        raw = direct(af)
        for resampling in ["pairwise_counts", "nominal_blocks"]:
            cached = admixpy.f4_ratio(blocks, **POPS, resampling=resampling, verbose=False)
            self.assertAlmostEqual(cached.summary.est[0], raw.summary.est[0])
            self.assertAlmostEqual(cached.summary.se[0], raw.summary.se[0])
            self.assertTrue(pd.isna(cached.summary.n[0]))
            self.assertEqual(cached.summary.snp_panel[0], "cached_pairwise")
        with tempfile.TemporaryDirectory() as tmp:
            admixpy.write_f2(blocks, tmp)
            disk = admixpy.f4_ratio(tmp, **POPS, verbose=False)
        self.assertAlmostEqual(disk.summary.est[0], raw.summary.est[0])

    def test_model_table_and_duplicates_preserve_order(self):
        models = pd.DataFrame([POPS, {**POPS, "reference": "R2"}, POPS])
        af = allele_data()
        with patch("admixpy.fstats.anygeno_to_afs", return_value=af):
            result = admixpy.f4_ratio("unused", models=models, verbose=False)
        self.assertEqual(result.summary.model.tolist(), [1, 2, 3])
        self.assertEqual(result.summary.reference.tolist(), ["R", "R2", "R"])
        np.testing.assert_allclose(result.summary.est, .25)
        self.assertEqual(result.components.model.tolist(), [1, 1, 2, 2, 3, 3])

    def test_weak_and_undefined_denominators_have_no_confidence_interval(self):
        af = allele_data()
        af.afs["R"] = af.afs.O
        with self.assertWarnsRegex(RuntimeWarning, "undefined"):
            zero = direct(af)
        self.assertTrue(np.isnan(zero.summary.est[0]))
        self.assertTrue(np.isnan(zero.summary.ci_low[0]))
        with self.assertWarnsRegex(RuntimeWarning, "weak_denominator"):
            weak = direct(allele_data(noisy=True), denominator_z_min=1e9)
        self.assertTrue(np.isfinite(weak.summary.est[0]))
        self.assertTrue(np.isnan(weak.summary.ci_low[0]))

    def test_zero_loo_denominator_is_not_silently_dropped(self):
        af = allele_data()
        af.afs["R"] = af.afs.O
        af.afs.loc[af.afs.index[:4], "R"] += .5
        with self.assertWarns(RuntimeWarning):
            result = direct(af, return_blocks=True)
        self.assertIn("unstable_denominator", result.summary.status[0])
        self.assertIn("undefined_jackknife", result.summary.status[0])
        self.assertTrue(np.isnan(result.summary.se[0]))

    def test_one_block_retains_point_estimate(self):
        af = allele_data()
        af.snpfile["CHR"] = 1
        af.snpfile["cm"] = np.arange(24) * .001
        with self.assertWarnsRegex(RuntimeWarning, "insufficient_blocks"):
            result = direct(af)
        self.assertAlmostEqual(result.summary.est[0], .25)
        self.assertTrue(np.isnan(result.summary.se[0]))

    def test_denominator_sign_changes_are_flagged_even_for_constant_ratio(self):
        af = allele_data()
        d = np.repeat([.2, -.1, .02, .02, .02, .02], 4)
        af.afs["R"] = af.afs.O + d / (af.afs.A - af.afs.B)
        with self.assertWarnsRegex(RuntimeWarning, "unstable_denominator"):
            result = direct(af)
        self.assertAlmostEqual(result.summary.est[0], .25)
        self.assertTrue(np.isnan(result.summary.ci_low[0]))

    def test_cached_missing_pairs_delete_same_physical_block(self):
        af = allele_data(noisy=True)
        af.afs.loc[af.afs.index[:3], "A"] = np.nan
        af.afs.loc[af.afs.index[8:12], "R"] = np.nan
        blocks = afs_to_f2_blocks(af, poly_only=False, apply_corr=False, verbose=False)["f2_blocks"]
        result = admixpy.f4_ratio(blocks, **POPS, verbose=False, return_blocks=True)
        # Independently pool each f2 pair using its own actual observations.
        def f4(first, keep):
            total = 0.
            for coefficient, p1, p2 in [(0.5, first, "O"), (0.5, "B", "R"),
                                        (-0.5, first, "R"), (-0.5, "B", "O")]:
                values, counts = blocks.pair(p1, p2)[keep], blocks.pair_counts(p1, p2)[keep]
                valid = np.isfinite(values) & (counts > 0)
                total += coefficient * np.average(values[valid], weights=counts[valid])
            return total
        full = f4("T", np.ones(6, bool)) / f4("A", np.ones(6, bool))
        loo = [f4("T", np.arange(6) != b) / f4("A", np.arange(6) != b) for b in range(6)]
        self.assertAlmostEqual(result.summary.est[0], full)
        np.testing.assert_allclose(result.blocks.ratio_loo, loo)
        # The block with no R data still contributes through pairs involving O.
        self.assertEqual(result.summary.n_blocks[0], 6)
        self.assertTrue(pd.isna(result.summary.n[0]))

    def test_unverifiable_f4_cache_and_cached_allsnps_are_rejected(self):
        cache = F4BlockCache(BlockStats(pd.DataFrame(), None, np.ones(3), "f4"))
        with self.assertRaisesRegex(ValueError, "not F4BlockCache"):
            admixpy.f4_ratio(cache, **POPS)
        blocks = afs_to_f2_blocks(allele_data(), verbose=False)["f2_blocks"]
        with self.assertRaisesRegex(ValueError, "direct genotype"):
            admixpy.f4_ratio(blocks, **POPS, allsnps=True)

    def test_invalid_arguments_fail_before_reading(self):
        bad = [dict(reference=[]), dict(source1="T"), dict(target=None),
               dict(confidence=1), dict(denominator_z_min=0), dict(covariance=True),
               dict(models=pd.DataFrame([POPS])), dict(reference=[None])]
        for opts in bad:
            with self.subTest(opts=opts), patch("admixpy.fstats.anygeno_to_afs") as reader:
                with self.assertRaises(ValueError):
                    admixpy.f4_ratio("unused", **{**POPS, **opts})
                reader.assert_not_called()

    def test_missing_population_errors_name_the_arguments(self):
        with patch("admixpy.fstats.anygeno_to_afs") as reader:
            with self.assertRaisesRegex(
                ValueError,
                r"^f4_ratio\(\) missing required population arguments: "
                r"target, source1, source2, reference, outgroup$",
            ):
                admixpy.f4_ratio("unused")
            with self.assertRaisesRegex(
                ValueError,
                r"^f4_ratio\(\) missing required population argument: outgroup$",
            ):
                admixpy.f4_ratio("unused", **{**POPS, "outgroup": None})
            with self.assertRaisesRegex(
                ValueError,
                "^reference must contain at least one population$",
            ):
                admixpy.f4_ratio("unused", **{**POPS, "reference": []})
            reader.assert_not_called()

    def test_streaming_real_genotypes_matches_in_memory(self):
        rng = np.random.default_rng(54)
        populations = ["T", "A", "B", "R", "O", "R2"]
        geno = rng.integers(0, 3, (60, 12))
        geno[:4, :2] = 9
        geno[6:9, 10:] = 9
        with tempfile.TemporaryDirectory() as tmp:
            pref = Path(tmp) / "test"
            pref.with_suffix(".ind").write_text("".join(
                f"{p}{i} U {p}\n" for p in populations for i in range(2)))
            pref.with_suffix(".snp").write_text("".join(
                f"s{i} {i // 5 + 1} {i % 5 * .01} {i * 100} A G\n" for i in range(60)))
            pref.with_suffix(".geno").write_text("".join(
                "".join(map(str, row)) + "\n" for row in geno))
            for allsnps in [False, True]:
                opts = {**POPS, "reference": ["R", "R2"], "verbose": False,
                        "adjust_pseudohaploid": False, "return_blocks": True, "allsnps": allsnps}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    raw = admixpy.f4_ratio(pref, **opts)
                    stream = admixpy.f4_ratio(pref, stream=True, chunk_size=7, **opts)
                pd.testing.assert_frame_equal(raw.summary, stream.summary)
                pd.testing.assert_frame_equal(raw.components, stream.components)
                pd.testing.assert_frame_equal(raw.blocks, stream.blocks)


if __name__ == "__main__":
    unittest.main()
