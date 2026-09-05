import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import admixpy.fstats as f
from admixpy.genotypes import AfData, _read_tgeno, discard_from_aftable, tgeno_to_afs


def allele_data():
    rng = np.random.default_rng(914)
    pops = ["T", "S1", "S2", "R0", "R1", "R2", "R3"]
    values = rng.uniform(.1, .9, (300, len(pops)))
    values[rng.random(values.shape) < .08] = np.nan
    values[:30, 5] = np.nan  # A contrast absent from one whole block.
    values[30, :] = 0
    values[31, :] = 1
    snp = pd.DataFrame({
        "SNP": [f"s{i}" for i in range(len(values))], "CHR": 1,
        "cm": np.repeat(np.arange(10) * .1, 30), "POS": np.arange(300) * 100,
        "A1": "A", "A2": "G",
    })
    return AfData(
        pd.DataFrame(values, columns=pops),
        pd.DataFrame(np.where(np.isfinite(values), 10., 0.), columns=pops), snp,
    )


def contrasts():
    return pd.DataFrame([
        {"pop1": s, "pop2": "T", "pop3": r, "pop4": "R0"}
        for s in ["S1", "S2"] for r in ["R1", "R2", "R3"]
    ])


def fitting_cache():
    rows = contrasts().iloc[[0, 1, 3, 4]].reset_index(drop=True)
    est = np.array([.3, .6, -.2, -.4])
    loo = est[:, None] + np.random.default_rng(16).normal(0, .001, (4, 8))
    return f.F4BlockCache(f.BlockStats(rows, loo, np.ones(8), "f4", loo, est, np.eye(4)))


class ModelValidationTests(unittest.TestCase):
    def test_indefinite_covariance_cannot_return_a_passing_model(self):
        rows = contrasts().iloc[:2].copy()
        cov = np.array([[1., 2.], [2., 1.]])
        cache = f.F4BlockCache(f.BlockStats(
            rows, np.zeros((2, 3)), np.ones(3), "f4", est=np.array([1., -1.]), cov=cov,
        ))
        for caller, kwargs in (
            (f.qpadm, {"target": "T", "left": ["S1"]}),
            (f.qpwave, {"left": ["T", "S1"]}),
        ):
            with self.subTest(caller=caller.__name__):
                with self.assertRaisesRegex(ValueError, "not positive semidefinite.*minimum eigenvalue"):
                    caller(cache, right=["R0", "R1", "R2"], verbose=False, **kwargs)
        with self.assertRaisesRegex(ValueError, "not positive semidefinite"):
            f._qinv_from_cov(cov, .0001)

    def test_covariance_roundoff_at_zero_is_allowed(self):
        cov = np.array([[1., 1.], [1., 1. - 1e-15]])
        self.assertTrue(np.isfinite(f._qinv_from_cov(cov, .0001)).all())

    def test_bad_population_arguments_fail_before_loading(self):
        cases = [
            ({"left_base": "S1"}, "left_base must be the target"),
            ({"right": ["R0", "R1", "R1"]}, "Duplicate right"),
            ({"right": ["R0"]}, "At least two right"),
            ({"left": ["S1", "S2", "S3", "S4"]}, "requested rank exceeds"),
        ]
        for extra, message in cases:
            with self.subTest(extra=extra), patch("admixpy.fstats.qpwave_f4stats") as read:
                args = {"left": ["S1", "S2"], "right": ["R0", "R1", "R2"], **extra}
                with self.assertRaisesRegex(ValueError, message):
                    f.qpadm("unused", "T", verbose=False, **args)
                read.assert_not_called()

    def test_bad_batch_model_fails_before_building_cache(self):
        models = pd.DataFrame([{"target": "T", "left": ["S1"], "right": ["R0", "R1", "R1"]}])
        with patch("admixpy.fstats.f4_model_cache") as build:
            with self.assertRaisesRegex(ValueError, "Model 1: Duplicate right"):
                f.qpadm_multi("unused", models, verbose=False)
            build.assert_not_called()

    def test_optional_popdrop_preserves_weights_and_rank_tests(self):
        cache = fitting_cache()
        full = f.qpadm(cache, "T", ["S1", "S2"], ["R0", "R1", "R2"], verbose=False)
        with patch("admixpy.fstats.qpadm_popdrop", side_effect=AssertionError("unneeded subset fit")):
            lean = f.qpadm(cache, "T", sources=["S1", "S2"], right=["R0", "R1", "R2"],
                           left_base="T", popdrop=False, verbose=False)
        self.assertIsNone(lean.popdrop)
        self.assertEqual(len(full.popdrop), 3)
        pd.testing.assert_frame_equal(full.weights, lean.weights)
        pd.testing.assert_frame_equal(full.rankdrop, lean.rankdrop)
        np.testing.assert_allclose(lean.weights.weight, [.4, .6], atol=1e-6)

    def test_summary_batch_skips_covariance_and_subset_fits(self):
        cache = fitting_cache()
        models = pd.DataFrame([{"target": "T", "left": ["S1", "S2"], "right": ["R0", "R1", "R2"]}])
        expected = f.qpadm_multi(cache, models, full_results=True, verbose=False).result.iloc[0].rankdrop
        with patch("admixpy.fstats.qpadm_popdrop", side_effect=AssertionError("unneeded subset fit")), \
             patch("admixpy.fstats._weights_covariance", side_effect=AssertionError("unneeded weight covariance")):
            actual = f.qpadm_multi(cache, models, full_results=False, verbose=False)
        pd.testing.assert_frame_equal(actual[expected.columns], expected)


class ExactPopdropTests(unittest.TestCase):
    @staticmethod
    def cache():
        sources = ["S1", "S2", "S3"]
        rows = pd.DataFrame([
            {"pop1": s, "pop2": "T", "pop3": r, "pop4": "R0"}
            for s in sources for r in ["R1", "R2", "R3"]
        ])
        matrix = np.array([[1., .3, -.2], [.1, -.5, .8], [-.4, .6, .2]])
        cov = np.kron(
            np.array([[1., .7, .3], [.7, 1., .2], [.3, .2, 1.]]),
            np.array([[1., .2, .1], [.2, 1., .3], [.1, .3, 1.]]),
        )
        return f.F4BlockCache(f.BlockStats(rows, None, np.ones(8), "f4",
                                          est=matrix.ravel(), cov=cov))

    def test_every_subset_matches_independent_fit_including_regularization(self):
        cache = self.cache()
        sources, right = ["S1", "S2", "S3"], ["R0", "R1", "R2", "R3"]
        for twice in [False, True]:
            with self.subTest(fudge_twice=twice):
                kwargs = dict(fudge=.03, fudge_twice=twice, iterations=40,
                              getcov=False, verbose=False)
                result = f.qpadm(cache, "T", sources, right, **kwargs)
                for row in result.popdrop.itertuples(index=False):
                    kept = [s for s, bit in zip(sources, row.pat) if bit == "0"]
                    standalone = f.qpadm(cache, "T", kept, right, popdrop=False, **kwargs)
                    fit = standalone.rankdrop.iloc[0]
                    self.assertAlmostEqual(row.chisq, fit.chisq, places=12)
                    self.assertAlmostEqual(row.p, fit.p, places=12)
                    self.assertEqual(row.dof, fit.dof)
                    np.testing.assert_allclose([getattr(row, s) for s in kept],
                                               standalone.weights.weight, atol=1e-12)

    def test_nested_tests_use_common_covariance_and_real_parents(self):
        cache = self.cache()
        sources = ["S1", "S2", "S3"]
        result = f.qpadm(cache, "T", sources, ["R0", "R1", "R2", "R3"],
                         fudge=.03, fudge_twice=True, iterations=40,
                         getcov=False, verbose=False)
        common_cov = f._regularize_covariance(cache.stats.cov, .03, True)
        matrix = cache.stats.est.reshape(3, 3)
        by_pattern = result.popdrop.set_index("pat")
        for child in result.popdrop.itertuples(index=False):
            if pd.isna(child.nested_parent):
                self.assertTrue(np.isnan(child.p_nested))
                continue
            parent = by_pattern.loc[child.nested_parent]
            self.assertEqual(sum(a != b for a, b in zip(child.pat, child.nested_parent)), 1)
            self.assertTrue(all(a == "1" or b == "0" for a, b in zip(child.pat, child.nested_parent)))
            for pattern, chisq in [(child.pat, child.nested_chisq),
                                    (child.nested_parent, parent.nested_chisq)]:
                keep = [i for i, bit in enumerate(pattern) if bit == "0"]
                flat = np.concatenate([np.arange(i*3, i*3+3) for i in keep])
                expected = f.qpadm_fit(matrix[keep], np.linalg.inv(common_cov[np.ix_(flat, flat)]),
                                       len(keep)-1, fudge=.03, iterations=40)
                self.assertAlmostEqual(chisq, expected["chisq"], places=12)
            self.assertEqual(child.dofdiff, 1)
            delta = child.nested_chisq - parent.nested_chisq
            self.assertAlmostEqual(child.chisqdiff, delta, places=12)
            if delta >= 0:
                self.assertAlmostEqual(child.p_nested, f._chi2_sf(delta, 1), places=12)
            else:
                self.assertTrue(np.isnan(child.p_nested))

    def test_parent_selection_does_not_compare_siblings_or_unrelated_models(self):
        out = pd.DataFrame({
            "pat": ["0000", "0001", "0010", "0100", "1000", "1100", "1110"],
            "chisq": [1., 2., 3., 4., 5., 7., 8.], "feasible": [True]*7,
        })
        for frame in [out, out.sample(frac=1, random_state=52).reset_index(drop=True)]:
            best, parents = f._popdrop_nested_parents(frame, 4)
            mapping = {frame.iloc[i].pat: frame.iloc[p].pat for i, p in enumerate(parents) if p >= 0}
            self.assertEqual(mapping["1100"], "0100")
            self.assertEqual(mapping["1110"], "1100")
            for pat in ["0001", "0010", "0100", "1000"]:
                self.assertEqual(mapping[pat], "0000")
            self.assertTrue(best.all())

    def test_qinv_only_entry_point_uses_marginal_covariance(self):
        matrix = np.array([[1., 0.], [.5, 0.]])
        cov = np.kron(np.array([[1., .8], [.8, 1.]]), np.eye(2))
        result = f.qpadm_popdrop(matrix, np.linalg.inv(cov), ["S1", "S2"], fudge=0)
        row = result.set_index("pat").loc["01"]
        self.assertAlmostEqual(row.chisq, 1.)
        self.assertAlmostEqual(row.p, f._chi2_sf(1., 2))
        with self.assertRaisesRegex(ValueError, "requires raw cov when qinv is singular"):
            f.qpadm_popdrop(matrix, np.zeros((4, 4)), ["S1", "S2"])

    def test_nonmonotonic_nested_fit_does_not_produce_p_one(self):
        matrix = np.array([[1., 0.], [.5, 0.]])
        original_fit = f.qpadm_fit
        def nonmonotonic(*args, **kwargs):
            fit = original_fit(*args, **kwargs)
            fit["chisq"] = 10. if args[0].shape[0] == 2 else 1.
            return fit
        with patch("admixpy.fstats.qpadm_fit", side_effect=nonmonotonic):
            result = f.qpadm_popdrop(matrix, np.eye(4), ["S1", "S2"])
        children = result.loc[result.nested_parent.notna()]
        self.assertTrue((children.chisqdiff < 0).all())
        self.assertTrue(children.p_nested.isna().all())


class OptimizationRegressionTests(unittest.TestCase):
    def test_streaming_fast_f4_matches_materialized_with_singletons(self):
        afdat = allele_data()
        afdat.counts = afdat.counts.clip(upper=1)

        def chunks(*args, **kwargs):
            for start in range(0, len(afdat.afs), 17):
                snp = afdat.snpfile.iloc[start:start+17].reset_index(drop=True)
                afs = afdat.afs.iloc[start:start+17].copy()
                counts = afdat.counts.iloc[start:start+17].copy()
                afs.index = counts.index = snp.SNP
                yield AfData(afs, counts, snp)

        for allsnps in [True, False]:
            with self.subTest(allsnps=allsnps):
                with patch("admixpy.fstats.anygeno_to_afs", return_value=afdat):
                    expected = f.f4_stats_from_geno("unused", contrasts(), allsnps=allsnps,
                                                    minac2=2, verbose=False)
                with patch("admixpy.fstats.iter_geno_to_afs", side_effect=chunks):
                    actual = f.f4_stats_from_geno("unused", contrasts(), allsnps=allsnps,
                                                  minac2=2, stream=True, verbose=False)
                for name in ("est", "cov", "blocks", "loo", "snp_counts"):
                    np.testing.assert_allclose(getattr(actual, name), getattr(expected, name),
                                               atol=1e-15, equal_nan=True)

    def test_f4_matrix_path_matches_scalar_with_missingness_and_weights(self):
        afdat = allele_data()
        combos = contrasts()
        # Preserve ordering, duplicate requests, and multiple model panels.
        combos = pd.concat([combos.iloc[::-1], combos.iloc[[0]]], ignore_index=True)
        second = contrasts().iloc[:4].copy()
        combos["model"] = 1
        second["model"] = 2
        combos = pd.concat([combos, second], ignore_index=True)
        for allsnps, poly_only, weighted in [(True, False, False), (True, False, True),
                                            (False, False, True), (False, True, False),
                                            (True, True, False)]:
            with self.subTest(allsnps=allsnps, poly_only=poly_only, weighted=weighted):
                kwargs = dict(allsnps=allsnps, poly_only=poly_only, verbose=False,
                              snpwt=np.linspace(.5, 2, 300) if weighted else None)
                with patch("admixpy.fstats._f4_matmul_groups", return_value=[]):
                    expected = f._f4_direct_blocks_from_afs(afdat, combos, **kwargs)
                actual = f._f4_direct_blocks_from_afs(afdat, combos, **kwargs)
                for field in ("blocks", "est", "loo", "cov", "snp_counts"):
                    np.testing.assert_allclose(getattr(actual, field), getattr(expected, field),
                                               rtol=1e-12, atol=1e-15, equal_nan=True)

    def test_repeated_population_keeps_bias_correction_with_fast_neighbors(self):
        combos = pd.concat([contrasts(), pd.DataFrame([
            {"pop1": "S1", "pop2": "T", "pop3": "S1", "pop4": "R0"},
        ])], ignore_index=True)
        afdat = allele_data()
        with patch("admixpy.fstats._f4_matmul_groups", return_value=[]):
            expected = f._f4_direct_blocks_from_afs(afdat, combos, allsnps=True, verbose=False)
        actual = f._f4_direct_blocks_from_afs(afdat, combos, allsnps=True, verbose=False)
        np.testing.assert_allclose(actual.blocks, expected.blocks, atol=1e-15, equal_nan=True)
        np.testing.assert_array_equal(actual.snp_counts, expected.snp_counts)

    def test_pair_counts_match_explicit_pairs_for_rectangular_blocks(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(size=(43, 7)), rng.normal(size=(43, 4))
        a[rng.random(a.shape) < .2] = np.nan
        b[rng.random(b.shape) < .2] = np.inf
        lengths = [1, 13, 29]
        actual = f.mats_to_ctarr(a, b, lengths)
        start = 0
        for block, n in enumerate(lengths):
            for i in range(7):
                for j in range(4):
                    self.assertEqual(actual[i, j, block], np.mean(
                        np.isfinite(a[start:start+n, i]) & np.isfinite(b[start:start+n, j])))
            start += n

    def test_tgeno_decoding_handles_padding_selection_and_unaligned_ranges(self):
        rng = np.random.default_rng(83)
        geno = rng.integers(0, 4, (19, 5), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            pref = Path(tmp) / "packed"
            payload = [b"TGENO 5 19 0 0".ljust(48, b"\0")]
            for col in geno.T:
                packed = bytearray(48)
                for i, value in enumerate(col):
                    packed[i // 4] |= int(value) << (6 - 2 * (i % 4))
                payload.append(bytes(packed))
            pref.with_suffix(".geno").write_bytes(b"".join(payload))
            pref.with_suffix(".ind").write_text("\n".join(f"i{i} U P{i}" for i in range(5)) + "\n")
            pref.with_suffix(".snp").write_text("\n".join(f"s{i} 1 {i*.01} {i*100} A G" for i in range(19)) + "\n")
            expected = geno.astype(float)
            expected[expected == 3] = np.nan
            for first, last in [(1, 19), (2, 17), (3, 18), (4, 4), (5, 19)]:
                actual = _read_tgeno(pref.with_suffix(".geno"), 19, 5, first, last, np.array([4, 1]))
                np.testing.assert_equal(actual, expected[first-1:last, [4, 1]])
            full = tgeno_to_afs(pref, pops=["P4", "P1"], adjust_pseudohaploid=False, verbose=False)
            chunked = tgeno_to_afs(pref, pops=["P4", "P1"], adjust_pseudohaploid=False,
                                   chunked=True, chunk_size=3, verbose=False)
            pd.testing.assert_frame_equal(full.afs, chunked.afs)
            pd.testing.assert_frame_equal(full.counts, chunked.counts)
            np.testing.assert_equal(full.afs.to_numpy(), expected[:, [4, 1]] / 2)

    def test_minac2_mode_two_exempts_all_singletons(self):
        afdat = allele_data()
        afdat.counts = afdat.counts.clip(upper=1)
        expected = discard_from_aftable(afdat, minac2=False)
        actual = discard_from_aftable(afdat, minac2=2)
        pd.testing.assert_frame_equal(actual.afs, expected.afs)
        with self.assertRaisesRegex(ValueError, "No SNPs remain"):
            discard_from_aftable(afdat, minac2=True)


class DeferredCacheTests(unittest.TestCase):
    def test_nondefault_right_base_matches_cached_and_uncached_batch(self):
        afdat = allele_data()
        # A complete panel ensures positive semidefinite covariance for fitting.
        afdat.afs = afdat.afs.fillna(.5)
        afdat.counts.iloc[:] = 10
        models = pd.DataFrame([{"target": "T", "left": ["S1", "S2"],
                                "right": ["R0", "R1", "R2", "R3"]}])
        with patch("admixpy.fstats.anygeno_to_afs", return_value=afdat):
            cached = f.qpadm_multi("unused", models, right_base="R1", left_base="T",
                                    full_results=False, verbose=False)
            direct = f.qpadm_multi("unused", models, right_base="R1", left_base="T",
                                    full_results=False, use_cache=False, verbose=False)
        pd.testing.assert_frame_equal(cached, direct, rtol=1e-10, atol=1e-12)

    def test_cache_resampling_mismatch_is_rejected(self):
        for method, other in [("pairwise_counts", "nominal_blocks"), ("nominal_blocks", "pairwise_counts")]:
            with self.subTest(method=method):
                stats = f._f4_direct_blocks_from_afs(allele_data(), contrasts(), allsnps=True, verbose=False)
                stats = f._set_direct_resampling(stats, method, covariance=True)
                cache = f.F4BlockCache(stats)
                with self.assertRaisesRegex(ValueError, "cache uses resampling=.*rebuild"):
                    f.f4_stats(cache, contrasts(), resampling=other, verbose=False)
                with self.assertRaisesRegex(ValueError, "cache uses resampling=.*rebuild"):
                    f.f4_model_cache(cache, [], resampling=other, verbose=False)

    def test_batch_cache_defers_covariance_and_matches_independent_models(self):
        models = pd.DataFrame([
            {"target": "T", "left": ["S1", "S2"], "right": ["R0", "R1", "R2", "R3"]},
            {"target": "T", "left": ["S1"], "right": ["R0", "R1", "R2"]},
        ])
        for method in ["pairwise_counts", "nominal_blocks"]:
            for allsnps in [True, False]:
                with self.subTest(method=method, allsnps=allsnps):
                    with patch("admixpy.fstats.anygeno_to_afs", return_value=allele_data()), \
                         patch("admixpy.fstats._influence_covariance", side_effect=AssertionError("dense batch covariance")), \
                         patch("admixpy.fstats.jackknife_cov", side_effect=AssertionError("dense nominal covariance")):
                        cache = f.f4_model_cache("unused", models, allsnps=allsnps,
                                                 resampling=method, verbose=False)
                    self.assertIsNone(cache.stats.cov)
                    self.assertTrue(cache.deferred_covariance)
                    for i, model in enumerate(models.itertuples(index=False), 1):
                        selected = f._select_f4_block_cache_model(cache, i)
                        actual = f.qpwave_f4stats(selected, ["T"] + model.left, model.right,
                                                 resampling=method, verbose=False)
                        with patch("admixpy.fstats.anygeno_to_afs", return_value=allele_data()):
                            expected = f.qpwave_f4stats("unused", ["T"] + model.left, model.right,
                                                       allsnps=allsnps, resampling=method, maxmiss=1,
                                                       verbose=False)
                        np.testing.assert_allclose(actual.matrix, expected.matrix, atol=1e-15)
                        np.testing.assert_allclose(actual.cov, expected.cov, atol=1e-15, equal_nan=True)
                        np.testing.assert_allclose(actual.loo, expected.loo, atol=1e-15, equal_nan=True)
                        self.assertEqual(selected.stats.resampling, method)
                        self.assertIsNone(selected.stats.cov)
                        lean = f.f4_stats(selected, actual.f4.rows, resampling=method,
                                          keep_blocks=False, keep_loo=False, verbose=False)
                        np.testing.assert_allclose(lean.cov, actual.cov, atol=1e-15, equal_nan=True)
                        self.assertIsNone(lean.blocks)
                        self.assertIsNone(lean.loo)


if __name__ == "__main__":
    unittest.main()
