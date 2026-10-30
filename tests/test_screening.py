import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import admixpy as ap


screening_module = importlib.import_module("admixpy.screening")


def basis_cache(estimates, covariance=None, candidates=None):
    candidates = candidates or ["A", "B", "C"]
    n_coordinates = estimates.shape[1]
    refs = [f"R{i}" for i in range(n_coordinates)]
    rows = pd.DataFrame([
        (source, "T", ref, "O")
        for source in candidates for ref in refs
    ], columns=["pop1", "pop2", "pop3", "pop4"])
    if covariance is None:
        rng = np.random.default_rng(512)
        noise = rng.normal(size=(estimates.size, estimates.size))
        covariance = (noise @ noise.T / estimates.size + np.eye(estimates.size)) * .01
    stats = ap.BlockStats(rows, None, np.ones(80), "f4",
                          est=estimates.flatten(), cov=covariance)
    return ap.F4BlockCache(stats), refs


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.sources = np.array([[.1, .5, -.3, .9], [.7, -.1, .4, .2], [-.4, .8, .2, -.3]])
        target = .35 * self.sources[0] + .65 * self.sources[1]
        self.estimates = self.sources - target
        self.cache, self.refs = basis_cache(self.estimates)

    def screen(self, **kwargs):
        options = dict(outgroup="O", verbose=False)
        options.update(kwargs)
        return ap.screening(self.cache, "T", ["A", "B", "C"], self.refs, **options)

    def test_recovers_mixture_and_preserves_numeric_tables_and_group_order(self):
        result = self.screen()
        self.assertIsInstance(result, ap.ScreeningResult)
        best = result.models.iloc[0]
        self.assertEqual(best.sources, ["A", "B"])
        self.assertLess(best.score, 1e-5)
        np.testing.assert_allclose(result.weights.query("model == @best.model").weight, [.35, .65], atol=1e-5)
        for _, rows in result.models.groupby("n_sources"):
            self.assertTrue(rows.score.is_monotonic_increasing)
        self.assertEqual(result.models.n_sources.tolist(), [2, 2, 2, 3])
        for _, weights in result.weights.groupby("model"):
            self.assertAlmostEqual(weights.weight.sum(), 1.)
            self.assertTrue(weights.weight.between(0, 1).all())
        self.assertTrue(pd.api.types.is_numeric_dtype(result.models.score))
        self.assertFalse(any("p_" in col or col == "p" for col in result.models.columns))
        self.assertFalse(result.settings["score_is_pvalue"])
        self.assertEqual(result.settings["admixpy_version"], ap.__version__)

    def test_default_outgroup_matches_explicit_and_records_resolved_panel(self):
        explicit = self.screen()
        default = ap.screening(self.cache, "T", ["A", "B", "C"], ["O", *self.refs], verbose=False)
        included = ap.screening(self.cache, "T", ["A", "B", "C"],
                                [*self.refs[:2], "O", *self.refs[2:]], outgroup="O", verbose=False)
        for result in [default, included]:
            pd.testing.assert_frame_equal(result.models, explicit.models)
            pd.testing.assert_frame_equal(result.weights, explicit.weights)
            self.assertEqual(result.settings["outgroup"], "O")
            self.assertEqual(result.settings["refs"], self.refs)

    def test_correlated_uncertain_sources_match_independent_grid_search(self):
        estimates = self.estimates[:2] + [.04, -.06, .02, .1]
        cache, refs = basis_cache(estimates, candidates=["A", "B"])
        result = ap.screening(cache, "T", ["A", "B"], refs, outgroup="O", verbose=False)
        weights = result.weights.weight.to_numpy()
        covariance = cache.stats.cov
        ridge = 1e-6 * np.mean(np.diag(covariance))
        regularized = covariance + np.eye(estimates.size) * ridge
        cov4 = regularized.reshape(2, 4, 2, 4)
        grid = np.linspace(0, 1, 5001)
        coefficients = np.column_stack([grid, 1 - grid])
        residuals = coefficients @ estimates
        covariances = np.einsum("ga,aibj,gb->gij", coefficients, cov4, coefficients)
        solved = np.linalg.solve(covariances, residuals[..., None])[..., 0]
        scores = np.sqrt(np.einsum("gi,gi->g", residuals, solved) / 4)
        best = np.argmin(scores)
        self.assertLessEqual(result.models.score.iloc[0], scores[best] + 1e-7)
        self.assertAlmostEqual(weights[0], grid[best], delta=.0003)
        # Independently propagate all cross-covariance blocks at the fitted weights.
        transform = np.hstack([weights[0] * np.eye(4), weights[1] * np.eye(4)])
        residual = weights @ estimates
        propagated = transform @ covariance @ transform.T
        z = residual / np.sqrt(np.diag(propagated))
        self.assertAlmostEqual(result.models.max_abs_z.iloc[0], max(abs(z)))
        expected = np.sqrt(residual @ np.linalg.solve(transform @ regularized @ transform.T, residual) / 4)
        self.assertAlmostEqual(result.models.score.iloc[0], expected)
        # Ignoring correlations between reference contrasts changes the score.
        naive = np.sqrt(np.mean(residual ** 2 / np.diag(propagated)))
        self.assertGreater(abs(expected - naive), .01)

    def test_source_order_and_global_scale_preserve_mixture(self):
        base = self.screen(source_sizes=[2])
        reordered = ap.screening(self.cache, "T", ["C", "B", "A"], self.refs,
                                  outgroup="O", source_sizes=[2], verbose=False)
        cache, _ = basis_cache(self.estimates * .001, self.cache.stats.cov * 1e-6)
        scaled = ap.screening(cache, "T", ["A", "B", "C"], self.refs,
                              outgroup="O", source_sizes=[2], verbose=False)
        for result in [reordered, scaled]:
            best = result.models.iloc[0]
            weights = result.weights.query("model == @best.model").set_index("source").weight
            self.assertAlmostEqual(weights["A"], .35, places=4)
            self.assertAlmostEqual(weights["B"], .65, places=4)
        np.testing.assert_allclose(base.models.score, scaled.models.score, atol=1e-7)

    def test_singular_covariance_and_zero_weight_boundary_are_visible(self):
        estimates = self.sources[:2] - self.sources[0]
        covariance = np.kron(np.eye(2), np.ones((4, 4))) * .01
        cache, refs = basis_cache(estimates, covariance, ["A", "B"])
        result = ap.screening(cache, "T", ["A", "B"], refs, outgroup="O", verbose=False)
        self.assertTrue(np.isfinite(result.models.score.iloc[0]))
        self.assertEqual(result.models.covariance_rank.iloc[0], 1)
        self.assertIn("singular_covariance", result.models.status.iloc[0])
        self.assertIn("boundary", result.models.status.iloc[0])
        np.testing.assert_allclose(result.weights.weight, [1, 0], atol=1e-6)

    def test_indistinguishable_models_and_optimizer_failures_are_retained(self):
        estimates = self.estimates.copy()
        estimates[1] = estimates[0]
        cache, refs = basis_cache(estimates)
        result = ap.screening(cache, "T", ["A", "B", "C"], refs, outgroup="O", verbose=False)
        degenerate = result.models.loc[result.models.sources.map(lambda s: s == ["A", "B"])].iloc[0]
        self.assertEqual(degenerate.status, "indistinguishable_sources")
        self.assertTrue(np.isnan(degenerate.score))
        self.assertFalse(result.weights.model.eq(degenerate.model).any())
        with patch.object(screening_module, "minimize", return_value=SimpleNamespace(
                success=False, message="iteration limit")):
            result = self.screen(source_sizes=[2])
        self.assertEqual(len(result.models), 3)
        self.assertTrue(result.models.status.eq("optimizer_failed").all())
        self.assertTrue(result.models.message.str.contains("iteration limit").all())
        self.assertTrue(result.weights.empty)
        self.assertIn("optimizer_failed", result.format(top=1))

    def test_explicit_single_source_models(self):
        result = self.screen(source_sizes=[1])
        self.assertEqual(len(result.models), 3)
        np.testing.assert_array_equal(result.weights.weight, 1.)
        self.assertTrue(result.models.score.notna().all())

    def test_basis_and_model_covariances_match_qpadm_exactly(self):
        with patch.object(screening_module, "_fit_mixture", wraps=screening_module._fit_mixture) as fit:
            result = self.screen(source_sizes=[1, 2, 3])
        self.assertEqual(result.settings["left_base"], "T")
        self.assertEqual(result.settings["right_base"], "O")
        self.assertEqual(result.coordinates.pop3.unique().tolist(), self.refs)
        self.assertTrue(result.models.n_contrasts.eq(len(self.refs)).all())
        for model, call in enumerate(fit.call_args_list, 1):
            row = result.models.set_index("model").loc[model]
            # Match the diagonal regularization for the single-source check.
            # qpAdm uses fudge * trace(C); screening uses ridge * mean(diag(C)).
            fudge = 1e-6 / (row.n_sources * len(self.refs))
            qpadm = ap.qpadm(self.cache, "T", row.sources, ["O", *self.refs],
                            return_stats=True, getcov=False, popdrop=False,
                            fudge=fudge, verbose=False)
            np.testing.assert_array_equal(call.args[0], qpadm.qpwave.matrix)
            np.testing.assert_array_equal(
                call.args[1].reshape(qpadm.qpwave.cov.shape), qpadm.qpwave.cov)
            expected_rows = qpadm.qpwave.f4.rows[["pop1", "pop2", "pop3", "pop4"]]
            selected = result.coordinates.loc[result.coordinates.pop1.isin(row.sources), expected_rows.columns]
            pd.testing.assert_frame_equal(selected.reset_index(drop=True), expected_rows)
            if row.n_sources == 1:
                self.assertAlmostEqual(row.score ** 2 * len(self.refs), qpadm.rankdrop.chisq.iloc[0])

    def test_retains_reference_direction_that_old_basis_would_cancel(self):
        estimates = np.ones((1, 4))
        cache, refs = basis_cache(estimates, np.eye(4), candidates=["A"])
        result = ap.screening(cache, "T", ["A"], ["O", *refs],
                              source_sizes=[1], verbose=False)
        # Taking differences relative to R0 would erase this entire residual.
        np.testing.assert_array_equal(estimates - estimates[:, :1], 0.)
        self.assertGreater(result.models.score.iloc[0], .99)
        self.assertEqual(result.models.n_contrasts.iloc[0], 4)

    def test_source_limit_uses_all_right_contrasts(self):
        sources = np.random.default_rng(61).normal(size=(4, 4))
        expected = np.array([.1, .2, .3, .4])
        candidates = ["A", "B", "C", "D"]
        cache, refs = basis_cache(sources - expected @ sources, candidates=candidates)
        result = ap.screening(cache, "T", candidates, ["O", *refs], verbose=False)
        self.assertEqual(result.settings["source_sizes"], [2, 3, 4])
        four = result.models.loc[result.models.n_sources == 4].iloc[0]
        self.assertEqual(four.n_contrasts, 4)
        np.testing.assert_allclose(result.weights.query("model == @four.model").weight, expected, atol=1e-4)
        # Two sources need two contrasts, not the old three-reference requirement.
        cache, refs = basis_cache(np.array([[1., 0.], [-1., 0.]]), candidates=["A", "B"])
        pair = ap.screening(cache, "T", ["A", "B"], ["O", *refs], verbose=False)
        self.assertEqual(pair.models.n_contrasts.iloc[0], 2)
        np.testing.assert_allclose(pair.weights.weight, [.5, .5], atol=1e-5)

    def test_display_limits_each_group_and_does_not_change_raw_data(self):
        result = self.screen()
        original = result.models.copy(deep=True)
        text = result.format(top=1)
        self.assertIn("S1 = A", text)
        self.assertIn("2-source models (1 of 3)", text)
        self.assertIn("3-source models (1 of 1)", text)
        self.assertIn("S1:35.0%", text)
        self.assertNotIn("p_approx", text)
        self.assertEqual(repr(result), result.format())
        pd.testing.assert_frame_equal(result.models, original)
        self.assertIn("score", result.models.to_csv(index=False))
        for top in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                result.format(top=top)

    def test_invalid_inputs_fail_before_io(self):
        cases = [dict(target=""), dict(candidates=["A", "A"]), dict(candidates=["T", "A"]),
                 dict(refs=["R0", "A", "R2"]), dict(refs=["R0", "R0"]),
                 dict(refs=[]), dict(refs=["O", "R0"], outgroup=None), dict(outgroup=""),
                 dict(source_sizes=[]), dict(source_sizes=[0]), dict(source_sizes=[True]),
                 dict(source_sizes=[4]), dict(source_sizes=[1.5]),
                 dict(covariance_ridge=0), dict(covariance_ridge=np.nan),
                 dict(covariance_ridge=True), dict(allsnps="yes"), dict(allsnps=1), dict(covariance=False),
                 dict(afprod=True), dict(left_base="T"), dict(right_base="O")]
        with patch("admixpy.fstats.f4_stats") as compute:
            for changes in cases:
                options = dict(target="T", candidates=["A", "B", "C"], refs=self.refs,
                               outgroup="O", verbose=False)
                options.update(changes)
                with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                    ap.screening("unused", **options)
            compute.assert_not_called()
        self.cache.allsnps = True
        with self.assertRaisesRegex(ValueError, "does not match the f4 cache"):
            self.screen()

    def test_cached_allsnps_setting_must_match_and_is_recorded(self):
        expected = self.screen()
        with self.assertRaisesRegex(ValueError, "does not match the f4 cache"):
            self.screen(allsnps=True)
        self.cache.allsnps = True
        actual = self.screen(allsnps=True)
        pd.testing.assert_frame_equal(actual.models, expected.models)
        self.assertTrue(actual.settings["allsnps"])

    def test_invalid_or_missing_covariance_is_not_silently_diagonalized(self):
        for bad in [None, np.full((12, 12), np.nan), np.zeros((12, 12)),
                    -np.eye(12)]:
            with self.subTest(covariance=bad):
                self.cache.stats.cov = bad
                with self.assertRaises(ValueError):
                    self.screen()

    def test_direct_genotype_path_loads_basis_once_and_respects_snp_panel_option(self):
        rng = np.random.default_rng(72)
        pops = ["T", "A", "B", "C", "O", *self.refs]
        af = pd.DataFrame(rng.uniform(.1, .9, (480, len(pops))), columns=pops)
        af["T"] = .35 * af.A + .65 * af.B
        af.loc[0, "C"] = np.nan
        af.loc[1, "R2"] = np.nan
        af.index = [f"s{i}" for i in range(len(af))]
        snps = pd.DataFrame(dict(SNP=af.index, CHR=np.repeat(np.arange(1, 21), 24),
                                 cm=np.tile(np.arange(24) * .001, 20), POS=np.arange(480) * 100,
                                 A1="A", A2="G"))
        data = ap.AfData(af, pd.DataFrame(20., index=af.index, columns=pops), snps)
        for allsnps in [False, True]:
            with self.subTest(allsnps=allsnps), patch(
                    "admixpy.fstats.anygeno_to_afs", return_value=data) as reader:
                # Leave the default mode implicit to check backwards compatibility.
                options = dict(allsnps=True) if allsnps else {}
                result = ap.screening("unused", "T", ["A", "B", "C"], ["O", *self.refs],
                                      verbose=False, blgsize=.05, **options)
            reader.assert_called_once()
            self.assertEqual(result.models.iloc[0].sources, ["A", "B"])
            for row in result.coordinates.itertuples(index=False):
                panel = af.dropna(subset=[row.pop1, row.pop2, row.pop3, row.pop4]) if allsnps else af.dropna()
                expected = ((panel[row.pop1] - panel[row.pop2]) *
                            (panel[row.pop3] - panel[row.pop4])).mean()
                self.assertAlmostEqual(row.est, expected)
                self.assertEqual(row.n, len(panel))
            self.assertEqual(result.settings["allsnps"], allsnps)
            self.assertEqual(result.settings["blgsize"], .05)
            # Compare against the actual qpAdm statistic builder in both modes.
            with patch("admixpy.fstats.anygeno_to_afs", return_value=data):
                qpw = ap.qpwave_f4stats("unused", left=["T", "A", "B", "C"],
                                        right=["O", *self.refs], allsnps=allsnps,
                                        blgsize=.05, verbose=False)
            columns = ["pop1", "pop2", "pop3", "pop4"]
            pd.testing.assert_frame_equal(result.coordinates[columns], qpw.f4.rows[columns])
            np.testing.assert_array_equal(result.coordinates.est, qpw.f4.est)
            np.testing.assert_allclose(result.coordinates.se, qpw.f4.se, atol=1e-15)

    def test_precomputed_f2_input(self):
        pops = ["T", "A", "B", "C", "O", *self.refs]
        af = np.random.default_rng(719).uniform(.1, .9, (len(pops), 120, 8))
        distances = ((af[:, None] - af[None, :]) ** 2).mean(axis=-1)
        blocks = ap.F2Blocks(distances, pops, pops, np.full(120, 8))
        result = ap.screening(blocks, "T", ["A", "B", "C"], ["O", *self.refs],
                              verbose=False, resampling="nominal_blocks")
        self.assertEqual(len(result.models), 4)
        self.assertTrue(np.isfinite(result.models.score).all())
        self.assertNotIn("n", result.coordinates)
        with self.assertRaisesRegex(ValueError, "allsnps=True is only available"):
            ap.screening(blocks, "T", ["A", "B", "C"], ["O", *self.refs],
                         allsnps=True, verbose=False)


if __name__ == "__main__":
    unittest.main()
