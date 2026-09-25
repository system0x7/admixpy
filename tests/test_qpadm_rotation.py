import importlib
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import admixpy


class RotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pops = ["T", "A", "B", "C", "R0", "R1"]
        af = np.random.default_rng(719).uniform(.1, .9, (len(pops), 120, 8))
        distances = ((af[:, None] - af[None, :]) ** 2).mean(axis=-1)
        cls.blocks = admixpy.F2Blocks(distances, pops, pops, np.full(120, 8))

    def rotate(self, **kwargs):
        return admixpy.qpadm_rotate(
            self.blocks, ["A", "B", "C"], "T", ["R0", "R1"],
            verbose=False, resampling="nominal_blocks", **kwargs,
        )

    def test_cached_and_direct_tables_agree(self):
        cached = self.rotate(full_results=True, getcov=True)
        direct = self.rotate(full_results=True, getcov=True, use_cache=False)
        pd.testing.assert_frame_equal(cached.models, direct.models)
        pd.testing.assert_frame_equal(cached.weights, direct.weights)
        self.assertTrue(cached.errors.empty)
        self.assertEqual(cached.settings["admixpy_version"], admixpy.__version__)
        self.assertTrue(all(isinstance(x, list) for x in cached.models.left))
        self.assertEqual(list(cached.models.n_sources), [1, 1, 1, 2, 2, 2])
        self.assertEqual(len(cached.weights), 9)

    def test_ids_are_stable_across_source_filters(self):
        all_models = self.rotate().models
        pairs = self.rotate(source_sizes=[2]).models
        self.assertEqual(list(all_models.loc[all_models.n_sources == 2, "model"]), list(pairs.model))
        self.assertTrue(all_models.model.is_unique)

    def test_screen_marks_feasibility_unavailable(self):
        result = self.rotate()
        self.assertTrue(result.weights.empty)
        self.assertTrue(result.models.feasible.isna().all())
        self.assertEqual(str(result.models.feasible.dtype), "boolean")
        self.assertFalse(hasattr(result, "to_csv"))

    def test_model_errors_record_or_raise(self):
        module = importlib.import_module("admixpy.qpadm")
        original = module.qpadm

        def failing(data, target, left, right, **kwargs):
            if left == ["B"]:
                raise ValueError("invalid model covariance")
            return original(data, target, left, right, **kwargs)

        with patch.object(module, "qpadm", side_effect=failing):
            result = self.rotate(source_sizes=[1], full_results=True, on_error="record")
            self.assertEqual(list(result.models.status), ["ok", "error", "ok"])
            self.assertEqual(list(result.weights.source), ["A", "C"])
            self.assertEqual(result.errors.iloc[0].model, result.models.iloc[1].model)
            self.assertEqual(result.errors.iloc[0].message, "invalid model covariance")
            self.assertTrue(pd.isna(result.models.iloc[1].feasible))
            with self.assertRaisesRegex(ValueError, "invalid model covariance"):
                self.rotate(source_sizes=[1])

    def test_invalid_options_fail_before_cache(self):
        with patch("admixpy.fstats.f4_model_cache") as cache:
            for options in (dict(getcov=True), dict(on_error="ignore"),
                            dict(right_base="A"), dict(left_base="A"),
                            dict(popdrop=True)):
                with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                    self.rotate(**options)
            cache.assert_not_called()

    def test_cache_errors_are_not_recorded_as_model_errors(self):
        with patch("admixpy.fstats.f4_model_cache", side_effect=ValueError("bad input")):
            with self.assertRaisesRegex(ValueError, "bad input"):
                self.rotate(on_error="record")


if __name__ == "__main__":
    unittest.main()
