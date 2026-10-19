import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import admixpy
from admixpy.fstats import BlockStats, F4BlockCache
from admixpy.genotypes import AfData


class F4ProximityTests(unittest.TestCase):
    def test_direct_estimates_match_per_snp_rms_and_report_minimum_counts(self):
        rng = np.random.default_rng(41)
        pops = ["O", "R1", "R2", "A", "B", "T"]
        af = pd.DataFrame(rng.uniform(.1, .9, (24, 6)), columns=pops)
        af.index = [f"s{i}" for i in range(24)]
        af.loc[af.index[:2], "R1"] = np.nan
        snps = pd.DataFrame(dict(
            SNP=af.index, CHR=np.repeat(np.arange(1, 7), 4),
            cm=np.tile([0, .01, .02, .03], 6), POS=np.arange(24) * 100,
            A1="A", A2="G",
        ))
        data = AfData(af, pd.DataFrame(20., index=af.index, columns=pops), snps)
        with patch("admixpy.fstats.anygeno_to_afs", return_value=data):
            result = admixpy.f4_proximity(
                "unused", "T", ["B", "A"], ["R1", "R2"],
                outgroup="O", allsnps=True, verbose=False,
            )
            individual = admixpy.f4(
                "unused", "O", ["R1", "R2"], ["B", "A"], "T",
                allsnps=True, verbose=False,
            )
            implicit = admixpy.f4_proximity(
                "unused", "T", ["B", "A"], ["O", "R1", "R2"],
                allsnps=True, verbose=False,
            )
        pd.testing.assert_frame_equal(implicit, result)
        for row in result.itertuples(index=False):
            expected = []
            for ref in ["R1", "R2"]:
                x = af.dropna(subset=["O", ref, row.population, "T"])
                expected.append(((x.O - x[ref]) * (x[row.population] - x["T"])).mean())
            self.assertAlmostEqual(row.rms_f4, np.sqrt(np.mean(np.square(expected))))
            z = individual.loc[individual.pop3 == row.population, "z"].to_numpy()
            self.assertAlmostEqual(row.rms_z, np.sqrt(np.mean(z ** 2)))
            self.assertEqual(row.n_snps, 22)
        self.assertTrue(result.rms_f4.is_monotonic_increasing)
        self.assertEqual(result.columns.tolist(), ["population", "rms_f4", "rms_z", "n_snps"])

    def test_cache_preserves_undefined_z_and_stable_ties_without_counts(self):
        rows = pd.DataFrame([
            ["O", "R1", "B", "T"], ["O", "R2", "B", "T"],
            ["O", "R1", "A", "T"], ["O", "R2", "A", "T"],
        ], columns=["pop1", "pop2", "pop3", "pop4"])
        stats = BlockStats(rows, None, np.ones(3), "f4",
                           est=np.array([0., 2., 0., -2.]),
                           variances=np.array([0., 1., 1., 0.]))
        result = admixpy.f4_proximity(
            F4BlockCache(stats), "T", ["B", "A"], ["R1", "R2"],
            outgroup="O", verbose=False,
        )
        self.assertEqual(result.population.tolist(), ["B", "A"])
        np.testing.assert_allclose(result.rms_f4, np.sqrt(2))
        self.assertTrue(np.isnan(result.rms_z.iloc[0]))
        self.assertTrue(np.isinf(result.rms_z.iloc[1]))
        self.assertTrue(result.n_snps.isna().all())

    def test_single_population_strings_and_forwarded_options(self):
        frame = pd.DataFrame(dict(pop3=["Candidate"], est=[-3.], z=[-2.], n=[42]))
        with patch("admixpy.fstats.f4", return_value=frame) as compute:
            result = admixpy.f4_proximity(
                "unused", "Target", "Candidate", "Reference",
                outgroup="O",
                comb=True, unique_only=True, covariance=True, apply_corr=False,
            )
        self.assertEqual(compute.call_args.args[2:4], (["Reference"], ["Candidate"]))
        self.assertTrue(compute.call_args.kwargs["covariance"])
        self.assertFalse(compute.call_args.kwargs["apply_corr"])
        self.assertEqual(result.iloc[0].tolist(), ["Candidate", 3., 2., 42])

    def test_invalid_inputs_fail_before_loading_data(self):
        cases = [
            ({"candidates": []}, "candidates must contain"),
            ({"refs": []}, "refs must contain"),
            ({"candidates": ["A", "A"]}, "Duplicate candidate"),
            ({"refs": ["R", "R"]}, "Duplicate reference"),
            ({"target": ["T", "T2"]}, "target must be"),
            ({"outgroup": ""}, "outgroup must be"),
            ({"comb": False}, "requires comb=True"),
        ]
        with patch("admixpy.fstats.f4") as compute:
            for changes, message in cases:
                options = dict(target="T", candidates=["A"], refs=["R"], outgroup="O")
                options.update(changes)
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                    admixpy.f4_proximity("unused", **options)
            compute.assert_not_called()

    def test_default_outgroup_uses_first_reference_without_mutating_input(self):
        frame = pd.DataFrame(dict(pop3=["A"], est=[3.], z=[2.], n=[42]))
        refs = ["O", "R"]
        with patch("admixpy.fstats.f4", return_value=frame) as compute:
            for options in ({}, {"outgroup": None}):
                result = admixpy.f4_proximity("unused", "T", ["A"], refs, **options)
                self.assertEqual(compute.call_args.args, ("unused", "O", ["R"], ["A"], "T"))
                self.assertEqual(result.iloc[0].tolist(), ["A", 3., 2., 42])
        self.assertEqual(refs, ["O", "R"])

    def test_default_outgroup_requires_remaining_unique_references(self):
        cases = [
            ([], "refs must contain"),
            (["O"], "at least two populations"),
            ("O", "at least two populations"),
            (["O", "O"], "Duplicate reference"),
        ]
        with patch("admixpy.fstats.f4") as compute:
            for refs, message in cases:
                with self.subTest(refs=refs), self.assertRaisesRegex(ValueError, message):
                    admixpy.f4_proximity("unused", "T", ["A"], refs)
            compute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
