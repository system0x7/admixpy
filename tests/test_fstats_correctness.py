import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from admixpy.fstats import (
    F2Blocks,
    _hudson_fst,
    _hudson_fst_components,
    afs_to_f2_blocks,
    f2,
    f2_from_geno,
    fst,
    mats_to_f2arr,
    qp3pop,
    qpdstat,
    read_f2,
    write_f2,
)
from admixpy.genotypes import AfData


class HudsonFstTests(unittest.TestCase):
    def test_apply_corr_false_uses_raw_hudson_denominator(self):
        p1, p2 = np.array([[0.25]]), np.array([[0.75]])
        c1 = c2 = np.array([[4.0]])
        got = _hudson_fst(p1, p2, c1, c2, [1], apply_corr=False, verbose=False)[0, 0, 0]
        raw = (0.25 - 0.75) ** 2
        expected = raw / (raw + 0.25 * 0.75 + 0.75 * 0.25)
        self.assertAlmostEqual(got, expected)

    def test_corrected_hudson_formula_is_unchanged_for_valid_counts(self):
        p1, p2 = np.array([[0.25]]), np.array([[0.75]])
        c1 = c2 = np.array([[4.0]])
        got = _hudson_fst(p1, p2, c1, c2, [1], apply_corr=True, verbose=False)[0, 0, 0]
        raw = (0.25 - 0.75) ** 2
        correction = 2 * (0.25 * 0.75 / 3)
        expected = (raw - correction) / (raw + 0.25 * 0.75 + 0.75 * 0.25)
        self.assertAlmostEqual(got, expected)

    def test_singletons_are_excluded_when_correcting(self):
        p1, p2 = np.array([[0.0]]), np.array([[0.5]])
        c1, c2 = np.array([[1.0]]), np.array([[4.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fst, counts, _, _ = _hudson_fst_components(
                p1, p2, c1, c2, [1], apply_corr=True, verbose=False
            )
            f2arr = mats_to_f2arr(p1, p2, c1, c2, [1], apply_corr=True, verbose=False)
        self.assertTrue(np.isnan(fst[0, 0, 0]))
        self.assertEqual(counts[0, 0, 0], 0)
        self.assertTrue(np.isnan(f2arr[0, 0, 0]))
        self.assertTrue(any("at least two" in str(item.message) for item in caught))

    def test_singletons_are_explicitly_biased_when_not_correcting(self):
        p1, p2 = np.array([[0.0]]), np.array([[0.5]])
        c1, c2 = np.array([[1.0]]), np.array([[4.0]])
        with self.assertWarnsRegex(RuntimeWarning, "sampling bias"):
            got = _hudson_fst(p1, p2, c1, c2, [1], apply_corr=False, verbose=False)[0, 0, 0]
        self.assertAlmostEqual(got, 0.5)


class PairCountTests(unittest.TestCase):
    def test_f2_resampling_modes_are_explicit(self):
        blocks = F2Blocks(
            data=np.array([[[0.0, 1.0]]]),
            pops1=["A"],
            pops2=["B"],
            block_lengths=np.array([10, 10]),
            stat="f2",
            snp_counts=np.array([[[10.0, 1.0]]]),
        )
        pairwise = f2(blocks, pop1="A", pop2="B")
        nominal = f2(blocks, pop1="A", pop2="B", resampling="nominal_blocks")
        self.assertAlmostEqual(nominal.loc[0, "est"], 0.5)
        self.assertNotIn("n", nominal.columns)
        self.assertAlmostEqual(pairwise.loc[0, "est"], 1 / 11)
        self.assertEqual(pairwise.loc[0, "n"], 11)

    def test_fst_resampling_and_aggregation_are_separate(self):
        blocks = F2Blocks(
            data=np.array([[[0.1, 0.9]]]),
            pops1=["A"],
            pops2=["B"],
            block_lengths=np.array([10, 10]),
            stat="fst",
            snp_counts=np.array([[[10.0, 1.0]]]),
            fst_num=np.array([[[1.0, 9.0]]]),
            fst_den=np.array([[[10.0, 10.0]]]),
        )
        pairwise = fst(blocks, pop1="A", pop2="B")
        nominal = fst(blocks, pop1="A", pop2="B", resampling="nominal_blocks")
        pooled = fst(
            blocks,
            pop1="A",
            pop2="B",
            resampling="pairwise_counts",
            fst_aggregation="pooled_components",
        )
        self.assertAlmostEqual(nominal.loc[0, "est"], 0.5)
        self.assertAlmostEqual(pairwise.loc[0, "est"], 1.9 / 11)
        self.assertAlmostEqual(pooled.loc[0, "est"], 0.5)
        self.assertNotIn("n", nominal.columns)
        self.assertEqual(pairwise.loc[0, "n"], 11)

    def test_pooled_fst_zero_denominator_is_nan_not_an_exception(self):
        blocks = F2Blocks(
            data=np.array([[[np.nan, np.nan]]]),
            pops1=["A"],
            pops2=["B"],
            block_lengths=np.array([5, 5]),
            stat="fst",
            snp_counts=np.array([[[5.0, 5.0]]]),
            fst_num=np.zeros((1, 1, 2)),
            fst_den=np.zeros((1, 1, 2)),
        )
        out = fst(
            blocks,
            pop1="A",
            pop2="B",
            resampling="pairwise_counts",
            fst_aggregation="pooled_components",
        )
        self.assertTrue(np.isnan(out.loc[0, "est"]))
        self.assertTrue(np.isnan(out.loc[0, "se"]))

    def test_default_poly_filter_uses_different_f2_and_fst_snp_sets(self):
        afs = pd.DataFrame({"A": [0.0, 0.25], "B": [0.0, 0.75]}, index=["s1", "s2"])
        counts = pd.DataFrame(4.0, index=afs.index, columns=afs.columns)
        snps = pd.DataFrame(
            {
                "SNP": afs.index,
                "CHR": [1, 1],
                "cm": [0.0, 0.01],
                "POS": [1, 2],
                "A1": ["A", "A"],
                "A2": ["G", "G"],
            }
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No usable map or base positions found")
            out = afs_to_f2_blocks(
                AfData(afs, counts, snps),
                pops1=["A"],
                pops2=["B"],
                stats=("f2", "fst"),
                poly_only=("f2",),
                verbose=False,
            )
            comparable = afs_to_f2_blocks(
                AfData(afs, counts, snps),
                pops1=["A"],
                pops2=["B"],
                stats=("f2", "fst"),
                poly_only=True,
                verbose=False,
            )
        self.assertEqual(out["f2_blocks"].snp_counts[0, 0, 0], 1)
        self.assertEqual(out["fst_blocks"].snp_counts[0, 0, 0], 2)
        np.testing.assert_array_equal(
            comparable["f2_blocks"].snp_counts,
            comparable["fst_blocks"].snp_counts,
        )


class CacheTests(unittest.TestCase):
    def test_cache_round_trip_preserves_counts_and_fst_components(self):
        blocks = F2Blocks(
            data=np.array([[[0.1, 0.2]]]),
            pops1=["A"],
            pops2=["B"],
            block_lengths=np.array([10, 20]),
            stat="fst",
            snp_counts=np.array([[[7.0, 11.0]]]),
            fst_num=np.array([[[0.7, 2.2]]]),
            fst_den=np.array([[[7.0, 11.0]]]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_f2(blocks, tmp)
            got = read_f2(tmp, pops=["A"], pops2=["B"], type="fst")
        np.testing.assert_array_equal(got.data, blocks.data)
        np.testing.assert_array_equal(got.snp_counts, blocks.snp_counts)
        np.testing.assert_array_equal(got.fst_num, blocks.fst_num)
        np.testing.assert_array_equal(got.fst_den, blocks.fst_den)

    def test_cache_without_real_counts_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "block_lengths_f2.npy", np.array([10, 20]))
            (root / "A").mkdir()
            np.savez_compressed(root / "A" / "B_f2.npz", est=np.array([0.1, 0.2]), counts=np.ones(2))
            with self.assertRaisesRegex(ValueError, "missing required per-pair SNP counts"):
                read_f2(root, pops=["A"], pops2=["B"], type="f2")

    def test_write_cache_without_real_counts_is_rejected(self):
        blocks = F2Blocks(
            data=np.array([[[0.1, 0.2]]]),
            pops1=["A"],
            pops2=["B"],
            block_lengths=np.array([10, 20]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "requires real per-pair SNP counts"):
                write_f2(blocks, tmp)


class CompositePairCountTests(unittest.TestCase):
    @staticmethod
    def _square_blocks(counts_ab):
        pops = ["A", "B", "C"]
        data = np.zeros((3, 3, 3), dtype=float)
        counts = np.full_like(data, 4.0)

        def pair(a, b, values, pair_counts=(4, 4, 4)):
            i, j = pops.index(a), pops.index(b)
            data[i, j] = data[j, i] = values
            counts[i, j] = counts[j, i] = pair_counts

        pair("A", "B", [0.0, 1.0, 0.0], counts_ab)
        pair("A", "C", [1.0, 1.0, 1.0])
        pair("B", "C", [0.5, 0.5, 0.5])
        return F2Blocks(data, pops, pops, np.array([4, 4, 4]), "f2", counts)

    def test_cached_f3_pairwise_counts_changes_thin_pair_weighting(self):
        blocks = self._square_blocks((4, 1, 2))
        nominal = qp3pop(blocks, "A", "B", "C", resampling="nominal_blocks", verbose=False)
        pairwise = qp3pop(
            blocks,
            "A",
            "B",
            "C",
            verbose=False,
        )
        self.assertAlmostEqual(nominal.loc[0, "est"], 5 / 12)
        self.assertAlmostEqual(pairwise.loc[0, "est"], 9 / 28)
        self.assertTrue(np.isfinite(pairwise.loc[0, "se"]))

    def test_cached_f3_pairwise_counts_matches_nominal_when_complete(self):
        blocks = self._square_blocks((4, 4, 4))
        nominal = qp3pop(blocks, "A", "B", "C", resampling="nominal_blocks", verbose=False)
        pairwise = qp3pop(
            blocks,
            "A",
            "B",
            "C",
            verbose=False,
        )
        self.assertAlmostEqual(nominal.loc[0, "est"], pairwise.loc[0, "est"])
        self.assertAlmostEqual(nominal.loc[0, "se"], pairwise.loc[0, "se"])

    @staticmethod
    def _f4_blocks(counts_ad):
        pops1, pops2 = ["A", "B"], ["C", "D"]
        data = np.zeros((2, 2, 3), dtype=float)
        counts = np.full_like(data, 4.0)
        values = {
            ("A", "D"): ([0.0, 1.0, 0.0], counts_ad),
            ("B", "C"): ([1.0, 1.0, 1.0], (4, 4, 4)),
            ("A", "C"): ([0.2, 0.2, 0.2], (4, 4, 4)),
            ("B", "D"): ([0.4, 0.4, 0.4], (4, 4, 4)),
        }
        for (a, b), (pair_values, pair_counts) in values.items():
            i, j = pops1.index(a), pops2.index(b)
            data[i, j] = pair_values
            counts[i, j] = pair_counts
        return F2Blocks(data, pops1, pops2, np.array([4, 4, 4]), "f2", counts)

    def test_cached_f4_pairwise_counts_handles_thin_blocks(self):
        blocks = self._f4_blocks((4, 1, 2))
        nominal = qpdstat(
            blocks, "A", "B", "C", "D", resampling="nominal_blocks", verbose=False
        )
        pairwise = qpdstat(
            blocks,
            "A",
            "B",
            "C",
            "D",
            verbose=False,
        )
        self.assertAlmostEqual(nominal.loc[0, "est"], 11 / 30)
        self.assertAlmostEqual(pairwise.loc[0, "est"], 19 / 70)
        self.assertTrue(np.isfinite(pairwise.loc[0, "se"]))

    def test_cached_f4_pairwise_counts_handles_an_absent_pair_block(self):
        blocks = self._f4_blocks((4, 0, 2))
        blocks.data[0, 1, 1] = np.nan
        pairwise = qpdstat(
            blocks,
            "A",
            "B",
            "C",
            "D",
            verbose=False,
        )
        self.assertAlmostEqual(pairwise.loc[0, "est"], 0.2)
        self.assertTrue(np.isfinite(pairwise.loc[0, "se"]))

    def test_cached_f4_pairwise_counts_matches_nominal_when_complete(self):
        blocks = self._f4_blocks((4, 4, 4))
        nominal = qpdstat(
            blocks, "A", "B", "C", "D", resampling="nominal_blocks", verbose=False
        )
        pairwise = qpdstat(
            blocks,
            "A",
            "B",
            "C",
            "D",
            verbose=False,
        )
        self.assertAlmostEqual(nominal.loc[0, "est"], pairwise.loc[0, "est"])
        self.assertAlmostEqual(nominal.loc[0, "se"], pairwise.loc[0, "se"])


class RawF4Tests(unittest.TestCase):
    @staticmethod
    def _write_complete_eigenstrat(root: Path) -> Path:
        pref = root / "complete"
        pref.with_suffix(".ind").write_text(
            "A1 M A\nA2 F A\nB1 M B\nB2 F B\n"
            "C1 M C\nC2 F C\nD1 M D\nD2 F D\n"
        )
        positions = [0.00, 0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.10, 0.11, 0.12, 0.13]
        pref.with_suffix(".snp").write_text(
            "".join(f"s{i} 1 {cm:.2f} {i * 100} A G\n" for i, cm in enumerate(positions, 1))
        )
        pref.with_suffix(".geno").write_text(
            "00111122\n" * 4 + "00221100\n" * 4 + "11002200\n" * 4
        )
        return pref

    def test_raw_f4_matches_complete_pairwise_cache_and_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pref = self._write_complete_eigenstrat(root)
            raw = qpdstat(
                pref,
                "A",
                "B",
                "C",
                "D",
                blgsize=0.04,
                adjust_pseudohaploid=False,
                verbose=False,
            )
            blocks = f2_from_geno(
                pref,
                pops=["A", "B"],
                pops2=["C", "D"],
                blgsize=0.04,
                maxmiss=0,
                remove_na=False,
                adjust_pseudohaploid=False,
                verbose=False,
            )
            cached = qpdstat(
                blocks,
                "A",
                "B",
                "C",
                "D",
                resampling="pairwise_counts",
                verbose=False,
            )
            cache_dir = root / "f2cache"
            write_f2(blocks, cache_dir)
            round_trip = qpdstat(
                cache_dir,
                "A",
                "B",
                "C",
                "D",
                resampling="pairwise_counts",
                verbose=False,
            )
        self.assertAlmostEqual(raw.loc[0, "est"], cached.loc[0, "est"])
        self.assertAlmostEqual(raw.loc[0, "se"], cached.loc[0, "se"])
        self.assertAlmostEqual(cached.loc[0, "est"], round_trip.loc[0, "est"])
        self.assertAlmostEqual(cached.loc[0, "se"], round_trip.loc[0, "se"])

    def test_raw_and_cached_pairwise_f4_are_finite_with_an_absent_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pref = self._write_complete_eigenstrat(root)
            pref.with_suffix(".geno").write_text(
                "00111122\n" * 4 + "00221199\n" * 4 + "11002200\n" * 4
            )
            raw = qpdstat(
                pref,
                "A",
                "B",
                "C",
                "D",
                blgsize=0.04,
                adjust_pseudohaploid=False,
                verbose=False,
            )
            blocks = f2_from_geno(
                pref,
                pops=["A", "B"],
                pops2=["C", "D"],
                blgsize=0.04,
                maxmiss=1,
                remove_na=False,
                adjust_pseudohaploid=False,
                verbose=False,
            )
            cached = qpdstat(
                blocks,
                "A",
                "B",
                "C",
                "D",
                resampling="pairwise_counts",
                verbose=False,
            )
        self.assertTrue(np.isfinite(raw.loc[0, "est"]))
        self.assertTrue(np.isfinite(raw.loc[0, "se"]))
        self.assertTrue(np.isfinite(cached.loc[0, "est"]))
        self.assertTrue(np.isfinite(cached.loc[0, "se"]))
        self.assertNotAlmostEqual(raw.loc[0, "est"], cached.loc[0, "est"])


class DirectF3Tests(unittest.TestCase):
    @staticmethod
    def _write_singleton_source_eigenstrat(root: Path) -> Path:
        pref = root / "singleton_source"
        pref.with_suffix(".ind").write_text(
            "A1 M A\nA2 F A\nB1 M B\nB2 F B\nC1 U C\n"
        )
        positions = [
            0.00,
            0.01,
            0.02,
            0.03,
            0.06,
            0.07,
            0.08,
            0.09,
            0.12,
            0.13,
            0.14,
            0.15,
        ]
        pref.with_suffix(".snp").write_text(
            "".join(f"s{i} 1 {cm:.2f} {i * 100} A G\n" for i, cm in enumerate(positions, 1))
        )
        pref.with_suffix(".geno").write_text(
            "\n".join(
                [
                    "02002",
                    "02200",
                    "20020",
                    "20222",
                    "00222",
                    "22000",
                    "00002",
                    "22220",
                    "02009",
                    "20029",
                    "00209",
                    "22029",
                ]
            )
            + "\n"
        )
        return pref

    def test_singleton_source_is_finite_and_matches_corrected_repeated_f4(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = self._write_singleton_source_eigenstrat(Path(tmp))
            f3 = qp3pop(pref, "A", "B", "C", blgsize=0.05, verbose=False)
            swapped = qp3pop(pref, "A", "C", "B", blgsize=0.05, verbose=False)
            repeated_f4 = qpdstat(
                pref,
                "A",
                "B",
                "A",
                "C",
                blgsize=0.05,
                apply_corr=True,
                verbose=False,
            )
        self.assertTrue(np.isfinite(f3.loc[0, "est"]))
        self.assertTrue(np.isfinite(f3.loc[0, "se"]))
        self.assertEqual(f3.loc[0, "n"], 8)
        self.assertAlmostEqual(f3.loc[0, "est"], swapped.loc[0, "est"])
        self.assertAlmostEqual(f3.loc[0, "se"], swapped.loc[0, "se"])
        self.assertAlmostEqual(f3.loc[0, "est"], repeated_f4.loc[0, "est"])
        self.assertAlmostEqual(f3.loc[0, "se"], repeated_f4.loc[0, "se"])

    def test_singleton_target_remains_unestimable_when_correcting(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = self._write_singleton_source_eigenstrat(Path(tmp))
            with self.assertWarnsRegex(RuntimeWarning, "at least two"):
                out = qp3pop(pref, "C", "A", "B", blgsize=0.05, verbose=False)
        self.assertTrue(np.isnan(out.loc[0, "est"]))
        self.assertTrue(np.isnan(out.loc[0, "se"]))
        self.assertEqual(out.loc[0, "n"], 0)

    def test_complete_direct_f3_matches_f2_derived_f3(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = RawF4Tests._write_complete_eigenstrat(Path(tmp))
            direct = qp3pop(
                pref,
                "A",
                "B",
                "C",
                blgsize=0.04,
                allsnps=False,
                poly_only=False,
                maxmiss=0,
                adjust_pseudohaploid=False,
                verbose=False,
            )
            direct_nominal = qp3pop(
                pref,
                "A",
                "B",
                "C",
                blgsize=0.04,
                allsnps=False,
                poly_only=False,
                maxmiss=0,
                adjust_pseudohaploid=False,
                resampling="nominal_blocks",
                verbose=False,
            )
            blocks = f2_from_geno(
                pref,
                pops=["A", "B", "C"],
                blgsize=0.04,
                poly_only=False,
                maxmiss=0,
                remove_na=False,
                adjust_pseudohaploid=False,
                verbose=False,
            )
            cached = qp3pop(blocks, "A", "B", "C", verbose=False)
            cached_nominal = qp3pop(
                blocks,
                "A",
                "B",
                "C",
                resampling="nominal_blocks",
                verbose=False,
            )
        self.assertAlmostEqual(direct.loc[0, "est"], cached.loc[0, "est"])
        self.assertAlmostEqual(direct.loc[0, "se"], cached.loc[0, "se"])
        self.assertAlmostEqual(direct_nominal.loc[0, "est"], cached_nominal.loc[0, "est"])
        self.assertAlmostEqual(direct_nominal.loc[0, "se"], cached_nominal.loc[0, "se"])
        self.assertNotIn("n", direct_nominal.columns)


if __name__ == "__main__":
    unittest.main()
