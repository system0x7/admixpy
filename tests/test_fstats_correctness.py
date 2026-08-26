import inspect
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd

from admixpy.fstats import (
    BlockStats,
    F2Blocks,
    F4BlockCache,
    QpWaveStats,
    _qinv_from_cov,
    _hudson_fst,
    _hudson_fst_components,
    afs_to_f2_blocks,
    f2,
    f2_from_geno,
    f3_stats_from_geno,
    fst,
    f4_stats,
    f4_stats_from_geno,
    f4_model_cache,
    _f3_direct_blocks_from_afs,
    _influence_covariance,
    jackknife_cov,
    mats_to_f2arr,
    qp3pop,
    qpadm,
    qpdstat,
    qpwave,
    qpwave_f4stats,
    read_f2,
    stats_to_loo,
    write_f2,
)
from admixpy.genotypes import (
    AfData,
    _detect_pseudohaploid,
    _parse_geno_header_with_kind,
    discard_from_aftable,
    get_block_lengths,
    iter_geno_to_afs,
)


def _reference_influence_covariance(influence, contributes):
    influence = np.asarray(influence, float)
    contributes = np.asarray(contributes, bool)
    cov = np.full((len(influence), len(influence)), np.nan)
    for i in range(len(influence)):
        for j in range(i, len(influence)):
            keep = (
                contributes[i]
                & contributes[j]
                & np.isfinite(influence[i])
                & np.isfinite(influence[j])
            )
            if keep.sum() >= 2:
                cov[i, j] = cov[j, i] = np.mean(influence[i, keep] * influence[j, keep])
    return cov


def _reference_jackknife_covariance(loo, block_lengths, estimates, contributes):
    loo = np.asarray(loo, float)
    block_lengths = np.asarray(block_lengths, float)
    estimates = np.asarray(estimates, float)
    contributes = np.asarray(contributes, bool)
    cov = np.full((len(loo), len(loo)), np.nan)
    for i in range(len(loo)):
        for j in range(i, len(loo)):
            keep = (
                contributes[i]
                & contributes[j]
                & np.isfinite(loo[i])
                & np.isfinite(loo[j])
            )
            if keep.sum() >= 2:
                h = block_lengths[keep].sum() / block_lengths[keep]
                cov[i, j] = cov[j, i] = np.mean(
                    (estimates[i] - loo[i, keep])
                    * (estimates[j] - loo[j, keep])
                    * (h - 1)
                )
    return cov


class LowRiskOptimizationEquivalenceTests(unittest.TestCase):
    def test_vectorized_influence_covariance_matches_pairwise_reference(self):
        rng = np.random.default_rng(123)
        influence = rng.normal(size=(12, 17))
        contributes = rng.random(influence.shape) > 0.25
        influence[rng.random(influence.shape) < 0.1] = np.nan
        expected = _reference_influence_covariance(influence, contributes)
        actual = _influence_covariance(influence, contributes)
        np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-13, atol=1e-13)

    def test_vectorized_jackknife_covariance_matches_pairwise_reference(self):
        rng = np.random.default_rng(456)
        loo = rng.normal(size=(10, 19))
        contributes = rng.random(loo.shape) > 0.2
        loo[rng.random(loo.shape) < 0.1] = np.nan
        block_lengths = rng.integers(1, 20, size=loo.shape[1]).astype(float)
        estimates = rng.normal(size=loo.shape[0])
        expected = _reference_jackknife_covariance(
            loo,
            block_lengths,
            estimates,
            contributes,
        )
        actual, returned_estimates = jackknife_cov(
            loo,
            block_lengths,
            est=estimates,
            contributes=contributes,
        )
        np.testing.assert_array_equal(returned_estimates, estimates)
        np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-13, atol=1e-13)

    def test_vectorized_pseudohaploid_detection_preserves_rules(self):
        geno = np.array(
            [
                [0.0, 0.0, np.nan, 0.0],
                [2.0, 1.0, np.nan, 2.0],
                [np.nan, 2.0, np.nan, 0.0],
            ]
        )
        actual = _detect_pseudohaploid(geno, np.array([0, 0, 1, -1]), True)
        np.testing.assert_array_equal(actual, [1.0, 2.0, 2.0, 2.0])

    def test_numeric_and_prefixed_chromosomes_produce_same_blocks(self):
        base = pd.DataFrame(
            {
                "CHR": [1, 1, 1, 2, 2],
                "cm": [0.0, 0.01, 0.07, 0.0, 0.08],
                "POS": [1, 2, 3, 1, 2],
            }
        )
        prefixed = base.copy()
        prefixed["CHR"] = "chr" + prefixed["CHR"].astype(str)
        np.testing.assert_array_equal(
            get_block_lengths(base),
            get_block_lengths(prefixed),
        )

    def test_prefixed_autosomes_still_pass_filtering(self):
        snp_ids = ["s1", "s22", "sx"]
        afs = pd.DataFrame({"A": 0.25, "B": 0.75}, index=snp_ids)
        counts = pd.DataFrame(4.0, index=snp_ids, columns=afs.columns)
        snps = pd.DataFrame(
            {
                "SNP": snp_ids,
                "CHR": ["chr1", "chr22", "chrX"],
                "cm": [0.0, 0.0, 0.0],
                "POS": [1, 2, 3],
                "A1": "A",
                "A2": "G",
            }
        )
        with self.assertWarnsRegex(UserWarning, "non-numeric chromosomes"):
            filtered = discard_from_aftable(AfData(afs, counts, snps))
        self.assertEqual(filtered.snpfile["SNP"].tolist(), ["s1", "s22"])

    def test_vectorized_mutation_classification_handles_both_orientations(self):
        pairs = [("A", "G"), ("G", "A"), ("C", "T"), ("T", "C"), ("A", "C"), ("C", "A")]
        snp_ids = [f"s{i}" for i in range(len(pairs))]
        afs = pd.DataFrame({"A": 0.25, "B": 0.75}, index=snp_ids)
        counts = pd.DataFrame(4.0, index=snp_ids, columns=afs.columns)
        snps = pd.DataFrame(
            {
                "SNP": snp_ids,
                "CHR": 1,
                "cm": np.arange(len(pairs), dtype=float),
                "POS": np.arange(len(pairs)),
                "A1": [pair[0] for pair in pairs],
                "A2": [pair[1] for pair in pairs],
            }
        )
        filtered = discard_from_aftable(
            AfData(afs, counts, snps),
            transitions=False,
            transversions=True,
        )
        self.assertEqual(filtered.snpfile["SNP"].tolist(), ["s4", "s5"])


class DirectGenotypeDefaultsTests(unittest.TestCase):
    def test_binary_header_parser_reads_only_48_bytes(self):
        opened = mock_open(read_data=b"TGENO 2 4 abc def".ljust(48, b"\0"))
        with patch.object(Path, "open", opened):
            self.assertEqual(_parse_geno_header_with_kind("large.tgeno"), ("TGENO", 2, 4))
        opened.return_value.read.assert_called_once_with(48)

    def test_materialized_mode_is_default(self):
        self.assertFalse(inspect.signature(f3_stats_from_geno).parameters["stream"].default)
        self.assertFalse(inspect.signature(f4_stats_from_geno).parameters["stream"].default)

    def test_streaming_chunk_default_is_250000(self):
        for func in (f3_stats_from_geno, f4_stats_from_geno, iter_geno_to_afs):
            self.assertEqual(inspect.signature(func).parameters["chunk_size"].default, 250_000)


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

    def test_singleton_warning_names_affected_population(self):
        p1, p2 = np.array([[0.0]]), np.array([[0.5]])
        c1, c2 = np.array([[1.0]]), np.array([[4.0]])
        with self.assertWarnsRegex(RuntimeWarning, "Affected populations: 'A'"):
            mats_to_f2arr(
                p1,
                p2,
                c1,
                c2,
                [1],
                apply_corr=True,
                verbose=False,
                population_labels=(["A"], ["B"]),
            )

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

    def test_direct_f4_diagonal_variance_matches_full_covariance(self):
        combos = pd.DataFrame(
            [
                {"pop1": "A", "pop2": "B", "pop3": "C", "pop4": "D"},
                {"pop1": "A", "pop2": "C", "pop3": "B", "pop4": "D"},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            pref = self._write_complete_eigenstrat(Path(tmp))
            full = qpdstat(
                pref,
                combos,
                unique_only=False,
                blgsize=0.04,
                adjust_pseudohaploid=False,
                verbose=False,
            )
            diagonal = qpdstat(
                pref,
                combos,
                unique_only=False,
                blgsize=0.04,
                adjust_pseudohaploid=False,
                covariance=False,
                verbose=False,
            )
            stats = f4_stats_from_geno(
                pref,
                combos,
                blgsize=0.04,
                adjust_pseudohaploid=False,
                covariance=False,
                verbose=False,
            )
        pd.testing.assert_frame_equal(diagonal, full)
        self.assertIsNone(stats.cov)
        self.assertTrue(np.isfinite(stats.variances).all())
        np.testing.assert_allclose(stats.se, full["se"])

    def test_f2_backed_f4_diagonal_variance_matches_full_covariance(self):
        combos = pd.DataFrame(
            [
                {"pop1": "A", "pop2": "B", "pop3": "C", "pop4": "D"},
                {"pop1": "A", "pop2": "C", "pop3": "B", "pop4": "D"},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            pref = self._write_complete_eigenstrat(Path(tmp))
            full = qpdstat(
                pref,
                combos,
                unique_only=False,
                allsnps=False,
                blgsize=0.04,
                adjust_pseudohaploid=False,
                verbose=False,
            )
            diagonal = qpdstat(
                pref,
                combos,
                unique_only=False,
                allsnps=False,
                blgsize=0.04,
                adjust_pseudohaploid=False,
                covariance=False,
                verbose=False,
            )
            stats = f4_stats(
                pref,
                combos,
                unique_only=False,
                allsnps=False,
                blgsize=0.04,
                adjust_pseudohaploid=False,
                covariance=False,
                verbose=False,
            )
        pd.testing.assert_frame_equal(diagonal, full)
        self.assertIsNone(stats.cov)
        self.assertTrue(np.isfinite(stats.variances).all())
        self.assertTrue(np.isfinite(stats.influence).all())
        np.testing.assert_allclose(stats.se, full["se"])

    def test_nominal_block_f4_diagonal_variance_matches_full_covariance(self):
        combos = pd.DataFrame(
            [
                {"pop1": "A", "pop2": "B", "pop3": "C", "pop4": "D"},
                {"pop1": "A", "pop2": "C", "pop3": "B", "pop4": "D"},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            pref = self._write_complete_eigenstrat(Path(tmp))
            full = f4_stats(
                pref,
                combos,
                unique_only=False,
                allsnps=False,
                resampling="nominal_blocks",
                blgsize=0.04,
                adjust_pseudohaploid=False,
                verbose=False,
            )
            diagonal = f4_stats(
                pref,
                combos,
                unique_only=False,
                allsnps=False,
                resampling="nominal_blocks",
                covariance=False,
                blgsize=0.04,
                adjust_pseudohaploid=False,
                verbose=False,
            )
        self.assertIsNone(diagonal.cov)
        self.assertTrue(np.isfinite(diagonal.variances).all())
        np.testing.assert_allclose(diagonal.est, full.est)
        np.testing.assert_allclose(diagonal.se, full.se)

    def test_f4_block_cache_preserves_diagonal_variances(self):
        combos = pd.DataFrame(
            [
                {"pop1": "A", "pop2": "B", "pop3": "C", "pop4": "D"},
                {"pop1": "A", "pop2": "C", "pop3": "B", "pop4": "D"},
            ]
        )
        cached_stats = BlockStats(
            rows=combos,
            blocks=np.zeros((2, 3)),
            block_lengths=np.ones(3),
            stat="f4",
            est=np.array([0.1, 0.2]),
            cov=np.array([[4.0, 1.0], [1.0, 9.0]]),
        )
        requested = combos.iloc[::-1].reset_index(drop=True)
        from_covariance = f4_stats(
            F4BlockCache(cached_stats),
            requested,
            unique_only=False,
            covariance=False,
            verbose=False,
        )
        self.assertIsNone(from_covariance.cov)
        np.testing.assert_allclose(from_covariance.variances, [9.0, 4.0])
        np.testing.assert_allclose(from_covariance.se, [3.0, 2.0])

        cached_stats.cov = None
        cached_stats.variances = np.array([16.0, 25.0])
        from_variances = f4_stats(
            F4BlockCache(cached_stats),
            requested,
            unique_only=False,
            covariance=False,
            verbose=False,
        )
        self.assertIsNone(from_variances.cov)
        np.testing.assert_allclose(from_variances.variances, [25.0, 16.0])
        np.testing.assert_allclose(from_variances.se, [5.0, 4.0])

    def test_covariance_dependent_workflows_reject_diagonal_only_mode(self):
        combos = pd.DataFrame(
            [
                {"pop1": "A", "pop2": "B", "pop3": "C", "pop4": "D"},
            ]
        )
        diagonal_stats = BlockStats(
            rows=combos,
            blocks=np.zeros((1, 3)),
            block_lengths=np.ones(3),
            stat="f4",
            est=np.array([0.1]),
            variances=np.array([4.0]),
        )
        with self.assertRaisesRegex(ValueError, "does not contain a full covariance"):
            f4_stats(
                F4BlockCache(diagonal_stats),
                combos,
                unique_only=False,
                covariance=True,
                verbose=False,
            )
        with self.assertRaisesRegex(ValueError, "qpWave and qpAdm require covariance=True"):
            qpwave_f4stats(
                None,
                left=["A", "B"],
                right=["C", "D"],
                covariance=False,
                verbose=False,
            )
        with self.assertRaisesRegex(ValueError, "f4_model_cache requires covariance=True"):
            f4_model_cache(
                None,
                pd.DataFrame([{"left": ["A", "B"], "right": ["C", "D"]}]),
                covariance=False,
                verbose=False,
            )

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

    def test_streamed_and_materialized_direct_f4_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = self._write_complete_eigenstrat(Path(tmp))
            streamed = qpdstat(
                pref, "A", "B", "C", "D", blgsize=0.04, chunk_size=3,
                adjust_pseudohaploid=False, stream=True, verbose=False,
            )
            materialized = qpdstat(
                pref, "A", "B", "C", "D", blgsize=0.04, stream=False,
                adjust_pseudohaploid=False, verbose=False,
            )
        self.assertEqual(streamed.loc[0, "n"], 12)
        pd.testing.assert_frame_equal(streamed, materialized)


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

    @staticmethod
    def _pack_high_bit_codes(codes, record_len):
        raw = bytearray(record_len)
        for i, code in enumerate(codes):
            raw[i // 4] |= int(code) << (6 - 2 * (i % 4))
        return bytes(raw)

    @staticmethod
    def _write_binary_equivalents(eigen_pref: Path, root: Path) -> list[Path]:
        lines = eigen_pref.with_suffix(".geno").read_text().splitlines()
        geno = np.array(
            [[np.nan if c == "9" else int(c) for c in row] for row in lines],
            dtype=float,
        )
        ind_lines = eigen_pref.with_suffix(".ind").read_text().splitlines()
        snp_rows = [line.split() for line in eigen_pref.with_suffix(".snp").read_text().splitlines()]
        nsnp, nind = geno.shape

        packed = root / "equiv_packed"
        packed.with_suffix(".ind").write_text(eigen_pref.with_suffix(".ind").read_text())
        packed.with_suffix(".snp").write_text(eigen_pref.with_suffix(".snp").read_text())
        record_len = max(48, (nind + 3) // 4)
        header = f"GENO {nind} {nsnp} 0 0".encode().ljust(record_len, b"\0")
        payload = [header]
        for row in geno:
            codes = [3 if np.isnan(g) else int(g) for g in row]
            payload.append(DirectF3Tests._pack_high_bit_codes(codes, record_len))
        packed.with_suffix(".geno").write_bytes(b"".join(payload))

        tgeno = root / "equiv_tgeno"
        tgeno.with_suffix(".ind").write_text(eigen_pref.with_suffix(".ind").read_text())
        tgeno.with_suffix(".snp").write_text(eigen_pref.with_suffix(".snp").read_text())
        record_len = max(48, (nsnp + 3) // 4)
        header = f"TGENO {nind} {nsnp} 0 0".encode().ljust(48, b"\0")
        payload = [header]
        for col in geno.T:
            codes = [3 if np.isnan(g) else int(g) for g in col]
            payload.append(DirectF3Tests._pack_high_bit_codes(codes, record_len))
        tgeno.with_suffix(".tgeno").write_bytes(b"".join(payload))

        plink = root / "equiv_plink"
        fam = []
        for line in ind_lines:
            iid, sex, pop = line.split()
            fam.append(f"{pop} {iid} 0 0 {1 if sex == 'M' else 2} -9\n")
        plink.with_suffix(".fam").write_text("".join(fam))
        plink.with_suffix(".bim").write_text(
            "".join(
                f"{chrom} {snp} {cm} {pos} {a1} {a2}\n"
                for snp, chrom, cm, pos, a1, a2 in snp_rows
            )
        )
        bed = bytearray(b"\x6c\x1b\x01")
        plink_code = {0: 3, 1: 2, 2: 0}
        bytes_per_snp = (nind + 3) // 4
        for row in geno:
            raw = bytearray(bytes_per_snp)
            for i, g in enumerate(row):
                code = 1 if np.isnan(g) else plink_code[int(g)]
                raw[i // 4] |= code << (2 * (i % 4))
            bed.extend(raw)
        plink.with_suffix(".bed").write_bytes(bytes(bed))
        return [packed, tgeno, plink]

    def test_singleton_source_is_finite_and_matches_corrected_repeated_f4(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = self._write_singleton_source_eigenstrat(Path(tmp))
            f3 = qp3pop(pref, "A", "B", "C", blgsize=0.05, outgroupmode=True, verbose=False)
            swapped = qp3pop(pref, "A", "C", "B", blgsize=0.05, outgroupmode=True, verbose=False)
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

    def test_repeated_source_matches_corrected_f2_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = RawF4Tests._write_complete_eigenstrat(Path(tmp))
            repeated_f3 = qp3pop(
                pref, "A", "B", "B", blgsize=0.04, outgroupmode=True,
                adjust_pseudohaploid=False, verbose=False,
            )
            repeated_f4 = qpdstat(
                pref, "A", "B", "A", "B", blgsize=0.04,
                adjust_pseudohaploid=False, verbose=False,
            )
        self.assertAlmostEqual(repeated_f3.loc[0, "est"], repeated_f4.loc[0, "est"])
        self.assertAlmostEqual(repeated_f3.loc[0, "se"], repeated_f4.loc[0, "se"])

    def test_target_equal_to_source_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = RawF4Tests._write_complete_eigenstrat(Path(tmp))
            out = qp3pop(
                pref, "A", "A", "C", blgsize=0.04, outgroupmode=True,
                adjust_pseudohaploid=False, verbose=False,
            )
        self.assertAlmostEqual(out.loc[0, "est"], 0.0)
        self.assertAlmostEqual(out.loc[0, "se"], 0.0)

    def test_default_normalizes_by_target_heterozygosity(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = RawF4Tests._write_complete_eigenstrat(Path(tmp))
            normalized = qp3pop(
                pref,
                "A",
                "B",
                "C",
                blgsize=0.04,
                adjust_pseudohaploid=False,
                verbose=False,
            )
            raw = qp3pop(
                pref,
                "A",
                "B",
                "C",
                blgsize=0.04,
                outgroupmode=True,
                adjust_pseudohaploid=False,
                verbose=False,
            )
        # Frozen analytical values for this fixture. The normalized value is
        # the block-jackknife ratio used by ADMIXTOOLS 2; raw is its
        # outgroupmode=TRUE numerator scale.
        self.assertAlmostEqual(normalized.loc[0, "est"], 0.0625)
        self.assertAlmostEqual(raw.loc[0, "est"], 5 / 36)
        self.assertNotAlmostEqual(normalized.loc[0, "est"], raw.loc[0, "est"])

    def test_singleton_target_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = self._write_singleton_source_eigenstrat(Path(tmp))
            for stream in (False, True):
                with self.subTest(stream=stream):
                    with self.assertRaises(ValueError) as caught:
                        qp3pop(
                            pref,
                            "C",
                            "A",
                            "B",
                            blgsize=0.05,
                            stream=stream,
                            verbose=False,
                        )
                    message = str(caught.exception)
                    self.assertIn("f3 target population cannot be estimated", message)
                    self.assertIn("'C'", message)
                    self.assertIn("pseudohaploid singleton", message)

    def test_singleton_target_allows_explicit_uncorrected_raw_f3(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = self._write_singleton_source_eigenstrat(Path(tmp))
            with self.assertWarnsRegex(RuntimeWarning, "apply_corr=False"):
                out = qp3pop(
                    pref,
                    "C",
                    "A",
                    "B",
                    blgsize=0.05,
                    apply_corr=False,
                    outgroupmode=True,
                    verbose=False,
                )
        self.assertTrue(np.isfinite(out.loc[0, "est"]))
        self.assertTrue(np.isfinite(out.loc[0, "se"]))
        self.assertGreater(out.loc[0, "n"], 0)

    def test_f2_remove_na_error_names_nonfinite_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = self._write_singleton_source_eigenstrat(Path(tmp))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with self.assertRaisesRegex(
                    ValueError,
                    r"No blocks remain with remove_na=True\. Non-finite f2 pairs: \('A', 'C'\)\.",
                ):
                    f2_from_geno(
                        pref,
                        pops=["A", "C"],
                        maxmiss=1,
                        poly_only=False,
                        remove_na=True,
                        verbose=False,
                    )

    def test_streaming_f4_singleton_warning_is_summarized_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = self._write_singleton_source_eigenstrat(Path(tmp))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with self.assertRaisesRegex(ValueError, "f4 produced 1 non-finite estimate"):
                    qpdstat(
                        pref,
                        "C",
                        "A",
                        "C",
                        "B",
                        blgsize=0.05,
                        stream=True,
                        verbose=False,
                    )
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, RuntimeWarning)
        self.assertIn("f4 bias correction requires at least two", str(caught[0].message))
        self.assertIn("count < 2 in 2 of 3 blocks", str(caught[0].message))
        self.assertIn("Affected populations: 'C'", str(caught[0].message))

    def test_pairwise_f4_errors_instead_of_returning_nonfinite_estimate(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = self._write_singleton_source_eigenstrat(Path(tmp))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with self.assertRaises(ValueError) as caught:
                    qpdstat(
                        pref,
                        "C",
                        "A",
                        "C",
                        "B",
                        allsnps=False,
                        blgsize=0.05,
                        verbose=False,
                    )
        message = str(caught.exception)
        self.assertIn("f4 produced 1 non-finite estimate", message)
        self.assertIn("f4(C, A; C, B)", message)
        self.assertIn("Bias correction needs allele count >= 2", message)
        self.assertIn("Use allsnps=True", message)
        self.assertIn("apply_corr=False", message)
        self.assertIn("all four populations", message)

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
                outgroupmode=True,
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
                outgroupmode=True,
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

    def test_poly_only_retains_equal_segregating_frequencies(self):
        snps = pd.DataFrame(
            {
                "SNP": ["s1", "s2", "s3", "s4"],
                "CHR": [1, 1, 1, 1],
                "cm": [0.00, 0.01, 0.06, 0.07],
                "POS": [1, 2, 3, 4],
                "A1": ["A"] * 4,
                "A2": ["G"] * 4,
            }
        )
        afs = pd.DataFrame(
            {
                "A": [0.5, 0.0, 1.0, 0.5],
                "B": [0.5, 0.0, 1.0, 0.5],
                "C": [0.5, 0.0, 1.0, 0.5],
            },
            index=snps["SNP"],
        )
        counts = pd.DataFrame(4.0, index=afs.index, columns=afs.columns)
        stats = _f3_direct_blocks_from_afs(
            AfData(afs, counts, snps),
            pd.DataFrame([{"pop1": "A", "pop2": "B", "pop3": "C"}]),
            blgsize=0.05,
            poly_only=True,
            outgroupmode=True,
            verbose=False,
        )
        self.assertEqual(int(stats.snp_counts.sum()), 2)
        self.assertTrue(np.isfinite(stats.est[0]))

    def test_normalized_f3_weights_numerator_and_denominator(self):
        snps = pd.DataFrame(
            {
                "SNP": ["s1", "s2"],
                "CHR": [1, 1],
                "cm": [0.00, 0.01],
                "POS": [1, 2],
                "A1": ["A", "A"],
                "A2": ["G", "G"],
            }
        )
        afs = pd.DataFrame(
            {"A": [0.25, 0.5], "B": [0.0, 0.25], "C": [1.0, 0.75]},
            index=snps["SNP"],
        )
        counts = pd.DataFrame(4.0, index=afs.index, columns=afs.columns)
        stats = _f3_direct_blocks_from_afs(
            AfData(afs, counts, snps),
            pd.DataFrame([{"pop1": "A", "pop2": "B", "pop3": "C"}]),
            blgsize=0.05,
            snpwt=[2.0, 3.0],
            verbose=False,
        )
        # Unbiased target heterozygosities are 1/2 and 2/3, so the
        # mean of their weighted components is (2*1/2 + 3*2/3) / 2.
        self.assertAlmostEqual(stats.ratio_den[0, 0], 1.5)

    def test_streamed_and_materialized_paths_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = RawF4Tests._write_complete_eigenstrat(Path(tmp))
            streamed = qp3pop(
                pref, "A", "B", "C", blgsize=0.04, chunk_size=3,
                adjust_pseudohaploid=False, stream=True, verbose=False,
            )
            materialized = qp3pop(
                pref, "A", "B", "C", blgsize=0.04, stream=False,
                adjust_pseudohaploid=False, verbose=False,
            )
        pd.testing.assert_frame_equal(streamed, materialized)

    def test_all_supported_formats_give_same_direct_f3(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eigen = RawF4Tests._write_complete_eigenstrat(root)
            prefixes = [eigen] + self._write_binary_equivalents(eigen, root)
            results = [
                qp3pop(
                    pref, "A", "B", "C", blgsize=0.04, chunk_size=3,
                    adjust_pseudohaploid=False, stream=True, verbose=False,
                )
                for pref in prefixes
            ]
        for result in results[1:]:
            pd.testing.assert_frame_equal(result, results[0])


class NominalMissingBlockTests(unittest.TestCase):
    def test_absent_block_deletion_leaves_full_estimate(self):
        loo = stats_to_loo(np.array([[1.0, np.nan, 3.0]]), [1, 10, 1])
        np.testing.assert_allclose(loo, [[3.0, 2.0, 1.0]])

    def test_single_finite_block_has_no_deletable_estimate(self):
        loo = stats_to_loo(np.array([[np.nan, 2.0, np.nan]]), [1, 10, 1])
        self.assertEqual(loo[0, 0], 2.0)
        self.assertTrue(np.isnan(loo[0, 1]))
        self.assertEqual(loo[0, 2], 2.0)

    def test_covariance_excludes_noncontributing_nominal_blocks(self):
        blocks = np.array([[1.0, np.nan, 3.0]])
        lengths = np.array([1.0, 10.0, 1.0])
        loo = stats_to_loo(blocks, lengths)
        cov, est = jackknife_cov(
            loo,
            lengths,
            contributes=np.isfinite(blocks),
        )
        self.assertAlmostEqual(est[0], 2.0)
        self.assertAlmostEqual(cov[0, 0], 1.0)


class QpAdmValidationTests(unittest.TestCase):
    def test_covariance_inverse_rejects_nonfinite_values_explicitly(self):
        with self.assertRaisesRegex(
            ValueError,
            "Covariance matrix contains 1 non-finite entry; cannot compute its inverse",
        ):
            _qinv_from_cov(np.array([[np.nan]]), fudge=0.0001)

    def test_qpadm_error_names_contrasts_with_nonfinite_covariance(self):
        rows = pd.DataFrame(
            [
                {"pop1": "S1", "pop2": "Target", "pop3": "R1", "pop4": "R0"},
                {"pop1": "S2", "pop2": "Target", "pop3": "R1", "pop4": "R0"},
            ]
        )
        f4 = BlockStats(
            rows=rows,
            blocks=np.zeros((2, 2)),
            block_lengths=np.ones(2),
            stat="f4",
            loo=np.zeros((2, 2)),
            est=np.array([0.1, 0.2]),
            cov=np.array([[1.0, np.nan], [np.nan, 1.0]]),
        )
        qpw = QpWaveStats(
            f4=f4,
            left=["Target", "S1", "S2"],
            right=["R0", "R1"],
            left_base="Target",
            right_base="R0",
            row_pops=["S1", "S2"],
            col_pops=["R1"],
        )
        with patch("admixpy.fstats.qpwave_f4stats", return_value=qpw):
            with self.assertRaises(ValueError) as ctx:
                qpadm(
                    "unused",
                    "Target",
                    ["S1", "S2"],
                    ["R0", "R1"],
                    allsnps=False,
                    verbose=False,
                )
        message = str(ctx.exception)
        self.assertIn("qpAdm cannot fit the model", message)
        self.assertIn("2 non-finite covariance entries", message)
        self.assertIn("f4(S1, Target; R1, R0)", message)
        self.assertIn("f4(S2, Target; R1, R0)", message)
        self.assertIn("pseudohaploid singletons", message)

    def test_qpwave_error_names_contrasts_with_nonfinite_covariance(self):
        rows = pd.DataFrame(
            [
                {"pop1": "S1", "pop2": "Target", "pop3": "R1", "pop4": "R0"},
                {"pop1": "S2", "pop2": "Target", "pop3": "R1", "pop4": "R0"},
            ]
        )
        f4 = BlockStats(
            rows=rows,
            blocks=np.zeros((2, 2)),
            block_lengths=np.ones(2),
            stat="f4",
            loo=np.zeros((2, 2)),
            est=np.array([0.1, 0.2]),
            cov=np.array([[1.0, np.nan], [np.nan, 1.0]]),
        )
        qpw = QpWaveStats(
            f4=f4,
            left=["Target", "S1", "S2"],
            right=["R0", "R1"],
            left_base="Target",
            right_base="R0",
            row_pops=["S1", "S2"],
            col_pops=["R1"],
        )
        with patch("admixpy.fstats.qpwave_f4stats", return_value=qpw):
            with self.assertRaises(ValueError) as ctx:
                qpwave(
                    "unused",
                    ["Target", "S1", "S2"],
                    ["R0", "R1"],
                    allsnps=False,
                    verbose=False,
                )
        message = str(ctx.exception)
        self.assertIn("qpWave cannot test ranks", message)
        self.assertIn("f4(S1, Target; R1, R0)", message)
        self.assertIn("f4(S2, Target; R1, R0)", message)


if __name__ == "__main__":
    unittest.main()
