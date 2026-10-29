import importlib
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import admixpy


dating = importlib.import_module("admixpy.admixture_date")


def write_eigenstrat(root, geno, position, chromosome, populations=None):
    pref = Path(root) / "data"
    populations = populations or ["T", "T", "A", "A", "B", "B"]
    pref.with_suffix(".ind").write_text("".join(
        f"i{i} U {pop}\n" for i, pop in enumerate(populations)))
    pref.with_suffix(".snp").write_text("".join(
        f"s{i} {ch} {pos:.12g} {i * 100} A G\n"
        for i, (ch, pos) in enumerate(zip(chromosome, position))))
    pref.with_suffix(".geno").write_text("".join(
        "".join("9" if np.isnan(g) else str(int(g)) for g in row) + "\n" for row in geno))
    return pref


def write_binary(root, eigen, geno):
    nsnp, nind = geno.shape
    codes = np.where(np.isnan(geno), 3, geno).astype(int)

    def pack(row, size):
        out = bytearray(size)
        for i, g in enumerate(row):
            out[i // 4] |= int(g) << (6 - 2 * (i % 4))
        return bytes(out)

    paths = []
    for name in ["packed", "tgeno"]:
        pref = Path(root) / name
        pref.with_suffix(".ind").write_text(eigen.with_suffix(".ind").read_text())
        pref.with_suffix(".snp").write_text(eigen.with_suffix(".snp").read_text())
        size = max(48, ((nind if name == "packed" else nsnp) + 3) // 4)
        header_size = size if name == "packed" else 48
        header = f"{'GENO' if name == 'packed' else 'TGENO'} {nind} {nsnp} 0 0".encode().ljust(header_size, b"\0")
        payload = header + b"".join(pack(row, size) for row in (codes if name == "packed" else codes.T))
        pref.with_suffix(".geno" if name == "packed" else ".tgeno").write_bytes(payload)
        paths.append(pref)
    pref = Path(root) / "plink"
    ind = [row.split() for row in eigen.with_suffix(".ind").read_text().splitlines()]
    pref.with_suffix(".fam").write_text("".join(f"{pop} {iid} 0 0 0 -9\n" for iid, sex, pop in ind))
    snps = [row.split() for row in eigen.with_suffix(".snp").read_text().splitlines()]
    pref.with_suffix(".bim").write_text("".join(
        f"{ch} {sid} {float(pos) * 100:.12g} {bp} {a} {b}\n" for sid, ch, pos, bp, a, b in snps))
    payload = bytearray(b"\x6c\x1b\x01")
    for row in codes:
        out = bytearray((nind + 3) // 4)
        for i, g in enumerate(row):
            out[i // 4] |= {0: 3, 1: 2, 2: 0, 3: 1}[g] << (2 * (i % 4))
        payload.extend(out)
    pref.with_suffix(".bed").write_bytes(payload)
    return paths + [pref]


def simulated_data(rate=40., missing=0., diploid=False):
    """Independent two-source haplotypes with a known ancestry transition rate."""
    rng = np.random.default_rng(173)
    nchrom, nsnp, ntarget = 12, 1600, 4
    position = np.tile(np.arange(nsnp) * .0005, nchrom)
    chromosome = np.repeat(np.arange(1, nchrom + 1), nsnp)
    n = len(position)
    # Fixed alternate source alleles with random orientation: exact reference AFs.
    a = rng.integers(0, 2, n)
    b = 1 - a
    target = np.empty((n, ntarget))
    for sample in range(ntarget):
        dosage = np.zeros(n)
        for hap in range(2 if diploid else 1):
            states = np.empty(n, int)
            for ch in range(nchrom):
                start = ch * nsnp
                state = int(rng.random() < .4)
                for j in range(nsnp):
                    if j and rng.random() < -np.expm1(-rate * .0005):
                        state = int(rng.random() < .4)
                    states[start + j] = state
            dosage += np.where(states, a, b)
        target[:, sample] = dosage if diploid else 2 * dosage
    target[rng.random(target.shape) < missing] = np.nan
    geno = np.column_stack([target, 2 * a, 2 * a, 2 * b, 2 * b])
    return geno, position, chromosome, ["T"] * ntarget + ["A", "A", "B", "B"]


def run_date(pref, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return admixpy.admixture_date(pref, "T", ["A", "B"], verbose=False, **kwargs)


class CovarianceTests(unittest.TestCase):
    def test_score_averages_allele_scores_and_source_swap(self):
        g = np.array([0., 1., 2., 2., np.nan])
        a, b = np.full(5, .8), np.full(5, .3)
        score, alpha = dating._ancestry_scores(g, a, b, 2)
        self.assertAlmostEqual(alpha, .65)
        p, q = .625, np.sqrt(.65 * .35)
        expected = q * .5 * np.array([-1 / (1 - p), .5 / p - .5 / (1 - p), 1 / p, 1 / p])
        np.testing.assert_allclose(score[:4], expected)
        self.assertTrue(np.isnan(score[4]))
        swapped, other_alpha = dating._ancestry_scores(g, b, a, 2)
        np.testing.assert_allclose(swapped, -score)
        self.assertAlmostEqual(other_alpha, 1 - alpha)

    def test_direct_matches_independent_pair_enumeration(self):
        x = np.array([0, .001, .001, .003, .008, .013])
        y = np.array([2., -1., .3, np.nan, 4., -3.])
        stats = dating._pair_stats(x, y, .004, .012, "direct", 10)
        bins = [[] for _ in range(3)]
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if np.isfinite(y[i] + y[j]) and x[j] - x[i] < .012:
                    bins[int((x[j] - x[i]) / .004)].append((y[i], y[j], x[j] - x[i]))
        cov, distance = dating._covariance(stats)
        for k, pairs in enumerate(bins):
            if pairs:
                pairs = np.array(pairs)
                self.assertEqual(stats[k, 0], len(pairs))
                self.assertAlmostEqual(cov[k], np.mean(pairs[:, 0] * pairs[:, 1]) - np.mean(pairs[:, :2]) ** 2)
                self.assertAlmostEqual(distance[k], np.mean(pairs[:, 2]))

    def test_fft_matches_mesh_enumeration_including_collisions_and_gaps(self):
        rng = np.random.default_rng(8)
        # Integer coordinates make every bin boundary exact in both methods.
        x = np.sort(rng.integers(0, 80, 70)).astype(float)
        y = rng.normal(size=len(x))
        y[::7] = np.nan
        exact = dating._pair_stats(x, y, 8., 64., "direct", 4)
        fft = dating._pair_stats(x, y, 8., 64., "fft", 4)
        # Quantize independently, then enumerate with integer coordinates.
        snapped = np.floor((x - x[0]) / 2 + .5) * 2
        reference = dating._pair_stats(snapped, y, 8., 64., "direct", 4)
        np.testing.assert_allclose(fft, reference, atol=1e-10)
        self.assertGreater(exact[:, 0].sum(), 0)

    def test_no_self_pairs_or_pairs_across_individuals(self):
        empty = dating._pair_stats(np.array([0.]), np.array([2.]), .01, .1, "fft", 10)
        self.assertEqual(empty.sum(), 0)
        stats = dating._pair_stats(np.array([0., .00001]), np.array([2., 3.]), .01, .1, "fft", 10)
        np.testing.assert_allclose(stats[0, :3], [1, 5, 6])

    def test_chromosome_reversal_leaves_covariance_unchanged(self):
        x, y = np.array([0., 10., 14., 34.]), np.array([1., 3., -4., 2.])
        for method in ["direct", "fft"]:
            original = dating._pair_stats(x, y, 11., 70., method, 11)
            reversed_ = dating._pair_stats(x[-1] - x[::-1], y[::-1], 11., 70., method, 11)
            np.testing.assert_allclose(original[:, :3], reversed_[:, :3], atol=1e-10)


class FitTests(unittest.TestCase):
    def test_exact_curve_fit_and_executable_generation_offset(self):
        x = np.linspace(.005, .3, 160)
        for intercept in [True, False]:
            y = .07 * np.exp(-42 * x) + (.003 if intercept else 0)
            fit = dating._fit_decay(x, y, intercept, (.001, 10000))
            self.assertEqual(fit["status"], "converged")
            np.testing.assert_allclose(fit["values"], [43, 42, .07, .003 if intercept else 0], rtol=1e-6, atol=1e-9)
            self.assertLess(fit["nrmsd"], 1e-8)
            self.assertFalse(fit["boundary"])

    def test_flat_increasing_sparse_and_boundary_curves(self):
        x = np.linspace(.005, .3, 80)
        for y in [np.full(80, .003), x.copy()]:
            fit = dating._fit_decay(x, y, True, (.001, 1000))
            self.assertEqual(fit["status"], "no_signal")
            self.assertTrue(np.isnan(fit["values"][0]))
        fit = dating._fit_decay(x[:3], x[:3], True, (.001, 1000))
        self.assertEqual(fit["status"], "insufficient_bins")
        fit = dating._fit_decay(x, .1 * np.exp(-200 * x), True, (1, 100))
        self.assertTrue(fit["boundary"])

    def test_jackknife_matches_equal_and_unequal_delete_group_formulas(self):
        full = np.array([40., .1])
        reps = np.array([[38, .11], [43, .08], [39, .12]])
        se = dating._jackknife_se(full, reps, [1, 1, 1])
        np.testing.assert_allclose(se, np.sqrt(2 / 3 * ((reps - reps.mean(axis=0)) ** 2).sum(axis=0)))
        weights = np.array([2., 3., 7.])
        h = weights.sum() / weights
        pseudo = full + (h[:, None] - 1) * (full - reps)
        center = full + np.sum((1 - weights[:, None] / weights.sum()) * (full - reps), axis=0)
        expected = np.sqrt(np.mean((pseudo - center) ** 2 / (h[:, None] - 1), axis=0))
        np.testing.assert_allclose(dating._jackknife_se(full, reps, weights), expected)
        reps[0] = np.nan
        self.assertTrue(np.isnan(dating._jackknife_se(full, reps, weights)).all())


class AdmixtureDateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = simulated_data()

    def test_recovers_simulated_decay_and_returns_reviewable_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = write_eigenstrat(tmp, *self.sim)
            result = run_date(pref, max_distance=.25, return_jackknife=True)
        self.assertIsInstance(result, admixpy.AdmixtureDateResult)
        row = result.summary.iloc[0]
        self.assertEqual(row.fit_status, "converged")
        self.assertLess(abs(row.est - 41), 7)
        self.assertTrue(np.isfinite(row.se))
        self.assertGreater(row.se, 0)
        self.assertEqual(row.n_individuals, 4)
        self.assertEqual(row.n_chromosomes, 12)
        self.assertEqual(len(result.jackknife), 12)
        self.assertAlmostEqual(result.jackknife.weight.sum(), 1)
        self.assertEqual(result.settings["generation_offset"], 1)
        self.assertEqual(result.parameters.parameter.tolist(), dating._PARAMETERS)
        self.assertFalse(result.curve.loc[result.curve.bin_left < .0045, "included"].any())
        self.assertIn("AdmixtureDateResult", repr(result))
        self.assertGreater(len(result.curve.to_csv(index=False)), 100)

    def test_missing_pseudohaploid_and_diploid_simulations(self):
        for diploid in [False, True]:
            with self.subTest(diploid=diploid), tempfile.TemporaryDirectory() as tmp:
                pref = write_eigenstrat(tmp, *simulated_data(missing=.4, diploid=diploid))
                result = run_date(pref, max_distance=.25, jackknife=None)
                self.assertLess(abs(result.summary.est[0] - 41), 12)
                self.assertTrue(np.isnan(result.summary.se[0]))
                self.assertIsNone(result.jackknife)
                self.assertEqual(result.summary.identifiability[0], "not_assessed")

    def test_formats_map_units_source_swap_and_target_selection(self):
        g, pos, chrom, pops = self.sim
        # Enough data for a fit, but keep direct enumeration inexpensive.
        keep = np.arange(len(g)) < 1600
        with tempfile.TemporaryDirectory() as tmp:
            pref = write_eigenstrat(tmp, g[keep], pos[keep], chrom[keep], pops)
            base = run_date(pref, max_distance=.1, jackknife=None, target_iids=["i0"])
            for path in write_binary(tmp, pref, g[keep]):
                other = run_date(path, max_distance=.1, jackknife=None, target_iids="i0")
                np.testing.assert_allclose(other.curve.covariance, base.curve.covariance, atol=1e-10)
                np.testing.assert_allclose(other.parameters.est, base.parameters.est, rtol=1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                swapped = admixpy.admixture_date(pref, "T", ["B", "A"], target_iids=["i0"],
                                                  max_distance=.1, jackknife=None, verbose=False)
            np.testing.assert_allclose(swapped.curve.covariance, base.curve.covariance, atol=1e-12)
            # Explicit cM override on an EIGENSTRAT map.
            table = pd.read_csv(str(pref) + ".snp", sep=r"\s+", header=None)
            table[2] *= 100
            table.to_csv(str(pref) + ".snp", sep=" ", header=False, index=False)
            other = run_date(pref, max_distance=.1, map_unit="cM", jackknife=None, target_iids=["i0"])
            np.testing.assert_allclose(other.curve.covariance, base.curve.covariance, atol=1e-10)

    def test_actual_fft_close_to_exact_pairs_and_pair_pooling(self):
        g, pos, chrom, pops = self.sim
        take = np.arange(600)
        with tempfile.TemporaryDirectory() as tmp:
            pref = write_eigenstrat(tmp, g[take], pos[take], chrom[take], pops)
            fft = run_date(pref, min_distance=.01, max_distance=.15, bin_size=.003,
                           method="fft", mesh_subdivisions=60, jackknife=None)
            direct = run_date(pref, min_distance=.01, max_distance=.15, bin_size=.003,
                              method="direct", jackknife=None)
            np.testing.assert_allclose(fft.curve.covariance, direct.curve.covariance, atol=.005)
            self.assertLess(abs(fft.summary.est[0] - direct.summary.est[0]), 3)
            self.assertLessEqual(fft.curve.n_pairs.sum(), 4 * 600 * 599 / 2)

    def test_validation_before_genotype_io(self):
        cases = [dict(sources=["A"]), dict(sources=["A", "A"]), dict(sources=["T", "B"]),
                 dict(min_distance=-1), dict(max_distance=.001), dict(bin_size=0),
                 dict(map_unit="bp"), dict(method="other"), dict(jackknife="snp"),
                 dict(mesh_subdivisions=0), dict(mesh_subdivisions=1.5), dict(confidence=1),
                 dict(rate_bounds=(100, 1)), dict(rate_bounds=(0, 1)), dict(min_snps=1)]
        for overrides in cases:
            options = dict(target="T", sources=["A", "B"], verbose=False)
            options.update(overrides)
            with self.subTest(overrides=overrides), patch.object(dating, "_read_input") as reader:
                with self.assertRaises(ValueError):
                    admixpy.admixture_date("unused", **options)
                reader.assert_not_called()
        for data in [pd.DataFrame(), Path(tempfile.gettempdir())]:
            with self.assertRaisesRegex(ValueError, "genotype dataset prefix"):
                run_date(data)

    def test_target_iids_selects_and_records_individual_ids(self):
        g, pos, chrom, pops = self.sim
        with tempfile.TemporaryDirectory() as tmp:
            pref = write_eigenstrat(tmp, g[:1600], pos[:1600], chrom[:1600], pops)
            canonical = run_date(pref, target_iids=["i0"], max_distance=.1, jackknife=None)
            self.assertEqual(canonical.settings["target_iids"], ["i0"])
            self.assertEqual(canonical.summary.n_individuals[0], 1)

    def test_multiple_individuals_pool_pairs_without_cross_individual_products(self):
        g, pos, chrom, pops = self.sim
        g = g[:220].copy()
        g[::3, 0] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            pref = write_eigenstrat(tmp, g, pos[:220], chrom[:220], pops)
            combined = run_date(pref, min_snps=100, max_distance=.05, jackknife=None)
            singles = [run_date(pref, min_snps=100, max_distance=.05, jackknife=None,
                                target_iids=[f"i{i}"]) for i in range(4)]
            expected = sum((r.curve.n_pairs.to_numpy() for r in singles))
            np.testing.assert_array_equal(combined.curve.n_pairs, expected)

    def test_all_missing_individual_and_indistinguishable_references_raise(self):
        g, pos, chrom, pops = self.sim
        with tempfile.TemporaryDirectory() as tmp:
            missing = g[:300].copy()
            missing[:, 0] = np.nan
            pref = write_eigenstrat(tmp, missing, pos[:300], chrom[:300], pops)
            with self.assertRaisesRegex(ValueError, "Target individual i0.*min_snps"):
                run_date(pref)
            same = g[:300].copy()
            same[:, 6:] = same[:, 4:6]
            pref = write_eigenstrat(tmp, same, pos[:300], chrom[:300], pops)
            with self.assertRaisesRegex(ValueError, "ancestry-informative"):
                run_date(pref)

    def test_default_autosome_filter_and_explicit_nonhuman_chromosomes(self):
        g, pos, chrom, pops = self.sim
        with tempfile.TemporaryDirectory() as tmp:
            pref = write_eigenstrat(tmp, g[:300], pos[:300], ["scaffoldA"] * 300, pops)
            with self.assertRaisesRegex(ValueError, "chromosome selection"):
                run_date(pref)
            result = run_date(pref, autosomes_only=False, jackknife=None, max_distance=.05)
            self.assertEqual(result.summary.n_chromosomes[0], 1)

    def test_invalid_maps_selection_and_unadmixed_targets(self):
        g, pos, chrom, pops = self.sim
        with tempfile.TemporaryDirectory() as tmp:
            pref = write_eigenstrat(tmp, g[:1600], np.zeros(1600), chrom[:1600], pops)
            with self.assertRaisesRegex(ValueError, "No usable genetic map"):
                run_date(pref)
            bad = pos[:1600].copy()
            bad[10] = np.inf
            pref = write_eigenstrat(tmp, g[:1600], bad, chrom[:1600], pops)
            with self.assertRaisesRegex(ValueError, "finite genetic map"):
                run_date(pref)
            pref = write_eigenstrat(tmp, g[:1600], pos[:1600][::-1], chrom[:1600], pops)
            with self.assertRaisesRegex(ValueError, "decreases"):
                run_date(pref)
            pref = write_eigenstrat(tmp, g[:1600], pos[:1600], chrom[:1600], pops)
            with self.assertRaisesRegex(ValueError, "missing from target"):
                run_date(pref, target_iids=["i4"])
            with self.assertRaisesRegex(ValueError, "Populations missing"):
                admixpy.admixture_date(pref, "Missing", ["A", "B"], verbose=False)
            homogeneous = g[:1600].copy()
            homogeneous[:, :4] = homogeneous[:, 4, None]
            pref = write_eigenstrat(tmp, homogeneous, pos[:1600], chrom[:1600], pops)
            with self.assertRaisesRegex(ValueError, "outside"):
                run_date(pref)

    def test_signed_map_origins_preserve_distances_covariance_and_date(self):
        g, pos, chrom, pops = self.sim
        with tempfile.TemporaryDirectory() as tmp:
            for method in ["direct", "fft"]:
                with self.subTest(method=method):
                    pref = write_eigenstrat(tmp, g[:1600], pos[:1600], chrom[:1600], pops)
                    base = run_date(pref, method=method, max_distance=.1,
                                    target_iids=["i0"], jackknife=None)
                    pref = write_eigenstrat(tmp, g[:1600], pos[:1600] - .125, chrom[:1600], pops)
                    shifted = run_date(pref, method=method, max_distance=.1,
                                       target_iids=["i0"], jackknife=None)
                    np.testing.assert_allclose(shifted.curve.distance, base.curve.distance, atol=1e-12)
                    np.testing.assert_array_equal(shifted.curve.n_pairs, base.curve.n_pairs)
                    np.testing.assert_allclose(shifted.curve.covariance, base.curve.covariance, atol=1e-12)
                    np.testing.assert_allclose(shifted.parameters.est, base.parameters.est, rtol=1e-6)
                    self.assertEqual(shifted.settings["map_origins_morgans"], {"1": -.125})
                    self.assertEqual(shifted.summary.n_snps[0], base.summary.n_snps[0])

    def test_one_chromosome_suppresses_uncertainty_and_reports_warning(self):
        g, pos, chrom, pops = self.sim
        with tempfile.TemporaryDirectory() as tmp:
            pref = write_eigenstrat(tmp, g[:1600], pos[:1600], chrom[:1600], pops)
            result = run_date(pref, max_distance=.15, return_jackknife=True)
            self.assertTrue(np.isnan(result.summary.se[0]))
            self.assertIn("insufficient_jackknife_chromosomes", result.summary.status[0])
            self.assertFalse(result.jackknife.usable.any())

    def test_failed_jackknife_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            pref = write_eigenstrat(tmp, *self.sim)
            real_fit = dating._fit_decay
            calls = []

            def fail_one(*args):
                fit = real_fit(*args)
                calls.append(1)
                if len(calls) == 3:
                    fit["status"] = "no_signal"
                    fit["values"][:] = np.nan
                return fit

            with patch.object(dating, "_fit_decay", side_effect=fail_one):
                result = run_date(pref, max_distance=.15, return_jackknife=True)
            self.assertTrue(np.isnan(result.summary.se[0]))
            self.assertIn("unstable_jackknife", result.summary.status[0])
            self.assertEqual(result.diagnostics.n_failed_jackknife[0], 1)


if __name__ == "__main__":
    unittest.main()
