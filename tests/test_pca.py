import importlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

import admixpy as ap
from test_admixture_date import write_eigenstrat, write_binary


pca_module = importlib.import_module('admixpy.pca')


class PCATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(83)
        self.g = rng.binomial(2, rng.uniform(.1, .9, 260)[:, None], (260, 15)).astype(float)
        self.pops = ['A'] * 6 + ['B'] * 6 + ['T'] * 3
        self.pref = self.write(self.g)

    def write(self, g, name='data', populations=None):
        folder = self.root / name
        folder.mkdir()
        return write_eigenstrat(folder, g, np.arange(len(g)) * .001,
                                np.ones(len(g), dtype=int), populations or self.pops)

    def fit(self, **kwargs):
        options = dict(pops=['A', 'B'], n_components=3, min_snps=1, verbose=False)
        options.update(kwargs)
        return ap.pca(self.pref, **options)

    def test_svd_and_reference_only_statistics(self):
        result = self.fit(project_pops='T')
        model = result.model
        ids = model.snps.SNP.str[1:].astype(int).to_numpy()
        g = self.g[ids, :12].T
        p = g.mean(axis=0) / 2
        z = (g - 2 * p) / np.sqrt(2 * p * (1-p))
        u, s, vt = np.linalg.svd(z, full_matrices=False)
        np.testing.assert_allclose(model.frequencies, p)
        np.testing.assert_allclose(model.eigenvalues, s[:3] ** 2 / len(ids))
        np.testing.assert_allclose(result.eigenvalues.explained_variance_ratio,
                                   s[:3] ** 2 / np.sum(s ** 2))
        np.testing.assert_allclose(result.scores.iloc[:12, 3:].to_numpy(), z @ model.loadings)
        self.assertLess(result.eigenvalues.explained_variance_ratio.sum(), 1)
        changed = self.g.copy()
        changed[:, 12:] = 0
        other = ap.pca(self.write(changed, 'changed'), pops=['A', 'B'],
                       project_pops='T', n_components=3, min_snps=1, verbose=False)
        np.testing.assert_allclose(other.model.loadings, model.loadings)
        self.assertEqual(result.scores.role.tolist(), ['reference'] * 12 + ['projected'] * 3)

    def test_combined_and_reused_model_projection_match(self):
        combined = self.fit(project_pops='T', chunk_size=17)
        later = ap.pca(self.pref, model=combined.model, project_pops='T',
                       min_snps=1, chunk_size=43, verbose=False)
        np.testing.assert_allclose(later.scores.iloc[:, 3:], combined.scores.iloc[12:, 3:], atol=1e-12)
        self.assertIs(later.model, combined.model)
        self.assertEqual(later.settings['mode'], 'project')
        self.assertEqual(later.scores.iid.tolist(), ['i12', 'i13', 'i14'])

    def test_missing_data_qr_matches_independent_lstsq(self):
        rng = np.random.default_rng(5)
        g = self.g.copy()
        g[rng.random(g.shape) < .25] = np.nan
        result = ap.pca(self.write(g, 'missing'), pops=['A', 'B'], project_pops='T',
                        n_components=3, maxmiss=.8, min_snps=1, chunk_size=19, verbose=False)
        m = result.model
        ids = m.snps.SNP.str[1:].astype(int).to_numpy()
        ref = g[ids, :12].T
        zref = np.nan_to_num((ref - 2*m.frequencies) / m.scales, nan=0)
        np.testing.assert_allclose(result.scores.iloc[:12, 3:], zref @ m.loadings, atol=1e-10)
        for sample in range(3):
            row = g[ids, sample + 12]
            keep = np.isfinite(row)
            z = (row[keep] - 2*m.frequencies[keep]) / m.scales[keep]
            expected = np.linalg.lstsq(m.loadings[keep], z, rcond=None)[0]
            np.testing.assert_allclose(result.scores.iloc[sample + 12, 3:].to_numpy(dtype=float), expected, atol=1e-10)

    def test_selection_intersection_and_validation(self):
        result = self.fit(pops='A', iids=['i4', 'i0', 'i7'], n_components=1,
                          project_pops='T', project_iids=['i14', 'i12'])
        self.assertEqual(result.scores.iid.tolist(), ['i0', 'i4', 'i12', 'i14'])
        self.assertEqual(result.scores.population.tolist(), ['A', 'A', 'T', 'T'])
        for kwargs in [dict(pops='absent'), dict(iids=['absent']),
                       dict(pops='A', iids=['i12']), dict(pops=[]),
                       dict(pops='A', project_iids=['i0']),
                       dict(pops=None, project_pops='T')]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.fit(**kwargs)
        duplicate = self.pref.with_suffix('.ind').read_text().replace('i1 U', 'i0 U')
        self.pref.with_suffix('.ind').write_text(duplicate)
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.fit()

    def test_model_rejects_fit_options_even_explicit_defaults(self):
        model = self.fit().model
        for key, value in pca_module._FIT_DEFAULTS.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Fitting'):
                ap.pca(self.pref, model=model, project_pops='T', **{key: value})
        for kwargs in [dict(), dict(pops='A', project_pops='T'), dict(iids=['i0'], project_iids=['i12'])]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ap.pca(self.pref, model=model, **kwargs)

    def test_snp_whitelist_qc_and_reproducible_cap(self):
        g = self.g.copy()
        g[0, :12] = 0
        g[1, 0] = np.nan
        pref = self.write(g, 'qc')
        options = dict(pops=['A', 'B'], n_components=3, keepsnps=[f's{i}' for i in range(100)],
                       max_snps=35, seed=12, verbose=False)
        a = ap.pca(pref, chunk_size=13, **options)
        b = ap.pca(pref, chunk_size=71, **options)
        self.assertEqual(a.model.snps.SNP.tolist(), b.model.snps.SNP.tolist())
        np.testing.assert_allclose(a.model.loadings, b.model.loadings, atol=1e-10)
        self.assertEqual(len(a.model.snps), 35)
        self.assertNotIn('s0', a.model.snps.SNP.tolist())
        self.assertNotIn('s1', a.model.snps.SNP.tolist())
        self.assertTrue(set(a.model.snps.SNP) <= set(options['keepsnps']))
        c = ap.pca(pref, **{**options, 'seed': 13})
        self.assertNotEqual(a.model.snps.SNP.tolist(), c.model.snps.SNP.tolist())
        all_ = ap.pca(pref, **{**options, 'max_snps': 1000})
        p = np.nanmean(g[:100, :12], axis=1) / 2
        eligible = (np.minimum(p, 1-p) >= .01) & np.isfinite(g[:100, :12]).all(axis=1)
        self.assertEqual(len(all_.model.snps), int(eligible.sum()))

    def test_metadata_filters(self):
        snp = pd.read_csv(self.pref.with_suffix('.snp'), sep=r'\s+', header=None)
        snp[1] = snp[1].astype(str)
        snp.loc[0, 1] = 'chrX'
        snp.loc[1, 1] = 'chr0'
        snp.loc[2, 1] = 'chr2'
        snp.loc[3, 5] = 'C'  # A/C transversion, other SNPs are A/G transitions.
        snp.to_csv(self.pref.with_suffix('.snp'), sep=' ', header=False, index=False)
        result = self.fit(transversions=False)
        self.assertFalse({'s0', 's1', 's3'} & set(result.model.snps.SNP))
        self.assertIn('s2', result.model.snps.SNP.tolist())
        result = self.fit(transitions=False, n_components=1)
        self.assertEqual(result.model.snps.SNP.tolist(), ['s3'])
        with self.assertRaisesRegex(ValueError, 'No polymorphic'):
            self.fit(transitions=False, transversions=False)

    def test_reordered_subset_and_reversed_alleles(self):
        model = self.fit().model
        order = np.arange(len(self.g)-1, -1, -1)[::2]
        pref = self.write(2-self.g[order], 'reversed')
        snp = self.pref.with_suffix('.snp').read_text().splitlines()
        pref.with_suffix('.snp').write_text(''.join(snp[i].replace('A G', 'G A')+'\n' for i in order))
        result = ap.pca(pref, model=model, project_pops='T', min_snps=1, chunk_size=29, verbose=False)
        by_id = {sid: i for i, sid in enumerate(model.snps.SNP)}
        retained = [i for i in order if f's{i}' in by_id]
        mi = [by_id[f's{i}'] for i in retained]
        z = (self.g[retained, 12:].T - 2*model.frequencies[mi]) / model.scales[mi]
        expected = np.linalg.lstsq(model.loadings[mi], z.T, rcond=None)[0].T
        np.testing.assert_allclose(result.scores.iloc[:, 3:], expected, atol=1e-10)
        self.assertEqual(result.settings['n_allele_flips'], len(mi))

    def test_mismatch_and_failed_projection_rows(self):
        model = self.fit().model
        pref = self.write(self.g, 'bad_alleles')
        pref.with_suffix('.snp').write_text(pref.with_suffix('.snp').read_text().replace(' A G', ' A C'))
        with self.assertRaisesRegex(ValueError, 'Incompatible alleles'):
            ap.pca(pref, model=model, project_pops='T', verbose=False)
        with warnings.catch_warnings(record=True):
            result = ap.pca(pref, model=model, project_pops='T', mismatch='drop', verbose=False)
        self.assertEqual(result.diagnostics.status.tolist(), ['no_observed_snps']*3)
        self.assertTrue(result.scores.iloc[:, 3:].isna().all().all())
        g = self.g.copy()
        g[:, 12:] = np.nan
        first = int(model.snps.SNP.iloc[0][1:])
        g[first, 12] = 2
        with warnings.catch_warnings(record=True):
            result = ap.pca(self.write(g, 'sparse'), model=model, project_pops='T', min_snps=1, verbose=False)
        self.assertEqual(result.diagnostics.status.tolist(), ['rank_deficient', 'no_observed_snps', 'no_observed_snps'])
        with warnings.catch_warnings(record=True):
            result = ap.pca(self.pref, model=model, project_pops='T', min_snps=1000, verbose=False)
        self.assertEqual(set(result.diagnostics.status), {'insufficient_snps'})

    def test_formats_and_shared_af_stream(self):
        expected = self.fit(project_pops='T')
        prefixes = write_binary(self.root, self.pref, self.g)
        for pref in prefixes:
            with self.subTest(pref=pref):
                actual = ap.pca(pref, pops=['A', 'B'], project_pops='T', n_components=3,
                                min_snps=1, chunk_size=37, verbose=False)
                np.testing.assert_allclose(actual.scores.iloc[:, 3:], expected.scores.iloc[:, 3:], atol=1e-10)
                chunks = list(ap.iter_geno_to_afs(pref, pops=['A', 'B'], chunk_size=37, verbose=False))
                full = ap.anygeno_to_afs(pref, pops=['A', 'B'], verbose=False)
                np.testing.assert_allclose(pd.concat([c.afs for c in chunks]), full.afs)

    def test_roundtrip_and_corrupt_model(self):
        model = self.fit().model
        path = self.root / 'saved'
        ap.write_pca(path, model)
        with self.assertRaises(FileExistsError):
            ap.write_pca(path, model)
        loaded = ap.read_pca(path)
        self.assertEqual(loaded.settings, model.settings)
        a = ap.pca(self.pref, model=model, project_pops='T', min_snps=1, verbose=False)
        b = ap.pca(self.pref, model=loaded, project_pops='T', min_snps=1, verbose=False)
        pd.testing.assert_frame_equal(a.scores, b.scores)
        metadata_path = path / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['format_version'] = 99
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'format version'):
            ap.read_pca(path)
        model.loadings[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'loadings'):
            ap.pca(self.pref, model=model, project_pops='T', verbose=False)

    def test_archive_shape_and_frequency_validation(self):
        model = self.fit().model
        path = self.root / 'invalid_archive'
        ap.write_pca(path, model)
        with np.load(path / 'model.npz', allow_pickle=False) as arrays:
            payload = {key: arrays[key] for key in arrays.files}
        payload['frequencies'][0] = 1.5
        np.savez_compressed(path / 'model.npz', **payload)
        with self.assertRaisesRegex(ValueError, 'frequencies'):
            ap.read_pca(path)
        payload['frequencies'] = payload['frequencies'][:-1]
        np.savez_compressed(path / 'model.npz', **payload)
        with self.assertRaisesRegex(ValueError, 'shape'):
            ap.read_pca(path)

    def test_numeric_ids_and_na_labels_are_preserved(self):
        self.pref.with_suffix('.ind').write_text(''.join(
            f'{i:03d} U {"NA" if i < 12 else "001"}\n' for i in range(15)))
        snp = self.pref.with_suffix('.snp').read_text().splitlines()
        self.pref.with_suffix('.snp').write_text(''.join(
            f'{i:04d} ' + row.split(' ', 1)[1] + '\n' for i, row in enumerate(snp)))
        result = self.fit(pops='NA', iids=['000', '001', '002', '003'],
                          n_components=2, project_pops='001', project_iids='012')
        self.assertEqual(result.scores.iid.tolist(), ['000', '001', '002', '003', '012'])
        self.assertEqual(result.scores.population.tolist(), ['NA']*4 + ['001'])
        self.assertTrue(all(isinstance(sid, str) and len(sid) == 4 for sid in result.model.snps.SNP))

    def test_formats_with_missing_and_pseudohaploid_calls(self):
        rng = np.random.default_rng(3)
        g = 2 * (self.g > 0).astype(float)
        g[rng.random(g.shape) < .2] = np.nan
        eigen = self.write(g, 'haploid')
        options = dict(pops=['A', 'B'], project_pops='T', n_components=3,
                       maxmiss=.6, min_snps=1, chunk_size=23, verbose=False)
        expected = ap.pca(eigen, **options)
        prefixes = write_binary(self.root, eigen, g)
        for pref in prefixes:
            with self.subTest(pref=pref):
                actual = ap.pca(pref, **options)
                np.testing.assert_allclose(actual.scores.iloc[:, 3:], expected.scores.iloc[:, 3:], atol=1e-10)
        tgeno = prefixes[1]
        payload = tgeno.with_suffix('.tgeno').read_bytes()
        tgeno.with_suffix('.tgeno').write_bytes(payload[48:])
        actual = ap.pca(tgeno, format='tgeno', **options)
        np.testing.assert_allclose(actual.scores.iloc[:, 3:], expected.scores.iloc[:, 3:], atol=1e-10)

    def test_invalid_parameters_rank_and_genotypes(self):
        for kwargs in [dict(n_components=0), dict(n_components=True), dict(n_components=12),
                       dict(max_snps=0), dict(max_snps=1), dict(maxmiss=np.nan),
                       dict(minmaf=.6), dict(seed=-1), dict(seed=None), dict(chunk_size=0),
                       dict(min_snps=0), dict(shrinkage='bad'), dict(mismatch='bad')]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.fit(**kwargs)
        with self.assertRaisesRegex(ValueError, 'rank'):
            ap.pca(self.write(np.ones_like(self.g), 'rank'), n_components=1, verbose=False)
        missing = self.g.copy()
        missing[:, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'no observed retained SNPs'):
            ap.pca(self.write(missing, 'missing_reference'), pops=['A', 'B'],
                    n_components=3, maxmiss=1, verbose=False)
        bad = self.pref.with_suffix('.geno').read_text().splitlines()
        bad[50] = '7' + bad[50][1:]
        self.pref.with_suffix('.geno').write_text('\n'.join(bad)+'\n')
        with self.assertRaisesRegex(ValueError, 'encoded'):
            self.fit()

    def test_autoshrink_reference_coordinate_convention(self):
        rng = np.random.default_rng(91)
        g = rng.binomial(2, .1, (1000, 300)).astype(float)
        g[:, :150] = rng.binomial(2, .9, (1000, 150))
        pref = self.write(g, 'shrink', populations=['R'] * 300)
        result = ap.pca(pref, n_components=2, shrinkage='autoshrink', verbose=False)
        m = result.model
        ids = m.snps.SNP.str[1:].astype(int).to_numpy()
        z = (g[ids].T - 2*m.frequencies) / m.scales
        np.testing.assert_allclose(result.scores.iloc[:, 3:], (z @ m.loadings) * m.shrinkage_factors, atol=1e-10)
        self.assertTrue(np.any(m.shrinkage_factors < 1))
        projected = ap.pca(pref, model=m, project_pops='R', min_snps=1, verbose=False)
        np.testing.assert_allclose(projected.scores.iloc[:, 3:], z @ m.loadings, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
