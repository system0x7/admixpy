# Changelog

All notable changes to AdmixPy are documented in this file.

## 1.0.6 - 2026-09-05

This release includes the qpAdm validation, covariance, population-drop, and
f-statistics performance improvements described below.

### Fixed

- Reject conflicting qpAdm source aliases, duplicate reference populations,
  invalid source/reference counts, and a left base different from the target
  before reading genotype data.
- Reject indefinite qpAdm/qpWave covariance matrices before testing model fit.
- Record f4 cache resampling methods and reject mismatched requests.
- Handle `minac2=2` when every population has at most one allele observation.
- Preserve explicit contrast bases in cached batch calculations.
- Fit population-drop models by inverting each subset covariance, with
  subset-specific regularization matching independent fits on the stored SNPs.
- Compare genuine parent/child models for population-drop nested tests, using
  one shared covariance treatment. Expose `nested_parent` and `nested_chisq`;
  suppress p-values for materially negative chi-square differences.

### Changed

- Batch suitable direct f4 contrasts using matrix products, preserving the
  general calculation for bias corrections and other unsupported cases.
- Compute pair counts without SNP-by-population-by-population arrays and decode
  packed TGENO bytes together, including unaligned SNP ranges.
- Compute genotype-batch covariance only for requested contrasts. The full
  batch cache initially has `stats.cov=None`.
- Add `qpadm(..., popdrop=False)` to skip source-subset fits. Summary-only
  `qpadm_multi` calls skip subset fits and weight covariance automatically.

## 1.0.4 - 2026-08-26

### Changed

- Simplified singleton-observation warnings by omitting affected population names.

## 1.0.3 - 2026-08-26

### Changed

- Improved diagnostics and added manual PyPI publishing.

## 1.0.2 - 2026-08-26

### Changed

- Preserve f2 cache blocks with missing pair estimates by default and retain
  pairwise SNP counts for missing-data-aware resampling.
- Report affected populations and contrasts when finite-sample correction or
  non-finite f4 inputs prevent f2, qpAdm, or qpWave calculations.

## 1.0.1 - 2026-08-21

### Changed

- Improved direct f3/f4, covariance, chromosome parsing, mutation filtering,
  and pseudohaploid-detection performance.
- Raise a clear error when a pseudohaploid singleton is used as an f3 target
  requiring bias correction or target-heterozygosity normalization.

## 1.0.0 - 2026-08-10

First stable release.

### Features

- Fast whole-dataset f-statistics, qpAdm, and qpWave analysis.
- f2, Hudson FST, f3, and f4 statistics with block-jackknife uncertainty.
- qpAdm and qpWave model fitting, rank tests, and qpAdm population-drop models.
- Direct EIGENSTRAT, PACKEDANCESTRYMAP, TGENO, and SNP-major PLINK input.
- Precomputed f2 block caches with per-pair SNP counts and FST components.
- Bounded-memory streaming for direct f3 and f4 workflows.
- Explicit missing-data, SNP-selection, and pseudohaploid correction options.

### Requirements

- Python 3.10 or newer is required.
