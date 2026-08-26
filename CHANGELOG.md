# Changelog

All notable changes to AdmixPy are documented in this file.

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
