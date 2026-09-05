# AdmixPy

Python implementation of ADMIXTOOLS-style f-statistics, qpAdm, and qpWave.

> Fast f-statistics, qpAdm, and qpWave in Python.

## Installation

Requires Python 3.10 or newer.

Using a virtual environment is recommended:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the latest release from PyPI:

```bash
python -m pip install --upgrade pip
python -m pip install admixpy
```

Check that the package imports:

```bash
python -c "import admixpy; print(admixpy.__version__)"
```

### Updating

Upgrade an existing installation to the latest release:

```bash
python -m pip install --upgrade admixpy
```

## Alternative: installing from source

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the package from `pyproject.toml`:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Examples

Supported input layouts include EIGENSTRAT text, packed AncestryMap, TGENO, and
SNP-major PLINK binary files (`.bed/.bim/.fam`). For `.geno/.snp/.ind` inputs,
the genotype layout is detected from the file header/size; TGENO can also be provided as `.tgeno/.snp/.ind`.

## Basic API

The main convenience wrappers are:

```python
admixpy.f2(data, pop1=None, pop2=None, *, unique_only=True,
           resampling="pairwise_counts", **kwargs)
admixpy.fst(data, pop1=None, pop2=None, *, unique_only=True,
            resampling="pairwise_counts", fst_aggregation="block_ratios",
            **kwargs)
admixpy.f3(data, pop1=None, pop2=None, pop3=None, *, unique_only=True,
           resampling="pairwise_counts", verbose=True, **kwargs)
admixpy.f4(data, pop1, pop2=None, pop3=None, pop4=None, *, comb=True,
           unique_only=True, afprod=False, verbose=True, **kwargs)
admixpy.qpwave(data, left, right, ranks=None, left_base=None,
               right_base=None, rcond=1e-10, diag=0.0, max_nfev=None,
               verbose=True, **kwargs)
admixpy.qpadm(data, target, left=None, right=None, sources=None,
              fudge=0.0001, fudge_twice=False, iterations=20, getcov=True,
              return_f4=False, return_stats=False, return_cov=False,
              verbose=True, *, popdrop=True, **kwargs)
```

`data` can be a supported genotype dataset prefix or precomputed f2 data.
Population arguments can be strings or lists where the wrapper supports multiple
combinations. For PLINK `.bed/.bim/.fam` input, population labels are read from
the FID column of the `.fam` file.

For direct genotype input, `f3`, `f4`, `qpwave`, and `qpadm` default to
`allsnps=True`, matching the ADMIXTOOLS1-style behavior of estimating each
statistic from its available SNPs. For precomputed f2 input, `allsnps` defaults
to `False` and the standard f2-based behavior is used. Pass `allsnps=False` to
restrict direct-genotype models to SNPs shared across the required populations.
Both direct-genotype modes use the direct per-SNP estimator and report the exact
SNP count for each statistic; raw input is reduced to pairwise f2 values only
when explicitly creating an f2 cache. An optional `model` column in an f4
combination table scopes the shared panel separately for each model.

Direct genotype `f3` also defaults to `allsnps=True` and is calculated per SNP.
By default, its corrected numerator is divided by unbiased target
heterozygosity. Set `outgroupmode=True` to return the unnormalized f3 numerator;
that raw mode is directly comparable to f2-derived f3 and to original `qp3Pop`
outgroup mode after removing the latter's factor of 1000.

Direct f3 and f4 genotype calculations (including the f4 calculations
for qpAdm and qpWave) read the genotype file once by default and hold the
complete SNP-by-population allele-frequency and count tables in memory. For
datasets that do not fit comfortably in RAM, set `stream=True` to use two
bounded-memory passes with 250,000 SNPs per chunk by default. The chunk size
can be adjusted with `chunk_size`.

qpAdm uses `qpadm(data, target, left, right)`; there is no separate positional
outgroup argument. `sources` is an alias for `left`, so supply only one of them.
Both population lists must contain unique names, and `left_base`, if supplied,
must equal the target. Invalid population counts are rejected before genotype
data is read.

By default, qpAdm also fits every nonempty subset of the sources. This requires
`2**len(left) - 1` subset fits. Use `popdrop=False` to skip this table while
retaining the weights and rank tests; `result.popdrop` will be `None`.
`qpadm_multi(..., full_results=False)` automatically skips subset fits and
weight standard errors because it returns only the rank-test tables.

Population-drop `chisq`, `p`, and weights use the covariance of each retained
subset, inverted after applying that subset's regularization. They match
independent fits using the same stored SNP panel and block layout, including
`fudge_twice`; dropping a source does not recover additional SNPs.

Nested comparisons include every one-source drop against the full model,
followed by a feasible child with the smallest chi-square at each smaller size.
`nested_parent` records the actual parent pattern. For these comparisons only,
selected models are refitted using submatrices of the full model's regularized
covariance. `nested_chisq` records those fit statistics, and `chisqdiff` and
`p_nested` use their differences. Consequently, `chisqdiff` need not equal the
difference between the standalone `chisq` entries. A materially negative
difference from incomplete numerical convergence produces `p_nested=NaN`.
These are nominal chi-square tests, without adjustment for model selection.
The selected comparisons require at most `2*len(left) - 2` extra fits.

For direct use of `qpadm_popdrop`, pass the original covariance as `cov=...`
alongside `qinv` to obtain the same subset regularization as `qpadm`. Without
`cov`, the inverse of the supplied `qinv` is treated as an already regularized
covariance; a singular `qinv` requires the original covariance explicitly.

For genotype batches, `f4_model_cache` retains block statistics and computes
covariance only for the requested contrasts. Its `cache.stats.cov` is initially
`None`; use `f4_stats(cache, ...)` or pass the cache to qpAdm/qpWave to obtain the
required covariance. The cache records its resampling method in
`cache.stats.resampling`. Requests using a different method raise an error;
rebuild the cache with the desired `resampling` setting.

Direct f4 automatically batches suitable contrasts into matrix products.
Repeated-population bias corrections, target-heterozygosity normalization, and
other unsupported combinations retain the general per-contrast calculation.
Pair-count computation and TGENO decoding also use fewer temporary arrays.

Lower-level helpers are also exported for direct use, including allele-frequency
conversion (`anygeno_to_afs`, `eigenstrat_to_afs`, `plink_to_afs`,
`packedancestrymap_to_afs`, `tgeno_to_afs`), f2 block IO and access
(`get_f2`, `read_f2`, `write_f2`), and block/statistical utilities such as
`iter_geno_to_afs`, `f3_stats_from_geno`, `block_covariance`,
`jackknife_cov`, `stats_to_loo`, and `est_to_loo`.

### SNP selection, missingness, and small samples

By default, `f2` excludes SNPs with identical allele frequencies in every
loaded population, while `fst` retains them. This matches the ADMIXTOOLS
default but means the two statistics can use different SNP sets. Use
`poly_only=True` to both calls when they should be directly comparable.

AdmixPy uses `resampling="pairwise_counts"` as default for data
with missing genotypes: each population pair is weighted by the SNP
observations actually available for that pair. Pairwise `f2` and `fst` result
tables include `n`. Set `resampling="nominal_blocks"` to reproduce the older
behavior in which every pair uses nominal block sizes. Direct-genotype f4 with
`allsnps=True` already uses per-statistic counts on a common SNP intersection.
Direct-genotype f4 with `allsnps=False` uses the intersection shared by the
requested model. Cached pairwise f3/f4 instead defines a pairwise-available
estimator and cannot reconstruct either common intersection.

Different missing-block patterns can produce a covariance matrix with negative
eigenvalues under pairwise resampling. qpAdm and qpWave reject such matrices
instead of returning an invalid chi-square statistic. Inspect population
coverage and usable blocks; recomputing from genotype input with
`allsnps=False` uses a common SNP panel. Tiny negative eigenvalues attributable
to floating-point roundoff are tolerated.

F2 cache creation and reading retain blocks with missing pair estimates by
default (`remove_na=False`). Set `remove_na=True` to discard every block that
is not finite (`NaN`) for all requested population pairs.

FST cache files additionally retain numerator and denominator sums. The
default `fst_aggregation="block_ratios"` averages stored block estimates.
Set `fst_aggregation="pooled_components"` to recompute full-data and
leave-one-block-out FST as ratios of pooled numerator and denominator sums.

Bias-corrected f2 and FST require at least two independent allele observations
in each population. SNP values with a count below two are excluded with a
warning when `apply_corr=True`. Setting `apply_corr=False` explicitly requests
the finite but sampling-biased raw estimate; the Hudson FST denominator remains
`(p1-p2)^2 + p1(1-p1) + p2(1-p2)` in either mode.

Cache files without real per-pair SNP counts are rejected and must be rebuilt.

With `minac2=2`, populations that never have two allele observations are exempt
from the minimum-count filter. If all populations are exempt, other SNP filters
still apply. `minac2=True` instead requires two observations in every population.

Run an f4 statistic from a supported genotype dataset prefix:

```python
import admixpy

prefix = "/path/to/dataset_prefix"

result = admixpy.f4(
    prefix,
    "Mbuti",
    "Germany_ViesenhaeuserHof_EN",
    "Sardinian",
    "French",
)

print(result)
```

qpAdm can be run the same way from a Python REPL:

```python
>>> import admixpy as a
>>> prefix = "/path/to/dataset_prefix"
>>> target = "Sardinian"
>>> left = ["Turkey_N", "Russia_Samara_EBA_Yamnaya", "Luxembourg_Loschbour_Mesolithic", "Iran_GanjDareh_N"]
>>> right = ["Chimp", "Turkey_Epipaleolithic", "Georgia_KotiasKlde_Mesolithic", "Russia_Vologda_Mesolithic", "Switzerland_Epipaleolithic", "Iran_BeltCave_Mesolithic"]
>>> res = a.qpadm(prefix, target=target, left=left, right=right)
>>> res
QpAdmResult(target='Sardinian')

weights:
                           left weight     se     z
                       Turkey_N  0.686  0.013 52.45
      Russia_Samara_EBA_Yamnaya  0.102  0.012  8.54
Luxembourg_Loschbour_Mesolithic  0.119 0.0064 18.57
               Iran_GanjDareh_N  0.094  0.013  7.11

rankdrop:
f4rank dof   chisq         p  p_nested
     3   2    0.53     0.769 9.39e-242
     2   6 1123.16 2.03e-239         0
     1  12 3317.58         0         0
     0  20 6849.91         0       NaN

popdrop:
 pat                                                                    dropped f4rank dof   chisq         p  feasible status
0000                                                                                 3   2    0.53     0.769      True   PASS
0001                                                           Iran_GanjDareh_N      2   3   58.58  1.18e-12      True   FAIL
0010                                            Luxembourg_Loschbour_Mesolithic      2   3   370.2  6.29e-80     False   FAIL
0100                                                  Russia_Samara_EBA_Yamnaya      2   3   73.09  9.29e-16      True   FAIL
1000                                                                   Turkey_N      2   3  825.23 1.46e-178     False   FAIL
 ...
```

## Citation

AdmixPy implements methods from Patterson et al. (2012) and Maier et al. (2023).

## License

[MIT License](LICENSE) .
