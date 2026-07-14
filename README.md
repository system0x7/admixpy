# AdmixPy

Fast Python implementation of ADMIXTOOLS-style f-statistics, qpAdm, and qpWave.

> Fast f-statistics, qpAdm, and qpWave in Python.

## Setup

Requires Python 3.10 or newer.

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

Check that the package imports:

```bash
python -c "import admixpy; print(admixpy.__file__)"
```

### Updating an existing clone

```bash
cd /path/to/admixpy
git pull --ff-only
```

After updating, run the two installation commands in Setup again.

## Alternative

If you only want to install the dependencies without installing the package:

```bash
python -m pip install -r requirements.txt
```

## Examples

Supported input layouts include EIGENSTRAT text, packed AncestryMap, TGENO, and
SNP-major PLINK binary files (`.bed/.bim/.fam`). For `.geno/.snp/.ind` inputs,
the genotype layout is detected from the file header/size; TGENO can also be provided as `.tgeno/.snp/.ind`.
For fastest repeated f-statistic workloads, prefer SNP-major binary formats such
as packed AncestryMap or PLINK `.bed/.bim/.fam`; TGENO is supported, but its
sample-major layout is usually less efficient for SNP-block scans.

## Basic API

The main convenience wrappers are:

```python
admixpy.f2(data, pop1=None, pop2=None, **kwargs)
admixpy.fst(data, pop1=None, pop2=None, **kwargs)
admixpy.f3(data, pop1, pop2, pop3, **kwargs)
admixpy.f4(data, pop1, pop2, pop3, pop4, allsnps=True, **kwargs)
admixpy.qpwave(data, left, right, ranks=None, allsnps=True, **kwargs)
admixpy.qpadm(data, target, left, right, allsnps=True, **kwargs)
```

`data` can be a supported genotype dataset prefix or precomputed f2 data.
Population arguments can be strings or lists where the wrapper supports multiple
combinations. For PLINK `.bed/.bim/.fam` input, population labels are read from
the FID column of the `.fam` file.

For direct genotype input, `f4`, `qpwave`, and `qpadm` default to
`allsnps=True`, matching the ADMIXTOOLS1-style behavior of estimating each f4
statistic from its available SNPs. Use `allsnps=False` to restrict the model to
SNPs shared across the required populations. Precomputed f2 input keeps the
standard f2-based behavior.

Direct genotype `f3` also defaults to `allsnps=True` and is calculated per SNP.
By default, its corrected numerator is divided by unbiased target
heterozygosity. Set `outgroupmode=True` to return the unnormalized f3 numerator;
that raw mode is directly comparable to f2-derived f3 and to original `qp3Pop`
outgroup mode after removing the latter's factor of 1000.

The relevant small-sample quantity is the number of independent allele
observations, not merely the number of individuals. AdmixPy corrects every
population repeated across the two factors of `f3(A; B, C)`. Thus the target
`A` is corrected, and in `f3(A; B, B)` source `B` is corrected as well.

| Data | Scope | Recommended setting |
|---|---|---|
| Known modern diploid data | All direct genotype statistics | `adjust_pseudohaploid=False` if known diploid (otherwise auto-detected); `apply_corr=True` |
| Ancient or mixed-ploidy data | All direct genotype statistics | `adjust_pseudohaploid=True, apply_corr=True` (auto-detected per sample) |
| ADMIXTOOLS2-normalized direct f3 | f3 only | `outgroupmode=False` (default) |
| Raw/original outgroup f3 | f3 only | `outgroupmode=True` |
| Diploid singleton target or repeated source | f3 only | Keep `apply_corr=True`; two called alleles make correction possible |
| Pseudohaploid singleton used only as a distinct source | f3 only | Allowed with `apply_corr=True` because it occurs linearly |
| Pseudohaploid singleton target or repeated source | f3 only | Unbiased correction is not possible; affected SNPs are excluded |
| Intentionally biased singleton estimate | f3 only | `outgroupmode=True, apply_corr=False`; exploratory/legacy |
| Missingness differs among populations or blocks | f2, FST, f3, and cached f4 | `resampling="pairwise_counts"` (default) |
| In-memory blocks without SNP counts | Precomputed-block workflows | `resampling="nominal_blocks"`; incomplete on-disk caches must be rebuilt |
| Maximum available SNPs per combination | Direct f3/f4, qpWave, and qpAdm | `allsnps=True` (direct-genotype default) |
| Common SNP set across a model | Direct f3/f4, qpWave, and qpAdm | `allsnps=False` |
| Only segregating sites | Direct f-statistics | `poly_only=True`; equal non-boundary frequencies are retained |

Direct f3 and f4 genotype calculations stream by physical SNP block by default.
They make two sequential passes over the genotype file so filtering and block
boundaries remain identical while memory stays bounded. Set `stream=False` to
use the materialized SNP-by-population path.

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
behavior in which every pair uses nominal block sizes. Raw-genotype f4 with
`allsnps=True` already uses per-statistic counts on a common SNP intersection.
Cached pairwise f3/f4 instead defines a pairwise-available estimator and cannot
reconstruct that common intersection.

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
