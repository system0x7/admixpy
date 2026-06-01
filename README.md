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

Lower-level helpers are also exported for direct use, including allele-frequency
conversion (`anygeno_to_afs`, `eigenstrat_to_afs`, `plink_to_afs`,
`packedancestrymap_to_afs`, `tgeno_to_afs`), f2 block IO and access
(`get_f2`, `read_f2`, `write_f2`), and block/statistical utilities such as
`block_covariance`, `jackknife_cov`, `stats_to_loo`, and `est_to_loo`.

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
                           left weight   se     z
                       Turkey_N  0.686 0.01 52.82
      Russia_Samara_EBA_Yamnaya  0.101 0.01  8.51
Luxembourg_Loschbour_Mesolithic  0.119 0.01 18.64
               Iran_GanjDareh_N  0.094 0.01   7.18

rankdrop:
 f4rank dof      chisq         p  p_nested
      3   2       0.46     0.796 4.53e-244
      2   6    1133.78 1.02e-241         0
      1  12    3349.77         0         0
      0  20    6886.04         0       NaN

popdrop:
  pat                                                                    dropped  f4rank  dof      chisq          p  feasible status
 0000                                                                                  3    2       0.46      0.796      True   PASS
 0001                                                           Iran_GanjDareh_N       2    3      60.12   5.54e-13      True   FAIL
 0010                                            Luxembourg_Loschbour_Mesolithic       2    3     372.86   1.67e-80     False   FAIL
 0100                                                  Russia_Samara_EBA_Yamnaya       2    3      72.69   1.13e-15      True   FAIL
 1000                                                                   Turkey_N       2    3     834.89  1.17e-180     False   FAIL
 ...
```

## Citation

AdmixPy implements methods from Patterson et al. (2012) and Maier et al. (2023).

## License

[MIT License](LICENSE) .
