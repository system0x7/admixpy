# AdmixPy

Fast Python implementation of ADMIXTOOLS-style f-statistics, qpAdm, and qpWave.

> Runs faster than ADMIXTOOLS 2 on equivalent workloads.

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

Supported input formats include EIGENSTRAT (`.geno/.snp/.ind`), packed
AncestryMap (`.geno/.snp/.ind`), TGENO (`.tgeno/.snp/.ind`), and SNP-major PLINK
binary files (`.bed/.bim/.fam`).
TGENO conversion streams samples by default for speed; use `tgeno_chunked=True`
with `anygeno_to_afs`, `f2_from_geno`, or `f4_stats_from_geno` to opt into
lower-memory SNP chunking.

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
>>> left = ["Turkey_N", "Luxembourg_Loschbour_Mesolithic", "Iran_GanjDareh_N", "Russia_Samara_EBA_Yamnaya", "Jordan_PPNB"]
>>> right = ["Mbuti", "Turkey_Epipaleolithic", "Georgia_KotiasKlde_Mesolithic", "Russia_Vologda_Mesolithic", "Switzerland_Epipaleolithic", "Iran_BeltCave_Mesolithic"]
>>> res = a.qpadm(prefix, target=target, left=left, right=right)
>>> res
```

## Citation

admixpy implements methods from Patterson et al. (2012) and Maier et al. (2023).

## License

[MIT License](LICENSE) .
