# AdmixPy

Fast population-genetic f-statistics, qpAdm, qpWave, and single-pulse admixture dating
in Python. Requires Python 3.10 or newer.

For function signatures, options, result objects, and method details, see the
[API reference](REFERENCE.md). See the [changelog](CHANGELOG.md) for changes.

## Installation

Using a virtual environment is recommended:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install admixpy
```

Upgrade an existing installation with `python -m pip install --upgrade admixpy`.
To install a source checkout, run `python -m pip install -e .` from its root.
Check the installed version with:

```bash
python -c "import admixpy; print(admixpy.__version__)"
```

## Input data

Use a dataset prefix without a file extension. Supported layouts are
EIGENSTRAT text, packed AncestryMap, TGENO, and SNP-major PLINK binary files.
Population names must match the `.ind` population column or PLINK `.fam` FID.
Genotype layout detection is automatic.

Most f-statistic and ancestry-model functions also accept precomputed f2 blocks.
Admixture dating requires individual genotypes and a genetic map. See
[input conventions](REFERENCE.md#input-conventions) and
[missingness and resampling](REFERENCE.md#missingness-and-resampling) before
comparing analyses with different input types.

## Quick start

The population names below are placeholders, replace them with labels in your data.

```python
import admixpy as ap

prefix = "/path/to/dataset_prefix"

# f4 statistic
stats = ap.f4(prefix, "Outgroup", "Reference", "Source", "Target")
print(stats)

# qpAdm: sources on the left, reference populations on the right
model = ap.qpadm(
    prefix,
    target="Target",
    sources=["SourceA", "SourceB"],
    right=["Outgroup", "ReferenceA", "ReferenceB", "ReferenceC"],
)
print(model)
model.weights.to_csv("ancestry_weights.csv", index=False)
```

qpAdm tests a proposed ancestry model relative to the selected references.
A model that is not rejected is not uniquely established as the historical model.
See [qpAdm and qpWave](REFERENCE.md#qpadm-and-qpwave) for rank tests, source
subsets, rotating models, batching, and result fields.

## Admixture dating

```python
date = ap.admixture_date(
    prefix,
    target="Target",
    sources=["SourceA", "SourceB"],
    # target_iids=["sample_id"],  # Optional individual selection
)
print(date)
date.curve.to_csv("admixture_covariance.csv", index=False)
```

This DATES-inspired estimator fits a single exponential to within-individual
ancestry covariance. It supports a single target individual or a group.
Use one representation per person and group individuals with similar sampling
ages and ancestry histories. Dates are in generations before sampling.
A `resolved` result summarizes numerical checks, it does not test whether the
chosen references describe a valid demographic model.

See [admixture dating](REFERENCE.md#admixture-dating) for every option, map units,
generation conventions, uncertainty, result tables, and diagnostic flags.

## Available analyses

| Analysis | Functions | Reference |
|---|---|---|
| f-statistics and differentiation | `f2`, `fst`, `f3`, `f4` | [Statistics](REFERENCE.md#statistics) |
| Relative affinity and ancestry ratios | `f4_proximity`, `f4_ratio` | [Statistics](REFERENCE.md#statistics) |
| Ancestry models and rank tests | `qpadm`, `qpwave`, batch and rotation helpers | [Models](REFERENCE.md#qpadm-and-qpwave) |
| Single-pulse dating | `admixture_date` | [Dating](REFERENCE.md#admixture-dating) |
| Genotype conversion and filtering | Readers, allele-frequency conversion, SNP selection | [Genotypes](REFERENCE.md#genotype-utilities) |
| Reusable statistics | f2 IO, f4 caches, block and jackknife helpers | [Caches and blocks](REFERENCE.md#caches-and-blocks) |

## License

[MIT License](LICENSE).
