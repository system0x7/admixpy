# AdmixPy reference

[Installation and quick start](README.md) · [Changelog](CHANGELOG.md)

This reference describes the public package API. All functions below are
available as `admixpy.<name>`. Signatures show parameter names and defaults;
type annotations are omitted for readability. `__version__` is the package
version string. The complete exported callable index follows the method notes.

## Contents

- [Input conventions](#input-conventions)
- [Statistics](#statistics)
- [qpAdm and qpWave](#qpadm-and-qpwave)
- [Missingness and resampling](#missingness-and-resampling)
- [Admixture dating](#admixture-dating)
- [Genotype utilities](#genotype-utilities)
- [Caches and blocks](#caches-and-blocks)
- [Result objects](#result-objects)
- [Public API signatures](#public-api-signatures)

## Input conventions

A genotype prefix names `.geno/.snp/.ind`, `.tgeno/.snp/.ind`, or
`.bed/.bim/.fam` files without their extensions. EIGENSTRAT-family formats are
detected from their headers and sizes. PLINK must be SNP-major.

An `.ind` row has three whitespace-separated fields: `iid`, `sex`, and
`population`. Labels cannot contain spaces. `.fam` population labels are read
from FID; individual identifiers come from IID. Use actual sample IDs when
selecting individuals, including suffixes such as `.SG` or `.DG`. Multiple
representations of the same person are not automatically deduplicated.

Genotype values use 0/1/2 for diploid allele dosage, 0/2 for pseudo-haploid
calls, and missing values internally as NaN. Readers decode the missing-value
codes of each file format. `adjust_pseudohaploid` controls reference/population
allele counting. With automatic detection, observed samples without
heterozygotes in the detection panel are treated as pseudo-haploid. This can
misclassify sparse diploid samples; disable adjustment when inappropriate.

Population-list support is function-specific: `f2`/`fst` generate population
pairs; `f3` generates triples; `f4` accepts lists or an explicit contrast table.
`f4(..., comb=True)` generates Cartesian products; `comb=False` combines entries
by position. `unique_only` controls symmetry deduplication. `f4_ratio` also
supports explicit model rows; its five populations must be distinct.

Most statistics accept a genotype prefix, an `F2Blocks` object, or a directory
written by `write_f2`. `AfData` is used by allele-frequency conversion and block
construction helpers. The direct f3/f4 readers accept prefixes; they do not
accept population allele-frequency tables in place of individual genotypes.
`admixture_date` accepts a genotype prefix only. f4 caches are accepted by the
f4/qpAdm/qpWave paths described below; they are not universal inputs.

## Statistics

| Function | Definition or purpose | Output |
|---|---|---|
| `f2` | Pairwise squared allele-frequency difference, with optional finite-sample correction | Population pairs, `est`, `se`; `n` under pair-count resampling |
| `fst` | Hudson FST, with block-ratio or pooled-component aggregation | Population pairs, `est`, `se`; `n` under pair-count resampling |
| `f3` / `qp3pop` | `f3(pop1; pop2, pop3)`; direct input normally normalizes by target heterozygosity | Population triples, `est`, `se`, `z`, `p`, `n` |
| `f4` / `qpdstat` | Contrast `f4(pop1, pop2; pop3, pop4)` | Population quadruples, `est`, `se`, `z`, `p`, `n` |
| `f4_proximity` | RMS f4 contrasts between target and each candidate across references | `population`, `rms_f4`, `rms_z`, `n_snps` |
| `f4_ratio` | `f4(target, source2; reference, outgroup) / f4(source1, source2; reference, outgroup)` | `F4RatioResult` |

`f3` is an alias of `qp3pop`; `f4` delegates to `qpdstat`. The exported
`qpdstat` computes f4 statistics; it does not expose a normalized D-statistic
switch. `format_fstats` returns a copy with formatted numeric display values;
keep the original numeric tables for further calculation.

For `f4_ratio`, the ratio estimates source1 ancestry under the assumed
two-source topology; it does not test that topology. `confidence` controls
normal confidence limits. `denominator_z_min=3.0` flags weak denominators;
sign changes across deletions flag instability. Undefined or unreliable ratios
retain diagnostics and suppress confidence intervals. `return_blocks=True`
retains block estimates, deletion ratios, weights, and SNP counts.

## qpAdm and qpWave

`qpadm` fits a target from `left` (or `sources`) relative to `right`. Its main
rank is `len(sources) - 1`. It reports ancestry weights, uncertainty, rank tests,
and optionally source-subset fits. `getcov=False` skips weight uncertainty.
`return_f4`, `return_stats`, and `return_cov` retain the fitted f4 table,
`QpWaveStats`, and weight covariance respectively. `fudge` regularizes the f4
covariance and fitting solves; `fudge_twice` controls the additional covariance
regularization path. `iterations` controls alternating fitting updates.

`qpwave` tests specified ranks of the matrix of f4 contrasts. Bases default
to the first population on each side. `ranks=None` tests all ranks with positive
degrees of freedom. `rcond` controls singular-value cutoffs; `diag` adds a
covariance diagonal term. `max_nfev` limits nonlinear optimizer evaluations.
The output contains `rank`, `dof`, `chisq`, `p`, `n_stats`, `converged`, and
`optimizer_cost`. Rank-test p-values depend on the chosen reference panel.

Batch functions accept a DataFrame or list of model dictionaries. qpAdm rows
require `target`, `left`, and `right`; `left` and `right` contain population
lists. qpWave rows use `left` and `right`, optionally with a `target` prepended
to the left panel. `use_cache=True` reuses genotype-derived statistics.
`qpadm_multi(full_results=True)` returns one row per model with a `result`
column containing `QpAdmResult`; `False` concatenates rank-test tables.
`qpwave_multi` returns a combined rank-test table with model IDs and panels.

### Shared behavior and detailed model options

The following notes also describe statistics and cache behavior shared by the
model wrappers. For exact defaults, use the [signature index](#public-api-signatures).

Genotype inputs and precomputed f2 inputs differ as follows.
`admixture_date` requires a genotype prefix; f2 caches and population
allele-frequency tables do not preserve its within-individual signal.
Population arguments can be strings or lists where the wrapper supports multiple
combinations. For PLINK `.bed/.bim/.fam` input, population labels are read from
the FID column of the `.fam` file.

`f4_ratio` accepts strings or lists for all five population arguments.
The default `comb=True` evaluates every combination; `comb=False` pairs
entries by position and requires equal lengths, treating a string as one entry.
Use `models=` for explicit rows instead of individual population arguments.

For direct genotype input, `f3`, `f4`, `qpwave`, and `qpadm` default to
`allsnps=True`, matching the ADMIXTOOLS1-style behavior of estimating each
statistic from its available SNPs. For precomputed f2 input, `allsnps` defaults
to `False` and the standard f2-based behavior is used. Pass `allsnps=False` to
restrict direct-genotype models to SNPs shared across the required populations.
Both direct-genotype modes use the direct per-SNP estimator and report the exact
SNP count for each statistic; raw input is reduced to pairwise f2 values only
when explicitly creating an f2 cache. An optional `model` column in an f4
combination table scopes the shared panel separately for each model.

`f4_proximity` ranks candidates by the RMS of
`f4(outgroup, ref; candidate, target)` across `refs`. It returns `population`,
`rms_f4`, `rms_z`, and the minimum contrast SNP count `n_snps`
(NaN for precomputed f2 data). Rankings depend on the reference panel.
The `outgroup` argument must be supplied explicitly. All entries in `refs`
are treated as references. Choose an outgroup appropriate to the comparison,
rankings depend on both the outgroup and reference panel.
It defaults to `allsnps=False`, so high missingness can reduce the shared SNP count.

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

Rotating candidate sources between the left and right panels is available with
`qpadm_rotate`. Unused candidates join `rightfix` on the right. Restrict source
counts with `source_sizes`; omit it to test all counts with positive degrees of
freedom. Preview the model table with `qpadm_rotate_models` using the same
population arguments, without a `data` argument.

```python
rotation = admixpy.qpadm_rotate(
    data, target="Target", leftright=["SourceA", "SourceB", "SourceC"],
    rightfix=["ReferenceA", "ReferenceB", "ReferenceC"],
    source_sizes=[1, 2], full_results=True, getcov=True,
)
rotation.models  # One row per model; population lists, fit and execution status
rotation.weights  # One row per model/source: weight, se, z
rotation.errors  # Model ID, error type and message
rotation.settings  # Version and run options
rotation.weights.to_csv("rotation_weights.csv", index=False)
```

Rotation results retain generation order and population lists. Model IDs are
stable across source-count filters and identify the ordered population panels
and bases, not the dataset or fitting settings. `full_results=False` returns
fit summaries with missing feasibility and an empty weights table.
`full_results=True` includes weights and feasibility; `getcov=True` additionally
computes standard errors. Population-drop fits are skipped. There is no custom
export method: use pandas on individual tables.

By default, rotation errors raise. Set `on_error="record"` to retain model-level
fit errors and continue; input validation and cache-construction errors still
raise. An execution status of `ok` means the fit completed, regardless of its
p-value. Sorting by p-value does not establish the best ancestry model because
the reference panel changes between rotations.

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


## Missingness and resampling

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


### Forwarded genotype and filtering options

`**kwargs` forwards options down the relevant input path, not to every
function in the package. `f2` and `fst` use `get_f2` / `f2_from_geno` for raw
input. Direct f3 uses `f3_stats_from_geno`; f4 and the ancestry-model wrappers
use `f4_stats` / `f4_stats_from_geno`. See those signatures before combining
options. Cached input cannot recover SNPs already filtered out.

| Option | Meaning |
|---|---|
| `blgsize` | Block span; default 0.05 in the genetic-position column. Values >=100 use physical positions in bp. |
| `maxmiss` | Maximum fraction of loaded populations with no allele observations at a SNP. Direct f3/f4 may choose a default from `allsnps` and model scope. |
| `minmaf`, `maxmaf` | Minor-allele-frequency limits across loaded populations, weighted by allele counts. |
| `minac2` | Require >=2 allele observations: `True` for all populations; `2` exempts populations never having two observations. |
| `poly_only` | Restrict to SNPs with differing frequencies across loaded populations. Block constructors can apply this selectively by statistic name. |
| `transitions`, `transversions` | Include each mutation class; both default to True. |
| `auto_only` | Apply the existing numeric-chromosome autosome filter; this is separate from dating's `autosomes_only`. |
| `keepsnps` | Explicit SNP-ID whitelist. In `discard_from_aftable`, this replaces the other SNP filters. |
| `outpop` | Restrict to SNPs polymorphic in this population; optionally scale by its heterozygosity. |
| `outpop_scale` | Apply `1 / (p_out * (1-p_out))` SNP weights when an `outpop` is supplied. |
| `apply_corr` | Apply finite-sample corrections where supported; False requests raw estimates. |
| `format` | Override genotype detection with `eigenstrat`, `packedancestrymap`, `tgeno`, or `plink`. |
| `adjust_pseudohaploid` | Detect pseudo-haploid reference samples for allele counting; integer values can set the detection-panel size in conversion/statistics helpers. Dating accepts a boolean. |
| `chunk_size` | SNPs decoded per chunk; default 10,000 for conversion/f2 helpers, 250,000 for streamed direct f3/f4. Chunked decoding alone does not imply bounded-memory statistics. |
| `stream` | Two bounded-memory passes for direct f3/f4 and wrappers using them. Not supported by dating. |
| `tgeno_chunked` | Select the chunked TGENO conversion path where exposed. |
| `remove_na` | Drop blocks missing any requested pair estimate when True; default False. |
| `afprod` | Use the allele-product path where exposed; see the cache construction helpers. |
| `keep_blocks`, `keep_loo`, `covariance` | Control retention/computation in `f4_stats`. qpAdm/qpWave require covariance. |
| `resampling` | `pairwise_counts` or `nominal_blocks`, subject to the estimator and stored cache metadata. |

The statistics block builder uses the supplied genetic-position column as-is;
it does not perform the dating function's format-specific M/cM conversion.
Supply positions and `blgsize` in consistent units. Unlike dating,
`get_block_lengths` can fall back to 2 Mb physical blocks when no useful genetic
map is present, or to chromosome blocks if neither map is usable.

## Admixture dating

Estimate a date using exactly two reference populations:

```python
result = admixpy.admixture_date(
    prefix,
    target="Target",
    sources=["SourceA", "SourceB"],
    return_jackknife=True,
)
print(result)
result.summary       # Date (est), SE, confidence limits, coverage and statuses
result.parameters    # generations, decay_rate, amplitude, offset
result.curve         # Observed/fitted covariance, residuals, distances, pair counts
result.diagnostics   # Fit error, condition, boundary and jackknife diagnostics
result.settings      # Estimator, units, generation convention and run options
result.jackknife     # Chromosome-deletion fits; optional table
result.curve.to_csv("admixture_covariance.csv", index=False)
```

`target` names one population. Use `target_iids=["sample_id"]` to restrict the
analysis to specific individual IDs (`iid`) in that population. Omit it to use
all individuals in the target population. Result settings record retained
IDs under `target_iids`. Target and source populations must be distinct.
Targets with fewer than `min_snps` observed informative SNPs are skipped with
a warning identifying the sample and observed count. Excluded IDs and reasons
are recorded in `settings['skipped_individuals']`, and their count in
`diagnostics.n_skipped_individuals`. Only retained targets contribute to the
curve, counts, and jackknife. If none remain, the function raises an error.
An ancestry regression coefficient outside (0, 1) still raises an error
identifying that sample. The regression coefficients stored in
`settings` are nuisance parameters, not validated ancestry-proportion estimates.

The function supports EIGENSTRAT text, packed AncestryMap, TGENO, and SNP-major
PLINK inputs through the existing genotype decoders. PLINK population labels
come from the `.fam` FID column. Selected genotypes are loaded into memory;
this function does not currently offer streaming. Pseudo-haploid calls must
use the existing 0/2 encoding; diploid calls use 0/1/2. Reference allele counts
use admixpy's existing pseudo-haploid detection, which can be disabled with
`adjust_pseudohaploid=False`.

All distance parameters and output distances are **Morgans**. Input genetic
positions default to Morgans for EIGENSTRAT-family files and centiMorgans for
PLINK `.bim` files. Set `map_unit="M"` or `map_unit="cM"` to override this.
Despite its internal name `cm`, the EIGENSTRAT SNP column is interpreted as
Morgans here. Maps must be finite and nondecreasing within each chromosome.
Negative starting coordinates are valid: each chromosome is normalized to its
first coordinate, preserving all pairwise distances. Original origins in Morgans
are retained in `settings["map_origins_morgans"]`.
An entirely flat map on a chromosome with multiple SNPs raises an
error; there is no conversion from physical distance. Human autosomes 1–22
are selected by default. Set `autosomes_only=False` to include other chromosome
labels, including nonhuman datasets; make that selection with appropriate
ploidy and map information.

The estimator independently fits a two-reference genome-wide ancestry
coefficient for each target individual. It then computes the DATES v4010
two-source ancestry score, averaging the two allele scores for heterozygotes.
For each distance bin it pools only within-individual, within-chromosome SNP
pairs. With score values `u` and `v` at the pair endpoints, the reported
covariance is `mean(u*v) - mean((u+v)/2)**2`, where the second mean is over
all pair endpoints. Chromosome contributions are retained internally for
jackknife reconstruction. Equal weighting of distance bins is used to fit:

```text
C(d) = amplitude * exp(-decay_rate * d) + offset
```

Amplitude is constrained to be nonnegative; the offset is unrestricted.
`fit_intercept=False` fixes the offset at zero. Only bins whose left edge is
at least `min_distance` are fitted, so the first included bin can start above
the requested cutoff. Empty bins are omitted. The fitting distance for each
bin is its mean pair distance. `max_distance` is exclusive; the final bin may
be shorter than `bin_size`. `rate_bounds` constrain the decay rate, in inverse
Morgans. Multiple search intervals are used after profiling out amplitude and
offset.

The fitted `decay_rate` is retained separately from the reported date.
We report `generations = decay_rate + 1`, following the generation-reporting
convention in the DATES v4010 executable
([`dates_expfit.c`](https://github.com/MoorjaniLab/DATES_v4010/blob/main/dates_expfit.c)).
This is a reporting convention, not an additional fitted parameter. Match the
generation-counting convention when comparing with simulations or other
programs. Dates refer to generations before the sampled individuals lived;
there is no automatic calendar-age conversion.

The default FFT calculation rounds positions to a mesh spaced at
`bin_size / mesh_subdivisions`. This approximates distances, including bin
membership near boundaries; its mean pair distances are measured on that mesh.
Distinct SNPs in one mesh cell are counted once, and self-pairs are excluded.
Use a finer mesh to check sensitivity or `method="direct"` to enumerate exact
SNP pairs on small datasets. The FFT mesh has a five-million-cell per-chromosome
limit; increase `bin_size`, reduce `mesh_subdivisions`, or use direct mode if
this is exceeded. The direct mode can be slow on dense datasets.

By default, a weighted chromosome-deletion jackknife refits the curve on the
same bin panel. Chromosome weights are their counts of observed informative
target-SNP calls, summed across individuals. Genome-wide ancestry coefficients
are held fixed across deletions. Standard errors therefore describe chromosome
variation conditional on these coefficients and the reference panel. Confidence
limits are approximate normal jackknife intervals, and are not clipped to the
parameter bounds. `jackknife=None` disables uncertainty estimation.

Inspect `fit_status` separately from `identifiability` and `status`. A converged
fit can still be unresolved. Boundary rates, ill-conditioned fits, failed
deletion fits, changes in the usable bin panel, and insufficient chromosomes
are flagged. These conditions suppress date uncertainty rather than silently
discarding replicates. Constant or increasing curves without a positive decay
return `no_signal` and an undefined date. Too few populated fitting bins return
`insufficient_bins`. Diagnostic warnings are emitted even with `verbose=False`.
Dates with a jackknife SE at least half the estimate are flagged as
`imprecise_date`; their uncertainty is retained. With `jackknife=None`, a
numerically stable fit has `identifiability="not_assessed"`. A `resolved` label
only summarizes these numerical and precision checks; it does not establish
the single-pulse demographic assumption.
NRMSD is descriptive; no universal acceptance threshold or formal p-value is
applied. Distance bins and SNP pairs are correlated, and their counts are not
independent sample sizes.

This is an independent DATES-inspired estimator. Its endpoint symmetrization,
bin-distance convention, optimization, filtering, and jackknife weighting are
explicit admixpy choices. Numerical equivalence to the upstream executable has
not been established. A single-exponential fit assumes a single admixture
pulse; population heterogeneity or samples of different ages can violate that
assumption. Inspect the curve and check sensitivity to fitting ranges and
reference populations. Two-pulse and continuous-history models are not included.


### Dating options

| Argument | Meaning |
|---|---|
| `data` | Genotype prefix, without extension. |
| `target` | One population label. |
| `sources` | Exactly two distinct reference population labels, both different from the target. |
| `target_iids` | Optional individual ID or sequence of IDs within the target population. |
| `min_distance`, `max_distance`, `bin_size` | Fitting cutoff, exclusive maximum distance, and bin width, all in Morgans. |
| `map_unit` | Input map unit, `M` or `cM`; None uses the format-specific default. |
| `fit_intercept` | Fit an unrestricted offset; False fixes it to zero. |
| `jackknife` | `chromosome` for uncertainty, or None. |
| `confidence` | Coverage used for approximate normal intervals. |
| `return_jackknife` | Retain chromosome-deletion fits in the result. |
| `method` | `fft` for mesh-based computation; `direct` for exact SNP-pair enumeration. |
| `mesh_subdivisions` | Positive integer; mesh spacing is `bin_size / mesh_subdivisions`. |
| `rate_bounds` | Two finite positive increasing bounds on decay rate, in inverse Morgans. |
| `min_snps` | Minimum observed informative SNPs per target individual; targets below this threshold are skipped with a warning. Integer >=2. |
| `autosomes_only` | Restrict to human chromosomes 1–22; False permits other labels. |
| `adjust_pseudohaploid` | Reference allele-count adjustment using the existing detection procedure. |
| `format` | Optional genotype-format override. |
| `verbose` | Progress output; diagnostic warnings are still emitted when False. |

### Dating status values

| Field or flag | Meaning |
|---|---|
| `fit_status=converged` | The bounded decay search returned a positive-amplitude fit. |
| `fit_status=no_signal` | A flat or nonpositive-decay solution gives no defined date. |
| `fit_status=insufficient_bins` | Too few usable distance bins for fitting. |
| `rate_at_boundary` | The estimated rate is near a search bound. |
| `ill_conditioned_fit` | Normalized fit-Jacobian condition exceeds 1e8. |
| `amplitude_overflow` | Extrapolating amplitude to zero distance is not finite. |
| `insufficient_jackknife_chromosomes` | Fewer than two chromosomes contribute fitting pairs. |
| `unstable_jackknife` | At least one deletion fit fails stability or common-bin checks. |
| `imprecise_date` | Date SE is at least half the date estimate; uncertainty is retained. |
| `identifiability=resolved` | No diagnostic flag was triggered with jackknife enabled. |
| `identifiability=unresolved` | One or more diagnostic flags were triggered. |
| `identifiability=not_assessed` | A numerically stable fit was obtained with jackknife disabled. |

`status` contains `ok` or semicolon-separated flags. These flags are numerical
diagnostics, not a calibrated admixture-detection test. A converged or resolved
curve does not validate the two-source demographic interpretation.

Group runs pool SNP-pair contributions, so individuals with more usable pairs
can contribute more information. They do not average individual date estimates.
Use one representation per person; suffixes such as `.AG`, `.DG`, and `.SG`
can refer to repeated datasets of the same person. The function does not infer
biological identity from these names. Mixed sampling ages can make one shared
number of generations inappropriate even when ancestry proportions are similar.

For calendar conversion, multiply generations by an explicitly chosen generation
length and add that interval to the target sampling age, propagating age and
model uncertainty separately. The reference population's age does not anchor the
date. Reference names do not establish where an admixture event occurred.

## Genotype utilities

`read_ind` returns `iid`, `sex`, and `population`. `read_snp` returns `SNP`,
`CHR`, `cm`, `POS`, `A1`, and `A2`; it preserves genetic-position units despite
the `cm` column name. `read_eigenstrat` and `read_packedancestrymap` return a
dictionary with `geno`, `ind`, and `snp`. `read_plink` returns `geno`, `fam`,
and `bim`. Genotype matrices are SNP-by-individual. `first` and `last`, where
supported, are inclusive one-based SNP bounds.

In conversion and genotype-reader helpers, `pops` selects population labels.
`inds` alone selects sample IDs and treats each selected individual as its own
output group. Supplying both `inds` and `pops` assigns one output group label
per ID; their lengths must match. This differs from dating's `target_iids`,
which restricts the existing target population without relabeling samples.

The `*_to_afs` functions return `AfData`: SNP-by-population allele frequencies,
allele-observation counts, and SNP metadata in matching row order.
`anygeno_to_afs` dispatches by detected format. `iter_geno_to_afs` yields chunks
of `AfData` for all supported formats. There is no public `read_tgeno` export;
use `tgeno_to_afs` or the automatic conversion path for allele frequencies.

`discard_from_aftable` returns a filtered `AfData` and raises if no SNPs remain.
`is_polymorphic` returns a boolean mask for rows with at least two finite,
differing frequencies. `detect_geno_format` checks dimensions against headers;
`validate=True` additionally scans all text rows and requires `nsnp`.

## Caches and blocks

`f2_from_geno` returns one `F2Blocks` object for f2, allele products (`afprod=True`),
or FST (`fst=True`). `afs_to_f2_blocks` can build several statistics from an
`AfData` and returns a dictionary keyed by `f2_blocks`, `ap_blocks`, and/or
`fst_blocks`. `stats` selects statistic names explicitly.

`write_f2` writes a block cache directory and refuses to overwrite existing
files unless `overwrite=True`. `read_f2` selects `pops`, `pops2`, and `type`
(`f2`, `ap`, or `fst`). `get_f2` dispatches between in-memory blocks, cache
objects, directories, and raw genotype prefixes. Preserve real SNP counts when
saving or constructing caches: pair-count resampling rejects missing counts.

`f4_model_cache` builds statistics for model tables. Raw genotype input produces
`F4BlockCache`; f2 input produces `F4ModelCache`. The direct cache stores
blocks/deletion statistics and defers dense covariance until a specific model
is selected. Supply the intended `allsnps` and `resampling` settings when
constructing the cache. Changing the resampling method requires rebuilding it.

`f4_stats` returns `BlockStats`; `f3_stats_from_geno` and `f4_stats_from_geno`
compute blocks directly using contrast tables with `pop1`–`pop3` or
`pop1`–`pop4` columns. `qpwave_f4stats` wraps f4 blocks as a contrast matrix.
The lower-level default for `allsnps` need not match its convenience wrapper;
use the exact signatures below and set it explicitly when reusing statistics.

For array helpers, allele-frequency/count matrices have shape
`(n_snps, n_populations)`. Pairwise block arrays have shape
`(n_populations1, n_populations2, n_blocks)`. `block_lengths` holds SNP-row
counts for contiguous blocks, not distances. `mats_to_ctarr` returns finite-pair
availability fractions; multiply by block lengths for SNP counts.
`mats_to_aparr` returns unscaled matching-allele-product block means.

`stats_to_loo` takes a statistic-by-block matrix and returns leave-one-block-out
estimates under nominal weights. `est_to_loo` handles pairwise block arrays or
`F2Blocks` and preserves the latter type. `jack_vec_stats` returns an estimate
and **variance**, not SE. `jackknife_cov` returns a covariance matrix and the
estimate vector; its `contributes` mask controls usable deletions.
`block_covariance` computes and, for a `BlockStats` input, stores covariance.
These general helpers do not automatically reproduce every estimator-specific
count-weighted or ratio jackknife path.

Low-level qpAdm helpers accept a two-dimensional f4 matrix `xmat` and the
inverse covariance `qinv` for its row-major flattened entries. Rank fits return
weights, fitted matrix, residuals and fit statistics as applicable.
`qpadm_weights` returns weights and low-rank factors `A` and `B`.
`qpadm_rankdrop` and `qpadm_popdrop` return DataFrames. `qpwave_ranktest` uses
`QpWaveStats` and additionally returns fitted/residual matrices and optimizer
convergence information.

## Result objects

Result tables are pandas DataFrames and support ordinary pandas indexing and
export. No universal `.save()` or plotting method is provided. Scalar estimates
and counts remain numeric; display formatting does not change stored values.

| Type | Main contents |
|---|---|
| `AfData` | `afs`, `counts`, `snpfile`, with matching SNP rows. |
| `F2Blocks` | `data`, `pops1`, `pops2`, `block_lengths`, `stat`; optional `snp_counts`, `fst_num`, `fst_den`. |
| `BlockStats` | Contrast `rows`, optional statistic-by-block `blocks`/`loo`, `block_lengths`, `stat`, `est`, `cov`, `variances`, `resampling`. |
| `F4ModelCache` | Pairwise `blocks`, optional `models`, `left_pops`, `right_pops`. |
| `F4BlockCache` | Direct `stats`, optional `models`, `allsnps`, `deferred_covariance`. |
| `QpWaveStats` | Underlying `f4`, original panels/bases, row/column population labels; `matrix`, `cov`, `blocks`, `loo` properties. |
| `QpAdmResult` | `target`, `left`, `right`, `weights`, `rankdrop`; optional `popdrop`, `f4`, `qpwave`, `weight_cov`. |
| `QpAdmRotationResult` | `models`, `weights`, `errors`, `settings`. |
| `F4RatioResult` | `summary`, `components`, `settings`; optional `blocks`. |
| `AdmixtureDateResult` | `summary`, `parameters`, `curve`, `diagnostics`, `settings`; optional `jackknife`. |

`F2Blocks` methods: `subset(pops=None, pops2=None)`, `pair(pop1, pop2)`,
`pair_counts(pop1, pop2)`, `pair_fst_components(pop1, pop2)`, and
`select_blocks(keep)`. Missing optional count/component information is returned
as None. `BlockStats` provides `se`, `z`, and `p` properties plus
`to_frame(round_z=None, round_p=None)`. `QpAdmResult.to_dict()` returns its
available result tables/objects; it is not a JSON serializer.

### Ancestry model tables

`QpAdmResult.weights` has `target`, `left`, `weight`, `se`, and `z`.
`rankdrop` includes rank, degrees of freedom, chi-square and p-values;
`popdrop` includes source-retention patterns, fitted weights, feasibility and
nested comparisons. Optional `f4` contains `est`, `fit`, and `diff`.
`weight_cov` is ordered like the source weights.

Rotation `models` includes stable model IDs, population panels, source count,
rank, degrees of freedom, chi-square, p-value, feasibility, and execution status.
`weights` is indexed by model/source and includes uncertainty when computed.
`errors` records model-level failures with their types and messages.
A rotation `status=ok` reports execution success, not statistical acceptance.

Ratio `summary` contains model and population identities, `est`, `se`,
`ci_low`, `ci_high`, `n`, `n_blocks`, `denominator_z`, `snp_panel`, and `status`.
`components` retains numerator/denominator contrasts with estimates and errors.
`n` is nullable when the estimator cannot identify one common SNP count.

### Admixture date tables

| Table | Fields |
|---|---|
| `summary` | `target`, `source1`, `source2`, `est`, `se`, `ci_low`, `ci_high`, `n_individuals`, `n_snps`, `n_chromosomes`, `n_bins`, `min_distance`, `max_distance`, `fit_status`, `identifiability`, `status`. |
| `parameters` | `parameter`, `est`, `se`, `ci_low`, `ci_high`, `unit`; rows for `generations`, `decay_rate`, `amplitude`, `offset`. |
| `curve` | `bin_left`, `bin_right`, `distance`, `covariance`, `fitted`, `residual`, `n_pairs`, `included`. Distances are Morgans. |
| `diagnostics` | `sse`, `nrmsd`, `condition`, `rate_at_boundary`, `n_jackknife`, `n_failed_jackknife`, `date_relative_se`, `min_ancestry_coefficient`, `max_ancestry_coefficient`. |
| `jackknife` | When requested: `chromosome`, `n_observations`, `weight`, the four parameter values, `fit_status`, `rate_at_boundary`, `same_bins`, `usable`. |
| `settings` | Version, estimator/reference identity, input format/unit, `map_origins_morgans`, distance/fit/mesh options, generation convention, jackknife scheme, nuisance ancestry coefficients, selected `target_iids`, filters, and compatibility status. |

`n_snps` is the union of observed informative SNPs across targets; it is not the
sum of individual counts. `n_pairs` sums within-individual pair counts.
`n_chromosomes` counts chromosomes contributing pairs in the fitting window.
`n_observations` counts observed informative target-SNP calls on the chromosome.
If `return_jackknife=True` but no deletions are performed, the table can be
empty. The full-data estimate is retained even when uncertainty is suppressed.

## Public API signatures

Defaults below are taken from the public Python signatures. Parameters routed
through `**kwargs` are described in the relevant method notes and lower-level
input signatures. Constructors are listed for inspection; normal workflows
obtain result objects by calling the analysis functions.

- [Statistics API](#statistics-api)
- [Model API](#model-api)
- [Dating API](#dating-api)
- [Genotype API](#genotype-api)
- [Cache API](#cache-api)
- [Numerical helper API](#numerical-helper-api)
- [Result constructors](#result-constructors)


### Statistics API

[f2](#f2) · [fst](#fst) · [f3](#f3) · [qp3pop](#qp3pop) · [f4](#f4) · [qpdstat](#qpdstat) · [f4_proximity](#f4_proximity) · [f4_ratio](#f4_ratio)

#### f2

```python
f2(data, pop1=None, pop2=None, *, unique_only=True, resampling='pairwise_counts', **kwargs)
```

Return pairwise f2 estimates and jackknife errors.

#### fst

```python
fst(
    data,
    pop1=None,
    pop2=None,
    *,
    unique_only=True,
    resampling='pairwise_counts',
    fst_aggregation='block_ratios',
    **kwargs,
)
```

Return Hudson FST; input caches must contain FST statistics.

#### f3

```python
f3(
    data,
    pop1=None,
    pop2=None,
    pop3=None,
    *,
    unique_only=True,
    resampling='pairwise_counts',
    verbose=True,
    **kwargs,
)
```

Alias of qp3pop; return f3 contrasts and uncertainty.

#### qp3pop

```python
qp3pop(
    data,
    pop1=None,
    pop2=None,
    pop3=None,
    *,
    unique_only=True,
    resampling='pairwise_counts',
    verbose=True,
    **kwargs,
)
```

Return f3 contrasts. Direct genotype input defaults to target-heterozygosity normalization; outgroupmode=True returns raw f3.

#### f4

```python
f4(
    data,
    pop1,
    pop2=None,
    pop3=None,
    pop4=None,
    *,
    comb=True,
    unique_only=True,
    afprod=False,
    verbose=True,
    **kwargs,
)
```

Return f4 contrasts; accepts a contrast table as pop1 or population arguments.

#### qpdstat

```python
qpdstat(
    data,
    pop1,
    pop2=None,
    pop3=None,
    pop4=None,
    *,
    comb=True,
    unique_only=True,
    afprod=False,
    verbose=True,
    **kwargs,
)
```

Return f4 contrasts using the same options as f4.

#### f4_proximity

```python
f4_proximity(data, target, candidates, refs, *, outgroup, allsnps=False, verbose=True, **kwargs)
```

Rank candidates by RMS contrasts across refs. outgroup is required.

#### f4_ratio

```python
f4_ratio(
    data,
    target=None,
    source1=None,
    source2=None,
    reference=None,
    outgroup=None,
    *,
    comb=True,
    models=None,
    allsnps=False,
    resampling='pairwise_counts',
    confidence=0.95,
    denominator_z_min=3.0,
    return_blocks=False,
    verbose=True,
    **kwargs,
)
```

Return F4RatioResult for explicit models or combinations of the five population arguments.

### Model API

[qpadm](#qpadm) · [qpadm_multi](#qpadm_multi) · [qpadm_rotate](#qpadm_rotate) · [qpadm_rotate_models](#qpadm_rotate_models) · [qpwave](#qpwave) · [qpwave_multi](#qpwave_multi)

#### qpadm

```python
qpadm(
    data,
    target,
    left=None,
    right=None,
    sources=None,
    fudge=0.0001,
    fudge_twice=False,
    iterations=20,
    getcov=True,
    return_f4=False,
    return_stats=False,
    return_cov=False,
    verbose=True,
    *,
    popdrop=True,
    **kwargs,
)
```

Return QpAdmResult for one target/source/reference model.

#### qpadm_multi

```python
qpadm_multi(data, models, use_cache=True, full_results=True, verbose=True, **kwargs)
```

Evaluate a model table; return full result objects or concatenated rank tests.

#### qpadm_rotate

```python
qpadm_rotate(
    data,
    target,
    leftright,
    rightfix=None,
    *,
    source_sizes=None,
    full_results=False,
    getcov=False,
    use_cache=True,
    on_error='raise',
    verbose=True,
    **kwargs,
)
```

Rotate unused candidate sources onto the reference panel and return QpAdmRotationResult.

#### qpadm_rotate_models

```python
qpadm_rotate_models(leftright, target, rightfix=None, *, source_sizes=None)
```

Return the rotation model table without reading genotype data.

#### qpwave

```python
qpwave(
    data,
    left,
    right,
    ranks=None,
    left_base=None,
    right_base=None,
    rcond=1e-10,
    diag=0.0,
    max_nfev=None,
    verbose=True,
    **kwargs,
)
```

Return rank tests for the left/right f4 contrast matrix.

#### qpwave_multi

```python
qpwave_multi(
    data,
    models,
    ranks=None,
    left_base=None,
    right_base=None,
    max_nfev=None,
    use_cache=True,
    verbose=True,
    **kwargs,
)
```

Return rank tests for multiple model rows.

### Dating API

[admixture_date](#admixture_date)

#### admixture_date

```python
admixture_date(
    data,
    target,
    sources,
    *,
    target_iids=None,
    min_distance=0.0045,
    max_distance=1.0,
    bin_size=0.001,
    map_unit=None,
    fit_intercept=True,
    jackknife='chromosome',
    confidence=0.95,
    return_jackknife=False,
    method='fft',
    mesh_subdivisions=10,
    rate_bounds=(0.001, 10000.0),
    min_snps=100,
    autosomes_only=True,
    adjust_pseudohaploid=True,
    format=None,
    verbose=True,
)
```

Return AdmixtureDateResult for a single-pulse covariance fit. Requires individual genotypes and a genetic map.

### Genotype API

[read_ind](#read_ind) · [read_snp](#read_snp) · [detect_geno_format](#detect_geno_format) · [read_eigenstrat](#read_eigenstrat) · [read_packedancestrymap](#read_packedancestrymap) · [read_plink](#read_plink) · [anygeno_to_afs](#anygeno_to_afs) · [iter_geno_to_afs](#iter_geno_to_afs) · [eigenstrat_to_afs](#eigenstrat_to_afs) · [packedancestrymap_to_afs](#packedancestrymap_to_afs) · [plink_to_afs](#plink_to_afs) · [tgeno_to_afs](#tgeno_to_afs) · [discard_from_aftable](#discard_from_aftable) · [is_polymorphic](#is_polymorphic) · [get_block_lengths](#get_block_lengths)

#### read_ind

```python
read_ind(path)
```

Read iid, sex, and population fields from an individual metadata file.

#### read_snp

```python
read_snp(path, plink=False)
```

Read SNP metadata in EIGENSTRAT or PLINK column order.

#### detect_geno_format

```python
detect_geno_format(path, nind=None, nsnp=None, validate=False)
```

Detect EIGENSTRAT text, packed AncestryMap, or TGENO from a genotype path.

#### read_eigenstrat

```python
read_eigenstrat(pref, inds=None, pops=None, first=1, last=None)
```

Read selected EIGENSTRAT genotypes and metadata into a dictionary.

#### read_packedancestrymap

```python
read_packedancestrymap(pref, inds=None, pops=None, first=1, last=None)
```

Read selected packed AncestryMap genotypes and metadata into a dictionary.

#### read_plink

```python
read_plink(pref, inds=None, pops=None)
```

Read selected SNP-major PLINK genotypes and fam/bim metadata into a dictionary.

#### anygeno_to_afs

```python
anygeno_to_afs(
    pref,
    inds=None,
    pops=None,
    format=None,
    adjust_pseudohaploid=True,
    chunk_size=10000,
    verbose=True,
    tgeno_chunked=False,
)
```

Detect genotype format and return population-level AfData.

#### iter_geno_to_afs

```python
iter_geno_to_afs(
    pref,
    inds=None,
    pops=None,
    format=None,
    adjust_pseudohaploid=True,
    chunk_size=250000,
    verbose=True,
)
```

Yield population-level AfData in bounded SNP chunks.

#### eigenstrat_to_afs

```python
eigenstrat_to_afs(
    pref,
    inds=None,
    pops=None,
    adjust_pseudohaploid=True,
    chunk_size=10000,
    verbose=True,
)
```

Convert EIGENSTRAT text genotypes to AfData.

#### packedancestrymap_to_afs

```python
packedancestrymap_to_afs(
    pref,
    inds=None,
    pops=None,
    adjust_pseudohaploid=True,
    chunk_size=10000,
    verbose=True,
)
```

Convert packed AncestryMap genotypes to AfData.

#### plink_to_afs

```python
plink_to_afs(
    pref,
    inds=None,
    pops=None,
    adjust_pseudohaploid=True,
    chunk_size=10000,
    verbose=True,
)
```

Convert SNP-major PLINK genotypes to AfData.

#### tgeno_to_afs

```python
tgeno_to_afs(
    pref,
    inds=None,
    pops=None,
    adjust_pseudohaploid=True,
    chunk_size=10000,
    verbose=True,
    chunked=False,
)
```

Convert TGENO genotypes to AfData; chunked selects the chunked conversion path.

#### discard_from_aftable

```python
discard_from_aftable(
    afdat,
    maxmiss=0,
    minmaf=0,
    maxmaf=0.5,
    minac2=False,
    outpop=None,
    auto_only=True,
    poly_only=False,
    transitions=True,
    transversions=True,
    keepsnps=None,
)
```

Filter an AfData by SNP availability, frequency, allele count, chromosome and mutation type, or a whitelist.

#### is_polymorphic

```python
is_polymorphic(afs)
```

Return a mask for SNP rows with at least two finite, differing allele frequencies.

#### get_block_lengths

```python
get_block_lengths(dat, blgsize=0.05)
```

Return contiguous SNP-row counts per block using map positions or the documented fallback.

### Cache API

[afs_to_f2_blocks](#afs_to_f2_blocks) · [f2_from_geno](#f2_from_geno) · [get_f2](#get_f2) · [read_f2](#read_f2) · [write_f2](#write_f2) · [f4_model_cache](#f4_model_cache) · [f4_stats](#f4_stats) · [f3_stats_from_geno](#f3_stats_from_geno) · [f4_stats_from_geno](#f4_stats_from_geno) · [qpwave_f4stats](#qpwave_f4stats)

#### afs_to_f2_blocks

```python
afs_to_f2_blocks(
    afdat,
    blgsize=0.05,
    pops1=None,
    pops2=None,
    outpop=None,
    afprod=False,
    fst=False,
    poly_only=('f2',),
    apply_corr=True,
    stats=None,
    verbose=True,
)
```

Construct a dictionary of requested pairwise statistic block objects from AfData.

#### f2_from_geno

```python
f2_from_geno(
    pref,
    inds=None,
    pops=None,
    blgsize=0.05,
    maxmiss=0,
    minmaf=0,
    maxmaf=0.5,
    pops2=None,
    outpop=None,
    outpop_scale=True,
    transitions=True,
    transversions=True,
    auto_only=True,
    keepsnps=None,
    afprod=False,
    fst=False,
    poly_only=('f2',),
    minac2=False,
    format=None,
    adjust_pseudohaploid=True,
    chunk_size=10000,
    tgeno_chunked=False,
    remove_na=False,
    apply_corr=True,
    verbose=True,
)
```

Construct one F2Blocks from a genotype prefix, selecting f2, allele-product, or FST mode.

#### get_f2

```python
get_f2(data, pops=None, pops2=None, **kwargs)
```

Load/select pairwise blocks from a prefix, cache directory, F2Blocks, or f4 cache.

#### read_f2

```python
read_f2(f2_dir, pops=None, pops2=None, type='f2', remove_na=False)
```

Read a block directory with optional population selection.

#### write_f2

```python
write_f2(blocks, outdir, overwrite=False)
```

Write F2Blocks and its available counts/components to a cache directory.

#### f4_model_cache

```python
f4_model_cache(data, models, resampling='pairwise_counts', verbose=True, **kwargs)
```

Build reusable direct-f4 or pairwise-f2 statistics for a model table.

#### f4_stats

```python
f4_stats(
    data,
    pop1,
    pop2=None,
    pop3=None,
    pop4=None,
    *,
    comb=True,
    unique_only=True,
    afprod=False,
    keep_blocks=True,
    keep_loo=True,
    covariance=True,
    allsnps=False,
    resampling='pairwise_counts',
    verbose=True,
    **kwargs,
)
```

Return BlockStats with optional blocks, deletion estimates, and covariance.

#### f3_stats_from_geno

```python
f3_stats_from_geno(
    pref,
    popcombs,
    *,
    blgsize=0.05,
    maxmiss=None,
    minmaf=0,
    maxmaf=0.5,
    minac2=False,
    outpop=None,
    outpop_scale=True,
    transitions=True,
    transversions=True,
    auto_only=True,
    keepsnps=None,
    allsnps=True,
    poly_only=False,
    apply_corr=True,
    outgroupmode=False,
    format=None,
    adjust_pseudohaploid=True,
    chunk_size=250000,
    tgeno_chunked=False,
    verbose=True,
    stream=False,
)
```

Compute direct-genotype f3 BlockStats from a popcombs table.

#### f4_stats_from_geno

```python
f4_stats_from_geno(
    pref,
    popcombs,
    *,
    blgsize=0.05,
    maxmiss=None,
    minmaf=0,
    maxmaf=0.5,
    minac2=False,
    outpop=None,
    outpop_scale=True,
    transitions=True,
    transversions=True,
    auto_only=True,
    keepsnps=None,
    allsnps=False,
    poly_only=False,
    apply_corr=True,
    format=None,
    adjust_pseudohaploid=True,
    chunk_size=250000,
    tgeno_chunked=False,
    verbose=True,
    stream=False,
    covariance=True,
)
```

Compute direct-genotype f4 BlockStats from a popcombs table.

#### qpwave_f4stats

```python
qpwave_f4stats(data, left, right, left_base=None, right_base=None, verbose=True, **kwargs)
```

Construct QpWaveStats and its f4 contrast matrix for the specified panels.

### Numerical helper API

[f3_from_f2](#f3_from_f2) · [f4_from_f2](#f4_from_f2) · [mats_to_f2arr](#mats_to_f2arr) · [mats_to_aparr](#mats_to_aparr) · [mats_to_ctarr](#mats_to_ctarr) · [est_to_loo](#est_to_loo) · [stats_to_loo](#stats_to_loo) · [jack_vec_stats](#jack_vec_stats) · [jackknife_cov](#jackknife_cov) · [block_covariance](#block_covariance) · [format_fstats](#format_fstats) · [qpadm_weights](#qpadm_weights) · [qpadm_fit](#qpadm_fit) · [qpadm_rankdrop](#qpadm_rankdrop) · [qpadm_popdrop](#qpadm_popdrop) · [qpwave_ranktest](#qpwave_ranktest)

#### f3_from_f2

```python
f3_from_f2(blocks, pop1, pop2, pop3)
```

Return per-block f3 values from pairwise f2 blocks, without direct-genotype target normalization.

#### f4_from_f2

```python
f4_from_f2(blocks, pop1, pop2, pop3, pop4)
```

Return per-block f4 values from pairwise f2 blocks.

#### mats_to_f2arr

```python
mats_to_f2arr(
    afmat1,
    afmat2,
    countmat1,
    countmat2,
    block_lengths,
    snpwt=None,
    apply_corr=True,
    verbose=False,
    population_labels=None,
)
```

Return a population1-by-population2-by-block f2 array from frequency/count matrices.

#### mats_to_aparr

```python
mats_to_aparr(
    afmat1,
    afmat2,
    countmat1,
    countmat2,
    block_lengths,
    snpwt=None,
    apply_corr=True,
    verbose=False,
)
```

Return matching-allele-product block means. The exposed apply_corr argument does not alter this helper's calculation.

#### mats_to_ctarr

```python
mats_to_ctarr(afmat1, afmat2, block_lengths, verbose=False)
```

Return per-block finite-pair availability fractions.

#### est_to_loo

```python
est_to_loo(blocks, block_lengths=None)
```

Construct nominal-weight deletion estimates from pairwise arrays or F2Blocks.

#### stats_to_loo

```python
stats_to_loo(blocks, block_lengths)
```

Construct nominal-weight deletion estimates from statistic-by-block arrays.

#### jack_vec_stats

```python
jack_vec_stats(loo_vec, block_lengths)
```

Return the weighted jackknife estimate and variance from a deletion vector.

#### jackknife_cov

```python
jackknife_cov(loo_mat, block_lengths, est=None, contributes=None)
```

Return covariance and an estimate vector from deletion arrays and block weights.

#### block_covariance

```python
block_covariance(stats, block_lengths=None)
```

Compute covariance from BlockStats or a statistic-by-block matrix.

#### format_fstats

```python
format_fstats(df)
```

Return a display-formatted copy of a statistics DataFrame.

#### qpadm_weights

```python
qpadm_weights(xmat, qinv, rank, fudge=0.0001, iterations=20)
```

Fit source weights and low-rank factors for an f4 matrix and inverse covariance.

#### qpadm_fit

```python
qpadm_fit(xmat, qinv, rank, fudge=0.0001, iterations=20)
```

Fit a specified matrix rank and return weights, fitted/residual matrices and fit statistics.

#### qpadm_rankdrop

```python
qpadm_rankdrop(xmat, qinv, fudge=0.0001, iterations=20)
```

Return qpAdm rank-test results for an f4 matrix and inverse covariance.

#### qpadm_popdrop

```python
qpadm_popdrop(xmat, qinv, sources, fudge=0.0001, iterations=20, *, cov=None, fudge_twice=False)
```

Fit nonempty source subsets on the existing SNP panel. Supply cov for original-covariance subset regularization.

#### qpwave_ranktest

```python
qpwave_ranktest(qpw, rank, rcond=1e-10, diag=0.0, max_nfev=None)
```

Test one rank of a QpWaveStats matrix; return fit statistics, matrices and convergence diagnostics.

### Result constructors

[AfData](#afdata) · [AdmixtureDateResult](#admixturedateresult) · [BlockStats](#blockstats) · [F2Blocks](#f2blocks) · [F4BlockCache](#f4blockcache) · [F4ModelCache](#f4modelcache) · [F4RatioResult](#f4ratioresult) · [QpAdmResult](#qpadmresult) · [QpAdmRotationResult](#qpadmrotationresult) · [QpWaveStats](#qpwavestats)

#### AfData

```python
AfData(afs, counts, snpfile)
```

Result/container constructor; fields are described under [result objects](#result-objects).

#### AdmixtureDateResult

```python
AdmixtureDateResult(summary, parameters, curve, diagnostics, settings, jackknife=None)
```

Result/container constructor; fields are described under [result objects](#result-objects).

#### BlockStats

```python
BlockStats(
    rows,
    blocks,
    block_lengths,
    stat,
    loo=None,
    est=None,
    cov=None,
    variances=None,
    resampling='pairwise_counts',
)
```

Result/container constructor; fields are described under [result objects](#result-objects).

#### F2Blocks

```python
F2Blocks(
    data,
    pops1,
    pops2,
    block_lengths,
    stat='f2',
    snp_counts=None,
    fst_num=None,
    fst_den=None,
)
```

Result/container constructor; fields are described under [result objects](#result-objects).

#### F4BlockCache

```python
F4BlockCache(stats, models=None, allsnps=False, deferred_covariance=False)
```

Result/container constructor; fields are described under [result objects](#result-objects).

#### F4ModelCache

```python
F4ModelCache(blocks, models=None, left_pops=None, right_pops=None)
```

Result/container constructor; fields are described under [result objects](#result-objects).

#### F4RatioResult

```python
F4RatioResult(summary, components, settings, blocks=None)
```

Result/container constructor; fields are described under [result objects](#result-objects).

#### QpAdmResult

```python
QpAdmResult(
    target,
    left,
    right,
    weights,
    rankdrop,
    popdrop=None,
    f4=None,
    qpwave=None,
    weight_cov=None,
)
```

Result/container constructor; fields are described under [result objects](#result-objects).

#### QpAdmRotationResult

```python
QpAdmRotationResult(models, weights, errors, settings)
```

Result/container constructor; fields are described under [result objects](#result-objects).

#### QpWaveStats

```python
QpWaveStats(f4, left, right, left_base, right_base, row_pops, col_pops)
```

Result/container constructor; fields are described under [result objects](#result-objects).

## Method references and license

The statistical methods draw on Patterson et al. (2012) and Maier et al. (2023).
The admixture dating method is inspired by
Chintalapati et al. (2022), DATES.

AdmixPy is distributed under the [MIT license](LICENSE).
