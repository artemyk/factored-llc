# Data provenance and publication checks

This repository was extracted on 11 November 2026 from the existing
`experiments/factored_transformer_llc` research directory. It includes the
same-data paired preparation, representation analysis, localized SMC, and
local prediction-only GD continuation experiments.

## Included reference assets

- The frozen independent-HMM context table and six final checkpoints
  (`factored_seed0` through `joint_seed2`). Binary files are byte-for-byte
  copies. Each fit includes its regularized precursor, preparation history,
  refinement diagnostics, and representation probes.
- All 54 standalone MALA/HMC JSON summaries from the six fits, including
  unusable runs. These preserve the 18 matched comparisons and sampled-state
  representation diagnostics. Raw MCMC chain arrays are omitted.
- Seven SMC pilots and all 96 completed production stage records: 24
  transformer populations and 72 toy populations. Original particle and RNG
  state files are omitted. New runs regenerate these states from their seeds.
- Original GD continuation histories and metadata, including the wider
  perturbation follow-ups used by the archived local-return summary.
- Original aggregated free-energy values in `reference_curves.csv`, used as
  an independent regression reference for the public aggregation code.

`assets.json` gives SHA-256 hashes before and after extraction and the
published byte count for each reference asset. JSON changes only replace
absolute machine paths with repository-relative locations. The reference CSV
uses LF line endings. Numerical values,
checkpoint/data hashes, source hashes, seeds, settings, diagnostics, and
reported versions are preserved. The local-return metadata's original
`source_sha256` identifies its original GD driver, rather than the newly
extracted `persistence.py`.

Generated plots and analysis tables are rebuilt by the public analysis
scripts. Their values are checked against the original records; these derived
files are not immutable assets in the manifest. Original `.pt` state files,
logs, unrelated experimental outputs, private workspace notes, and credentials
are not distributed.

## Code extraction

- `experiment.py` retains the exact source generation, transformer, losses,
  probes, Fisher computation, and MALA/HMC kernels. The earlier, different-data
  fitting campaign and its driver were removed.
- `paired_experiment.py` retains the paired training and prediction-only
  refinement algorithm. Added guards reject missing or mismatched fits and
  handle archived summaries without chain arrays when resampling failures.
  Its default stage is training; standalone MCMC is requested explicitly.
- `geometry.py` retains the numerical checkpoint-rank diagnostic used by the
  paired training report. Its conditional coefficient requires exact-fit and
  rank assumptions; it is not asserted as the LLC of these checkpoints.
- `problems.py` contains the original checkpoint-loading and toy-integral
  routines extracted from the variational driver. Checkpoint loading now
  requires and validates the frozen dataset, rather than regenerating it
  silently. Unrelated variational optimization code is excluded.
- `smc.py` retains the annealing, weighting, resampling, and corrected HMC
  algorithms. Added validation rejects malformed schedules and pilot/target
  mismatches; production cannot reuse the pilot seed. Resume identities now
  include relevant source fingerprints. A missing or stale JSON record can
  be recovered from a committed particle state.
- Fresh training and SMC output defaults use `runs/`, separate from the
  distributed reference assets. Analysis defaults inspect the reference data.
- `analyze_smc.py` retains evidence pooling, covariance-aware errors,
  decomposition, and plots. Added guards reject accidental pooling of
  incompatible targets, HMC settings, or duplicate population seeds.
- `persistence.py` extracts the plain population-GD continuation and records
  all parameters and probes. The exploratory random-initialization basin
  classifier is excluded.

The manifest also records hashes of the original source files. The original
long-HMC source snapshot differs from the final pre-extraction SMC source only
in when its source-file hash is captured; its integration algorithm is the same.
Changes made for this repository are covered by additional workflow and
scientific-control tests.

## Validation before publication

The publication checks include:

1. The complete 31-test suite, including eleven added scientific/workflow tests.
2. Recalculation of all six checkpoint losses, nonadditivity scores, product
   probes, and pair-eligibility screens from the distributed weights.
3. Regeneration of 255 free-energy curve points and the figures from all
   96 raw production populations, matching the original numerical summaries.
4. Fresh full preparation and prediction-only refinement for all three paired
   seeds. All six final weight vectors match the original weights **bitwise**
   in the checked environment.
5. A fresh 20,000-step plain-GD continuation of all six final checkpoints,
   retaining all six prepared classes and reproducing the saved unperturbed
   endpoint diagnostics.
6. Fresh transformer pilot and production SMC CLI runs through precision 100.
7. Tests of the full SMC estimator against exact Gaussian normalization,
   leapfrog reversibility, and identical particle/RNG recovery after interruption.

The long high-precision production integrations were not rerun as part of
publication. Their saved stage records were reaggregated and checked; the
upper-end sampling qualification remains necessary. The repository's CI runs
the tests and reference-data verification on each push.
