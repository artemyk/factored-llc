# Reproducing the experiments

Run commands from the repository root. Scripts use CPU float64 and one Torch
thread per process. No external dataset, model download, API key, or training
service is needed. `runs/` is ignored by Git and is the destination for fresh
computations. The reference records are in `results_paired/` and `results_smc/`.

## Environment

The saved SMC records report Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1, and
PyTorch 2.14.1. Publication checks used Python 3.12.16 with those numerical
package versions, Matplotlib 3.11.2, and tqdm 4.70.1.

In an environment you manage, install the pinned dependencies with:

```sh
python -m pip install -r requirements.txt
```

Using an existing Conda environment is also sufficient; prefix the following
commands with `conda run -n base` when needed. Different operating systems,
BLAS implementations, or package versions can change optimization paths,
rank thresholds, and HMC decisions. Recompute numerical convergence and
representation screens when retraining. Saved checkpoints define the exact
centers of the published SMC comparison.

For a process budget of one numerical-library thread, set:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Torch compilation is optional. `--compile-loss` checks compiled values and
gradients against eager float64 before use. Eager mode computes the same target
and avoids a compiler dependency; floating-point differences can change an
individual Monte Carlo path. MPS does not support the float64 precision used here.

## Tests, saved records, and plots

```sh
python -m unittest discover -s tests -v
python verify_results.py
python analyze_paired.py
python analyze_smc.py
```

Tests cover latent-path enumeration, normalization and independence of the
source, equivalence to standard attention, vectorized gradients, stable KL,
matched auxiliary targets, prediction-only refinement, coordinate symmetries,
leapfrog reversibility, HMC invariance, SMC normalization and evidence pooling,
schedule validation, interrupted-run recovery, and the distributed results.

`verify_results.py` checks all distributed asset hashes and reproduces the
checkpoint probes and free-energy summaries from the raw records. It checks
the original GD histories; it does not execute 20,000 steps. `persistence.py`
performs that new computation.

`analyze_paired.py` writes coordinate controls and matched standalone MCMC
comparisons to `results_paired/analysis/`. `analyze_smc.py` writes curve,
slope, decomposition, and diagnostic tables and plots to `results_smc/`.
Their `--output` option selects another results directory. For SMC, that
directory must contain the raw production JSON records, not just a pilot.

The saved records include 24 transformer production populations, 72 toy
production populations, and seven pilots. Pilots are excluded from estimates.
The transformer SMC calculation uses training seed zero only; the other
training seeds support the preparation and persistence checks and earlier
standalone MCMC comparisons.

## Train new checkpoints

```sh
python paired_experiment.py --stage train --seeds 0 1 2 --output runs/paired
python analyze_paired.py --output runs/paired
```

The default training settings reproduce the preparation protocol in
[METHODS.md](./METHODS.md). Every arm saves its regularized checkpoint, final
checkpoint, objective history, representation metrics, numerical Hessian
diagnostics, and convergence flag. `pairs.json` records the operational
pair-eligibility screen. Completed fits are reused; choose a new output
directory when changing the training protocol.

The final stage uses prediction alone, including its numerical refinement.
The separate plain-GD continuation is:

```sh
python persistence.py --results runs/paired --output runs/persistence
```

By default this continues all six prepared checkpoints for 20,000 steps at
learning rate 0.2. To test the distributed checkpoints directly, omit
`--results runs/paired`. The script saves probe histories, parameter snapshots,
metadata, and one summary row per trajectory. It refuses to overwrite an
existing run.

To reproduce the original 20,000-step perturbation experiment, including the
original perturbation seed and draw ordering, use:

```sh
python persistence.py --scales 0 0.01 0.03 0.1 --output runs/perturbed-persistence
```

Seed zero uses all requested scales and four repeats for each positive scale;
seeds one and two are continued without perturbations. This is a local return
check around prepared solutions, with noise applied only at the start.

## A short transformer SMC check

This exercises the full pilot/production CLI without estimating the published
high-precision curves:

```sh
python smc.py --pilot --names factored_seed0 joint_seed0 \
  --particles 16 --seeds 20261102 --moves 2 --leapfrog-steps 4 \
  --max-tau 100 --label smoke --output runs/smc-smoke
python smc.py --names joint_seed0 --particles 16 --seeds 20261501 \
  --moves 2 --leapfrog-steps 4 \
  --schedule runs/smc-smoke/joint_seed0_smoke_n16_seed20261102.json \
  --label smoke --output runs/smc-smoke
```

For fresh training centers, pass `--results runs/paired` to **both** pilot
and production. A pilot is specific to its name, localization, checkpoint,
and frozen dataset. The production CLI checks their hashes. It also rejects
reuse of the pilot seed. A schedule fixes all reported temperatures; if
`--max-tau` is also supplied to production, it must match the schedule.

## Primary transformer free-energy curves

The distributed pilot schedules exactly specify the bridge temperatures and
HMC step sizes of the reported comparison. Rerun its eight primary populations
with these schedules and the distributed checkpoints:

```sh
for arm in factored joint; do
  python smc.py --names "${arm}_seed0" --particles 128 \
    --seeds 20261501 20261502 20261503 20261504 \
    --moves 8 --leapfrog-steps 16 \
    --schedule "results_smc/${arm}_seed0_pilot_fast_n32_seed20261102.json" \
    --label long_hmc --high-tau 1000000 --high-moves 8 --high-length 64 \
    --compile-loss --output runs/smc
 done
python analyze_smc.py --output runs/smc
```

Omit `--compile-loss` to use eager mode. SMC saves per-stage progress,
normalizing-constant increments, incremental ESS, HMC acceptance/divergences,
ancestry counts, energies, and displacements. Longer trajectories matter
particularly at the upper temperatures. In the original records, one primary
population took approximately several minutes on the recorded CPU;
runtime depends strongly on compilation and hardware. Full replication
requires all independent populations, rather than one successful run.

To regenerate the discarded fast pilots instead of using the distributed ones:

```sh
python smc.py --pilot --names factored_seed0 joint_seed0 \
  --particles 32 --seeds 20261102 --moves 6 --leapfrog-steps 16 \
  --label pilot_fast --output runs/pilots
```

A secondary pilot uses 128 particles, seed 20261101, 12 HMC moves, and
`--leapfrog-steps 24`. Both pilots are excluded from the production estimates.
If you regenerate pilots or retrain the checkpoints, point production at the
new schedules. Schedule hashes and run metadata identify the resulting protocol.

## Particle-count and HMC-effort comparisons

The shorter-trajectory comparison uses the same fast pilots and four seeds
at each of two particle counts:

```sh
for arm in factored joint; do
  for particles in 128 256; do
    python smc.py --names "${arm}_seed0" --particles "$particles" \
      --seeds 20261401 20261402 20261403 20261404 \
      --moves 8 --leapfrog-steps 16 \
      --schedule "results_smc/${arm}_seed0_pilot_fast_n32_seed20261102.json" \
      --label main --output runs/smc
  done
done
python analyze_smc.py --output runs/smc
```

This adds sixteen populations. The primary longer-HMC runs above hold the
schedule and particle count fixed while changing trajectory effort at high
temperature. Pool only complete independent populations with matched targets
and settings. The aggregator rejects mixed checkpoint hashes, dataset hashes,
HMC effort, or duplicate seeds within an estimation group.

## Exact toy calibrations

The toys use a normalized standard Gaussian prior and losses
\((x^2+4y^2)/2\), \(x^4\), and \((y-x^2)^2\). Analytic integration or
one-dimensional quadrature gives their reference free energies and endpoint
energies. Their limiting learning coefficients are 1, 1/4, and 1/2.

The saved toy pilot uses 256 particles, seed 20261101, 12 HMC moves, and
`--leapfrog-steps 24`. To regenerate it:

```sh
python smc.py --pilot --names quadratic quartic curved --particles 256 \
  --seeds 20261101 --moves 12 --leapfrog-steps 24 \
  --label pilot --output runs/toy-pilots
```

The following reruns the 72 recorded toy production populations using the
distributed schedules:

```sh
for toy in quadratic quartic curved; do
  for particles in 256 1024; do
    python smc.py --names "$toy" --particles "$particles" \
      --seeds 20261201 20261202 20261203 20261204 \
      --moves 12 --leapfrog-steps 24 \
      --schedule "results_smc/${toy}_pilot_n256_seed20261101.json" \
      --output runs/toys
  done
  python smc.py --names "$toy" --particles 4096 \
    --seeds 20261301 20261302 20261303 20261304 20261305 20261306 \
      20261307 20261308 20261309 20261310 20261311 20261312 \
      20261313 20261314 20261315 20261316 \
    --moves 12 --leapfrog-steps 24 \
    --schedule "results_smc/${toy}_pilot_n256_seed20261101.json" \
    --output runs/toys
done
python analyze_smc.py --output runs/toys
```

## Resuming and provenance

Rerun an identical SMC command to resume from its `.pt` state and random
generator state. Changes to target metadata, code fingerprints, or HMC
settings are rejected for that state file. Use a fresh output or label after
changing code or settings. A lost JSON record is reconstructed from the saved
state. State files and logs from newly executed runs are ignored by Git.

The repository includes original JSON stage records, but omits original
particle/state `.pt` files and standalone MCMC chain arrays. These are not
needed to reproduce tables or start independent integrations. The distributed
JSON records alone are not resumable particle populations; new computations
start from the prior and create their own resumable states.

[provenance/assets.json](../provenance/assets.json) records original and
published hashes. Absolute machine paths in original JSON metadata were made
relative. Numerical values, dataset fingerprints, binary checkpoints, and
original source hashes were preserved. [Extraction notes](../provenance/README.md)
describe the code cleanup and publication checks.

## Earlier standalone MCMC comparison

`paired_experiment.py --stage sample` retains the earlier MALA/HMC endpoint
energy estimator, which measures \(\tau\mathbb E[K-K(w_*)]\), rather than
SMC free energy. Its archived JSON diagnostics and probes are included for
all three training seeds. The archived analysis selects the usable chain
with highest energy ESS in each configuration. An example fresh HMC command is:

```sh
python paired_experiment.py --stage sample --output runs/paired --seeds 0 1 2 \
  --sampler hmc --burnin 1000 --draws 4000 \
  --leapfrog-steps 40 --step-size 0.03 --compile-loss
```

Numerical acceptance and mixing must be checked for each configuration.
Failed records remain in the analysis and are excluded from matched usable
comparisons. The primary README free-energy table comes from SMC, not this
endpoint-energy estimator.
