"""Same-data transformer comparison with temporary representation supervision.

Both arms see one frozen, exactly weighted independent-HMM dataset. Auxiliary
hidden targets differ only during preparation. Subsequent optimization and
LLC sampling use the common unregularized prediction loss and all 628 weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.optimize
import torch
from tqdm import tqdm

from experiment import (HERE, SIGNS, TOKEN_TO_STATE, Transformer, fisher_geometry,
                        hmc, mala, population_kl, representation_metrics, source_table,
                        tensors, write_json)
from geometry import inspect


STATUS_FILENAME = "status.json"


def dataset_hash(table):
    digest = hashlib.sha256()
    for array in table:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def hidden_targets(table, width, seed):
    """Matched-scale additive and joint codes, in the same rotated coordinates.

    The shared four features are the two predictive logit contrasts and two
    posterior marginal moments. The extra joint feature is the posterior
    product moment. Thus the joint code linearly represents all four latent
    state probabilities, while both codes permit a linear optimal readout.
    Each target has total weighted centered variance one. These are training
    targets, never constraints on the final architecture or sampler.
    """
    _, probability, target, belief = table
    signs = SIGNS[TOKEN_TO_STATE]
    features = np.column_stack([np.log(target) @ signs / 2,
                                belief @ SIGNS,
                                belief @ SIGNS.prod(axis=1)])
    features -= probability @ features
    features /= np.sqrt(probability @ (features**2))
    rotation, _ = np.linalg.qr(np.random.default_rng(1000+seed).normal(size=(width, width)))
    result = {}
    for arm, count in (("factored", 4), ("joint", 5)):
        code = np.zeros((len(probability), width))
        code[:, :count] = features[:, :count] / np.sqrt(count)
        result[arm] = torch.tensor(code @ rotation, dtype=torch.float64)
    return result


def auxiliary_loss(model, weight, data, target):
    hidden = model.forward(weight, data[0], return_hidden=True)[1][0]
    return (data[1][:, None] * (hidden-target).square()).sum()


def refine_prediction(model, original, data, iterations=100, method="newton"):
    """Damped Gauss-Newton solves the common teacher-logit equations.

    All three output contrasts and all weights vary. An initial least-squares
    readout solve exploits the network's linear output layer, without choosing
    factor or joint activation targets. The equations have exactly the same
    zero set as prediction KL. Final optimization uses prediction KL itself.
    """
    basis = torch.tensor(np.column_stack([SIGNS[TOKEN_TO_STATE],
                         SIGNS[TOKEN_TO_STATE].prod(axis=1)])/2, dtype=original.dtype)
    target = data[2].log()@basis

    def residual(weight):
        return ((model.forward(weight, data[0])[0]@basis-target)*data[1].sqrt()[:, None]).flatten()

    if method == "trf":
        cache, history = {}, []

        def evaluate(value):
            if "weight" not in cache or not np.array_equal(value, cache["weight"]):
                weight = torch.tensor(value, dtype=original.dtype, requires_grad=True)
                cache.update(weight=value.copy(), residual=residual(weight).detach().numpy(),
                             jacobian=torch.autograd.functional.jacobian(residual, weight, vectorize=True).numpy())
                history.append(float(np.linalg.norm(cache["residual"])))
                if len(history) % 50 == 0:
                    print(f"  prediction-only TRF {len(history)}: logit residual={history[-1]:.3g}", flush=True)
            return cache["residual"], cache["jacobian"]

        optimum = scipy.optimize.least_squares(
            lambda x:evaluate(x)[0], original.numpy(), jac=lambda x:evaluate(x)[1],
            method="trf", tr_solver="exact", max_nfev=iterations,
            ftol=1e-14, xtol=1e-14, gtol=1e-14)
        weight = torch.from_numpy(optimum.x)
        return weight, dict(method=method, history=history, movement=float((weight-original).norm()),
                            final_kl=float(population_kl(model, weight, data)[0]),
                            optimality=float(optimum.optimality), evaluations=optimum.nfev,
                            termination=str(optimum.message))
    if method not in ("newton", "root"):
        raise ValueError(method)

    weight, history = original.clone(), []
    # Variable projection: solve for every affine readout contrast with the
    # hidden network held fixed. Inactive teacher contrasts emerge as zero
    # from this ordinary unconstrained least-squares solve.
    with torch.no_grad():
        hidden = model.forward(weight, data[0], True)[1][0]
        design = torch.cat([hidden, torch.ones(len(hidden), 1, dtype=weight.dtype)], dim=1)
        coefficients = torch.linalg.lstsq(design*data[1].sqrt()[:, None],
                                          target*data[1].sqrt()[:, None],
                                          driver="gelsd", rcond=1e-14).solution
        pieces = model.unpack(weight[None])
        pieces[4][0] += (coefficients[:-1]-pieces[4][0]@basis)@basis.T
        pieces[5][0] += (coefficients[-1]-pieces[5][0]@basis)@basis.T
    for iteration in range(iterations):
        error = residual(weight)
        kl = float(population_kl(model, weight, data)[0])
        history.append(dict(iteration=iteration, kl=kl, residual=float(error.norm())))
        if kl < 1e-20:
            break
        jacobian = torch.autograd.functional.jacobian(residual, weight, vectorize=True)
        left, singular, right = torch.linalg.svd(jacobian, full_matrices=False)
        # This cutoff stabilizes the optimization solve only. No mode is
        # removed from the architecture, loss, or subsequent sampler.
        singular = torch.where(singular > singular[0]*1e-12, singular, 0.0)
        projected = left.T@error
        # Solve the linearized least-squares trust-region problem. Simply
        # clipping an undamped Newton step lets extremely weak directions
        # consume the entire step while barely reducing prediction error.
        def coordinates(damping):
            denominator = (singular.square()+damping).clamp_min(torch.finfo(weight.dtype).tiny)
            return -singular*projected/denominator

        damping = 0.0
        if float(coordinates(damping).norm()) > 0.2:
            low, high = 0.0, 1.0
            for _ in range(100):
                middle = (low+high)/2
                if float(coordinates(middle).norm()) > 0.2:
                    low = middle
                else:
                    high = middle
            damping = high
        step = right.T@coordinates(damping)
        if method == "root":
            # A bounded Newton root step may temporarily increase loss while
            # following the curved solution set. This optional refinement is
            # kept separate from the monotone training procedure.
            weight = weight+step
            if iteration % 10 == 0:
                print(f"  prediction-only root {iteration}: KL={kl:.3g}", flush=True)
            continue
        for attempt in range(20):
            candidate = weight + 0.5**attempt*step
            if float(population_kl(model, candidate, data)[0]) < kl:
                weight = candidate
                break
        else:
            break
        if iteration % 10 == 0:
            print(f"  prediction-only Newton {iteration}: KL={kl:.3g}", flush=True)
    return weight, dict(history=history, movement=float((weight-original).norm()),
                        final_kl=float(population_kl(model, weight, data)[0]))


def converged(fit):
    return (fit["stable_kl"] <= 1e-10 and fit["gradient_norm"] <= 1e-7
            and fit["hessian_min"] >= -1e-7)


def distinct_pair(factored, joint):
    """Predeclared operational screen, independent of the LLC estimates."""
    return (converged(factored) and converged(joint)
            and factored["additive_hidden_residual_fraction"] < 0.01
            and joint["additive_hidden_residual_fraction"] > 0.08
            and joint["belief_linear_r2"][2] > 0.99)


def status(output, stage, **details):
    write_json(output/STATUS_FILENAME, dict(stage=stage,
               updated_utc=datetime.now(timezone.utc).isoformat(), process_id=os.getpid(), **details))


def train_arm(model, table, seed, arm, run, output, args):
    data = tensors(table)
    code = hidden_targets(table, model.width, seed)[arm]
    weight = torch.nn.Parameter(model.initialize(seed))
    history, snapshots = [], {}
    started = time.monotonic()

    def objective(coefficient):
        loss = population_kl(model, weight, data).sum()
        if coefficient:
            loss = loss + coefficient*auxiliary_loss(model, weight, data, code)
        return loss

    def adam(phase, steps, lr, coefficient):
        status(output, phase, seed=seed, arm=arm, running=True)
        optimizer = torch.optim.Adam([weight], lr=lr)
        for step in tqdm(range(steps), desc=f"{arm} seed={seed} {phase}", mininterval=10):
            regularization = coefficient(step, steps)
            optimizer.zero_grad()
            loss = objective(regularization)
            loss.backward()
            optimizer.step()
            if step % 250 == 0 or step == steps-1:
                row = dict(phase=phase, step=step, coefficient=regularization,
                           objective=float(loss.detach()),
                           kl=float(population_kl(model, weight, data)[0].detach()))
                history.append(row)
                status(output, phase, seed=seed, arm=arm, running=True, progress=row)

    def lbfgs(phase, coefficient, steps):
        status(output, phase, seed=seed, arm=arm, running=True)
        optimizer = torch.optim.LBFGS([weight], max_iter=steps, history_size=80,
                                     tolerance_grad=1e-8, tolerance_change=1e-15,
                                     line_search_fn="strong_wolfe")
        evaluations = 0

        def closure():
            nonlocal evaluations
            optimizer.zero_grad()
            loss = objective(coefficient)
            (1e4*loss).backward()
            evaluations += 1
            if evaluations % 200 == 0:
                print(f"{arm} seed={seed} {phase}: eval={evaluations} objective={float(loss.detach()):.4g}", flush=True)
                status(output, phase, seed=seed, arm=arm, running=True,
                       evaluations=evaluations, objective=float(loss.detach()))
            return 1e4*loss

        optimizer.step(closure)
        history.append(dict(phase=phase, coefficient=coefficient, evaluations=evaluations,
                            objective=float(objective(coefficient).detach())))

    adam("auxiliary_adam", args.aux_steps, 0.01, lambda step, count: args.regularization)
    lbfgs("auxiliary_lbfgs", args.regularization, args.aux_lbfgs_steps)
    snapshots["regularized"] = representation_metrics(model, weight.detach(), table)
    np.savez_compressed(run/"regularized_checkpoint.npz", weights=weight.detach().numpy())
    # Anneal to zero, then use a fresh optimizer with exactly zero auxiliary loss.
    adam("annealing", args.anneal_steps, 0.001,
         lambda step, count: args.regularization*(1-(step+1)/count)**2)
    snapshots["removed"] = representation_metrics(model, weight.detach(), table)
    adam("prediction_only_adam", args.off_steps, 0.001, lambda step, count: 0.0)
    lbfgs("prediction_only_lbfgs", 0.0, args.off_lbfgs_steps)
    status(output, "prediction_only_newton", seed=seed, arm=arm, running=True)
    refined, refinement = refine_prediction(model, weight.detach(), data)
    with torch.no_grad():
        weight.copy_(refined)
    lbfgs("prediction_only_final", 0.0, 300)
    center = weight.detach().clone()
    energy = population_kl(model, weight, data).sum()
    gradient = torch.autograd.grad(energy, weight)[0]
    hessian = torch.autograd.functional.hessian(
        lambda w: population_kl(model, w, data).sum(), center, vectorize=True)
    eigenvalues, eigenvectors = torch.linalg.eigh((hessian+hessian.T)/2)
    result = dict(arm=arm, seed=seed, source="hmm", coupling=0,
                  dataset_sha256=dataset_hash(table), width=model.width, heads=model.heads,
                  length=model.length, mlp_width=model.mlp_width, parameters=model.dimension,
                  stable_kl=float(energy.detach()), gradient_norm=float(gradient.norm()),
                  hessian_min=float(eigenvalues.min()), parameter_norm=float(center.norm()),
                  final_auxiliary_coefficient=0.0, training_seconds=time.monotonic()-started,
                  snapshots=snapshots, history=history, prediction_refinement=refinement,
                  **representation_metrics(model, center, table))
    result["converged"] = converged(result)
    result["local_geometry"] = inspect(model, center, data, product=True)
    np.savez_compressed(run/"checkpoint.npz", weights=center.numpy(),
                        eigenvalues=eigenvalues.numpy(), eigenvectors=eigenvectors.numpy(),
                        auxiliary_target=code.numpy())
    write_json(run/"fit.json", result)
    print(f"FIT {arm} seed={seed}: KL={result['stable_kl']:.4g}, grad={result['gradient_norm']:.3g}, "
          f"nonadditive={result['additive_hidden_residual_fraction']:.3g}, "
          f"joint probe R2={result['belief_linear_r2'][2]:.5f}", flush=True)
    return result


def report(output):
    fits = [json.loads(p.read_text()) for p in sorted(output.glob("*_seed*/fit.json"))]
    pairs = []
    for seed in sorted({fit["seed"] for fit in fits}):
        arms = {fit["arm"]: fit for fit in fits if fit["seed"] == seed}
        if len(arms) == 2:
            pairs.append(dict(seed=seed, eligible=distinct_pair(arms["factored"], arms["joint"])))
    lines = ["# Same-data paired experiment", "",
             "Both arms use the identical weighted independent-HMM table and unregularized prediction loss.",
             "Auxiliary supervision is absent during final optimization and sampling.", "",
             "| Seed | Arm | KL | Gradient norm | Nonadditive variance (%) | Joint belief probe R² | Converged |",
             "|---|---|---:|---:|---:|---:|---|"]
    for fit in fits:
        lines.append(f"| {fit['seed']} | {fit['arm']} | {fit['stable_kl']:.3g} | {fit['gradient_norm']:.3g} | "
                     f"{100*fit['additive_hidden_residual_fraction']:.5g} | {fit['belief_linear_r2'][2]:.4f} | {fit['converged']} |")
    lines += ["", "Pair eligibility requires convergence, less than 1% nonadditive activation variance in the factored arm,",
              "more than 8% in the joint arm, and joint-moment linear probe R² above 0.99 in the joint arm.", ""]
    for pair in pairs:
        lines.append(f"- Seed {pair['seed']}: eligible for paired LLC comparison = {pair['eligible']}.")
    lines += ["", "## Sampling", "",
              "Estimates below are finite-precision localized estimates. Failed mixing checks are retained but excluded from conclusions.", "",
              "| Seed | Arm | Sampler | Precision | Localization | Estimate | MCSE | R-hat | ESS | Usable |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for path in sorted(output.glob("*_seed*/*_tau*.json")):
        row = json.loads(path.read_text())
        lines.append(f"| {row['seed']} | {row['arm']} | {row.get('sampler', 'mala')} | {row['tau']:.0e} | {row['gamma']:g} | "
                     f"{row['estimate']:.3f} | {row['mcse']:.3f} | {row['rhat']:.3f} | {row['ess']:.0f} | {row['usable']} |")
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n")
    write_json(output/"pairs.json", pairs)
    return pairs


def sample_arm(model, table, run, output, args):
    fit = json.loads((run/"fit.json").read_text())
    if not fit["converged"]:
        raise ValueError(f"Unconverged fit: {run}")
    checkpoint = np.load(run/"checkpoint.npz")
    center = torch.from_numpy(checkpoint["weights"])
    data = tensors(table)
    eigenvalues, eigenvectors = fisher_geometry(model, center, data)
    bits = (SIGNS[TOKEN_TO_STATE][table[0]] > 0).astype(int)
    codes = np.einsum("nlf,l->nf", bits, 2**np.arange(model.length))
    design = np.column_stack([np.ones(len(table[0])), np.eye(8)[codes[:, 0]], np.eye(8)[codes[:, 1]]])
    left, singular, _ = np.linalg.svd(design*np.sqrt(table[1])[:, None], full_matrices=False)
    additive = torch.tensor(left[:, singular > singular[0]*1e-12], dtype=center.dtype)
    raw_energy = lambda w: population_kl(model, w, data)
    if args.compile_loss:
        compiled_energy = torch.compile(raw_energy, fullgraph=True)

        def energy(w):
            return raw_energy(w) if w.shape[0] == 1 else compiled_energy(w)

        probe = center[None].repeat(4, 1).clone()
        probe += 0.001*torch.sin(torch.arange(probe.numel(), dtype=probe.dtype)).reshape(probe.shape)
        probe.requires_grad_(True)
        fast, reference = energy(probe), raw_energy(probe)
        torch.testing.assert_close(fast, reference, rtol=1e-8, atol=1e-20)
        torch.testing.assert_close(torch.autograd.grad(fast.sum(), probe)[0],
                                   torch.autograd.grad(reference.sum(), probe)[0], rtol=1e-8, atol=1e-12)
    else:
        energy = raw_energy
    for tau in args.taus:
        if tau*fit["stable_kl"] > 0.001:
            raise ValueError("Center accuracy insufficient at requested precision")
        for gamma in args.gammas:
            path = run/f"{args.sampler}_tau{tau:g}_gamma{gamma:g}.json"
            if path.exists():
                if not args.resample_failed or json.loads(path.read_text())["usable"]:
                    continue
                backup = run/f"sampler_pilot_{time.time_ns()}"
                backup.mkdir()
                path.rename(backup/path.name)
                if path.with_suffix(".npz").exists():
                    path.with_suffix(".npz").rename(backup/path.with_suffix(".npz").name)
            previous = run/f"mala_tau{tau:g}_gamma{gamma:g}.json"
            if args.only_failed and previous.exists() and json.loads(previous.read_text())["usable"]:
                continue
            status(output, "sampling", arm=fit["arm"], seed=fit["seed"], tau=tau, gamma=gamma, running=True)
            observed = 0

            def observer(weights):
                nonlocal observed
                hidden = model.forward(weights, data[0], True)[1]
                centered = hidden-(hidden*data[1][None, :, None]).sum(1, keepdim=True)
                weighted = centered*data[1].sqrt()[None, :, None]
                projection = torch.einsum("nr,cnd->crd", additive, weighted)
                variance = weighted.square().sum((1, 2))
                fractions = (1-projection.square().sum((1, 2))/variance).clamp_min(0)
                observed += 100
                row = dict(draw=observed, nonadditive_fraction=fractions.tolist(),
                           activation_variance=variance.tolist())
                status(output, "sampling", arm=fit["arm"], seed=fit["seed"], tau=tau,
                       gamma=gamma, running=True, last_observation=row)
                return row

            sampler = hmc if args.sampler == "hmc" else mala
            extra = dict(leapfrog_steps=args.leapfrog_steps,
                         target_acceptance=args.target_acceptance) if args.sampler == "hmc" else {}
            samples, result = sampler(energy, center, eigenvalues, eigenvectors, tau, gamma,
                                   seed=20000+fit["seed"], chains=4, burnin=args.burnin,
                                   draws=args.draws, step_size=args.step_size,
                                   description=f"paired {args.sampler} {fit['arm']} seed={fit['seed']} tau={tau:g} gamma={gamma:g}",
                                   observer=observer, **extra)
            result.update(arm=fit["arm"], seed=fit["seed"], dataset_sha256=dataset_hash(table),
                          sampler=args.sampler, compiled_loss=args.compile_loss,
                          final_auxiliary_coefficient=0.0,
                          usable=bool(result["rhat"] <= 1.03 and result["ess"] >= 200
                                      and 0.5 <= result["acceptance"] <= 0.95
                                      and result.get("divergence_fraction", 0) <= 0.01))
            np.savez_compressed(path.with_suffix(".npz"), energy=samples)
            write_json(path, result)
            print(f"LLC {fit['arm']} seed={fit['seed']}: tau={tau:g} gamma={gamma:g} "
                  f"{result['estimate']:.3f} +/- {result['mcse']:.3f}; "
                  f"Rhat={result['rhat']:.3f} ESS={result['ess']:.0f}", flush=True)
            report(output)


def main():
    global STATUS_FILENAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"runs/paired")
    parser.add_argument("--stage", choices=["train", "sample", "all", "report"], default="train")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", choices=["factored", "joint"], default=["factored", "joint"],
                        help="Arms to sample; training still prepares and checks the pair")
    parser.add_argument("--aux-steps", type=int, default=3000)
    parser.add_argument("--aux-lbfgs-steps", type=int, default=1000)
    parser.add_argument("--anneal-steps", type=int, default=1000)
    parser.add_argument("--off-steps", type=int, default=1000)
    parser.add_argument("--off-lbfgs-steps", type=int, default=2000)
    parser.add_argument("--regularization", type=float, default=0.3)
    parser.add_argument("--taus", type=float, nargs="+", default=[1e4, 1e5, 1e6])
    parser.add_argument("--gammas", type=float, nargs="+", default=[100, 400])
    parser.add_argument("--burnin", type=int, default=2500)
    parser.add_argument("--draws", type=int, default=6000)
    parser.add_argument("--sampler", choices=["mala", "hmc"], default="mala")
    parser.add_argument("--leapfrog-steps", type=int, default=40)
    parser.add_argument("--step-size", type=float, default=0.15)
    parser.add_argument("--only-failed", action="store_true", help="Skip configurations with usable MALA estimates")
    parser.add_argument("--compile-loss", action="store_true")
    parser.add_argument("--resample-failed", action="store_true")
    parser.add_argument("--target-acceptance", type=float, default=0.8)
    parser.add_argument("--status-filename", default="status.json")
    args = parser.parse_args()
    if any(n < 0 for n in (args.aux_steps, args.aux_lbfgs_steps, args.anneal_steps,
                           args.off_steps, args.off_lbfgs_steps)):
        parser.error('Training iteration counts must be nonnegative.')
    STATUS_FILENAME = args.status_filename
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    model = Transformer()
    table = source_table("hmm", 0, model.length)
    table_path = args.output/"dataset.npz"
    if table_path.exists():
        saved = np.load(table_path)
        restored = tuple(saved[k] for k in ("contexts", "context_probabilities", "target_probabilities", "beliefs"))
        if dataset_hash(restored) != dataset_hash(table):
            raise ValueError("The frozen dataset differs from the source definition")
        table = restored
    else:
        np.savez_compressed(table_path, contexts=table[0], context_probabilities=table[1],
                            target_probabilities=table[2], beliefs=table[3])
    write_json(args.output/f"invocation_{time.time_ns()}.json",
               dict(arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    dataset_sha256=dataset_hash(table), python=__import__("sys").executable,
                    torch=torch.__version__, numpy=np.__version__, scipy=__import__("scipy").__version__,
                    dtype="float64", device="cpu", full_batch=True))
    if args.stage == "report":
        report(args.output)
        return
    try:
        for seed in args.seeds:
            for arm in ("factored", "joint"):
                run = args.output/f"{arm}_seed{seed}"
                run.mkdir(exist_ok=True)
                if args.stage in ("train", "all") and not (run/"fit.json").exists():
                    train_arm(model, table, seed, arm, run, args.output, args)
                    report(args.output)
                elif (run/"fit.json").exists():
                    saved_fit = json.loads((run/"fit.json").read_text())
                    if saved_fit['dataset_sha256'] != dataset_hash(table):
                        raise ValueError(f'Fit uses a different frozen dataset: {run}')
                    if not (run/"checkpoint.npz").exists():
                        raise FileNotFoundError(f'Fit is missing its checkpoint: {run}')
                else:
                    raise FileNotFoundError(f'No fitted checkpoint at {run}; run --stage train first.')
            pairs = report(args.output)
            eligible = next(p["eligible"] for p in pairs if p["seed"] == seed)
            if args.stage in ("sample", "all") and eligible:
                for arm in args.arms:
                    sample_arm(model, table, args.output/f"{arm}_seed{seed}", args.output, args)
            elif args.stage in ("sample", "all"):
                print(f"Seed {seed}: pair not eligible; skipping LLC comparison", flush=True)
        status(args.output, "complete", running=False, pairs=report(args.output))
    except BaseException as error:
        status(args.output, "failed", running=False, error=repr(error))
        raise


if __name__ == "__main__":
    main()
