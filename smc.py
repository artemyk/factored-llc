"""Annealed SMC evidence estimates with Metropolis-corrected HMC moves.

A separate adaptive pilot selects temperatures and HMC step sizes. Production
runs freeze that schedule, start from independent exact Gaussian-prior draws,
and use unbiased systematic resampling at every stage. Their product-of-means
estimator of Z is unbiased conditional on the pilot schedule (its logarithm
is not). No variational density or quadratic loss approximation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import torch

from experiment import HERE, write_json
from problems import checkpoint_problem, toy_references

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CODE_SHA256 = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
               for name in ('smc.py', 'experiment.py', 'problems.py')}


@dataclass
class Problem:
    name: str
    energy: object
    center: torch.Tensor
    eigenvalues: torch.Tensor
    rotation: torch.Tensor
    gamma: float
    metadata: dict

    @property
    def dimension(self):
        return len(self.center)

    def loss(self, x):
        """Stable excess KL in fixed orthonormal displacement coordinates."""
        return self.energy(self.center+x@self.rotation.T)-self.metadata['baseline']


def make_problem(name, gamma=100., results=HERE/'results_paired'):
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('Localization precision must be finite and positive.')
    if name in ('quadratic', 'quartic', 'curved'):
        cases = {
            'quadratic': (2, lambda w:(w.square()*w.new_tensor([1., 4.])).sum(-1)/2, [1., 4.]),
            'quartic': (1, lambda w:w[:, 0]**4, [0.]),
            'curved': (2, lambda w:(w[:, 1]-w[:, 0]**2)**2, [0., 2.]),
        }
        d, energy, h = cases[name]
        return Problem(name, energy, torch.zeros(d, dtype=torch.float64),
                       torch.tensor(h, dtype=torch.float64), torch.eye(d, dtype=torch.float64),
                       1., dict(baseline=0., toy=True))
    run = results/name
    fit, center, h, u, energy = checkpoint_problem(run)
    if not fit.get('converged', True):
        raise ValueError('Checkpoint did not pass its training convergence check.')
    metadata = dict(baseline=float(energy(center[None])[0]), toy=False,
                    checkpoint=str(run/'checkpoint.npz'),
                    checkpoint_sha256=hashlib.sha256((run/'checkpoint.npz').read_bytes()).hexdigest(),
                    dataset_sha256=fit.get('dataset_sha256'), training_seed=fit['seed'])
    return Problem(name, energy, center, h.clamp_min(0), u, gamma, metadata)


def validate_schedule(schedule, endpoints):
    """Require an increasing, finite schedule containing every reported endpoint."""
    if not endpoints or any(not math.isfinite(t) or t <= 0 for t in endpoints):
        raise ValueError('Endpoints must be finite and positive.')
    if any(b <= a for a, b in zip(endpoints, endpoints[1:])):
        raise ValueError('Endpoints must be strictly increasing.')
    if schedule is None:
        return
    temperatures = [row['tau'] for row in schedule]
    if not temperatures or temperatures[-1] != endpoints[-1]:
        raise ValueError('Schedule must end at the final endpoint.')
    previous = 0.
    for row in schedule:
        if not math.isfinite(row['tau']) or row['tau'] <= previous:
            raise ValueError('Schedule temperatures must be finite and strictly increasing.')
        if not math.isfinite(row['step_size']) or row['step_size'] <= 0:
            raise ValueError('Schedule step sizes must be finite and positive.')
        previous = row['tau']
    if not set(endpoints).issubset(temperatures):
        raise ValueError('Schedule must contain every reported endpoint.')


def incremental_weights(loss, delta_tau):
    log_weight = -delta_tau*loss
    log_mean = torch.logsumexp(log_weight, 0)-math.log(len(loss))
    weights = torch.softmax(log_weight, 0)
    return weights, float(log_mean), float(1/weights.square().sum())


def next_temperature(loss, tau, endpoint, ess_fraction):
    """Largest step up to the endpoint with the requested incremental ESS."""
    target = ess_fraction*len(loss)
    if incremental_weights(loss, endpoint-tau)[2] >= target:
        return float(endpoint)
    lower, upper = tau, endpoint
    for _ in range(60):
        middle = (lower+upper)/2
        if incremental_weights(loss, middle-tau)[2] < target:
            upper = middle
        else:
            lower = middle
    if lower <= tau:
        raise FloatingPointError('Temperature increment underflowed.')
    return float(lower)


def systematic_resample(weights, generator):
    n = len(weights)
    positions = (torch.arange(n, dtype=weights.dtype)+
                 torch.rand((), generator=generator, dtype=weights.dtype))/n
    cumulative = weights.cumsum(0)
    cumulative[-1] = 1.
    return torch.searchsorted(cumulative, positions).clamp_max(n-1)


def potential_gradient(problem, z, scales, tau):
    z = z.detach().requires_grad_(True)
    x = z*scales
    loss = problem.loss(x)
    potential = tau*loss+problem.gamma*x.square().sum(-1)/2
    gradient = torch.autograd.grad(potential.sum(), z)[0]
    return potential.detach(), gradient.detach(), loss.detach()


def leapfrog(z, momentum, potential, gradient, evaluate, step, length):
    """Reversible volume-preserving proposal, with unflipped final momentum."""
    position = z.clone()
    velocity = momentum-step*gradient/2
    for index in range(length):
        position = position+step*velocity
        new_potential, new_gradient, loss = evaluate(position)
        velocity = velocity-step*(.5 if index == length-1 else 1.)*new_gradient
    change = new_potential+velocity.square().sum(-1)/2-potential-momentum.square().sum(-1)/2
    return position, velocity, new_potential, new_gradient, loss, change


def rejuvenate(problem, x, tau, step, moves, length, generator, *, adapt=False):
    """HMC in Fisher-scaled coordinates; isotropic prior stays in original weights.

    Step adaptation is permitted only in the discarded schedule pilot.
    Production trajectory lengths are randomly jittered independently of state.
    """
    scales = (problem.gamma+tau*problem.eigenvalues).rsqrt()
    z = x/scales
    evaluate = lambda q:potential_gradient(problem, q, scales, tau)
    potential, gradient, loss = evaluate(z)
    rates, probabilities, divergent, displacements = [], [], [], []
    for _ in range(moves):
        momentum = torch.randn(z.shape, generator=generator, dtype=z.dtype)
        count = int(torch.randint(max(2, length//2), 3*length//2+1, (), generator=generator))
        proposal, _, new_potential, new_gradient, new_loss, change = leapfrog(
            z, momentum, potential, gradient, evaluate, step, count)
        finite = torch.isfinite(change) & torch.isfinite(new_gradient).all(-1)
        probability = torch.where(finite, torch.exp((-change).clamp(max=0)), torch.zeros_like(change))
        accept = torch.rand(len(x), generator=generator, dtype=z.dtype) < probability
        displacements.append(float(((proposal-z)*scales).square().sum(-1)[accept].sum()/len(x)))
        z = torch.where(accept[:, None], proposal, z)
        potential = torch.where(accept, new_potential, potential)
        gradient = torch.where(accept[:, None], new_gradient, gradient)
        loss = torch.where(accept, new_loss, loss)
        rates.append(float(accept.double().mean()))
        probabilities.append(float(probability.mean()))
        divergent.append(float((~finite | (change.abs()>100)).double().mean()))
        if adapt:
            # Fast reduction after rejection; gentle growth after good acceptance.
            step *= math.exp(1.5*(probabilities[-1]-.85))
            step = float(np.clip(step, 1e-9, .8))
    return z*scales, loss, dict(acceptance=float(np.mean(rates)),
        acceptance_probability=float(np.mean(probabilities)), divergence_fraction=float(np.mean(divergent)),
        mean_squared_jump=float(np.mean(displacements)), next_step=step)


def run_smc(problem, output, *, particles, seed, endpoints, moves=8, length=24,
            ess_fraction=.85, schedule=None, label='main', high_tau=None,
            high_moves=8, high_length=64):
    """Run or resume one pilot/production population, saving every stage.

    Production uncertainties must come from independent full populations;
    resampled particles are dependent. Terminal particle SD/sqrt(N) is not
    reported as an evidence error bar.
    """
    validate_schedule(schedule, endpoints)
    if particles < 2 or moves < 1 or length < 2 or not 0 < ess_fraction < 1:
        raise ValueError('Require particles >= 2, moves >= 1, length >= 2, and 0 < ESS fraction < 1.')
    if high_tau is not None and (not math.isfinite(high_tau) or high_tau <= 0
                                 or high_moves < 1 or high_length < 2):
        raise ValueError('Invalid high-temperature HMC settings.')
    output.mkdir(parents=True, exist_ok=True)
    pilot = schedule is None
    stem = f'{problem.name}_{label}_n{particles}_seed{seed}'
    record_path, state_path = output/f'{stem}.json', output/f'{stem}.pt'
    schedule_hash = None if pilot else hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()
    config = dict(name=problem.name, pilot=pilot, particles=particles, seed=seed,
                  endpoints=list(endpoints), moves=moves, leapfrog_steps=length, gamma=problem.gamma,
                  ess_fraction=ess_fraction, dimension=problem.dimension, label=label,
                  schedule_sha256=schedule_hash, code_sha256=CODE_SHA256, **problem.metadata)
    if high_tau is not None:
        config.update(high_tau=high_tau, high_moves=high_moves, high_length=high_length)
    config_key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    generator = torch.Generator().manual_seed(seed)
    started = time.monotonic()
    if state_path.exists():
        saved = torch.load(state_path, weights_only=True)
        if saved['config_key'] != config_key:
            raise ValueError(f'Refusing to overwrite a different configuration: {state_path}')
        x, ancestry, interval_ancestry = saved['x'], saved['ancestry'], saved['interval_ancestry']
        rows, tau, log_z, step = saved['rows'], saved['tau'], saved['log_z'], saved['step']
        generator.set_state(saved['generator'])
        elapsed = saved['elapsed']
        # The state is committed first. Recover a missing/stale JSON record
        # after interruption between the two writes, including at completion.
        if rows:
            write_json(record_path, dict(config=config, rows=rows,
                complete=tau >= endpoints[-1], seconds=elapsed,
                versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                              numpy=np.__version__, scipy=scipy.__version__),
                script_sha256=SOURCE_SHA256, source_fingerprinted_at_import=True))
    else:
        x = torch.randn(particles, problem.dimension, generator=generator, dtype=torch.float64)/math.sqrt(problem.gamma)
        ancestry = torch.arange(particles)
        interval_ancestry = ancestry.clone()
        rows, tau, log_z, step, elapsed = [], 0., 0., .25, 0.
    with torch.no_grad():
        loss = problem.loss(x)
    if not torch.isfinite(loss).all():
        raise FloatingPointError('Nonfinite starting loss.')
    while tau < endpoints[-1]:
        endpoint = next(t for t in endpoints if t > tau)
        if pilot:
            new_tau = next_temperature(loss, tau, endpoint, ess_fraction)
            used_step = step
        else:
            stage = schedule[len(rows)]
            new_tau, used_step = stage['tau'], stage['step_size']
        weights, increment, ess = incremental_weights(loss, new_tau-tau)
        weighted_energy = float(new_tau*(weights*loss).sum())
        indices = systematic_resample(weights, generator)
        x, ancestry, interval_ancestry = x[indices], ancestry[indices], interval_ancestry[indices]
        # Pilot adaptation is excluded from evidence inference. A frozen smaller
        # step is saved for independent production runs at each temperature.
        stage_moves = high_moves if high_tau is not None and new_tau>=high_tau else moves
        stage_length = high_length if high_tau is not None and new_tau>=high_tau else length
        x, loss, diagnostic = rejuvenate(problem, x, new_tau, used_step, stage_moves, stage_length,
                                         generator, adapt=pilot)
        step = diagnostic.pop('next_step')
        log_z += increment
        row = dict(stage=len(rows)+1, tau=new_tau, previous_tau=tau, step_size=step*.9 if pilot else used_step,
            log_z=log_z, free_energy=-log_z, log_z_increment=increment, incremental_ess=ess,
            weighted_energy_before_moves=weighted_energy, energy=float(new_tau*loss.mean()),
            energy_sd=float(new_tau*loss.std()), original_weight_rms_displacement=float(x.square().mean().sqrt()),
            initial_ancestors=int(ancestry.unique().numel()), interval_ancestors=int(interval_ancestry.unique().numel()),
            endpoint=new_tau in endpoints, hmc_moves=stage_moves, leapfrog_steps=stage_length, **diagnostic)
        if problem.metadata['toy']:
            row['exact_free_energy'], row['exact_energy'] = toy_references(problem.name, new_tau)
        rows.append(row)
        tau = new_tau
        if row['endpoint']:
            interval_ancestry = torch.arange(particles)
        total_seconds = elapsed+time.monotonic()-started
        record = dict(config=config, rows=rows, complete=tau >= endpoints[-1], seconds=total_seconds,
                      versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                                    numpy=np.__version__, scipy=scipy.__version__),
                      script_sha256=SOURCE_SHA256, source_fingerprinted_at_import=True)
        state = dict(config_key=config_key, x=x, ancestry=ancestry, interval_ancestry=interval_ancestry,
                     rows=rows, tau=tau, log_z=log_z, step=step, generator=generator.get_state(), elapsed=total_seconds)
        temporary = state_path.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(state_path)
        write_json(record_path, record)
        if row['endpoint'] or len(rows) % 5 == 0:
            print(f'{stem} stage={len(rows)} tau={tau:.5g} F={-log_z:.4f} '
                  f'energy={row["energy"]:.3f} ESS={ess:.1f} acc={row["acceptance"]:.3f} '
                  f'eps={row["step_size"]:.4g} elapsed={total_seconds:.0f}s', flush=True)
        if len(rows)>3000:
            raise RuntimeError('Too many bridging temperatures; inspect degeneracy.')
    return json.loads(record_path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--names', nargs='+', default=['quadratic', 'quartic', 'curved'])
    parser.add_argument('--results', type=Path, default=HERE/'results_paired')
    parser.add_argument('--output', type=Path, default=HERE/'runs/smc')
    parser.add_argument('--gamma', type=float, default=100.)
    parser.add_argument('--particles', type=int, default=128)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261101])
    parser.add_argument('--moves', type=int, default=8)
    parser.add_argument('--leapfrog-steps', type=int, default=24)
    parser.add_argument('--ess-fraction', type=float, default=.85)
    parser.add_argument('--max-tau', type=float,
                        help='Pilot maximum precision (default 1e8); a production schedule fixes its own maximum.')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--schedule', type=Path)
    parser.add_argument('--label', default='main')
    parser.add_argument('--compile-loss', action='store_true', help='Compile CPU loss after checking float64 values and gradients.')
    parser.add_argument('--high-tau', type=float, help='Fixed precision threshold for a stronger HMC effort check.')
    parser.add_argument('--high-moves', type=int, default=8)
    parser.add_argument('--high-length', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.pilot == bool(args.schedule):
        parser.error('Specify exactly one of --pilot and --schedule.')
    if args.schedule and len(args.names) != 1:
        parser.error('A saved schedule belongs to one problem; pass exactly one --names value.')
    if args.max_tau is not None and (not math.isfinite(args.max_tau) or args.max_tau < 1):
        parser.error('--max-tau must be finite and at least 1.')
    for name in args.names:
        problem = make_problem(name, args.gamma, args.results)
        if args.compile_loss:
            raw = problem.energy
            compiled = torch.compile(raw, fullgraph=True)
            probe = (problem.center+.01*torch.randn(args.particles,problem.dimension,
                     generator=torch.Generator().manual_seed(579),dtype=torch.float64)).requires_grad_()
            reference, actual = raw(probe), compiled(probe)
            reference_gradient = torch.autograd.grad(reference.sum(),probe)[0]
            actual_gradient = torch.autograd.grad(actual.sum(),probe)[0]
            torch.testing.assert_close(reference,actual,rtol=1e-10,atol=1e-13)
            torch.testing.assert_close(reference_gradient,actual_gradient,rtol=1e-9,atol=1e-12)
            problem.energy=compiled
            problem.metadata.update(compiled_loss=True,
                compilation_value_max_error=float((reference-actual).abs().max().detach()),
                compilation_gradient_max_error=float((reference_gradient-actual_gradient).abs().max()))
        maximum = args.max_tau if args.max_tau is not None else 1e8
        endpoints = [float(x) for x in np.logspace(0, math.log10(maximum),
                                                  int(2*math.log10(maximum))+1)]
        schedule = None
        if args.schedule:
            pilot = json.loads(args.schedule.read_text())
            if not pilot['complete'] or not pilot['config']['pilot']:
                raise ValueError('Schedule must come from a completed pilot.')
            for key, value in [('name', name), ('gamma', problem.gamma),
                               ('checkpoint_sha256', problem.metadata.get('checkpoint_sha256')),
                               ('dataset_sha256', problem.metadata.get('dataset_sha256'))]:
                if pilot['config'].get(key) != value:
                    raise ValueError(f'Pilot does not match target: {key}')
            schedule = [dict(tau=r['tau'], step_size=r['step_size']) for r in pilot['rows']]
            endpoints = pilot['config']['endpoints']
            if args.max_tau is not None and args.max_tau != endpoints[-1]:
                raise ValueError('--max-tau differs from the saved pilot schedule.')
            if any(seed == pilot['config']['seed'] for seed in args.seeds):
                raise ValueError('Production seeds must differ from the discarded pilot seed.')
        for seed in args.seeds:
            run_smc(problem, args.output, particles=args.particles, seed=seed, endpoints=endpoints,
                    moves=args.moves, length=args.leapfrog_steps, ess_fraction=args.ess_fraction,
                    schedule=schedule, label=args.label, high_tau=args.high_tau,
                    high_moves=args.high_moves, high_length=args.high_length)


if __name__ == '__main__':
    main()
