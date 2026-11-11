"""Check finite-time retention of prepared codes under prediction-only GD.

All original parameters follow w[t+1] = w[t] - eta grad K(w[t]). Gaussian
perturbations, when requested, are applied once before this deterministic flow.
Operational class thresholds measure feature prominence in the saved basis.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiment import HERE, Transformer, population_kl, representation_metrics, tensors, write_json
from problems import load_frozen_table


def classify(metrics):
    fraction = metrics['additive_hidden_residual_fraction']
    return 'factored' if fraction < .01 else 'joint' if fraction > .08 else 'intermediate'


def gradient_descent(model, initial, table, *, steps, learning_rate, output, checkpoints=()):
    """Run independent fixed-step trajectories and save probes at selected steps.

    Each row receives its own full population gradient, without averaging over
    trajectories. Failed trajectories are marked explicitly and then frozen.
    Their probe values are omitted if the weights themselves are nonfinite.
    """
    if steps < 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Require nonnegative steps and a finite, positive learning rate.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = tensors(table)
    weight = initial.clone().requires_grad_(True)
    failed = np.zeros(len(initial), dtype=bool)
    history, snapshots = [], {}
    started = time.monotonic()
    desired = set(checkpoints) | {0, steps}
    for step in range(steps+1):
        loss = population_kl(model, weight, data)
        gradient, = torch.autograd.grad(loss.sum(), weight)
        valid = (torch.isfinite(loss) & torch.isfinite(gradient).all(dim=1)).cpu().numpy()
        failed |= ~valid
        if step in desired:
            row = dict(step=step, wall_seconds=time.monotonic()-started,
                       kl=loss.detach().cpu().numpy().tolist(),
                       gradient_norm=gradient.detach().norm(dim=1).cpu().numpy().tolist(),
                       failed=failed.tolist(), representation=[])
            for w in weight.detach().cpu():
                row['representation'].append(representation_metrics(model, w, table)
                                             if torch.isfinite(w).all() else None)
            for key in ('kl', 'gradient_norm'):
                row[key] = [v if np.isfinite(v) else None for v in row[key]]
            history.append(row)
            snapshots[str(step)] = weight.detach().cpu().numpy().copy()
            write_json(output/'history.json', history)
            np.savez_compressed(output/'trajectories.npz', **snapshots)
            finite = loss.detach()[torch.isfinite(loss)]
            print(f'GD step={step}/{steps}, median KL='
                  f'{float(finite.median()) if len(finite) else float("nan"):.6g}, '
                  f'failed={failed.sum()}, elapsed={row["wall_seconds"]:.1f}s', flush=True)
        if step == steps:
            break
        with torch.no_grad():
            active = torch.tensor(~failed, device=weight.device)
            weight[active] -= learning_rate*gradient[active]
    return weight.detach(), history


def summarize(history, labels):
    """Report initial/final class and whether every recorded class was retained."""
    result = []
    for i, label in enumerate(labels):
        metrics = [row['representation'][i] for row in history]
        classes = [classify(m) if m is not None else 'failed' for m in metrics]
        failed = any(row['failed'][i] for row in history)
        result.append(dict(**label, steps=history[-1]['step'], failed=failed,
            initial_class=classes[0], final_class=classes[-1],
            retained_at_recorded_steps=not failed and all(c == label['arm'] for c in classes),
            initial_nonadditivity=metrics[0]['additive_hidden_residual_fraction'] if metrics[0] else None,
            final_nonadditivity=metrics[-1]['additive_hidden_residual_fraction'] if metrics[-1] else None,
            final_joint_probe_r2=metrics[-1]['belief_linear_r2'][2] if metrics[-1] else None,
            final_kl=history[-1]['kl'][i], final_gradient_norm=history[-1]['gradient_norm'][i]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE/'results_paired')
    parser.add_argument('--output', type=Path, default=HERE/'runs/persistence')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--learning-rate', type=float, default=.2)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--scales', type=float, nargs='+', default=[0.],
                        help='Coordinate SDs for seed zero; other seeds are unperturbed.')
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--noise-seed', type=int, default=202610300)
    args = parser.parse_args()
    if args.repeats < 1 or any(not np.isfinite(s) or s < 0 for s in args.scales):
        parser.error('Require positive repeats and finite, nonnegative perturbation scales.')
    if (args.output/'metadata.json').exists():
        parser.error('Output already contains a run; choose a fresh --output directory.')
    torch.set_num_threads(1)
    model = Transformer()
    table = load_frozen_table(args.results)
    generator = torch.Generator().manual_seed(args.noise_seed)
    initial, labels, hashes = [], [], {}
    for arm in ('factored', 'joint'):
        for seed in args.seeds:
            run = args.results/f'{arm}_seed{seed}'
            fit = json.loads((run/'fit.json').read_text())
            load_frozen_table(args.results, fit['dataset_sha256'])
            path = run/'checkpoint.npz'
            with np.load(path, allow_pickle=False) as checkpoint:
                center = torch.from_numpy(checkpoint['weights'])
            hashes[str(path.relative_to(args.results))] = hashlib.sha256(path.read_bytes()).hexdigest()
            for scale in (args.scales if seed == 0 else [0.]):
                for repeat in range(1 if scale == 0 else args.repeats):
                    # Consume a draw also at scale zero to reproduce original perturbation seeds.
                    initial.append(center+scale*torch.randn(center.shape, generator=generator, dtype=center.dtype))
                    labels.append(dict(arm=arm, checkpoint_seed=seed, coordinate_sd=scale, replicate=repeat))
    from paired_experiment import dataset_hash
    write_json(args.output/'metadata.json', dict(
        arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        dataset_sha256=dataset_hash(table), checkpoint_sha256=hashes,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=dict(torch=str(torch.__version__), numpy=np.__version__),
        dynamics='full_batch_euclidean_gradient_descent', dtype='float64', device='cpu',
        trajectory_labels=labels))
    checkpoints = [s for s in (0,100,500,1000,2000,5000,10000,20000,50000,100000) if s <= args.steps]
    _, history = gradient_descent(model, torch.stack(initial), table, steps=args.steps,
                                 learning_rate=args.learning_rate, output=args.output, checkpoints=checkpoints)
    summary = summarize(history, labels)
    write_json(args.output/'summary.json', summary)
    print(f'{sum(r["retained_at_recorded_steps"] for r in summary)}/{len(summary)} '
          'retain their prepared class at all recorded steps.')


if __name__ == '__main__':
    main()
