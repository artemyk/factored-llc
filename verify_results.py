"""Verify distributed assets and recompute the central numerical results.

This performs numerical regression checks, not a mathematical certification.
It does not retrain or rerun the long SMC populations.
"""

import csv
import hashlib
import json

import numpy as np
import torch

from analyze_smc import collect
from experiment import HERE, Transformer, population_kl, representation_metrics, tensors
from paired_experiment import dataset_hash, distinct_pair
from problems import load_frozen_table


def verify_assets():
    manifest = json.loads((HERE/'provenance/assets.json').read_text())
    for item in manifest['assets']:
        path = HERE/item['path']
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item['sha256']:
            raise ValueError(f'Asset differs from manifest: {item["path"]}')
    return len(manifest['assets'])


def verify_checkpoints():
    table = load_frozen_table(HERE/'results_paired')
    model, data = Transformer(), tensors(table)
    result = {}
    for seed in range(3):
        pair = {}
        for arm in ('factored','joint'):
            run = HERE/f'results_paired/{arm}_seed{seed}'
            fit = json.loads((run/'fit.json').read_text())
            assert dataset_hash(table) == fit['dataset_sha256']
            with np.load(run/'checkpoint.npz', allow_pickle=False) as saved:
                weight = torch.from_numpy(saved['weights'])
            assert weight.shape == (628,)
            kl = float(population_kl(model, weight, data)[0])
            metrics = representation_metrics(model, weight, table)
            np.testing.assert_allclose(kl, fit['stable_kl'], rtol=1e-3, atol=1e-15)
            np.testing.assert_allclose(metrics['additive_hidden_residual_fraction'],
                                       fit['additive_hidden_residual_fraction'], rtol=1e-6, atol=1e-12)
            np.testing.assert_allclose(metrics['belief_linear_r2'], fit['belief_linear_r2'], atol=1e-6)
            pair[arm] = dict(fit, **metrics)
            result[f'{arm}_seed{seed}'] = dict(kl=kl,
                nonadditive_fraction=metrics['additive_hidden_residual_fraction'],
                product_probe_r2=metrics['belief_linear_r2'][2])
        assert distinct_pair(pair['factored'],pair['joint']), f'Pair {seed} failed its screen.'
    return result


def verify_curves():
    curves, _, diagnostics, _ = collect(HERE/'results_smc')
    def key(row):
        return row['name'], float(row['particles']), row['label'], float(row['tau'])
    recomputed = {key(row):row for row in curves}
    with (HERE/'provenance/reference_curves.csv').open() as handle:
        reference = list(csv.DictReader(handle))
    assert len(reference) == len(curves)
    for row in reference:
        actual = recomputed[key(row)]
        for field in ('free_energy','free_energy_se','energy','concentration'):
            np.testing.assert_allclose(actual[field], float(row[field]), rtol=1e-11, atol=1e-11)
    return len(curves),len(diagnostics)


def verify_persistence():
    root = HERE/'results_paired/basin_gd/lr02'
    labels = json.loads((root/'return_metadata.json').read_text())['trajectory_labels']
    history = json.loads((root/'return_history.json').read_text())
    assert history[-1]['step'] == 20000
    zero = [i for i,row in enumerate(labels) if row['coordinate_sd'] == 0]
    assert len(zero) == 6
    from persistence import classify
    for i in zero:
        for row in history:
            assert not row['failed'][i]
            assert classify(row['representation'][i]) == labels[i]['arm']
        assert history[-1]['kl'][i] < 1e-10
    return len(zero)


def main():
    torch.set_num_threads(1)
    print(f'Verified {verify_assets()} distributed asset hashes.')
    checkpoints = verify_checkpoints()
    for name,row in checkpoints.items():
        print(f'{name}: KL={row["kl"]:.6g}, nonadditive={100*row["nonadditive_fraction"]:.6g}%, '
              f'product probe R2={row["product_probe_r2"]:.6f}')
    points,populations = verify_curves()
    print(f'Recomputed {points} free-energy curve points from {populations} production populations.')
    print(f'Checked recorded 20,000-step persistence for {verify_persistence()} prepared checkpoints.')


if __name__ == '__main__':
    main()
