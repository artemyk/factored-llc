"""Independent scientific controls and regression checks for the public workflow."""

import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from analyze_smc import collect, summarize
from experiment import HERE, SIGNS, TOKEN_TO_STATE, Transformer, population_kl, source_table, tensors, write_json
from persistence import gradient_descent, summarize as summarize_persistence
from problems import checkpoint_problem, load_frozen_table, toy_references
from smc import make_problem, run_smc, validate_schedule
from verify_results import verify_assets, verify_checkpoints, verify_curves, verify_persistence


class ScientificControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hmm_matches_independent_latent_path_enumeration(self):
        # Sum all 16 latent paths of each one-factor HMM, independently of
        # source_table's recursive filtering implementation.
        contexts, weights, targets, beliefs = source_table('hmm', 0., 3)
        observations = ((SIGNS[TOKEN_TO_STATE][contexts]+1)/2).astype(int)
        for row in range(64):
            factor_weight, factor_prediction, factor_belief = [], [], []
            for factor,(stay,accuracy) in enumerate(((.88,.9),(.78,.85))):
                joint_next = np.zeros(2)
                posterior = np.zeros(2)
                for path in itertools.product(range(2), repeat=4):
                    probability = .5
                    for t in range(3):
                        if t:
                            probability *= stay if path[t] == path[t-1] else 1-stay
                        probability *= accuracy if path[t] == observations[row,t,factor] else 1-accuracy
                    posterior[path[2]] += probability
                    probability *= stay if path[3] == path[2] else 1-stay
                    for next_observation in range(2):
                        joint_next[next_observation] += probability*(
                            accuracy if path[3] == next_observation else 1-accuracy)
                # Posterior above sums over the two values of the future state.
                posterior /= posterior.sum()
                factor_weight.append(joint_next.sum())
                factor_prediction.append(joint_next/joint_next.sum())
                factor_belief.append(posterior)
            self.assertAlmostEqual(weights[row], np.prod(factor_weight), places=14)
            expected = np.kron(*factor_prediction)[TOKEN_TO_STATE]
            np.testing.assert_allclose(targets[row],expected,rtol=1e-13,atol=1e-14)
            np.testing.assert_allclose(beliefs[row],np.kron(*factor_belief),rtol=1e-13,atol=1e-14)

    def test_toy_reference_energy_is_free_energy_derivative(self):
        for name in ('quadratic','quartic','curved'):
            for tau in (1.,100.,1e6):
                epsilon=1e-4
                upper=toy_references(name,tau*math.exp(epsilon))[0]
                lower=toy_references(name,tau*math.exp(-epsilon))[0]
                energy=toy_references(name,tau)[1]
                self.assertAlmostEqual((upper-lower)/(2*epsilon),energy,delta=1e-8)

    def test_full_smc_normalization_against_exact_gaussian(self):
        problem=make_problem('quadratic')
        endpoints=[float(t) for t in np.geomspace(.01,100.,33)]
        schedule=[dict(tau=t,step_size=.25) for t in endpoints]
        estimates=[]
        with tempfile.TemporaryDirectory() as directory:
            for seed in (71,72):
                result=run_smc(problem,Path(directory),particles=1024,seed=seed,
                               endpoints=endpoints,schedule=schedule,moves=3,length=8)
                estimates.append(result['rows'][-1]['log_z'])
        actual=summarize(estimates)[0]
        self.assertAlmostEqual(actual,toy_references('quadratic',100.)[0],delta=.12)

    def test_persistence_updates_are_plain_population_gd(self):
        model=Transformer(width=4,heads=2,mlp_width=4)
        table=source_table('hmm',0.,3)
        initial=torch.stack([model.initialize(seed) for seed in (8,9)])
        expected=initial.clone().requires_grad_()
        for _ in range(3):
            gradient,=torch.autograd.grad(population_kl(model,expected,tensors(table)).sum(),expected)
            expected=(expected-.2*gradient).detach().requires_grad_()
        with tempfile.TemporaryDirectory() as directory:
            actual,history=gradient_descent(model,initial,table,steps=3,learning_rate=.2,output=directory)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        summary=summarize_persistence(history,[dict(arm='factored',checkpoint_seed=i) for i in (8,9)])
        self.assertEqual(len(summary),2)
        self.assertFalse(any(row['failed'] for row in summary))


class WorkflowChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_distributed_checkpoint_metrics_and_eligibility(self):
        self.assertEqual(len(verify_checkpoints()),6)

    def test_saved_free_energy_aggregation_and_persistence_records(self):
        self.assertEqual(verify_curves(),(255,96))
        self.assertEqual(verify_persistence(),6)

    def test_asset_checksums(self):
        self.assertGreater(verify_assets(),100)

    def test_missing_or_changed_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=root/'joint_seed0'
            run.mkdir()
            original=HERE/'results_paired/joint_seed0'
            (run/'fit.json').write_bytes((original/'fit.json').read_bytes())
            (run/'checkpoint.npz').write_bytes((original/'checkpoint.npz').read_bytes())
            with self.assertRaises(FileNotFoundError):
                checkpoint_problem(run)
            table=list(load_frozen_table(HERE/'results_paired'))
            table[2]=table[2].copy()
            table[2][0,0]+=1e-8
            np.savez(root/'dataset.npz',contexts=table[0],context_probabilities=table[1],
                     target_probabilities=table[2],beliefs=table[3])
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                checkpoint_problem(run)

    def test_invalid_schedules_are_rejected_before_running(self):
        for schedule in ([],[dict(tau=2.,step_size=.1)],
                         [dict(tau=1.,step_size=0.),dict(tau=2.,step_size=.1)],
                         [dict(tau=2.,step_size=.1),dict(tau=1.,step_size=.1),dict(tau=2.,step_size=.1)]):
            with self.assertRaises(ValueError):
                validate_schedule(schedule,[1.,2.])
        with self.assertRaises(ValueError):
            validate_schedule(None,[2.,1.])
        with self.assertRaises(ValueError):
            make_problem('quadratic',gamma=0.)

    def test_interrupted_smc_resumes_identical_particle_and_rng_state(self):
        problem=make_problem('quadratic')
        args=dict(particles=16,seed=184,endpoints=[1.,10.,100.],moves=2,length=6,
                  schedule=[dict(tau=t,step_size=.2) for t in (1.,10.,100.)])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            uninterrupted=run_smc(problem,root/'full',**args)
            # Simulate interruption just after committing the first state file.
            with patch('smc.write_json',side_effect=InterruptedError('simulated interruption')):
                with self.assertRaises(InterruptedError):
                    run_smc(problem,root/'resumed',**args)
            resumed=run_smc(problem,root/'resumed',**args)
            self.assertEqual(uninterrupted['rows'],resumed['rows'])
            states=[]
            for name in ('full','resumed'):
                states.append(torch.load(next((root/name).glob('*.pt')),weights_only=True))
            torch.testing.assert_close(states[0]['x'],states[1]['x'],rtol=0,atol=0)
            self.assertTrue(torch.equal(states[0]['generator'],states[1]['generator']))
            # Recover even if interruption left a complete state without JSON.
            next((root/'resumed').glob('*.json')).unlink()
            recovered=run_smc(problem,root/'resumed',**args)
            self.assertEqual(recovered['rows'],resumed['rows'])
            with self.assertRaisesRegex(ValueError,'different configuration'):
                run_smc(problem,root/'resumed',**dict(args,moves=3))

    def test_aggregation_refuses_mixed_targets_or_hmc_effort(self):
        source=HERE/'results_smc/factored_seed0_long_hmc_n128_seed20261501.json'
        first=json.loads(source.read_text())
        for field,value in (('high_length',32),('checkpoint_sha256','different-checkpoint')):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                second=json.loads(source.read_text())
                second['config']['seed']+=1
                second['config'][field]=value
                write_json(root/'a.json',first)
                write_json(root/'b.json',second)
                with self.assertRaisesRegex(ValueError,'Incompatible'):
                    collect(root)


if __name__ == '__main__':
    unittest.main()
