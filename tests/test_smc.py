"""Checks of the annealing weights, resampling, and corrected HMC kernel."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from smc import (Problem, incremental_weights, leapfrog, make_problem, next_temperature,
                 potential_gradient, rejuvenate, run_smc, systematic_resample)


class SMCChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_incremental_gaussian_partition_ratio(self):
        nodes, weights = np.polynomial.hermite.hermgauss(80)
        x = torch.tensor(nodes*math.sqrt(2))
        # Exact Gaussian integration, independent of Monte Carlo noise.
        ratio = float((torch.tensor(weights/math.sqrt(math.pi))*torch.exp(-.7*x.square()/2)).sum())
        self.assertAlmostEqual(ratio, 1/math.sqrt(1.7), places=13)
        loss = torch.tensor([0., 1., 4.], dtype=torch.float64)
        w, log_mean, ess = incremental_weights(loss, .7)
        self.assertAlmostEqual(math.exp(log_mean), float(torch.exp(-.7*loss).mean()), places=14)
        self.assertAlmostEqual(ess, 1/float(w.square().sum()), places=14)
        # A constant loss shift changes F, not weights or temperature selection.
        w2, shifted, _ = incremental_weights(loss+12, .7)
        torch.testing.assert_close(w, w2)
        self.assertAlmostEqual(shifted-log_mean, -.7*12, places=13)

    def test_temperature_and_systematic_resampling(self):
        loss = torch.linspace(0, 2, 100, dtype=torch.float64)
        tau = next_temperature(loss, 1., 100., .85)
        self.assertAlmostEqual(incremental_weights(loss, tau-1)[2], 85., places=10)
        self.assertEqual(next_temperature(torch.zeros(20), 0., 10., .85), 10.)
        weights = torch.tensor([.05, .15, .3, .5], dtype=torch.float64)
        rng = torch.Generator().manual_seed(981)
        counts = torch.zeros(4)
        for _ in range(4000):
            counts += torch.bincount(systematic_resample(weights, rng), minlength=4)
        torch.testing.assert_close(counts/16000, weights.float(), atol=.004, rtol=0)

    def test_leapfrog_reversibility_and_fisher_coordinate_gradient(self):
        p = make_problem('curved')
        scales = (p.gamma+17*p.eigenvalues).rsqrt()
        z = torch.tensor([[.2, -.5], [-.3, .8]], dtype=torch.float64)
        v = torch.tensor([[.7, -.2], [.1, -.4]], dtype=torch.float64)
        evaluate = lambda q:potential_gradient(p, q, scales, 17.)
        potential, gradient, _ = evaluate(z)
        x = z*scales
        residual = x[:, 1]-x[:, 0]**2
        analytic = torch.stack([-68*x[:, 0]*residual+x[:, 0], 34*residual+x[:, 1]], -1)*scales
        torch.testing.assert_close(gradient, analytic)
        q, momentum, u, g, _, _ = leapfrog(z, v, potential, gradient, evaluate, .02, 17)
        reverse, reversed_v, *_ = leapfrog(q, -momentum, u, g, evaluate, .02, 17)
        torch.testing.assert_close(reverse, z, atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(reversed_v, -v, atol=1e-13, rtol=1e-13)

    def test_hmc_preserves_exact_quadratic_target(self):
        p = make_problem('quadratic')
        rng = torch.Generator().manual_seed(453)
        scales = (1+100*p.eigenvalues).rsqrt()
        x = torch.randn(12000, 2, generator=rng, dtype=torch.float64)*scales
        x, loss, diagnostics = rejuvenate(p, x, 100., .3, 4, 12, rng)
        normalized = x/scales
        self.assertLess(float(normalized.mean(0).abs().max()), .03)
        self.assertLess(float((normalized.square().mean(0)-1).abs().max()), .04)
        self.assertGreater(diagnostics['acceptance'], .95)
        self.assertAlmostEqual(float(100*loss.mean()), float((100*p.eigenvalues/(1+100*p.eigenvalues)).sum()/2), delta=.03)

    def test_exact_normalization_for_constant_loss_and_resuming(self):
        p = Problem('constant', lambda w:2.+0*w[:,0], torch.zeros(2, dtype=torch.float64),
                    torch.zeros(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64),
                    7., dict(baseline=.3, toy=False))
        schedule = [dict(tau=t, step_size=.2) for t in [1., 10., 100.]]
        with tempfile.TemporaryDirectory() as directory:
            args = dict(particles=16, seed=87, endpoints=[1., 10., 100.], moves=2,
                        length=4, schedule=schedule, high_tau=10., high_moves=3, high_length=6)
            result = run_smc(p, Path(directory), **args)
            for row in result['rows']:
                self.assertAlmostEqual(row['free_energy'], 1.7*row['tau'], places=12)
                self.assertAlmostEqual(row['incremental_ess'], 16., places=12)
                self.assertEqual(row['hmc_moves'], 2 if row['tau']<10 else 3)
                self.assertEqual(row['leapfrog_steps'], 4 if row['tau']<10 else 6)
            self.assertEqual(result, run_smc(p, Path(directory), **args))

    def test_evidence_aggregation_and_correlated_slope_error(self):
        from analyze_smc import summarize, summarize_energy
        log_z = np.log([.1, .2, .4, .8])
        f, se, influence = summarize(log_z)
        self.assertAlmostEqual(f, -math.log(.375), places=13)
        self.assertAlmostEqual(se, np.std([.1,.2,.4,.8], ddof=1)/2/.375, places=13)
        later_f, later_se, later_influence = summarize(log_z-12)
        self.assertAlmostEqual(later_f-f, 12., places=13)
        self.assertAlmostEqual(later_se, se, places=13)
        # A common multiplicative error in Z cancels in the interval ratio.
        self.assertLess(float(np.std(later_influence-influence)), 1e-14)
        energy,energy_se,energy_influence=summarize_energy(log_z,[2.,3.,4.,5.])
        self.assertAlmostEqual(energy, (.1*2+.2*3+.4*4+.8*5)/1.5, places=13)
        self.assertAlmostEqual(energy_se, energy_influence.std(ddof=1)/2, places=13)


if __name__ == '__main__':
    unittest.main()
