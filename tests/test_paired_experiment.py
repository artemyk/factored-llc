"""Controls for the same-data representation intervention."""

import unittest

import numpy as np
import torch

from experiment import SIGNS, TOKEN_TO_STATE, Transformer, population_kl, source_table, tensors
from paired_experiment import dataset_hash, distinct_pair, hidden_targets, refine_prediction
from analyze_paired import activation_controls, residual_basis_transform


class PairedTests(unittest.TestCase):
    def test_residual_basis_change_preserves_predictions(self):
        model = Transformer()
        weight = model.initialize(6)
        context = tensors(source_table("hmm", 0, 3))[0]
        matrix = torch.diag(torch.linspace(0.4, 2, 8, dtype=torch.float64))
        matrix[0, 1] = 0.3
        transformed = residual_basis_transform(model, weight, matrix)
        before = model.forward(weight, context, True)
        after = model.forward(transformed, context, True)
        torch.testing.assert_close(before[0], after[0], rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(before[1]@matrix, after[1], rtol=1e-12, atol=1e-14)

    def test_whitened_control_is_invariant_to_affine_coordinates(self):
        table = source_table("hmm", 0, 3)
        rng = np.random.default_rng(16)
        hidden = rng.normal(size=(64, 8))
        transform = np.diag(np.linspace(0.1, 10, 8))
        changed = hidden@transform+np.arange(8)
        controls = activation_controls(table, hidden, changed)
        self.assertAlmostEqual(controls["factored"]["whitened_nonadditive_fraction"],
                               controls["joint"]["whitened_nonadditive_fraction"], places=12)
        self.assertLess(controls["factored_to_joint_affine_residual_fraction"], 1e-20)

    def test_prediction_refinement_preserves_a_nonproduct_teacher(self):
        torch.set_num_threads(1)
        model = Transformer()
        data = list(tensors(source_table("hmm", 0, 3)))
        true_weight = model.initialize(5)
        data[2] = model.forward(true_weight, data[0])[0].softmax(-1).detach()
        initial = true_weight.clone()
        # Perturb only the readout. The unconstrained readout solve must
        # recover all teacher contrasts, including its nonzero interaction.
        model.unpack(initial[None])[4][0].add_(0.1)
        fitted, _ = refine_prediction(model, initial, data, iterations=5)
        self.assertLess(float(population_kl(model, fitted, data)[0]), 1e-20)
        interaction = data[2].log()@torch.tensor(SIGNS[TOKEN_TO_STATE].prod(1)/2)
        self.assertGreater(float(interaction.abs().max()), 1e-5)

    def test_codes_have_matched_scale_and_distinct_interaction(self):
        table = source_table("hmm", 0, 3)
        targets = hidden_targets(table, 8, 0)
        bits = (SIGNS[TOKEN_TO_STATE][table[0]] > 0).astype(int)
        codes = np.einsum("nlf,l->nf", bits, 2**np.arange(3))
        design = np.column_stack([np.ones(64), np.eye(8)[codes[:, 0]], np.eye(8)[codes[:, 1]]])
        root = np.sqrt(table[1])[:, None]
        for arm, expected_fraction in (("factored", 0), ("joint", 0.2)):
            code = targets[arm].numpy()
            np.testing.assert_allclose(table[1]@code, 0, atol=1e-14)
            self.assertAlmostEqual(float(np.sum(table[1][:, None]*code**2)), 1)
            coefficient = np.linalg.lstsq(design*root, code*root, rcond=None)[0]
            error = np.sum(table[1][:, None]*(code-design@coefficient)**2)
            self.assertAlmostEqual(error, expected_fraction)

    def test_both_targets_support_exact_common_prediction(self):
        table = source_table("hmm", 0, 3)
        logits = np.log(table[2])
        logits -= logits.mean(axis=1, keepdims=True)
        for target in hidden_targets(table, 8, 1).values():
            design = np.column_stack([np.ones(64), target.numpy()])
            coefficient = np.linalg.lstsq(design, logits, rcond=None)[0]
            np.testing.assert_allclose(design@coefficient, logits, atol=1e-13)

    def test_dataset_fingerprint_detects_changed_targets(self):
        table = source_table("hmm", 0, 3)
        copy = tuple(x.copy() for x in table)
        self.assertEqual(dataset_hash(table), dataset_hash(copy))
        copy[2][0, 0] += 1e-8
        self.assertNotEqual(dataset_hash(table), dataset_hash(copy))

    def test_unconverged_or_collapsed_pair_is_not_compared(self):
        factor = dict(stable_kl=1e-14, gradient_norm=1e-9, hessian_min=-1e-10,
                      additive_hidden_residual_fraction=0.001)
        joint = dict(**factor)
        joint.update(additive_hidden_residual_fraction=0.2, belief_linear_r2=[1, 1, 0.999])
        self.assertTrue(distinct_pair(factor, joint))
        joint["additive_hidden_residual_fraction"] = 0.001
        self.assertFalse(distinct_pair(factor, joint))
        joint.update(additive_hidden_residual_fraction=0.2, gradient_norm=1e-4)
        self.assertFalse(distinct_pair(factor, joint))


if __name__ == "__main__":
    unittest.main()
