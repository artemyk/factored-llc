"""Checks for the generative distributions and vectorized inference/sampling."""

import unittest

import numpy as np
import torch

from experiment import SIGNS, TOKEN_TO_STATE, Transformer, hmc, mala, population_kl, population_loss, source_table, tensors


class ExperimentTests(unittest.TestCase):
    def test_sampling_observer_does_not_change_target_chain(self):
        torch.set_num_threads(1)
        center = torch.zeros(2, dtype=torch.float64)
        eigenvalues = torch.ones_like(center)
        eigenvectors = torch.eye(2, dtype=center.dtype)
        energy = lambda w: w.square().sum(-1)/2
        for sampler in (mala, hmc):
            options = dict(seed=12, burnin=20, draws=20, chains=2, step_size=0.1)
            if sampler is hmc:
                options["leapfrog_steps"] = 3
            original, _ = sampler(energy, center, eigenvalues, eigenvectors, 10, 2, **options)
            observed, result = sampler(energy, center, eigenvalues, eigenvectors, 10, 2,
                                       observer=lambda w: dict(norm=w.norm().item()), observe_every=5,
                                       **options)
            np.testing.assert_array_equal(original, observed)
            self.assertEqual(len(result["observations"]), 4)

    def test_stable_kl_matches_cross_entropy_gradient(self):
        model = Transformer(width=4, heads=2)
        data = tensors(source_table("hmm", 0.25, 3))
        weight = model.initialize(8).requires_grad_(True)
        kl = population_kl(model, weight, data).sum()
        loss = population_loss(model, weight, data).sum()
        entropy = -(data[1][:, None]*data[2]*data[2].log()).sum()
        torch.testing.assert_close(kl, loss-entropy, rtol=1e-12, atol=1e-14)
        grad_kl = torch.autograd.grad(kl, weight)[0]
        grad_ce = torch.autograd.grad(loss, weight)[0]
        torch.testing.assert_close(grad_kl, grad_ce, rtol=1e-10, atol=1e-13)

    def test_stable_kl_resolves_tiny_prediction_errors(self):
        class DirectLogits:
            def forward(self, flat, contexts):
                return flat.reshape(-1, 1, 4)
        q = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
        perturbation = 1e-9*torch.tensor([1., -2., 3., -1.], dtype=torch.float64)
        data = (torch.zeros((1, 1), dtype=torch.long), torch.ones(1, dtype=torch.float64), q)
        kl = population_kl(DirectLogits(), q.log().flatten()+perturbation, data)[0]
        centered = perturbation-(q*perturbation).sum()
        leading_term = (q*centered.square()).sum()/2
        self.assertGreater(float(kl), 0)
        self.assertAlmostEqual(float(kl/leading_term), 1, places=5)

    def test_sources_are_normalized_and_product_at_zero(self):
        token_signs = SIGNS[TOKEN_TO_STATE]
        for source in ("markov", "hmm"):
            context, weight, target, belief = source_table(source, 0, 3)
            self.assertAlmostEqual(weight.sum(), 1)
            np.testing.assert_allclose(target.sum(axis=1), 1)
            np.testing.assert_allclose(belief.sum(axis=1), 1)
            means = target @ token_signs
            product = ((1+means[:, 0, None]*token_signs[None, :, 0])
                       *(1+means[:, 1, None]*token_signs[None, :, 1]))/4
            np.testing.assert_allclose(target, product, atol=1e-14)
            coupled = source_table(source, 0.5, 3)[2]
            means = coupled @ token_signs
            covariance = coupled @ (token_signs[:, 0]*token_signs[:, 1])-means.prod(axis=1)
            self.assertGreater(np.max(np.abs(covariance)), 1e-3)

    def test_markov_prediction_uses_only_final_token(self):
        contexts, weights, targets, _ = source_table("markov", 0.25, 3)
        for last in range(4):
            rows = targets[contexts[:, -1] == last]
            np.testing.assert_allclose(rows, np.broadcast_to(rows[0], rows.shape))

    def test_vectorized_chains_match_individual_forward_and_gradient(self):
        torch.set_num_threads(1)
        model = Transformer(width=4, heads=2)
        data = tensors(source_table("hmm", 0.25, 3))
        batched = torch.stack([model.initialize(seed) for seed in (0, 1)]).requires_grad_(True)
        losses = population_loss(model, batched, data)
        gradients = torch.autograd.grad(losses.sum(), batched)[0]
        for index in range(2):
            single = batched[index].detach().requires_grad_(True)
            loss = population_loss(model, single, data).sum()
            gradient = torch.autograd.grad(loss, single)[0]
            torch.testing.assert_close(loss, losses[index])
            torch.testing.assert_close(gradient, gradients[index])

    def test_last_query_matches_standard_pytorch_attention(self):
        model = Transformer(width=4, heads=2, mlp_width=8)
        flat = model.initialize(7)
        context = tensors(source_table("hmm", 0.25, 3))[0][:5]
        pieces = [piece[0] for piece in model.unpack(flat[None])]
        embedding, position, qkv, output, readout, bias, w1, b1, w2, b2 = pieces
        attention = torch.nn.MultiheadAttention(4, 2, bias=False, batch_first=True, dtype=torch.float64)
        with torch.no_grad():
            attention.in_proj_weight.copy_(qkv.T)
            attention.out_proj.weight.copy_(output.T)
        hidden = embedding[context]+position
        mask = torch.triu(torch.full((3, 3), -torch.inf, dtype=torch.float64), diagonal=1)
        attended, _ = attention(hidden, hidden, hidden, attn_mask=mask)
        final = hidden[:, -1]+attended[:, -1]
        final = final+torch.nn.functional.gelu(final@w1+b1)@w2+b2
        torch.testing.assert_close(model.forward(flat, context)[0], final@readout+bias)


if __name__ == "__main__":
    unittest.main()
