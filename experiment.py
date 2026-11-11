"""Exact two-factor sources, vectorized transformer, representation probes, and MCMC.

All model calculations use population targets; the default transformer has
628 trainable parameters. Stable KL evaluation retains tiny prediction errors.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.special
import torch
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
SIGNS = np.array(list(itertools.product([-1.0, 1.0], repeat=2)))
# Token labels deliberately hide the Cartesian-product order from the model.
TOKEN_TO_STATE = np.array([2, 0, 3, 1])
STATE_TO_TOKEN = np.argsort(TOKEN_TO_STATE)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_table(source: str, coupling: float, length: int):
    """Return every context, its probability, next-token law, and latent belief.

    Markov: correlated spin increments, with J=(0.8,0.55).
    HMM: independent latent chains with stay probabilities (0.88,0.78),
    local emission accuracies (0.9,0.85), and a shared uniform-replacement
    observation channel of strength coupling. Initial states are stationary.
    """
    contexts = np.array(list(itertools.product(range(4), repeat=length)))
    if source == "markov":
        s, t = SIGNS.T
        log_transition = (
            0.8 * np.outer(s, s)
            + 0.55 * np.outer(t, t)
            + coupling * np.outer(s * t, s * t)
        )
        transition = scipy.special.softmax(log_transition, axis=1)
        emission = np.eye(4)
    elif source == "hmm":
        transitions = [np.array([[p, 1-p], [1-p, p]]) for p in (0.88, 0.78)]
        emissions = [np.array([[p, 1-p], [1-p, p]]) for p in (0.9, 0.85)]
        transition = np.kron(*transitions)
        emission = (1-coupling) * np.kron(*emissions) + coupling / 4
    else:
        raise ValueError(source)
    weights, targets, beliefs = [], [], []
    for context in contexts:
        belief = np.ones(4) / 4
        weight = 1.0
        for position, token in enumerate(context):
            if position:
                belief = belief @ transition
            belief = belief * emission[:, TOKEN_TO_STATE[token]]
            probability = belief.sum()
            weight *= probability
            belief /= probability
        weights.append(weight)
        targets.append((belief @ transition @ emission)[TOKEN_TO_STATE])
        beliefs.append(belief)
    weights = np.array(weights)
    targets = np.array(targets)
    beliefs = np.array(beliefs)
    assert np.allclose(weights.sum(), 1)
    assert np.allclose(targets.sum(axis=1), 1)
    return contexts, weights, targets, beliefs


class Transformer:
    """One causal attention layer, learned embeddings/positions, MLP, readout.

    Only the final query is evaluated: other queries cannot affect its output
    in a single layer. All attention, embedding, position, and readout weights
    are trained. There are no factor labels, tied weights, LayerNorm,
    dropout, weight decay, or auxiliary losses. The leading weight axis permits
    vectorized independent MCMC chains.
    """

    def __init__(self, width=8, heads=2, length=3, mlp_width=16):
        assert width % heads == 0
        self.width, self.heads, self.length = width, heads, length
        self.mlp_width = mlp_width
        self.shapes = [(4, width), (length, width), (width, 3*width),
                       (width, width), (width, 4), (4,)]
        if mlp_width:
            self.shapes += [(width, mlp_width), (mlp_width,), (mlp_width, width), (width,)]
        self.sizes = [math.prod(shape) for shape in self.shapes]
        self.dimension = sum(self.sizes)

    def unpack(self, flat):
        return [x.reshape(flat.shape[0], *shape)
                for x, shape in zip(flat.split(self.sizes, dim=-1), self.shapes)]

    def initialize(self, seed, device="cpu", dtype=torch.float64, random_biases=False):
        """Gaussian initialization; optionally randomize all 28 bias coordinates.

        Random biases use the same 0.2/sqrt(width) scale as dense weights.
        Non-bias draws are identical between protocols for a fixed seed.
        """
        generator = torch.Generator(device=device).manual_seed(seed)
        pieces = []
        for index, shape in enumerate(self.shapes):
            scale = 0.15 if index in (0, 1) else 0.2 / math.sqrt(self.width)
            value = torch.randn(shape, generator=generator, device=device, dtype=dtype) * scale
            if index in (5, 7, 9) and not random_biases:
                value.zero_()
            pieces.append(value.flatten())
        return torch.cat(pieces)

    def forward(self, flat, contexts, return_hidden=False):
        if flat.ndim == 1:
            flat = flat[None]
        pieces = self.unpack(flat)
        embedding, position, qkv_weight, output, readout, bias = pieces[:6]
        hidden = embedding[:, contexts] + position[:, None]
        count, examples, length, _ = hidden.shape
        qkv = torch.einsum("cnld,cdf->cnlf", hidden, qkv_weight)
        qkv = qkv.reshape(count, examples, length, 3, self.heads, self.width//self.heads)
        query, key, value = qkv[:, :, -1, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
        score = torch.einsum("cnhi,cnlhi->cnhl", query, key) / math.sqrt(self.width//self.heads)
        attention = score.softmax(dim=-1)
        attended = torch.einsum("cnhl,cnlhi->cnhi", attention, value)
        attended = attended.reshape(count, examples, self.width)
        hidden = hidden[:, :, -1] + torch.einsum("cnd,cdf->cnf", attended, output)
        if self.mlp_width:
            w1, b1, w2, b2 = pieces[6:]
            intermediate = torch.nn.functional.gelu(torch.einsum("cnd,cdf->cnf", hidden, w1)+b1[:, None])
            hidden = hidden+torch.einsum("cnf,cfd->cnd", intermediate, w2)+b2[:, None]
        logits = torch.einsum("cnd,cdv->cnv", hidden, readout) + bias[:, None]
        return (logits, hidden, attention) if return_hidden else logits


def tensors(table, device="cpu", dtype=torch.float64):
    contexts, weights, targets, _ = table
    return (torch.tensor(contexts, dtype=torch.long, device=device),
            torch.tensor(weights, dtype=dtype, device=device),
            torch.tensor(targets, dtype=dtype, device=device))


def population_loss(model, flat, data):
    contexts, weights, targets = data
    log_probability = model.forward(flat, contexts).log_softmax(dim=-1)
    return -(log_probability * targets[None] * weights[None, :, None]).sum(dim=(-1, -2))


def population_kl(model, flat, data):
    """Population KL without subtracting two order-one cross entropies.

    If d is the teacher-centered logit error, KL=log E_q exp(d). Subtracting
    E_q d first leaves a quantity of second order in the prediction error.
    The exp remainder is evaluated by a Taylor polynomial near zero so that
    precisions well above 1e6 do not amplify cross-entropy cancellation.
    """
    contexts, weights, targets = data
    difference = model.forward(flat, contexts)-targets.log()[None]
    d = difference-(difference*targets[None]).sum(dim=-1, keepdim=True)
    polynomial = d.square()*(0.5+d*(1/6+d*(1/24+d*(1/120+d*(1/720+d*(1/5040+d/40320))))))
    remainder = torch.where(d.abs() < 0.01, polynomial, torch.expm1(d)-d)
    return (torch.log1p((targets[None]*remainder).sum(dim=-1))*weights[None]).sum(dim=-1)


def fisher_geometry(model, center, data):
    """SVD-based Fisher preconditioner, retaining weak directions accurately.

    This changes only MCMC proposals. Squaring singular values of the score
    Jacobian avoids losing weak curvature when forming a dense Gram matrix.
    No numerical rank cutoff is used to define the sampling target or proposal.
    """
    contexts, weights, _ = data
    with torch.no_grad():
        probability = model.forward(center, contexts)[0].softmax(dim=-1)
    jacobian = torch.autograd.functional.jacobian(
        lambda w: model.forward(w, contexts)[0], center, vectorize=True)
    score = jacobian-(probability[:, :, None]*jacobian).sum(dim=1, keepdim=True)
    score = score*torch.sqrt(weights[:, None, None]*probability[:, :, None])
    _, singular, right = torch.linalg.svd(score.flatten(0, 1), full_matrices=True)
    eigenvalues = torch.zeros_like(center)
    eigenvalues[:singular.numel()] = singular.square()
    return eigenvalues, right.T


def representation_metrics(model, flat, table):
    contexts, weights, targets, beliefs = table
    with torch.no_grad():
        logits, hidden, attention = model.forward(flat, tensors(table)[0], True)
    hidden = hidden[0].numpy()
    logits = logits[0].numpy()
    probability = scipy.special.softmax(logits, axis=-1)
    centered = hidden - weights @ hidden
    _, singular_values, components = scipy.linalg.svd(centered * np.sqrt(weights[:, None]), full_matrices=False)
    variance = singular_values**2
    # Fit sums of arbitrary functions of each observed factor's history.
    bits = (SIGNS[TOKEN_TO_STATE][contexts] > 0).astype(int)
    codes = np.einsum("nlf,l->nf", bits, 2**np.arange(model.length))
    design = np.concatenate([np.ones((len(contexts), 1)),
                             np.eye(2**model.length)[codes[:, 0]],
                             np.eye(2**model.length)[codes[:, 1]]], axis=1)
    weighted_design = design * np.sqrt(weights[:, None])
    coefficients = scipy.linalg.lstsq(weighted_design, hidden * np.sqrt(weights[:, None]))[0]
    additive_error = np.sum(weights[:, None] * (hidden-design@coefficients)**2)
    variance_total = np.sum(weights[:, None] * centered**2)
    sign_tokens = SIGNS[TOKEN_TO_STATE]
    interaction = sign_tokens[:, 0] * sign_tokens[:, 1] / 4
    teacher_logit_interaction = np.log(targets) @ interaction
    student_logit_interaction = logits @ interaction
    marginal = np.stack([targets @ sign_tokens[:, i] for i in range(2)], axis=-1)
    product = ((1+marginal[:, 0, None]*sign_tokens[None, :, 0])
               *(1+marginal[:, 1, None]*sign_tokens[None, :, 1]))/4
    teacher_mi = np.sum(weights[:, None] * targets * np.log(targets/product))
    hidden_design = np.column_stack([np.ones(len(hidden)), hidden])
    belief_targets = np.column_stack([beliefs@SIGNS[:, 0], beliefs@SIGNS[:, 1],
                                     beliefs@(SIGNS[:, 0]*SIGNS[:, 1])])
    beta = scipy.linalg.lstsq(hidden_design*np.sqrt(weights[:, None]),
                             belief_targets*np.sqrt(weights[:, None]))[0]
    residual = np.sum(weights[:, None] * (hidden_design@beta-belief_targets)**2, axis=0)
    total = np.sum(weights[:, None] * (belief_targets-weights@belief_targets)**2, axis=0)
    top2_design = np.column_stack([np.ones(len(hidden)), centered@components[:2].T])
    beta2 = scipy.linalg.lstsq(top2_design*np.sqrt(weights[:, None]),
                              belief_targets*np.sqrt(weights[:, None]))[0]
    residual2 = np.sum(weights[:, None]*(top2_design@beta2-belief_targets)**2, axis=0)
    return dict(
        hidden_singular_values=singular_values.tolist(),
        hidden_dimension_95=int(np.searchsorted(np.cumsum(variance)/variance.sum(), 0.95)+1),
        additive_hidden_residual_fraction=float(additive_error/variance_total),
        teacher_logit_interaction_rms=float(np.sqrt(weights@teacher_logit_interaction**2)),
        student_logit_interaction_rms=float(np.sqrt(weights@student_logit_interaction**2)),
        teacher_conditional_mutual_information=float(teacher_mi),
        belief_linear_r2=(1-residual/np.maximum(total, 1e-30)).tolist(),
        belief_linear_r2_top2=(1-residual2/np.maximum(total, 1e-30)).tolist(),
        joint_belief_probe_weight_norm=float(np.linalg.norm(beta[1:, -1])),
        mean_attention=attention[0].numpy().mean(axis=0).tolist(),
        maximum_probability_error=float(np.max(np.abs(probability-targets))),
    )



def split_rhat_ess(samples):
    """Split R-hat and initial-positive-sequence ESS for scalar energy traces."""
    chains, count = samples.shape
    half = count//2
    x = np.concatenate([samples[:, :half], samples[:, -half:]], axis=0)
    within = np.mean(np.var(x, axis=1, ddof=1))
    between = half * np.var(x.mean(axis=1), ddof=1)
    var_hat = (half-1)/half * within + between/half
    rhat = math.sqrt(var_hat/within) if within else 1.0
    centered = x-x.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(centered, n=2*half, axis=1)
    covariance = np.fft.irfft(fft*np.conj(fft), axis=1)[:, :half]/half
    rho = 1 - (within-covariance.mean(axis=0))/max(var_hat, 1e-30)
    pair_sums = []
    for k in range(1, half-1, 2):
        pair = float(rho[k]+rho[k+1])
        if pair < 0:
            break
        pair_sums.append(min(pair, pair_sums[-1]) if pair_sums else pair)
    autocorrelation_time = max(1, 1+2*sum(pair_sums))
    ess = min(x.size, x.size/autocorrelation_time)
    mcse = math.sqrt(var_hat/ess)
    return dict(rhat=rhat, ess=ess, mcse=mcse)


def mala(energy_loss, center, eigenvalues, eigenvectors, tau, gamma, *, seed,
         chains=4, burnin=1500, draws=3000, thin=1, step_size=0.3, description="MALA",
         observer=None, observe_every=100):
    """Exact-target MALA in fixed Hessian-preconditioned coordinates.

    The Gaussian localization is gamma/2 times the Euclidean squared distance
    in ORIGINAL weight coordinates. The preconditioner changes proposal
    efficiency only; the Metropolis correction preserves the specified target.
    Adaptation occurs during burn-in only. The baseline is always the fitted
    center's loss, never a chain-dependent empirical minimum.
    An optional observer records diagnostics of retained states without
    changing proposals or the target distribution.
    """
    generator = torch.Generator().manual_seed(seed)
    root = eigenvectors * (tau*eigenvalues.clamp_min(0)+gamma).rsqrt()[None]
    baseline = float(energy_loss(center[None]).detach()[0])
    z = 0.1*torch.randn(chains, center.numel(), generator=generator, dtype=center.dtype)

    def value_gradient(position):
        position = position.detach().requires_grad_(True)
        delta = position @ root.T
        loss = energy_loss(center[None]+delta)
        potential = tau*(loss-baseline)+gamma/2*(delta**2).sum(dim=-1)
        gradient = torch.autograd.grad(potential.sum(), position)[0]
        return potential.detach(), gradient.detach(), loss.detach()

    potential, gradient, loss = value_gradient(z)
    values, displacements, accepted, raw_log_accept, observations = [], [], [], [], []
    step = step_size
    recent_accept = []
    total_steps = burnin + draws*thin
    for iteration in tqdm(range(total_steps), desc=description, mininterval=10):
        noise = torch.randn(z.shape, generator=generator, dtype=z.dtype)
        proposal = z-step/2*gradient+math.sqrt(step)*noise
        new_potential, new_gradient, new_loss = value_gradient(proposal)
        log_accept = (-new_potential+potential
                      -(z-proposal+step/2*new_gradient).square().sum(dim=-1)/(2*step)
                      +(proposal-z+step/2*gradient).square().sum(dim=-1)/(2*step))
        log_accept = torch.nan_to_num(log_accept, nan=-torch.inf, posinf=-torch.inf)
        probability = log_accept.clamp(max=0).exp()
        accept = torch.log(torch.rand(chains, generator=generator, dtype=z.dtype)) < log_accept
        z = torch.where(accept[:, None], proposal, z)
        potential = torch.where(accept, new_potential, potential)
        gradient = torch.where(accept[:, None], new_gradient, gradient)
        loss = torch.where(accept, new_loss, loss)
        if iteration < burnin:
            recent_accept.append(float(probability.mean()))
            if (iteration+1) % 50 == 0:
                step *= math.exp(0.5*(np.mean(recent_accept)-0.75))
                step = float(np.clip(step, 1e-12, 3))
                recent_accept = []
        else:
            accepted.append(accept.numpy())
            raw_log_accept.append(log_accept.clamp(max=0).numpy())
            if (iteration-burnin) % thin == 0:
                values.append((tau*(loss-baseline)).numpy())
                delta = z@root.T
                displacements.append(torch.sqrt(delta.square().mean(dim=-1)).numpy())
                if observer is not None and len(values) % observe_every == 0:
                    with torch.no_grad():
                        observations.append(observer(center[None]+delta))
    samples = np.array(values).T
    diagnostic = split_rhat_ess(samples)
    return samples, dict(
        estimate=float(samples.mean()), chain_estimates=samples.mean(axis=1).tolist(),
        chain_mcse=[split_rhat_ess(x[None])["mcse"] for x in samples],
        acceptance=float(np.mean(accepted)), final_step_size=step,
        original_weight_rms_displacement=float(np.mean(displacements)),
        mean_log_accept=float(np.mean(raw_log_accept)),
        baseline_loss=baseline, tau=tau, gamma=gamma,
        quadratic_reference=float(0.5*torch.sum(tau*eigenvalues.clamp_min(0)/(tau*eigenvalues.clamp_min(0)+gamma))),
        burnin=burnin, draws=draws, thin=thin, chains=chains, sampler_seed=seed,
        observations=observations,
        **diagnostic,
    )


def hmc(energy_loss, center, eigenvalues, eigenvectors, tau, gamma, *, seed,
        chains=4, burnin=500, draws=2000, thin=1, step_size=0.03,
        leapfrog_steps=25, description="HMC", observer=None, observe_every=100,
        target_acceptance=0.8):
    """Fixed-mass HMC for the same localized posterior as MALA.

    Randomizing the trajectory length avoids periodic Gaussian trajectories.
    Step-size adaptation is confined to burn-in. A Metropolis test corrects
    leapfrog error. All original weight coordinates, including singular
    directions, remain in the target.
    """
    generator = torch.Generator().manual_seed(seed)
    root = eigenvectors*(tau*eigenvalues.clamp_min(0)+gamma).rsqrt()[None]
    baseline = float(energy_loss(center[None]).detach()[0])
    z = 0.1*torch.randn(chains, center.numel(), generator=generator, dtype=center.dtype)

    def evaluate(position):
        position = position.detach().requires_grad_(True)
        delta = position@root.T
        loss = energy_loss(center[None]+delta)
        potential = tau*(loss-baseline)+gamma/2*delta.square().sum(dim=-1)
        gradient = torch.autograd.grad(potential.sum(), position)[0]
        return potential.detach(), gradient.detach(), loss.detach()

    potential, gradient, loss = evaluate(z)
    energies, distances, acceptance, divergences, observations = [], [], [], [], []
    step, recent = step_size, []
    for iteration in tqdm(range(burnin+draws*thin), desc=description, mininterval=10):
        momentum = torch.randn(z.shape, generator=generator, dtype=z.dtype)
        proposal = z.clone()
        velocity = momentum-step/2*gradient
        number = int(torch.randint(max(2, leapfrog_steps//2), 3*leapfrog_steps//2+1,
                                   (1,), generator=generator))
        for leap in range(number):
            proposal = proposal+step*velocity
            new_potential, new_gradient, new_loss = evaluate(proposal)
            velocity = velocity-step*(0.5 if leap == number-1 else 1)*new_gradient
        energy_change = new_potential+velocity.square().sum(dim=-1)/2-potential-momentum.square().sum(dim=-1)/2
        log_accept = torch.nan_to_num(-energy_change, nan=-torch.inf, posinf=-torch.inf)
        probability = log_accept.clamp(max=0).exp()
        accept = torch.log(torch.rand(chains, generator=generator, dtype=z.dtype)) < log_accept
        z = torch.where(accept[:, None], proposal, z)
        potential = torch.where(accept, new_potential, potential)
        gradient = torch.where(accept[:, None], new_gradient, gradient)
        loss = torch.where(accept, new_loss, loss)
        if iteration < burnin:
            recent.append(float(probability.mean()))
            if (iteration+1) % 10 == 0:
                step *= math.exp(0.6*(np.mean(recent)-target_acceptance))
                step = float(np.clip(step, 1e-8, 1))
                recent = []
        else:
            acceptance.append(accept.numpy())
            divergences.append((~torch.isfinite(energy_change) | (energy_change.abs()>100)).numpy())
            if (iteration-burnin) % thin == 0:
                energies.append((tau*(loss-baseline)).numpy())
                distances.append((z@root.T).square().mean(dim=-1).sqrt().numpy())
                if observer is not None and len(energies) % observe_every == 0:
                    with torch.no_grad():
                        observations.append(observer(center[None]+z@root.T))
    samples = np.array(energies).T
    return samples, dict(estimate=float(samples.mean()), chain_estimates=samples.mean(axis=1).tolist(),
        chain_mcse=[split_rhat_ess(x[None])["mcse"] for x in samples],
        acceptance=float(np.mean(acceptance)), divergence_fraction=float(np.mean(divergences)),
        final_step_size=step, leapfrog_steps=leapfrog_steps,
        target_acceptance=target_acceptance,
        original_weight_rms_displacement=float(np.mean(distances)), baseline_loss=baseline,
        tau=tau, gamma=gamma, burnin=burnin, draws=draws, thin=thin, chains=chains, sampler_seed=seed,
        observations=observations,
        quadratic_reference=float(0.5*torch.sum(tau*eigenvalues.clamp_min(0)/(tau*eigenvalues.clamp_min(0)+gamma))),
        **split_rhat_ess(samples))
