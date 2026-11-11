"""Load the common frozen prediction problem and evaluate toy reference integrals."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy.integrate
import torch

from experiment import Transformer, fisher_geometry, population_kl, tensors


def load_frozen_table(root, expected_hash=None):
    """Read exact context probabilities, prediction targets, and latent beliefs.

    A fitted checkpoint must travel with its frozen dataset. Refusing a missing
    or changed table avoids silently sampling a different statistical problem.
    """
    with np.load(Path(root)/"dataset.npz", allow_pickle=False) as saved:
        table = tuple(saved[k] for k in
                      ("contexts", "context_probabilities", "target_probabilities", "beliefs"))
    digest = hashlib.sha256()
    for array in table:
        digest.update(np.ascontiguousarray(array).tobytes())
    if expected_hash is not None and digest.hexdigest() != expected_hash:
        raise ValueError("Frozen data fingerprint does not match the fitted checkpoint.")
    return table


def toy_references(name, tau):
    """Exact localized free energy and energy, with a normalized N(0,I) prior."""
    if name == "quadratic":
        h = np.array([1., 4.])
        return float(np.log1p(tau*h).sum()/2), float((tau*h/(1+tau*h)).sum()/2)
    if name == "quartic":
        def integrand(u):
            return math.exp(-u**4-u*u/(2*math.sqrt(tau)))
        normalization = scipy.integrate.quad(integrand, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
        moment = scipy.integrate.quad(lambda u:u**4*integrand(u), -np.inf, np.inf,
                                      epsabs=1e-12, epsrel=1e-12)[0]/normalization
        return math.log(tau)/4+math.log(2*math.pi)/2-math.log(normalization), moment
    if name == "curved":
        a = tau/(1+2*tau)
        density = lambda x:math.exp(-x*x/2-a*x**4)/math.sqrt(2*math.pi)
        normalization = scipy.integrate.quad(density, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
        fourth = scipy.integrate.quad(lambda x:x**4*density(x), -np.inf, np.inf,
                                      epsabs=1e-12, epsrel=1e-12)[0]/normalization
        return math.log1p(2*tau)/2-math.log(normalization), tau/(1+2*tau)+tau*fourth/(1+2*tau)**2
    raise ValueError(name)


def checkpoint_problem(run):
    """Return fit metadata, center, full Fisher basis, and population KL callable.

    The Fisher basis preconditions proposals only. Every original parameter
    remains free, and the nonlinear prediction objective is evaluated directly.
    """
    run = Path(run)
    fit = json.loads((run/"fit.json").read_text())
    model = Transformer(fit["width"], fit["heads"], fit["length"], fit["mlp_width"])
    table = load_frozen_table(run.parent, fit["dataset_sha256"])
    data = tensors(table)
    with np.load(run/"checkpoint.npz", allow_pickle=False) as checkpoint:
        center = torch.from_numpy(checkpoint["weights"])
    if center.shape != (model.dimension,) or not torch.isfinite(center).all():
        raise ValueError("Checkpoint has invalid parameter shape or values.")
    eigenvalues, rotation = fisher_geometry(model, center, data)
    energy = lambda w:population_kl(model, w, data)
    return fit, center, eigenvalues, rotation, energy
