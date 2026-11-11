"""Numerical rank diagnostics at a fitted checkpoint.

The conditional coefficient assumes an exact realizing product-source minimum
and full ranks. Finite precision singular values do not prove these hypotheses.
"""

import numpy as np
import torch

from experiment import SIGNS, TOKEN_TO_STATE, population_kl

BASIS = torch.tensor(np.column_stack([SIGNS[TOKEN_TO_STATE],
                    SIGNS[TOKEN_TO_STATE].prod(axis=1)])/2, dtype=torch.float64)

def inspect(model, weight, data, product):
    with torch.no_grad():
        _, hidden, _ = model.forward(weight, data[0], True)
        augmented = torch.cat([hidden[0], torch.ones(len(data[0]), 1, dtype=weight.dtype)], dim=1)
        augmented *= data[1].sqrt()[:, None]
        pieces = model.unpack(weight[None])
        readout = torch.cat([pieces[4][0], pieces[5][0][None]], dim=0)@BASIS
    contrasts = BASIS[:, :2] if product else BASIS

    def feature(w):
        return ((model.forward(w, data[0])[0]@contrasts)*data[1].sqrt()[:, None]).flatten()

    jacobian = torch.autograd.functional.jacobian(feature, weight, vectorize=True)
    singular = torch.linalg.svdvals(jacobian)
    active_dimension = len(data[0])*contrasts.shape[1]
    inactive_dimension = (model.width+1)*(3-contrasts.shape[1])
    return dict(
        hidden_affine_rank=int((torch.linalg.svdvals(augmented)>1e-12).sum()),
        hidden_affine_min_singular=float(torch.linalg.svdvals(augmented).min()),
        readout_singular_values=torch.linalg.svdvals(readout).tolist(),
        inactive_readout_norm=float(readout[:, 2].norm()) if product else None,
        active_jacobian_rank=int((singular>1e-12).sum()),
        active_jacobian_min_singular=float(singular.min()),
        active_normal_directions=active_dimension, inactive_normal_directions=inactive_dimension,
        conditional_local_llc=(active_dimension+inactive_dimension)/2,
        conditional_multiplicity=1, stable_kl=float(population_kl(model, weight, data)[0]),
    )
