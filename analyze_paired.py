"""Matched comparisons and representation controls for the same-data experiment."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import torch

from experiment import HERE, SIGNS, TOKEN_TO_STATE, Transformer, tensors, write_json


def residual_basis_transform(model, weight, matrix):
    """Exact GL(width) symmetry of this transformer without normalization.

    Predictions remain identical and hidden activations become h @ matrix.
    This provides a control for interpreting scale-sensitive geometry metrics.
    """
    transformed = weight.clone()
    pieces = model.unpack(transformed[None])
    with torch.no_grad():
        for index in (0, 1, 3):
            pieces[index][0].copy_(pieces[index][0]@matrix)
        for index in (2, 4):
            pieces[index][0].copy_(torch.linalg.solve(matrix, pieces[index][0]))
        if model.mlp_width:
            pieces[6][0].copy_(torch.linalg.solve(matrix, pieces[6][0]))
            pieces[8][0].copy_(pieces[8][0]@matrix)
            pieces[9][0].copy_(pieces[9][0]@matrix)
    return transformed


def plot_interaction_geometry(table, hidden_pair, output):
    """Show a learned interaction component across the 8 by 8 history grid."""
    contexts, probability, _, _ = table
    bits = (SIGNS[TOKEN_TO_STATE][contexts] > 0).astype(int)
    codes = np.einsum("nlf,l->nf", bits, 2**np.arange(contexts.shape[1]))
    design = np.column_stack([np.ones(len(contexts)), np.eye(8)[codes[:, 0]], np.eye(8)[codes[:, 1]]])
    root = np.sqrt(probability)[:, None]
    images, fractions = [], []
    for hidden in hidden_pair:
        coefficient = scipy.linalg.lstsq(design*root, hidden*root)[0]
        residual = hidden-design@coefficient
        _, _, right = scipy.linalg.svd(residual*root, full_matrices=False)
        component = residual@right[0]
        image = np.empty((8, 8))
        image[codes[:, 0], codes[:, 1]] = component
        images.append(image)
        fractions.append(np.sum(probability[:, None]*residual**2)/np.sum(probability[:, None]*(hidden-probability@hidden)**2))
    limit = max(np.max(np.abs(image)) for image in images)
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.5), constrained_layout=True)
    labels = [format(code, "03b")[::-1] for code in range(8)]
    for axis, arm, image, fraction in zip(axes, ("Factored", "Joint"), images, fractions):
        artist = axis.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower")
        axis.set(xticks=range(8), yticks=range(8), xticklabels=labels, yticklabels=labels,
                 xlabel="Observed factor 2 history", ylabel="Observed factor 1 history",
                 title=f"{arm}\nNonadditive variance: {100*fraction:.3g}%")
        axis.tick_params(axis="x", rotation=90)
    fig.colorbar(artist, ax=axes, label="Leading interaction component")
    fig.savefig(output/"interaction_geometry.png", dpi=180)
    fig.savefig(output/"interaction_geometry.pdf")
    plt.close(fig)


def activation_controls(table, factored, joint):
    """Distinguish dominant geometry from changes of activation basis.

    Whitened nonadditivity and column-space angles are invariant to invertible
    affine activation transforms, subject to the stated numerical rank cutoff.
    Raw nonadditive variance, used by the training screen, is scale sensitive.
    """
    contexts, probability, _, _ = table
    bits = (SIGNS[TOKEN_TO_STATE][contexts] > 0).astype(int)
    codes = np.einsum("nlf,l->nf", bits, 2**np.arange(contexts.shape[1]))
    design = np.column_stack([np.ones(len(contexts)), np.eye(8)[codes[:, 0]], np.eye(8)[codes[:, 1]]])
    root = np.sqrt(probability)[:, None]
    additive = scipy.linalg.orth(design*root, rcond=1e-12)
    whitened = {}
    result = {}
    for arm, hidden in (("factored", factored), ("joint", joint)):
        centered = (hidden-probability@hidden)*root
        basis = scipy.linalg.orth(centered, rcond=1e-12)
        whitened[arm] = basis
        result[arm] = dict(numerical_centered_rank=basis.shape[1], rank_relative_cutoff=1e-12,
                           whitened_nonadditive_fraction=float(np.linalg.norm(basis-additive@(additive.T@basis))**2/basis.shape[1]),
                           weighted_centered_variance=float(np.linalg.norm(centered)**2))
    result["column_space_cosines"] = scipy.linalg.svdvals(whitened["factored"].T@whitened["joint"]).tolist()
    for source, target in (("factored", "joint"), ("joint", "factored")):
        h = factored if source == "factored" else joint
        desired = joint if target == "joint" else factored
        affine = np.column_stack([np.ones(len(h)), h])
        coefficients = scipy.linalg.lstsq(affine*root, desired*root, cond=1e-12)[0]
        residual = np.sum(probability[:, None]*(affine@coefficients-desired)**2)
        total = np.sum(probability[:, None]*(desired-probability@desired)**2)
        result[f"{source}_to_{target}_affine_residual_fraction"] = float(residual/total)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"results_paired")
    args = parser.parse_args()
    torch.set_num_threads(1)
    analysis = args.output/"analysis"
    analysis.mkdir(exist_ok=True)
    saved = np.load(args.output/"dataset.npz")
    table = tuple(saved[k] for k in ("contexts", "context_probabilities", "target_probabilities", "beliefs"))
    data = tensors(table)
    model = Transformer()
    fits = [json.loads(p.read_text()) for p in sorted(args.output.glob("*_seed*/fit.json"))]
    controls = []
    for seed in sorted({fit["seed"] for fit in fits}):
        paths = [args.output/f"{arm}_seed{seed}"/"checkpoint.npz" for arm in ("factored", "joint")]
        if not all(p.exists() for p in paths):
            continue
        hidden = []
        for path in paths:
            weight = torch.from_numpy(np.load(path)["weights"])
            with torch.no_grad():
                hidden.append(model.forward(weight, data[0], True)[1][0].numpy())
        controls.append(dict(seed=seed, **activation_controls(table, *hidden)))
        if seed == 0:
            plot_interaction_geometry(table, hidden, analysis)
    write_json(analysis/"activation_controls.json", controls)
    samples = [dict(path=str(p), **json.loads(p.read_text()))
               for p in sorted(args.output.glob("*_seed*/*_tau*.json"))]
    diagnostic_rows = []
    for path in sorted(args.output.rglob("*_tau*.json")):
        row = json.loads(path.read_text())
        keys = ("seed", "arm", "sampler", "tau", "gamma", "estimate", "mcse", "rhat", "ess",
                "acceptance", "divergence_fraction", "usable", "sampler_seed")
        record = {key:row.get(key, "mala" if key == "sampler" else "") for key in keys}
        record.update(archived=path.parent != args.output/f"{row['arm']}_seed{row['seed']}",
                      path=str(path.relative_to(args.output)))
        diagnostic_rows.append(record)
    if diagnostic_rows:
        with (analysis/"all_sampler_diagnostics.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(diagnostic_rows)
    neighborhoods = []
    for row in samples:
        observations = row.get("observations", [])
        if not row["usable"] or not observations:
            continue
        fractions = np.array([record["nonadditive_fraction"] for record in observations])
        neighborhoods.append(dict(seed=row["seed"], arm=row["arm"], tau=row["tau"], gamma=row["gamma"],
                                  samples=int(fractions.size), mean_fraction=float(fractions.mean()),
                                  lower_quantile=float(np.quantile(fractions, 0.05)),
                                  upper_quantile=float(np.quantile(fractions, 0.95))))
    write_json(analysis/"posterior_geometry.json", neighborhoods)
    selected = {}
    for row in samples:
        if not row["usable"]:
            continue
        key = row["seed"], row["arm"], row["tau"], row["gamma"]
        if key not in selected or row["ess"] > selected[key]["ess"]:
            selected[key] = row
    paired = []
    for seed, arm, tau, gamma in sorted(selected):
        if arm != "factored" or (seed, "joint", tau, gamma) not in selected:
            continue
        f, j = selected[seed, arm, tau, gamma], selected[seed, "joint", tau, gamma]
        paired.append(dict(seed=seed, tau=tau, gamma=gamma, factored=f["estimate"], joint=j["estimate"],
                           joint_minus_factored=j["estimate"]-f["estimate"],
                           factored_mcse=f["mcse"], joint_mcse=j["mcse"],
                           factored_sampler=f.get("sampler", "mala"), joint_sampler=j.get("sampler", "mala")))
    if paired:
        with (analysis/"matched_comparisons.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(paired)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    colors = dict(factored="#2576ad", joint="#d26a28")
    for arm in colors:
        group = [f for f in fits if f["arm"] == arm]
        axes[0].plot([f["seed"] for f in group],
                     [100*f["additive_hidden_residual_fraction"] for f in group], "o", color=colors[arm], label=arm)
    axes[0].set(xlabel="Training seed", ylabel="Nonadditive activation variance (%)", title="After auxiliary loss is removed")
    axes[0].set_xticks(sorted({fit["seed"] for fit in fits}))
    axes[0].legend(frameon=False)
    for axis, gamma in zip(axes[1:], (100, 400)):
        for seed in sorted({r["seed"] for r in paired}):
            rows = sorted([r for r in paired if r["seed"] == seed and r["gamma"] == gamma], key=lambda r:r["tau"])
            for arm in colors:
                axis.errorbar([r["tau"] for r in rows], [r[arm] for r in rows],
                              yerr=[r[f"{arm}_mcse"] for r in rows], marker="o", markersize=3,
                              alpha=0.75, linewidth=1, color=colors[arm])
        axis.set(xscale="log", xlabel="Loss precision", ylabel="Localized estimate", title=f"Localization {gamma}")
    fig.savefig(analysis/"paired_comparison.png", dpi=180)
    fig.savefig(analysis/"paired_comparison.pdf")
    plt.close(fig)
    lines = ["# Matched same-data results", "",
             "Only configurations with usable sampling diagnostics in both arms enter these comparisons.",
             "Values are finite-precision localized estimates, not established asymptotic LLCs.", "",
             "![Paired comparison](./paired_comparison.png)", "",
             "Each curve is one training seed; error bars are Monte Carlo standard errors. Only matched usable configurations are plotted.", "",
             "The following seed-zero plot shows the leading activation component left after fitting additive factor contributions, on a common color scale.",
             "![Interaction geometry](./interaction_geometry.png)", "",
             "| Precision | Localization | Matched training seeds | Factored mean | Joint mean | Mean joint minus factored | SD of paired differences |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for tau, gamma in sorted({(r["tau"],r["gamma"]) for r in paired}):
        group = [r for r in paired if r["tau"] == tau and r["gamma"] == gamma]
        delta = [r["joint_minus_factored"] for r in group]
        sd = f"{np.std(delta, ddof=1):.3f}" if len(group)>1 else "—"
        lines.append(f"| {tau:.0e} | {gamma:g} | {len(group)} | {np.mean([r['factored'] for r in group]):.3f} | "
                     f"{np.mean([r['joint'] for r in group]):.3f} | {np.mean(delta):.3f} | {sd} |")
    lines += ["", "The standard deviation is across paired training seeds, not Monte Carlo error.",
              "Individual estimates and Monte Carlo errors are in matched_comparisons.csv.", "",
              "## Activation-coordinate controls", "",
              "The raw variance screen measures dominant geometry and is sensitive to coordinate scaling.",
              "Weak joint features may be present in the factored arm. The following whitened diagnostic",
              "is invariant to invertible affine activation transformations at numerical rank cutoff 1e-12.", "",
              "| Seed | Factored whitened nonadditivity | Joint whitened nonadditivity | Joint-code affine reconstruction error from factored code |",
              "|---|---:|---:|---:|"]
    for row in controls:
        lines.append(f"| {row['seed']} | {row['factored']['whitened_nonadditive_fraction']:.4f} | "
                     f"{row['joint']['whitened_nonadditive_fraction']:.4f} | {row['factored_to_joint_affine_residual_fraction']:.4g} |")
    lines += ["", "These controls prevent a change in the prominence of weak features from being interpreted",
              "as proof that one code lacks joint information. They do not by themselves classify the LLC."]
    if neighborhoods:
        lines += ["", "## Representation within sampled neighborhoods", "",
                  "Only usable runs with retained-state observations are shown. Quantiles describe the observed states and are not confidence intervals.", "",
                  "| Seed | Arm | Precision | Localization | Mean nonadditive variance (%) | 5%–95% range (%) |",
                  "|---|---|---:|---:|---:|---:|"]
        for row in sorted(neighborhoods, key=lambda r:(r["seed"],r["tau"],r["gamma"],r["arm"])):
            lines.append(f"| {row['seed']} | {row['arm']} | {row['tau']:.0e} | {row['gamma']:g} | "
                         f"{100*row['mean_fraction']:.3f} | {100*row['lower_quantile']:.3f}–{100*row['upper_quantile']:.3f} |")
    (analysis/"COMPARISON.md").write_text("\n".join(lines)+"\n")
    print(f"{len(fits)} fits, {len(samples)} sampler results, {len(paired)} matched comparisons", flush=True)


if __name__ == "__main__":
    main()
