"""Aggregate independent SMC populations and plot free-energy curves.

Z estimates, not log Z estimates, are averaged. Error bars use variation
between complete independent populations; particles within one resampled
population are not treated as independent. Slope errors retain the covariance
between temperatures along each population's annealing path.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from experiment import HERE, write_json


def summarize(log_z):
    """Delta-method standard error for -log of independent mean Z estimates."""
    log_z = np.asarray(log_z)
    normalized_z = np.exp(log_z-logsumexp(log_z)+math.log(len(log_z)))
    free_energy = -float(logsumexp(log_z)-math.log(len(log_z)))
    se = float(normalized_z.std(ddof=1)/math.sqrt(len(log_z))) if len(log_z)>1 else math.nan
    return free_energy, se, -normalized_z


def summarize_energy(log_z, energies):
    """Pool unnormalized SMC moments, then divide by the pooled evidence.

    Each Z_hat times the terminal particle mean is an unbiased unnormalized
    moment in the ideal SMC algorithm. The ratio is consistent as independent
    populations increase, even at fixed particle count. Equal weighting of
    normalized population means does not have that fixed-N property.
    """
    energies=np.asarray(energies)
    _, _, negative_weights=summarize(log_z)
    weights=-negative_weights
    value=float(np.mean(weights*energies))
    influence=weights*(energies-value)
    se=float(influence.std(ddof=1)/math.sqrt(len(energies))) if len(energies)>1 else math.nan
    return value,se,influence


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def collect(root):
    grouped = defaultdict(list)
    for path in sorted(root.glob('*.json')):
        r = json.loads(path.read_text())
        if not isinstance(r, dict) or 'config' not in r or not r.get('complete') or r['config']['pilot']:
            continue
        c = r['config']
        key = (c['name'], c['gamma'], c['particles'], c['moves'], c['leapfrog_steps'],
               c['label'], c['schedule_sha256'])
        grouped[key].append(r)
    curves, slopes, diagnostics = [], [], []
    for key, runs in sorted(grouped.items()):
        name, gamma, n, moves, length, label, schedule_hash = key
        # A label is not enough to establish that runs estimate the same target
        # with the same HMC effort. Refuse accidental pooling across protocols.
        identity_keys = ('checkpoint_sha256', 'dataset_sha256', 'endpoints',
                         'high_tau', 'high_moves', 'high_length', 'code_sha256')
        identity = {k:runs[0]['config'].get(k) for k in identity_keys}
        if any({k:r['config'].get(k) for k in identity_keys} != identity for r in runs[1:]):
            raise ValueError(f'Incompatible targets or HMC settings in group {key}; use separate labels or directories.')
        if len({r['config']['seed'] for r in runs}) != len(runs):
            raise ValueError(f'Duplicate population seeds in group {key}.')
        settings = dict(name=name, gamma=gamma, particles=n, moves=moves, leapfrog_steps=length,
                        label=label, repetitions=len(runs), schedule_sha256=schedule_hash)
        if 'high_tau' in runs[0]['config']:
            settings.update({k:runs[0]['config'][k] for k in ('high_tau','high_moves','high_length')})
        taus = runs[0]['config']['endpoints']
        endpoint_rows = [{r['tau']:r for r in run['rows'] if r['endpoint']} for run in runs]
        previous = None
        previous_slope = None
        for tau in taus:
            endpoint = [r[tau] for r in endpoint_rows]
            free_energy, se, influence = summarize([r['log_z'] for r in endpoint])
            energies = np.array([r['energy'] for r in endpoint])
            energy,energy_se,energy_influence=summarize_energy([r['log_z'] for r in endpoint],energies)
            concentration_influence=influence-energy_influence
            row = dict(**settings, tau=tau, free_energy=free_energy, free_energy_se=se,
                       energy=energy, energy_se=energy_se,
                       concentration=free_energy-energy,
                       concentration_se=float(concentration_influence.std(ddof=1)/math.sqrt(len(runs)))
                       if len(runs)>1 else math.nan,
                       relative_entropy=energy-free_energy,
                       mean_population_energy=float(energies.mean()),
                       mean_population_energy_se=float(energies.std(ddof=1)/math.sqrt(len(runs)))
                       if len(runs)>1 else math.nan,
                       replicate_ess=float(len(runs)**2/np.square(influence).sum()),
                       largest_evidence_fraction=float((-influence).max()/len(runs)),
                       mean_log_free_energy=float(np.mean([r['free_energy'] for r in endpoint])),
                       mean_interval_ancestors=float(np.mean([r['interval_ancestors'] for r in endpoint])))
            if 'exact_free_energy' in endpoint[0]:
                row.update(exact_free_energy=endpoint[0]['exact_free_energy'], exact_energy=endpoint[0]['exact_energy'])
                row['free_energy_error'] = free_energy-row['exact_free_energy']
            curves.append(row)
            if previous is not None:
                old, old_influence = previous
                width = math.log(tau/old['tau'])
                slope_se = float((influence-old_influence).std(ddof=1)/math.sqrt(len(runs))/width) if len(runs)>1 else math.nan
                slope = dict(**settings, tau_from=old['tau'], tau_to=tau, midpoint=math.sqrt(tau*old['tau']),
                             slope=(free_energy-old['free_energy'])/width, slope_se=slope_se,
                             max_evidence_fraction=max(row['largest_evidence_fraction'],old['largest_evidence_fraction']))
                slope_influence=(influence-old_influence)/width
                if previous_slope is not None:
                    old_slope,old_slope_influence=previous_slope
                    slope.update(change_from_previous=slope['slope']-old_slope,
                        change_se=float((slope_influence-old_slope_influence).std(ddof=1)/math.sqrt(len(runs)))
                        if len(runs)>1 else math.nan)
                previous_slope=slope['slope'],slope_influence
                if 'exact_free_energy' in row:
                    slope['exact_slope'] = (row['exact_free_energy']-old['exact_free_energy'])/width
                slopes.append(slope)
            previous = row, influence
        for run in runs:
            records = run['rows']
            # Integrate energy against log tau on all positive bridge
            # temperatures. This is a consistency check, not independent data.
            log_temperature = np.log([r['tau'] for r in records])
            ti_tail = float(np.trapezoid([r['energy'] for r in records], log_temperature))
            tail_f = records[-1]['free_energy']-records[0]['free_energy']
            diagnostics.append(dict(**settings, seed=run['config']['seed'], seconds=run['seconds'],
                stages=len(records), min_incremental_ess_fraction=min(r['incremental_ess']/n for r in records),
                mean_acceptance=float(np.mean([r['acceptance'] for r in records])),
                min_acceptance=min(r['acceptance'] for r in records),
                mean_divergence_fraction=float(np.mean([r['divergence_fraction'] for r in records])),
                final_initial_ancestors=records[-1]['initial_ancestors'],
                final_interval_ancestors=records[-1]['interval_ancestors'],
                ti_free_energy_difference=ti_tail, smc_free_energy_difference=tail_f,
                ti_minus_smc=ti_tail-tail_f))
    paired = []
    arms = defaultdict(dict)
    for key, runs in grouped.items():
        name, *settings = key
        if name in ('factored_seed0', 'joint_seed0'):
            # Pilot hashes differ between arms, so compare matching numerical
            # settings and seeds, not the schedule hash.
            arms[tuple(settings[:-1])][name] = {r['config']['seed']:r for r in runs}
    for settings, pair in arms.items():
        if len(pair)!=2:continue
        gamma,n,moves,length,label = settings
        common = sorted(set(pair['factored_seed0']) & set(pair['joint_seed0']))
        if not common:continue
        previous = None
        for tau in pair['factored_seed0'][common[0]]['config']['endpoints']:
            rows = {}
            for arm in pair:
                rows[arm] = [next(r for r in pair[arm][seed]['rows'] if r['tau']==tau) for seed in common]
            f, _, fi = summarize([r['log_z'] for r in rows['factored_seed0']])
            j, _, ji = summarize([r['log_z'] for r in rows['joint_seed0']])
            influence = ji-fi
            fe,_,fei=summarize_energy([r['log_z'] for r in rows['factored_seed0']], [r['energy'] for r in rows['factored_seed0']])
            je,_,jei=summarize_energy([r['log_z'] for r in rows['joint_seed0']], [r['energy'] for r in rows['joint_seed0']])
            concentration_influence=influence-(jei-fei)
            row = dict(gamma=gamma,particles=n,moves=moves,leapfrog_steps=length,label=label,
                       repetitions=len(common),tau=tau,free_energy_difference=j-f,
                       free_energy_difference_se=float(influence.std(ddof=1)/math.sqrt(len(common))) if len(common)>1 else math.nan,
                       energy_difference=je-fe,
                       energy_difference_se=float((jei-fei).std(ddof=1)/math.sqrt(len(common))) if len(common)>1 else math.nan,
                       concentration_difference=(j-f)-(je-fe),
                       concentration_difference_se=float(concentration_influence.std(ddof=1)/math.sqrt(len(common)))
                       if len(common)>1 else math.nan,
                       largest_evidence_fraction=max(float((-fi).max()),float((-ji).max()))/len(common))
            if previous:
                old, old_influence = previous
                width=math.log(tau/old['tau'])
                row.update(tau_from=old['tau'],slope_difference=(row['free_energy_difference']-old['free_energy_difference'])/width,
                           slope_difference_se=float((influence-old_influence).std(ddof=1)/math.sqrt(len(common))/width)
                           if len(common)>1 else math.nan)
            paired.append(row)
            previous=row,influence
    return curves, slopes, diagnostics, paired


def figures(root, curves, slopes):
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    toys = [name for name in ('quadratic', 'quartic', 'curved') if any(r['name']==name for r in curves)]
    if toys:
        fig, axes = plt.subplots(2, len(toys), figsize=(12, 7), squeeze=False)
        for j, name in enumerate(toys):
            data = [r for r in curves if r['name']==name]
            for n in sorted(set(r['particles'] for r in data)):
                part = sorted([r for r in data if r['particles']==n], key=lambda r:r['tau'])
                axes[0,j].errorbar([r['tau'] for r in part], [r['free_energy_error'] for r in part],
                    yerr=[2*r['free_energy_se'] for r in part], marker='.', capsize=2, label=f'N={n}, R={part[0]["repetitions"]}')
                ss = sorted([r for r in slopes if r['name']==name and r['particles']==n], key=lambda r:r['midpoint'])
                axes[1,j].errorbar([r['midpoint'] for r in ss], [r['slope'] for r in ss],
                    yerr=[2*r['slope_se'] for r in ss], marker='.', capsize=2)
            axes[0,j].axhline(0, color='black', lw=1)
            axes[0,j].set_title(name.capitalize())
            ref = sorted([r for r in slopes if r['name']==name and r['particles']==max(s['particles'] for s in data)], key=lambda r:r['midpoint'])
            axes[1,j].plot([r['midpoint'] for r in ref], [r['exact_slope'] for r in ref], color='black', label='Exact finite-temperature slope')
            axes[0,j].legend(fontsize=8)
            for ax in axes[:,j]:
                ax.set_xscale('log'); ax.grid(alpha=.2)
            axes[1,j].set_xlabel(r'Precision $\tau$ (interval midpoint)')
        axes[0,0].set_ylabel('SMC free energy − exact free energy')
        axes[1,0].set_ylabel(r'$\Delta F / \Delta\log\tau$')
        axes[1,-1].legend(fontsize=8)
        fig.suptitle('Exact toy calibration; error bars: twice the between-run standard error')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(root/f'toy_free_energy.{ext}', dpi=180)
        plt.close(fig)
    neural = [r for r in curves if r['name'].endswith('_seed0')]
    if neural:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        styles = ['-', '--', ':', '-.']
        settings = sorted(set((r['particles'], r['moves'], r['label']) for r in neural))
        for name, color in [('factored_seed0', '#2166ac'), ('joint_seed0', '#b2182b')]:
            for i, (n, moves, label) in enumerate(settings):
                part = sorted([r for r in neural if (r['name'],r['particles'],r['moves'],r['label'])==(name,n,moves,label)], key=lambda r:r['tau'])
                if not part:continue
                display = f'{name.split("_")[0]}, N={n}, '+('longer HMC' if label=='long_hmc' else 'shorter HMC')
                for ax, field, error in [(axes[0],'free_energy','free_energy_se'),(axes[2],'energy','energy_se')]:
                    ax.errorbar([r['tau'] for r in part], [r[field] for r in part], yerr=[2*r[error] for r in part],
                                color=color, ls=styles[i%len(styles)], marker='.', capsize=2, label=display)
                    concentrated=[r for r in part if r['largest_evidence_fraction']>.75]
                    ax.plot([r['tau'] for r in concentrated],[r[field] for r in concentrated],
                            ls='none',marker='o',markersize=5,markerfacecolor='white',markeredgecolor=color,zorder=4)
                ss = sorted([r for r in slopes if (r['name'],r['particles'],r['moves'],r['label'])==(name,n,moves,label)], key=lambda r:r['midpoint'])
                axes[1].errorbar([r['midpoint'] for r in ss], [r['slope'] for r in ss], yerr=[2*r['slope_se'] for r in ss],
                                color=color, ls=styles[i%len(styles)], marker='.', capsize=2)
                concentrated=[r for r in ss if r['max_evidence_fraction']>.75]
                axes[1].plot([r['midpoint'] for r in concentrated],[r['slope'] for r in concentrated],
                             ls='none',marker='o',markersize=5,markerfacecolor='white',markeredgecolor=color,zorder=4)
            mcmc = {}
            for path in (HERE/'results_paired'/name).glob('*_tau*_gamma100.json'):
                saved=json.loads(path.read_text())
                if not saved.get('usable') or saved.get('gamma')!=100:continue
                tau=saved['tau']
                if tau not in mcmc or saved['ess']>mcmc[tau]['ess']:mcmc[tau]=saved
            if mcmc:
                values=sorted(mcmc.values(),key=lambda r:r['tau'])
                axes[2].errorbar([r['tau'] for r in values],[r['estimate'] for r in values],
                                yerr=[2*r['mcse'] for r in values],color=color,marker='x',ls='none',
                                markersize=7,capsize=3,label=f'{name.split("_")[0]}: earlier MCMC')
        for ax in axes:
            ax.set_xscale('log'); ax.grid(alpha=.2); ax.set_xlabel(r'Precision $\tau$')
        axes[0].set_ylabel(r'Free energy $F=-\log Z$')
        axes[1].set_ylabel(r'$\Delta F / \Delta\log\tau$')
        axes[1].set_xlabel(r'Precision $\tau$ (interval midpoint)')
        axes[2].set_ylabel(r'Endpoint mean energy $\tau E[\Delta K]$')
        axes[0].plot([],[],ls='none',marker='o',markerfacecolor='white',markeredgecolor='black',
                     label='Open: >75% evidence from one run')
        axes[0].legend(fontsize=7)
        axes[2].legend(fontsize=7)
        fig.suptitle(r'Paired checkpoints, fixed localization $\gamma=100$; error bars: twice between-run SE')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(root/f'transformer_free_energy.{ext}', dpi=180)
        plt.close(fig)


def decomposition_figures(root, curves, paired):
    """Separate prediction cost E=tau<K> and relative entropy cost C=F-E.

    The primary figure uses the longer-HMC runs. A second figure exposes
    sensitivity of the component differences to particle count and HMC effort.
    Shading marks the upper decade where earlier convergence checks were weaker.
    """
    primary=[r for r in curves if r['name'].endswith('_seed0') and
             r['label']=='long_hmc' and r['particles']==128]
    if not primary:
        return
    terms=[('free_energy_difference','free_energy_difference_se','black',r'Total $\Delta F$'),
           ('concentration_difference','concentration_difference_se','#6a3d9a',r'Entropy cost $\Delta C$'),
           ('energy_difference','energy_difference_se','#d87500',r'Prediction cost $\Delta E$')]

    def draw(ax, rows, value, error, color, label, mark_concentration=False):
        rows=sorted(rows,key=lambda r:r['tau'])
        x=np.array([r['tau'] for r in rows])
        y=np.array([r[value] for r in rows])
        se=np.array([r[error] for r in rows])
        ax.plot(x,y,color=color,lw=2,marker='.',ms=5,label=label)
        ax.fill_between(x,y-2*se,y+2*se,color=color,alpha=.15,linewidth=0)
        if mark_concentration:
            flag=np.array([r['largest_evidence_fraction']>.75 for r in rows])
            ax.plot(x[flag],y[flag],ls='none',marker='o',ms=5,mfc='white',mec=color,zorder=4)

    def decorate(ax):
        ax.set_xscale('log')
        ax.axvspan(1e7,1e8,color='#777777',alpha=.09,zorder=0)
        ax.axhline(0,color='#777777',lw=.7)
        ax.set_xlim(1,1e8)
        ax.set_xlabel(r'Inverse temperature $\tau$')
        ax.grid(alpha=.18)

    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for name,color in [('factored_seed0','#2166ac'),('joint_seed0','#b2182b')]:
        rows=[r for r in primary if r['name']==name]
        draw(axes[0],rows,'energy','energy_se',color,name.split('_')[0].capitalize())
        draw(axes[1],rows,'concentration','concentration_se',color,name.split('_')[0].capitalize())
    gap=[r for r in paired if r['label']=='long_hmc' and r['particles']==128]
    for value,error,color,label in terms:
        draw(axes[2],gap,value,error,color,label)
    for ax in axes:decorate(ax)
    axes[0].set_title('Prediction cost')
    axes[0].set_ylabel(r'$E(\tau)=\tau\,\mathbb{E}_{\pi_\tau}[\Delta K]$')
    axes[1].set_title('Entropy contribution')
    axes[1].set_ylabel(r'$C(\tau)=-S_{\rm rel}=D_{\rm KL}(\pi_\tau\Vert p_0)$')
    axes[2].set_title('Joint minus factored')
    axes[2].set_ylabel('Contribution to the free-energy difference')
    axes[0].legend(frameon=False,fontsize=9)
    axes[2].legend(frameon=False,fontsize=9)
    fig.suptitle(r'Free-energy decomposition: $F=E+C$  |  Fixed localization $\gamma=100$',fontsize=13)
    fig.text(.5,.015,'Longer HMC, 128 particles, four runs per arm. Bands: ±2 Monte Carlo SE with covariance. Shading: upper-range sensitivity.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95))
    for ext in ('png','pdf'):fig.savefig(root/f'free_energy_decomposition.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
    for ax,(label,n,title) in zip(axes,[('main',128,'128 particles, shorter HMC'),
                                      ('main',256,'256 particles, shorter HMC'),
                                      ('long_hmc',128,'128 particles, longer HMC')]):
        rows=[r for r in paired if r['label']==label and r['particles']==n]
        for value,error,color,legend in terms:
            draw(ax,rows,value,error,color,legend,mark_concentration=True)
        decorate(ax)
        ax.set_title(title)
    axes[0].set_ylabel('Joint minus factored')
    axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Sensitivity of the free-energy decomposition to sampling effort',fontsize=13)
    fig.text(.5,.015,'Bands: ±2 Monte Carlo SE. Open markers: >75% of evidence from one run in either arm. Shading: upper-range sensitivity.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95))
    for ext in ('png','pdf'):fig.savefig(root/f'free_energy_decomposition_sensitivity.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def simple_free_energy_figure(root, curves):
    """Show only the primary paired curves, marking the unsettled upper range."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, color, label in [('factored_seed0', '#2166ac', 'Factored'),
                               ('joint_seed0', '#b2182b', 'Joint')]:
        rows = sorted([r for r in curves if r['name'] == name and
                       r['label'] == 'long_hmc' and r['particles'] == 128],
                      key=lambda r: r['tau'])
        if not rows:
            plt.close(fig)
            return
        x = np.array([r['tau'] for r in rows])
        y = np.array([r['free_energy'] for r in rows])
        se = np.array([r['free_energy_se'] for r in rows])
        ax.plot(x[x <= 1e7], y[x <= 1e7], color=color, lw=2.2, label=label)
        ax.plot(x[x >= 1e7], y[x >= 1e7], color=color, lw=2.2, ls='--')
        ax.fill_between(x, y-2*se, y+2*se, color=color, alpha=.2)
    ax.set(xscale='log', xlim=(1, 1.25e8), ylim=(0, 255),
           xlabel=r'Inverse temperature $\tau$', ylabel=r'Free energy $F=-\log Z$')
    ax.legend(frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.16)
    fig.text(.5, .015, 'Fixed localization γ=100. Longer HMC, 4 independent runs per checkpoint.\n'
             'Bands: ±2 Monte Carlo SE. Dashed range: upper-end estimates provisional.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(root/f'free_energy_simple.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def curve_slope_fits(root):
    """Fit F = intercept + slope * log(tau) over prespecified upper windows.

    Ordinary least squares describes the curve within each window. Monte Carlo
    errors propagate independent-population influence vectors through the fit,
    retaining all temperature correlations; they exclude extrapolation bias.
    """
    rows = []
    windows = [(1e5, 1e6), (1e6, 1e7), (1e7, 1e8), (1e6, 1e8)]
    for name in ('factored_seed0', 'joint_seed0'):
        runs = [json.loads(p.read_text()) for p in
                sorted(root.glob(f'{name}_long_hmc_n128_seed*.json'))]
        runs = [r for r in runs if r.get('complete') and not r['config']['pilot']]
        if len(runs) < 2:
            continue
        endpoints = [{r['tau']: r for r in run['rows'] if r['endpoint']} for run in runs]
        for low, high in windows:
            taus = sorted(t for t in endpoints[0] if low <= t <= high)
            x = np.log(taus)
            centered = x-x.mean()
            slope_weights = centered/np.dot(centered, centered)
            summaries = [summarize([r[t]['log_z'] for r in endpoints]) for t in taus]
            y = np.array([r[0] for r in summaries])
            influence = np.array([r[2] for r in summaries])
            slope = float(slope_weights@y)
            intercept = float(y.mean()-slope*x.mean())
            slope_influence = slope_weights@influence
            rows.append(dict(name=name, gamma=100., label='long_hmc', particles=128,
                             repetitions=len(runs), tau_from=low, tau_to=high,
                             points=len(taus), slope=slope,
                             slope_se=float(slope_influence.std(ddof=1)/math.sqrt(len(runs))),
                             intercept=intercept,
                             max_absolute_residual=float(np.abs(y-intercept-slope*x).max())))
    write_csv(root/'curve_slope_fits.csv', rows)
    write_json(root/'curve_slope_fits.json', rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'results_smc')
    args = parser.parse_args()
    curves, slopes, diagnostics, paired = collect(args.output)
    for name, rows in [('curves',curves), ('slopes',slopes), ('diagnostics',diagnostics), ('paired_differences',paired)]:
        write_csv(args.output/f'{name}.csv', rows)
        write_json(args.output/f'{name}.json', rows)
    figures(args.output, curves, slopes)
    decomposition_figures(args.output, curves, paired)
    simple_free_energy_figure(args.output, curves)
    curve_slope_fits(args.output)
    print(f'{len(curves)} curve points, {len(slopes)} intervals, {len(diagnostics)} completed production runs.')


if __name__ == '__main__':
    main()
