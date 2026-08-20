"""
Cross-tier analysis of live/shadow readout dispersion.

Combines T1, T2, T3 results from the new readout-based metric and produces:
  - Live/shadow ratio table at α=0.03 across (tier, regime)
  - Shadow scaling exponent p (shadow_disp ~ α^p) per (tier, regime)
  - Side-by-side scaling plot showing dimension dependence

Run after T3 jobs complete.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = '/Users/bsovocool/Documents/Claude/Projects/Dreaming LoRA/dreaming_lora_handoff'

REGIMES = [
    ('T1', 'fixed',     'sim_tier1_readout/fixed/results.json'),
    ('T1', 'sd_g1.0',   'sim_tier1_readout/sd/results.json'),
    ('T2', 'fixed',     'sim_tier2_readout/fixed/results.json'),
    ('T2', 'sd_g0.3',   'sim_tier2_readout/sd_g03/results.json'),
    ('T2', 'sd_g1.0',   'sim_tier2_readout/sd_g10/results.json'),
    ('T2', 'sd_g3.0',   'sim_tier2_readout/sd_g30/results.json'),
    ('T3', 'fixed',     'sim_tier3_readout/fixed/results.json'),
    ('T3', 'sd_g1.0',   'sim_tier3_readout/sd_g10/results.json'),
    ('T3', 'sd_g3.0',   'sim_tier3_readout/sd_g30/results.json'),
]


def fit_power_law(alphas, disps, alpha_min=None, alpha_max=None):
    """Fit dispersion ~ C * α^p in log-log space. Optionally restrict range."""
    a = np.array(alphas, dtype=float)
    d = np.array(disps, dtype=float)
    mask = np.isfinite(d) & (d > 0)
    if alpha_min is not None:
        mask &= a >= alpha_min
    if alpha_max is not None:
        mask &= a <= alpha_max
    if mask.sum() < 2:
        return None, None
    log_a = np.log(a[mask])
    log_d = np.log(d[mask])
    slope, intercept = np.polyfit(log_a, log_d, 1)
    return slope, np.exp(intercept)


def load(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


def main():
    rows = []
    print('=' * 100)
    print(f'{"Tier":<6} {"Regime":<10} {"α=0.03 live":<13} {"α=0.03 shadow":<15} {"ratio":<8} {"live exp":<10} {"shadow exp":<12}')
    print('=' * 100)

    plots_per_tier = {'T1': [], 'T2': [], 'T3': []}

    for tier, regime, path in REGIMES:
        data = load(path)
        if data is None:
            print(f'{tier:<6} {regime:<10} (results.json not yet present at {path})')
            continue

        main_data = data['main']
        scaling = data['scaling_dispersion_vs_alpha']
        alphas = sorted([float(a) for a in scaling.keys()])
        live_disps = [scaling[str(a)].get('live_mean', scaling[str(a)].get('mean'))
                      for a in alphas]
        shadow_disps = [scaling[str(a)].get('shadow_mean') for a in alphas]

        # Power-law exponent fit, excluding α=0.005 which often hasn't converged
        live_p, _ = fit_power_law(alphas, live_disps, alpha_min=0.01)
        shadow_p, _ = fit_power_law(alphas, shadow_disps, alpha_min=0.01)

        # α=0.03 numbers from main (if present)
        if '0.03' in main_data:
            m = main_data['0.03']
            live03 = m['live_dispersion_mean']
            shadow03 = m['shadow_dispersion_mean']
            ratio = m['live_over_shadow_ratio']
        else:
            live03 = shadow03 = ratio = float('nan')

        rows.append({
            'tier': tier, 'regime': regime,
            'live_03': live03, 'shadow_03': shadow03, 'ratio_03': ratio,
            'live_exp': live_p, 'shadow_exp': shadow_p,
            'alphas': alphas, 'live_disps': live_disps,
            'shadow_disps': shadow_disps,
        })
        plots_per_tier[tier].append((regime, alphas, live_disps, shadow_disps))

        print(f'{tier:<6} {regime:<10} '
              f'{live03:<13.5f} {shadow03:<15.5f} {ratio:<8.3f} '
              f'{live_p if live_p else "—":<10} '
              f'{shadow_p if shadow_p else "—":<12}')

    print()
    print('Dispersion ~ α^p fitted on α ≥ 0.01 (excluding small-α transient).')
    print('Live exponent should be ~0.5 (theory). Shadow exponent measures the EMA smoothing benefit.')
    print()

    # Three-panel scaling plot, one per tier
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, tier in zip(axes, ['T1', 'T2', 'T3']):
        for regime, alphas, live, shadow in plots_per_tier[tier]:
            color = {'fixed': 'C0', 'sd_g0.3': 'C2', 'sd_g1.0': 'C1', 'sd_g3.0': 'C3'}.get(regime, 'C4')
            ax.plot(alphas, live, 'o-', color=color, label=f'{regime} live', alpha=0.9)
            ax.plot(alphas, shadow, 's--', color=color, label=f'{regime} shadow', alpha=0.6)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('α')
        ax.set_ylabel('Final-window dispersion')
        ax.set_title(f'{tier}')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    out_path = os.path.join(ROOT, 'sim_tier3_readout/scaling_cross_tier.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f'Cross-tier scaling plot: {out_path}')

    # Save table to JSON
    json_path = os.path.join(ROOT, 'sim_tier3_readout/cross_tier_analysis.json')
    serializable = []
    for r in rows:
        serializable.append({k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                              for k, v in r.items() if k not in ('alphas',)})
        serializable[-1]['alphas'] = r['alphas']
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f'JSON: {json_path}')


if __name__ == '__main__':
    main()
