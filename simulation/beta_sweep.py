"""
Shadow-EMA β sweep at fixed (tier, α, regime).

Tests whether the shadow EMA rate β is well-tuned at the default 0.05.
Theory (autocorrelation argument in SHADOW_SCALING_NOTES.md): shadow's
smoothing peaks when the EMA averaging window τ_shadow = 1/β is matched
to the iterate's autocorrelation time τ_θ ≈ 1/α. So optimal β should
be ~α.

Output: a table and a plot of live/shadow dispersion vs β.

Usage:
  python3 beta_sweep.py                          # T2 fixed, α=0.03
  python3 beta_sweep.py --tier 3 --state-dep --strength 1.0 --alpha 0.1
"""

import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import simulation as sim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', type=int, default=2, choices=[1, 2, 3])
    parser.add_argument('--alpha', type=float, default=0.03)
    parser.add_argument('--state-dep', action='store_true')
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--betas', type=str,
                        default='0.005,0.01,0.02,0.05,0.1,0.2,0.5',
                        help='Comma-separated β values to sweep')
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--n-cycles', type=int, default=None,
                        help='Override n_cycles (default uses tier preset)')
    parser.add_argument('--out-dir', type=str,
                        default='dreaming_lora_handoff/beta_sweep')
    args = parser.parse_args()

    sim.apply_tier(args.tier)
    if args.n_cycles is not None:
        sim.N_CYCLES = args.n_cycles

    os.makedirs(args.out_dir, exist_ok=True)
    betas = [float(b) for b in args.betas.split(',')]

    rng = np.random.default_rng(sim.GLOBAL_SEED)
    target = sim.initialize_target(rng)
    print(f'Tier {args.tier}: D={sim.D}, R={sim.R}, n_cycles={sim.N_CYCLES}, '
          f'n_seeds={args.n_seeds}, α={args.alpha}, '
          f'state_dep={args.state_dep}, γ={args.strength}')
    print(f'β values: {betas}')
    print()

    results = {}
    t_total = time.time()
    for beta in betas:
        sim.SHADOW_BETA = beta
        seed_results = []
        t0 = time.time()
        for seed in range(args.n_seeds):
            cfg = sim.make_cfg(
                alpha=args.alpha,
                state_dependent=args.state_dep,
                state_dependent_strength=args.strength,
            )
            res = sim.run_simulation(
                cfg, target, seed=seed + 7000 * int(beta * 10000),
                return_full_history=False,
            )
            seed_results.append({
                'live': res['final_window_dispersion'],
                'shadow': res['final_window_dispersion_shadow'],
                'dist_live': res.get('final_distance_to_target', np.nan),
                'dist_shadow': res.get('final_distance_to_target_shadow', np.nan),
                'diverged': res['diverged'],
            })
        wall = time.time() - t0

        finite_live = [r['live'] for r in seed_results if np.isfinite(r['live'])]
        finite_shadow = [r['shadow'] for r in seed_results if np.isfinite(r['shadow'])]
        live_mean = float(np.mean(finite_live)) if finite_live else float('inf')
        shadow_mean = float(np.mean(finite_shadow)) if finite_shadow else float('inf')
        ratio = (live_mean / shadow_mean
                 if np.isfinite(live_mean) and np.isfinite(shadow_mean) and shadow_mean > 0
                 else float('nan'))
        results[str(beta)] = {
            'live_mean': live_mean,
            'live_std': float(np.std(finite_live)) if finite_live else float('nan'),
            'shadow_mean': shadow_mean,
            'shadow_std': float(np.std(finite_shadow)) if finite_shadow else float('nan'),
            'live_over_shadow_ratio': ratio,
            'wall_time_s': wall,
        }
        print(f'  β={beta:6.4f}: live={live_mean:.4f}, shadow={shadow_mean:.4f}, '
              f'ratio={ratio:.3f}, wall {wall:.1f}s')

    print(f'\nTotal wall: {time.time() - t_total:.1f}s')

    summary = {
        'tier': args.tier, 'alpha': args.alpha,
        'state_dep': args.state_dep, 'strength': args.strength,
        'n_seeds': args.n_seeds, 'n_cycles': sim.N_CYCLES,
        'betas': betas,
        'results_per_beta': results,
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    live = [results[str(b)]['live_mean'] for b in betas]
    shadow = [results[str(b)]['shadow_mean'] for b in betas]
    ax.plot(betas, live, 'o-', label='Live readout', linewidth=1.5, markersize=8)
    ax.plot(betas, shadow, 's-', label='Shadow readout', linewidth=1.5, markersize=8)
    ax.axvline(args.alpha, color='gray', linestyle='--', alpha=0.5,
               label=f'β = α = {args.alpha}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('β (shadow EMA rate)')
    ax.set_ylabel('Final-window dispersion')
    title = f'T{args.tier} '
    title += f'state-dep γ={args.strength} ' if args.state_dep else 'fixed '
    title += f'α={args.alpha}'
    ax.set_title(title + ' — β sweep, dispersions')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ratios = [results[str(b)]['live_over_shadow_ratio'] for b in betas]
    ax.plot(betas, ratios, 'D-', linewidth=2, markersize=10)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.4)
    ax.axvline(args.alpha, color='gray', linestyle='--', alpha=0.5,
               label=f'β = α = {args.alpha}')
    ax.set_xscale('log')
    ax.set_xlabel('β')
    ax.set_ylabel('live / shadow dispersion ratio')
    ax.set_title(title + ' — shadow benefit')
    ax.legend()
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    tag = f'T{args.tier}_'
    tag += f'sd_g{args.strength:.1f}_' if args.state_dep else 'fixed_'
    tag += f'a{args.alpha:.2f}'
    plot_path = os.path.join(args.out_dir, f'beta_sweep_{tag}.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()

    json_path = os.path.join(args.out_dir, f'beta_sweep_{tag}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f'Plot: {plot_path}')
    print(f'JSON: {json_path}')


if __name__ == '__main__':
    main()
