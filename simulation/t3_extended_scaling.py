"""
T3 Extended-Cycle Scaling Rerun
================================

The original T3 (state-dependent) run reported broken √α scaling at d=128,
but at α ≤ 0.02 the trajectories had not reached steady state (10000 cycles
not enough). This script reruns just the small-α scaling subset at 50000
cycles to firm up the exponent.

Outputs a JSON record and a scaling plot extending the original.
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
    parser.add_argument('--alphas', type=str, default='0.005,0.01,0.02',
                        help='Comma-separated stepsizes')
    parser.add_argument('--n-cycles', type=int, default=30000)
    parser.add_argument('--n-seeds', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--out-dir', type=str,
                        default='dreaming_lora_handoff/sim_tier3_extended')
    args = parser.parse_args()

    sim.apply_tier(3)
    sim.N_CYCLES = args.n_cycles
    os.makedirs(args.out_dir, exist_ok=True)

    alphas = [float(x) for x in args.alphas.split(',')]

    rng = np.random.default_rng(sim.GLOBAL_SEED)
    target = sim.initialize_target(rng)
    print(f'Tier 3 extended: D={sim.D}, R={sim.R}, '
          f'n_cycles={args.n_cycles}, n_seeds={args.n_seeds}', flush=True)
    print(f'||W*||_F={np.linalg.norm(target):.4f}', flush=True)
    print(f'α values: {alphas}', flush=True)

    results = {}
    t_total = time.time()
    for alpha in alphas:
        cfg = sim.make_cfg(
            alpha=alpha,
            state_dependent=True,
            state_dependent_strength=args.gamma,
            n_cycles=args.n_cycles,
        )
        seed_results = []
        t0 = time.time()
        for seed in range(args.n_seeds):
            res = sim.run_simulation(
                cfg, target,
                seed=seed + 2000 * int(alpha * 1000),
                return_full_history=False,
            )
            seed_results.append({
                'final_window_dispersion': res['final_window_dispersion'],
                'final_window_dispersion_shadow': res.get(
                    'final_window_dispersion_shadow', float('nan')),
                'final_distance_to_target': res.get(
                    'final_distance_to_target', float('nan')),
                'final_distance_to_target_shadow': res.get(
                    'final_distance_to_target_shadow', float('nan')),
                'diverged': res['diverged'],
            })
        wall = time.time() - t0
        finite_live = [r['final_window_dispersion'] for r in seed_results
                       if np.isfinite(r['final_window_dispersion'])]
        finite_shadow = [r['final_window_dispersion_shadow'] for r in seed_results
                         if np.isfinite(r['final_window_dispersion_shadow'])]
        n_div = sum(1 for r in seed_results if r['diverged'])
        live_mean = float(np.mean(finite_live)) if finite_live else float('inf')
        shadow_mean = float(np.mean(finite_shadow)) if finite_shadow else float('inf')
        ratio = (live_mean / shadow_mean
                 if np.isfinite(live_mean) and np.isfinite(shadow_mean) and shadow_mean > 0
                 else float('nan'))
        results[str(alpha)] = {
            'n_seeds_diverged': n_div,
            'n_seeds_total': args.n_seeds,
            'live_disp_mean': live_mean,
            'live_disp_std': float(np.std(finite_live)) if finite_live else float('nan'),
            'shadow_disp_mean': shadow_mean,
            'shadow_disp_std': float(np.std(finite_shadow)) if finite_shadow else float('nan'),
            'live_over_shadow_ratio': ratio,
            'wall_time_s': wall,
        }
        print(f'  α={alpha}: live={live_mean:.4f} '
              f'(CV {100*results[str(alpha)]["live_disp_std"]/live_mean:.1f}%), '
              f'shadow={shadow_mean:.4f}, '
              f'ratio={ratio:.3f}, '
              f'wall {wall:.1f}s, div {n_div}/{args.n_seeds}', flush=True)

    summary = {
        'tier': 3, 'D': sim.D, 'R': sim.R,
        'n_cycles': args.n_cycles, 'n_seeds': args.n_seeds,
        'gamma': args.gamma,
        'results_per_alpha': results,
        'total_wall_time_s': time.time() - t_total,
    }

    # Add C estimates (both readouts)
    summary['C_estimates_live'] = {
        a: results[a]['live_disp_mean'] / np.sqrt(float(a))
        for a in results if np.isfinite(results[a]['live_disp_mean'])
    }
    summary['C_estimates_shadow'] = {
        a: results[a]['shadow_disp_mean'] / np.sqrt(float(a))
        for a in results if np.isfinite(results[a]['shadow_disp_mean'])
    }

    # Pull in the new readout-metric T3 sd_g10 results for comparison plot
    # (matches the gamma we're rerunning at; fallback to legacy if not present)
    if abs(args.gamma - 1.0) < 1e-6:
        orig_path = 'dreaming_lora_handoff/sim_tier3_readout/sd_g10/results.json'
    elif abs(args.gamma - 3.0) < 1e-6:
        orig_path = 'dreaming_lora_handoff/sim_tier3_readout/sd_g30/results.json'
    else:
        orig_path = None
    orig = None
    if orig_path and os.path.exists(orig_path):
        with open(orig_path) as f:
            orig = json.load(f)

    # Plot: live + shadow, 10k vs extended
    fig, ax = plt.subplots(figsize=(10, 6))
    if orig:
        short_alphas = sorted([float(a) for a in orig['scaling_dispersion_vs_alpha']])
        short_live = [orig['scaling_dispersion_vs_alpha'][str(a)].get('live_mean',
                          orig['scaling_dispersion_vs_alpha'][str(a)].get('mean'))
                      for a in short_alphas]
        short_shadow = [orig['scaling_dispersion_vs_alpha'][str(a)].get('shadow_mean', np.nan)
                        for a in short_alphas]
        ax.plot(short_alphas, short_live, 'o-', color='C0', alpha=0.5,
                label='Original 10k, live', linewidth=1.2, markersize=7)
        ax.plot(short_alphas, short_shadow, 's--', color='C0', alpha=0.5,
                label='Original 10k, shadow', linewidth=1.2, markersize=7)

    ext_alphas = sorted([float(a) for a in results.keys()])
    ext_live = [results[str(a)]['live_disp_mean'] for a in ext_alphas]
    ext_live_std = [results[str(a)]['live_disp_std'] for a in ext_alphas]
    ext_shadow = [results[str(a)]['shadow_disp_mean'] for a in ext_alphas]
    ext_shadow_std = [results[str(a)]['shadow_disp_std'] for a in ext_alphas]
    ax.errorbar(ext_alphas, ext_live, yerr=ext_live_std, fmt='o-', color='C3',
                label=f'Extended {args.n_cycles}, live',
                linewidth=2.0, markersize=10, capsize=5)
    ax.errorbar(ext_alphas, ext_shadow, yerr=ext_shadow_std, fmt='s--', color='C3',
                label=f'Extended {args.n_cycles}, shadow',
                linewidth=2.0, markersize=10, capsize=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Stepsize α')
    ax.set_ylabel('Final-window dispersion')
    ax.set_title(f'T3 state-dep γ={args.gamma}, extended cycles ({args.n_cycles})')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, 'scaling_extended.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()

    json_path = os.path.join(args.out_dir, 'results_extended.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f'\nTotal wall: {summary["total_wall_time_s"]:.1f}s')
    print(f'JSON: {json_path}')
    print(f'Plot: {plot_path}')


if __name__ == '__main__':
    main()
