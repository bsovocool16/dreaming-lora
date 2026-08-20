"""
Shadow Adapter Dispersion Check
================================

The B3/Main metric (final-window dispersion of the *live* adapter) showed
no inflation under ANY sampling regime. But the §4.4 toy plots clearly
show shadow smoothing the live trajectory (live-vs-shadow distance is
~half of live dispersion). The right way to detect "shadow does something"
is to compare the SHADOW's own dispersion to the LIVE's dispersion.

Run a single trajectory in each regime and report shadow-vs-live dispersion
ratios.
"""

import os
import json
import numpy as np

import simulation as sim


def trajectory_with_shadow(cfg, target, seed):
    """Reproduce run_simulation but record the shadow trajectory too."""
    rng = np.random.default_rng(seed)
    full_rank = not cfg.use_rank_constraint
    A, B = sim.initialize_factors(rng, full_rank)
    A_shadow = A.copy()
    B_shadow = B.copy() if not full_rank else np.eye(sim.D)

    T = cfg.n_cycles + 1
    live_history = np.zeros((T, sim.D, sim.D))
    shadow_history = np.zeros((T, sim.D, sim.D))
    live_history[0] = sim.deformation(A, B, full_rank)
    shadow_history[0] = sim.deformation(A_shadow, B_shadow, full_rank)

    for t in range(cfg.n_cycles):
        if cfg.target_noise_scale > 0:
            current_target = target + rng.normal(
                scale=cfg.target_noise_scale, size=target.shape)
        else:
            current_target = target

        delta_W_curr = sim.deformation(A, B, full_rank)
        try:
            h = sim.sample_dream_batch(rng, cfg, delta_W_curr, 1.0, target=current_target)
        except (np.linalg.LinAlgError, FloatingPointError):
            break

        grad_A, grad_B = sim.per_sample_gradients(A, B, h, current_target, full_rank)
        if cfg.trust_region is not None:
            grad_A = sim.operator_norm_clip(grad_A, cfg.trust_region)
            if not full_rank:
                grad_B = sim.operator_norm_clip(grad_B, cfg.trust_region)
        A = A - cfg.alpha * grad_A
        if not full_rank:
            B = B - cfg.alpha * grad_B
        A_shadow = (1 - sim.SHADOW_BETA) * A_shadow + sim.SHADOW_BETA * A
        if not full_rank:
            B_shadow = (1 - sim.SHADOW_BETA) * B_shadow + sim.SHADOW_BETA * B
        live_history[t+1] = sim.deformation(A, B, full_rank)
        shadow_history[t+1] = sim.deformation(A_shadow, B_shadow, full_rank)

    # Final-window dispersion for both trajectories
    fw = int(T * 0.25)
    burn = int(T * 0.5)
    start = max(T - fw, burn)
    n = T - start
    flat_live = live_history[start:].reshape(n, -1)
    flat_shadow = shadow_history[start:].reshape(n, -1)
    mean_live = flat_live.mean(axis=0)
    mean_shadow = flat_shadow.mean(axis=0)
    live_disp = np.linalg.norm(flat_live - mean_live[None, :], axis=1).mean()
    shadow_disp = np.linalg.norm(flat_shadow - mean_shadow[None, :], axis=1).mean()
    live_to_shadow = np.linalg.norm(live_history[start:] - shadow_history[start:],
                                     axis=(1, 2)).mean()
    return {
        'live_dispersion': float(live_disp),
        'shadow_dispersion': float(shadow_disp),
        'mean_live_to_shadow_dist': float(live_to_shadow),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()
    sim.apply_tier(args.tier)
    rng = np.random.default_rng(sim.GLOBAL_SEED)
    target = sim.initialize_target(rng)

    regimes = {
        'fixed': dict(state_dependent=False, symmetry_breaking=False),
        'state_dep_g1': dict(state_dependent=True, state_dependent_strength=1.0,
                              symmetry_breaking=False),
        'state_dep_g3': dict(state_dependent=True, state_dependent_strength=3.0,
                              symmetry_breaking=False),
        'sb_tau4': dict(state_dependent=True, state_dependent_strength=1.0,
                         symmetry_breaking=True, sb_temperature=4.0),
        'sb_tau1': dict(state_dependent=True, state_dependent_strength=1.0,
                         symmetry_breaking=True, sb_temperature=1.0),
    }
    results = {}
    for name, kwargs in regimes.items():
        # Average over a few seeds for stability
        seed_results = []
        for seed in range(5):
            cfg = sim.make_cfg(alpha=0.03, **kwargs)
            r = trajectory_with_shadow(cfg, target, seed=42 + seed)
            seed_results.append(r)
        mean = {k: float(np.mean([s[k] for s in seed_results]))
                for k in seed_results[0]}
        std = {k + '_std': float(np.std([s[k] for s in seed_results]))
                for k in seed_results[0]}
        results[name] = {**mean, **std}
        results[name]['shadow_over_live_ratio'] = (
            mean['shadow_dispersion'] / mean['live_dispersion'])
        print(f'{name}: live={mean["live_dispersion"]:.4f}, '
              f'shadow={mean["shadow_dispersion"]:.4f}, '
              f'ratio={results[name]["shadow_over_live_ratio"]:.3f}, '
              f'mean_lv_dist={mean["mean_live_to_shadow_dist"]:.4f}')

    os.makedirs('dreaming_lora_handoff/shadow_check', exist_ok=True)
    out_path = f'dreaming_lora_handoff/shadow_check/shadow_dispersion_T{args.tier}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults: {out_path}')


if __name__ == '__main__':
    main()
