"""
Direct Rotation Detection in Iterate Trajectories
==================================================

If the §4.4 toy has rotational dynamics, the iterate trajectory should
exhibit oscillation, which would show as:

  (a) Negative excursions in the autocorrelation function (AC) of the
      deviation-from-mean signal at some lag.
  (b) A peak in the power spectrum at a non-zero frequency.

A purely drifting+jittering system (Ornstein–Uhlenbeck-like) has positive
monotonically decaying AC and a 1/f^2 power spectrum with no peak.

Run several regimes at T2 and report:
- AC at lag 1, 10, 50, 100
- Whether AC ever goes below -0.1 (would indicate rotation)
- Position of power spectrum peak (excluding DC)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

import simulation as sim


def trajectory(cfg, target, seed):
    rng = np.random.default_rng(seed)
    full_rank = not cfg.use_rank_constraint
    A, B = sim.initialize_factors(rng, full_rank)
    T = cfg.n_cycles + 1
    history = np.zeros((T, sim.D, sim.D))
    history[0] = sim.deformation(A, B, full_rank)
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
        history[t+1] = sim.deformation(A, B, full_rank)
    return history


def analyze_trajectory(history, burn_frac=0.5):
    """Compute AC and power spectrum of dispersion-from-mean signal."""
    T = history.shape[0]
    burn = int(T * burn_frac)
    h = history[burn:]
    flat = h.reshape(h.shape[0], -1)
    mean = flat.mean(axis=0)
    devs = flat - mean[None, :]
    # Project onto top-2 PCA directions to get a 2D signal
    U, S, Vt = np.linalg.svd(devs, full_matrices=False)
    pc_signal = U[:, :2] * S[:2]  # T x 2
    # AC of the PC1 signal
    x = pc_signal[:, 0]
    x = x - x.mean()
    n = len(x)
    var = (x * x).mean()
    if var == 0:
        return None
    ac_lags = [1, 5, 10, 25, 50, 100, 200, 500]
    ac_vals = []
    for lag in ac_lags:
        if lag >= n:
            ac_vals.append(np.nan)
            continue
        c = (x[:-lag] * x[lag:]).mean() / var
        ac_vals.append(float(c))
    # Power spectrum
    spec = np.abs(np.fft.rfft(x))**2
    freqs = np.fft.rfftfreq(n)
    # Find peak excluding DC
    peak_idx = np.argmax(spec[1:]) + 1
    peak_freq = float(freqs[peak_idx])
    peak_period = 1.0 / peak_freq if peak_freq > 0 else np.inf
    # AC minimum (most negative point)
    full_ac = np.array([(x[:-lag] * x[lag:]).mean() / var
                         for lag in range(1, min(n // 4, 500))])
    return {
        'ac_at_lags': dict(zip(ac_lags, ac_vals)),
        'min_ac': float(full_ac.min()),
        'min_ac_lag': int(np.argmin(full_ac) + 1),
        'peak_freq': peak_freq,
        'peak_period_cycles': peak_period,
        'log_dc_to_peak_ratio': float(np.log10((spec[0] + 1e-10) / (spec[peak_idx] + 1e-10))),
        'pc_signal': pc_signal,
    }


def main():
    sim.apply_tier(2)
    rng = np.random.default_rng(sim.GLOBAL_SEED)
    target = sim.initialize_target(rng)

    regimes = {
        'fixed': dict(state_dependent=False, symmetry_breaking=False),
        'state_dep_g3': dict(state_dependent=True, state_dependent_strength=3.0,
                              symmetry_breaking=False),
        'sb_tau1': dict(state_dependent=True, state_dependent_strength=1.0,
                         symmetry_breaking=True, sb_temperature=1.0),
        'sb_tau1_no_shadow': dict(state_dependent=True, state_dependent_strength=1.0,
                                    symmetry_breaking=True, sb_temperature=1.0,
                                    use_shadow=False),
    }

    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for i, (name, kwargs) in enumerate(regimes.items()):
        cfg = sim.make_cfg(alpha=0.03, **kwargs)
        h = trajectory(cfg, target, seed=42)
        an = analyze_trajectory(h)
        if an is None:
            results[name] = {'error': 'no variance'}
            continue
        pc = an['pc_signal']
        results[name] = {k: v for k, v in an.items() if k != 'pc_signal'}
        # 2D PC trajectory plot (steady-state portion)
        ax = axes[i // 2, i % 2]
        ax.plot(pc[:, 0], pc[:, 1], lw=0.6, alpha=0.7)
        ax.scatter(pc[0, 0], pc[0, 1], color='green', s=40, label='start', zorder=5)
        ax.scatter(pc[-1, 0], pc[-1, 1], color='red', s=40, label='end', zorder=5)
        ax.set_xlabel('PC1 of steady-state dispersion')
        ax.set_ylabel('PC2 of steady-state dispersion')
        ax.set_title(f'{name}\nmin_AC={an["min_ac"]:.3f} @ lag {an["min_ac_lag"]}')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    os.makedirs('dreaming_lora_handoff/rotation_check', exist_ok=True)
    plot_path = 'dreaming_lora_handoff/rotation_check/pc_trajectories.png'
    plt.savefig(plot_path, dpi=120)
    plt.close()

    json_path = 'dreaming_lora_handoff/rotation_check/rotation_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f'\nPlot: {plot_path}')
    print(f'JSON: {json_path}')


if __name__ == '__main__':
    main()
