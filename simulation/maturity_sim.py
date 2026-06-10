"""
Dreaming LoRA — Maturity Signal Diagnostic (§5.2)
==================================================

Records the developmental signals the paper proposes:

  μ_struct (eq. 8)     — participation ratio of S_Q + S_V (we have Q only).
  μ_struct ratcheted   — high-water-mark version.
  μ_exp                — 1 - exp(-N_t / N_sat), normalized consolidation history.

This is a diagnostic instrumented over a single trajectory rather than an
ablation battery: the goal is to characterize the maturity signals' dynamics
in the toy regime and check whether they behave the way the paper assumes
(monotone, well-distributed across [0,1], usable as a phase transition gate).

Outputs a multi-panel plot and a JSON record.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

import simulation as sim


def participation_ratio(sigma: np.ndarray, r_eff: int) -> float:
    """
    μ_struct = (Σ σ_i)^2 / (r_eff · Σ σ_i^2).

    For our toy with only Q-side adapter, sigma are the singular values
    of S = ΔW^T ΔW (which has rank ≤ R). The paper's r_eff = min(2r, d_in)
    accommodates S_Q + S_V where each has rank ≤ r; we have a single Q-side
    adapter, so the analog is r_eff = min(r, D) = R.
    """
    sum1 = sigma.sum()
    sum2 = (sigma * sigma).sum()
    if sum2 <= 0:
        return 0.0
    return float(sum1 * sum1 / (r_eff * sum2))


def run_with_maturity_tracking(
    cfg: sim.Config,
    target: np.ndarray,
    seed: int,
    n_sat: int,
) -> dict:
    """
    Drop-in equivalent of sim.run_simulation, augmented with per-cycle
    recording of μ_struct, μ_struct_ratcheted, μ_exp, and the singular-value
    spectrum of S = ΔW^T ΔW (top sim.R entries, since S has rank ≤ R).
    """
    rng = np.random.default_rng(seed)
    full_rank = not cfg.use_rank_constraint

    A, B = sim.initialize_factors(rng, full_rank)
    T = cfg.n_cycles + 1
    distance_curve = np.zeros(T)
    mu_struct_curve = np.zeros(T)
    mu_struct_ratched = np.zeros(T)
    mu_exp_curve = np.zeros(T)
    eff_rank_curve = np.zeros(T, dtype=int)
    # Top-R singular values of S over time (S has rank ≤ R).
    top_sigma_curve = np.zeros((T, sim.R))

    r_eff = sim.R  # see participation_ratio docstring

    def record_maturity(idx, delta_W, n_consol):
        S = delta_W.T @ delta_W
        # Eigenvalues of a symmetric PSD matrix — use eigvalsh.
        eigs = np.linalg.eigvalsh(S)
        eigs = eigs[::-1]  # descending
        # Singular values of ΔW are sqrt(eigenvalues of ΔW^T ΔW), but the
        # paper defines μ_struct over the eigenvalues of S itself; those are
        # what we want to use.
        # Use only the top sim.R values; the rest are numerically zero.
        sigma = np.maximum(eigs[:sim.R], 0.0)
        top_sigma_curve[idx] = sigma
        mu = participation_ratio(sigma, r_eff)
        mu_struct_curve[idx] = mu
        mu_struct_ratched[idx] = max(mu, mu_struct_ratched[idx-1]) if idx > 0 else mu
        mu_exp_curve[idx] = 1.0 - np.exp(-n_consol / n_sat)
        eff_rank_curve[idx] = sim.effective_rank(delta_W)
        distance_curve[idx] = np.linalg.norm(delta_W - target)

    delta_W = sim.deformation(A, B, full_rank)
    record_maturity(0, delta_W, n_consol=0)

    for t in range(cfg.n_cycles):
        if cfg.target_noise_scale > 0:
            current_target = target + rng.normal(
                scale=cfg.target_noise_scale, size=target.shape)
        else:
            current_target = target

        delta_W_curr = sim.deformation(A, B, full_rank)
        try:
            h_batch = sim.sample_dream_batch(
                rng, cfg, delta_W_curr, scale_mult=1.0, target=current_target)
        except (np.linalg.LinAlgError, FloatingPointError):
            break

        grad_A, grad_B = sim.per_sample_gradients(
            A, B, h_batch, current_target, full_rank)

        if cfg.trust_region is not None:
            grad_A = sim.operator_norm_clip(grad_A, cfg.trust_region)
            if not full_rank:
                grad_B = sim.operator_norm_clip(grad_B, cfg.trust_region)

        A = A - cfg.alpha * grad_A
        if not full_rank:
            B = B - cfg.alpha * grad_B

        delta_W = sim.deformation(A, B, full_rank)
        record_maturity(t + 1, delta_W, n_consol=t + 1)

    return {
        'distance_curve': distance_curve,
        'mu_struct_curve': mu_struct_curve,
        'mu_struct_ratched': mu_struct_ratched,
        'mu_exp_curve': mu_exp_curve,
        'eff_rank_curve': eff_rank_curve,
        'top_sigma_curve': top_sigma_curve,
    }


def plot_maturity_panels(results_dict: dict, out_path: str, n_sat: int):
    """
    Multi-panel plot:
      (0,0) distance to target
      (0,1) μ_struct (raw + ratcheted), μ_exp
      (1,0) effective rank
      (1,1) top-R singular values of S
    One curve per condition.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {
        'fixed':        ('C0', '-'),
        'state_dep':    ('C1', '-'),
        'sb':           ('C2', '-'),
    }

    # (0,0) Distance to target
    ax = axes[0, 0]
    for name, res in results_dict.items():
        c, ls = colors[name]
        t = np.arange(len(res['distance_curve']))
        ax.plot(t, res['distance_curve'], color=c, linestyle=ls, label=name, lw=1.2)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('||ΔW - W*||_F')
    ax.set_title('Distance to target')
    ax.legend()
    ax.grid(alpha=0.3)

    # (0,1) μ_struct and μ_exp
    ax = axes[0, 1]
    for name, res in results_dict.items():
        c, _ = colors[name]
        t = np.arange(len(res['mu_struct_curve']))
        ax.plot(t, res['mu_struct_curve'], color=c, lw=0.8, alpha=0.5,
                label=f'{name} μ_struct (raw)')
        ax.plot(t, res['mu_struct_ratched'], color=c, lw=1.5,
                linestyle='--', label=f'{name} μ_struct (ratcheted)')
    # μ_exp depends only on cycle count, same for all
    t = np.arange(len(next(iter(results_dict.values()))['mu_exp_curve']))
    ax.plot(t, next(iter(results_dict.values()))['mu_exp_curve'],
            color='k', linestyle=':', lw=1.5, label=f'μ_exp (N_sat={n_sat})')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Maturity signal')
    ax.set_title('μ_struct (raw & ratcheted) and μ_exp')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

    # (1,0) Effective rank
    ax = axes[1, 0]
    for name, res in results_dict.items():
        c, _ = colors[name]
        t = np.arange(len(res['eff_rank_curve']))
        ax.plot(t, res['eff_rank_curve'], color=c, lw=1.2, label=name)
    ax.axhline(sim.R, color='gray', linestyle='--', alpha=0.5,
               label=f'target rank R={sim.R}')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Effective rank of ΔW')
    ax.set_title('Effective rank trajectory')
    ax.legend()
    ax.grid(alpha=0.3)

    # (1,1) Top-R singular values of S
    ax = axes[1, 1]
    # Use the last condition for the spectrum plot to avoid clutter; pick
    # state_dep if present.
    pref = 'state_dep' if 'state_dep' in results_dict else next(iter(results_dict))
    res = results_dict[pref]
    sig = res['top_sigma_curve']
    t = np.arange(sig.shape[0])
    for i in range(sig.shape[1]):
        ax.plot(t, sig[:, i], lw=1.0, label=f'σ_{i+1}')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Eigenvalue of S = ΔW^T ΔW')
    ax.set_title(f'Top-R spectrum of S ({pref} condition)')
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Maturity signal diagnostic.')
    parser.add_argument('--tier', type=int, default=2, choices=[1, 2, 3])
    parser.add_argument('--alpha', type=float, default=0.03)
    parser.add_argument('--n-sat', type=int, default=1000,
                        help='μ_exp saturation scale')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str,
                        default='dreaming_lora_handoff/maturity_sim')
    args = parser.parse_args()

    sim.apply_tier(args.tier)
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    target = sim.initialize_target(rng)
    print(f'Tier {args.tier}: D={sim.D}, R={sim.R}, n_cycles={sim.N_CYCLES}')
    print(f'||W*||_F={np.linalg.norm(target):.4f}, '
          f'effective rank(W*)={sim.effective_rank(target)}')

    # Three conditions
    results = {}

    print('\nRunning fixed-distribution baseline...', flush=True)
    cfg_fixed = sim.make_cfg(alpha=args.alpha,
                             state_dependent=False,
                             symmetry_breaking=False)
    results['fixed'] = run_with_maturity_tracking(
        cfg_fixed, target, seed=args.seed, n_sat=args.n_sat)

    print('Running state-dependent (γ=1.0)...', flush=True)
    cfg_sd = sim.make_cfg(alpha=args.alpha,
                          state_dependent=True,
                          state_dependent_strength=1.0,
                          symmetry_breaking=False)
    results['state_dep'] = run_with_maturity_tracking(
        cfg_sd, target, seed=args.seed, n_sat=args.n_sat)

    print('Running state-dependent + symmetry-breaking...', flush=True)
    cfg_sb = sim.make_cfg(alpha=args.alpha,
                          state_dependent=True,
                          state_dependent_strength=1.0,
                          symmetry_breaking=True,
                          sb_temperature=4.0,
                          sb_oversample=4)
    results['sb'] = run_with_maturity_tracking(
        cfg_sb, target, seed=args.seed, n_sat=args.n_sat)

    # Summary numerical record
    summary = {
        'tier': args.tier, 'alpha': args.alpha, 'n_sat': args.n_sat,
        'final_mu_struct_raw': {k: float(v['mu_struct_curve'][-1])
                                for k, v in results.items()},
        'final_mu_struct_ratched': {k: float(v['mu_struct_ratched'][-1])
                                    for k, v in results.items()},
        'final_mu_exp': float(results['fixed']['mu_exp_curve'][-1]),
        'max_mu_struct_raw': {k: float(v['mu_struct_curve'].max())
                              for k, v in results.items()},
        'cycle_at_mu_struct_0p5': {},
        'final_eff_rank': {k: int(v['eff_rank_curve'][-1])
                           for k, v in results.items()},
        'final_distance': {k: float(v['distance_curve'][-1])
                           for k, v in results.items()},
    }
    for k, v in results.items():
        cross = np.where(v['mu_struct_ratched'] >= 0.5)[0]
        summary['cycle_at_mu_struct_0p5'][k] = (
            int(cross[0]) if len(cross) else None)

    plot_path = os.path.join(args.out_dir, f'maturity_T{args.tier}.png')
    plot_maturity_panels(results, plot_path, n_sat=args.n_sat)

    json_path = os.path.join(args.out_dir, f'maturity_T{args.tier}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nResults:\n{json.dumps(summary, indent=2)}')
    print(f'\nPlot saved to {plot_path}')
    print(f'JSON saved to {json_path}')


if __name__ == '__main__':
    main()
