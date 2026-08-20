"""
Dreaming LoRA — Bounded-Tracking Simulation (State-Dependent Dreams)
=====================================================================

Extends the Tier 2 simulation with state-dependent dream generation:
the dream sample covariance is biased toward the adapter's current
receptive field. This is the minimal mechanism that captures the
co-constitution between adapter and dream distribution discussed in §5.6.

Differences from the prior simulation.py (in dreaming_lora_handoff/):
  - Config has new fields: `state_dependent`, `state_dependent_strength`.
  - run_simulation samples dreams from N(0, sigma^2 I + gamma * Delta_W^T Delta_W)
    when state_dependent is True; otherwise behavior is identical to before.
  - Tier-aware CLI: --tier {1,2,3} switches D/R/n_cycles/n_seeds.
  - Output paths are configurable so we don't clobber the prior baselines.

Output:
  - One PNG per result figure, with optional filename prefix
  - results.json with summary numerical results
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ===========================================================================
# Hyperparameters (tier-overridable via CLI)
# ===========================================================================

# Architecture (tier-overridable)
D = 32                      # input dimension
R = 4                       # adapter rank

# Dream distribution
BATCH_SIZE_DEFAULT = 32     # samples per consolidation cycle (n)
DREAM_NOISE_SCALE = 1.0     # std of the isotropic dream component

# Update parameters
TRUST_REGION_DEFAULT = 1.0  # operator norm cap on per-cycle update
SHADOW_BETA = 0.05          # Polyak-Ruppert EMA rate

# Trajectory length (tier-overridable)
N_CYCLES = 5000
N_SEEDS = 10                # for variability estimation

# Stepsizes for main result
STEPSIZES_MAIN = [0.01, 0.03, 0.1]
ALPHA_MID = 0.03            # used for ablations

# Stepsizes for scaling experiment
STEPSIZES_SCALING = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]

# Final-window for computing converged dispersion
FINAL_WINDOW_FRACTION = 0.25
BURN_IN_FRACTION = 0.5

# Output (overridable via CLI)
OUTPUT_DIR = 'plots'
RESULTS_FILE = 'results.json'
PLOT_PREFIX = ''            # prepended to plot filenames (e.g. 'state_dep_')

# State-dependent dream generation (overridable via CLI)
STATE_DEPENDENT = False
STATE_DEPENDENT_STRENGTH = 1.0

# Symmetry-breaking importance reweighting (paper §5.4)
# When enabled, sample a larger candidate pool, compute per-candidate
# adapted-vs-base loss ratios, and resample by p ∝ (L_θ/L_W)^(1/τ).
# This upweights contexts the adapter currently performs poorly on.
SYMMETRY_BREAKING = False
SB_TEMPERATURE = 4.0
SB_OVERSAMPLE = 4

# Memory tightening (auto-on for tier 3; --tight-memory CLI flag forces on).
# When True, retain delta_W_history only for the seed-0 runs that the plots
# actually require: main α=ALPHA_MID (used as the dispersion / ablation
# reference) and B3 (used for the PCA trajectory plot). All other seed-0
# runs still compute their metric curves but skip full-history retention.
MEMORY_TIGHT = False

# Reproducibility
GLOBAL_SEED = 42


# ===========================================================================
# Tier specs — set via --tier CLI flag
# ===========================================================================

TIER_SPECS = {
    1: dict(D=8,   R=2, n_cycles=5000,  n_seeds=10),
    2: dict(D=32,  R=4, n_cycles=5000,  n_seeds=10),
    3: dict(D=128, R=8, n_cycles=10000, n_seeds=20),
}


def apply_tier(tier: int):
    """Mutate module globals so existing functions see tier-specific values."""
    global D, R, N_CYCLES, N_SEEDS
    spec = TIER_SPECS[tier]
    D = spec['D']
    R = spec['R']
    N_CYCLES = spec['n_cycles']
    N_SEEDS = spec['n_seeds']


# ===========================================================================
# Core update logic
# ===========================================================================

@dataclass
class Config:
    """Hyperparameter bundle for one experimental run."""
    alpha: float                              # stepsize
    batch_size: int = BATCH_SIZE_DEFAULT      # n dream samples per cycle
    trust_region: Optional[float] = TRUST_REGION_DEFAULT
    use_shadow: bool = True
    use_rank_constraint: bool = True
    dream_noise_scale: float = DREAM_NOISE_SCALE
    n_cycles: int = N_CYCLES
    occasional_large_input: bool = False      # for B2 ablation
    target_noise_scale: float = 0.05          # bounded fluctuation in target per cycle
    # New: state-dependent dream sampler
    state_dependent: bool = False
    state_dependent_strength: float = 1.0
    # New: symmetry-breaking importance reweighting (paper §5.4)
    symmetry_breaking: bool = False
    sb_temperature: float = 4.0
    sb_oversample: int = 4


def initialize_target(rng: np.random.Generator) -> np.ndarray:
    """Generate a fixed rank-R target deformation."""
    A_star = rng.normal(scale=0.3, size=(R, D))
    B_star = rng.normal(scale=0.3, size=(D, R))
    return B_star @ A_star  # (D, D), rank R


def initialize_factors(rng: np.random.Generator, full_rank: bool = False
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize adapter factors. Standard LoRA: A small random, B zero."""
    if full_rank:
        # B4 ablation: ignore rank constraint, use d x d directly
        A = rng.normal(scale=0.05, size=(D, D))
        B = np.eye(D)
        return A, B
    A = rng.normal(scale=0.05, size=(R, D))
    B = np.zeros((D, R))
    return A, B


def deformation(A: np.ndarray, B: np.ndarray, full_rank: bool = False
                ) -> np.ndarray:
    """Compute Delta_W = B @ A (or just A in full_rank case)."""
    if full_rank:
        return A
    return B @ A


def per_sample_gradients(A: np.ndarray, B: np.ndarray,
                         h_batch: np.ndarray, target: np.ndarray,
                         full_rank: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-form factor gradients for L_i = 0.5 * ||(BA - W*) h_i||^2.
    Returns mean gradient over the batch.
    """
    delta_W = deformation(A, B, full_rank)
    residual = delta_W - target
    n = h_batch.shape[0]

    if full_rank:
        H = h_batch.T @ h_batch / n
        grad_A = residual @ H
        grad_B = np.zeros_like(B)
        return grad_A, grad_B

    H = h_batch.T @ h_batch / n
    res_H = residual @ H
    grad_A = B.T @ res_H
    grad_B = res_H @ A.T
    return grad_A, grad_B


def operator_norm_clip(M: np.ndarray, max_norm: float) -> np.ndarray:
    s = np.linalg.norm(M, ord=2)
    if s > max_norm:
        return M * (max_norm / s)
    return M


def _draw_gaussian(rng: np.random.Generator,
                   cfg: Config,
                   delta_W_curr: np.ndarray,
                   scale_mult: float,
                   n_samples: int) -> np.ndarray:
    """Draw n_samples Gaussian dream candidates (fixed or state-dependent)."""
    sigma2 = cfg.dream_noise_scale ** 2
    gamma = cfg.state_dependent_strength

    if not cfg.state_dependent:
        return rng.normal(
            scale=scale_mult * cfg.dream_noise_scale,
            size=(n_samples, D),
        )

    # Adapter at LoRA init (B=0): structured component is zero — fall back
    # to isotropic. Skip the matmul as a small optimization.
    if not delta_W_curr.any():
        return rng.normal(
            scale=scale_mult * cfg.dream_noise_scale,
            size=(n_samples, D),
        )

    with np.errstate(invalid='ignore', over='ignore'):
        S = delta_W_curr.T @ delta_W_curr
    if not np.isfinite(S).all():
        raise FloatingPointError(
            'Non-finite S in state-dependent sampler — adapter has diverged '
            'before the run\'s divergence guard could catch it')
    Sigma = (scale_mult ** 2) * (sigma2 * np.eye(D) + gamma * S)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(D))
    z = rng.normal(size=(n_samples, D))
    return z @ L.T


def sample_dream_batch(rng: np.random.Generator,
                       cfg: Config,
                       delta_W_curr: np.ndarray,
                       scale_mult: float,
                       target: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Sample a batch of dream inputs.

    Three sampling regimes:

    - Fixed (state_dependent=False, symmetry_breaking=False):
        h_i ~ N(0, (scale_mult*sigma)^2 I).

    - State-dependent (state_dependent=True, symmetry_breaking=False):
        h_i ~ N(0, Sigma) with
          Sigma = scale_mult^2 * (sigma^2 I + gamma * Delta_W^T Delta_W).
        See paper §5.6 / state-dependent report.

    - Symmetry-breaking (symmetry_breaking=True, paper §5.4):
        Draw `batch_size * sb_oversample` candidates from the underlying
        Gaussian (state-dep or fixed), then resample `batch_size` of them
        with weights p_j ∝ (L_θ(h_j) / L_W(h_j))^(1/τ_sb). Here L_θ is
        the per-sample loss under the current adapter and L_W is the loss
        under the base model. In the toy regime, "base model" means
        ΔW = 0, so L_W(h) = 0.5 * ||W* h||^2 and
        L_θ(h) = 0.5 * ||(ΔW - W*) h||^2.

        This upweights candidates the *adapted* model is currently bad at
        relative to the *base* model — the audit mechanism. The min-max-ish
        structure (sampler favors adapter's weak spots; adapter then
        consolidates against those weak spots) is the structural condition
        the paper argues is needed for rotational dynamics, which shadow
        averaging is meant to damp.

    Cholesky may fail if Sigma is numerically singular. We retry with
    a small ridge.
    """
    if not cfg.symmetry_breaking:
        return _draw_gaussian(rng, cfg, delta_W_curr, scale_mult, cfg.batch_size)

    if target is None:
        raise ValueError('symmetry_breaking requires target to be passed')

    # Oversample candidates and reweight by L_θ/L_W.
    n_cand = cfg.batch_size * cfg.sb_oversample
    h_cand = _draw_gaussian(rng, cfg, delta_W_curr, scale_mult, n_cand)

    # Per-sample losses (closed form for the toy linear regression objective).
    # L_W(h) = 0.5 * ||W* h||^2
    base_proj = h_cand @ target.T
    L_W = 0.5 * (base_proj * base_proj).sum(axis=1)

    # L_θ(h) = 0.5 * ||(ΔW - W*) h||^2
    diff = delta_W_curr - target
    adapt_proj = h_cand @ diff.T
    L_theta = 0.5 * (adapt_proj * adapt_proj).sum(axis=1)

    # Tempered ratio (eq. 13). Small ε prevents div-by-zero for h-vectors
    # in W*'s null space (rare in random Gaussians but the paper specifies ε).
    eps = 1e-12
    ratio = (L_theta + eps) / (L_W + eps)
    # Guard against overflow when raising large ratios to 1/τ (τ>=1 so this
    # is a non-issue, but be defensive).
    log_w = np.log(ratio) / cfg.sb_temperature
    # Numerically stable softmax-like normalization.
    log_w -= log_w.max()
    w = np.exp(log_w)
    w_sum = w.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        # Pathological: fall through to uniform sampling.
        idx = rng.integers(0, n_cand, size=cfg.batch_size)
    else:
        p = w / w_sum
        idx = rng.choice(n_cand, size=cfg.batch_size, replace=True, p=p)
    return h_cand[idx]


def run_simulation(cfg: Config, target: np.ndarray, seed: int,
                   return_full_history: bool = False
                   ) -> dict:
    """
    Run one full simulation trajectory and return computed metrics.

    Both the live adapter and its Polyak-Ruppert shadow EMA are always
    computed (the shadow has no feedback into live updates, so this costs
    only one extra matmul per cycle and lets us report `shadow_dispersion`
    natively). The `cfg.use_shadow` flag is retained for backward
    compatibility but does not gate computation.
    """
    rng = np.random.default_rng(seed)
    full_rank = not cfg.use_rank_constraint

    A, B = initialize_factors(rng, full_rank)
    A_shadow = A.copy()
    B_shadow = B.copy() if not full_rank else np.eye(D)

    T = cfg.n_cycles + 1
    distance_curve = np.zeros(T)
    iterate_norm_curve = np.zeros(T)
    eff_rank_curve = np.zeros(T, dtype=int)
    shadow_distance_curve = np.zeros(T)  # ||live - shadow||

    final_window_size = int(T * FINAL_WINDOW_FRACTION)
    burn_in = int(T * BURN_IN_FRACTION)
    final_window_start = max(T - final_window_size, burn_in)
    steady_window_size = T - final_window_start
    steady_buffer = np.zeros((steady_window_size, D, D))
    steady_buffer_shadow = np.zeros((steady_window_size, D, D))

    full_history = [] if return_full_history else None
    full_shadow_history = [] if return_full_history else None

    def record(idx, delta_W, shadow_W):
        if not np.isfinite(delta_W).all():
            distance_curve[idx] = np.inf
            iterate_norm_curve[idx] = np.inf
            eff_rank_curve[idx] = -1
            shadow_distance_curve[idx] = np.inf
            return True
        diff = delta_W - target
        distance_curve[idx] = np.linalg.norm(diff)
        iterate_norm_curve[idx] = np.linalg.norm(delta_W)
        eff_rank_curve[idx] = effective_rank(delta_W)
        shadow_distance_curve[idx] = np.linalg.norm(delta_W - shadow_W)
        if idx >= final_window_start:
            steady_buffer[idx - final_window_start] = delta_W
            steady_buffer_shadow[idx - final_window_start] = shadow_W
        if return_full_history:
            full_history.append(delta_W.copy())
            full_shadow_history.append(shadow_W.copy())
        return False

    delta_W_init = deformation(A, B, full_rank)
    shadow_W_init = deformation(A_shadow, B_shadow, full_rank)
    record(0, delta_W_init, shadow_W_init)

    diverged = False
    diverged_at = None
    for t in range(cfg.n_cycles):
        if cfg.target_noise_scale > 0:
            target_perturbation = rng.normal(
                scale=cfg.target_noise_scale, size=target.shape)
            current_target = target + target_perturbation
        else:
            current_target = target

        # Numerical divergence safeguard for the no-trust-region ablation.
        # Must run BEFORE sampling, because under B2 + state-dependent dreams
        # an exploding ||A||, ||B|| produces a Sigma with such large eigenvalues
        # that Cholesky fails (matrix becomes numerically non-PD even with a
        # ridge regularizer). Catching divergence early prevents that crash.
        DIVERGENCE_CAP = 1e8
        if cfg.trust_region is None:
            if (np.linalg.norm(A) > DIVERGENCE_CAP or
                (not full_rank and np.linalg.norm(B) > DIVERGENCE_CAP)):
                diverged = True
                diverged_at = t
                distance_curve[t+1:] = np.inf
                iterate_norm_curve[t+1:] = np.inf
                eff_rank_curve[t+1:] = -1
                shadow_distance_curve[t+1:] = np.inf
                break

        # B2 large-input mechanism: amplify covariance every 50 cycles.
        if cfg.occasional_large_input and t > 0 and t % 50 == 0:
            scale_mult = 10.0
        else:
            scale_mult = 1.0

        # Sample dream batch (fixed or state-dependent).
        # State-dependent uses CURRENT delta_W (pre-update), which is the
        # source of the adapter↔dream coupling. If the sampler raises
        # LinAlgError despite the ridge, treat it as a divergence event —
        # this can happen at intermediate ||A|| values where Sigma is
        # numerically singular but A hasn't crossed DIVERGENCE_CAP yet.
        delta_W_curr = deformation(A, B, full_rank)
        try:
            h_batch = sample_dream_batch(
                rng, cfg, delta_W_curr, scale_mult, target=current_target)
        except (np.linalg.LinAlgError, FloatingPointError):
            diverged = True
            diverged_at = t
            distance_curve[t+1:] = np.inf
            iterate_norm_curve[t+1:] = np.inf
            eff_rank_curve[t+1:] = -1
            shadow_distance_curve[t+1:] = np.inf
            break

        grad_A, grad_B = per_sample_gradients(A, B, h_batch, current_target, full_rank)

        if cfg.trust_region is not None:
            grad_A = operator_norm_clip(grad_A, cfg.trust_region)
            if not full_rank:
                grad_B = operator_norm_clip(grad_B, cfg.trust_region)

        A = A - cfg.alpha * grad_A
        if not full_rank:
            B = B - cfg.alpha * grad_B

        A_shadow = (1 - SHADOW_BETA) * A_shadow + SHADOW_BETA * A
        if not full_rank:
            B_shadow = (1 - SHADOW_BETA) * B_shadow + SHADOW_BETA * B

        delta_W = deformation(A, B, full_rank)
        shadow_W = deformation(A_shadow, B_shadow, full_rank)
        if record(t + 1, delta_W, shadow_W):
            diverged = True
            diverged_at = t + 1
            distance_curve[t+2:] = np.inf
            iterate_norm_curve[t+2:] = np.inf
            eff_rank_curve[t+2:] = -1
            shadow_distance_curve[t+2:] = np.inf
            break

    if diverged:
        long_run_mean = np.full((D, D), np.nan)
        long_run_mean_shadow = np.full((D, D), np.nan)
        final_window_dispersion_value = float('inf')
        final_window_dispersion_shadow_value = float('inf')
        final_distance_value = float('inf')
        final_distance_shadow_value = float('inf')
        dispersion_curve = np.full(T, np.nan)
        dispersion_curve_shadow = np.full(T, np.nan)
    else:
        # Live final-window dispersion (around its own long-run mean).
        long_run_mean = steady_buffer.mean(axis=0)
        flat_steady = steady_buffer.reshape(steady_window_size, -1)
        final_mean_flat = long_run_mean.flatten()
        deviations = np.linalg.norm(flat_steady - final_mean_flat[None, :], axis=1)
        final_window_dispersion_value = float(deviations.mean())

        # Shadow final-window dispersion (around its own long-run mean).
        long_run_mean_shadow = steady_buffer_shadow.mean(axis=0)
        flat_steady_shadow = steady_buffer_shadow.reshape(steady_window_size, -1)
        final_mean_flat_shadow = long_run_mean_shadow.flatten()
        deviations_shadow = np.linalg.norm(
            flat_steady_shadow - final_mean_flat_shadow[None, :], axis=1)
        final_window_dispersion_shadow_value = float(deviations_shadow.mean())

        # Final-window mean distance to target for both readouts (averages
        # bias + variance into one number; useful as a "how well consolidated"
        # summary).
        final_distance_value = float(np.linalg.norm(
            steady_buffer - target[None, :, :], axis=(1, 2)).mean())
        final_distance_shadow_value = float(np.linalg.norm(
            steady_buffer_shadow - target[None, :, :], axis=(1, 2)).mean())

        if return_full_history:
            full_history_arr = np.array(full_history)
            full_shadow_arr = np.array(full_shadow_history)
            flat_all = full_history_arr.reshape(T, -1)
            flat_all_shadow = full_shadow_arr.reshape(T, -1)
            dispersion_curve = np.linalg.norm(
                flat_all - final_mean_flat[None, :], axis=1)
            dispersion_curve_shadow = np.linalg.norm(
                flat_all_shadow - final_mean_flat_shadow[None, :], axis=1)
        else:
            dispersion_curve = np.full(T, np.nan)
            dispersion_curve[final_window_start:] = deviations
            dispersion_curve_shadow = np.full(T, np.nan)
            dispersion_curve_shadow[final_window_start:] = deviations_shadow

    result = {
        'distance_to_target_curve': distance_curve,
        'dispersion_curve': dispersion_curve,
        'dispersion_curve_shadow': dispersion_curve_shadow,
        'iterate_norm_curve': iterate_norm_curve,
        'effective_rank_curve': eff_rank_curve,
        'shadow_distance_curve': shadow_distance_curve,
        'final_window_dispersion': final_window_dispersion_value,
        'final_window_dispersion_shadow': final_window_dispersion_shadow_value,
        'final_distance_to_target': final_distance_value,
        'final_distance_to_target_shadow': final_distance_shadow_value,
        'long_run_mean': long_run_mean,
        'long_run_mean_shadow': long_run_mean_shadow,
        'diverged': diverged,
        'diverged_at': diverged_at,
        'cfg': asdict(cfg),
    }

    if return_full_history:
        if diverged and len(full_history) < T:
            pad = T - len(full_history)
            nan_mat = np.full((D, D), np.nan)
            full_history.extend([nan_mat] * pad)
            full_shadow_history.extend([nan_mat] * pad)
        result['delta_W_history'] = np.array(full_history)
        result['shadow_history'] = np.array(full_shadow_history)

    return result


# ===========================================================================
# Metrics
# ===========================================================================

def effective_rank(M: np.ndarray, threshold_ratio: float = 0.01) -> int:
    if not np.isfinite(M).all():
        return -1
    s = np.linalg.svd(M, compute_uv=False)
    if s[0] < 1e-10:
        return 0
    return int(np.sum(s > threshold_ratio * s[0]))


# ===========================================================================
# Helper: build a Config with state-dependent fields if the global flag is set
# ===========================================================================

def make_cfg(**kwargs) -> Config:
    """Build Config, layering in state-dependent globals if set."""
    kwargs.setdefault('state_dependent', STATE_DEPENDENT)
    kwargs.setdefault('state_dependent_strength', STATE_DEPENDENT_STRENGTH)
    kwargs.setdefault('symmetry_breaking', SYMMETRY_BREAKING)
    kwargs.setdefault('sb_temperature', SB_TEMPERATURE)
    kwargs.setdefault('sb_oversample', SB_OVERSAMPLE)
    kwargs.setdefault('n_cycles', N_CYCLES)
    return Config(**kwargs)


# ===========================================================================
# Experimental runs
# ===========================================================================

def run_main(target: np.ndarray) -> dict:
    """Run A: main result with three stepsizes and multiple seeds."""
    results = {}
    for alpha in STEPSIZES_MAIN:
        cfg = make_cfg(alpha=alpha)
        seed_results = []
        for seed in range(N_SEEDS):
            keep_full = (seed == 0)
            if MEMORY_TIGHT and alpha != ALPHA_MID:
                keep_full = False
            res = run_simulation(
                cfg, target,
                seed=seed + 1000 * int(alpha * 1000),
                return_full_history=keep_full,
            )
            seed_results.append(res)
        results[alpha] = seed_results
        print(f'  main α={alpha} done', flush=True)
    return results


def run_scaling(target: np.ndarray) -> dict:
    """Run C: stepsize-dispersion scaling. No full-history needed."""
    results = {}
    for alpha in STEPSIZES_SCALING:
        cfg = make_cfg(alpha=alpha)
        seed_results = []
        for seed in range(N_SEEDS):
            res = run_simulation(
                cfg, target,
                seed=seed + 2000 * int(alpha * 1000),
                return_full_history=False,
            )
            seed_results.append(res)
        results[alpha] = seed_results
        print(f'  scaling α={alpha} done', flush=True)
    return results


def run_ablations(target: np.ndarray) -> dict:
    """
    Runs B1, B2, B4: each removes one safeguard.

    B3 (no shadow averaging) is no longer a separate config. Shadow is now
    always computed, so its effect on the *readout* is captured in every
    config's `final_window_dispersion_shadow` vs `final_window_dispersion`
    pair. The B3 effect at any α/regime is `live/shadow` of the corresponding
    Main row — no extra run needed.
    """
    seeds = list(range(N_SEEDS))
    cfgs = {
        'B1_insufficient_samples': make_cfg(alpha=ALPHA_MID, batch_size=4),
        'B2_no_trust_region': make_cfg(alpha=ALPHA_MID, trust_region=None,
                                        occasional_large_input=True),
        'B4_no_rank_constraint': make_cfg(alpha=ALPHA_MID, use_rank_constraint=False),
    }
    results = {}
    for name, cfg in cfgs.items():
        seed_results = []
        for seed in seeds:
            keep_full = (seed == 0)
            # Under memory pressure (tier 3), only retain full history for the
            # one ablation config whose plot needs it. Was B3 historically;
            # B4's effective-rank panel is the most informative ablation plot
            # now, and rank is recorded in the metric curves either way, so
            # no ablation config requires full history.
            if MEMORY_TIGHT:
                keep_full = False
            res = run_simulation(
                cfg, target,
                seed=seed + 3000,
                return_full_history=keep_full,
            )
            seed_results.append(res)
        results[name] = seed_results
        print(f'  ablation {name} done', flush=True)
    return results


# ===========================================================================
# Plotting
# ===========================================================================

def setup_plot_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def _plot_path(name: str) -> str:
    return f'{OUTPUT_DIR}/{PLOT_PREFIX}{name}'


def plot_main_trajectory(main_results: dict, target: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0.01: 'C0', 0.03: 'C1', 0.1: 'C2'}
    for alpha in STEPSIZES_MAIN:
        seed_results = main_results[alpha]
        curves = np.array([r['distance_to_target_curve'] for r in seed_results])
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        t = np.arange(len(mean))
        ax.plot(t, mean, color=colors[alpha], label=f'α = {alpha}', linewidth=1.5)
        ax.fill_between(t, mean - std, mean + std, alpha=0.2, color=colors[alpha])
    ax.set_xlabel('Consolidation cycle')
    ax.set_ylabel('||ΔW_t - W*||_F')
    ax.set_title('Distance to target deformation over time')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_plot_path('01_main_trajectory.png'), dpi=120)
    plt.close()


def plot_dispersion(main_results: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0.01: 'C0', 0.03: 'C1', 0.1: 'C2'}
    for alpha in STEPSIZES_MAIN:
        seed0 = main_results[alpha][0]
        dispersion = seed0['dispersion_curve']
        t = np.arange(len(dispersion))
        ax.plot(t, dispersion, color=colors[alpha], label=f'α = {alpha}',
                linewidth=1.0, alpha=0.7)
    ax.set_xlabel('Consolidation cycle')
    ax.set_ylabel('Iterate dispersion ||ΔW_t - long-run mean||_F')
    ax.set_title('Bounded-tracking dispersion signature')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_plot_path('02_dispersion.png'), dpi=120)
    plt.close()


def plot_scaling(scaling_results: dict):
    alphas = sorted(scaling_results.keys())
    dispersions, dispersion_stds = [], []
    shadow_dispersions, shadow_stds = [], []
    for alpha in alphas:
        seed_results = scaling_results[alpha]
        live = [r['final_window_dispersion'] for r in seed_results
                if np.isfinite(r['final_window_dispersion'])]
        sh = [r['final_window_dispersion_shadow'] for r in seed_results
              if np.isfinite(r['final_window_dispersion_shadow'])]
        dispersions.append(np.mean(live) if live else np.nan)
        dispersion_stds.append(np.std(live) if live else np.nan)
        shadow_dispersions.append(np.mean(sh) if sh else np.nan)
        shadow_stds.append(np.std(sh) if sh else np.nan)
    dispersions = np.array(dispersions)
    dispersion_stds = np.array(dispersion_stds)
    shadow_dispersions = np.array(shadow_dispersions)
    shadow_stds = np.array(shadow_stds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(alphas, dispersions, yerr=dispersion_stds, fmt='o-',
                label='Live readout', linewidth=1.5, capsize=4)
    ax.errorbar(alphas, shadow_dispersions, yerr=shadow_stds, fmt='s-',
                label='Shadow readout', linewidth=1.5, capsize=4, color='C2')

    finite_mask = np.isfinite(dispersions)
    if finite_mask.any():
        finite_idx = np.where(finite_mask)[0]
        mid_idx = finite_idx[len(finite_idx) // 2]
        C = dispersions[mid_idx] / np.sqrt(alphas[mid_idx])
        alpha_fine = np.linspace(min(alphas), max(alphas), 50)
        ax.plot(alpha_fine, C * np.sqrt(alpha_fine), '--',
                color='gray', label=f'$C\\sqrt{{α}}$ reference, live (C={C:.3f})')

    ax.set_xlabel('Stepsize α')
    ax.set_ylabel('Final-window dispersion radius')
    ax.set_title('Stepsize–dispersion scaling (live and shadow readouts)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(_plot_path('03_scaling.png'), dpi=120)
    plt.close()
    return alphas, dispersions, dispersion_stds, shadow_dispersions, shadow_stds


def plot_ablations(main_results: dict, ablation_results: dict, target: np.ndarray):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    main_seed0 = main_results[ALPHA_MID][0]
    main_dispersion = main_seed0['dispersion_curve']

    # B1
    ax = axes[0, 0]
    b1_seed0 = ablation_results['B1_insufficient_samples'][0]
    b1_dispersion = b1_seed0['dispersion_curve']
    t = np.arange(len(main_dispersion))
    ax.plot(t, main_dispersion, label='Main (n=32)', linewidth=1.0)
    ax.plot(t, b1_dispersion, label='B1 (n=4)', linewidth=1.0)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Dispersion')
    ax.set_title('B1: Insufficient dream samples')
    ax.legend()
    ax.grid(alpha=0.3)

    # B2
    ax = axes[0, 1]
    b2_seed0 = ablation_results['B2_no_trust_region'][0]
    main_norms = main_seed0['iterate_norm_curve']
    b2_norms = b2_seed0['iterate_norm_curve']
    ax.plot(main_norms, label='Main (with trust region)', linewidth=1.0)
    ax.plot(b2_norms, label='B2 (no trust region)', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('||ΔW_t||_F')
    ax.set_title('B2: No trust region (large inputs every 50 cycles)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Shadow vs live (replaces former B3 PCA panel; B3 ablation deprecated)
    ax = axes[1, 0]
    main_disp_live = main_seed0['dispersion_curve']
    main_disp_shadow = main_seed0['dispersion_curve_shadow']
    finite = np.isfinite(main_disp_live) & np.isfinite(main_disp_shadow)
    t = np.arange(len(main_disp_live))
    ax.plot(t, main_disp_live, label='Live readout', linewidth=1.2)
    ax.plot(t, main_disp_shadow, label='Shadow readout', linewidth=1.2)
    # Report seed-averaged ratio in the title, not just seed-0
    main_seeds = main_results[ALPHA_MID]
    live_disps = [r['final_window_dispersion'] for r in main_seeds
                  if np.isfinite(r['final_window_dispersion'])]
    shadow_disps = [r['final_window_dispersion_shadow'] for r in main_seeds
                    if np.isfinite(r['final_window_dispersion_shadow'])]
    if shadow_disps:
        ratio = float(np.mean(live_disps) / np.mean(shadow_disps))
        title = f'Shadow readout dispersion (live/shadow = {ratio:.3f} avg across seeds)'
    else:
        title = 'Shadow readout dispersion'
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Dispersion (around own long-run mean)')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    # B4
    ax = axes[1, 1]
    b4_seed0 = ablation_results['B4_no_rank_constraint'][0]
    main_eff_ranks = main_seed0['effective_rank_curve']
    b4_eff_ranks = b4_seed0['effective_rank_curve']
    ax.plot(main_eff_ranks, label='Main (rank-r constraint)', linewidth=1.5)
    ax.plot(b4_eff_ranks, label='B4 (no rank constraint)', linewidth=1.0, alpha=0.7)
    ax.axhline(y=R, color='gray', linestyle='--', alpha=0.5, label=f'r = {R}')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Effective rank of ΔW')
    ax.set_title('B4: No rank constraint')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(_plot_path('04_ablations.png'), dpi=120)
    plt.close()


def plot_shadow_smoothing(main_results: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    for alpha in STEPSIZES_MAIN:
        curves = np.array([r['shadow_distance_curve'] for r in main_results[alpha]])
        mean = curves.mean(axis=0)
        ax.plot(mean, label=f'α = {alpha}', linewidth=1.5)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('||live - shadow||_F')
    ax.set_title('Shadow adapter smoothing of live trajectory')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_plot_path('05_shadow_smoothing.png'), dpi=120)
    plt.close()


# ===========================================================================
# Smoke test entry
# ===========================================================================

def smoke_test():
    """Single-seed run at d=8, α=0.03, state_dep on. Sanity check the sampler."""
    apply_tier(1)
    rng = np.random.default_rng(GLOBAL_SEED)
    target = initialize_target(rng)
    cfg = Config(alpha=0.03, n_cycles=2000,
                 state_dependent=True, state_dependent_strength=1.0)

    print(f'Smoke test: d={D}, r={R}, α={cfg.alpha}, '
          f'state_dependent={cfg.state_dependent}, γ={cfg.state_dependent_strength}',
          flush=True)
    t0 = time.time()
    res = run_simulation(cfg, target, seed=42, return_full_history=True)
    dt = time.time() - t0

    diverged = res['diverged']
    fin_disp = res['final_window_dispersion']
    final_dist = res['distance_to_target_curve'][-1]
    final_norm = res['iterate_norm_curve'][-1]
    eff_rank_final = int(res['effective_rank_curve'][-1])

    # Verify Cholesky-based sampler covariance with a spot check
    # (handoff §"Failure modes" called out N(0, L^T L) vs N(0, L L^T) as a
    # likely sampler bug — this directly checks E[h h^T] ≈ Sigma).
    rng_check = np.random.default_rng(0)
    A_check, B_check = initialize_factors(rng_check, full_rank=False)
    delta_W_curr = deformation(A_check, B_check, full_rank=False)
    # Stack ~10k samples for a low-variance covariance estimate.
    h_big = np.concatenate([
        sample_dream_batch(rng_check, cfg, delta_W_curr, 1.0)
        for _ in range(312)
    ], axis=0)
    emp_cov = h_big.T @ h_big / h_big.shape[0]
    sigma2 = cfg.dream_noise_scale ** 2
    Sigma_target = sigma2 * np.eye(D) + cfg.state_dependent_strength * (
        delta_W_curr.T @ delta_W_curr)
    cov_err = np.linalg.norm(emp_cov - Sigma_target) / np.linalg.norm(Sigma_target)

    print(f'\n--- smoke test results ---')
    print(f'wallclock: {dt:.2f}s')
    print(f'diverged: {diverged}')
    print(f'final_window_dispersion (live):   {fin_disp:.4f}')
    print(f'final_window_dispersion (shadow): {res["final_window_dispersion_shadow"]:.4f}')
    print(f'live/shadow ratio:                '
          f'{fin_disp / max(res["final_window_dispersion_shadow"], 1e-12):.3f}')
    print(f'final ||ΔW - W*||_F: {final_dist:.4f}')
    print(f'final ||ΔW||_F: {final_norm:.4f}')
    print(f'final effective rank: {eff_rank_final} (target rank R={R})')
    print(f'sampler covariance relative error: {cov_err:.4f}')
    print('--- end smoke test ---')


# ===========================================================================
# Main entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Dreaming LoRA bounded-tracking simulation.')
    parser.add_argument('--tier', type=int, choices=[1, 2, 3], default=2,
                        help='Tier preset: 1=d8/r2, 2=d32/r4, 3=d128/r8 (longer/more seeds)')
    parser.add_argument('--state-dep', action='store_true',
                        help='Enable state-dependent dream sampler')
    parser.add_argument('--strength', type=float, default=1.0,
                        help='State-dependent coupling strength γ (default 1.0)')
    parser.add_argument('--symmetry-breaking', action='store_true',
                        help='Enable §5.4 symmetry-breaking importance reweighting')
    parser.add_argument('--sb-tau', type=float, default=4.0,
                        help='Symmetry-breaking temperature τ_sb (default 4.0)')
    parser.add_argument('--sb-oversample', type=int, default=4,
                        help='Symmetry-breaking candidate oversampling factor (default 4)')
    parser.add_argument('--out-dir', type=str, default='plots',
                        help='Directory for plots and results.json')
    parser.add_argument('--results-name', type=str, default='results.json',
                        help='Filename for JSON summary')
    parser.add_argument('--plot-prefix', type=str, default='',
                        help='Prefix prepended to plot filenames')
    parser.add_argument('--tight-memory', action='store_true',
                        help='Retain full delta_W history only for the seed-0 '
                             'runs that the plots require (main α=ALPHA_MID and '
                             'B3). Auto-enabled for --tier 3.')
    parser.add_argument('--smoke', action='store_true',
                        help='Run smoke test only and exit')
    args = parser.parse_args()

    if args.smoke:
        # Override globals for the smoke test path.
        global STATE_DEPENDENT, STATE_DEPENDENT_STRENGTH
        global SYMMETRY_BREAKING, SB_TEMPERATURE, SB_OVERSAMPLE
        STATE_DEPENDENT = True
        STATE_DEPENDENT_STRENGTH = args.strength
        SYMMETRY_BREAKING = args.symmetry_breaking
        SB_TEMPERATURE = args.sb_tau
        SB_OVERSAMPLE = args.sb_oversample
        smoke_test()
        return

    # Apply tier and CLI globals.
    apply_tier(args.tier)
    global OUTPUT_DIR, RESULTS_FILE, PLOT_PREFIX, MEMORY_TIGHT
    OUTPUT_DIR = args.out_dir
    RESULTS_FILE = os.path.join(args.out_dir, args.results_name)
    PLOT_PREFIX = args.plot_prefix
    STATE_DEPENDENT = args.state_dep
    STATE_DEPENDENT_STRENGTH = args.strength
    SYMMETRY_BREAKING = args.symmetry_breaking
    SB_TEMPERATURE = args.sb_tau
    SB_OVERSAMPLE = args.sb_oversample
    MEMORY_TIGHT = args.tight_memory or (args.tier == 3)

    np.random.seed(GLOBAL_SEED)
    setup_plot_dir()

    # Generate fixed target
    rng = np.random.default_rng(GLOBAL_SEED)
    target = initialize_target(rng)
    print(f'Tier {args.tier}: D={D}, R={R}, n_cycles={N_CYCLES}, n_seeds={N_SEEDS}',
          flush=True)
    print(f'State-dependent={STATE_DEPENDENT}, strength={STATE_DEPENDENT_STRENGTH}',
          flush=True)
    if SYMMETRY_BREAKING:
        print(f'Symmetry-breaking ON: τ_sb={SB_TEMPERATURE}, '
              f'oversample={SB_OVERSAMPLE}', flush=True)
    print(f'Output dir: {OUTPUT_DIR}, results file: {RESULTS_FILE}', flush=True)
    print(f'Target deformation: ||W*||_F={np.linalg.norm(target):.4f}, '
          f'effective rank={effective_rank(target)}', flush=True)

    t_total = time.time()

    print('\nRunning Run A (main result)...', flush=True)
    main_results = run_main(target)

    print('\nRunning Run C (stepsize scaling)...', flush=True)
    scaling_results = run_scaling(target)

    print('\nRunning ablations B1-B4...', flush=True)
    ablation_results = run_ablations(target)

    print(f'\nAll runs complete in {time.time() - t_total:.1f}s. Generating plots...',
          flush=True)
    plot_main_trajectory(main_results, target)
    plot_dispersion(main_results)
    alphas, dispersions, dispersion_stds, shadow_disps, shadow_disps_std = plot_scaling(scaling_results)
    plot_ablations(main_results, ablation_results, target)
    plot_shadow_smoothing(main_results)

    def _summarize_seed_results(seed_results):
        """Per-config summary of live + shadow dispersion and final distance."""
        def _aggregate(key):
            vals = [r[key] for r in seed_results if np.isfinite(r[key])]
            if vals:
                return float(np.mean(vals)), float(np.std(vals))
            return float('inf'), float('nan')

        n_diverged = sum(1 for r in seed_results if r['diverged'])
        live_mean, live_std = _aggregate('final_window_dispersion')
        shadow_mean, shadow_std = _aggregate('final_window_dispersion_shadow')
        dist_live_mean, _ = _aggregate('final_distance_to_target')
        dist_shadow_mean, _ = _aggregate('final_distance_to_target_shadow')

        # live/shadow dispersion ratio (the new B3-equivalent metric)
        if np.isfinite(live_mean) and np.isfinite(shadow_mean) and shadow_mean > 0:
            live_over_shadow = live_mean / shadow_mean
        else:
            live_over_shadow = float('nan')

        return {
            'n_seeds_diverged': n_diverged,
            'n_seeds_total': len(seed_results),
            'live_dispersion_mean': live_mean,
            'live_dispersion_std': live_std,
            'shadow_dispersion_mean': shadow_mean,
            'shadow_dispersion_std': shadow_std,
            'live_over_shadow_ratio': live_over_shadow,
            'final_distance_live_mean': dist_live_mean,
            'final_distance_shadow_mean': dist_shadow_mean,
        }

    summary = {
        'config': {
            'tier': args.tier,
            'D': D, 'R': R,
            'batch_size': BATCH_SIZE_DEFAULT,
            'n_cycles': N_CYCLES,
            'n_seeds': N_SEEDS,
            'shadow_beta': SHADOW_BETA,
            'trust_region_default': TRUST_REGION_DEFAULT,
            'state_dependent': STATE_DEPENDENT,
            'state_dependent_strength': STATE_DEPENDENT_STRENGTH,
            'symmetry_breaking': SYMMETRY_BREAKING,
            'sb_temperature': SB_TEMPERATURE,
            'sb_oversample': SB_OVERSAMPLE,
        },
        'main': {
            str(alpha): _summarize_seed_results(main_results[alpha])
            for alpha in STEPSIZES_MAIN
        },
        'scaling_dispersion_vs_alpha': {
            str(alphas[i]): {
                'live_mean': float(dispersions[i]),
                'live_std': float(dispersion_stds[i]),
                'shadow_mean': float(shadow_disps[i]),
                'shadow_std': float(shadow_disps_std[i]),
            }
            for i in range(len(alphas))
        },
        'ablations': {
            name: _summarize_seed_results(ablation_results[name])
            for name in ablation_results
        },
    }

    # Compute proportionality constant C from scaling data, for both readouts.
    finite_alphas = [alphas[i] for i in range(len(alphas))
                     if np.isfinite(dispersions[i])]
    finite_disps = [dispersions[i] for i in range(len(alphas))
                    if np.isfinite(dispersions[i])]
    finite_shadow_disps = [shadow_disps[i] for i in range(len(alphas))
                           if np.isfinite(shadow_disps[i])]
    if finite_alphas:
        Cs_live = [d / np.sqrt(a) for a, d in zip(finite_alphas, finite_disps)]
        summary['scaling_C_estimates_live'] = {
            str(a): float(c) for a, c in zip(finite_alphas, Cs_live)
        }
        summary['scaling_C_mean_live'] = float(np.mean(Cs_live))
        if len(finite_shadow_disps) == len(finite_alphas):
            Cs_sh = [d / np.sqrt(a) for a, d
                     in zip(finite_alphas, finite_shadow_disps)]
            summary['scaling_C_estimates_shadow'] = {
                str(a): float(c) for a, c in zip(finite_alphas, Cs_sh)
            }
            summary['scaling_C_mean_shadow'] = float(np.mean(Cs_sh))

    with open(RESULTS_FILE, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print('\n=== Summary ===', flush=True)
    print(json.dumps(summary, indent=2, default=str))
    print(f'\nPlots saved to {OUTPUT_DIR}/', flush=True)
    print(f'Numerical results saved to {RESULTS_FILE}', flush=True)


if __name__ == '__main__':
    main()
