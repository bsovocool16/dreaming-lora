"""
Track 2, Experiment A — Coherence loss in a linear content/context regime
==========================================================================

Tests §3.4's claim in the simplest regime that can express it: "content"
has identity across "contexts," and we ask whether the Fisher-discriminant
coherence term is load-bearing for cross-context representational
consistency, or whether reconstruction alone achieves it given diverse
training contexts (the issues-tracker Option B retreat).

Regime
------
  contents:   c_k in R^d, k = 1..K, unit norm
  contexts:   T_j = I + sigma_T * G_j   (G_j iid N(0, 1/d) entries)
  observation: h = T_j c_k + eps,  eps ~ N(0, sigma_n^2 I)
  adapter:    z = h + B A h            (LoRA-style residual, rank r, B init 0)
  base task:  y = omega * P c_k + (1 - omega) * Q T_j c_k
              predicted by frozen random readout R applied to z.
              omega controls how content-determined the prediction target is;
              the (1-omega) term is the surface/context-bound component that
              real next-token prediction always has.

Losses
------
  L_recon  = mean ||R z - y||^2
  L_cohere = Fisher-discriminant on L2-normalized z grouped by content
             (within-content dispersion / between-content variance), as in
             paper Eq. (6).
  composite: (1 - lam) * L_recon + lam * L_cohere

Evaluation — always on HELD-OUT contexts T_j' never seen in training
---------------------------------------------------------------------
  fisher_holdout:    same Fisher ratio computed on held-out contexts
                     (lower = more coherent). Reported for raw h as floor.
  retrieval_acc:     nearest-centroid content classification: centroids from
                     TRAIN contexts, queries from HELD-OUT contexts. The
                     functional consequence of coherence (recognizing same
                     content as same in a novel setting).
  recon_holdout:     reconstruction loss on held-out contexts (control:
                     the comparison is only meaningful at similar recon).

Pre-registered predictions (written before first full run)
-----------------------------------------------------------
  P-A1 (paper's claim): recon-only training leaves fisher_holdout near the
       raw-h floor and retrieval_acc near its raw-h value; composite
       improves both substantially at <10% recon_holdout penalty for
       moderate lam.
  P-A2 (Option B alternative): as n_ctx_train grows, recon-only closes the
       gap. If at n_ctx_train = 32 the recon-only/composite gap in
       retrieval_acc is < 5 points, the coherence term is NOT load-bearing
       in this regime and the result supports Option B here.
  P-A3: the effect of omega is monotone: the more content-determined the
       task (high omega), the more coherence recon-only inherits for free.
       At omega = 1 the task itself demands content abstraction and the
       coherence term should be nearly redundant; at low omega the gap
       should be largest. (This parameterizes WHEN coherence is needed,
       which is more useful than a binary answer.)

Decisions committed here rather than tuned post hoc:
  - adapter updated by plain gradient descent, fixed alpha, fixed steps,
    no early stopping (no peeking at eval).
  - lam swept over a small grid; we report the full frontier, not a
    cherry-picked point.
  - 5 seeds; mean +/- sd reported for every cell.
"""

import json
import os
import sys
import numpy as np

# ---------------------------------------------------------------- config
D = 32                 # ambient dim
R_RANK = 4             # adapter rank
K = 8                  # number of contents
SIGMA_T = 8.0          # context distortion strength (in the shared subspace)
CTX_DECAY = 0.75       # per-direction decay of distortion energy: rank-4
                       # adapter can cancel most (calibrated: raw retrieval
                       # ~0.61, rank-4 oracle ~0.95, rank-8 oracle 1.0)
M_CTX = 8              # dimension of the shared context-distortion subspace
DELTA_MAX = 0.5        # trust region: max operator norm of per-step update
SIGMA_N = 0.05         # observation noise
D_OUT = 16             # readout dim
ALPHA = 0.01           # GD stepsize
N_STEPS = 4000         # GD steps
BATCH = 64             # samples per step
N_CTX_HOLDOUT = 16     # held-out contexts (fixed)
EPS = 1e-6

N_CTX_TRAIN_GRID = [2, 4, 8, 16, 32]
LAM_GRID = [0.0, 0.1, 0.3, 0.5, 0.7]   # 0.0 == recon-only
OMEGA_GRID = [0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2, 3, 4]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_a")


def make_world(rng):
    c = rng.normal(size=(K, D))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    P = rng.normal(size=(D_OUT, D)) / np.sqrt(D)
    Q = rng.normal(size=(D_OUT, D)) / np.sqrt(D)
    Rd = rng.normal(size=(D_OUT, D)) / np.sqrt(D)   # frozen readout
    # shared context-distortion subspace: all contexts distort along the same
    # m directions (with context-specific strengths). This is what makes
    # cross-context coherence ACHIEVABLE for a rank-r linear adapter: the
    # adapter can learn to suppress the shared subspace. Unstructured
    # per-context distortion would be linearly unrecoverable and the
    # experiment would test nothing.
    Uc, _ = np.linalg.qr(rng.normal(size=(D, M_CTX)))
    Vc, _ = np.linalg.qr(rng.normal(size=(D, M_CTX)))
    return c, P, Q, Rd, Uc, Vc


def make_contexts(rng, n, Uc, Vc):
    w = CTX_DECAY ** np.arange(M_CTX)
    s = SIGMA_T * rng.normal(size=(n, M_CTX)) * w
    return np.eye(D)[None, :, :] + np.einsum("dm,nm,em->nde", Uc, s, Vc)


def sample_batch(rng, c, Ts, P, Q, omega, n):
    # balanced over contents so every Fisher group is populated
    ks = np.tile(np.arange(K), n // K + 1)[:n]
    rng.shuffle(ks)
    js = rng.integers(0, len(Ts), size=n)
    base = c[ks]                                   # (n, D)
    h = np.einsum("njk,nk->nj", Ts[js], base)
    h += SIGMA_N * rng.normal(size=h.shape)
    y = omega * base @ P.T + (1 - omega) * h @ Q.T
    return h, y, ks


def fisher_ratio(z, ks):
    """Within-content dispersion / between-content variance on normalized z."""
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + EPS)
    mus = np.stack([zn[ks == k].mean(axis=0) for k in range(K)])
    within = np.mean([np.mean(np.sum((zn[ks == k] - mus[k]) ** 2, axis=1))
                      for k in range(K)])
    between = np.mean(np.sum((mus - mus.mean(axis=0)) ** 2, axis=1))
    return within / (between + EPS)


def retrieval_acc(z_train, ks_train, z_hold, ks_hold):
    zn_tr = z_train / (np.linalg.norm(z_train, axis=1, keepdims=True) + EPS)
    zn_ho = z_hold / (np.linalg.norm(z_hold, axis=1, keepdims=True) + EPS)
    cents = np.stack([zn_tr[ks_train == k].mean(axis=0) for k in range(K)])
    cents /= (np.linalg.norm(cents, axis=1, keepdims=True) + EPS)
    pred = np.argmax(zn_ho @ cents.T, axis=1)
    return float(np.mean(pred == ks_hold))


def grads(A, B, h, y, ks, Rd, lam):
    """Analytic gradients of composite loss wrt LoRA factors A, B."""
    n = h.shape[0]
    z = h + h @ A.T @ B.T                          # (n, D)
    # --- recon
    err = z @ Rd.T - y                             # (n, D_OUT)
    dz_recon = 2.0 * err @ Rd / n                  # dL_recon/dz
    L_recon = float(np.mean(np.sum(err ** 2, axis=1)))
    # --- coherence (Fisher) on normalized z; gradient through normalization
    nz = np.linalg.norm(z, axis=1, keepdims=True) + EPS
    zn = z / nz
    mus = np.stack([zn[ks == k].mean(axis=0) for k in range(K)])
    mu_bar = mus.mean(axis=0)
    between = np.mean(np.sum((mus - mu_bar) ** 2, axis=1)) + EPS
    diffs = zn - mus[ks]                           # (n, D)
    within = float(np.mean(np.sum(diffs ** 2, axis=1)))
    L_coh = within / between
    # treat centroids and 'between' as slowly-varying (stop-grad), the
    # standard simplification; gradient pushes each zn toward its centroid
    dzn = (2.0 / (n * between)) * diffs
    # back through normalization: d zn / d z = (I - zn zn^T)/||z||
    dz_coh = (dzn - zn * np.sum(dzn * zn, axis=1, keepdims=True)) / nz
    dz = (1 - lam) * dz_recon + lam * dz_coh
    # back into factors
    gB = dz.T @ (h @ A.T)                          # (D, r)
    gA = B.T @ dz.T @ h                            # (r, D)
    return gA, gB, L_recon, L_coh


def run_cell(seed, n_ctx_train, lam, omega):
    # world and data are seeded WITHOUT lam, so all lam values in a cell see
    # the identical world, contexts, and training stream: paired comparison.
    world_key = 1_000_003 * seed + 1_009 * n_ctx_train + int(omega * 100)
    rng_world = np.random.default_rng(world_key)
    c, P, Q, Rd, Uc, Vc = make_world(rng_world)
    Ts_train = make_contexts(rng_world, n_ctx_train, Uc, Vc)
    Ts_hold = make_contexts(rng_world, N_CTX_HOLDOUT, Uc, Vc)
    rng = np.random.default_rng(world_key + 7)

    A = rng.normal(size=(R_RANK, D)) / np.sqrt(D)
    B = np.zeros((D, R_RANK))

    for _ in range(N_STEPS):
        h, y, ks = sample_batch(rng, c, Ts_train, P, Q, omega, BATCH)
        gA, gB, _, _ = grads(A, B, h, y, ks, Rd, lam)
        # norm-bounded trust region on the effective update (paper SS2.4)
        dW = ALPHA * (B @ gA + gB @ A)
        op = np.linalg.norm(dW, 2)
        scale = min(1.0, DELTA_MAX / (op + EPS))
        A -= ALPHA * scale * gA
        B -= ALPHA * scale * gB

    # ---- evaluation
    h_tr, y_tr, ks_tr = sample_batch(rng, c, Ts_train, P, Q, omega, 2048)
    h_ho, y_ho, ks_ho = sample_batch(rng, c, Ts_hold, P, Q, omega, 2048)
    z_tr = h_tr + h_tr @ A.T @ B.T
    z_ho = h_ho + h_ho @ A.T @ B.T
    err_ho = z_ho @ Rd.T - y_ho
    # rank-4 oracle reference: project out the top-4 shared context directions
    P4 = np.eye(D) - Uc[:, :4] @ Uc[:, :4].T
    return {
        "retrieval_acc_oracle_r4": retrieval_acc(h_tr @ P4.T, ks_tr, h_ho @ P4.T, ks_ho),
        "fisher_holdout": fisher_ratio(z_ho, ks_ho),
        "fisher_holdout_raw_h": fisher_ratio(h_ho, ks_ho),
        "retrieval_acc": retrieval_acc(z_tr, ks_tr, z_ho, ks_ho),
        "retrieval_acc_raw_h": retrieval_acc(h_tr, ks_tr, h_ho, ks_ho),
        "recon_holdout": float(np.mean(np.sum(err_ho ** 2, axis=1))),
        "adapter_norm": float(np.linalg.norm(B @ A)),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    total = len(OMEGA_GRID) * len(N_CTX_TRAIN_GRID) * len(LAM_GRID) * len(SEEDS)
    done = 0
    for omega in OMEGA_GRID:
        for n_ctx in N_CTX_TRAIN_GRID:
            for lam in LAM_GRID:
                for seed in SEEDS:
                    m = run_cell(seed, n_ctx, lam, omega)
                    m.update(omega=omega, n_ctx_train=n_ctx, lam=lam, seed=seed)
                    results.append(m)
                    done += 1
                if done % 25 == 0 or done == total:
                    print(f"[{done}/{total}] omega={omega} n_ctx={n_ctx} lam={lam} "
                          f"retr={np.mean([r['retrieval_acc'] for r in results[-len(SEEDS):]]):.3f}",
                          flush=True)
    with open(os.path.join(OUT_DIR, "results_a.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", os.path.join(OUT_DIR, "results_a.json"))


if __name__ == "__main__":
    main()
