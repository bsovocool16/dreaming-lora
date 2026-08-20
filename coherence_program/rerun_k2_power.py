"""
Powered rerun of the k=2 functional cells, addressing reviewer findings:

  C2: 15 seeds (0-14), lam fixed in advance at {0.0, 0.3} — no max-over-lam
      selection. Paired per-seed gaps + a paired t-test.
  M1: adds a loss-independent representational probe: ridge classifier on
      UN-normalized name-token states, trained on dream-family states,
      tested on eval-family states (linear separation geometry, not the
      cosine-centroid geometry the Fisher loss optimizes).
  M3: adds a shuffled-label coherence arm at lam=0.3: identical auxiliary
      loss magnitude and identical (1-lam) LM downweighting, but entity
      labels are randomly permuted per batch, destroying the entity
      structure. If shuffled-coherence reproduces part of the recon-vs-
      composite gap, that part is generic-regularization, not coherence.

Run with K_BASE_FAMS=2. Writes results_b/results_b_k2_power.json.
"""

import json
import os
import time

import numpy as np
import torch

import exp_b_transformer as eb

SEEDS = list(range(15))
ARMS = [("recon", 0.0, False), ("cohere", 0.3, False), ("shuffled", 0.3, True)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results_b", "results_b_k2_power.json")


def ridge_probe(z_tr, k_tr, z_ho, k_ho, reg=1.0):
    """Linear probe on un-normalized states: ridge regression to one-hot."""
    ents = sorted(set(k_tr.tolist()))
    Y = np.stack([(k_tr == e).astype(float) for e in ents], axis=1)
    X = np.concatenate([z_tr, np.ones((len(z_tr), 1))], axis=1)
    W = np.linalg.solve(X.T @ X + reg * np.eye(X.shape[1]), X.T @ Y)
    Xh = np.concatenate([z_ho, np.ones((len(z_ho), 1))], axis=1)
    pred = np.array(ents)[np.argmax(Xh @ W, axis=1)]
    return float(np.mean(pred == k_ho))


def consolidate(model, world, seed, n_dream, lam, shuffle):
    torch.manual_seed(1000 + seed)
    import math
    for b in model.blocks:
        b.lora["qa"].data = torch.randn(eb.LORA_RANK, eb.D_MODEL, device=eb.DEV) / math.sqrt(eb.D_MODEL)
        b.lora["va"].data = torch.randn(eb.LORA_RANK, eb.D_MODEL, device=eb.DEV) / math.sqrt(eb.D_MODEL)
        b.lora["qb"].data.zero_()
        b.lora["vb"].data.zero_()
    model.set_lora(True)
    new_ents = list(range(eb.N_BASE, eb.N_BASE + eb.N_NEW))
    dream_fams = eb.DREAM_FAM_POOL[:n_dream]
    rng = np.random.default_rng(777 + seed)          # same stream across arms
    shuffle_rng = np.random.default_rng(31337 + seed)
    opt = torch.optim.AdamW(model.lora_params(), lr=eb.CONSOL_LR)
    for step in range(eb.CONSOL_STEPS):
        idx, meta = world.batch(new_ents, dream_fams, eb.CONSOL_BATCH, rng)
        logits, h = model(idx, return_hidden=True)
        L_rec = eb.lm_loss(logits, idx)
        ents = np.array([m[0] for m in meta])
        if shuffle:
            ents = shuffle_rng.permutation(ents)
        L_coh = eb.fisher_coherence(h[:, eb.POS_NAME_B], ents)
        loss = (1 - lam) * L_rec + lam * L_coh
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lora_params(), 1.0)
        opt.step()

    base_ents = list(range(eb.N_BASE))
    res = {
        "attr_eval_acc": eb.attr_acc(model, world, new_ents, eb.EVAL_FAMS),
        "attr_dream_acc": eb.attr_acc(model, world, new_ents, dream_fams),
        "base_attr_acc": eb.attr_acc(model, world, base_ents, list(range(eb.N_FAM))),
    }
    with torch.no_grad():
        idx, _ = world.batch(new_ents, dream_fams, 512, np.random.default_rng(99))
        res["lm_dream"] = float(eb.lm_loss(model(idx), idx).item())
    z_tr, k_tr = eb.name_reps(model, world, new_ents, dream_fams)
    z_ho, k_ho = eb.name_reps(model, world, new_ents, eb.EVAL_FAMS)
    res["fisher_eval"] = eb.np_fisher(z_ho, k_ho)
    res["retrieval_eval"] = eb.np_retrieval(z_tr, k_tr, z_ho, k_ho)
    res["probe_eval"] = ridge_probe(z_tr, k_tr, z_ho, k_ho)
    model.set_lora(False)
    return res


def main():
    assert eb.K_BASE_FAMS == 2, "run with K_BASE_FAMS=2"
    all_results = []
    for seed in SEEDS:
        t0 = time.time()
        world = eb.World(seed)
        model = eb.pretrain(world, seed)
        base_acc = eb.attr_acc(model, world, list(range(eb.N_BASE)), list(range(eb.N_FAM)))
        zs = eb.attr_acc(model, world, list(range(eb.N_BASE, eb.N_BASE + eb.N_NEW)),
                         list(range(eb.N_FAM)))
        print(f"seed {seed}: base={base_acc:.3f} zs={zs:.3f} ({time.time()-t0:.0f}s)",
              flush=True)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.lora_params():
            p.requires_grad_(True)
        for n_dream in [1, 2, 4]:
            for arm, lam, shuf in ARMS:
                m = consolidate(model, world, seed, n_dream, lam, shuf)
                m.update(seed=seed, n_dream=n_dream, arm=arm, lam=lam,
                         shuffled=shuf, k_base=2, base_acc_pre=base_acc,
                         new_zero_shot=zs)
                all_results.append(m)
                with open(OUT, "w") as f:
                    json.dump(all_results, f, indent=1)
        print(f"  seed {seed} cells done ({time.time()-t0:.0f}s)", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
