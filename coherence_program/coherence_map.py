"""
Layer x position coherence map — the gate check for the use/reusability recut.

Question: is recon-only consolidation incoherent EVERYWHERE, or does it build
entity structure at sites other than the one the coherence loss trains
(last layer, name-B token)? If recon-only is coherent somewhere (most
plausibly the rel-token position, where the functional route must carry
entity->attribute), the paper claim needs re-SITING, not re-leveling.

Design (pre-registered):
  - k=2 arm (K_BASE_FAMS=2), n_dream=4, lam in {0.0 (recon), 0.3 (cohere)},
    5 seeds, same pairing discipline as rerun_k2_power.py.
  - Capture hidden states at 6 depths (embedding, blocks 1-4, final LN) x
    4 positions (2: pre-name filler control; 3: name-A; 4: name-B; 5: rel
    token). Coherence loss trains ONLY depth=lnf, pos=4.
  - Per site: ridge-probe entity accuracy (train dream-family states, test
    eval-family states, un-normalized) and Fisher ratio on eval families.
  - Position 2 (family filler, before any name token) is the floor: causal
    attention means NO entity information can exist there; probe accuracy
    at pos 2 estimates the probe's false-transfer baseline (~1/8).
  - Position 6 (attr token) is EXCLUDED: teacher-forced input there is the
    attribute itself, which correlates with entity -> label leakage.

Pre-registered readings:
  R1 (recut safe): recon-only probe accuracy stays near its last-layer
      level (low) at ALL post-embedding sites, including pos 5; cohere arm
      improves pos-4 sites strongly and others weakly or not at all.
  R2 (re-site the claim): recon-only shows substantially higher probe
      accuracy at pos 5 (or another site) than at the trained site — the
      route carries legible entity structure somewhere else, and
      "reconstruction never produces coherence" is false as stated.
  R3 (halo): cohere arm improves non-trained sites nearly as much as the
      trained site -> coherence propagates through the LoRA-modified
      circuit rather than being a local geometric patch. (Bears on whether
      the loss builds structure or decorates one readout.)
"""

import json
import math
import os
import time

import numpy as np
import torch

assert os.environ.get("K_BASE_FAMS") == "2", "run with K_BASE_FAMS=2"
import exp_b_transformer as eb
from rerun_k2_power import ridge_probe

SEEDS = [0, 1, 2, 3, 4]
N_DREAM = 4
ARMS = [("recon", 0.0), ("cohere", 0.3)]
POSITIONS = [2, 3, 4, 5]
DEPTH_NAMES = ["emb", "b1", "b2", "b3", "b4", "lnf"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results_b", "coherence_map.json")


def consolidate(model, world, seed, lam):
    torch.manual_seed(1000 + seed)
    for b in model.blocks:
        b.lora["qa"].data = torch.randn(eb.LORA_RANK, eb.D_MODEL, device=eb.DEV) / math.sqrt(eb.D_MODEL)
        b.lora["va"].data = torch.randn(eb.LORA_RANK, eb.D_MODEL, device=eb.DEV) / math.sqrt(eb.D_MODEL)
        b.lora["qb"].data.zero_()
        b.lora["vb"].data.zero_()
    model.set_lora(True)
    new_ents = list(range(eb.N_BASE, eb.N_BASE + eb.N_NEW))
    dream_fams = eb.DREAM_FAM_POOL[:N_DREAM]
    rng = np.random.default_rng(777 + seed)
    opt = torch.optim.AdamW(model.lora_params(), lr=eb.CONSOL_LR)
    for _ in range(eb.CONSOL_STEPS):
        idx, meta = world.batch(new_ents, dream_fams, eb.CONSOL_BATCH, rng)
        logits, h = model(idx, return_hidden=True)
        L_rec = eb.lm_loss(logits, idx)
        L_coh = eb.fisher_coherence(h[:, eb.POS_NAME_B],
                                    np.array([m[0] for m in meta]))
        loss = (1 - lam) * L_rec + lam * L_coh
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lora_params(), 1.0)
        opt.step()
    return dream_fams


@torch.no_grad()
def all_depth_states(model, world, ents, fams, n=512, rng_seed=54321):
    """Returns states[depth][n, seq, d] for depths emb, b1..b4, lnf."""
    rng = np.random.default_rng(rng_seed)
    idx, meta = world.batch(ents, fams, n, rng)
    acts = []
    hooks = [b.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
             for b in model.blocks]
    _, h_final = model(idx, return_hidden=True)
    for hk in hooks:
        hk.remove()
    T = idx.shape[1]
    emb = model.emb(idx) + model.pos(torch.arange(T, device=idx.device))[None]
    states = [emb] + acts + [h_final]
    ks = np.array([m[0] for m in meta])
    return [s.cpu().numpy() for s in states], ks


def main():
    results = []
    for seed in SEEDS:
        t0 = time.time()
        world = eb.World(seed)
        model = eb.pretrain(world, seed)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.lora_params():
            p.requires_grad_(True)
        new_ents = list(range(eb.N_BASE, eb.N_BASE + eb.N_NEW))
        for arm, lam in ARMS:
            dream_fams = consolidate(model, world, seed, lam)
            st_tr, k_tr = all_depth_states(model, world, new_ents, dream_fams)
            st_ho, k_ho = all_depth_states(model, world, new_ents, eb.EVAL_FAMS,
                                           rng_seed=54322)
            for d, dname in enumerate(DEPTH_NAMES):
                for pos in POSITIONS:
                    z_tr, z_ho = st_tr[d][:, pos], st_ho[d][:, pos]
                    results.append({
                        "seed": seed, "arm": arm, "depth": dname, "pos": pos,
                        "probe": ridge_probe(z_tr, k_tr, z_ho, k_ho),
                        "fisher": eb.np_fisher(z_ho, k_ho),
                    })
            model.set_lora(False)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=1)
        print(f"seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    # summary map
    print("\nPROBE accuracy map (mean over seeds) — recon | cohere")
    print(f"{'depth':>5} | " + " | ".join(f"pos {p}" for p in POSITIONS))
    for dname in DEPTH_NAMES:
        row = f"{dname:>5} | "
        for pos in POSITIONS:
            v = {}
            for arm, _ in ARMS:
                xs = [r["probe"] for r in results
                      if r["arm"] == arm and r["depth"] == dname and r["pos"] == pos]
                v[arm] = np.mean(xs)
            row += f"{v['recon']:.2f}/{v['cohere']:.2f} | "
        print(row)


if __name__ == "__main__":
    main()
