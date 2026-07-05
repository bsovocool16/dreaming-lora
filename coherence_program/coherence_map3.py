"""
Coherence map, round 3 — the corrected FACT probe.

Round 2's color probe failed its own pre-registered sanity check (R-b2):
with 8 new entities and 8 colors, color labels nearly relabel identity, so
a probe trained on new-entity states memorizes entity->color itself and
rides the ambient identity signal (zero-adapter arm decoded "facts" the
model never learned, 0.84).

Corrected design: train the color probe on BASE-entity states (32 entities,
colors the model genuinely knows from pretraining), test on NEW-entity
states in EVAL families. Identity memorization cannot transfer to unseen
entities; only a shared color ENCODING can. Where the probe transfers is
where the model writes consolidated facts in the same code it uses for
pretrained facts.

Also reports base->base transfer (24 train / 8 held-out base entities) as
the probe ceiling per site: if base->base is low at a site, color isn't
linearly coded there for anyone and a low new-entity number means nothing.

Pre-registered readings:
  R3-1 (route story, corrected): under recon-only, new-entity color
        transfers at pos 5 deep (the model must represent the fact there to
        predict the attr token) and NOT at pos 4; cohere may or may not add
        pos-4 fact legibility — the coherence loss never sees color, so a
        positive result would mean identity-clustering co-locates facts.
  R3-2: zero-adapter arm now shows ~chance transfer for new entities at all
        sites (the model does not know these facts), while base->base
        ceiling is high wherever facts are coded. This is the sanity check;
        if it fails again the probe methodology is wrong in a deeper way.
"""

import json
import os
import time

import numpy as np
import torch

assert os.environ.get("K_BASE_FAMS") == "2", "run with K_BASE_FAMS=2"
import exp_b_transformer as eb
from rerun_k2_power import ridge_probe
from coherence_map import (SEEDS, N_DREAM, POSITIONS, DEPTH_NAMES,
                           consolidate, all_depth_states)

ARMS = [("none", None), ("recon", 0.0), ("cohere", 0.3)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results_b", "coherence_map3.json")


def zero_adapter(model):
    for b in model.blocks:
        b.lora["qb"].data.zero_()
        b.lora["vb"].data.zero_()
    model.set_lora(True)


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
        base_ents = list(range(eb.N_BASE))
        base_tr, base_te = base_ents[:24], base_ents[24:]
        new_ents = list(range(eb.N_BASE, eb.N_BASE + eb.N_NEW))
        color = {e: int(world.color[e]) for e in range(eb.N_BASE + eb.N_NEW)}
        dream_fams = eb.DREAM_FAM_POOL[:N_DREAM]
        for arm, lam in ARMS:
            if lam is None:
                zero_adapter(model)
            else:
                consolidate(model, world, seed, lam)
            # base-entity states from all 8 fams (restrict_base handles k=2
            # allowed-family membership internally)
            st_btr, k_btr = all_depth_states(model, world, base_tr,
                                             list(range(eb.N_FAM)), n=1024)
            st_bte, k_bte = all_depth_states(model, world, base_te,
                                             list(range(eb.N_FAM)), n=512,
                                             rng_seed=54323)
            st_new, k_new = all_depth_states(model, world, new_ents,
                                             eb.EVAL_FAMS, n=512,
                                             rng_seed=54322)
            for d, dname in enumerate(DEPTH_NAMES):
                for pos in POSITIONS:
                    z_btr = st_btr[d][:, pos]
                    c_btr = np.array([color[k] for k in k_btr])
                    z_bte = st_bte[d][:, pos]
                    c_bte = np.array([color[k] for k in k_bte])
                    z_new = st_new[d][:, pos]
                    c_new = np.array([color[k] for k in k_new])
                    results.append({
                        "seed": seed, "arm": arm, "depth": dname, "pos": pos,
                        "fact_new": ridge_probe(z_btr, c_btr, z_new, c_new),
                        "fact_base_ceiling": ridge_probe(z_btr, c_btr, z_bte, c_bte),
                    })
            model.set_lora(False)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=1)
        print(f"seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    for metric, label in [("fact_new", "FACT transfer to NEW entities"),
                          ("fact_base_ceiling", "base->base ceiling")]:
        print(f"\n{label} (mean over seeds) — none | recon | cohere")
        print(f"{'depth':>5} | " + " | ".join(f"pos {p}" for p in POSITIONS))
        for dname in DEPTH_NAMES:
            row = f"{dname:>5} | "
            for pos in POSITIONS:
                vals = []
                for arm, _ in ARMS:
                    xs = [r[metric] for r in results
                          if r["arm"] == arm and r["depth"] == dname
                          and r["pos"] == pos]
                    vals.append(np.mean(xs))
                row += "/".join(f"{v:.2f}" for v in vals) + " | "
            print(row)


if __name__ == "__main__":
    main()
