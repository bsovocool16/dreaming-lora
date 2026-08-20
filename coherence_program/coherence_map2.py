"""
Coherence map, round 2 — two controls that gate interpretation of round 1.

Round-1 map found recon-only carries fully legible entity identity at the
rel-token position (0.86–0.98 deep), while losing it with depth at the name
token. Two confounds before concluding anything:

C-a. ZERO-ADAPTER CONTROL: is rel-position identity legibility created by
     consolidation at all, or is it the frozen base circuit aggregating
     name embeddings (which it learned to do for base entities)? Add arm
     "none": zero adapter, no consolidation, same probes.
C-b. FACT PROBE: identity-decodability is upstream of memory. The
     consolidated content is the entity->color binding. Probe COLOR (8-way,
     chance 0.125) at every site, same train-dream/test-eval protocol.
     Where is the consolidated FACT legible, per arm?

Pre-registered readings:
  R-a1: if arm "none" already shows high identity probe at pos 5 deep, the
        round-1 pos-5 result is base circuitry, and neither loss "built"
        it. Identity is ambient at aggregation sites.
  R-b1 (route story): under recon-only, color is legible at pos 5 deep
        (where the attr prediction happens) and NOT at pos 4 (the entity's
        own representation); under cohere, same at pos 5, and pos 4 color
        legibility tells us whether identity-clustering incidentally makes
        the fact readable at the content position (not obviously — the
        coherence loss never sees color).
  R-b2: arm "none" color probe ~0.125 everywhere for new entities (facts
        not yet consolidated) — sanity check that color probes measure
        consolidated memory, not leakage.
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
                   "results_b", "coherence_map2.json")


def zero_adapter(model):
    for b in model.blocks:
        b.lora["qb"].data.zero_()
        b.lora["vb"].data.zero_()
    model.set_lora(True)   # active but zero: identical to base


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
        colors = {e: int(world.color[e]) for e in new_ents}
        dream_fams = eb.DREAM_FAM_POOL[:N_DREAM]
        for arm, lam in ARMS:
            if lam is None:
                zero_adapter(model)
            else:
                consolidate(model, world, seed, lam)
            st_tr, k_tr = all_depth_states(model, world, new_ents, dream_fams)
            st_ho, k_ho = all_depth_states(model, world, new_ents, eb.EVAL_FAMS,
                                           rng_seed=54322)
            c_tr = np.array([colors[k] for k in k_tr])
            c_ho = np.array([colors[k] for k in k_ho])
            for d, dname in enumerate(DEPTH_NAMES):
                for pos in POSITIONS:
                    z_tr, z_ho = st_tr[d][:, pos], st_ho[d][:, pos]
                    results.append({
                        "seed": seed, "arm": arm, "depth": dname, "pos": pos,
                        "probe_id": ridge_probe(z_tr, k_tr, z_ho, k_ho),
                        "probe_color": ridge_probe(z_tr, c_tr, z_ho, c_ho),
                    })
            model.set_lora(False)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=1)
        print(f"seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    for metric, label in [("probe_id", "IDENTITY"), ("probe_color", "COLOR (fact)")]:
        print(f"\n{label} probe map (mean over seeds) — none | recon | cohere")
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
