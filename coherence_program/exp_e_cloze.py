"""Supplementary pass for exp_e: IN-FORMAT cloze recall per seed x arm.

Retrains each arm exactly as exp_e_realm.consolidate does (same paired
seeds, streams, inits), then measures city recall by restricted choice on
the template-2 prefix ("{name} is a {occ} who lives and works in") — a
surface form the dreams contained. Together with exp_e's QA and
composition numbers this gives the format-boundness gradient:
in-format -> QA-format -> composition.

Also records QA again on the same retrained model (sanity: should match
the main run's numbers, same pairing).
"""

import json
import os
import time

import numpy as np
import torch

import exp_e_realm as ee
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results_e", "results_e_cloze.json")


def cloze_city(model, tok, world, loras):
    ee.set_lora(loras, True)
    correct = []
    for e in range(ee.N_ENTS):
        prefix = (f"{world.names[e]} is a {ee.OCCS[world.occ[e]]} "
                  f"who lives and works in")
        ans = ee.pick(model, tok, prefix, [" " + c for c in ee.CITIES])
        correct.append(ans == world.city[e])
    return float(np.mean(correct))


def main():
    tok = AutoTokenizer.from_pretrained(ee.MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ee.MODEL_NAME, torch_dtype=torch.float32).to(ee.DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    loras = ee.add_lora(model, seed=0)
    results = []
    for seed in ee.SEEDS:
        world = ee.World(seed)
        for arm, lam, shuf in ee.ARMS:
            t0 = time.time()
            ee.consolidate(model, tok, world, loras, seed, lam, shuf)
            row = {"seed": seed, "arm": arm,
                   "cloze_city": cloze_city(model, tok, world, loras)}
            row["qa_city"], row["qa_occ"] = ee.declarative_qa(model, tok, world, loras)
            results.append(row)
            print(f"seed={seed} {arm:>8}: cloze={row['cloze_city']:.3f} "
                  f"qa_city={row['qa_city']:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
            with open(OUT, "w") as f:
                json.dump(results, f, indent=1)
    print("done.")


if __name__ == "__main__":
    main()
