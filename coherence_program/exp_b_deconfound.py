"""
Track 2, Experiment B follow-up — deconfounding k (families-per-entity)
=======================================================================

Reviewer objection (M2, accepted): the k=8 vs k=2 contrast confounds
per-entity family diversity with DATA COMPOSITION — at k=2 each family is
pretrained on only ~8 base entities (32 entities x 2 fams / 8 fams) instead
of 32, so the eval families' binding circuitry sees 4x fewer entities.

Deconfound: hold ENTITIES-PER-FAMILY constant while k varies. This driver
runs k=2 with N_BASE=128 (and name pools widened to 16x16 so 136 unique
names exist): 128 entities x 2 fams / 8 fams = 32 entities per family in
expectation — matching the k=8 arm's per-family exposure.

Pre-registered predictions (written before the full run; also in
DECONFOUND_REPORT.md):
  P1 (data-composition story): recon-only attr_eval_acc rises to near the
     k=8 arm's levels (~0.75-0.95). The k-effect was a training-data
     artifact.
  P2 (binding-entanglement story): recon-only attr_eval_acc stays near the
     original k=2 levels (~0.48-0.81) despite matched per-family exposure.
     Per-entity family diversity in pretraining is what buys functional
     headroom, not raw per-family entity count.

Committed decisions:
  - 3 seeds (0-2), n_dream in {1,2,4}, lam fixed at {0.0, 0.3} — no lam
    grid, no max-over-lam (house rule after the winner's-curse incident).
  - Paired arms exactly as rerun_k2_power.py: its consolidate() and
    ridge_probe() are imported and reused verbatim, so all metrics
    (including probe_eval) are computed identically.
  - Pretraining budget: 128 entities x 2 attrs is 4x the memorization load
    of the original arms, so start at 12000 steps (vs default 8000); if the
    >=0.95 base-accuracy gate fails, extend in 4000-step chunks up to 28000
    total and report what was needed. Gate is checked per seed BEFORE any
    consolidation.
  - No existing file is edited: all constant overrides happen on the
    imported module object, after import and before any World/model exists.
  - CPU forced (eb.DEV = "cpu") for reproducibility of this run.

Usage:
  python3.10 -W ignore exp_b_deconfound.py            # full run
  DECONF_SMOKE=1 python3.10 -W ignore exp_b_deconfound.py   # plumbing test

Writes results_b/results_b_deconfound.json (smoke: ..._smoke.json).
"""

import json
import os
import time

# MUST precede the exp_b_transformer import: K_BASE_FAMS is read at import.
os.environ["K_BASE_FAMS"] = "2"

import numpy as np
import torch

import exp_b_transformer as eb

# ---- overrides: after import, before any World/model is constructed ------
eb.DEV = "cpu"                    # force CPU (no MPS nondeterminism)
eb.N_FIRST = 16                   # name pools 8x8 -> 16x16 (256 pairs)
eb.N_LAST = 16
eb.N_BASE = 128                   # 128 base entities x k=2 fams / 8 fams
assert eb.N_NEW == 8              # unchanged                 = 32 per fam
assert eb.K_BASE_FAMS == 2

SMOKE = os.environ.get("DECONF_SMOKE") == "1"
eb.PRETRAIN_STEPS = 300 if SMOKE else 12000
if SMOKE:
    eb.CONSOL_STEPS = 50

# imported AFTER the overrides for clarity (its functions read eb.* at call
# time either way, so ordering is not load-bearing).
from rerun_k2_power import consolidate, ridge_probe  # noqa: E402,F401

SEEDS = [0, 1, 2]
N_DREAM_GRID = [1, 2, 4]
ARMS = [("recon", 0.0), ("cohere", 0.3)]      # fixed lam, no grid
GATE = 0.95
EXTEND_CHUNK = 4000
MAX_PRETRAIN_TOTAL = 28000

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_b",
                   "results_b_deconfound_smoke.json" if SMOKE
                   else "results_b_deconfound.json")


def verify_world(world, seed):
    """Confirm the module-level overrides actually reached Vocab/World."""
    expected_vocab = (2 + eb.N_FIRST + eb.N_LAST + eb.N_COLOR + eb.N_CITY
                      + eb.N_FAM * eb.FILLERS_PER_FAM + eb.N_FAM * 2)
    assert expected_vocab == 114, expected_vocab
    assert world.v.size == expected_vocab, \
        f"vocab={world.v.size}, override did not take (expected {expected_vocab})"
    n_ent = len(world.entities)
    assert n_ent == eb.N_BASE + eb.N_NEW == 136, \
        f"entity count {n_ent} != 136 — N_BASE override did not take"
    assert len(set(world.entities)) == 136, "duplicate entity name pairs"
    cov = {f: 0 for f in range(eb.N_FAM)}
    for e, fams in world.base_fams.items():
        assert e < eb.N_BASE
        assert len(fams) == 2 and len(set(fams)) == 2
        for f in fams:
            cov[f] += 1
    counts = [cov[f] for f in range(eb.N_FAM)]
    assert sum(counts) == 2 * eb.N_BASE == 256
    print(f"  [verify seed {seed}] vocab={world.v.size} entities={n_ent} "
          f"unique_names=OK", flush=True)
    print(f"  [verify seed {seed}] base entities per family: "
          f"{counts} (target ~32; eval fams 6,7 -> {cov[6]},{cov[7]})",
          flush=True)
    return cov


def extend_pretrain(model, world, steps):
    """Continue pretraining (same loop as eb.pretrain; fresh AdamW state)."""
    opt = torch.optim.AdamW((p for n, p in model.named_parameters()
                             if "lora" not in n), lr=eb.PRETRAIN_LR)
    base_ents = list(range(eb.N_BASE))
    fams = list(range(eb.N_FAM))
    for step in range(steps):
        idx, _ = world.batch(base_ents, fams, eb.PRETRAIN_BATCH)
        loss = eb.lm_loss(model(idx), idx)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 2000 == 0:
            acc = eb.attr_acc(model, world, base_ents, fams)
            print(f"  extend {step+1}: lm={loss.item():.3f} "
                  f"base_attr={acc:.3f}", flush=True)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print(f"deconfound run: k={eb.K_BASE_FAMS} N_BASE={eb.N_BASE} "
          f"N_FIRST/LAST={eb.N_FIRST}/{eb.N_LAST} DEV={eb.DEV} "
          f"pretrain_steps={eb.PRETRAIN_STEPS} smoke={SMOKE}", flush=True)
    all_results = []
    for seed in SEEDS:
        t0 = time.time()
        world = eb.World(seed)
        cov = verify_world(world, seed)
        print(f"=== seed {seed}: pretraining ({eb.PRETRAIN_STEPS} steps) ===",
              flush=True)
        model = eb.pretrain(world, seed)
        steps_used = eb.PRETRAIN_STEPS
        base_acc = eb.attr_acc(model, world, list(range(eb.N_BASE)),
                               list(range(eb.N_FAM)))
        # gate check per seed; extend if needed (report what was required)
        while base_acc < GATE and not SMOKE and steps_used < MAX_PRETRAIN_TOTAL:
            print(f"  gate FAILED at {steps_used} steps "
                  f"(base_attr={base_acc:.3f} < {GATE}) — extending "
                  f"{EXTEND_CHUNK}", flush=True)
            extend_pretrain(model, world, EXTEND_CHUNK)
            steps_used += EXTEND_CHUNK
            base_acc = eb.attr_acc(model, world, list(range(eb.N_BASE)),
                                   list(range(eb.N_FAM)))
        gate_ok = base_acc >= GATE
        zs = eb.attr_acc(model, world,
                         list(range(eb.N_BASE, eb.N_BASE + eb.N_NEW)),
                         list(range(eb.N_FAM)))
        print(f"  seed {seed}: base_attr={base_acc:.3f} gate_ok={gate_ok} "
              f"steps_used={steps_used} new-entity zero-shot={zs:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if not gate_ok:
            print(f"  GATE STILL FAILED at {steps_used} steps — recording "
                  f"and continuing (flag for review)", flush=True)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.lora_params():
            p.requires_grad_(True)
        for n_dream in N_DREAM_GRID:
            for arm, lam in ARMS:
                t1 = time.time()
                m = consolidate(model, world, seed, n_dream, lam,
                                shuffle=False)
                m.update(seed=seed, n_dream=n_dream, arm=arm, lam=lam,
                         k_base=2, n_base=eb.N_BASE,
                         ents_per_fam=[cov[f] for f in range(eb.N_FAM)],
                         pretrain_steps_used=steps_used, gate_ok=gate_ok,
                         base_acc_pre=base_acc, new_zero_shot=zs)
                all_results.append(m)
                print(f"  seed={seed} n_dream={n_dream} {arm} (lam={lam}): "
                      f"eval={m['attr_eval_acc']:.3f} "
                      f"dream={m['attr_dream_acc']:.3f} "
                      f"probe={m['probe_eval']:.3f} "
                      f"retr={m['retrieval_eval']:.3f} "
                      f"fish={m['fisher_eval']:.2f} "
                      f"base={m['base_attr_acc']:.3f} "
                      f"({time.time()-t1:.0f}s)", flush=True)
                with open(OUT, "w") as f:
                    json.dump(all_results, f, indent=1)
        print(f"  seed {seed} done ({time.time()-t0:.0f}s total)", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
