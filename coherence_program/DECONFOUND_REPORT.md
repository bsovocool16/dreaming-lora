# Experiment B deconfound — entities-per-family held constant while k varies

Status: COMPLETE. The pre-registration below was written 2026-07-02 before
the run and is unedited; results and verdict were appended 2026-07-03 after
the run finished. Process note: the first launch died with its shell
overnight; the identical script was relaunched and ran to completion
(log: `deconfound_run.log`; data: `results_b/results_b_deconfound.json`).

## The confound being removed

The original k=8 vs k=2 contrast (RESULTS.md §Experiment B) varied
families-per-entity, but at fixed N_BASE=32 it ALSO varied data
composition: at k=2 each family is pretrained on only ~8 base entities
instead of 32, so the eval families' (6, 7) binding circuitry saw 4× fewer
entities. The "binding entanglement determines functional headroom" reading
is therefore confounded with "eval families were undertrained."

## Design

k=2 with N_BASE=128 (name pools widened to 16×16): 128 entities × 2
families / 8 families ≈ 32 base entities per family in expectation —
matching the k=8 arm's per-family exposure while keeping per-entity family
diversity at k=2. Driver: `exp_b_deconfound.py` (imports
`exp_b_transformer` with K_BASE_FAMS=2, overrides N_FIRST=N_LAST=16,
N_BASE=128 on the module before any World/model exists; reuses
`consolidate()` and `ridge_probe()` from `rerun_k2_power.py` unmodified).

- 3 seeds (0–2); n_dream ∈ {1, 2, 4}; λ fixed ∈ {0.0, 0.3}. No λ grid, no
  max-over-λ (house rule).
- Pretraining: 12000 steps to start (4× memorization load vs the original
  arms' 8000); per-seed ≥0.95 base-accuracy gate checked before
  consolidation, extended in 4000-step chunks (cap 28000) if it fails, with
  the required budget reported.
- Verification per seed before training: vocab size 114, entity count 136,
  all name pairs unique, per-family base-entity coverage printed from
  `world.base_fams`.

## Pre-registered predictions (before running)

The decision variable is recon-only (λ=0) `attr_eval_acc` on never-dreamed
families 6–7, compared against the two prior arms:

- **P1 (data-composition story):** if the k-effect was driven by data
  composition, recon-only attr_eval_acc here rises to near the k=8 arm's
  levels, ~0.75–0.95 across n_dream ∈ {1,2,4}. The k-contrast was a
  training-data artifact and the binding-entanglement interpretation should
  be retracted.
- **P2 (binding-entanglement story):** if per-entity family diversity is
  what matters, recon-only attr_eval_acc stays near the original k=2
  levels, ~0.48–0.81 (powered rerun: 0.476 / 0.672 / 0.810 at n_dream
  1/2/4) despite matched per-family exposure. The interpretation survives
  the confound.
- **Ambiguous zone:** results falling materially between the two prior
  arms' envelopes (e.g., a partial rise) mean both factors contribute;
  report as such, no forced call.
- Secondary (not decision-bearing, recorded for consistency): the λ=0.3
  arm's probe/retrieval advantages over recon should persist if the k=2
  representational findings were not composition artifacts either.

With 3 seeds this is directional, not definitive — per-seed spreads will be
reported alongside means, and no significance claims will be made.

---

# Results (appended 2026-07-03, after the run)

## Run integrity

- All per-seed verifications passed: vocab 114, 136 entities, all name
  pairs unique. Base entities per family (target ~32): seed 0
  [24,39,31,37,25,25,34,41], seed 1 [32,35,34,27,31,29,35,33], seed 2
  [32,31,32,31,25,36,36,33] — binomial spread around 32; eval families
  6–7 saw 33–41 base entities per seed, i.e. at least the k=8 arm's 32.
- Pretraining gate: base_attr = 1.000 for all 3 seeds at the planned
  12000 steps (already 1.000 at the step-2000 checkpoint, so the 4×
  memorization load was comfortably absorbed; no extension needed).
- New-entity zero-shot before consolidation: 0.191 / 0.137 / 0.125
  (chance ≈ 0.125–0.17 in prior arms) — all transfer is
  consolidation-driven, as before.

## Headline: recon-only (λ=0) attr_eval_acc, never-dreamed families 6–7

| n_dream | original k=2 (32 ents, 15 seeds) | **deconfounded k=2 (128 ents, 3 seeds)** | k=8 (32 ents, 3 seeds) | gap closed |
|---|---|---|---|---|
| 1 | 0.476 (sd 0.14) | **0.498** [0.287, 0.590, 0.617] | 0.747 [0.455, 0.838, 0.947] | **8%** |
| 2 | 0.672 (sd 0.12) | **0.791** [0.773, 0.777, 0.822] | 0.947 [0.914, 0.957, 0.971] | **43%** |
| 4 | 0.810 (sd 0.10) | **0.911** [0.875, 0.910, 0.949] | 0.954 [0.885, 0.979, 0.998] | **70%** |

"Gap closed" = (deconf − k2) / (k8 − k2), using arm means. Per-seed
values in brackets, sorted. Prior sources: `results_b_k2_power.json`
(recon arm) and `results_b.json` (λ=0.0 cells).

Reading by cell:

- **n_dream=1: no gap closure.** 0.498 sits on the original k=2 value
  (0.476); the k=8 mean (0.747) is far above, though that reference is
  itself noisy (3 seeds spanning 0.455–0.947).
- **n_dream=2: partial closure.** 0.791 is ~1 sd above the k=2 mean, but
  the seed ranges of the deconfounded run (0.773–0.822) and the k=8 arm
  (0.914–0.971) do not overlap — still clearly below k=8.
- **n_dream=4: near-complete closure.** 0.911 vs 0.954; per-seed ranges
  overlap (0.875–0.949 vs 0.885–0.998) — indistinguishable from k=8 at
  this seed count.

## Verdict: ambiguous per the pre-registered rule — both factors real, with a diversity-dependent split

Neither prediction holds across the board, which is exactly the
pre-registered "ambiguous zone" (results materially between the two
envelopes → both factors contribute, no forced call):

- **P1 (data composition) is mostly right at n_dream ≥ 2.** Matching
  per-family entity exposure recovers 43–70% of the k-contrast, and at
  n_dream=4 the deconfounded k=2 run is functionally indistinguishable
  from k=8. The original claim that "base-model binding entanglement
  determines functional headroom" was substantially inflated by the
  data-composition confound — the reviewer's objection was correct and
  material.
- **P2 (binding entanglement) survives only at n_dream=1.** With
  per-family exposure matched, the lowest-diversity cell shows
  essentially none of the k-effect closing (8%). A residual, genuinely
  k-linked deficit remains when consolidation itself provides no
  cross-family signal.
- Synthesis: per-entity family diversity in pretraining and dream
  diversity at consolidation appear to be **substitutes**. When dreams
  span several families, well-trained per-family binding circuitry
  (which entity count buys) is enough for transfer; only when dreams are
  confined to one family does the base model's family-invariant binding
  (which k buys) still matter.
- RESULTS.md's k-arm paragraph should be revised accordingly: from
  "suggestive that base abstraction determines functional headroom" to
  "the k-contrast was mostly data composition at n_dream ≥ 2; a residual
  base-abstraction effect persists at n_dream = 1."

## Secondary (pre-registered, not decision-bearing): λ=0.3 vs recon

The powered-k=2 representational findings persist in this regime.
Paired per-seed gaps (cohere − recon):

| n_dream | attr_eval Δ | probe_eval Δ | retrieval_eval Δ |
|---|---|---|---|
| 1 | −0.003 [−0.023, +0.002, +0.012] | +0.193 [+0.049, +0.100, +0.432] | +0.113 [−0.088, +0.172, +0.256] |
| 2 | −0.092 [−0.223, −0.041, −0.014] | +0.127 [−0.164, +0.195, +0.350] | +0.212 [−0.049, +0.262, +0.422] |
| 4 | −0.027 [−0.094, −0.016, +0.027] | +0.279 [+0.045, +0.377, +0.414] | +0.459 [+0.344, +0.486, +0.547] |

- At n_dream=4 the coherence arm's probe (0.867 vs 0.589) and retrieval
  (0.788 vs 0.329) advantages are large and positive in all 3 seeds,
  with fisher_eval 0.70 vs 6.73 — same pattern as the 15-seed powered
  run. The functional metric shows no coherence advantage (small
  negative means, per-seed mixed; the −0.092 at n_dream=2 is driven by
  one seed's −0.223), also consistent with the powered run's functional
  null.
- Controls: attr_dream_acc = 1.000 in every cell (consolidation itself
  always works); lm_dream cost of λ=0.3 ≤ ~5% (1.214 vs 1.155 worst
  cell); base-entity forgetting is similar across arms (0.37–0.49,
  comparable to the powered k=2 run's 0.38–0.53).

## Caveats

- **3 seeds — directional, not definitive.** No significance claims.
  The n_dream=1 verdict cell has a wide per-seed spread on both sides
  (deconf 0.287–0.617; k=8 reference 0.455–0.947, also only 3 seeds), so
  the "residual entanglement at n_dream=1" reading is the most fragile
  part; the n_dream=2 non-overlap and n_dream=4 overlap are more solid.
- The deconfound necessarily changed more than N_BASE: vocab grew 98 →
  114 (name pools 8×8 → 16×16), and name-pair density rose (136/256
  pairs used vs 40/64). Per-token exposure statistics therefore differ
  somewhat from both prior arms; entities-per-family is the quantity
  that was matched.
- Pretraining used 12000 steps vs the prior arms' 8000. Gate saturated
  (1.000 everywhere, as in prior arms), so all arms compare
  fully-memorized base models.
- Runtime: 6–42 min/seed pretrain on CPU (2518 s / 1483 s / 363 s;
  wall-clock variance was system load, not the model).
