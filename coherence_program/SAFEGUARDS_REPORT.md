# Track 2, Experiment C — Safeguards vs Catastrophic Forgetting

**Status: complete. 110 rows in `results_b/results_c_safeguards.json`;
analysis in `analyze_c_safeguards.py` (raw tables:
`results_b/analysis_c_safeguards.txt`); driver `exp_c_safeguards.py`.**

RESULTS.md finding #5: naive LoRA consolidation of 8 new entities collapses
base-entity attribute accuracy from 1.00 to ~0.35–0.40 (k=2, n_dream=4).
This experiment ablates the paper's omitted safeguards against that measured
baseline. Question: which safeguard buys the most forgetting reduction per
unit of new-fact acquisition lost?

Design: k=2, n_dream=4, λ ∈ {0.0, 0.3} fixed in advance (no max-over-arm
selection anywhere), seeds 0–4, paired (same pretrained base, dream stream,
and LoRA re-init per seed), 600 consolidation steps, CPU. Variants: naive
(= rerun_k2_power loop, verified bit-for-bit in a smoke test), trust
(per-step spectral cap on each block's Δ(BA)), decay (A,B ← (1−η)(A,B) per
step; η ∈ {1e-4, 1e-3}), shadow (factor EMA read out at β ∈ {0.05, 0.005};
training untouched — shadow-live == naive bit-for-bit, verified), combo
(trust + decay(best η=1e-3 by the committed λ=0 rule) + shadow readouts),
plus post-hoc calibrated arms trust_cal / combo_cal (see §Trust).

## Process notes (read first — one honest failure)

1. The original driver died overnight when its shell exited; the coordinator
   relaunched it from the cached pretrained models. Results are unaffected
   (all consolidation runs are deterministic given the cache).
2. **The trust-region δ_max was never calibrated before the grid ran.**
   The spec was to set δ_max so ~10–30% of steps clip. The placeholder guess
   δ_max=0.004 went into the grid, and post-hoc calibration showed it sits
   near the *4th percentile* of actual step norms: **the `trust` and `combo`
   arms clipped on 100.0% of steps in every run**. They therefore measure a
   *hard per-step movement cap*, not the spec'd trust region. The spec'd
   regime was added afterwards as `trust_cal`/`combo_cal` (δ_max=0.02 ≈ p80;
   realized clip rate 0.17–0.31 across all 20 runs — inside the band).
   Both regimes are reported; the accident turned out to be informative.
3. Predictions below were pre-registered in the `exp_c_safeguards.py`
   docstring before the grid ran; P-C2's prediction assumed a *calibrated*
   trust region, so it is scored against `trust_cal`.

## Pre-registered predictions vs outcomes

| # | Prediction (abbreviated) | Outcome |
|---|---|---|
| P-C1 | naive reproduces powered-rerun seeds 0–4 numbers | **Supported.** base 0.350/0.368 vs 0.405/0.402 (λ=0/0.3), eval 0.724/0.836 vs 0.761/0.837, dream 1.000 — within seed noise (CPU vs MPS RNG paths differ) |
| P-C2 | trust: +0.05–0.15 forgetting protection at small cost | **Refuted — null.** Calibrated trust_cal: +0.019 (t=2.3) / +0.014 (t=1.0), mixed sign across seeds (7/10 positive) |
| P-C3 | decay: η=1e-4 ≈ naive; η=1e-3 real protection, visible cost | **Mostly supported.** η=1e-3: +0.136/+0.121; but η=1e-4 is a real (tiny) effect, +0.021, 10/10 seeds — not indistinguishable. Cost landed entirely on eval transfer; dream acc never moved (1.000 in all 20 decay runs) |
| P-C4 | shadow: live==naive; β=0.05 ≈ live; β=0.005 protects +0.02–0.10; clearest effect = seed-spread reduction (Track-1 B3 analogy) | **Means supported, variance claim refuted.** live==naive bit-for-bit; β=0.05 ≈ naive (+0.003/+0.011); β=0.005 +0.087/+0.109 (10/10 seeds). But per-seed SDs are *not* smaller (similar or slightly larger in 6/8 metric×λ cells) — B3's variance-reduction story does not transfer |
| P-C5 | combo: roughly additive, most protective, worst acquisition | **Supported.** Hard-cap combo: base 0.741/0.803 live, 0.904/0.936 at β=0.005 — but dream 0.32–0.61. Calibrated combo_cal ≈ decay+shadow sum (+0.230 vs 0.136+0.087=0.223 at λ=0) |
| P-C6 | efficiency ordering: trust ≥ shadow(.005) > decay(1e-3) ≫ decay(1e-4) ≈ 0 | **Refuted.** Observed: decay(1e-3) ≥ shadow(.005) ≫ trust_cal ≈ 0 per-unit; hard-cap trust protects a lot but at >1:1 cost. decay(1e-4) is small-but-free, not zero |
| P-C7 | no safeguard changes the recon-vs-cohere comparison | **Representationally supported, one functional exception.** Retrieval/probe contrasts stay large positive under every safeguard (+0.26 to +0.53 / +0.30 to +0.39). Under the *binding hard cap only*, the functional contrast flips negative (trust d_eval −0.099, t=−5.3) and cohere becomes base-protective (+0.070); the flip disappears under calibrated trust (d_eval +0.076) |

## Forgetting vs acquisition — the tradeoff table

Means over 5 seeds, [min–max]. Forgetting metric: `base_attr` (naive floor
~0.35; pre-consolidation ceiling 1.00). Acquisition: `dream` = attribute acc
in dreamed families, `eval` = transfer to never-dreamed families. Paired
deltas vs naive; "cost" = naive − arm (positive = acquisition lost).

**λ = 0.0 (reconstruction-only):**

| arm | base_attr | dream | eval | Δforget (t) | cost dream | cost eval | Δf/(Σcost) |
|---|---|---|---|---|---|---|---|
| naive | 0.350 [0.18–0.47] | 1.000 | 0.724 | — | — | — | — |
| decay η=1e-4 | 0.371 [0.19–0.48] | 1.000 | 0.725 | +0.021 (5.1) | 0.000 | −0.002 | ~free |
| decay η=1e-3 | 0.486 [0.25–0.59] | 1.000 | 0.664 | **+0.136 (6.5)** | 0.000 | +0.059 | **2.3** |
| shadow β=0.005 | 0.436 [0.20–0.57] | 0.999 | 0.685 | +0.087 (4.3) | +0.001 | +0.039 | 2.2 |
| shadow β=0.05 | 0.352 | 1.000 | 0.729 | +0.003 (1.4) | 0.000 | −0.005 | ~0 (null) |
| trust_cal (δ=.02) | 0.369 [0.20–0.50] | 1.000 | 0.681 | +0.019 (2.3) | 0.000 | +0.043 | 0.4 (null-ish) |
| trust (hard cap δ=.004) | 0.581 [0.37–0.69] | 0.821 | 0.545 | +0.232 (8.6) | +0.179 | +0.179 | 0.6 |
| combo_cal live | 0.485 [0.28–0.62] | 1.000 | 0.639 | +0.135 (7.0) | 0.000 | +0.085 | 1.6 |
| combo_cal β=0.005 | 0.579 [0.33–0.69] | 0.977 | 0.589 | +0.230 (8.2) | +0.023 | +0.134 | 1.5 |
| combo (hard cap) live | 0.741 [0.56–0.86] | 0.605 | 0.398 | +0.392 | +0.395 | +0.325 | 0.5 |
| combo (hard cap) β=0.005 | 0.904 [0.81–0.96] | 0.385 | 0.280 | +0.554 | +0.615 | +0.444 | 0.5 |

**λ = 0.3 (composite):** same ordering, slightly worse trust economics:
naive base 0.368; decay 1e-3 +0.121 at eval cost +0.045 (ratio 2.7);
shadow .005 +0.109 at costs +0.009/+0.082 (1.2); trust_cal +0.014 n.s.;
hard-cap trust +0.284 at costs +0.316/+0.390 (0.4); combo_cal live +0.130
at 0/+0.099 (1.3), β=0.005 readout +0.249 at +0.039/+0.223 (1.0); hard-cap
combo β=0.005 +0.568 at +0.681/+0.582 (0.4). Full tables in
`results_b/analysis_c_safeguards.txt`.

### The frontier, ordered by efficiency (Δforget per unit acquisition lost)

1. **decay η=1e-4 — tiny but free** (+0.021, 10/10 seeds, zero measurable
   cost). A null-adjacent finding worth keeping: mild homeostatic decay is
   never harmful here.
2. **decay η=1e-3 — the efficiency winner at meaningful effect size**:
   recovers ~0.13 of base accuracy (10/10 seeds), *zero* dream-family cost
   in all 20 runs, lm_dream +0.008–0.011 only; the entire cost is −0.05
   eval-family transfer.
3. **shadow β=0.005** — close second (+0.09/+0.11, 10/10 seeds, near-zero
   dream cost, −0.04/−0.08 eval); β=0.05 is a null (EMA time constant ~20
   steps just tracks the live weights).
4. **decay + shadow stack additively** (= combo_cal, since its trust
   component is ~null): +0.23/+0.25 protection at dream −0.02/−0.04 and
   eval −0.13/−0.22; ratio ~1.0–1.6.
5. **trust region, calibrated per spec (10–30% clip): ~null** for
   forgetting (+0.019/+0.014, mixed sign) with a small eval cost.
6. **hard per-step cap** (the miscalibrated arm): the only way we got
   large single-safeguard protection (+0.23/+0.28; combo to 0.90+), but it
   pays >1:1 — every point of base accuracy costs more than a point of
   acquisition, and lm_dream degrades 35–100%.

No safeguard is Pareto-dominant: the frontier is monotone (more protection
always costs more acquisition). But the *slope* varies ~5x, and everything
past decay+shadow buys protection at a losing exchange rate.

## The trust-region calibration finding (and what δ_max should be)

The main grid's trust arms ran at δ_max=0.004 and clipped on **100.0% of
steps in all 20 runs** — the calibration never ran before launch (see
Process notes). Post-hoc calibration on unclipped seed-0 dynamics (9,600
pair-steps): p50=0.0090, p75=0.0173, **p80=0.0195**, p90=0.0259,
p95=0.0325. To sit in the spec'd 10–30% clip band, **δ_max ≈ 0.02–0.026**;
the rerun at δ_max=0.02 realized clip rates 0.17–0.31, confirming the
calibration transfers across seeds and λ.

The substantive result this accident produced: **clipping only the tail of
large steps (the paper's §2.4 mechanism, properly calibrated) does
essentially nothing for forgetting in this toy, while capping *every* step
protects strongly but at >1:1 acquisition cost.** Together with decay being
the efficiency winner, this says forgetting damage here is not carried by a
minority of outlier updates — it accumulates with *total adapter
movement*. Safeguards that bound the endpoint (decay shrinks the norm,
shadow averages the path, hard cap bounds path length at 600·δ_max) work in
proportion to how much endpoint movement they remove; a tail-clipper that
leaves total movement nearly intact removes nearly nothing. If the paper
wants §2.4's trust region to double as a forgetting control, it needs to
either bind on most steps (and own the acquisition cost) or be motivated
purely as an optimization-stability device, with §3.4's homeostatic decay
carrying the forgetting-control load.

## Does any safeguard change the recon-vs-cohere comparison?

Mostly no — the paper's representational claim is robust to every safeguard
tested. Cohere−recon at fixed λ=0.3 vs 0.0 (paired, 5 seeds):

- retrieval: +0.26 to +0.53 across all 13 arms (naive +0.51, decay 1e-3
  +0.50, combo_cal +0.35); probe: +0.30 to +0.39 everywhere. All positive,
  most t > 3.
- base_attr: coherence remains ~forgetting-neutral under naive/decay/
  shadow/calibrated arms (+0.00 to +0.04), mildly protective under the
  hard-cap arms (+0.06/+0.07).
- The one interaction: under the *binding hard cap*, coherence's small
  functional gain flips to a cost (trust d_eval −0.099, t=−5.3; combo
  −0.05) — with a fixed movement budget, the coherence term competes with
  reconstruction for it. Under calibrated trust the contrast is positive
  again (+0.076). So: coherence and safeguards are orthogonal *except*
  when a safeguard rations total movement hard enough that the two loss
  terms become rivals.

## Per-seed spreads

Protection effects are highly consistent: decay 1e-4, decay 1e-3, shadow
β=0.005, combo_cal, and hard-cap arms are positive in 10/10 seed×λ cells
each (per-seed Δbase for decay 1e-3 at λ=0: +0.19, +0.13, +0.17, +0.07,
+0.12). trust_cal is 7/10 positive with sign flips — a null. Seed 3 is
uniformly the worst seed (naive base 0.18–0.21; every safeguard's absolute
level is lowest there, but deltas hold). The shadow-as-variance-reducer
prediction failed: SD across seeds for β=0.005 readouts is within noise of
naive (e.g., base_attr SD 0.126 vs 0.099 at λ=0, 0.115 vs 0.098 at λ=0.3).

## Honest caveats

- **The calibration failure is mine.** δ_max=0.004 entered the grid
  unvalidated; the docstring's calibration section originally contained
  placeholder numbers. Both are corrected and disclosed (here and in the
  docstring). The hard-cap arms are still well-defined experiments — just
  not the pre-registered one; the calibrated arms are the pre-registered
  test and were run after seeing the hard-cap results (they inherit
  post-hoc status).
- 5 seeds. Small effects (decay 1e-4's +0.021, shadow β=0.05's +0.003–.011)
  have consistent signs but toy-scale certainty only; t-values are
  descriptive.
- Single cell of the diversity grid (n_dream=4, k=2). At n_dream=1–2 the
  naive forgetting is milder (0.50–0.53 base) and the tradeoff could
  differ.
- No decay+shadow-without-trust arm was run; the claim combo_cal ≈
  decay+shadow rests on trust_cal ≈ null plus additivity, not a direct
  measurement.
- Best-η selection used the committed λ=0 rule on this same data before
  running combo; only combo inherits that mild selection, and both η values
  are reported standalone.
- The trust region approximates the paper's operator-norm constraint by
  factor interpolation after an AdamW step (first-order in step size), with
  Adam moments left untouched; the paper's plain-GD trust region could
  behave differently, particularly because Adam's moments keep pushing
  along clipped directions on subsequent steps.
- "Acquisition cost" splits into dream-family accuracy (never harmed by
  decay/shadow/calibrated arms) and eval-family transfer (always the first
  casualty). If cross-context transfer is the quantity of interest — as the
  rest of Track 2 argues — the relevant costs are the eval ones quoted.
- Forgetting here is adapter-induced and fully reversible by unplugging the
  LoRA (base weights are frozen); safeguards matter for the regime where
  the adapter must stay plugged in. Shadow readouts exploit exactly this.
- Toy scale (4 layers, d=128, 8-token grammar); the "forgetting tracks
  total movement, not step outliers" conclusion is a hypothesis at LLM
  scale, not a result.

## Reproduction

```
K_BASE_FAMS is set inside the driver; CPU is forced.
python3.10 -W ignore exp_c_safeguards.py            # full grid (sequential)
python3.10 -W ignore exp_c_safeguards.py calibrate  # step-norm percentiles
python3.10 -W ignore exp_c_safeguards.py supplement # trust_cal/combo_cal
python3.10 -W ignore analyze_c_safeguards.py        # tables
```
