# Track 2 — Coherence Loss Validation (§3.4 / §8.1): Results

**Status: draft for discussion. No paper edits made.**
Run overnight 2026-06-10: Experiment A (linear toy, 525 cells), Experiment B
(toy transformer + LoRA, three arms: k=8, k=2, and a 15-seed powered rerun
with a shuffled-label control). An adversarial reviewer pass was taken on the
first draft; its two critical findings (a pre-registration violation in my
Option B verdict, and 3-seed gaps inflated by max-over-λ selection) were
accepted and drove the powered rerun. This document reports the post-review
numbers. All code, raw results, pre-registered predictions, and the review
are in this directory.

## The question

Contribution #2 of the paper claims reconstruction-family losses do not
produce cross-context representational consistency, and that the
Fisher-discriminant coherence loss is required for cross-session symbolic
operation. The pre-registered alternative (issues tracker, Option B): with
sufficiently diverse dreams, reconstruction alone produces coherence
emergently, and the term could be dropped.

## Headline answers

1. **The representational claim is supported, with large effects, across
   three metrics including one in a geometry the loss does not optimize —
   in graded form.** Reconstruction-only consolidation never matched the
   composite's cross-context consistency in any regime tested; the
   layer×position maps show it recovers roughly half of the coherence
   term's effect at the content position incidentally (identity probe
   0.31 base → 0.59 recon → 0.88–0.94 cohere at depth), so the correct
   claim is graded, not binary: coherence is the strong, targeted form of
   a pressure reconstruction exerts weakly. With the coherence term
   (λ=0.3): cross-context entity retrieval +0.34/+0.41 over recon-only
   at n_dream=2/4 (14/15 and 15/15 seeds, t≈6–8); a ridge-probe classifier
   on *un-normalized* states — objective-distinct, though not fully
   independent, since cosine clustering tends to induce linear separability
   — +0.25/+0.29 (13/15 seeds, t≈4). A shuffled-entity-label control at
   identical λ shows the gains mostly require the entity structure: the
   dissociation is clean at n_dream=4 (shuffled retrieval +0.01 n.s., probe
   +0.07 n.s.) and partial at n_dream=2 (shuffled recovers +0.09 of the
   +0.34 retrieval gain, significant, and +0.14 of the +0.25 probe gain,
   n.s.).

2. **The functional-transfer claim is NOT supported at this toy scale.**
   This is the night's most important negative. An earlier 3-seed run showed
   "+5–10 points" of attribute transfer from coherence at k=2; the reviewer
   flagged it as max-over-λ selection on noisy cells, and the 15-seed
   powered rerun at fixed λ=0.3 confirms the flag: cohere−recon on attribute
   transfer is −0.006 / +0.005 / +0.041 at n_dream = 1 / 2 / 4 — null, null,
   weak non-significant trend (t=1.69). In this grammar, the functional task
   can evidently be solved through routes that do not require coherent
   entity representations, even when those representations are demonstrably
   incoherent. Function and representation dissociate.

3. **Option B, judged per the pre-registered rule (gap < 5 points at highest
   diversity ⇒ term not load-bearing), splits by what you care about:**
   *functionally*, Option B is **supported** — at k=8 the gap is +0.04 under
   ceiling, and at k=2 the powered fixed-λ gaps are ≤ +0.04; *representationally*,
   Option B is **refuted everywhere**, and in Experiment A diversity
   *amplifies* the recon-vs-composite retrieval gap (+0.00 at 2 contexts →
   +0.09–0.14 at 32 for λ=0.3, up to +0.19 at λ=0.7; positive in 20/20
   seed×ω cells at λ=0.3). Diversity is a complement to the coherence term,
   not a substitute for it.

4. **The shuffled-label control rules out (1−λ) LM-downweighting as the
   source of the gains, in both directions.** Real-label coherence: large
   representational gains, no functional cost. Shuffled-label coherence at
   the same λ: *damages* functional transfer (−0.07 to −0.13, significant)
   while *protecting* base-entity facts (+0.07 to +0.12, significant).
   Note the control is adversarial rather than neutral — it is an actively
   harmful auxiliary loss (dream-domain LM cost +0.11–0.17), so it shows
   the composite's gains are not generic regularization, not that any
   auxiliary loss would behave this way. What the coherence term does is
   specific to the content structure it is given.

5. **Consolidated memory looks route-bound at toy scale, with one caveat
   under test (compositional eval, final).** A pretrained comparison
   circuit that consumed pretrained facts at 0.97+ before consolidation
   (0.72–0.85 after — consolidation damages it) shows chance-level
   consumption of LoRA-consolidated facts from either arm, while the
   declarative route reads the same facts at 0.96+ in the same models.
   Coherent type tokens did not make facts consumable by a circuit outside
   the consolidation pathway. Two free analyses support route-boundness
   over the damaged-consumer alternative: mixed new×base pairs (one fully
   consumable operand) are also at chance in all arms (0.50–0.52), and
   within the cohere arm better-preserved circuits do no better on new
   pairs (per-seed corr(comp\_bb, comp\_nn) = −0.38). The decisive check ran:
   rerunning the full eval with the decay safeguard (η=1e-3, 15 seeds)
   partially restores the substrate (base facts 0.53 → 0.62–0.65;
   comparison circuit 0.72 → 0.75–0.82, though not to its 0.97
   pre-consolidation level) and new-pair comparison REMAINS at chance in
   both arms (cohere−recon −0.010/−0.019, n.s., same-class ≈ 0.2 both) —
   a single-seed early cell suggesting the opposite (0.635) did not
   survive aggregation, continuing this program's perfect record of
   early-cell mirages. Route-boundness is now supported by four
   independent analyses (chance-level new×new under partial substrate
   protection, chance-level mixed pairs, the −0.38 circuit-health
   correlation, and the declarative route reading the same facts at
   0.95+); the residual caveat is that no arm restored the consumer
   fully. On the functional ledger overall: no functional gain of
   practical size anywhere in the program; one tiny but robust effect
   (declarative transfer at n\_dream=2: +0.019, t=6.2, 15/15 seeds,
   error-halving near ceiling; absent at n\_dream=4 and in the 4-layer
   model). The coherence term's case rests on the representational
   evidence and the §9 identity argument, with functional benefit an
   explicit conjecture for the real-LM stage.

6. **Unplanned finding: severe catastrophic forgetting in every transformer
   cell.** Naive LoRA consolidation on new-entity dreams drove base-entity
   accuracy from 1.00 to 0.12–0.83 across all configurations and arms.
   Coherence has at most a small effect on this: −0.028 at n_dream=1
   (significant, t=−3.1), ≈0 at n_dream=2/4 (n.s.) — i.e., if anything a
   slight worsening at low diversity, not the mitigation an early
   single-seed run hinted at. This toy deliberately omitted the paper's
   safeguards; the follow-up ablation ran (safeguards section below):
   homeostatic decay recovers a third of the loss at zero new-fact cost,
   and the calibrated trust region does not move forgetting.

## What this means for the paper (proposals, for discussion)

The paper's §3.4 motivates coherence *representationally* (type-level
structure; same content recognizable as same across contexts) but §8.1's
predicted test is *functional* (paraphrase recognition, entity tracking).
The toy results split exactly along that line: the representational
mechanism works as designed and reconstruction provides at most a weak
incidental fraction of it; the functional payoff did not materialize in a grammar where prediction
has coherence-free routes. Three options for the paper, roughly in order of
how much I'd defend them:

- **Sharpen the claim to the representational level** and treat functional
  consequences as regime-dependent: coherence buys the representational
  substrate (which is what §9's identity argument actually needs), and
  whether that substrate is functionally load-bearing depends on whether
  the deployment's tasks have coherence-free shortcuts. The toy's relation
  tokens are exactly such a shortcut; open-ended language mostly is not —
  but that is now an argument, not a result.
- **Keep the functional claim as a prediction** explicitly flagged as
  untested at scale, citing the toy dissociation as the reason the test
  must be run on a real model.
- **Option B (drop the term)** is now closable for the representational
  claim but, honestly, *open* for the functional one. If the paper's
  ambition is functional, the retreat path is still live.

## Experiment A — linear content/context regime

Contents observed through context transforms whose distortion occupies a
shared low-rank subspace (calibrated: raw retrieval ≈ 0.6, rank-4 oracle
≈ 0.95); rank-4 residual adapter; composite loss per paper Eq. (6); paired
worlds across λ; 5 seeds; evaluation on held-out contexts. Selected table
(ω = 0.5, retrieval accuracy on held-out contexts):

| train contexts | raw h | recon-only | λ=0.1 | λ=0.3 | λ=0.7 |
|---|---|---|---|---|---|
| 1  | 0.54 | 0.52 | 0.52 | 0.51 | 0.49 |
| 2  | 0.50 | 0.54 | 0.57 | 0.56 | 0.54 |
| 8  | 0.78 | 0.77 | 0.85 | 0.90 | 0.96 |
| 32 | 0.80 | 0.84 | 0.94 | 0.97 | 0.99 |

- Recon-only tracks raw hidden states at every diversity; the composite
  approaches the oracle, with the gap *growing* in diversity.
- Cost frontier: λ=0.1–0.3 costs −0.2% to 7.9% held-out reconstruction
  across cells; λ=0.7 up to ~50%. The knee is around λ≈0.3.
- At one training context the term does nothing or mildly hurts (−0.00 to
  −0.03 across ω); at ω=1.0, n_ctx=2 the composite is also slightly
  *negative* (−0.01 to −0.03, 4–5/5 seeds) — with too little context
  variety, the term has nothing valid to compress and only constrains.
- Caveat: held-out contexts are new coefficients on the *same* fixed
  distortion subspace — interpolation within a context family, not novel
  context structure. The regime was constructed so a rank-4 adapter *can*
  achieve coherence; the result is that only the coherence term finds it.

## Experiment B — toy transformer + LoRA + context transplantation

4-layer/128-d causal LM pretrained on an entity-attribute grammar (8
template families with private filler and relation tokens), frozen; LoRA
rank 8 on W_Q/W_V consolidates 8 new entities from transplanted dream
sentences in n_dream ∈ {1,2,4} families; families 6–7 eval-only. New-entity
zero-shot before consolidation is chance-level (0.17 in the 3-seed arms,
0.14 in the powered arm), so all transfer is consolidation-driven.

**Powered k=2 rerun (the load-bearing numbers): 15 seeds, fixed λ=0.3,
paired arms recon / cohere / shuffled-cohere.**

| metric | n_dream | recon | cohere | Δ (t, seeds+) | shuffled Δ |
|---|---|---|---|---|---|
| attr transfer (eval fams) | 1 | 0.476 | 0.470 | −0.006 (n.s., 5/15) | −0.125 (t=−4.1) |
| | 2 | 0.672 | 0.677 | +0.005 (n.s., 9/15) | −0.077 (t=−2.2) |
| | 4 | 0.810 | 0.852 | +0.041 (t=1.7, 7/15) | −0.073 (t=−3.1) |
| ridge probe (eval fams) | 1 | 0.411 | 0.473 | +0.061 (n.s., 11/15) | −0.123 (t=−2.4) |
| | 2 | 0.426 | 0.676 | **+0.250 (t=3.9, 13/15)** | +0.137 (n.s.) |
| | 4 | 0.565 | 0.859 | **+0.294 (t=4.4, 13/15)** | +0.072 (n.s.) |
| retrieval (eval fams) | 1 | 0.387 | 0.504 | +0.117 (t=2.5, 11/15) | −0.043 (n.s.) |
| | 2 | 0.355 | 0.691 | **+0.336 (t=6.4, 14/15)** | +0.090 (t=3.1) |
| | 4 | 0.398 | 0.804 | **+0.406 (t=8.4, 15/15)** | +0.012 (n.s.) |
| base-entity acc (forgetting) | 1 | 0.529 | 0.502 | −0.028 (t=−3.1, 3/15) | +0.117 (t=4.4) |
| | 2–4 | 0.38–0.43 | ≈ same (n.s.) | | +0.07–0.10 (protective) |

The representational effects are weakest at n_dream=1 and grow with
diversity — the within-transformer version of Experiment A's
diversity-amplification pattern.

- LM-loss control at λ=0.3: ≤ 6.2% dream-domain cost (worst cell), so the
  representational gains are not bought with reconstruction quality.
- Two analyses that would settle the functional question at this scale,
  not run tonight: n_dream=6 (the pool has 6 dreamable families; the
  attr-transfer trend is monotone in diversity and the +0.041 at n_dream=4
  is below the ~0.07 minimum detectable effect at 15 seeds — more seeds or
  more diversity could resolve it), and a direct test of the
  relation-token-shortcut explanation for the functional null, which is
  currently an assertion.
- The first-pass arms (3 seeds, λ grid; `results_b/results_b*.json`) agree
  representationally (recon-only Fisher 4–8 and retrieval 0.18–0.44 vs
  composite Fisher < 1.3 and retrieval 0.44–1.00 from λ≥0.1 at n_dream≥2;
  note retrieval at n_dream=1, λ=0.1 stays low, ~0.29–0.45). Their
  *functional* gaps should be disregarded in favor of the powered rerun.
  Recon-only retrieval of 0.28–0.37 is 2–3× the 0.125 chance floor —
  above chance but far below the composite.
- k=8 vs k=2 arms: with each base entity pretrained in all 8 families
  (k=8), new bindings transfer functionally ~for free (attr transfer
  0.75–0.95 recon-only) — the base model already has family-invariant
  binding circuitry. The k-arm contrast is suggestive that base-model
  abstraction determines how much functional headroom coherence has, but
  the reviewer correctly notes k also changes data composition
  (entities-per-family 32 → 8, repetitions ×4), so this was a hypothesis,
  not a finding — and the deconfound run (own section below) subsequently
  showed the confound explains most of the effect at n_dream ≥ 2.

## Coherence map (layer × position program, three rounds — 2026-07-02/03)

The gate check for the use/reusability recut, run as three successive maps
(`coherence_map*.py`, results in `results_b/coherence_map*.json`), each
round's interpretation corrected by the next round's control:

1. **Identity at the point of use is ambient.** A zero-adapter control
   showed the frozen base circuit already aggregates entity identity at the
   relation position (probe 0.86–0.90 deep, no consolidation). Round 1's
   apparent "recon-only builds legible structure at the route position" was
   pre-existing circuitry.
2. **Identity at the content position is graded.** The base circuit washes
   name-token identity out with depth (0.89 → 0.31); recon-only
   consolidation half-restores it (→ ~0.60); coherence pins it
   (→ 0.88–0.94), and the effect propagates to intermediate layers
   (blocks 2–3 ≈ 0.99–1.00 vs recon 0.66–0.78; at block 1 both arms are
   high, 0.92 vs 1.00). **The binary claim
   "reconstruction never produces coherence" is wrong; the correct claim:
   coherence is the strong, targeted form of a pressure reconstruction
   exerts weakly and incidentally.**
3. **Facts are linearly localized nowhere, for anyone.** A first fact-probe
   design failed its own pre-registered sanity check (a zero adapter
   "decoded" never-consolidated facts at 0.84 — with 8 entities × 8 colors
   the probe memorizes entity→color and rides identity). The corrected
   probe (train on 32 base entities' known facts, test on unseen new
   entities — only a shared *encoding* can transfer) shows: new-entity fact
   transfer ≈ chance at every site in every arm, AND the base→base ceiling
   is itself low everywhere (0.2–0.5 vs 0.125 chance). Conclusion:
   attribute bindings live in the computation (attention/head pathways),
   not in a linearly readable code in the residual stream — for pretrained
   facts as much as consolidated ones. Additionally, the one weak shared
   code the frozen model had (0.52 at the route position, deep) is
   *halved* by consolidation in both arms — collateral scrambling of the
   base fact-encoding, a second face of the forgetting finding.

**Synthesis for the recut:** what the coherence term demonstrably
manufactures is a durable **type token** — a stable, legible identity
representation at the content's own position, surviving contextualization,
readable by consumers other than the trained route (and by future
consolidation cycles). Facts ride in routes regardless of arm. The paper's
§9 argument ("the same content gets a representational region") is a claim
about exactly this property; the functional-transfer prediction is a
separate claim that keeps failing where tasks have routes.

## Deconfounded k-arm (2026-07-03)

`exp_b_deconfound.py` / `DECONFOUND_REPORT.md`: 128 base entities × 2
families (entities-per-family matched to k=8 at 32). Verdict is graded:
matching data composition closes roughly **8% / 43% / 70%** of the
k=8-vs-k=2 recon-only transfer gap at n_dream = 1 / 2 / 4 (3-seed ratios of
noisy differences — treat as coarse). The round-1 reviewer's
confound (M2) explains most of the effect at n_dream ≥ 2; a residual
k-linked deficit survives at n_dream = 1. Reading: **pretraining context
diversity and dream context diversity are substitutes** — either supplies
the family-general scaffold that new bindings ride on. (Connects to §5.7:
early external diversity buys what later self-generated diversity would
otherwise have to.) 3 seeds, directional. Coherence's representational
advantage persists in this regime (probe 0.867 vs 0.589 at n_dream=4).

## Safeguards vs forgetting (2026-07-03, final — see SAFEGUARDS_REPORT.md)

`exp_c_safeguards.py`, 110 rows, 5 seeds, both λ, calibrated supplement
included. Findings:

- **Homeostatic decay (η=1e-3) is the most efficient safeguard**: +0.12–0.14
  base-fact retention at zero dream-family acquisition cost (its cost lands
  entirely on cross-context transfer, −0.05). Shadow readout (β=0.005)
  stacks approximately additively (+0.09–0.11 more protection for
  −0.04–0.08 eval). Best recipe: decay + slow shadow readout.
- **The trust region, properly calibrated (~20% of steps clipping), shows
  no reliable forgetting protection** (+0.014–0.019, bounded well below the
  +0.05–0.15 a movement-based account predicted). Miscalibrated as a hard cap (100%
  clipping) it "protects" (+0.23–0.28) only by strangling acquisition
  (−0.18–0.39). Reading: forgetting tracks *total adapter movement*, not
  outlier steps. Important nuance for the paper: this does NOT contradict
  the trust region's anti-divergence role (§4.4 sims: every seed diverges
  without it at d ≥ 32) — the two failure modes are different, and the
  four-safeguards framework's division of labor is empirically real:
  trust region prevents blowup; decay and averaging manage forgetting.
- The recon-vs-cohere representational gap survives every safeguard; under
  a binding movement budget (hard-cap trust region), coherence flips to a
  functional cost (eval −0.10, t=−5.3) — coherence spends movement budget.
- Shadow readout at β=0.05 is null here; β=0.005 is where protection lives
  (consistent with the §2.5 monotone-in-β finding from the §4.4 sims).

## Compositional eval (2026-07-03, final — exp_d attempt 3, gate passed)

Attempt 3 (6 layers, 64 base entities → ~4× same-pair coverage) passed the
pretrain gate (held-out base comparison ≈ 0.98–1.00). 15 seeds × n_dream ∈
{2,4} × arms recon/cohere(λ=0.3)/shuffled, paired. Chance = 0.5.

**Verdict: P-D2 — the deflationary pre-registered reading.** Comparison
accuracy on new×new entity pairs is at chance for every arm (recon
0.51/0.55, cohere 0.46/0.49, shuffled 0.51/0.51 at n_dream=2/4), with
heavy diff-bias (same-class accuracy 0.16–0.40): the frozen comparison
circuit consumes pretrained facts (base×base pairs: 0.72–0.85 post-
consolidation) but cannot consume LoRA-consolidated facts *regardless of
representational geometry*. The declarative route reads the same facts at
0.96–0.99 in the same models. **Consolidated memory at toy scale is
route-bound no matter how coherent the representations are: making the
type token legible does not by itself make the fact consumable by circuits
that never trained on the consolidated pathway.** P-D1 is refuted
outright — coherence is slightly *negative* on new-pair comparison
(−0.05/−0.06, t≈−2.4, 6/15 seeds).

Two informative twists:

1. **The shuffled arm significantly protects the base comparison circuit**
   (base×base +0.11/+0.13 over recon, t≈4.4–4.8, 13–14/15 seeds). This is
   the second experiment showing the shuffled-coherence-as-brake pattern
   (first: base_attr in the powered rerun; within this experiment,
   base_attr and comp_bb are correlated metrics from the same runs, so
   they count once). The safeguards grid contains no shuffled arm but
   shows the analogous mechanism: movement-limiting auxiliaries protect
   base circuitry at the cost of new consolidation. Both direct
   appearances share one grammar and one shuffle construction — a
   replication in a different regime would make this a claim; for now it
   is a consistent observation.
2. First statistically significant *functional* coherence gain anywhere in
   the program, though tiny and near ceiling: declarative eval transfer at
   n_dream=2, +0.019, t=6.2, 15/15 seeds (0.961 → 0.980).

Implication for the paper: contribution #2's functional cash-out is now
0-for-2 at toy scale (declarative transfer: null; compositional
consumption: null-to-negative). The representational claim stands on its
own evidence; functional benefit is conjecture pending the real-LM stage,
where the sharpest question is exactly this one: do a real model's richer
in-context comparison circuits consume coherent consolidated
representations, or is route-boundness fundamental to Q/V-adapter
consolidation?

## Compositional eval — decay control (final cell of the program)

`results_d/results_d_decay0.001.json`: the round-3 reviewer's demanded
control. Both arms rerun with homeostatic decay η=1e-3. Decay lifts
base-fact retention (0.53 → 0.62–0.65) and modestly recovers the
comparison circuit (0.75–0.82 vs 0.72–0.76 unprotected; pre-consolidation
0.97+), but new×new comparison stays at chance for both arms and the
cohere−recon difference is null (−0.010 ± 0.119 at n_dream=2, −0.019 ±
0.069 at n_dream=4). Declarative coherence gains trend positive under
decay (+0.020/+0.016, t=1.3/2.0, n.s.) — directionally consistent with
the one tiny significant effect in the main run, still not claimable.

## Compositional eval — design history

`exp_d_compositional.py`: same/diff-color comparison pretrained as a base
capability, dreams declarative-only, so eval has no pre-trainable route.
Two pretrain-gate failures preceded the final run: same-class accuracy
pinned at 0.406 at both 16k steps/30% mixture (4 layers, 32 entities) and
32k steps/50% mixture — the equality-test circuit does not form in a
4-layer model with ~96 ordered same-color training pairs. Attempt 3
(6 layers, 64 base entities → ~4× same-pair coverage) passed the gate at
0.98–1.00; its results are the final section above. The two failures are
themselves a scale finding: compositional circuitry is the binding
constraint on toy-model experimental design here.

## Experiment E — the pivotal real-LM test (Qwen2.5-1.5B-Instruct, 2026-07-04)

First test on a real pretrained instruct model (`exp_e_realm.py`,
`results_e/`): 12 invented entities, facts expressed across 8 English
registers; surgical Q/V LoRA (rank 8, all layers, decay η=1e-3); arms
recon/cohere(λ=0.3)/shuffled, 5 paired seeds; restricted-choice logprob
scoring; composition gate passed at 1.5B (1.00/0.93 with facts in context;
GPT-2 117M–774M and Qwen-0.5B all FAIL the gate — the consumer capability
emerges with scale, which is why toy-scale consumers kept failing us).

**P-P1 — type-token formation REPLICATES on a real LM, strongly.**
Cohere vs recon on held-out registers: probe 0.975 vs 0.721 (t=4.2, 5/5),
retrieval 0.917 vs 0.571 (t=11.6, 5/5), Fisher 0.18 vs 6.03 (t=−14.7).
Specificity: shuffled recovers part of the retrieval gain (+0.21 of
+0.35), cohere beats shuffled on all three metrics (probe +0.09 t=2.6,
fisher −3.3 t=−6.3) — same partial-specificity pattern as the toy.

**P-P2/P-P4 — format-boundness replicates and sharpens; notes dominate.**
The gradient, weights-only: in-format cloze recall **1.000** (diagnostic;
full per-arm supplement running) → QA one format away **0.36/0.24** →
composition **0.50/0.52 = chance**. The same facts supplied as context:
**0.99/0.95** on composition. Storage is perfect; retrieval is bound to
the trained surface forms even at the declarative level — a sharper
version of the toy's route-boundness. With both weights and context,
composition is 0.91/0.88 (the adapter costs a little in-context
performance but does not break it).

**P-P3 — resolved negative at 1.5B.** Coherent type tokens do not unlock
compositional consumption of weight-facts (cohere−recon on weights-only
composition: −0.019/+0.015, n.s.-to-trivial). The interaction the toy
could not test is now tested: representational coherence and functional
consumability remain dissociated on a real model.

**Forgetting: negligible at this scale** (ppl 17.9 → 18.1, all arms) —
24 facts with decay is far below the interference regime Lin et al.
document at ~1K facts; our toy forgetting numbers came from proportionally
much larger injections. Scale-dependence of interference should be stated,
not implied away.

**Reading, per the synthesis (notes_recut_ideas.md):** weight
consolidation on a real LM produces exactly what the dispositional-memory
account predicts — perfectly reliable in-pathway behavior, near-zero
token-like accessibility — and the coherence term manufactures the
representational (type) layer without converting dispositions into
consumable tokens. Facts are the wrong cargo for weights; the
dispositional experiment (exp_f) is where weights get their home game.

## Honest caveats

- Toy scale throughout (vocab ~100, 8-token sentences, 4 layers); both
  regimes constructed so context-invariant structure is learnable by a
  rank-limited adapter.
- The Fisher evaluation metric is the training objective on held-out data
  and the retrieval metric shares its normalized-cosine geometry; the
  ridge probe (un-normalized, linear-separation geometry) is
  objective-distinct but not fully independent — cosine clustering tends
  to induce linear separability — and the functional metric (attribute
  transfer) is fully independent. Conclusions above lean on the probe and
  the functional metric, in that order of representational weight.
- AdamW rather than the paper's plain-GD-with-trust-region; 600 fixed
  consolidation steps; no early stopping.
- For a real LLM, the k=8 arm (strong base abstraction) is probably the
  more realistic regime — which cuts both ways: less functional headroom
  for coherence, but the representational-stability argument (what the
  paper actually builds on) is unaffected.

## Process note

Two adversarial review rounds (round 1 preserved at `REVIEW_round1.md`).
Round 1: I accepted both critical findings — my Option B verdict violated
the pre-registered P-B2 decision rule, and the 3-seed functional gaps were
selection-inflated — plus the probe, shuffled-control, and n_ctx=1 fixes,
all of which ran tonight; the k-arm deconfound (M2) is acknowledged but not
run. The reviewer's skepticism about the functional gaps proved correct in
the powered rerun. Round 2 verified the powered-rerun statistics
cell-for-cell and caught four errors in my revision (an overstated
shuffled-control claim at n_dream=2, a significant forgetting effect at
n_dream=1 that I had reported as null, missing n_dream=1 table rows, and an
analysis-script fix I had marked done that had silently no-opped), plus a
batch of stale numbers — all corrected in this version. Round 2's bottom
line: the load-bearing numbers are real and the revision is safe to present
after those edits.

## Reproduction

```
python3.10 -W ignore exp_a_linear.py                       # ~12 min
python3.10 -W ignore exp_b_transformer.py                  # ~10 min (k=8)
K_BASE_FAMS=2 python3.10 -W ignore exp_b_transformer.py    # ~10 min (k=2)
K_BASE_FAMS=2 python3.10 -W ignore rerun_k2_power.py       # ~35 min (powered)
python3.10 -W ignore analyze_track2.py
```
