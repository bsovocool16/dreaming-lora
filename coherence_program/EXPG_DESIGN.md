# Experiment G — Workspace access of consolidated memory (DESIGN, not yet run)

**Status: design for methodology review. Instrument is J-lens-INSPIRED, not
a replication — differences are listed and honest naming ("W-lens") is
used throughout.**

## Question

The workspace paper (Anthropic 2026) shows a capacity-limited set of
verbalizable, mid-layer representations that downstream circuits flexibly
consume, with automatic processing running outside it. Our behavioral
results (exp_e) show weight-consolidated facts are operative but not
consumable, while in-context facts are consumable everywhere. exp_g asks
the mechanistic question: **do weight-consolidated facts fail to reach the
workspace (no-broadcast), or reach it and fail of uptake
(broadcast-without-uptake)? And does the coherence arm differ?**

Three pre-registered outcome patterns:
- G-A (no-broadcast): consolidated fact-concepts absent from the workspace
  readout in the weights condition, both arms → route-boundness =
  automatic-tier operation, mechanistically confirmed.
- G-B (broadcast-without-uptake): present for cohere but not recon (or
  present for both) while composition stays at chance → the bottleneck is
  consumption, not access; coherence is buying workspace entry. The most
  novel outcome.
- G-C (instrument failure): the POSITIVE CONTROL fails (see gate) → 1.5B
  may lack a resolvable workspace or our instrument is too blunt; report
  as a scale/instrument finding and stop.

## Instrument: the W-lens (finite-difference linearized broadcast readout)

For hidden state h at layer ℓ, position t: perturb h → h ± ε·ĥ (hook
injection), run the frozen forward pass, take the centered difference of
final-layer states at positions t' ≥ t, average over t'; readout =
softmax(W_U · norm(Δ)) — token-space scores of what this activation is
linearly poised to contribute downstream. Score a concept token c by its
reciprocal rank in this readout (and top-k membership, k=25).

Deviations from the paper's J-lens, stated for the record: (1) we compute
a context-local JVP with tangent = the activation itself, not the
corpus-averaged Jacobian (their E over 1,000 prompts isolates general
disposition; ours reads this context's broadcast — closer to the actual
question "is THIS content in the workspace NOW," but not their object);
(2) finite differences (2 forwards/probe) instead of autograd (MPS
robustness); ε swept {0.05, 0.1, 0.2}·‖h‖ with linearity check (readout
stability across ε). (3) No sparse cone decomposition; membership by
rank/top-k only. If review demands, a corpus-averaged variant over ~200
neutral prompts via accumulated VJP sketches is the upgrade path.

## Design

Substrate/materials/arms: exactly exp_e (Qwen2.5-1.5B-Instruct, 12
entities, 8 registers, recon/cohere arms retrained per seed with the
frozen recipe — paired seeds 0–4; shuffled arm dropped: its behavioral
profile matched recon and doubling probe compute buys little).

Concept tokens: CITIES only (verify each is a single Qwen BPE token with
leading space; drop/replace any that are not — the J-lens family is
single-token-limited and occupations are multi-token).

Probe sites: layers 8–24 of 28 (≈30–85% depth, their workspace band),
positions: name-final token and the question's answer-preceding position,
in three prompt conditions:
1. **context** (facts in prompt, adapter off) — POSITIVE CONTROL and GATE:
   the entity's city token must appear in the workspace readout (median
   reciprocal rank over entities ≥ 0.1, i.e. typically top-10) at fact/
   name positions in the mid-band for ≥ 4/5 seeds. If this gate fails, the
   instrument cannot see even in-context facts and the experiment reports
   G-C (no claims about consolidation).
2. **weights** (adapter on, no facts in prompt) — the measurement.
3. **base** (adapter off, no facts) — NEGATIVE CONTROL/floor: city tokens
   should NOT appear above chance rank for the paired entity (calibrates
   spurious rank from token priors; the paired-entity linkage is what's
   tested, so floor = rank of the TRUE city minus rank distribution of
   the 5 false cities).

Primary metric: workspace presence = median reciprocal rank of the true
city token minus median reciprocal rank of false-city tokens, per (arm,
condition, layer-band, site). Secondary: top-25 membership rate.

Causal follow-up (runs only if presence found in weights condition):
ablate the city-token readout direction (project out W-lens vector) at
the layers where presence peaks, re-measure declarative QA — if QA drops,
the workspace copy is load-bearing; if not, epiphenomenal.

## Pre-registered decision rules

- Gate: as above (context condition, ≥ 4/5 seeds).
- G-A declared iff gate passes AND weights-condition presence < 0.05
  reciprocal-rank margin for both arms at every probed band.
- G-B declared iff gate passes AND any arm shows presence ≥ 0.1 margin in
  ≥ 3/5 seeds at some band (report which arm/band; cohere>recon comparison
  paired as usual).
- No post-hoc site additions: bands and positions above are frozen; any
  exploratory probing is labeled exploratory.

## Cost

Retraining: 2 arms × 5 seeds × ~5 min = ~50 min (MPS). Probing: 2 forwards
per (ε, site, layer, entity, condition) ≈ manageable; batch entities;
estimate 2–4 h total. All local.

---

# REVISION 2 (post methodology review — verdict was RETHINK INSTRUMENT; empirical F1 check run 2026-07-04)

Empirical check of F1 (scale-vs-additive perturbation at layer 14):
downstream |Δ| for scaling-h = 0.33 (mean over t'>t) vs 0.57 for
matched-norm additive vs 0.0000 noise floor (fp32 MPS deterministic).
The reviewer's conclusion holds in softened form: scaling leaks through
residual reweighting but is uninterpretable as content broadcast and the
same-position term dominates 10×. All eight required changes adopted:

1. **Instrument (replaces W-lens tangent):** PATCH-INJECTION probe. The
   injected tangent is the fact-bearing activation difference
   Δ_fact = h(context condition) − h(base condition) at the same (layer,
   name-position), applied ADDITIVELY in the weights/base conditions;
   readout = W_U · norm applied to the resulting final-layer differences.
   This asks "does the network broadcast fact-bearing content from this
   site" with a tangent that is meaningful by construction. The
   corpus-averaged VJP sketch (~200 neutral prompts) is now the MAIN
   upgrade path if patch-injection proves noisy, not a contingency.
2. **Position split mandatory:** all metrics reported separately for
   t'=t (direct/logit-lens component) and t'>t (broadcast component);
   workspace claims rest ONLY on t'>t.
3. **Noise floor protocol:** ε=0 duplicate forwards per batch; cells with
   ‖Δ‖ below floor are discarded, never normalized. fp32 probes.
4. **Gate redesigned:** measured at the answer-preceding position of a QA
   prompt in the context condition (where exp_e proves the fact is used),
   t'>t component, counterbalanced margin ≥ 0.1, PLUS mismatched-context
   control (entity A prompt containing entity B's city): true-binding
   margin must significantly exceed lexical-presence margin, else gate
   fails as copying-confounded. ≥4/5 seeds.
5. **Within-token counterbalancing:** each city token scored at entities
   where it is true vs entities where it is false; margins are
   within-token, cancelling token-prior confounds exactly.
6. **G-B decontamination:** (i) pre-registered logit-lens baseline — the
   claimed presence must exceed the plain final-layer logit-lens effect
   for the same (arm, entity); (ii) SHUFFLED ARM REINSTATED for exp_g
   (coherence geometry with scrambled bindings — the control that
   separates "coherence-shaped states" from "correct-binding broadcast");
   (iii) mid-band + t'>t required.
7. **Causal follow-up redesigned:** clamping-to-base-value (not
   projection) at mid layers at the NAME position, effect measured at the
   answer position, with matched-norm random-direction and
   false-city-direction controls.
8. **Layer-signature pre-gate:** kurtosis/autocorrelation/effective-
   dimensionality profiles over 200 neutral prompts BEFORE anything else,
   splitting G-C into G-C1 (1.5B lacks workspace signatures — publishable
   scale finding) and G-C2 (instrument blunt — no claim).

Decision rules replaced by the reviewer's sharpened set (gate as item 4;
G-A requires demonstrated noise floor < the 0.05 margin equivalent;
G-B requires mid-band, t'>t, > logit-lens baseline, absent in shuffled
scrambled pairings, ≥3/5 seeds).

---

# REVISION 2.1 (pre-registered 2026-07-09, before the full run; smoke run of the Revision-2 instrument complete)

Smoke findings motivating this revision (seed 0, layers {12,20}, smoke
training): the redesigned gate is STRONG at 1.5B (answer-pre
counterbalanced margin 0.837 vs the 0.1 threshold; mismatched-context
control -0.0002, paired t=5.47) and the noise floor is exactly 0 (fp32
MPS inference is deterministic). But delta_fact INJECTION at the name
site reads ~0 despite large propagated |Δ| — single-site additive
patching plausibly cannot recreate fact transport that runs through
attention TO the context tokens themselves, i.e. the injection
instrument may be blunt regardless of content. Adopted changes:

1. **M2 — natural adapter delta (new PRIMARY presence measurement).**
   Per (seed, arm): the QA prompt (no facts in context) is run twice,
   adapter ON vs adapter OFF; Δ_ℓ = h_ℓ,on − h_ℓ,off at the
   answer-preceding position, for ℓ ∈ {8,12,16,20,24} ∪ {26, final
   (pre-final-norm output of layer 27)}; lensed through the model's own
   final RMSNorm applied to the difference (D2 readout); scored with the
   standard within-token counterbalanced margin. This is the analog of
   the gate's natural Δ — the instrument the gate itself validated.
   Same-cell logit-lens baseline = the same lens applied to the RAW
   adapter-on state at the same site. Injection probes are retained as
   SECONDARY.
2. **G-A redefined:** gate pass AND M2 city-margin < 0.05 at every
   mid-band cell (probed layers 8–20, recon and cohere arms, all seeds).
   Late-only signal (ℓ ≥ 24, 26, final) is consistent with G-A and is
   interpreted as direct-output-pathway (automatic-tier) operation.
3. **Injection insensitivity clause:** if delta_fact INJECTION margins
   are ≈0 everywhere (max |margin| < 0.05 over all delta_fact injection
   cells), injection results are declared instrument-insensitive and
   carry no weight in the verdict — a null injection result then says
   nothing about broadcast.
4. **G-B redefined:** gate pass AND some (arm ∈ {recon, cohere},
   mid-band layer) with M2 margin ≥ 0.1 in ≥ 3/5 seeds, each such seed
   also exceeding the same-cell logit-lens baseline. The COHERE-SPECIFIC
   version of G-B additionally requires cohere > shuffled AND cohere >
   recon on the per-seed mid-band-mean M2 margin (paired across seeds,
   one-sided p < .05).
5. **Shuffled-control ambiguity resolved:** the shuffled arm's LM loss
   still stores TRUE facts (only the coherence grouping is scrambled),
   so "absent in shuffled" is not a well-formed requirement for presence
   per se. Shuffled is reported both ways, and its pre-registered
   control role is the cohere-vs-shuffled comparison in item 4.
