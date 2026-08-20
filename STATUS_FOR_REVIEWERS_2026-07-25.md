# Dreaming LoRA — Status Brief for Reviewers

**Date: 2026-07-25.** Purpose: fix the review target and prevent staleness.
Anything not listed as CANONICAL below is either superseded or in flight.

## Canonical review targets (stable, reviewable now)

1. **Paper:** branch `v2026-06-10` at github.com/bsovocool16/dreaming-lora
   (three commits: June revision, July recut, workspace connection).
   `main` still shows the April version and is **stale** — do not review it.
   Note: a full structural rewrite is planned once the last experiment
   lands; prose-level comments may be better saved for that, but
   claim-level and technical criticism of the current branch remains fully
   useful and will carry over.
2. **Evidence record:** `coherence_program/RESULTS.md` in the same branch
   (mirrored locally at `track2_coherence/RESULTS.md`) — the authoritative
   findings document: six headline results, per-experiment sections
   (A/B/maps/deconfound/safeguards/compositional/E/G), pre-registered
   predictions with outcomes including failures, and the record of three
   adversarial review rounds. All statistics were verified by hostile
   recomputation from raw JSONs (round 3).
3. **Key verdicts, one line each:**
   - Type tokens: the coherence loss reliably creates cross-context entity
     representations (replicated linear → toy transformer → Qwen2.5-1.5B;
     large effects, 5/5 seeds).
   - Route-boundness: weight-consolidated facts are perfectly recallable
     in trained formats, near-floor one format away, chance at
     composition; the same facts in context score 0.95+ everywhere.
   - Mechanistic completion (Experiment G): the adapter's contribution to
     mid-layer "workspace-band" states is machine-zero; only a late-layer
     output-pathway trace exists. Consolidated memory operates at the
     automatic tier; coherence does not buy workspace entry.
   - Functional benefit of coherence: null at tested scales — an explicit
     open prediction, not a claim.
   - Forgetting: reproduced and scale-dependent; homeostatic decay is the
     efficient mitigation; calibrated trust region is null for forgetting
     (its role is divergence prevention).
4. **Literature audit:** `LIT_AUDIT_2026-07-03.md` (verified citations; O-LoRA/Merge-before-
   Forget/Lin/Biderman/Co2L/SDRL/Trans-LoRA positioning; note the Biderman
   weight-decay claim was REFUTED in verification and is deliberately not
   cited for that).

## In flight (do NOT treat as findings yet)

**Experiment F (dispositional memory — "weights on their home field").**
Tests whether consolidated demonstrated-never-stated behavioral profiles
beat an honest note-taking pipeline, with a tier-of-verbalizability ×
condition design, note-length and demo-count sweeps, dual cross-family
LLM judges, foil-bundle and minimal-pair controls, frozen pre-registered
decision rules (`track2_coherence/EXPF_DESIGN.md`, Revisions 1–2). Status:
pipeline running on local hardware; demonstrations, consolidations
(including sweeps), and most generations complete for 8 seeds; note
generation and judging in progress. Projection ~3–7 days from 07-25.
Operational history (two OOM crashes, a silent phase under-delivery, and
their fixes) is logged and will be reported in the writeup; the
pre-registered protocol has not been altered.

Known interpretive caveats already on file for Experiment F (reviewers
should expect these addressed in the writeup): low demonstration
clean-yield (0.10–0.19 vs V/S checkers), imperfect pilot compliance for
unbounded description-following (0.50–0.78), near-zero constraint coverage
in early note drafts (notes-arm strawman risk — the design's published-
notes audit adjudicates), and Phase D (analysis) being the least-exercised
code path.

## Pending decisions / next steps

- Full paper rewrite (de-novo structure around the routes/type-tokens/
  dispositional-memory results) begins when Experiment F lands; it will
  absorb reviewer comments on the current branch.
- `main`-branch update: pending author decision.
- Candidate follow-on work, designed but not run: epiplexity-weighted
  dream salience (after Zhang & Levin 2026), intermediate-layer coherence
  variant (motivated by the workspace-band geography).

## Process notes reviewers may want

Every experiment in the program was pre-registered (predictions in module
docstrings before first runs) and passed through independent adversarial
review before results were accepted; several headline-shaping errors were
caught this way (a pre-registration violation, an invalid probe design, a
degenerate instrument). Fixed-λ/fixed-rule analysis throughout; the one
early winner's-curse result was retracted and is documented. Raw results
(JSON) and all code ship in the repository (`simulation/`,
`coherence_program/`).
