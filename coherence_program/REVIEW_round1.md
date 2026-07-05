# Adversarial review, round 1 (agent a2b36706508f84fb7, 2026-06-10 ~02:00)

[Preserved verbatim from the reviewer agent's report. Disposition notes by
Claude follow each finding in brackets.]

**Summary judgment.** Experiment A's core result is solid: the recon-vs-composite retrieval gap at n_ctx=32 is positive in 20/20 seed×ω paired cells (λ=0.3), so "diversity amplifies rather than substitutes" survives scrutiny *in the linear regime*. But the transformer's functional headline (the k=2 "+5–10 points") is statistically indistinguishable from noise, the "Option B refuted in every regime" claim contradicts the experiment's own pre-registered decision rule, and the "representational" metrics are near-circular with the training objective. Several quantitative claims in RESULTS.md are contradicted by the raw JSONs.

## Critical
- **C1. "Option B refuted in every regime" violates pre-registered P-B2** — at k=8/n_dream=4 the gap (+0.04) is under the committed 5-point threshold; Option B is supported functionally in that arm. [ACCEPTED — headline #3 rewritten per-arm and per-level (functional vs representational).]
- **C2. k=2 functional gaps within noise, inflated by max-over-λ selection** (per-seed sign flips; winner's curse at n=3). Fix: 10–15 seeds at fixed λ. [ACCEPTED — 15-seed powered rerun at fixed λ=0.3 ran tonight; the reviewer was right: functional gaps collapsed to null/weak-trend.]

## Major
- **M1. Retrieval metric near-circular with the Fisher loss** (same token, same layer, same normalized-cosine geometry). Fix: loss-independent probe. [ACCEPTED — ridge probe on un-normalized states added in the rerun; representational claim survived it.]
- **M2. k=8 vs k=2 confounds binding entanglement with data composition.** [ACKNOWLEDGED — flagged in RESULTS.md as hypothesis-not-finding; N_BASE-rescaling control named as follow-up, not run tonight.]
- **M3. Forgetting + (1−λ) reweighting contaminate cross-λ comparisons.** Fix: shuffled-label coherence control. [ACCEPTED — control ran in the rerun; shuffled coherence damages function and protects base facts, dissociating coherence from generic regularization in both directions.]
- **M4. "Coherence needs ≥2 contexts" unsupported (A never ran n_ctx=1) and contradicted by B.** [ACCEPTED — headline dropped; n_ctx=1 row run (coherence flat-to-mildly-harmful at one context in A); restated as magnitude pattern.]

## Minor
1. Aggregate ranges contradicted by raw cells (retrieval "0.65–1.00", "≤5%" lm cost, "near chance"). [ACCEPTED — all corrected.]
2. analyze_track2.py never loads the k2 file. [ACCEPTED — first patch attempt silently no-opped (string-match failure); caught by round-2 review and fixed properly, verified to load all three arms.]
3. Unreported negative gaps at ω=1.0/n_ctx=2. [ACCEPTED — sentence added.]
4. "Unconditionally" needs the interpolation qualifier. [ACCEPTED — added.]

Full original text available in the session transcript; the substance is
reproduced above without alteration to the findings.
