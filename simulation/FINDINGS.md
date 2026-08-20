# Dreaming LoRA — Simulation Findings Synthesis

Consolidated writeup of the §4.4 simulation work, suitable as a
single-document handback to the paper. Supersedes the layered prior
reports (`STATE_DEPENDENT_REPORT.md`, `EXTENSIONS_REPORT.md`,
`READOUT_METRIC_REPORT.md`, `SHADOW_SCALING_NOTES.md`,
`T3_READOUT_RESULTS.md`, `BETA_AND_EXTENDED_REPORT.md`), which remain
on disk as the chronological record.

## Scope

Three dimensions of the §4.4 toy simulation, beyond the original
fixed-distribution baseline:

- **State-dependent dream sampler** (§5.6 coupling): dream covariance
  `Σ = σ²I + γ·ΔWᵀΔW`, parameterized by γ. The covariance is rank-≤R
  by construction, capturing the §5 mechanism that dream samples
  concentrate in the adapter's receptive field.
- **§5.4 symmetry-breaking importance reweighting**: oversample
  candidates, weight by `(L_θ/L_W)^(1/τ_sb)`, resample. Tests whether
  the "audit" mechanism produces rotational dynamics.
- **Shadow-readout-aware metric**: shadow EMA always computed,
  `final_window_dispersion_shadow` reported natively alongside live's
  dispersion. The right comparison for what shadow actually does.

Runs at three tiers (T1: d=8/r=2, T2: d=32/r=4, T3: d=128/r=8) across
multiple γ values (0, 0.3, 1.0, 3.0), three SB temperatures (∞, 4, 1),
and seven β values (5×10⁻⁴ to 5×10⁻¹). Total compute: ~6 hours wall
across two machines.

## Summary in one paragraph

Shadow readout dispersion is substantively lower than live readout
dispersion in every regime tested, by a factor that ranges from ~1.1× to
~9× depending on (α, D, γ, β). The default β=0.05 from the paper is
substantively suboptimal in stationary regimes — much smaller β gives
much more smoothing, with no peak in the tested range. The §4.4 B3
ablation as written ("remove shadow, measure live dispersion") cannot
detect what shadow does, because shadow has no feedback into the live
trajectory; the right comparison is shadow's own dispersion vs live's.
State-dep dream coupling at high D caps shadow's smoothing ceiling
because rank-R noise confinement limits the effective dimensionality
that shadow can average over. None of the proposed §4 safeguards have
been falsified; the toy regime simply does not produce the rotational
dynamics that shadow is theoretically meant to damp, so shadow's role
in the toy is purely variance reduction at the readout.

## Findings ordered by paper-section impact

### §2.5 — Shadow Adapter

**Status: needs reframing.** Currently presents shadow as Polyak–Ruppert
averaging at fixed β=0.05. Empirically:

| Tier / regime | β=0.05 (default) | β=0.005 | β=0.002 (best feasible) |
|---------------|-------------------|---------|--------------------------|
| T2 fixed α=0.03 | 1.63× | 4.71× | 9.06× |
| T2 fixed α=0.10 | 2.58× | 8.68× | — |
| T3 fixed α=0.10 | 2.07× | 6.28× | — |
| T3 sd γ=1.0 α=0.10 | 1.45× | 2.31× | — |
| T3 sd γ=1.0 α=0.01 (30k) | ~1.10× | 1.95× | 2.35× |

Ratio is `live_dispersion / shadow_dispersion`. Shadow benefit
**monotonically increases as β decreases** across [0.005, 0.5]. No peak
at β ≈ α (the matched-timescale intuition is wrong: the toy attractor is
stationary, so slower EMA → better averaging without lag-bias).

A finite-cycle peak at β ≈ 2 × 10⁻³ in the simulation is a
*burn-in equilibration artifact* (the EMA needs 1/β cycles to forget
initialization), not a real dynamical limit. In continuous deployment,
there's no burn-in cost, and β should be set by adaptation-speed
requirements alone.

**Recommended rewrite for §2.5:** β is a tuning knob trading off
smoothing strength against adaptation speed. The paper's prior
"40% reduction" claim corresponds to β=0.05 — a *conservative* choice.
Substantially better smoothing is available when the user can tolerate
slower adaptation. The bounded-tracking guarantee on the readout is
much tighter than on the live iterate.

### §4.3.3 — Shadow as Rotational Damping

**Status: empirically not testable in this toy.** The paper's argument
cites Mescheder et al. on GAN training, where discriminator and
generator have opposing objectives and the gradient field has complex
eigenvalues at fixed points. The §4.4 toy has a single descent
objective — minimize `||(ΔW − W*)h||²` — so the structural condition
for rotation is absent.

Direct rotation diagnostics (`rotation_detect.py`) confirm: no
orbital structure in 2D PC projections, autocorrelation of the
steady-state iterate is monotonically positive at short lags, no
spectral peak distinguishable from 1/f noise. The shadow trajectory
is bit-identical with and without shadow EMA — confirming shadow has
no feedback into live by construction.

In the toy, shadow does **variance reduction at the readout** — a
substantive and quantifiable effect — but not rotational damping. The
paper should split these two roles. The variance-reduction claim is
empirically supported and well-characterized; the rotational-damping
claim is theoretically plausible for deployment but not tested by the
§4.4 toy.

### §4.4 — Ablation Battery

**Status: requires structural revision.** As currently written, the
B3 ablation removes the shadow EMA and measures the live trajectory's
dispersion. But shadow has no feedback into live, so this measurement
cannot detect shadow's role. The B3 ablation is constitutively null
in the toy and would be in deployment too.

Replace with: compute both `live_dispersion` and `shadow_dispersion`
in every run, and report the live/shadow ratio as the shadow-ablation
metric. This is implemented in the updated simulation.

The other ablations behave as expected, with refinements:

- **B1 (insufficient samples)**: inflation 1.3–1.5× across all tiers
  and regimes; ratio invariant to γ at T2. Confirms variance reduction
  in gradient estimation requires adequate batch size.
- **B2 (no trust region)**: 0/10 diverged at T1, 10/10 at T2, 20/20 at
  T3. Trust region is load-bearing at high D regardless of dream
  regime, as predicted.
- **B4 (no rank constraint)**: positive inflation 1.2× at T2 fixed,
  but *negative* inflation 0.91 at T2 sd γ=1.0 (and 0.77 at γ=3.0).
  Under state-dep dreams, the unconstrained adapter self-regularizes
  to effective rank R because dream noise is rank-R confined. This
  is a substantive finding for §5: the rank constraint becomes a
  capacity bound rather than a directional-complexity constraint when
  dream noise inherits the adapter's rank structure.

**Recommended panel structure** in the paper:

1. B1 — dispersion vs cycle, n=32 vs n=4
2. B2 — `‖ΔW‖` vs cycle, with/without trust region
3. B4 — effective rank vs cycle, with/without rank constraint, plus
   a note that this panel is regime-dependent
4. **Shadow vs live readout dispersion** (replaces former B3 panel) —
   trajectories of `dispersion_curve_live` and `dispersion_curve_shadow`
   overlaid, with the seed-averaged ratio in the title

### §4 Bounded-Tracking Thesis

**Status: confirmed across the parameter space, with refined scaling.**
The bounded-tracking signature (iterates concentrate around a stable
attractor with dispersion scaling with stepsize) obtains under all
non-pathological configurations.

**Live readout scaling**: dispersion ∝ α^p with p ≈ 0.5 at small α,
growing toward p ≈ 0.6–0.7 at large α and large D. The "broken √α"
finding at T3 from the first round of reports was real (a large-α
correction) but the small-α part of that finding was a transient
artifact — α ≤ 0.02 at T3 + state-dep needs > 30k cycles to converge.

**Shadow readout scaling**: dispersion grows much more slowly with α,
~ α^0.2 at moderate β=0.05 down to ~ α^0.0 (essentially flat) at small
β=0.005. Shadow effectively *decouples readout dispersion from
stepsize* in benign regimes. This is the substantive deployment
implication.

**Convergence-time caveat at high D + state-dep**: at T3 with γ ≥ 1,
the iterate needs > 30k cycles to reach steady state at α ≤ 0.02.
This is not a failure of bounded tracking — no seeds diverge — but
indicates that the asymptotic rate is dimension-dependent in
state-dep regimes. Worth noting in §4 as a practical observation
about convergence time vs the simpler 1/α rule.

### §5.2 — Maturity Signals

**Status: behave as designed, with one calibration warning.**
`maturity_sim.py` instruments μ_struct (participation ratio of S_Q) and
μ_exp (cumulative-consolidation exponential) across all three tiers.
Both signals are monotone, well-bounded, and saturate as expected.

The Phase 1 exit threshold (currently flagged as an open question
in §5.7) is **tier-dependent**: μ_struct=0.5 is crossed at cycle 1
(T1), cycle ~100 (T2), and cycle ~5 (T3). A fixed threshold like
"exit at μ_struct ≥ 0.5" produces wildly different behavior across
ranks. Recommend a *fraction-of-saturation* rule (exit when μ_struct
reaches X% of its asymptotic value) — though identifying the asymptote
in a deployed system without ground truth is itself non-trivial.

### §5.4 — Symmetry-Breaking Sampler

**Status: implementable in closed form for the toy; not productive of
new dynamics.** The §5.4 audit mechanism reweights candidate contexts
by `(L_θ/L_W)^(1/τ_sb)`, upweighting those the adapter struggles with.
Tested at T1 and T2 with τ_sb ∈ {1, 4}.

The reweighting does what it's designed to do — concentrates the
sampling distribution on residual-large contexts. But it does not
produce the rotational dynamics shadow would need to damp. As
ΔW → W*, L_θ → 0 uniformly and the reweighting vanishes; no min-max
structure emerges because there's no actual two-player objective.

Empirically, B3-style ablations under §5.4 reweighting at every
τ_sb still give live/shadow ratio ≈ 1.0 (shadow no effect on live)
and shadow's own dispersion still shows the same ~40% reduction. The
§5.4 mechanism doesn't change what shadow does in the toy.

This doesn't refute the §5.4 design — the paper's argument is about
deployment with actual generative pipelines, which the toy can't
emulate. It does say the §4.4 toy cannot serve as evidence that §5.4
produces the structural conditions for rotation.

### §5.6 — Co-Constitution and State Dependence

**Status: state-dep coupling produces several substantive effects, all
confirmed across γ ∈ {0.3, 1.0, 3.0}.**

1. **Stabilization**: state-dep noise *reduces* live dispersion
   compared to fixed-distribution dreams at every tier. The mechanism
   is gradient preconditioning — structured covariance aligns with the
   adapter's sensitive directions, so gradient signal is more
   informative per sample.
2. **B4 self-regularization** (above): unconstrained adapter collapses
   to effective rank R, with the effect γ-monotone (B4/Main = 1.18 →
   1.07 → 0.91 → 0.77 across γ = 0, 0.3, 1.0, 3.0 at T2).
3. **Shadow ceiling**: state-dep at high D caps shadow's smoothing at
   ~2.3× even with optimal β, because the iterate is rank-R confined
   and shadow can only average across that limited subspace.
4. **Slow convergence at small α**: at T3 + state-dep, α ≤ 0.02 needs
   > 30k cycles to reach steady state — much slower than the 1/α rule
   would predict.

All four are direct consequences of the rank-R structure that
state-dep dream covariance imposes on the iterate's noise. They
empirically validate §5.6's argument that state-dep coupling is
substantive (rather than cosmetic), while clarifying which effects
go in which direction.

## Methodology

The simulation script `simulation.py` was extended along five
backward-compatible axes:

1. **State-dep sampler** (`Σ = σ²I + γ·ΔWᵀΔW`), Cholesky-based, with
   FP-flag suppression for the spurious "invalid" warnings emitted by
   numpy 2.x on macOS BLAS at small ΔW magnitudes. Pre-sampling
   divergence cap to handle B2 + state-dep numerical failure modes.
2. **§5.4 symmetry-breaking reweighting**: oversample factor 4,
   numerically stable log-weight normalization, optional τ_sb tuning.
3. **Shadow EMA always computed**: the `use_shadow` flag is retained
   for backward compatibility but no longer gates computation. Shadow's
   own final-window dispersion is reported natively as
   `final_window_dispersion_shadow`.
4. **B3 dropped from `run_ablations`**: redundant under always-on
   shadow.
5. **β as CLI argument** (via `beta_sweep.py`): module-level
   `SHADOW_BETA` is overridable per-run.

Supporting scripts at project root:
- `simulation.py` — main simulation with all the above changes
- `maturity_sim.py` — §5.2 diagnostic over a single trajectory
- `shadow_dispersion_check.py` — early shadow-vs-live diagnostic (now
  subsumed by always-on shadow in simulation.py)
- `rotation_detect.py` — autocorrelation + PC trajectory diagnostic
- `t3_extended_scaling.py` — small-α reruns at 30k cycles to escape
  T3 + state-dep transient
- `beta_sweep.py` — β sweep across one (tier, α, regime) tuple
- `analyze_t3.py` — cross-tier table generator

Total wall time across all runs: ~6 hours, split across the
original 8-core Mac (memory-constrained, had to delegate T3 via
`t3_portable.tar.gz` bundle) and a 12-core Mac mini.

## Data files

All raw results are in `dreaming_lora_handoff/`:

```
sim_tier{1,2,3}/                        Original fixed-dist baseline + state-dep extension
sim_tier{1,2}_sb*/                      §5.4 SB sampler runs at T1, T2 (τ=4, τ=1)
sim_tier2_gamma_sweep/                  γ ∈ {0.3, 3.0} at T2 (γ=1.0 already in sim_tier2/)
sim_tier{1,2,3}_readout/                Readout-metric runs (new schema, always-on shadow)
sim_tier3_extended/                     30k-cycle T3 fixed scaling rerun (small α)
sim_tier3_extended_readout/             30k-cycle T3 sd γ=1.0 scaling rerun
maturity_sim/                           §5.2 trajectory diagnostic at T1/T2/T3
shadow_check/                           Early shadow-vs-live spot check (T1, T2)
rotation_check/                         AC + PC trajectory analysis
beta_sweep/                             β ∈ [0.005, 0.5] at four (tier, regime, α)
beta_sweep_low/                         β ∈ [5e-4, 5e-3] at T2 fixed α=0.03
beta_sweep_ext/                         β sweep at T3 sd γ=1.0 α=0.01, 30k cycles
```

Each directory contains a `results*.json` (numerical summary) and
typically 5 plots (`*_01_*` through `*_05_*` for the standard battery,
or single `*.png` for the focused experiments).

## Headline numbers for the paper

If the paper rewrite wants three numbers per claim:

- **Shadow reduces readout dispersion**: ~40% at default β=0.05 across
  fixed regimes; up to ~90% at best feasible β=0.002.
- **Live readout dispersion scales as α^p**: p ≈ 0.5 (theory match) at
  small α and small D; p ≈ 0.7 at large α and d=128.
- **Shadow readout dispersion is much flatter in α**: p ≈ 0.2 at
  default β; ~0 at small β. Shadow effectively decouples dispersion
  from stepsize.
- **State-dep dreams stabilize the live iterate** (12% lower
  dispersion at T2 vs fixed at default α=0.03), with the effect
  γ-monotone (γ=3.0 gives 20% lower).
- **State-dep dreams cap shadow's smoothing ceiling at ~2.3× at
  T3** (rank-R bottleneck), down from T3 fixed's 6.3× ceiling.
- **Default β=0.05 is conservative**; β=0.002 gives 5–6× more shadow
  reduction at T2 fixed and is the best feasible in a finite
  simulation. Real deployment can tolerate smaller β.

## What's left undone (out of scope for this pass)

1. **Coherence loss (§3.4)** in a toy regime where "content" has
   identity across "contexts." The current linear-regression toy
   doesn't naturally support this. Would require constructing a
   regime where the dream distribution has compositional structure.
2. **Phase 1 / curriculum (§5.7)** — cold-start dynamics. Maturity
   signal infrastructure is in place; need to add explicit external
   supervision and measure receptive-field bias.
3. **Toy transformer** — moving from this synthetic linear-regression
   adapter toward a small actual transformer with LoRA. Out of scope
   for numpy/no-torch environment.
4. **Rotational dynamics test** — the toy can't produce them; a system
   with explicit two-player gradient structure (e.g. discriminator
   + generator, or adapter + samplers with slow-timescale decoupling)
   would be needed. Different paper.

## File map for the paper's revision

The current `dreaming_lora.tex` would benefit from edits at:

- §2.5 (Shadow Adapter): note β as tuning knob, default is
  conservative, monotone smoothing-vs-β in stationary regimes
- §4.3.3 (Shadow safeguard): split into "variance reduction at
  readout" (empirically supported) + "rotational damping"
  (theoretically motivated, not testable in the toy)
- §4.4 (Empirical Validation): change B3 ablation; new panel
  structure; report shadow-readout dispersion as the headline
  bounded-tracking metric
- §5.2 (Maturity signals): note Phase 1 threshold should be
  tier-aware, recommend fraction-of-saturation rule
- §5.6 (Co-constitution): cite the γ-monotone B4 self-regularization
  and the rank-R confinement of shadow's smoothing ceiling as
  empirical evidence

The rest of the prose stands.
