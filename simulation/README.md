# Dreaming LoRA — Stability Simulation (§4.4 / §8.1)

Toy simulation of the consolidation dynamics: a rank-constrained adapter
ΔW = BA tracking a drifting low-rank target under noisy gradient updates,
with the paper's four architectural safeguards individually ablatable.
This is the completed empirical program behind §4.4 and §8.1 of the paper.

**[FINDINGS.md](./FINDINGS.md)** is the consolidated results writeup — read
that first. The headline: bounded tracking obtains everywhere
non-pathological; the trust region is unambiguously load-bearing at scale;
the shadow adapter's measurable role is variance reduction at the readout
(not rotational damping, which a single-objective toy cannot exhibit); and
state-dependent dream coupling is substantive (it stabilizes the live
iterate and makes the rank constraint partially self-enforcing).

## Requirements

Python ≥ 3.10, numpy, matplotlib. No torch, no GPU. A full Tier 1 run takes
minutes on a laptop; Tier 3 takes hours (use `--tight-memory`,
auto-enabled at tier 3).

Note: on macOS, numpy 2.x emits spurious `invalid value encountered in
matmul` RuntimeWarnings from BLAS at small ‖ΔW‖; these are suppressed where
they are known to be harmless.

## Running

```bash
python3 simulation.py --smoke                 # sanity check (<1s)
python3 simulation.py --tier 1                # fixed-distribution baseline, d=8/r=2
python3 simulation.py --tier 2 --state-dep --strength 1.0   # state-dependent dreams, d=32/r=4
python3 simulation.py --tier 2 --symmetry-breaking --sb-tau 4   # §5.4 sampler
python3 beta_sweep.py ...                     # shadow EMA rate sweep
python3 maturity_sim.py                       # §5.2 maturity signal diagnostics
python3 rotation_detect.py                    # rotation diagnostics (autocorrelation, PC trajectories)
```

Each run writes plots and a `results*.json` summary to `--out-dir`.

## Scripts

| Script | Purpose |
|---|---|
| `simulation.py` | Main battery: trajectory, dispersion-vs-α scaling, ablations B1/B2/B4, live and shadow readout dispersion |
| `beta_sweep.py` | Shadow EMA rate β sweep for one (tier, regime, α) |
| `maturity_sim.py` | μ_struct / μ_exp trajectories (§5.2) |
| `rotation_detect.py` | Tests for rotational dynamics in the steady-state iterate |
| `t3_extended_scaling.py` | 30k-cycle reruns at small α (tier 3 + state-dep convergence is slow) |
| `shadow_dispersion_check.py` | Early shadow-vs-live diagnostic (superseded by always-on shadow in `simulation.py`) |
| `analyze_t3.py` | Cross-tier summary table generator |

## Results

`results/` contains the raw outputs (plots + JSON) for every run reported
in the paper. Directory map and interpretation are in
[FINDINGS.md](./FINDINGS.md) under "Data files."
