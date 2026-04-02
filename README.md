# Dreaming LoRA

**A Biologically-Motivated Memory Architecture for Transformers**

Ben Sovocool · Developed in collaboration with Claude (Anthropic)

Working Draft — March 2026

---

## Summary

Current LLMs have no mechanism for experiential memory that persists across conversations and modifies processing. Context injection gives models a cheat sheet; it doesn't reshape how they think. This paper proposes an architecture that bridges short-term and long-term memory through a consolidation mechanism analogous to sleep.

The core idea: the transformer's K/V cache functions as transient working memory. A LoRA adapter — a small, low-rank modification to the attention weights — functions as persistent long-term memory by deforming *how the model pays attention*, not by storing facts. A "dreaming" process transfers information between the two: after each conversation, the system generates structured variations of surprising content via adapter-biased context transplantation, extracts the consistent gradient signal across those variations via truncated SVD, and applies the result as a small adapter update. Memory becomes altered perception, not stored content.

The experiential consolidation loss is composite: reconstruction (next-token prediction) and representational coherence (contrastive alignment across dream contexts), with the balance self-regulated by the spectral energy of the mean gradient modulated by adapter maturity — a design motivated by the infant-sleep inversion and independently justified by the cold-start dynamics of the architecture. Homeostatic weight decay is applied directly to the LoRA factors after the SVD update, ensuring it does not compete with experiential signal for the rank-constrained gradient subspace.

The paper analyzes stability via constant-stepsize stochastic approximation with Markovian noise, argues for bounded tracking rather than fixed-point convergence, identifies a three-phase developmental trajectory (curriculum → transition → mature consolidation), and derives all scheduling from a single gauge-invariant maturity signal (combined effective deformation norm across Q and V projections).

## Files

- **[dreaming_lora.pdf](dreaming_lora.pdf)** — The paper (v5.2, 23 pages)
- **[dreaming_lora.tex](dreaming_lora.tex)** — LaTeX source

## What Changed

### v5.2 — Third review response (claims calibration)

- **SVD caveat upgraded:** Now explicitly positioned as the least principled component of the pipeline — curvature-blind, chosen for simplicity/interpretability over optimality. Hessian/Fisher alternatives named concretely
- **Coherence loss limitation acknowledged:** InfoNCE's tendency toward context-insensitivity in the low-temperature limit is now stated as a known risk, not defended away. Variance-bounded and Fisher discriminant alternatives suggested
- **Stability analysis scoped as aspirational:** Safeguards are necessary conditions argued by analogy; joint sufficiency is undemonstrated. GAN analogy noted as cutting both ways
- **μ limitation sharpened:** "Known architectural weakness" — narrow over-commitment and broad calibration are indistinguishable. Spectral maturity signal would require redesigning §5.2/5.5
- **Biological correspondences pruned:** Table cut from 16 to 11, removing purely metaphorical mappings (shadow adapter/character, embedding noise/stochastic firing, trust region/trauma, rank/synaptic modification). Caption explains omissions
- **Infant-sleep claim softened:** "Biology suggested the correction; cold-start dynamics justify it independently" — throughout §6, novel contributions, and conclusion
- **New failure modes in §8:** Chimeric sequence construction underspecification, adversarial/garbage input vulnerability, multi-layer consolidation considerations

### v5 / v5.1 — First and second review responses

- **Gauge invariance fix:** All scheduling depends on ||BA||_F not ||A||_F
- **V projection incorporated:** μ and noise use combined Q+V deformation norm
- **Homeostatic loss separated** from gradient pipeline
- **Two-timescale SA reframed** as single-timescale with Markovian noise
- **Rank growth addressed** via periodic re-projection
- **ρ circularity resolved** via one-cycle lag with explicit initialization
- **Anti-bias correction reframed** as stabilized hard-example mining heuristic
- **All equation cross-references verified**
- **Sleep-time compute** (Lin et al. 2025) cited and distinguished

### v4 — Composite consolidation objective

- Three-component loss (reconstruction, coherence, homeostatic) with self-regulating balance
- Infant-sleep inversion correction
- Homeostatic rejuvenation dynamics
- Coherence floor (λ_base)

## How This Was Made

I'm a practicing lawyer, not an academic. This was an iterative process with Claude (Anthropic) and adversarial review from multiple LLMs, starting from a philosophical analysis of biological memory and progressively formalizing. The architecture is theoretical — no experiments yet, which is where I run out of road. Comments, criticism, and especially empirical collaborations are very welcome.
