# Dreaming LoRA

**A Biologically-Motivated Memory Architecture for Transformers**

Ben Sovocool · Developed in collaboration with Claude (Anthropic)

Working Draft — March 2026

---

## Summary

Current LLMs have no mechanism for experiential memory that persists across conversations and modifies processing. Context injection gives models a cheat sheet; it doesn't reshape how they think. This paper proposes an architecture that bridges short-term and long-term memory through a consolidation mechanism analogous to sleep.

The core idea: the transformer's K/V cache functions as transient working memory. A LoRA adapter — a small, low-rank modification to the attention weights — functions as persistent long-term memory by deforming *how the model pays attention*, not by storing facts. A "dreaming" process transfers information between the two: after each conversation, the system generates structured variations of surprising content via adapter-biased context transplantation, extracts the consistent gradient signal across those variations via truncated SVD, and applies the result as a small adapter update. Memory becomes altered perception, not stored content.

The consolidation loss is composite: reconstruction (next-token prediction), representational coherence (contrastive alignment across dream contexts), and homeostatic regularization (weight decay enabling rejuvenation). The balance is self-regulated by the spectral energy of the mean gradient modulated by adapter maturity — a design corrected by the infant-sleep inversion, where the biological evidence revealed that the computationally naive balance was wrong.

The paper analyzes stability via two-timescale stochastic approximation, argues for bounded tracking rather than fixed-point convergence, identifies a three-phase developmental trajectory (curriculum → transition → mature consolidation), and derives all scheduling from a single endogenous maturity signal (adapter Frobenius norm).

## Files

- **[dreaming_lora.pdf](dreaming_lora.pdf)** — The paper (v4, 20 pages)
- **[dreaming_lora.tex](dreaming_lora.tex)** — LaTeX source

## What Changed (v4)

Since v3 (dream content generation, developmental phases, pseudocode):

- **Composite consolidation objective (§3.6):** Three-component loss (reconstruction, representational coherence via InfoNCE, homeostatic regularization) with self-regulating balance governed by spectral energy of the mean gradient, modulated by adapter maturity
- **Infant-sleep inversion correction:** Naive spectral-energy regulation makes young adapters reconstruction-dominant; biological evidence that infants have more REM (abstraction) led to maturity modulation λ_recon = μρ/(1+λ_base), highlighted as a case where biology did genuine corrective theoretical work
- **Homeostatic rejuvenation dynamics:** Weight decay provides a slow restoring force — unreinforced adapters gradually become more exploratory, the computational analog of cognitive flexibility during rest
- **Coherence floor (λ_base):** Prevents "perpetual cramming" under sustained moderate novelty
- **Notation and cross-reference fixes:** Equation numbers updated, coherence loss denominator corrected, SVD justification added, base model frozen status clarified
- **Novel contributions merged:** Single maturity signal governing all developmental scheduling (noise, importance weights, salience, loss balance) presented as one unified design principle

## How This Was Made

I'm a practicing lawyer, not an academic. This was an iterative process with Claude Opus 4.6 (and some generous off-task use of my work Copilot account for adversarial review), starting from a philosophical analysis of biological memory and progressively formalizing. The architecture is theoretical — no experiments yet, which is where I run out of road. Comments, criticism, and especially empirical collaborations are very welcome.
