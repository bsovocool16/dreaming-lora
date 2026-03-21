# Dreaming LoRA

**A Biologically-Motivated Memory Architecture for Transformers**

Ben Sovocool · Developed in collaboration with Claude (Anthropic)

Working Draft — March 2026

---

## Summary

Current LLMs have no mechanism for experiential memory that persists across conversations and modifies processing. Context injection gives models a cheat sheet; it doesn't reshape how they think. This paper proposes an architecture that bridges short-term and long-term memory through a consolidation mechanism analogous to sleep.

The core idea: the transformer's K/V cache functions as transient working memory. A LoRA adapter — a small, low-rank modification to the attention weights — functions as persistent long-term memory by deforming *how the model pays attention*, not by storing facts. A "dreaming" process transfers information between the two: after each conversation, the system generates structured variations of surprising content via adapter-biased context transplantation, extracts the consistent gradient signal across those variations via truncated SVD, and applies the result as a small adapter update. Memory becomes altered perception, not stored content.

The consolidation loss is composite: reconstruction (next-token prediction) and representational coherence (contrastive alignment across dream contexts), with the balance self-regulated by the spectral energy of the mean gradient modulated by adapter maturity — a design corrected by the infant-sleep inversion, where the biological evidence revealed that the computationally naive balance was wrong. Homeostatic weight decay is applied directly to the LoRA factors after the SVD update, ensuring it does not compete with experiential signal for the rank-constrained gradient subspace.

The paper analyzes stability via constant-stepsize stochastic approximation with Markovian noise, argues for bounded tracking rather than fixed-point convergence, identifies a three-phase developmental trajectory (curriculum → transition → mature consolidation), and derives all scheduling from a single gauge-invariant maturity signal (effective deformation norm).

## Files

- **[dreaming_lora.pdf](dreaming_lora.pdf)** — The paper (v5, 21 pages)
- **[dreaming_lora.tex](dreaming_lora.tex)** — LaTeX source

## What Changed (v5)

Since v4 (composite consolidation objective, infant-sleep inversion):

- **Gauge invariance fix:** All developmental scheduling (maturity signal μ, sensitivity matrix S_Q, salience weights, noise covariance, importance weighting, phase transitions) now depends on the effective deformation ||BA||_F rather than the individual factor ||A||_F, ensuring the architecture indexes functional properties of the learned memory rather than an arbitrary factorization gauge
- **Homeostatic loss separated from gradient pipeline:** Weight decay is applied directly to LoRA factors after the SVD update, not as part of the composite loss that feeds gradient extraction. This prevents decay gradients from competing with experiential signal for the limited rank-r subspace
- **Two-timescale SA reframed as single-timescale:** The stability analysis now correctly characterizes the system as constant-stepsize SA with Markovian noise and variance-reduced gradient estimation, not two-timescale SA. The prior framing was technically incorrect — dream sampling is batch Monte Carlo, not a second dynamical timescale
- **Rank growth addressed:** The update recursion now includes rank re-projection (SVD of the accumulated adapter after each additive update), with the re-projection itself framed as structured forgetting. Open questions section discusses the cost
- **Gradient materialization cost:** New open question acknowledging that materializing full d_out × d_in gradient matrices circumvents LoRA's memory efficiency, with suggestions for native low-rank or randomized SVD approaches
- **Anti-bias support limitation acknowledged:** The anti-bias correction mines hard cases within the adapted model's active support but cannot audit genuinely avoided regions outside that support. Base-model sampling mixture suggested as mitigation
- **Coherence loss clarified:** Passage-level invariance (not token-level) is the target, preserving legitimate context-sensitive token representations while enforcing consistent processing of salient subsequences as functional units
- **All equation cross-references verified and corrected**

## How This Was Made

I'm a practicing lawyer, not an academic. This was an iterative process with Claude Opus 4.6 (and adversarial review from Claude and GPT), starting from a philosophical analysis of biological memory and progressively formalizing. The architecture is theoretical — no experiments yet, which is where I run out of road. Comments, criticism, and especially empirical collaborations are very welcome.
