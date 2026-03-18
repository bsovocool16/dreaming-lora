# Dreaming LoRA

**A Biologically-Motivated Memory Architecture for Transformers**

Ben Sovocool · Developed in collaboration with Claude (Anthropic)

Working Draft — March 2026

---

## Summary

Current LLMs have no mechanism for experiential memory that persists across conversations and modifies processing. Context injection gives models a cheat sheet; it doesn't reshape how they think. This paper proposes an architecture that bridges short-term and long-term memory through a consolidation mechanism analogous to sleep.

The core idea: the transformer's K/V cache functions as transient working memory. A LoRA adapter — a small, low-rank modification to the attention weights — functions as persistent long-term memory by deforming *how the model pays attention*, not by storing facts. A "dreaming" process transfers information between the two: after each conversation, the system generates structured variations of surprising content, extracts the consistent gradient signal across those variations via truncated SVD, and applies the result as a small adapter update. Memory becomes altered perception, not stored content.

The paper formalizes the consolidation objective in gradient space (not error space), treats LoRA rank as an epistemic parameter determining how many deficiency modes are closed per cycle, specifies dream content generation as adapter-biased context transplantation with an anti-bias correction for confirmation bias, analyzes stability via two-timescale stochastic approximation, and argues that the correct stability target is bounded tracking rather than fixed-point convergence. A three-phase developmental trajectory (curriculum → transition → mature consolidation) is derived from the architecture's own feedback dynamics.

## Files

- **[dreaming_lora.pdf](dreaming_lora.pdf)** — The paper (17 pages)
- **[dreaming_lora.tex](dreaming_lora.tex)** — LaTeX source

## What Changed (March 2026 update)

Since the initial release, the paper has been substantially expanded based on iterative review:

- **Dream content generation (§5):** Concrete mechanism specified — adapter-biased context transplantation with anti-bias correction, embedding-space structured noise, and maturity-dependent scheduling governed by a single adapter-norm signal
- **Developmental phases (§5.7):** Three-phase trajectory (curriculum → transition → mature consolidation) derived from the cold-start problem in the architecture's self-reinforcing dynamics
- **Pseudocode (Algorithm 1):** Complete consolidation pipeline with SVD-to-LoRA factoring, smooth importance-weight interpolation, and operator-norm clipping
- **Revised §3.5:** Honest separation of what the formalism gives you from what it doesn't on the intensity → rank question
- **Stability sharpened:** SVD discontinuity flagged as specific obstacle; Titans distinction clarified as within-session vs. cross-session
- **Compute cost analysis** added to open questions

## How This Was Made

I'm a practicing lawyer, not an academic. This was an iterative process with Claude Opus 4.6 (and some off-task use of my work Copilot account for review), starting with a philosophical analysis of biological memory and phenomenological reports (i.e., my own self-reporting), and then formalized. Comments, criticism, and revisions are 100% welcome.
