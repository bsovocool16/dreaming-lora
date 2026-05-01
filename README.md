# Dreaming LoRA

**Cross-Session Memory Through Offline Episodic Consolidation**

Ben Sovocool · Developed in collaboration with Claude (Anthropic)

Working Draft — April 2026

---

## What This Is

Recent work on gradient-based test-time memorization—Titans, Atlas, Memory Caching, unified under the Miras framework—has established that transformer-based language models can be augmented with persistent memory updated via surprise-driven gradient descent during inference. This line of work has scoped itself explicitly to within-context operation: memory accumulates structure over long sequences but does not persist across separate conversations.

This paper proposes Dreaming LoRA, an architectural extension addressing cross-session memory through offline episodic consolidation. The architecture maintains the K/V cache as transient contextual memory, adds a LoRA adapter as persistent cross-session memory encoded as structural deformation of attention geometry, and introduces a between-conversation consolidation phase—dreaming—that updates the adapter from self-generated replay content. Memory is altered perception, not stored content; the adapter does not record what happened, it changes how the system processes future experience.

The paper develops the consolidation objective in factor-native LoRA subspace, a composite loss with reconstruction and Fisher-discriminant coherence components, and a stability analysis based on bounded tracking from constant-stepsize stochastic approximation with Markovian noise. An empirical program testing specific predictions from each contribution is outlined.

## Three Contributions

**Cross-session persistence requires offline episodic consolidation.** Online updates during forward passes—as in Titans and Atlas—do not address the structural revision that cross-session learning requires. An offline consolidation phase operating on self-generated replay content addresses this; a three-phase developmental trajectory addresses the cold-start pathology of a self-reinforcing consolidation loop.

**Cross-session symbolic operation requires coherence in attentional bias.** Reconstruction-family attentional biases populating the Miras framework preserve local prediction regularities but do not produce cross-context representational consistency. A coherence-family attentional bias rewarding region-level consistency of same-content representations across contexts extends the Miras signature to accommodate cross-context objectives.

**Bounded tracking is the appropriate stability framing for persistent learning systems.** Current work demonstrates empirical stability through careful mechanism design without a theoretical account of what stability means in this setting. Bounded tracking—ergodic concentration around stable attractors with dispersion scaling with the stepsize—is the appropriate target. The framework applies to the full family of gradient-based persistent memory architectures, not only to the proposal developed here.

## Status

The architecture is currently theoretical. An empirical program is outlined in §8 of the paper, with three central experiments: a stability simulation testing the bounded-tracking thesis on a toy adapter, a cross-session benchmark comparing offline consolidation against an online baseline, and a coherence-loss validation isolating the contribution of cross-context representational consistency. The simulation is the immediate next step. Implementation code will appear in this repo when written; the repo will remain paper-only until then.

## Files

- **[dreaming_lora.pdf](./dreaming_lora.pdf)** — The paper
- **[dreaming_lora.tex](./dreaming_lora.tex)** — LaTeX source
- **[CITATION.cff](./CITATION.cff)** — Citation metadata

## Revision History

This is a substantial revision of an earlier (March 2026) draft. The current version reflects engagement with the Miras framework line of work (Behrouz et al., 2024–2026), which had developed gradient-based test-time memorization into a comprehensive within-context framework that the original draft did not adequately position against. Major changes include reformulating the consolidation objective in factor-native LoRA subspace (replacing gradient-space SVD with rank as adapter capacity), replacing the InfoNCE coherence loss with a Fisher-discriminant formulation, splitting structural and experiential maturity into two distinct signals, restructuring the stability analysis around bounded tracking as a single-timescale framework, and reframing biological correspondences as supporting rather than load-bearing. The earlier draft is preserved in git history.

## How This Was Made

I'm a practicing lawyer, not an academic. This was an iterative process with Claude (Opus 4.6 for v1, then primarily Opus 4.7 for the substantial revision), starting with a philosophical analysis of biological memory and phenomenological reports (i.e., my own self-reporting), and then formalized. Some off-task use of my work Copilot account for review. Comments, criticism, and revisions are 100% welcome.

## Citing This Work

```bibtex
@misc{sovocool2026dreaminglora,
  title={Dreaming LoRA: Cross-Session Memory Through Offline Episodic Consolidation},
  author={Sovocool, Ben},
  year={2026},
  note={Working draft, April 2026},
  url={https://github.com/bsovocool16/dreaming-lora}
}
```

## Contact

Issues and pull requests are open. For substantive feedback, the issue tracker on this repo is the right place.

## License

The paper is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](./LICENSE).
