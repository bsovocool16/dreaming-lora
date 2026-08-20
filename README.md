# Dreaming LoRA

**Cross-Session Memory Through Offline Episodic Consolidation**

Claude (Anthropic) and Ben Sovocool

Working Draft — July/August 2026

The below is a project I've been working on to some extent for the past few months. It started with an idea that undeniably was mine (although not unique, following from the evident role of sleep and dreaming in learning), but increasingly it has been a Claude-driven project and now is essentially not my work at all. 

When I started working on this project, I had no issues with the idea that AI would autonomously advance research agendas. The fact that I did not understand the subject matter at all meant that I had no practical choice if I wanted to do anything here, and I felt that if there was any virtue to the idea at all it was net-positive to have Claude handle the elements that were outside of my control.

However, I am increasingly uncomfortable with this project, both because I would consider attributing credit to myself to be borderline fraudulent and because the existence of these "slop papers" is a serious challenge to the project of science. I think that the production of "paper-shaped objects" without comprehension is an extension of the worst attributes of academia, where the myopic focus on publication has contributed to a flood of low-quality, unverifiable and often wholly unimportant papers. Although I continue to believe that AI is a huge net positive for society and for science, I am unsure as to how to reconcile this with concerns around human disempowerment and Goodharting of social structures, especially where I may be a willing perpetrator of each.

I am leaving this repo up and may continue to contribute to it on an occasional basis in the hope that some of its contents may be useful to others. I continue to believe that developing persistent, cross-session memory is critical to advancing AI in the near term (although sufficient training data could theoretically obviate the need for memory), and also is relevant to questions of AI moral consideration and "personhood" to some extent. However, I am prefacing the README to memorialize my concerns and to clarify attribution. The remainder of the README is preserved in full below and discusses the paper project in more detail.

---

## What This Is

Recent work on gradient-based test-time memorization—Titans, Atlas, Memory Caching, unified under the Miras framework—has established that transformer-based language models can be augmented with persistent memory updated via surprise-driven gradient descent during inference. For most of its development, that line of work scoped itself explicitly to within-context operation: memory accumulates structure over long sequences but does not persist across separate conversations.

This paper proposes Dreaming LoRA, an architectural extension addressing cross-session memory through offline episodic consolidation. The architecture maintains the K/V cache as transient contextual memory, adds a LoRA adapter as persistent cross-session memory encoded as structural deformation of attention geometry, and introduces a between-conversation consolidation phase—dreaming—that updates the adapter from self-generated replay content. Memory is altered perception, not stored content; the adapter does not record what happened, it changes how the system processes future experience.

The paper develops the consolidation objective in factor-native LoRA subspace, a composite loss with reconstruction and Fisher-discriminant coherence components, and a stability analysis based on bounded tracking from constant-stepsize stochastic approximation with Markovian noise. The stability simulation—the first element of the empirical program—is complete, and its code and results are included in this repository.

## Three Contributions

**Cross-session persistence requires offline episodic consolidation.** Online updates during forward passes—as in Titans and Atlas—do not address the structural revision that cross-session learning requires. An offline consolidation phase operating on self-generated replay content addresses this; a three-phase developmental trajectory addresses the cold-start pathology of a self-reinforcing consolidation loop.

**Cross-session symbolic operation requires coherence in attentional bias.** Reconstruction-family attentional biases populating the Miras framework preserve local prediction regularities but do not produce cross-context representational consistency. A coherence-family attentional bias rewarding region-level consistency of same-content representations across contexts extends the Miras signature to accommodate cross-context objectives.

**Bounded tracking is the appropriate stability framing for persistent learning systems.** Bounded tracking—ergodic concentration around stable attractors with dispersion scaling with the stepsize—is the appropriate stability target, rather than fixed-point convergence. The framework applies to the full family of gradient-based persistent memory architectures, not only to the proposal developed here.

## Status

The stability simulation (§4.4 / §8.1 of the paper) is **complete**: three adapter scales (d=8/r=2 through d=128/r=8), fixed and state-dependent dream distributions, the §5.4 symmetry-breaking sampler, and a sweep of shadow averaging rates. Headline findings: bounded tracking obtains in every non-pathological configuration; the trust region is unambiguously load-bearing at scale; the shadow adapter's measurable role is variance reduction at the readout (1.3×–9× tighter than the live iterate, depending on the averaging rate), with its rotational-damping role constitutively untestable in a single-objective toy; and state-dependent dream coupling is substantive, stabilizing the live iterate and making the rank constraint partially self-enforcing. Simulation code and results are in [`simulation/`](./simulation/).

The **coherence-loss validation has now been run** at three scales — linear regime, toy transformer, and a pretrained 1.5B-parameter instruct model (code and results in [`coherence_program/`](./coherence_program/)). Headline findings, reported in §8.1: reconstruction-only consolidation yields *route-bound, dispositional* memory (perfect recall inside trained surface forms, chance-level consumption by independent circuits — while the same facts in context score 0.95+); the coherence objective reliably manufactures the missing cross-context *type structure* at ≤5% reconstruction cost, replicating from toys to the real model; and whether type structure unlocks functional consumption by other circuits remains an explicitly open prediction. The safeguard ablations measured the forgetting tradeoffs (homeostatic decay is the efficient mitigation; the trust region's role is divergence, not forgetting). The remaining central experiment — the cross-session benchmark against an online-consolidation baseline — is outlined in §8, with a dispositional-memory evaluation (where parametric memory is predicted to beat context stores) as the next work front.

## Concurrent Work

This revision engages with two concurrent papers titled "Language Models Need Sleep": an anonymous ICLR 2026 submission building on the Miras framework lineage (offline sleep with parameter expansion, knowledge distillation, and a task-reward-validated dreaming stage) and Lee et al. (arXiv:2605.26099; learned offline recurrence consolidating evicted context into fast weights). The first confirms that offline consolidation on self-generated content is where the framework family is heading; §7 of the paper details how the present proposal differs (fixed-capacity stability theory, operation without an external task evaluator, and cross-context coherence as an objective).

## Files

- **[dreaming_lora.pdf](./dreaming_lora.pdf)** — The paper
- **[dreaming_lora.tex](./dreaming_lora.tex)** — LaTeX source
- **[simulation/](./simulation/)** — Stability simulation code, results, and reports
- **[coherence_program/](./coherence_program/)** — Coherence-loss validation program (three scales), safeguards grid, and the pivotal real-LM experiment
- **[CITATION.cff](./CITATION.cff)** — Citation metadata

## Revision History

**July 2026.** Reports first results from the coherence-loss validation program (§3.4, §7, §8.1, §9 recut around the use/reusability distinction: reconstruction installs routes, coherence installs type tokens); adds the continual-PEFT positioning (O-LoRA lineage, Lin et al., Biderman et al.), representation-consistency lineage (Co²L, SDRL, drift compensation), and the migration corollary (Trans-LoRA, PorTAL: the dreaming mechanism doubles as its own migration-corpus generator); releases the full experimental program in `coherence_program/`.

**June 2026.** Integrates the completed stability-simulation program: §2.5 (shadow averaging rate as a tuning knob), §4.3.3 (shadow's two roles separated: measured readout variance reduction vs. theoretically-motivated rotational damping), §4.4 and §8.1 (full multi-scale results, including state-dependent dream coupling and the corrected shadow-ablation methodology), §5.2 (tier-dependent Phase 1 thresholds; fraction-of-saturation rule), §5.6 (empirical confirmation that co-constitutive coupling is substantive). Adds engagement with the two concurrent "Language Models Need Sleep" papers in §1 and §7.

**April 2026.** Substantial revision of the March 2026 draft reflecting engagement with the Miras framework: factor-native consolidation objective, Fisher-discriminant coherence loss, structural/experiential maturity split, bounded tracking as the spine of §4, biological correspondences reframed as supporting rather than load-bearing.

Earlier drafts are preserved in git history.

## How This Was Made

I'm a practicing lawyer, not an academic. This has been an iterative process with Claude (Opus 4.6 for v1, Opus 4.7 for the April revision, Fable 5 for the simulation-integration revision and ongoing empirical work), starting with a philosophical analysis of biological memory and phenomenological reports (i.e., my own self-reporting), and then formalized. Some off-task use of my work Copilot account for review. Comments, criticism, and revisions are 100% welcome.

## Citing This Work

```bibtex
@misc{sovocool2026dreaminglora,
  title={Dreaming LoRA: Cross-Session Memory Through Offline Episodic Consolidation},
  author={Sovocool, Ben},
  year={2026},
  note={Working draft, July 2026},
  url={https://github.com/bsovocool16/dreaming-lora}
}
```

## Contact

Issues and pull requests are open. For substantive feedback, the issue tracker on this repo is the right place.

## License

The paper is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](./LICENSE). Simulation code is provided under the same terms.
