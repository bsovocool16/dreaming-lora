# Coherence-Loss Validation Program (§3.4 / §8.1)

The complete empirical program behind the paper's §8.1 "Coherence loss
validation (first results)": three scales (linear regime, from-scratch toy
transformer, pretrained Qwen2.5-1.5B-Instruct), the layer×position
coherence maps, the safeguards-vs-forgetting grid, the deconfound control,
the compositional evaluations, and the pivotal real-LM run.

**[RESULTS.md](./RESULTS.md) is the authoritative record** — headline
findings, per-experiment sections, pre-registered predictions and their
outcomes (including the ones we got wrong), and the three adversarial
review rounds this program was subjected to (round 1 preserved at
REVIEW_round1.md with disposition notes; every headline statistic survived
hostile recomputation).

Experiments (each script's docstring carries its pre-registered
predictions, committed before first full run):

| Script | What it tests |
|---|---|
| `exp_a_linear.py` | Coherence vs reconstruction in a linear content/context regime |
| `exp_b_transformer.py` | Toy transformer + LoRA + context transplantation (k-arm variants) |
| `rerun_k2_power.py` | 15-seed powered rerun with shuffled-label control + ridge probe |
| `coherence_map*.py` | Layer×position identity/fact legibility maps (3 rounds) |
| `exp_b_deconfound.py` | Data-composition control for the k-arm contrast |
| `exp_c_safeguards.py` | Trust region / decay / shadow vs forgetting (110 cells) |
| `exp_d_compositional.py` | Same/diff comparison circuit vs consolidated facts |
| `exp_e_realm.py` | The pivotal test on Qwen2.5-1.5B-Instruct (notes-vs-weights) |
| `exp_e_cloze.py` | In-format cloze supplement (the format-boundness gradient) |

Requirements: Python ≥3.10, torch (MPS or CUDA or CPU), numpy;
`transformers` for exp_e. Runtimes: minutes (exp_a) to ~4h (exp_e, 5
seeds on Apple M-series). Raw outputs in `results_*/`.
