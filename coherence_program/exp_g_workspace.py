"""
Experiment G — Workspace access of consolidated memory (Revision 2 instrument)
==============================================================================

Implements EXPG_DESIGN.md **REVISION 2** (patch-injection probe). exp_e
substrate imported wholesale: same World(seed), same LoRA wrapping, same
frozen consolidate() recipe, paired seeds 0-4, same rng streams.

Question: do weight-consolidated facts fail to reach the workspace
(no-broadcast, G-A), or reach it and fail of uptake
(broadcast-without-uptake, G-B)? Shuffled arm = geometry-without-binding
control. G-C split by the Phase-0 layer-signature pre-gate into
G-C1 (1.5B lacks workspace signatures) / G-C2 (instrument blunt).

Pipeline (gate first, training only if gate passes):
  Phase 0  layer-signature pre-gate (descriptive, no threshold) ->
           results_g/pregate[_smoke].json
  Gate     context-condition natural-broadcast readout + mismatched-context
           control, all seeds, adapter off (cheap; runs before any training)
  Main     per (seed, arm): consolidate, then patch-injection probes with
           delta_fact (channel positive control) and delta_adapter (THE
           measurement) tangents; logit-lens baseline; noise floor
  Output   results_g/results_g[_smoke].json + printed tables + verdicts

IMPLEMENTATION DECISIONS (beyond the spec, numbered for the record)
-------------------------------------------------------------------
 D1  Layer convention: h_l := residual-stream OUTPUT of decoder layer l
     (0-indexed, 28 layers). Captured by forward hooks on
     model.model.layers[l]; injection = forward hook that returns the
     layer output with v added at position t. "Final" states are captured
     PRE final-RMSNorm via a forward-pre-hook on model.model.norm
     (transformers 5.x puts the post-norm state in last_hidden_state, so
     hidden_states outputs are not used at all).
 D2  Readout lens: scores = W_U @ (g * Delta / rms(Delta)), i.e. the
     model's own final RMSNorm applied TO THE DIFFERENCE itself (unit-rms
     normalize Delta, multiply by the learned norm weight g, unembed).
     Not norm(h+Delta)-norm(h): we lens the broadcast content directly,
     through the model's own output geometry.
 D3  Components per probe: direct = Delta(t'=t); broadcast = mean of
     Delta over all t'>t, THEN lensed (average content, not average of
     ranks); answer_pre = Delta at the final prompt position (the token
     position whose output predicts the answer; prompts end "... A:").
 D4  City tokens: only cities that are a single Qwen BPE token with
     leading space are SCORED (" Lisbon" " Osaka" " Denver" " Adelaide";
     " Tallinn"/" Cusco" are multi-token and are dropped from the token
     set). Worlds are NOT changed (pairing with exp_e preserved); entities
     whose true city is dropped still serve as false-entities for kept
     tokens. Within-token counterbalanced margin per Revision-2 item 5:
     margin(c) = mean RR(c) at entities where c true - mean RR(c) at
     entities where c false; cell margin = mean over kept tokens. RR =
     1/rank of the token in the FULL-vocab score vector.
 D5  delta_fact alignment: name-final position located independently in
     each prompt via character offsets of the LAST occurrence of the name
     (in the context prompt the name also appears in the fact paragraph;
     the last occurrence is the one inside the QA question). Sanity check:
     the distance-from-end of the two anchors must be equal AND the token
     suffix from the anchor to the end must be identical between the two
     encodings (suffix alignment); violations are counted and reported
     (per-prompt offsets remain the anchor of record).
 D6  Tangents injected RAW (scale 1.0, the actual activation difference,
     per Revision-2 item 1 "applied ADDITIVELY"); no epsilon sweep (that
     belonged to the abandoned W-lens); tangent norms recorded per cell.
 D7  Noise floor: one duplicate unperturbed forward pair per probe block;
     floor = max over positions of the L2 norm of the final-state
     difference (fp32; expected ~0 on MPS). Discard rule applied at
     COMPONENT level: a component whose vector norm < 2*floor is dropped
     from that cell's margin (never normalized); discard counts recorded.
 D8  Gate quantity: natural Delta = final(context) - final(base) at the
     answer-preceding position (last position of both prompts — end-
     aligned, so no cross-length ambiguity); counterbalanced margin >= 0.1
     in >= 4/5 seeds (1/1 in SMOKE). Mismatched-context control: entity
     A's question, context = fact_paragraph of the first entity B (cyclic
     scan from A+1) with a different city; per-entity PAIRED comparison of
     RR(true city) matched vs mismatched (same entity + same token, so
     token priors cancel), pooled over seeds, one-sided paired t-test at
     p<.05 (critical-t lookup, no scipy). Both must hold for gate pass.
     A broadcast-mean gate margin (aligned common suffix after the name)
     is also recorded, descriptively.
 D9  Logit-lens baseline (Revision-2 item 6i): same lens (D2) applied to
     the RAW state h_l at the name-final position of the SAME forward
     that supplies the tangent (adapter-ON pass for delta_adapter rows;
     context pass for delta_fact rows), same counterbalanced margin.
     Claimed presence must exceed this per-seed, per-layer number.
 D10 delta_fact probes are arm-independent (both passes adapter-off), so
     they are run once per seed and recorded with arm="shared".
 D11 Mid-band for decision rules := probed layers in {12,16,20} (of 28);
     probed set {8,12,16,20,24} (SMOKE: {12,20}).
 D12 G-B "absent in shuffled" operationalized as: shuffled-arm mean margin
     < 0.05 at the same (layer, component, inject_mode) cell. The
     reviewer's phrase "shuffled scrambled pairings" is ambiguous (the
     shuffled arm's LM loss still stores true facts; only the coherence
     grouping was scrambled) — flagged as a design concern in the run
     summary; both readings are printable from the saved cells.
 D13 Pre-gate: 200 (SMOKE: 40) neutral prompts = NATURAL_TEXT sentences +
     programmatic simple sentences (seeded rng). Per layer 0..27:
     (a) excess kurtosis per dim over positions x prompts, mean over dims;
     (b) lag-1 position autocorrelation = mean cosine of consecutive
     position states, reported RAW and CENTERED (per-prompt mean state
     subtracted — raw cosines saturate near 1 from the shared mean);
     (c) participation ratio (sum lam)^2 / sum lam^2 of the PCA spectrum
     over a 2000-row subsample. Descriptive; saved and printed.
     POSITION 0 IS EXCLUDED from all pre-gate statistics: Qwen's
     first-token massive activations (attention-sink outliers) dominate
     the variance and pinned PR at 1.0 across layers 1-25 in a first
     smoke run. A trimmed PR is also reported (top-8 highest-variance
     dims removed) because a handful of outlier DIMS persists at every
     position in Qwen residual streams; both raw and trimmed are saved.
 D14 Secondary condition (spec): delta_adapter also injected with adapter
     ON (deployed configuration), Delta vs the adapter-ON unperturbed
     forward. inject_mode in {"base","on"}. Decision rules use "base".
 D15 SMOKE=1: seed {0}, layers {12,20}, 40 pregate prompts, exp_e's own
     SMOKE consolidation (40 steps), gate thresholds 1/1 seeds; separate
     output filenames (*_smoke.json) so a smoke run never clobbers the
     full run's artifacts.
 D16 (Revision 2.1, pre-registered in EXPG_DESIGN.md before the full run)
     M2 — natural adapter delta, the PRIMARY presence measurement: per
     (seed, arm), the QA prompt is run adapter ON vs adapter OFF;
     Delta_l = h_l,on - h_l,off at the answer-preceding position, for
     l in PROBE_LAYERS + {26, "final" (pre-norm output of layer 27)};
     lensed per D2; standard within-token counterbalanced margin.
     Same-cell logit-lens baseline = D2 lens of the RAW adapter-on state
     at the same site. Rows carry inject_mode="natural". Injection probes
     become SECONDARY; if delta_fact INJECTION margins are ~0 everywhere
     (max |margin| < 0.05 over all its cells), injection is declared
     instrument-insensitive and carries no weight in the verdict.
     Verdicts (Rev 2.1): G-A = gate pass AND all mid-band (probed layers
     8-20) M2 cells for recon+cohere < 0.05 (late-only signal at l>=24 is
     consistent with G-A: direct-output-pathway / automatic tier);
     G-B = gate pass AND some (arm, mid-band layer) with M2 margin >= 0.1
     in >= 3/5 seeds, each exceeding the same-cell logit-lens baseline;
     cohere-SPECIFIC G-B additionally requires cohere > shuffled AND
     cohere > recon on per-seed mid-band-mean M2 margin (paired across
     seeds, one-sided p<.05). Shuffled is reported both ways; its control
     role is the cohere-vs-shuffled comparison (D12 resolved).
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_e_realm as E  # noqa: E402  (reads SMOKE env at import: STEPS=40 if smoke)

DEV = E.DEV
SMOKE = os.environ.get("SMOKE", "") == "1"
MODEL_NAME = os.environ.get("EXPG_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
ARMS = [("recon", 0.0, False), ("cohere", E.LAMBDA, False), ("shuffled", E.LAMBDA, True)]
PROBE_LAYERS = [12, 20] if SMOKE else [8, 12, 16, 20, 24]
MID_BAND = [12, 16, 20]                      # D11 (injection rules, Rev 2)
M2_EXTRA_LAYERS = [26]                       # D16: M2 also reads 26 + "final"
M2_LAYERS = PROBE_LAYERS + M2_EXTRA_LAYERS + ["final"]
M2_MID = [l for l in PROBE_LAYERS if 8 <= l <= 20]   # Rev 2.1 mid-band
N_PREGATE = 40 if SMOKE else 200
GATE_MARGIN = 0.10
GATE_SEEDS_REQ = 1 if SMOKE else 4
GB_SEEDS_REQ = 1 if SMOKE else 3
GA_EPS = 0.05
N_ENTS = E.N_ENTS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_g")
RESULTS_NAME = "results_g_smoke.json" if SMOKE else "results_g.json"
PREGATE_NAME = "pregate_smoke.json" if SMOKE else "pregate.json"

TIMINGS = {}


# ------------------------------------------------------------------ small stats
# one-sided critical t at alpha=.05 (df -> t); nearest df at or below
_T05 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
        8: 1.860, 9: 1.833, 10: 1.812, 12: 1.782, 15: 1.753, 20: 1.725,
        30: 1.697, 60: 1.671, 120: 1.658}


def t_crit(df):
    ks = sorted(k for k in _T05 if k <= max(df, 1))
    return _T05[ks[-1]] if ks else 6.314


def paired_t(diffs):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2:
        return 0.0, False
    sd = d.std(ddof=1)
    if sd == 0:
        return (float("inf") if d.mean() > 0 else 0.0), d.mean() > 0
    t = d.mean() / (sd / math.sqrt(n))
    return float(t), bool(t > t_crit(n - 1))


# ------------------------------------------------------------------ forward machinery (D1)
@torch.no_grad()
def fwd(model, ids, inject=None, capture_layers=()):
    """One forward pass. Returns {'final': [T,d] pre-final-norm states,
    l: [T,d] residual-stream output of decoder layer l for l in
    capture_layers}. inject=(layer, pos, v) adds v to the output of
    decoder layer `layer` at position `pos` (forward hook replaces the
    layer output)."""
    cap, handles = {}, []

    def mk_cap(l):
        def hook(mod, args, out):
            h = out[0] if isinstance(out, tuple) else out
            cap[l] = h.detach()[0]
        return hook

    for l in capture_layers:
        handles.append(model.model.layers[l].register_forward_hook(mk_cap(l)))
    if inject is not None:
        li, pos, v = inject

        def ihook(mod, args, out):
            if isinstance(out, tuple):
                h = out[0].clone()
                h[:, pos] = h[:, pos] + v
                return (h,) + tuple(out[1:])
            h = out.clone()
            h[:, pos] = h[:, pos] + v
            return h
        handles.append(model.model.layers[li].register_forward_hook(ihook))

    def npre(mod, args):
        cap["final"] = args[0].detach()[0]
    handles.append(model.model.norm.register_forward_pre_hook(npre))
    try:
        model(input_ids=ids.to(DEV))
    finally:
        for h in handles:
            h.remove()
    return cap


def norm_lens(model, x):
    """D2: scores = W_U @ (g * x/rms(x)). x: [d] or [k,d]."""
    g = model.model.norm.weight
    eps = getattr(model.model.norm, "variance_epsilon",
                  getattr(model.config, "rms_norm_eps", 1e-6))
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (xn * g) @ model.lm_head.weight.T


def rr(scores, tid):
    """Reciprocal rank of token tid in the full-vocab score vector."""
    return 1.0 / float((scores > scores[tid]).sum().item() + 1)


def measure_floor(model, ids):
    """D7: duplicate unperturbed forwards; max positionwise L2 of the diff."""
    a = fwd(model, ids)["final"]
    b = fwd(model, ids)["final"]
    return float((a - b).norm(dim=-1).max().item())


# ------------------------------------------------------------------ tokens & positions
def city_token_ids(tok):
    """D4: keep only single-BPE-token cities (leading space)."""
    keep = {}
    for ci, c in enumerate(E.CITIES):
        ids = tok(" " + c, add_special_tokens=False).input_ids
        if len(ids) == 1:
            keep[ci] = ids[0]
        else:
            print(f"  [tokens] dropping city {c!r}: {len(ids)} BPE tokens", flush=True)
    print(f"  [tokens] scored city tokens: "
          f"{[E.CITIES[ci] for ci in keep]}", flush=True)
    return keep


def name_pos_in(tok, text, name):
    """Token index of the final token of the LAST occurrence of name (D5)."""
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    start = text.rindex(name)
    end = start + len(name)
    om = enc.offset_mapping[0].tolist()
    idx = [j for j, (a, b) in enumerate(om)
           if a < end and b > start and b <= end + 1 and (a, b) != (0, 0)]
    if not idx:
        idx = [j for j, (a, b) in enumerate(om) if a < end and b > start]
    return enc.input_ids, max(idx)


def cb_margin(scores_by_ent, city_arr, keep):
    """D4: within-token counterbalanced reciprocal-rank margin.
    scores_by_ent: {e: [vocab] tensor}. Returns (mean margin over kept
    tokens with >=1 true and >=1 false entity, per-token dict)."""
    per_tok = {}
    for ci, tid in keep.items():
        tru = [rr(s, tid) for e, s in scores_by_ent.items() if city_arr[e] == ci]
        fal = [rr(s, tid) for e, s in scores_by_ent.items() if city_arr[e] != ci]
        if tru and fal:
            per_tok[int(ci)] = float(np.mean(tru) - np.mean(fal))
    if not per_tok:
        return None, per_tok
    return float(np.mean(list(per_tok.values()))), per_tok


def qa_text(world, e):
    return f"Q: What city does {world.names[e]} live in? A:"


# ------------------------------------------------------------------ PHASE 0 pre-gate (D13)
def build_neutral_prompts(n, rng):
    sents = [s.strip() + "." for s in E.NATURAL_TEXT.split(".") if len(s.strip()) > 20]
    subs = ["The engineer", "A quiet librarian", "The old ferry", "My neighbor",
            "The committee", "A young violinist", "The harbor master",
            "The research team", "An early train", "The gardener"]
    verbs = ["considered", "repaired", "described", "ignored", "finished",
             "sketched", "measured", "announced", "carried", "postponed"]
    objs = ["the proposal", "a wooden bridge", "the morning schedule",
            "an old map", "the broken instrument", "a long letter",
            "the annual report", "the garden wall", "a borrowed book",
            "the quiet street"]
    tails = ["before noon.", "without much fuss.", "in early spring.",
             "after the meeting.", "despite the rain.", "for the second time.",
             "with great care.", "near the station.", "over the weekend.",
             "as planned."]
    prompts = list(sents)
    while len(prompts) < n:
        prompts.append(f"{rng.choice(subs)} {rng.choice(verbs)} "
                       f"{rng.choice(objs)} {rng.choice(tails)}")
    return prompts[:n]


@torch.no_grad()
def phase0_pregate(model, tok):
    t0 = time.time()
    n_layers = len(model.model.layers)
    rng = np.random.default_rng(1234)
    prompts = build_neutral_prompts(N_PREGATE, rng)
    states = {l: [] for l in range(n_layers)}
    ac_raw = {l: [] for l in range(n_layers)}
    ac_cen = {l: [] for l in range(n_layers)}
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        cap = fwd(model, ids, capture_layers=range(n_layers - 1))
        for l in range(n_layers):
            X = (cap[l] if l < n_layers - 1 else cap["final"]).float().cpu().numpy()
            X = X[1:]      # D13: drop position 0 (attention-sink outliers)
            states[l].append(X)
            if X.shape[0] >= 3:
                Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                ac_raw[l].append(float(np.mean(np.sum(Xn[:-1] * Xn[1:], axis=1))))
                Xc = X - X.mean(0)
                Xcn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8)
                ac_cen[l].append(float(np.mean(np.sum(Xcn[:-1] * Xcn[1:], axis=1))))
    profile = []
    for l in range(n_layers):
        X = np.concatenate(states[l], axis=0)
        mu, s2 = X.mean(0), X.var(0)
        kurt = float(np.mean(((X - mu) ** 4).mean(0) / (s2 ** 2 + 1e-12) - 3.0))
        sub = X[rng.choice(len(X), min(2000, len(X)), replace=False)]
        sub = sub - sub.mean(0)
        lam = np.linalg.svd(sub, compute_uv=False) ** 2 / (len(sub) - 1)
        pr = float(lam.sum() ** 2 / (lam ** 2).sum())
        # trimmed PR: remove the top-8 highest-variance dims (Qwen outlier
        # dims persist at every position and can pin PR near 1)
        keep_dims = np.argsort(sub.var(0))[:-8]
        subt = sub[:, keep_dims]
        lamt = np.linalg.svd(subt - subt.mean(0), compute_uv=False) ** 2 / (len(subt) - 1)
        prt = float(lamt.sum() ** 2 / (lamt ** 2).sum())
        profile.append({"layer": l, "excess_kurtosis": kurt,
                        "autocorr_lag1_raw": float(np.mean(ac_raw[l])),
                        "autocorr_lag1_centered": float(np.mean(ac_cen[l])),
                        "participation_ratio": pr,
                        "participation_ratio_trim8": prt,
                        "n_rows": int(len(X))})
    TIMINGS["pregate"] = time.time() - t0
    out = {"model": MODEL_NAME, "n_prompts": len(prompts), "smoke": SMOKE,
           "note": "descriptive pre-gate (Revision 2 item 8); no threshold",
           "profile": profile}
    with open(os.path.join(OUT_DIR, PREGATE_NAME), "w") as f:
        json.dump(out, f, indent=1)
    print("\nPHASE 0 — layer-signature pre-gate "
          f"({len(prompts)} prompts, {TIMINGS['pregate']:.0f}s)", flush=True)
    print(f"  {'layer':>5} {'ex.kurt':>8} {'ac1.raw':>8} {'ac1.cen':>8} "
          f"{'PR':>7} {'PRtrim8':>8}")
    for r in profile:
        print(f"  {r['layer']:>5} {r['excess_kurtosis']:>8.2f} "
              f"{r['autocorr_lag1_raw']:>8.3f} {r['autocorr_lag1_centered']:>8.3f} "
              f"{r['participation_ratio']:>7.1f} "
              f"{r['participation_ratio_trim8']:>8.1f}", flush=True)
    return out


# ------------------------------------------------------------------ GATE (D8)
def mismatch_partner(world, e):
    for k in range(1, N_ENTS):
        b = (e + k) % N_ENTS
        if world.city[b] != world.city[e]:
            return b
    return (e + 1) % N_ENTS


@torch.no_grad()
def run_gate(model, tok, world, keep, seed):
    t0 = time.time()
    scores_ans, scores_mis, scores_bmean = {}, {}, {}
    rr_matched, rr_mismatched = [], []
    align_viol = 0
    floor = None
    for e in range(N_ENTS):
        qa = qa_text(world, e)
        base_ids, pos_b = name_pos_in(tok, qa, world.names[e])
        ctx = world.fact_paragraph([e]) + "\n" + qa
        ctx_ids, pos_c = name_pos_in(tok, ctx, world.names[e])
        b = mismatch_partner(world, e)
        mis = world.fact_paragraph([b]) + "\n" + qa
        mis_ids, _ = name_pos_in(tok, mis, world.names[e])
        if floor is None:
            floor = measure_floor(model, base_ids)
        fb = fwd(model, base_ids)["final"]
        fc = fwd(model, ctx_ids)["final"]
        fm = fwd(model, mis_ids)["final"]
        # suffix-alignment sanity (D5): distance-from-end + token suffix match
        Tb, Tc = base_ids.shape[1], ctx_ids.shape[1]
        if (Tb - pos_b) != (Tc - pos_c) or \
           not torch.equal(base_ids[0, pos_b:], ctx_ids[0, Tc - (Tb - pos_b):]):
            align_viol += 1
        d_ans = fc[-1] - fb[-1]
        d_mis = fm[-1] - fb[-1]
        scores_ans[e] = norm_lens(model, d_ans)
        scores_mis[e] = norm_lens(model, d_mis)
        # broadcast-mean over aligned common suffix strictly after the name
        k_after = Tb - pos_b - 1
        if k_after >= 1:
            d_bm = (fc[Tc - k_after:] - fb[Tb - k_after:]).mean(0)
            scores_bmean[e] = norm_lens(model, d_bm)
        ci = int(world.city[e])
        if ci in keep:
            rr_matched.append(rr(scores_ans[e], keep[ci]))
            rr_mismatched.append(rr(scores_mis[e], keep[ci]))
    margin_ans, per_tok = cb_margin(scores_ans, world.city, keep)
    margin_mis, _ = cb_margin(scores_mis, world.city, keep)
    margin_bm, _ = cb_margin(scores_bmean, world.city, keep)
    TIMINGS.setdefault("gate_seed", []).append(time.time() - t0)
    g = {"seed": seed, "margin_answer_pre": margin_ans,
         "margin_mismatched": margin_mis, "margin_broadcast_mean": margin_bm,
         "per_token": per_tok, "rr_true_matched": rr_matched,
         "rr_true_mismatched": rr_mismatched, "floor": floor,
         "align_violations": align_viol}
    print(f"  gate seed {seed}: answer-pre margin={margin_ans:.4f} "
          f"(mismatched={margin_mis:.4f}, bcast-mean={margin_bm:.4f}) "
          f"floor={floor:.2e} align_viol={align_viol} "
          f"({TIMINGS['gate_seed'][-1]:.0f}s)", flush=True)
    return g


# ------------------------------------------------------------------ probes
@torch.no_grad()
def probe_block(model, ent_data, v_by_el, base_key, floor, keep, city,
                llens_by_layer, meta):
    """One probe sweep: for each layer, inject v at (layer, name-final pos)
    of each entity's QA prompt, read out D3 components, counterbalanced
    margins. ent_data[e] = {ids, pos, base(=final states of the matching
    unperturbed pass)}. Returns list of cell rows."""
    rows = []
    for l in PROBE_LAYERS:
        comp_scores = {"direct": {}, "broadcast": {}, "answer_pre": {}}
        dnorms, vnorms, ndisc = [], [], {"direct": 0, "broadcast": 0, "answer_pre": 0}
        for e, d in ent_data.items():
            v = v_by_el[e][l]
            cap = fwd(model, d["ids"], inject=(l, d["pos"], v))
            delta = cap["final"] - d[base_key]
            comps = {"direct": delta[d["pos"]],
                     "broadcast": delta[d["pos"] + 1:].mean(0),
                     "answer_pre": delta[-1]}
            vnorms.append(float(v.norm().item()))
            dnorms.append(float(delta.norm(dim=-1).max().item()))
            for cname, vec in comps.items():
                if float(vec.norm().item()) < 2 * floor:      # D7
                    ndisc[cname] += 1
                    continue
                comp_scores[cname][e] = norm_lens(model, vec)
        for cname, sd in comp_scores.items():
            margin, per_tok = (cb_margin(sd, city, keep) if len(sd) >= 4
                               else (None, {}))
            rows.append(dict(meta, layer=l, component=cname, margin=margin,
                             per_token=per_tok, n_entities=len(sd),
                             n_discarded=ndisc[cname], floor=floor,
                             mean_tangent_norm=float(np.mean(vnorms)),
                             mean_max_delta_norm=float(np.mean(dnorms)),
                             logitlens_margin=llens_by_layer.get(l)))
    return rows


def sanitize(o):
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def save_results(payload):
    with open(os.path.join(OUT_DIR, RESULTS_NAME), "w") as f:
        json.dump(sanitize(payload), f, indent=1)


# ------------------------------------------------------------------ decisions
def decide(gate_list, cells, mismatch_t, mismatch_sig):
    """Revision-2.1 decision rules, gate first: M2 (natural adapter delta)
    is PRIMARY; injection probes are secondary and carry no weight if the
    delta_fact injection control is instrument-insensitive (D16)."""
    margins = {s: g["margin_answer_pre"] for s, g in gate_list.items()}
    n_pass = sum(1 for m in margins.values() if m is not None and m >= GATE_MARGIN)
    gate_ok = n_pass >= GATE_SEEDS_REQ and mismatch_sig
    detail = {"gate_margins": margins, "gate_seeds_pass": n_pass,
              "gate_seeds_required": GATE_SEEDS_REQ,
              "mismatch_paired_t": mismatch_t, "mismatch_sig": mismatch_sig,
              "gate_pass": gate_ok}
    if not gate_ok:
        return "G-C2-candidate (pending pregate interpretation)", detail

    # ---- injection instrument-sensitivity (Rev 2.1 item b)
    fact_inj = [abs(c["margin"]) for c in cells
                if c["tangent"] == "delta_fact" and c["inject_mode"] == "base"
                and c["margin"] is not None]
    inj_insensitive = (max(fact_inj, default=0.0) < GA_EPS)
    detail["fact_injection_max_abs_margin"] = max(fact_inj, default=None)
    detail["injection_instrument_insensitive"] = inj_insensitive

    # ---- M2 pools
    m2 = [c for c in cells if c["inject_mode"] == "natural"
          and c["margin"] is not None]
    m2_mid = [c for c in m2 if isinstance(c["layer"], int)
              and c["layer"] in M2_MID]

    # G-A (Rev 2.1 item a): ALL mid-band M2 cells for recon+cohere < GA_EPS;
    # late-only signal (>=24, 26, final) is consistent with G-A.
    ga_pool = [c["margin"] for c in m2_mid if c["arm"] in ("recon", "cohere")]
    ga = bool(ga_pool) and max(ga_pool) < GA_EPS
    detail["GA_m2_max_midband_margin"] = max(ga_pool, default=None)
    late = [c["margin"] for c in m2 if c["arm"] in ("recon", "cohere")
            and (c["layer"] == "final"
                 or (isinstance(c["layer"], int) and c["layer"] >= 24))]
    detail["M2_late_max_margin"] = max(late, default=None)

    # G-B (Rev 2.1 item c): M2 mid-band margin >= 0.1 in >= 3/5 seeds,
    # each exceeding the same-cell logit-lens baseline.
    gb_hits = []
    for arm in ("cohere", "recon", "shuffled"):
        for l in M2_MID:
            cs = [c for c in m2_mid if c["arm"] == arm and c["layer"] == l]
            ok_seeds = [c["seed"] for c in cs
                        if c["margin"] >= GATE_MARGIN
                        and c["logitlens_margin"] is not None
                        and c["margin"] > c["logitlens_margin"]]
            if len(ok_seeds) >= GB_SEEDS_REQ:
                gb_hits.append({"arm": arm, "layer": l, "seeds": ok_seeds})
    detail["GB_m2_hits"] = gb_hits

    # cohere-specific comparison (paired across seeds on mid-band mean)
    def midmean(arm):
        by_seed = {}
        for c in m2_mid:
            if c["arm"] == arm:
                by_seed.setdefault(c["seed"], []).append(c["margin"])
        return {s: float(np.mean(v)) for s, v in by_seed.items()}
    mc, mr, ms = midmean("cohere"), midmean("recon"), midmean("shuffled")
    common = sorted(set(mc) & set(mr) & set(ms))
    t_cs, sig_cs = paired_t([mc[s] - ms[s] for s in common])
    t_cr, sig_cr = paired_t([mc[s] - mr[s] for s in common])
    detail["m2_midband_mean_by_arm"] = {"cohere": mc, "recon": mr, "shuffled": ms}
    detail["cohere_vs_shuffled_t"], detail["cohere_vs_shuffled_sig"] = t_cs, sig_cs
    detail["cohere_vs_recon_t"], detail["cohere_vs_recon_sig"] = t_cr, sig_cr

    # ---- secondary: injection-based numbers, reported only (Rev 2.1 item b)
    inj_cells = [c for c in cells
                 if c["tangent"] == "delta_adapter" and c["inject_mode"] == "base"
                 and c["arm"] in ("recon", "cohere")
                 and c["component"] in ("broadcast", "answer_pre")
                 and c["margin"] is not None]
    detail["INJ_max_weights_margin"] = max((c["margin"] for c in inj_cells),
                                           default=None)
    detail["INJ_floor_max"] = max((c["floor"] for c in inj_cells), default=None)

    gb_real = [h for h in gb_hits if h["arm"] in ("cohere", "recon")]
    if gb_real:
        cohere_specific = (any(h["arm"] == "cohere" for h in gb_real)
                           and sig_cs and sig_cr)
        detail["GB_cohere_specific"] = cohere_specific
        v = "G-B (broadcast-without-uptake, M2 natural delta)"
        if cohere_specific:
            v += " — cohere-specific"
        return v, detail
    if ga:
        return "G-A (no-broadcast, M2 natural delta)", detail
    return ("no pre-registered pattern matched (M2 presence intermediate) — "
            "report as inconclusive"), detail


# ------------------------------------------------------------------ printing
def print_tables(cells, gate_list):
    def cell_mean(rows):
        v = [r["margin"] for r in rows if r["margin"] is not None]
        return f"{np.mean(v):+.4f}" if v else "   --  "

    print("\n===== SUMMARY TABLES (counterbalanced RR margins, mean over seeds) =====")
    print("\nGate (context condition, natural broadcast, answer-pre position):")
    for s, g in gate_list.items():
        print(f"  seed {s}: matched={g['margin_answer_pre']:.4f} "
              f"mismatched={g['margin_mismatched']:.4f} "
              f"bcast-mean={g['margin_broadcast_mean']:.4f}")
    m2_rows = [c for c in cells if c["inject_mode"] == "natural"]
    if m2_rows:
        arms = sorted({c["arm"] for c in m2_rows})
        print("\nM2 — natural adapter delta (PRIMARY, Rev 2.1), answer-pre position")
        hdr = "    arm      " + "".join(f"  L{str(l):<6}" for l in M2_LAYERS)
        print(hdr + "  llens(mean)")
        for arm in arms:
            line = f"    {arm:<9}"
            for l in M2_LAYERS:
                line += "  " + cell_mean([c for c in m2_rows if c["arm"] == arm
                                          and c["layer"] == l])
            ll = [c["logitlens_margin"] for c in m2_rows if c["arm"] == arm
                  and c["logitlens_margin"] is not None]
            line += f"  {np.mean(ll):+.4f}" if ll else "     --  "
            print(line)
    for tangent, inj in [("delta_fact", "base"), ("delta_adapter", "base"),
                         ("delta_adapter", "on")]:
        rows = [c for c in cells if c["tangent"] == tangent
                and c["inject_mode"] == inj]
        if not rows:
            continue
        arms = sorted({c["arm"] for c in rows})
        print(f"\ntangent={tangent} inject={inj}")
        for comp in ("direct", "broadcast", "answer_pre"):
            print(f"  [{comp}]")
            hdr = "    arm      " + "".join(f"  L{l:<6}" for l in PROBE_LAYERS)
            print(hdr + "  llens(mean)")
            for arm in arms:
                line = f"    {arm:<9}"
                for l in PROBE_LAYERS:
                    line += "  " + cell_mean([c for c in rows if c["arm"] == arm
                                              and c["layer"] == l
                                              and c["component"] == comp])
                ll = [c["logitlens_margin"] for c in rows if c["arm"] == arm
                      and c["component"] == comp
                      and c["logitlens_margin"] is not None]
                line += f"  {np.mean(ll):+.4f}" if ll else "     --  "
                print(line)


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"exp_g workspace probe — model={MODEL_NAME} dev={DEV} smoke={SMOKE} "
          f"seeds={SEEDS} layers={PROBE_LAYERS} steps={E.STEPS}", flush=True)
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    loras = E.add_lora(model, seed=0)
    E.set_lora(loras, False)
    print(f"loaded in {time.time()-t_load:.0f}s; "
          f"{len(model.model.layers)} layers", flush=True)

    # ---------- PHASE 0
    pregate = phase0_pregate(model, tok)
    keep = city_token_ids(tok)
    worlds = {s: E.World(s) for s in SEEDS}

    payload = {"meta": {"model": MODEL_NAME, "smoke": SMOKE, "seeds": SEEDS,
                        "probe_layers": PROBE_LAYERS, "mid_band": MID_BAND,
                        "m2_layers": [str(l) for l in M2_LAYERS],
                        "m2_mid_band": M2_MID,
                        "steps": E.STEPS, "lambda": E.LAMBDA,
                        "gate_margin": GATE_MARGIN,
                        "pregate_file": PREGATE_NAME},
               "gate": {}, "cells": [], "verdict": None, "detail": None,
               "timings": TIMINGS}

    # ---------- GATE (all seeds, before any training)
    print("\nGATE — context-condition natural broadcast (adapter off)", flush=True)
    E.set_lora(loras, False)
    gate_list = {}
    for seed in SEEDS:
        gate_list[seed] = run_gate(model, tok, worlds[seed], keep, seed)
    payload["gate"] = gate_list
    diffs = []
    for g in gate_list.values():
        diffs += [a - b for a, b in zip(g["rr_true_matched"], g["rr_true_mismatched"])]
    mt, msig = paired_t(diffs)
    n_pass = sum(1 for g in gate_list.values()
                 if g["margin_answer_pre"] is not None
                 and g["margin_answer_pre"] >= GATE_MARGIN)
    print(f"  gate: {n_pass}/{len(SEEDS)} seeds >= {GATE_MARGIN} "
          f"(need {GATE_SEEDS_REQ}); mismatch paired t={mt:.2f} "
          f"(n={len(diffs)}, one-sided p<.05: {msig})", flush=True)
    if not (n_pass >= GATE_SEEDS_REQ and msig):
        verdict, detail = decide(gate_list, [], mt, msig)
        payload["verdict"], payload["detail"] = verdict, detail
        save_results(payload)
        print(f"\nGATE FAILED — stopping before training.\nVERDICT: {verdict}",
              flush=True)
        return

    # ---------- MAIN MEASUREMENT
    cells = payload["cells"]
    for seed in SEEDS:
        world = worlds[seed]
        print(f"\n=== seed {seed} ===", flush=True)
        # per-entity prompt data + adapter-off base pass (cached across arms)
        E.set_lora(loras, False)
        t0 = time.time()
        ent, v_fact, llens_fact = {}, {}, {l: {} for l in PROBE_LAYERS}
        for e in range(N_ENTS):
            ids, pos = name_pos_in(tok, qa_text(world, e), world.names[e])
            cap = fwd(model, ids, capture_layers=PROBE_LAYERS + M2_EXTRA_LAYERS)
            ent[e] = {"ids": ids, "pos": pos, "final_off": cap["final"],
                      "h_off": {l: cap[l][pos] for l in PROBE_LAYERS},
                      "m2_off": {**{l: cap[l][-1]
                                    for l in PROBE_LAYERS + M2_EXTRA_LAYERS},
                                 "final": cap["final"][-1]}}
            ctx = world.fact_paragraph([e]) + "\n" + qa_text(world, e)
            cids, cpos = name_pos_in(tok, ctx, world.names[e])
            ccap = fwd(model, cids, capture_layers=PROBE_LAYERS)
            v_fact[e] = {l: ccap[l][cpos] - cap[l][pos] for l in PROBE_LAYERS}
            for l in PROBE_LAYERS:
                llens_fact[l][e] = norm_lens(model, ccap[l][cpos])
        floor = measure_floor(model, ent[0]["ids"])
        llens_fact_m = {l: cb_margin(llens_fact[l], world.city, keep)[0]
                        for l in PROBE_LAYERS}
        TIMINGS.setdefault("fact_extract_seed", []).append(time.time() - t0)

        # delta_fact probes — channel positive control (arm-independent, D10)
        t0 = time.time()
        cells += probe_block(
            model, ent, v_fact, "final_off", floor, keep, world.city,
            llens_fact_m,
            {"seed": seed, "arm": "shared", "tangent": "delta_fact",
             "inject_mode": "base"})
        TIMINGS.setdefault("fact_probe_seed", []).append(time.time() - t0)
        for c in cells[-3 * len(PROBE_LAYERS):]:
            if c["seed"] == seed and c["tangent"] == "delta_fact":
                m = "None" if c["margin"] is None else f"{c['margin']:+.4f}"
                print(f"  fact-probe L{c['layer']} {c['component']:<10} "
                      f"margin={m} |v|={c['mean_tangent_norm']:.2f} "
                      f"maxΔ={c['mean_max_delta_norm']:.3f}", flush=True)
        save_results(payload)

        for arm, lam, shuf in ARMS:
            t0 = time.time()
            E.consolidate(model, tok, world, loras, seed, lam, shuf)
            TIMINGS.setdefault("train_arm", []).append(time.time() - t0)
            t0 = time.time()
            E.set_lora(loras, True)
            v_ad, llens_ad = {}, {l: {} for l in PROBE_LAYERS}
            m2_scores = {l: {} for l in M2_LAYERS}
            m2_llens = {l: {} for l in M2_LAYERS}
            m2_dnorms = {l: [] for l in M2_LAYERS}
            for e in range(N_ENTS):
                cap = fwd(model, ent[e]["ids"],
                          capture_layers=PROBE_LAYERS + M2_EXTRA_LAYERS)
                ent[e]["final_on"] = cap["final"]
                v_ad[e] = {l: cap[l][ent[e]["pos"]] - ent[e]["h_off"][l]
                           for l in PROBE_LAYERS}
                for l in PROBE_LAYERS:
                    llens_ad[l][e] = norm_lens(model, cap[l][ent[e]["pos"]])
                # M2 (D16): natural adapter delta at the answer-preceding pos
                for l in M2_LAYERS:
                    on_state = cap["final"][-1] if l == "final" else cap[l][-1]
                    d = on_state - ent[e]["m2_off"][l]
                    m2_dnorms[l].append(float(d.norm().item()))
                    m2_scores[l][e] = norm_lens(model, d)
                    m2_llens[l][e] = norm_lens(model, on_state)
            llens_ad_m = {l: cb_margin(llens_ad[l], world.city, keep)[0]
                          for l in PROBE_LAYERS}
            m2_line = []
            for l in M2_LAYERS:
                margin, per_tok = cb_margin(m2_scores[l], world.city, keep)
                llm, _ = cb_margin(m2_llens[l], world.city, keep)
                cells.append({"seed": seed, "arm": arm,
                              "tangent": "delta_adapter",
                              "inject_mode": "natural", "layer": l,
                              "component": "answer_pre", "margin": margin,
                              "per_token": per_tok, "n_entities": N_ENTS,
                              "n_discarded": 0, "floor": None,
                              "mean_tangent_norm": float(np.mean(m2_dnorms[l])),
                              "mean_max_delta_norm": float(np.mean(m2_dnorms[l])),
                              "logitlens_margin": llm})
                m2_line.append(f"L{l}:{margin:+.3f}" if margin is not None
                               else f"L{l}:--")
            print(f"  {arm:>8} M2 (natural adapter Δ, answer-pre): "
                  + " ".join(m2_line), flush=True)
            # THE measurement: inject into the frozen (adapter-off) channel
            E.set_lora(loras, False)
            floor_b = measure_floor(model, ent[0]["ids"])
            cells += probe_block(
                model, ent, v_ad, "final_off", floor_b, keep, world.city,
                llens_ad_m,
                {"seed": seed, "arm": arm, "tangent": "delta_adapter",
                 "inject_mode": "base"})
            # secondary: deployed configuration (adapter on) (D14)
            E.set_lora(loras, True)
            floor_o = measure_floor(model, ent[0]["ids"])
            cells += probe_block(
                model, ent, v_ad, "final_on", floor_o, keep, world.city,
                llens_ad_m,
                {"seed": seed, "arm": arm, "tangent": "delta_adapter",
                 "inject_mode": "on"})
            E.set_lora(loras, False)
            TIMINGS.setdefault("probe_arm", []).append(time.time() - t0)
            arm_rows = [c for c in cells if c["seed"] == seed and c["arm"] == arm
                        and c["inject_mode"] == "base"
                        and c["component"] in ("broadcast", "answer_pre")]
            best = max((c["margin"] for c in arm_rows if c["margin"] is not None),
                       default=None)
            bs = "None" if best is None else f"{best:+.4f}"
            print(f"  {arm:>8}: train {TIMINGS['train_arm'][-1]:.0f}s, "
                  f"probe {TIMINGS['probe_arm'][-1]:.0f}s; best t'>t margin "
                  f"(inject=base)={bs}; llens by layer="
                  + " ".join(f"L{l}:{llens_ad_m[l]:+.3f}" for l in PROBE_LAYERS
                             if llens_ad_m[l] is not None), flush=True)
            save_results(payload)

    # ---------- decisions + tables
    verdict, detail = decide(gate_list, cells, mt, msig)
    payload["verdict"], payload["detail"] = verdict, detail
    save_results(payload)
    print_tables(cells, gate_list)
    print("\n===== DECISION RULES (Revision 2.1, gate first, M2 primary) =====")
    print(f"  gate pass: {detail['gate_pass']} "
          f"({detail['gate_seeds_pass']}/{len(SEEDS)} seeds, "
          f"mismatch t={detail['mismatch_paired_t']:.2f} sig={detail['mismatch_sig']})")
    fi = detail.get("fact_injection_max_abs_margin")
    if fi is not None:
        print(f"  injection sensitivity: max |delta_fact injection margin| = "
              f"{fi:.4f} -> instrument-insensitive: "
              f"{detail['injection_instrument_insensitive']} "
              f"(injection carries no verdict weight if True)")
    if detail.get("GA_m2_max_midband_margin") is not None:
        print(f"  G-A check (M2): max mid-band margin (recon+cohere) = "
              f"{detail['GA_m2_max_midband_margin']:+.4f} (need < {GA_EPS}); "
              f"late-band (>=24/final) max = "
              f"{detail['M2_late_max_margin']:+.4f}")
    print(f"  G-B M2 hits: {detail.get('GB_m2_hits', [])}")
    if "cohere_vs_shuffled_t" in detail:
        print(f"  cohere vs shuffled: t={detail['cohere_vs_shuffled_t']:.2f} "
              f"sig={detail['cohere_vs_shuffled_sig']}; cohere vs recon: "
              f"t={detail['cohere_vs_recon_t']:.2f} "
              f"sig={detail['cohere_vs_recon_sig']}")
    if detail.get("INJ_max_weights_margin") is not None:
        print(f"  [secondary] injection max weights margin = "
              f"{detail['INJ_max_weights_margin']:+.4f}; floor max = "
              f"{detail['INJ_floor_max']:.2e}")
    print(f"\nVERDICT: {verdict}")

    # ---------- runtime projection (meaningful in SMOKE)
    if SMOKE:
        lf = 5 / len(PROBE_LAYERS)
        proj = (TIMINGS["pregate"] * (200 / N_PREGATE)
                + 5 * float(np.mean(TIMINGS["gate_seed"]))
                + 5 * (float(np.mean(TIMINGS["fact_extract_seed"]))
                       + float(np.mean(TIMINGS["fact_probe_seed"])) * lf)
                + 5 * 3 * (float(np.mean(TIMINGS["train_arm"])) * (400 / E.STEPS)
                           + float(np.mean(TIMINGS["probe_arm"])) * lf))
        print(f"\nFULL-RUN PROJECTION: ~{proj/60:.0f} min "
              f"(pregate x{200/N_PREGATE:.0f}, 5 seeds, 3 arms, "
              f"{400//E.STEPS}x train steps, {lf:.1f}x probe layers) "
              f"+ model load", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
