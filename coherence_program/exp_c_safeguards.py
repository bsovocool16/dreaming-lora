"""
Track 2, Experiment C — architectural safeguards vs catastrophic forgetting
===========================================================================

RESULTS.md finding #5: naive LoRA consolidation of 8 new entities drives
base-entity attribute accuracy from 1.00 to ~0.40 at k=2, n_dream=4. The toy
deliberately omitted the paper's safeguards. This experiment ablates them
against that measured baseline and asks: which safeguard buys the most
forgetting reduction per unit of new-fact acquisition lost?

Design
------
Fixed: K_BASE_FAMS=2, n_dream=4, lam in {0.0, 0.3}, seeds 0-4, CPU only.
Paired: same pretrained model per seed; same dream stream
(np.random.default_rng(777+seed)); same LoRA re-init (torch.manual_seed(1000
+seed)). Consolidation loop copied from rerun_k2_power.consolidate (which is
itself the exp_b loop + ridge probe); safeguards are inserted around
opt.step(). Smoke test verified my naive == rerun_k2_power.consolidate
(shuffle=False) bit-for-bit. No max-over-arm selection: all comparisons at
fixed lam.

Variants
--------
  naive   exactly the rerun_k2_power consolidation (sanity: should land near
          the powered-rerun seeds 0-4 numbers — but NOT bit-for-bit, because
          that run drew its LoRA A-init on MPS and did its arithmetic on
          MPS; this run is CPU, so RNG streams and float paths differ).
  trust   norm-bounded trust region (paper SS2.4): after opt.step(), for each
          LoRA pair (per block, per projection: DeltaW = B@A, 8 pairs), if
          the step's spectral-norm change ||Delta(BA)||_2 > DELTA_MAX,
          interpolate both factors back toward their pre-step snapshot by
          s = DELTA_MAX/||Delta(BA)||_2. (First-order in the step size,
          B'A' - B0A0 ~ s*Delta; exact enough at these step sizes.)
          Adam moments are left untouched (projection-style trust region).
  decay   homeostatic decay (paper SS3.4): after each step,
          A <- (1-eta)A, B <- (1-eta)B for all factors. eta in {1e-4, 1e-3}.
          Cumulative over 600 steps: factor shrink x0.94 / x0.55, i.e. BA
          shrink ~x0.89 / ~x0.30 on whatever survives re-learning.
  shadow  EMA copy of all LoRA factors, theta_sh <- (1-beta)theta_sh +
          beta*theta after each step, initialized at the zero-init adapter
          (B=0 => effective DeltaW starts at 0). beta in {0.05, 0.005}.
          Training is untouched, so the LIVE readout of this variant equals
          naive bit-for-bit (verified in smoke test). Both betas are
          tracked in one run; each metric is reported live and shadow.
  combo   trust + decay(best eta) + shadow(both beta readouts).
          Per-step order: opt.step() -> trust clip -> decay -> shadow EMA.
          Best eta rule (committed before running): the eta whose lam=0.0
          mean(base_attr_acc + attr_dream_acc) over seeds is higher.

DELTA_MAX calibration — HONEST RECORD OF A PROCESS FAILURE
----------------------------------------------------------
Intended procedure: calibrate DELTA_MAX on seed-0 naive step norms so that
~10-30% of pair-steps clip. What actually happened: DELTA_MAX = 0.004 was
written in from a rough pre-run guess; the calibration pass had not yet
run when the original driver process died overnight and the grid was
relaunched with the guess still in place. Post-hoc calibration (mode
`calibrate`, seed-0 naive runs, lam 0.0 and 0.3, 9,600 pair-steps, no
clipping) measured the real distribution of ||Delta(BA)||_2:
p50=0.0090, p70=0.0148, p75=0.0173, p80=0.0195, p85=0.0220, p90=0.0259,
p95=0.0325, p99=0.0485; fraction exceeding 0.004 = 0.896 (~p10 from the
bottom). So the `trust` and `combo` arms in the main grid clipped on 100%
of steps (recorded per run as `clip_frac`): they implement a HARD
per-step spectral-norm cap of 0.004 (~0.44x the median unclipped step;
total per-pair movement bounded by 600 x 0.004 = 2.4 spectral units), NOT
the spec'd 10-30% trust region. They are reported as such. The spec'd
regime was added afterwards as supplementary arms `trust_cal` /
`combo_cal` with DELTA_MAX_CAL = 0.02 ~= p80 (expected ~20% clip rate on
naive dynamics; realized clip_frac recorded per run), run via mode
`supplement` and appended to the same results file.

Pre-registered predictions (written before the full grid ran)
-------------------------------------------------------------
  P-C1 (naive): reproduces the powered-rerun seeds 0-4 n_dream=4 numbers
       within seed noise: base_attr ~0.40 (range 0.26-0.56), attr_dream
       = 1.00, attr_eval ~0.76 (lam=0) / ~0.84 (lam=0.3).
  P-C2 (trust): moderate forgetting reduction (base_attr +0.05 to +0.15)
       at small acquisition cost (attr_dream -0.00 to -0.05): capping the
       largest per-step DeltaW excursions removes the steps that do
       disproportionate damage to base circuits, and 600 steps of capped
       updates still reach the (easy) new facts.
  P-C3 (decay): eta=1e-4 indistinguishable from naive (BA shrink ~x0.89 is
       within re-learning capacity); eta=1e-3 gives real forgetting
       protection (base_attr +0.10 to +0.25) with a visible acquisition
       cost (attr_dream and/or attr_eval down), since it shrinks the whole
       adapter toward zero — the only variant that attacks the endpoint
       norm rather than the path.
  P-C4 (shadow): live == naive exactly. beta=0.05 (time constant ~20 steps)
       shadow ~= live on all metrics. beta=0.005 (time constant ~200 steps,
       retains ~5% of the zero-init adapter) forgets less (base_attr +0.02
       to +0.10) and acquires slightly less. Informed by the Track-1 sim
       program (B3: shadow's main effect was READOUT VARIANCE reduction,
       not mean shift), secondary prediction: shadow's clearest effect is
       a smaller per-seed spread, not a big mean gain.
  P-C5 (combo): protections roughly additive — highest base_attr of all
       variants, lowest acquisition; whether it beats its components
       per-unit-cost is the open question.
  P-C6 (tradeoff ordering, forgetting reduction per unit acquisition lost):
       trust >= shadow(0.005) > decay(1e-3) >> decay(1e-4) ~ 0.
  P-C7 (recon-vs-cohere): no safeguard changes the sign or rough size of
       the lam=0.3 minus lam=0.0 contrast on retrieval/probe (still large
       positive) or on base_attr (still ~0); safeguards and the coherence
       term act on different axes.

Output: results_b/results_c_safeguards.json, one row per seed x variant x
lam (decay: one row per eta; shadow/combo: one row per beta, live metrics
duplicated across the two beta rows of the same underlying run).

Usage
-----
  python3.10 exp_c_safeguards.py                  # full grid, sequential
Parallel path actually used for the reported run (the host was shared with
three sibling experiment processes; CPU steps ran ~20x slower than an idle
machine, so seeds were run as concurrent single-thread processes — the
design is unchanged, seeds are independent and pairing is within-seed):
  python3.10 exp_c_safeguards.py pretrain SEED    # cache pretrained model
  python3.10 exp_c_safeguards.py calibrate        # Delta(BA) percentiles
  python3.10 exp_c_safeguards.py phase1 SEED      # naive/trust/decay/shadow
  python3.10 exp_c_safeguards.py besteta          # committed eta rule
  python3.10 exp_c_safeguards.py phase2 SEED ETA  # combo
  python3.10 exp_c_safeguards.py merge            # -> final json
  python3.10 exp_c_safeguards.py supplement       # trust_cal/combo_cal
  SMOKE=1 ... reduces steps for pipeline tests.
"""

import json
import math
import os
import sys
import time

os.environ["K_BASE_FAMS"] = "2"          # must precede the eb import
os.environ.setdefault("OMP_NUM_THREADS", "1")   # before torch import

import numpy as np
import torch

torch.set_num_threads(1)                 # shared host; see docstring

import exp_b_transformer as eb

eb.DEV = "cpu"                            # force CPU before any model exists

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    eb.PRETRAIN_STEPS = 400
    eb.CONSOL_STEPS = 50

SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
N_DREAM = 4
LAMS = [0.0, 0.3]
DELTA_MAX = 0.004          # spectral-norm cap; see calibration note above
DELTA_MAX_CAL = 0.02       # post-hoc calibrated ~p80 cap (supplement mode)
DECAY_ETAS = [1e-4, 1e-3]
SHADOW_BETAS = [0.05, 0.005]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_b",
                   "results_c_safeguards_smoke.json" if SMOKE
                   else "results_c_safeguards.json")
SCRATCH = ("/private/tmp/claude-501/-Users-bsovocool-Documents-Claude-"
           "Projects-Dreaming-LoRA/94521d74-6851-4c86-9c7d-1fe12bb96d48/"
           "scratchpad")
CACHE = os.path.join(SCRATCH, "expc_pretrained")
PART = os.path.join(SCRATCH, "expc_partials")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(PART, exist_ok=True)


def ridge_probe(z_tr, k_tr, z_ho, k_ho, reg=1.0):
    """Copied verbatim from rerun_k2_power (linear probe, un-normalized)."""
    ents = sorted(set(k_tr.tolist()))
    Y = np.stack([(k_tr == e).astype(float) for e in ents], axis=1)
    X = np.concatenate([z_tr, np.ones((len(z_tr), 1))], axis=1)
    W = np.linalg.solve(X.T @ X + reg * np.eye(X.shape[1]), X.T @ Y)
    Xh = np.concatenate([z_ho, np.ones((len(z_ho), 1))], axis=1)
    pred = np.array(ents)[np.argmax(Xh @ W, axis=1)]
    return float(np.mean(pred == k_ho))


def evaluate(model, world, new_ents, dream_fams):
    """Eval block copied from rerun_k2_power.consolidate."""
    base_ents = list(range(eb.N_BASE))
    res = {
        "attr_eval_acc": eb.attr_acc(model, world, new_ents, eb.EVAL_FAMS),
        "attr_dream_acc": eb.attr_acc(model, world, new_ents, dream_fams),
        "base_attr_acc": eb.attr_acc(model, world, base_ents,
                                     list(range(eb.N_FAM))),
    }
    with torch.no_grad():
        idx, _ = world.batch(new_ents, dream_fams, 512,
                             np.random.default_rng(99))
        res["lm_dream"] = float(eb.lm_loss(model(idx), idx).item())
    z_tr, k_tr = eb.name_reps(model, world, new_ents, dream_fams)
    z_ho, k_ho = eb.name_reps(model, world, new_ents, eb.EVAL_FAMS)
    res["fisher_eval"] = eb.np_fisher(z_ho, k_ho)
    res["retrieval_eval"] = eb.np_retrieval(z_tr, k_tr, z_ho, k_ho)
    res["probe_eval"] = ridge_probe(z_tr, k_tr, z_ho, k_ho)
    return res


def consolidate(model, world, seed, n_dream, lam, *, trust=False,
                decay_eta=0.0, shadow_betas=(), record_norms=None):
    """rerun_k2_power.consolidate (shuffle=False) + safeguard hooks."""
    torch.manual_seed(1000 + seed)
    for b in model.blocks:
        b.lora["qa"].data = torch.randn(eb.LORA_RANK, eb.D_MODEL,
                                        device=eb.DEV) / math.sqrt(eb.D_MODEL)
        b.lora["va"].data = torch.randn(eb.LORA_RANK, eb.D_MODEL,
                                        device=eb.DEV) / math.sqrt(eb.D_MODEL)
        b.lora["qb"].data.zero_()
        b.lora["vb"].data.zero_()
    model.set_lora(True)
    new_ents = list(range(eb.N_BASE, eb.N_BASE + eb.N_NEW))
    dream_fams = eb.DREAM_FAM_POOL[:n_dream]
    rng = np.random.default_rng(777 + seed)          # same stream across arms
    opt = torch.optim.AdamW(model.lora_params(), lr=eb.CONSOL_LR)

    # (A, B) per block per projection: effective DeltaW = B @ A  (D x D)
    pairs = [(b.lora[a], b.lora[bb])
             for b in model.blocks for a, bb in (("qa", "qb"), ("va", "vb"))]
    shadows = {beta: [p.detach().clone() for p in model.lora_params()]
               for beta in shadow_betas}
    watch = trust or (record_norms is not None)
    n_clip = n_tot = 0

    for step in range(eb.CONSOL_STEPS):
        idx, meta = world.batch(new_ents, dream_fams, eb.CONSOL_BATCH, rng)
        logits, h = model(idx, return_hidden=True)
        L_rec = eb.lm_loss(logits, idx)
        ents = np.array([m[0] for m in meta])
        L_coh = eb.fisher_coherence(h[:, eb.POS_NAME_B], ents)
        loss = (1 - lam) * L_rec + lam * L_coh
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lora_params(), 1.0)
        if watch:
            snaps = [(A.detach().clone(), B.detach().clone(),
                      (B.detach() @ A.detach()).clone()) for A, B in pairs]
        opt.step()
        if watch:
            for (A, B), (A0, B0, BA0) in zip(pairs, snaps):
                d = B.data @ A.data - BA0
                nrm = float(torch.linalg.matrix_norm(d, ord=2))
                if record_norms is not None:
                    record_norms.append(nrm)
                n_tot += 1
                if trust and nrm > DELTA_MAX:
                    s = DELTA_MAX / nrm
                    A.data.copy_(A0 + s * (A.data - A0))
                    B.data.copy_(B0 + s * (B.data - B0))
                    n_clip += 1
        if decay_eta:
            for p in model.lora_params():
                p.data.mul_(1.0 - decay_eta)
        for beta, sh in shadows.items():
            for st, p in zip(sh, model.lora_params()):
                st.mul_(1.0 - beta).add_(p.detach(), alpha=beta)

    res = evaluate(model, world, new_ents, dream_fams)
    if trust:
        res["clip_frac"] = n_clip / max(n_tot, 1)
    for beta, sh in shadows.items():
        live = [p.detach().clone() for p in model.lora_params()]
        for p, st in zip(model.lora_params(), sh):
            p.data.copy_(st)
        sres = evaluate(model, world, new_ents, dream_fams)
        for k, v in sres.items():
            res[f"{k}_shadow_b{beta:g}"] = v
        for p, lv in zip(model.lora_params(), live):
            p.data.copy_(lv)
    model.set_lora(False)
    return res


def get_pretrained(seed):
    """Pretrain (or load cached) and freeze; returns (model, world, gates)."""
    world = eb.World(seed)
    path = os.path.join(CACHE, f"seed{seed}_p{eb.PRETRAIN_STEPS}.pt")
    if os.path.exists(path):
        model = eb.TinyLM(world.v.size).to(eb.DEV)
        model.load_state_dict(torch.load(path))
        # world.rng is only consumed by pretraining batches; consolidation
        # and every eval pass their own explicit rngs, so cache-reload is
        # trajectory-equivalent for everything downstream.
        print(f"seed {seed}: loaded cached pretrained model", flush=True)
    else:
        t0 = time.time()
        model = eb.pretrain(world, seed)
        torch.save(model.state_dict(), path)
        print(f"seed {seed}: pretrained in {time.time()-t0:.0f}s", flush=True)
    base_acc = eb.attr_acc(model, world, list(range(eb.N_BASE)),
                           list(range(eb.N_FAM)))
    zs = eb.attr_acc(model, world,
                     list(range(eb.N_BASE, eb.N_BASE + eb.N_NEW)),
                     list(range(eb.N_FAM)))
    print(f"seed {seed}: base_attr={base_acc:.3f} zero_shot={zs:.3f}",
          flush=True)
    if base_acc < 0.95:
        print(f"seed {seed}: GATE FAILED (base_attr < 0.95) — flag for review",
              flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.lora_params():
        p.requires_grad_(True)
    return model, world, base_acc, zs


def calibrate():
    """Record ||Delta(BA)||_2 during naive seed-0 runs; print percentiles."""
    model, world, _, _ = get_pretrained(0)
    norms = []
    for lam in LAMS:
        t0 = time.time()
        consolidate(model, world, 0, N_DREAM, lam, record_norms=norms)
        print(f"  calib lam={lam}: {time.time()-t0:.0f}s", flush=True)
    norms = np.array(norms)
    print(f"pair-steps recorded: {len(norms)}")
    for q in (50, 70, 75, 80, 85, 90, 95, 99):
        print(f"  p{q} = {np.percentile(norms, q):.4f}")
    for cand in (0.003, 0.004, 0.005, 0.006):
        print(f"  clip frac at delta_max={cand}: "
              f"{float(np.mean(norms > cand)):.3f}")


def run_variant(model, world, seed, lam, variant, **kw):
    t0 = time.time()
    res = consolidate(model, world, seed, N_DREAM, lam, **kw)
    res.update(seed=seed, variant=variant, lam=lam, n_dream=N_DREAM, k_base=2)
    print(f"  s{seed} lam={lam} {variant:<12} "
          f"base={res['base_attr_acc']:.3f} dream={res['attr_dream_acc']:.3f} "
          f"eval={res['attr_eval_acc']:.3f} retr={res['retrieval_eval']:.3f}"
          + (f" clip={res['clip_frac']:.2f}" if "clip_frac" in res else "")
          + f" ({time.time()-t0:.0f}s)", flush=True)
    return res


def explode_shadow_rows(res, variant, extra):
    """One row per beta; live metrics duplicated, shadow metrics inlined."""
    rows = []
    for beta in SHADOW_BETAS:
        row = {k: v for k, v in res.items() if "_shadow_b" not in k}
        suf = f"_shadow_b{beta:g}"
        for k, v in res.items():
            if k.endswith(suf):
                row[k[: -len(suf)] + "_shadow"] = v
        row.update(variant=variant, beta=beta, **extra)
        rows.append(row)
    return rows


def phase1_rows(model, world, seed, base_acc, zs):
    rows = []
    gate = dict(base_acc_pre=base_acc, new_zero_shot=zs)
    for lam in LAMS:
        r = run_variant(model, world, seed, lam, "naive")
        rows.append({**r, **gate})
        r = run_variant(model, world, seed, lam, "trust", trust=True)
        rows.append({**r, **gate, "delta_max": DELTA_MAX})
        for eta in DECAY_ETAS:
            r = run_variant(model, world, seed, lam, f"decay_eta{eta:g}",
                            decay_eta=eta)
            r["variant"] = "decay"
            rows.append({**r, **gate, "eta": eta})
        r = run_variant(model, world, seed, lam, "shadow",
                        shadow_betas=SHADOW_BETAS)
        rows.extend(explode_shadow_rows(r, "shadow", gate))
    return rows


def phase2_rows(model, world, seed, base_acc, zs, best_eta):
    rows = []
    gate = dict(base_acc_pre=base_acc, new_zero_shot=zs)
    for lam in LAMS:
        r = run_variant(model, world, seed, lam, "combo", trust=True,
                        decay_eta=best_eta, shadow_betas=SHADOW_BETAS)
        rows.extend(explode_shadow_rows(
            r, "combo", {**gate, "eta": best_eta, "delta_max": DELTA_MAX}))
    return rows


def best_eta_from(rows):
    """Committed rule: lam=0 mean(base_attr + attr_dream), higher wins."""
    scores = {}
    for eta in DECAY_ETAS:
        rs = [r for r in rows if r["variant"] == "decay"
              and r["lam"] == 0.0 and r["eta"] == eta]
        scores[eta] = float(np.mean([r["base_attr_acc"] + r["attr_dream_acc"]
                                     for r in rs]))
    best = max(scores, key=scores.get)
    print("eta scores: " + ", ".join(f"{e:g}={s:.3f}"
                                     for e, s in scores.items())
          + f" -> best {best:g}", flush=True)
    return best


def load_partials(pattern):
    rows = []
    for fn in sorted(os.listdir(PART)):
        if fn.startswith(pattern) and fn.endswith(".json"):
            with open(os.path.join(PART, fn)) as f:
                rows.extend(json.load(f))
    return rows


def main_sequential():
    all_rows, models = [], {}
    for seed in SEEDS:
        model, world, base_acc, zs = get_pretrained(seed)
        models[seed] = (model, world, base_acc, zs)
        all_rows.extend(phase1_rows(model, world, seed, base_acc, zs))
        with open(OUT, "w") as f:
            json.dump(all_rows, f, indent=1)
    best_eta = best_eta_from(all_rows)
    for seed in SEEDS:
        model, world, base_acc, zs = models[seed]
        all_rows.extend(phase2_rows(model, world, seed, base_acc, zs,
                                    best_eta))
        with open(OUT, "w") as f:
            json.dump(all_rows, f, indent=1)
    print(f"done. {len(all_rows)} rows -> {OUT}")


def supplement():
    """trust/combo rerun at the post-hoc calibrated DELTA_MAX_CAL (p80).

    Appends rows (variant trust_cal / combo_cal, delta_max=DELTA_MAX_CAL)
    to the existing results file. Uses the same best eta the main grid's
    combo used (read from the combo rows for consistency).
    """
    global DELTA_MAX
    rows = json.load(open(OUT))
    combo_etas = {r["eta"] for r in rows if r["variant"] == "combo"}
    assert len(combo_etas) == 1
    best_eta = combo_etas.pop()
    rows = [r for r in rows if r["variant"] not in ("trust_cal", "combo_cal")]
    DELTA_MAX = DELTA_MAX_CAL
    for seed in SEEDS:
        model, world, base_acc, zs = get_pretrained(seed)
        gate = dict(base_acc_pre=base_acc, new_zero_shot=zs)
        for lam in LAMS:
            r = run_variant(model, world, seed, lam, "trust_cal", trust=True)
            rows.append({**r, **gate, "delta_max": DELTA_MAX_CAL})
            r = run_variant(model, world, seed, lam, "combo_cal", trust=True,
                            decay_eta=best_eta, shadow_betas=SHADOW_BETAS)
            rows.extend(explode_shadow_rows(
                r, "combo_cal", {**gate, "eta": best_eta,
                                 "delta_max": DELTA_MAX_CAL}))
            with open(OUT, "w") as f:
                json.dump(rows, f, indent=1)
    print(f"supplement done. {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        main_sequential()
    elif args[0] == "pretrain":
        get_pretrained(int(args[1]))
    elif args[0] == "calibrate":
        calibrate()
    elif args[0] == "phase1":
        seed = int(args[1])
        model, world, base_acc, zs = get_pretrained(seed)
        rows = phase1_rows(model, world, seed, base_acc, zs)
        with open(os.path.join(PART, f"phase1_seed{seed}.json"), "w") as f:
            json.dump(rows, f, indent=1)
    elif args[0] == "besteta":
        best_eta_from(load_partials("phase1_"))
    elif args[0] == "phase2":
        seed, eta = int(args[1]), float(args[2])
        model, world, base_acc, zs = get_pretrained(seed)
        rows = phase2_rows(model, world, seed, base_acc, zs, eta)
        with open(os.path.join(PART, f"phase2_seed{seed}.json"), "w") as f:
            json.dump(rows, f, indent=1)
    elif args[0] == "supplement":
        supplement()
    elif args[0] == "merge":
        rows = load_partials("phase1_") + load_partials("phase2_")
        with open(OUT, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"merged {len(rows)} rows -> {OUT}")
    else:
        raise SystemExit(f"unknown mode {args[0]}")
