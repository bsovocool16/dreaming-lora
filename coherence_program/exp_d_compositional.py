"""
Track 2, Experiment D — Compositional comparison: does coherence pay off
when the eval task has no pre-trainable route?
=========================================================================

Motivation (from RESULTS.md): consolidating new entities via LM loss alone
produces working attribute recall that transfers across surface contexts
("routes"), but geometrically incoherent entity representations; the
coherence loss fixes the geometry but showed no functional benefit —
because the eval task (replaying the trained mapping in a new context) is
exactly what routes are good at. HYPOTHESIS: coherence's functional payoff
appears on tasks that are novel at eval time and require COMPARING
consolidated representations — tasks with no pre-trainable route.

Design
------
Grammar: the 8 declarative template families of exp_b, unchanged
(<bos> g1_f g2_f NAME_A NAME_B rel_f ATTR <eos>, SEQ_LEN=8), PLUS a 9th
COMPARISON form, same SEQ_LEN=8:

    <bos> cmp_q NAME1_A NAME1_B NAME2_A NAME2_B cmp_rel ANSWER

with two new dedicated tokens cmp_q / cmp_rel and ANSWER in
{tok_same, tok_diff}: do entity1 and entity2 have the SAME COLOR?
(4 new vocab tokens total.)

Pretraining: base entities (32), k=8 regime (every base entity in all 8
declarative families), on a mixture of ~70% declarative and ~30%
comparison sentences over base-entity pairs, comparisons balanced 50/50
same/diff color. Gate: base declarative attr acc >= 0.95 AND base
comparison acc >= 0.90 on held-out pairs. If gates fail at 8000 steps,
extend to 16000; if still failing, STOP (RuntimeError) — do not proceed
with a broken gate.

Consolidation: identical to rerun_k2_power.consolidate — 8 new entities,
dream sentences DECLARATIVE ONLY (n_dream in {2,4} families; never the
comparison form — that is the whole point), arms recon (lam=0) / cohere
(lam=0.3) / shuffled (lam=0.3, entity labels permuted per batch), paired
seeds (shared pretrained base, shared LoRA init, shared dream stream).

Evaluation (the new part; chance is 0.5 here, not 0.125):
  comp_new_new:   comparison accuracy on new x new pairs (hardest: both
                  operands consolidated, format never dreamed)
  comp_new_base:  new x base pairs (one consolidated operand; both orders)
  comp_base_base: base x base held-out pairs post-consolidation
                  (forgetting control for the comparison circuit)
plus the standard declarative metrics from rerun_k2_power (attr_dream_acc,
attr_eval_acc declarative transfer, probe_eval, retrieval_eval,
base_attr_acc, fisher_eval, lm_dream) so results join the existing tables.
Every comparison eval is balanced 50/50 same/diff, and reports accuracy
AND the same/diff confusion breakdown (a model answering all-"diff"
scores 0.5 — each arm is flagged if it degenerates like that).

Pre-registered predictions (written before any full run)
---------------------------------------------------------
  P-D1 (paper's hope): cohere > recon on comp_new_new by more than the
       declarative-transfer gap (which was ~null: -0.006/+0.005/+0.041 at
       n_dream=1/2/4 in the powered k=2 rerun). Mechanism: the coherent
       arm's entity representations plug into the base comparison circuit;
       routes don't — there is no route to replay because the comparison
       form was never dreamed.
  P-D2 (deflationary): both arms ~ 0.5 on comp_new_new (neither
       representation is readable by the frozen comparison circuit —
       consolidation into Q/V LoRA may not expose entity color to cmp
       positions at all).
  P-D3: shuffled arm <= recon everywhere (destroying entity structure in
       the auxiliary loss cannot help a task that requires entity
       structure).
  Secondary comparison of interest: comp_new_base vs comp_new_new. With
  one known operand the circuit only needs to read ONE consolidated
  color; if comp_new_base >> comp_new_new, partial readability is the
  story (each consolidated color is noisy; comparing two noisy operands
  compounds the noise). If both are flat at 0.5, P-D2's "not exposed at
  all" reading wins.

Implementation decisions (choices the spec did not fix)
--------------------------------------------------------
  1. Comparison scoring: restricted argmax over {tok_same, tok_diff} at
     the answer position, so chance is exactly 0.5. The fraction of eval
     items whose FULL-vocab argmax falls outside {tok_same, tok_diff} is
     tracked separately as *_off_menu (a high value means the circuit
     doesn't even recognize the format).
  2. 50/50 balance in eval sets is implemented as macro-averaged
     (balanced) accuracy: acc = (acc_same + acc_diff) / 2 over the
     deterministic pair sentences of each class. This is exactly a 50/50
     class weighting with no resampling noise. Per-class accuracies,
     predicted-"same" rate, and class sizes are all reported.
  3. Held-out comparison pairs are held out at the UNORDERED level
     (16 same-color + 16 diff-color unordered base pairs; both orders
     excluded from pretraining). The gate and comp_base_base evaluate
     both orders of these (64 sentences, 32 per class).
  4. "Rejection sampling" of balanced pretraining pairs is implemented as
     uniform draws from precomputed qualifying ordered-pair lists (train
     pairs only). This is distributionally identical to rejection
     sampling (uniform over qualifying pairs) and faster.
  5. World color assignment is redrawn (rejection at construction) until:
     >= 24 same-color unordered base pairs (expected ~62; guarantees
     held-out split), >= 2 same-color unordered new-new pairs (expected
     3.5; guarantees comp_new_new's same class is non-empty), and
     >= 4 same-color new-base unordered pairs (expected ~32). This mildly
     biases color multiplicity upward; without it some seeds would have
     an EMPTY same class in comp_new_new. n_same is reported per cell so
     small-n cells are visible.
  6. comp_new_new same class can be as small as 4 ordered sentences
     (2 unordered pairs) — inherent to 8 new entities over 8 colors.
     Reported, not hidden; interpret per-seed same-class accuracy as
     coarse, and lean on the seed-aggregated numbers.
  7. comp_new_base includes BOTH operand orders (new-first and
     base-first); the diff class is subsampled to <= 256 ordered pairs
     with a fixed rng for eval-size sanity. Same class is never
     subsampled.
  8. Comparison sentences have no <eos>: ANSWER occupies the final
     position of the fixed SEQ_LEN=8 frame, exactly as specified.
  9. LM loss on comparison sentences covers all positions (the name
     tokens after cmp_q are unpredictable — irreducible entropy, same
     situation as declarative filler tokens; harmless constant).
 10. Self-pairs (e1 == e2) are excluded everywhere: trivially "same" via
     surface identity, no representation comparison needed.
 11. Gate-extension rule: extend to 16000 if EITHER gate fails at 8000
     (spec names the comparison gate; declarative is included since
     proceeding with broken declaratives is equally invalid). Hard stop
     (RuntimeError) if either still fails after extension. In SMOKE mode
     the extension logic still runs (so it is exercised end-to-end) but a
     final failure warns and continues with gate_passed=False recorded,
     since 800-step gates are expected to fail.
 12. Seeds: 15 (0..14), matching rerun_k2_power. n_dream in {2,4}.
 13. Zero-shot (pre-consolidation, LoRA off) attr and comparison
     accuracies are recorded per seed: new-entity comparisons should sit
     at ~0.5 before consolidation, making all comparison transfer
     consolidation-driven by construction.
 14. Forced CPU (DEV="cpu") per spec; MPS nondeterminism avoided.

Run
---
  python3.10 -W ignore exp_d_compositional.py            # full (GPU-wait + review first!)
  SMOKE=1 python3.10 -W ignore exp_d_compositional.py    # smoke: 800/50 steps, 1 seed
Writes results_d/results_d.json (smoke: results_d/results_d_smoke.json).
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = os.environ.get("EXPD_DEV", "cpu")   # cpu default per spec; EXPD_DEV=mps
                                           # for the GPU full run (design review
                                           # 2026-07-03: CPU 15-seed projection
                                           # was 9-18h; MPS ~1/3 of that)

SMOKE = os.environ.get("SMOKE", "0") == "1"

# ----------------------------------------------------------------- grammar
N_FAM = 8
EVAL_FAMS = [6, 7]
DREAM_FAM_POOL = [0, 1, 2, 3, 4, 5]
N_BASE = int(os.environ.get("EXPD_N_BASE", "32"))
N_NEW = 8
N_FIRST = N_LAST = int(os.environ.get("EXPD_NAME_POOL", "8"))   # name token pools
N_COLOR, N_CITY = 8, 8
FILLERS_PER_FAM = 6
K_BASE_FAMS = 8                  # k=8 regime, fixed by the exp-D spec

SEQ_LEN = 8
# declarative: <bos> g1 g2 A B rel ATTR <eos>
POS_NAME_B = 4
POS_ATTR = 6
# comparison:  <bos> cmp_q N1A N1B N2A N2B cmp_rel ANSWER
POS_CMP_ANS = 7

CMP_FRAC = float(os.environ.get("CMP_FRAC", "0.30"))  # comparison share of pretrain mixture
N_HELDOUT_SAME = 16              # unordered held-out base pairs per class
N_HELDOUT_DIFF = 16
GATE_ATTR = 0.95
GATE_CMP = 0.90

# ----------------------------------------------------------------- model
D_MODEL, N_HEAD = 128, 4
N_LAYER = int(os.environ.get("EXPD_N_LAYER", "4"))
LORA_RANK = 8
PRETRAIN_STEPS = 800 if SMOKE else int(os.environ.get("PRETRAIN_STEPS", "8000"))
PRETRAIN_STEPS_EXT = 1600 if SMOKE else int(os.environ.get("PRETRAIN_EXT", "16000"))  # extension ceiling
PRETRAIN_LR = 3e-4
PRETRAIN_BATCH = 128
CONSOL_STEPS = 50 if SMOKE else 600
CONSOL_LR = 1e-3
CONSOL_BATCH = 64

N_DREAM_GRID = [2, 4]
ARMS = [("recon", 0.0, False), ("cohere", 0.3, False), ("shuffled", 0.3, True)]
_arm_filter = os.environ.get("EXPD_ARMS")          # e.g. "recon,cohere"
if _arm_filter:
    keep = set(_arm_filter.split(","))
    ARMS = [a for a in ARMS if a[0] in keep]
CONSOL_DECAY = float(os.environ.get("EXPD_DECAY", "0"))   # homeostatic decay eta
SEEDS = [0] if SMOKE else list(range(15))
EPS = 1e-6

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_d")
RESULTS_NAME = "results_d_smoke.json" if SMOKE else "results_d.json"
if CONSOL_DECAY:
    RESULTS_NAME = RESULTS_NAME.replace(".json", f"_decay{CONSOL_DECAY:g}.json")


class Vocab:
    def __init__(self):
        toks = ["<bos>", "<eos>"]
        toks += [f"first{i}" for i in range(N_FIRST)]
        toks += [f"last{i}" for i in range(N_LAST)]
        toks += [f"color{i}" for i in range(N_COLOR)]
        toks += [f"city{i}" for i in range(N_CITY)]
        toks += [f"f{f}_g{i}" for f in range(N_FAM) for i in range(FILLERS_PER_FAM)]
        toks += [f"f{f}_rel{a}" for f in range(N_FAM) for a in range(2)]
        toks += ["cmp_q", "cmp_rel", "tok_same", "tok_diff"]   # exp-D additions
        self.t2i = {t: i for i, t in enumerate(toks)}
        self.size = len(toks)

    def __getitem__(self, t):
        return self.t2i[t]


class World:
    """Entity roster, declarative sampler, and comparison-pair bookkeeping."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.v = Vocab()
        names = [(a, b) for a in range(N_FIRST) for b in range(N_LAST)]
        self.rng.shuffle(names)
        self.entities = names[: N_BASE + N_NEW]   # (first, last) pairs
        # color redraw until same-class pairs exist everywhere (decision 5)
        for _ in range(1000):
            self.color = self.rng.integers(0, N_COLOR, size=N_BASE + N_NEW)
            if (self._n_same_unordered(range(N_BASE)) >= 24
                    and self._n_same_unordered(range(N_BASE, N_BASE + N_NEW)) >= 2
                    and self._n_same_cross() >= 4):
                break
        else:
            raise RuntimeError("could not draw colors satisfying pair constraints")
        self.city = self.rng.integers(0, N_CITY, size=N_BASE + N_NEW)
        self.base_fams = {e: self.rng.permutation(N_FAM)[:K_BASE_FAMS].tolist()
                          for e in range(N_BASE)}    # k=8: all fams, kept for parity

        # ---- comparison pair sets -------------------------------------
        base = list(range(N_BASE))
        new = list(range(N_BASE, N_BASE + N_NEW))
        same = lambda a, b: self.color[a] == self.color[b]
        # unordered base pairs by class
        bb_same_u = [(a, b) for i, a in enumerate(base) for b in base[i + 1:] if same(a, b)]
        bb_diff_u = [(a, b) for i, a in enumerate(base) for b in base[i + 1:] if not same(a, b)]
        self.rng.shuffle(bb_same_u)
        self.rng.shuffle(bb_diff_u)
        held_same_u = bb_same_u[:N_HELDOUT_SAME]
        held_diff_u = bb_diff_u[:N_HELDOUT_DIFF]
        both_orders = lambda pairs: [(a, b) for (x, y) in pairs for (a, b) in ((x, y), (y, x))]
        self.held_bb_same = both_orders(held_same_u)          # comp_base_base + gate
        self.held_bb_diff = both_orders(held_diff_u)
        self.train_bb_same = both_orders(bb_same_u[N_HELDOUT_SAME:])
        self.train_bb_diff = both_orders(bb_diff_u[N_HELDOUT_DIFF:])
        # new x new ordered pairs by class
        self.nn_same = [(a, b) for a in new for b in new if a != b and same(a, b)]
        self.nn_diff = [(a, b) for a in new for b in new if a != b and not same(a, b)]
        # new x base ordered pairs by class, both operand orders (decision 7)
        nb = [(a, b) for a in new for b in base] + [(a, b) for a in base for b in new]
        self.nb_same = [p for p in nb if same(*p)]
        nb_diff = [p for p in nb if not same(*p)]
        sub = np.random.default_rng(4242 + seed)
        if len(nb_diff) > 256:
            nb_diff = [nb_diff[i] for i in sub.choice(len(nb_diff), 256, replace=False)]
        self.nb_diff = nb_diff

    def _n_same_unordered(self, ents):
        ents = list(ents)
        return sum(self.color[a] == self.color[b]
                   for i, a in enumerate(ents) for b in ents[i + 1:])

    def _n_same_cross(self):
        return sum(self.color[a] == self.color[b]
                   for a in range((N_BASE)) for b in range(N_BASE, N_BASE + N_NEW))

    # ---- sentence builders --------------------------------------------
    def sentence(self, ent, fam, attr_type, rng=None):
        rng = rng or self.rng
        v, (fa, la) = self.v, self.entities[ent]
        g = rng.integers(0, FILLERS_PER_FAM, size=2)
        attr = (f"color{self.color[ent]}" if attr_type == 0
                else f"city{self.city[ent]}")
        return [v["<bos>"], v[f"f{fam}_g{g[0]}"], v[f"f{fam}_g{g[1]}"],
                v[f"first{fa}"], v[f"last{la}"], v[f"f{fam}_rel{attr_type}"],
                v[attr], v["<eos>"]]

    def cmp_sentence(self, e1, e2):
        v = self.v
        (f1, l1), (f2, l2) = self.entities[e1], self.entities[e2]
        ans = "tok_same" if self.color[e1] == self.color[e2] else "tok_diff"
        return [v["<bos>"], v["cmp_q"], v[f"first{f1}"], v[f"last{l1}"],
                v[f"first{f2}"], v[f"last{l2}"], v["cmp_rel"], v[ans]]

    # ---- batch samplers ------------------------------------------------
    def batch(self, ents, fams, n, rng=None, restrict_base=True):
        """Declarative batch (identical to exp_b)."""
        rng = rng or self.rng
        rows, meta = [], []
        for _ in range(n):
            e = ents[rng.integers(len(ents))]
            allowed = ([f for f in fams if f in self.base_fams[e]]
                       if (e < N_BASE and restrict_base) else fams)
            if not allowed:
                continue
            f = allowed[rng.integers(len(allowed))]
            a = int(rng.integers(2))
            rows.append(self.sentence(e, f, a, rng))
            meta.append((e, f, a))
        return (torch.tensor(rows, dtype=torch.long, device=DEV),
                np.array(meta))

    def mixed_batch(self, n, rng):
        """Pretraining mixture: ~70% declarative, ~30% balanced comparisons."""
        base_ents = list(range(N_BASE))
        fams = list(range(N_FAM))
        rows = []
        for _ in range(n):
            if rng.random() < CMP_FRAC:
                pool = self.train_bb_same if rng.random() < 0.5 else self.train_bb_diff
                e1, e2 = pool[rng.integers(len(pool))]
                rows.append(self.cmp_sentence(e1, e2))
            else:
                e = base_ents[rng.integers(N_BASE)]
                allowed = self.base_fams[e]
                f = allowed[rng.integers(len(allowed))]
                a = int(rng.integers(2))
                rows.append(self.sentence(e, f, a, rng))
        return torch.tensor(rows, dtype=torch.long, device=DEV)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.vp = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.o = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                nn.Linear(4 * D_MODEL, D_MODEL))
        self.lora = nn.ParameterDict({
            "qa": nn.Parameter(torch.randn(LORA_RANK, D_MODEL) / math.sqrt(D_MODEL)),
            "qb": nn.Parameter(torch.zeros(D_MODEL, LORA_RANK)),
            "va": nn.Parameter(torch.randn(LORA_RANK, D_MODEL) / math.sqrt(D_MODEL)),
            "vb": nn.Parameter(torch.zeros(D_MODEL, LORA_RANK)),
        })
        self.lora_on = False

    def attn(self, x):
        Bb, T, C = x.shape
        q = self.q(x)
        vv = self.vp(x)
        if self.lora_on:
            q = q + x @ self.lora["qa"].T @ self.lora["qb"].T
            vv = vv + x @ self.lora["va"].T @ self.lora["vb"].T
        k = self.k(x)
        hd = C // N_HEAD
        q, k, vv = (t.view(Bb, T, N_HEAD, hd).transpose(1, 2) for t in (q, k, vv))
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        return self.o((att @ vv).transpose(1, 2).reshape(Bb, T, C))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)

    def forward(self, idx, return_hidden=False):
        T = idx.shape[1]
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        for b in self.blocks:
            x = b(x)
        h = self.lnf(x)
        logits = self.head(h)
        return (logits, h) if return_hidden else logits

    def set_lora(self, on):
        for b in self.blocks:
            b.lora_on = on

    def lora_params(self):
        return [p for b in self.blocks for p in b.lora.values()]


def lm_loss(logits, idx):
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           idx[:, 1:].reshape(-1))


def fisher_coherence(h_nameB, ents):
    """Within-entity dispersion / between-entity variance, normalized reps."""
    zn = F.normalize(h_nameB, dim=-1)
    uniq = sorted(set(ents.tolist()))
    mus, within = [], []
    for e in uniq:
        m = torch.tensor([x == e for x in ents.tolist()], device=zn.device)
        grp = zn[m]
        mu = grp.mean(0)
        mus.append(mu)
        within.append(((grp - mu) ** 2).sum(-1).mean())
    mus = torch.stack(mus)
    within = torch.stack(within).mean()
    between = ((mus - mus.mean(0)) ** 2).sum(-1).mean()
    return within / (between + EPS)


# ----------------------------------------------------------------- metrics
@torch.no_grad()
def attr_acc(model, world, ents, fams, n=512):
    rng = np.random.default_rng(12345)
    idx, meta = world.batch(ents, fams, n, rng)
    logits = model(idx)
    pred = logits[:, POS_ATTR - 1].argmax(-1)
    return float((pred == idx[:, POS_ATTR]).float().mean())


@torch.no_grad()
def name_reps(model, world, ents, fams, n=512):
    rng = np.random.default_rng(54321)
    idx, meta = world.batch(ents, fams, n, rng)
    _, h = model(idx, return_hidden=True)
    return h[:, POS_NAME_B].cpu().numpy(), np.array([m[0] for m in meta])


def np_fisher(z, ks):
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + EPS)
    uniq = sorted(set(ks.tolist()))
    mus = np.stack([zn[ks == k].mean(0) for k in uniq])
    within = np.mean([np.mean(np.sum((zn[ks == k] - mus[i]) ** 2, axis=1))
                      for i, k in enumerate(uniq)])
    between = np.mean(np.sum((mus - mus.mean(0)) ** 2, axis=1))
    return float(within / (between + EPS))


def np_retrieval(z_tr, k_tr, z_ho, k_ho):
    zn = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)
    uniq = sorted(set(k_tr.tolist()))
    cents = np.stack([zn(z_tr)[k_tr == k].mean(0) for k in uniq])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + EPS
    pred = np.array(uniq)[np.argmax(zn(z_ho) @ cents.T, axis=1)]
    return float(np.mean(pred == k_ho))


def ridge_probe(z_tr, k_tr, z_ho, k_ho, reg=1.0):
    """Linear probe on un-normalized states (from rerun_k2_power)."""
    ents = sorted(set(k_tr.tolist()))
    Y = np.stack([(k_tr == e).astype(float) for e in ents], axis=1)
    X = np.concatenate([z_tr, np.ones((len(z_tr), 1))], axis=1)
    W = np.linalg.solve(X.T @ X + reg * np.eye(X.shape[1]), X.T @ Y)
    Xh = np.concatenate([z_ho, np.ones((len(z_ho), 1))], axis=1)
    pred = np.array(ents)[np.argmax(Xh @ W, axis=1)]
    return float(np.mean(pred == k_ho))


@torch.no_grad()
def comp_eval(model, world, pairs_same, pairs_diff):
    """Balanced comparison eval on deterministic pair sentences.

    Primary accuracy is the macro-average of per-class accuracies under a
    restricted argmax over {tok_same, tok_diff} (chance = 0.5 exactly).
    Also reports the same/diff confusion breakdown, the predicted-"same"
    rate, class sizes, and the off-menu rate (full-vocab argmax outside
    the answer pair).
    """
    v = world.v
    i_same, i_diff = v["tok_same"], v["tok_diff"]
    pairs = list(pairs_same) + list(pairs_diff)
    labels = np.array([1] * len(pairs_same) + [0] * len(pairs_diff))  # 1 = same
    idx = torch.tensor([world.cmp_sentence(a, b) for a, b in pairs],
                       dtype=torch.long, device=DEV)
    logits = model(idx)[:, POS_CMP_ANS - 1]                 # predict ANSWER slot
    pred_same = (logits[:, i_same] > logits[:, i_diff]).cpu().numpy().astype(int)
    full_arg = logits.argmax(-1).cpu().numpy()
    off_menu = float(np.mean(~np.isin(full_arg, [i_same, i_diff])))
    acc_same = float(np.mean(pred_same[labels == 1] == 1)) if len(pairs_same) else float("nan")
    acc_diff = float(np.mean(pred_same[labels == 0] == 0)) if len(pairs_diff) else float("nan")
    return {
        "acc": 0.5 * (acc_same + acc_diff),                 # balanced accuracy
        "acc_same": acc_same,
        "acc_diff": acc_diff,
        "pred_same_rate": float(np.mean(pred_same)),
        "off_menu": off_menu,
        "n_same": len(pairs_same),
        "n_diff": len(pairs_diff),
    }


def comp_metrics(model, world, key, res):
    """Run the three comparison evals and flatten into res under prefixes."""
    evals = {
        "comp_nn": (world.nn_same, world.nn_diff),          # new x new
        "comp_nb": (world.nb_same, world.nb_diff),          # new x base
        "comp_bb": (world.held_bb_same, world.held_bb_diff),  # held-out base x base
    }
    for prefix, (ps, pd) in evals.items():
        d = comp_eval(model, world, ps, pd)
        for k2, v2 in d.items():
            res[f"{prefix}_{k2}"] = v2
    return res


def degenerate_flags(res):
    """Flag arms answering (nearly) all-same or all-diff (score 0.5 trap)."""
    flags = []
    for prefix in ("comp_nn", "comp_nb", "comp_bb"):
        r = res[f"{prefix}_pred_same_rate"]
        if r <= 0.05 or r >= 0.95:
            flags.append(f"{prefix}:pred_same_rate={r:.2f}")
    return flags


# ----------------------------------------------------------------- phases
def gate_check(model, world):
    base_ents = list(range(N_BASE))
    attr = attr_acc(model, world, base_ents, list(range(N_FAM)))
    cmp_ho = comp_eval(model, world, world.held_bb_same, world.held_bb_diff)
    passed = (attr >= GATE_ATTR) and (cmp_ho["acc"] >= GATE_CMP)
    return passed, attr, cmp_ho


def pretrain(world, seed):
    """Mixture pretraining with the two-gate / extend / stop logic."""
    torch.manual_seed(seed)
    model = TinyLM(world.v.size).to(DEV)
    model.set_lora(False)
    opt = torch.optim.AdamW((p for n, p in model.named_parameters()
                             if "lora" not in n), lr=PRETRAIN_LR)
    rng = np.random.default_rng(20000 + seed)
    log_every = max(PRETRAIN_STEPS // 4, 1)

    def run(n_steps):
        for step in range(n_steps):
            idx = world.mixed_batch(PRETRAIN_BATCH, rng)
            loss = lm_loss(model(idx), idx)
            opt.zero_grad(); loss.backward(); opt.step()
            if (step + 1) % log_every == 0:
                _, a, c = gate_check(model, world)
                print(f"  pretrain +{step+1}: lm={loss.item():.3f} "
                      f"base_attr={a:.3f} cmp_ho={c['acc']:.3f}", flush=True)

    run(PRETRAIN_STEPS)
    passed, attr, cmp_ho = gate_check(model, world)
    extended = False
    if not passed:
        extended = True
        print(f"  GATE MISS at {PRETRAIN_STEPS} steps "
              f"(attr={attr:.3f}/{GATE_ATTR}, cmp={cmp_ho['acc']:.3f}/{GATE_CMP}) "
              f"— extending to {PRETRAIN_STEPS_EXT}", flush=True)
        run(PRETRAIN_STEPS_EXT - PRETRAIN_STEPS)
        passed, attr, cmp_ho = gate_check(model, world)
    if not passed:
        msg = (f"seed {seed}: pretrain gate FAILED after {PRETRAIN_STEPS_EXT} steps: "
               f"base_attr={attr:.3f} (>= {GATE_ATTR}), "
               f"cmp_heldout={cmp_ho['acc']:.3f} (>= {GATE_CMP}, "
               f"same={cmp_ho['acc_same']:.3f} diff={cmp_ho['acc_diff']:.3f})")
        if SMOKE:
            print(f"  [SMOKE] {msg} — continuing anyway (gate_passed=False)",
                  flush=True)
        else:
            raise RuntimeError(msg + " — stopping, do not run consolidation "
                               "on a broken gate.")
    gate_info = {"gate_passed": bool(passed), "gate_extended": extended,
                 "gate_attr": attr, "gate_cmp": cmp_ho["acc"],
                 "gate_cmp_same": cmp_ho["acc_same"],
                 "gate_cmp_diff": cmp_ho["acc_diff"]}
    return model, gate_info


def consolidate(model, world, seed, n_dream, lam, shuffle):
    """Identical to rerun_k2_power.consolidate: declarative dreams only."""
    torch.manual_seed(1000 + seed)
    for b in model.blocks:
        b.lora["qa"].data = torch.randn(LORA_RANK, D_MODEL, device=DEV) / math.sqrt(D_MODEL)
        b.lora["va"].data = torch.randn(LORA_RANK, D_MODEL, device=DEV) / math.sqrt(D_MODEL)
        b.lora["qb"].data.zero_()
        b.lora["vb"].data.zero_()
    model.set_lora(True)
    new_ents = list(range(N_BASE, N_BASE + N_NEW))
    dream_fams = DREAM_FAM_POOL[:n_dream]
    rng = np.random.default_rng(777 + seed)          # same dream stream per arm
    shuffle_rng = np.random.default_rng(31337 + seed)
    opt = torch.optim.AdamW(model.lora_params(), lr=CONSOL_LR)
    for step in range(CONSOL_STEPS):
        idx, meta = world.batch(new_ents, dream_fams, CONSOL_BATCH, rng)
        logits, h = model(idx, return_hidden=True)
        L_rec = lm_loss(logits, idx)
        ents = np.array([m[0] for m in meta])
        if shuffle:
            ents = shuffle_rng.permutation(ents)
        L_coh = fisher_coherence(h[:, POS_NAME_B], ents)
        loss = (1 - lam) * L_rec + lam * L_coh
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lora_params(), 1.0)
        opt.step()
        if CONSOL_DECAY:
            with torch.no_grad():
                for p in model.lora_params():
                    p.data.mul_(1.0 - CONSOL_DECAY)

    # ------- evaluation (LoRA stays ON: consolidated model)
    base_ents = list(range(N_BASE))
    res = {
        "attr_eval_acc": attr_acc(model, world, new_ents, EVAL_FAMS),
        "attr_dream_acc": attr_acc(model, world, new_ents, dream_fams),
        "base_attr_acc": attr_acc(model, world, base_ents, list(range(N_FAM))),
    }
    with torch.no_grad():
        idx, _ = world.batch(new_ents, dream_fams, 512, np.random.default_rng(99))
        res["lm_dream"] = float(lm_loss(model(idx), idx).item())
    z_tr, k_tr = name_reps(model, world, new_ents, dream_fams)
    z_ho, k_ho = name_reps(model, world, new_ents, EVAL_FAMS)
    res["fisher_eval"] = np_fisher(z_ho, k_ho)
    res["retrieval_eval"] = np_retrieval(z_tr, k_tr, z_ho, k_ho)
    res["probe_eval"] = ridge_probe(z_tr, k_tr, z_ho, k_ho)
    comp_metrics(model, world, "comp", res)          # the new evals
    model.set_lora(False)
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []
    for seed in SEEDS:
        t0 = time.time()
        world = World(seed)
        print(f"=== seed {seed}: pretraining (mixture, k={K_BASE_FAMS}) ===", flush=True)
        model, gate_info = pretrain(world, seed)
        new_ents = list(range(N_BASE, N_BASE + N_NEW))
        # zero-shot (pre-consolidation, LoRA off) baselines
        zs_attr = attr_acc(model, world, new_ents, list(range(N_FAM)))
        zs = {}
        comp_metrics(model, world, "comp", zs)
        print(f"  gate: attr={gate_info['gate_attr']:.3f} "
              f"cmp={gate_info['gate_cmp']:.3f} passed={gate_info['gate_passed']} "
              f"| zero-shot: attr={zs_attr:.3f} comp_nn={zs['comp_nn_acc']:.3f} "
              f"comp_nb={zs['comp_nb_acc']:.3f} | pairs: nn_same={zs['comp_nn_n_same']} "
              f"nb_same={zs['comp_nb_n_same']} | {time.time()-t0:.0f}s", flush=True)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.lora_params():
            p.requires_grad_(True)
        for n_dream in N_DREAM_GRID:
            for arm, lam, shuf in ARMS:
                t1 = time.time()
                m = consolidate(model, world, seed, n_dream, lam, shuf)
                m.update(seed=seed, n_dream=n_dream, arm=arm, lam=lam,
                         shuffled=shuf, k_base=K_BASE_FAMS, smoke=SMOKE,
                         new_zero_shot_attr=zs_attr,
                         zs_comp_nn_acc=zs["comp_nn_acc"],
                         zs_comp_nb_acc=zs["comp_nb_acc"],
                         zs_comp_bb_acc=zs["comp_bb_acc"],
                         **gate_info)
                flags = degenerate_flags(m)
                m["degenerate_flags"] = flags
                all_results.append(m)
                print(f"  seed={seed} n_dream={n_dream} arm={arm}: "
                      f"comp_nn={m['comp_nn_acc']:.3f} "
                      f"(s={m['comp_nn_acc_same']:.2f}/d={m['comp_nn_acc_diff']:.2f}) "
                      f"comp_nb={m['comp_nb_acc']:.3f} comp_bb={m['comp_bb_acc']:.3f} "
                      f"| attr_eval={m['attr_eval_acc']:.3f} "
                      f"dream={m['attr_dream_acc']:.3f} "
                      f"probe={m['probe_eval']:.3f} retr={m['retrieval_eval']:.3f} "
                      f"base={m['base_attr_acc']:.3f}"
                      f"{' DEGEN[' + ','.join(flags) + ']' if flags else ''} "
                      f"({time.time()-t1:.0f}s)", flush=True)
                with open(os.path.join(OUT_DIR, RESULTS_NAME), "w") as f:
                    json.dump(all_results, f, indent=1)
    print("done.")


if __name__ == "__main__":
    main()
