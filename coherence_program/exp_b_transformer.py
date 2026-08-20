"""
Track 2, Experiment B — Coherence loss on a toy transformer with LoRA
======================================================================

The first test of SS3.4 on an actual transformer. A small causal LM is
pretrained on a synthetic entity-attribute grammar, frozen, and then a LoRA
adapter on the Q/V projections consolidates facts about NEW entities from
"dream" sentences produced by context transplantation (the SS5.4 mechanism,
operationalized): the same (entity, attribute) content realized in the
surface phrasing of different template FAMILIES (= contexts).

Grammar
-------
  sentence(family f, entity e, attr type a):
    <bos> g1_f g2_f NAME_A(e) NAME_B(e) rel_f(a) ATTR(e, a) <eos>
  where rel_f(a) is family f's private relation token for attribute type a
  (the cue for WHICH attribute follows — without it the attr slot is
  irreducibly 50/50 ambiguous, which capped the first run at 0.49) and
  g*_f are drawn per-sentence from family f's private filler pool
  (within-family variability), and the filler POOLS are disjoint across
  families (between-family context signal). Each entity is a unique
  2-token name with a color and a city attribute.

Phases
------
  1. Pretrain (per seed): base entities (32) in ALL 8 families, both attrs.
     Frozen afterwards. Gate: held-out base attribute accuracy >= 0.95.
  2. Consolidate: 8 new entities, dream sentences in n_dream in {1,2,4}
     families (families 0..5 only; 6,7 are EVAL-ONLY, never dreamed).
     LoRA rank 8 on W_Q, W_V of all layers, base frozen.
     Loss: (1-lam)*LM + lam*Fisher-coherence on the last-layer hidden state
     at the final name token, grouped by entity across the batch.
     lam = 0 is the reconstruction-only baseline. All lam share the same
     pretrained base, same dream stream (paired comparison).

Evaluation
----------
  attr_eval_acc:   attribute prediction for NEW entities in families 6,7
                   (never dreamed) — cross-context transfer, the headline.
  attr_dream_acc:  same in dreamed families (did consolidation work at all).
  fisher_eval:     Fisher ratio of name representations across eval families.
  retrieval_eval:  nearest-centroid entity ID of eval-family name reps using
                   dream-family centroids.
  base_attr_acc:   base entities, all families (forgetting control).
  lm_dream:        LM loss on dream-domain text (matched-recon control).

Pre-registered predictions (written before first full run)
-----------------------------------------------------------
  P-B1 (paper): at low dream diversity (n_dream=1,2), composite lam>0 beats
       recon-only on attr_eval_acc and retrieval_eval at <=2% base_attr_acc
       cost and similar lm_dream.
  P-B2 (Option B): at n_dream=4, recon-only closes the gap (emergent
       abstraction from dream diversity alone). Gap < 5 points => coherence
       not load-bearing at that diversity.
  P-B3: fisher_eval correlates with attr_eval_acc across cells — the
       geometric property the loss targets is the one that carries the
       functional transfer. If it doesn't correlate, the coherence story
       needs rework regardless of who wins.

Committed decisions:
  - AdamW for both pretrain and consolidation (deviation from the paper's
    plain-GD update, chosen for optimization robustness; does not affect
    the recon-vs-composite comparison, which is the question).
  - consolidation steps fixed (600), no early stopping, no eval peeking.
  - 3 seeds; every (n_dream, lam) cell shares the seed's pretrained base.
"""

import json
import math
import os
import time

import numpy as np
import torch

K_BASE_FAMS = int(os.environ.get("K_BASE_FAMS", "8"))
OUT_SUFFIX = "" if K_BASE_FAMS == 8 else f"_k{K_BASE_FAMS}"
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

# ----------------------------------------------------------------- grammar
N_FAM = 8
EVAL_FAMS = [6, 7]
DREAM_FAM_POOL = [0, 1, 2, 3, 4, 5]
N_BASE = 32
N_NEW = 8
N_FIRST, N_LAST = 8, 8           # name token pools
N_COLOR, N_CITY = 8, 8
FILLERS_PER_FAM = 6

SEQ_LEN = 8                      # <bos> g1 g2 A B g3 ATTR <eos>
POS_NAME_B = 4                   # index of second name token
POS_ATTR = 6

# ----------------------------------------------------------------- model
D_MODEL, N_HEAD, N_LAYER = 128, 4, 4
LORA_RANK = 8
PRETRAIN_STEPS = 8000
PRETRAIN_LR = 3e-4
PRETRAIN_BATCH = 128
CONSOL_STEPS = 600
CONSOL_LR = 1e-3
CONSOL_BATCH = 64

N_DREAM_GRID = [1, 2, 4]
LAM_GRID = [0.0, 0.1, 0.3, 0.5, 0.7]
SEEDS = [0, 1, 2]
EPS = 1e-6

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_b")
RESULTS_NAME = f"results_b{OUT_SUFFIX}.json"


class Vocab:
    def __init__(self):
        toks = ["<bos>", "<eos>"]
        toks += [f"first{i}" for i in range(N_FIRST)]
        toks += [f"last{i}" for i in range(N_LAST)]
        toks += [f"color{i}" for i in range(N_COLOR)]
        toks += [f"city{i}" for i in range(N_CITY)]
        toks += [f"f{f}_g{i}" for f in range(N_FAM) for i in range(FILLERS_PER_FAM)]
        toks += [f"f{f}_rel{a}" for f in range(N_FAM) for a in range(2)]
        self.t2i = {t: i for i, t in enumerate(toks)}
        self.size = len(toks)

    def __getitem__(self, t):
        return self.t2i[t]


class World:
    """Entity roster and sentence sampler for one seed."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.v = Vocab()
        names = [(a, b) for a in range(N_FIRST) for b in range(N_LAST)]
        self.rng.shuffle(names)
        self.entities = names[: N_BASE + N_NEW]   # (first, last) pairs
        self.color = self.rng.integers(0, N_COLOR, size=N_BASE + N_NEW)
        self.city = self.rng.integers(0, N_CITY, size=N_BASE + N_NEW)
        # base-abstraction knob: each base entity appears in only K_BASE_FAMS
        # families during pretraining. k=8 -> base binding circuit is fully
        # family-invariant (run 1 showed eval transfer is then free and the
        # functional metric ceilings); k<8 -> binding is family-entangled and
        # cross-family transfer of NEW bindings must be earned.
        self.base_fams = {e: self.rng.permutation(N_FAM)[:K_BASE_FAMS].tolist()
                          for e in range(N_BASE)}

    def sentence(self, ent, fam, attr_type, rng=None):
        rng = rng or self.rng
        v, (fa, la) = self.v, self.entities[ent]
        g = rng.integers(0, FILLERS_PER_FAM, size=2)
        attr = (f"color{self.color[ent]}" if attr_type == 0
                else f"city{self.city[ent]}")
        return [v["<bos>"], v[f"f{fam}_g{g[0]}"], v[f"f{fam}_g{g[1]}"],
                v[f"first{fa}"], v[f"last{la}"], v[f"f{fam}_rel{attr_type}"],
                v[attr], v["<eos>"]]

    def batch(self, ents, fams, n, rng=None, restrict_base=True):
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
        # LoRA factors for q and v projections (zero-init B per paper SS5.7)
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


# ----------------------------------------------------------------- phases
def pretrain(world, seed):
    torch.manual_seed(seed)
    model = TinyLM(world.v.size).to(DEV)
    model.set_lora(False)
    opt = torch.optim.AdamW((p for n, p in model.named_parameters()
                             if "lora" not in n), lr=PRETRAIN_LR)
    base_ents = list(range(N_BASE))
    fams = list(range(N_FAM))
    for step in range(PRETRAIN_STEPS):
        idx, _ = world.batch(base_ents, fams, PRETRAIN_BATCH)
        loss = lm_loss(model(idx), idx)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 2000 == 0:
            acc = attr_acc(model, world, base_ents, fams)
            print(f"  pretrain {step+1}: lm={loss.item():.3f} base_attr={acc:.3f}",
                  flush=True)
    return model


@torch.no_grad()
def attr_acc(model, world, ents, fams, n=512):
    rng = np.random.default_rng(12345)
    idx, meta = world.batch(ents, fams, n, rng)
    logits = model(idx)
    pred = logits[:, POS_ATTR - 1].argmax(-1)        # token predicted AT attr slot
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


def consolidate(model, world, seed, n_dream, lam):
    """LoRA training on dream sentences. Returns metrics dict."""
    # reset LoRA to init (same init per seed: paired across lam)
    torch.manual_seed(1000 + seed)
    for b in model.blocks:
        b.lora["qa"].data = torch.randn(LORA_RANK, D_MODEL, device=DEV) / math.sqrt(D_MODEL)
        b.lora["va"].data = torch.randn(LORA_RANK, D_MODEL, device=DEV) / math.sqrt(D_MODEL)
        b.lora["qb"].data.zero_()
        b.lora["vb"].data.zero_()
    model.set_lora(True)
    new_ents = list(range(N_BASE, N_BASE + N_NEW))
    dream_fams = DREAM_FAM_POOL[:n_dream]
    rng = np.random.default_rng(777 + seed)          # same dream stream per lam
    opt = torch.optim.AdamW(model.lora_params(), lr=CONSOL_LR)
    for step in range(CONSOL_STEPS):
        idx, meta = world.batch(new_ents, dream_fams, CONSOL_BATCH, rng)
        logits, h = model(idx, return_hidden=True)
        L_rec = lm_loss(logits, idx)
        L_coh = fisher_coherence(h[:, POS_NAME_B], np.array([m[0] for m in meta]))
        loss = (1 - lam) * L_rec + lam * L_coh
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lora_params(), 1.0)
        opt.step()

    # ------- evaluation
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
    model.set_lora(False)
    return res


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []
    for seed in SEEDS:
        t0 = time.time()
        world = World(seed)
        print(f"=== seed {seed}: pretraining ===", flush=True)
        model = pretrain(world, seed)
        base_acc = attr_acc(model, world, list(range(N_BASE)), list(range(N_FAM)))
        # also: zero-shot performance on new entities before any consolidation
        zs = attr_acc(model, world, list(range(N_BASE, N_BASE + N_NEW)),
                      list(range(N_FAM)))
        print(f"  base_attr={base_acc:.3f} (gate >=0.95) | new-entity zero-shot={zs:.3f} "
              f"| {time.time()-t0:.0f}s", flush=True)
        if base_acc < 0.95:
            print("  GATE FAILED — recording and continuing (flag for review)")
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.lora_params():
            p.requires_grad_(True)
        for n_dream in N_DREAM_GRID:
            for lam in LAM_GRID:
                t1 = time.time()
                m = consolidate(model, world, seed, n_dream, lam)
                m.update(seed=seed, n_dream=n_dream, lam=lam, k_base=K_BASE_FAMS,
                         base_acc_pre=base_acc, new_zero_shot=zs)
                all_results.append(m)
                print(f"  seed={seed} n_dream={n_dream} lam={lam}: "
                      f"eval={m['attr_eval_acc']:.3f} dream={m['attr_dream_acc']:.3f} "
                      f"retr={m['retrieval_eval']:.3f} fish={m['fisher_eval']:.2f} "
                      f"base={m['base_attr_acc']:.3f} ({time.time()-t1:.0f}s)",
                      flush=True)
                with open(os.path.join(OUT_DIR, RESULTS_NAME), "w") as f:
                    json.dump(all_results, f, indent=1)
    print("done.")


if __name__ == "__main__":
    main()
