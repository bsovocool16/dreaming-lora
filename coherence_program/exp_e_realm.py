"""
Experiment E — The pivotal real-LM test (GPT-2, surgical Q/V LoRA)
===================================================================

First test of the program's two live questions on a real pretrained LM:

  Q1 (type token): does the coherence loss still pin cross-context entity
     identity when "contexts" are genuine English registers?
  Q2 (route-boundness / notes-vs-weights): can a real LM's in-context
     machinery CONSUME weight-consolidated facts — compose them, answer in
     novel formats — or does route-boundness replicate at scale? Measured
     head-to-head against the same facts supplied in context ("notes").

Design
------
  Model: GPT-2 (117M) via HF, frozen. LoRA rank 8 on the Q and V slices of
  c_attn only (columns 0:768 and 1536:2304), all 12 layers. Homeostatic
  decay eta=1e-3 on all arms (program standard after the safeguards grid).

  World (per seed): 12 consolidation entities + 6 extra context-only
  entities for few-shot format examples; 2-token-friendly invented names;
  attributes: occupation (8 pool) and city (6 pool -> guaranteed same-city
  pairs). 8 surface registers (news, dialogue, bio, list, memo, story,
  interview, reference); dreams use DREAM_REGISTERS, eval uses held-out
  registers + QA/composition formats never dreamed.

  Arms (paired seeds, fixed lambda): recon (LM loss only), cohere
  (+0.3 * Fisher on name-final-token last-hidden states grouped by
  entity), shuffled (labels permuted per batch).

Evaluations (restricted-choice logprob scoring throughout; no free
generation):
  gate/composition, 3 conditions x 2 formats:
    same-city yes/no ("Do A and B live in the same city?"), balanced;
    two-choice ("Which of them lives in CITY: A or B?").
    conditions: CONTEXT (facts paragraph in prompt, adapter off),
    WEIGHTS (adapter on, no facts in prompt), BOTH. All prompts carry
    2 few-shot examples built from the extra entities with their facts
    stated in-prompt (format teaching without leaking eval facts).
    GATE: CONTEXT condition must clear 0.70 on both formats, else the
    consumer circuit doesn't exist at this scale -> escalate model size.
  declarative QA (weights-only): "What city does N live in?" restricted
    to the city pool; same for occupation. The direct-recall measure.
  type-token battery: ridge probe + nearest-centroid retrieval + Fisher on
    name-final-token states, train on dream registers / test on held-out
    registers.
  forgetting: perplexity on a fixed embedded natural-text sample, pre/post.

Pre-registered predictions (written before the first full run)
---------------------------------------------------------------
  P-P1: type-token replicates — cohere > recon on held-out-register probe
        and retrieval, direction as in the toy (large, most seeds).
  P-P2: route-boundness replicates — WEIGHTS-only composition ~ chance
        (0.5) for both arms while CONTEXT clears the gate; declarative QA
        (weights) substantially above chance (the route works).
  P-P3: cohere x composition interaction — OPEN. If cohere-WEIGHTS
        composition beats recon-WEIGHTS, that is the field-relevant
        positive the toy could not produce.
  P-P4: honest expectation — CONTEXT >= WEIGHTS on everything at this
        scale (notes win; the 8B run is where this gets its real test).

Committed: fixed lambda=0.3; 5 seeds; fixed steps, no early stopping;
paired dream streams and LoRA inits across arms per seed.
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = os.environ.get("EXPE_MODEL", "gpt2")
SMOKE = os.environ.get("SMOKE", "") == "1"

LORA_RANK = 8
LAMBDA = 0.3
DECAY = 1e-3
LR = float(os.environ.get("EXPE_LR", "3e-4"))
STEPS = 40 if SMOKE else int(os.environ.get("EXPE_STEPS", "400"))
BATCH_ENTS = 6          # entities per batch
BATCH_REGS = 2          # registers per entity per batch (coherence needs >=2)
SEEDS = [0] if SMOKE else [0, 1, 2, 3, 4]
ARMS = [("recon", 0.0, False), ("cohere", LAMBDA, False), ("shuffled", LAMBDA, True)]

N_ENTS = 12             # consolidation entities
N_EXTRA = 6             # few-shot/context-only entities
DREAM_REGISTERS = [0, 1, 2, 3]
HELD_REGISTERS = [4, 5, 6, 7]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_e")
RESULTS_NAME = "results_e_smoke.json" if SMOKE else "results_e.json"

FIRST = ["Mara", "Tobin", "Rena", "Calder", "Isla", "Dorian",
         "Petra", "Silas", "Vera", "Orin", "Lena", "Hollis",
         "Nadia", "Ewan", "Tessa", "Bram", "Odile", "Caspar"]
LAST = ["Voss", "Ashe", "Hale", "Brandt", "Marchetti", "Okafor",
        "Lindqvist", "Farrow", "Kessler", "Duval", "Mercer", "Ito",
        "Reyes", "Thorne", "Novak", "Whitfield", "Aldana", "Grieve"]
OCCS = ["architect", "marine biologist", "pastry chef", "violinist",
        "cartographer", "electrician", "translator", "beekeeper"]
CITIES = ["Lisbon", "Osaka", "Denver", "Tallinn", "Cusco", "Adelaide"]

TEMPLATES = [
    # 0 news
    "Local sources report that {name}, a {occ} based in {city}, attended the ceremony on Friday.",
    # 1 dialogue
    "\"Have you met {name}?\" she asked. \"The {occ}? Sure — moved to {city} a while back.\"",
    # 2 bio
    "{name} is a {occ} who lives and works in {city}.",
    # 3 list
    "Attendees: {name} ({occ}, {city}); others to be confirmed.",
    # 4 memo (held out)
    "MEMO: Please forward the contract to {name}, our {occ} in {city}, before Thursday.",
    # 5 story (held out)
    "The morning fog had barely lifted over {city} when {name} set out, thinking about the {occ}'s work that waited at home.",
    # 6 interview (held out)
    "Interviewer: And your profession? {name}: I work as a {occ}. Interviewer: Still in {city}? {name}: That's right.",
    # 7 reference (held out)
    "{name} — occupation: {occ}; city of residence: {city}.",
]

NATURAL_TEXT = (
    "The history of bridge construction is in large part a history of "
    "materials. Early builders worked with stone and timber, shaping arches "
    "whose geometry did the work that later generations would assign to "
    "steel. When iron became cheap enough to use at scale, spans grew "
    "longer and the profession of engineering grew with them, developing "
    "methods for calculating loads that had previously been matters of "
    "judgment and inherited rule. The twentieth century added reinforced "
    "concrete and high-strength cable, and with them forms that would have "
    "seemed impossible a century earlier. A suspension bridge is, among "
    "other things, an argument about tension: the deck hangs from cables "
    "that pull inward on their towers, and the towers survive by pushing "
    "that force down into rock. Modern designers model these forces in "
    "software, but the underlying logic has not changed since the first "
    "rope was slung across a river gorge. What has changed is the margin "
    "of error, and the confidence with which a designer can promise that a "
    "structure carrying thousands of strangers will stand for a hundred "
    "years in wind, rain, and the slow chemistry of corrosion."
)


# ------------------------------------------------------------------ LoRA
class QVLoRA(nn.Module):
    """Wraps GPT-2's c_attn (Conv1D: x @ W + b, W (768, 2304)) with rank-r
    LoRA on the Q (0:d) and V (2d:3d) output slices only."""

    def __init__(self, c_attn, d_model, rank, seed_offset):
        super().__init__()
        self.base = c_attn
        for p in self.base.parameters():
            p.requires_grad_(False)
        g = torch.Generator().manual_seed(seed_offset)
        self.qa = nn.Parameter(torch.randn(rank, d_model, generator=g) / math.sqrt(d_model))
        self.qb = nn.Parameter(torch.zeros(d_model, rank))
        self.va = nn.Parameter(torch.randn(rank, d_model, generator=g) / math.sqrt(d_model))
        self.vb = nn.Parameter(torch.zeros(d_model, rank))
        self.d = d_model
        self.on = True

    def forward(self, x):
        out = self.base(x)
        if self.on:
            dq = (x @ self.qa.T) @ self.qb.T
            dv = (x @ self.va.T) @ self.vb.T
            out = torch.cat([out[..., :self.d] + dq,
                             out[..., self.d:2 * self.d],
                             out[..., 2 * self.d:] + dv], dim=-1)
        return out

    def reinit(self, seed_offset):
        g = torch.Generator().manual_seed(seed_offset)
        with torch.no_grad():
            self.qa.copy_(torch.randn_like(self.qa.cpu(), memory_format=torch.contiguous_format).normal_(generator=g) / math.sqrt(self.d))
            self.va.copy_(torch.randn_like(self.va.cpu(), memory_format=torch.contiguous_format).normal_(generator=g) / math.sqrt(self.d))
            self.qb.zero_()
            self.vb.zero_()


class LinLoRA(nn.Module):
    """Rank-r LoRA on a plain nn.Linear (used for q_proj / v_proj)."""

    def __init__(self, lin, rank, seed_offset):
        super().__init__()
        self.base = lin
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_in = lin.in_features
        g = torch.Generator().manual_seed(seed_offset)
        self.a = nn.Parameter(torch.randn(rank, d_in, generator=g) / math.sqrt(d_in))
        self.b = nn.Parameter(torch.zeros(lin.out_features, rank))
        self.on = True

    def forward(self, x):
        out = self.base(x)
        if self.on:
            out = out + (x @ self.a.T) @ self.b.T
        return out


def add_lora(model, seed):
    loras = []
    if hasattr(model, "transformer"):          # GPT-2 family
        for i, block in enumerate(model.transformer.h):
            wrap = QVLoRA(block.attn.c_attn, model.config.n_embd, LORA_RANK,
                          seed_offset=10_000 * seed + i).to(DEV)
            block.attn.c_attn = wrap
            loras.append(wrap)
    else:                                       # Llama/Qwen-style: q_proj/v_proj
        for i, layer in enumerate(model.model.layers):
            for j, name in enumerate(("q_proj", "v_proj")):
                wrap = LinLoRA(getattr(layer.self_attn, name), LORA_RANK,
                               seed_offset=10_000 * seed + 2 * i + j).to(DEV)
                setattr(layer.self_attn, name, wrap)
                loras.append(wrap)
    return loras


def lora_params(loras):
    out = []
    for l in loras:
        out += [l.qa, l.qb, l.va, l.vb] if isinstance(l, QVLoRA) else [l.a, l.b]
    return out


def set_lora(loras, on):
    for l in loras:
        l.on = on


def reinit_lora(loras, seed):
    for i, l in enumerate(loras):
        g = torch.Generator().manual_seed(10_000 * seed + i)
        with torch.no_grad():
            if isinstance(l, QVLoRA):
                l.qa.copy_((torch.randn(l.qa.shape, generator=g) / math.sqrt(l.d)).to(l.qa.device))
                l.va.copy_((torch.randn(l.va.shape, generator=g) / math.sqrt(l.d)).to(l.va.device))
                l.qb.zero_(); l.vb.zero_()
            else:
                d_in = l.a.shape[1]
                l.a.copy_((torch.randn(l.a.shape, generator=g) / math.sqrt(d_in)).to(l.a.device))
                l.b.zero_()


# ------------------------------------------------------------------ world
class World:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        pairs = [(f, l) for f in range(len(FIRST)) for l in range(len(LAST))]
        rng.shuffle(pairs)
        self.names = [f"{FIRST[f]} {LAST[l]}" for f, l in pairs[:N_ENTS + N_EXTRA]]
        self.occ = rng.integers(0, len(OCCS), N_ENTS + N_EXTRA)
        # cities: force at least 3 same-city pairs among consolidation ents
        city = rng.integers(0, len(CITIES), N_ENTS + N_EXTRA)
        for k in range(3):
            city[2 * k + 1] = city[2 * k]
        self.city = city
        self.rng = rng

    def sentence(self, e, reg):
        return TEMPLATES[reg].format(name=self.names[e],
                                     occ=OCCS[self.occ[e]],
                                     city=CITIES[self.city[e]])

    def fact_paragraph(self, ents):
        return " ".join(f"{self.names[e]} is a {OCCS[self.occ[e]]} who lives in "
                        f"{CITIES[self.city[e]]}." for e in ents)


# ------------------------------------------------------------------ tokenization helpers
def name_final_positions(tok, texts, names):
    """Token index of the last name token for each (text, name)."""
    enc = tok(texts, return_offsets_mapping=True, padding=True,
              return_tensors="pt")
    pos = []
    for i, (text, name) in enumerate(zip(texts, names)):
        start = text.index(name)
        end = start + len(name)
        idx = [j for j, (a, b) in enumerate(enc.offset_mapping[i].tolist())
               if a < end and b > start and b <= end + 1 and (a, b) != (0, 0)]
        if not idx:   # fallback: any overlap
            idx = [j for j, (a, b) in enumerate(enc.offset_mapping[i].tolist())
                   if a < end and b > start]
        pos.append(max(idx))
    return enc, torch.tensor(pos)


def option_logprob(model, tok, prompt, option):
    """Mean logprob of option tokens continuing the prompt."""
    full = prompt + option
    ids = tok(full, return_tensors="pt").input_ids.to(DEV)
    n_prompt = tok(prompt, return_tensors="pt").input_ids.shape[1]
    with torch.no_grad():
        logits = model(ids).logits
    lp = F.log_softmax(logits[0, :-1], dim=-1)
    tgt = ids[0, 1:]
    span = range(n_prompt - 1, ids.shape[1] - 1)
    return float(np.mean([lp[t, tgt[t]].item() for t in span]))


def pick(model, tok, prompt, options):
    return int(np.argmax([option_logprob(model, tok, prompt, o) for o in options]))


# ------------------------------------------------------------------ evals
def fewshot_prefix(world):
    """Format-teaching examples using EXTRA entities with facts in-prompt.
    Balanced: one Yes and one No same-city example; two two-choice examples
    with the correct answer in first and second position respectively."""
    a, b, c, d = N_ENTS, N_ENTS + 1, N_ENTS + 2, N_ENTS + 3
    # force a/b same city, c/d different (extra entities are ours to set)
    world.city[b] = world.city[a]
    if world.city[d] == world.city[c]:
        world.city[d] = (world.city[c] + 1) % len(CITIES)
    p = world.fact_paragraph([a, b, c, d]) + "\n"
    p += (f"Q: Do {world.names[a]} and {world.names[b]} live in the same city? A: Yes\n")
    p += (f"Q: Do {world.names[c]} and {world.names[d]} live in the same city? A: No\n")
    p += (f"Q: Which of them lives in {CITIES[world.city[c]]}: {world.names[c]} "
          f"or {world.names[d]}? A: {world.names[c]}\n")
    p += (f"Q: Which of them lives in {CITIES[world.city[b]]}: {world.names[d]} "
          f"or {world.names[b]}? A: {world.names[b]}\n")
    return p


def composition_eval(model, tok, world, condition, loras):
    """Returns (samecity_acc, twochoice_acc). condition: context|weights|both."""
    set_lora(loras, condition in ("weights", "both"))
    ents = list(range(N_ENTS))
    rng = np.random.default_rng(4242)
    same_pairs = [(i, j) for i in ents for j in ents if i < j
                  and world.city[i] == world.city[j]]
    diff_pairs = [(i, j) for i in ents for j in ents if i < j
                  and world.city[i] != world.city[j]]
    rng.shuffle(diff_pairs)
    diff_pairs = diff_pairs[:len(same_pairs)]
    fs = fewshot_prefix(world)
    correct_sc, yes_picks = [], []
    for (i, j), is_same in ([(p, True) for p in same_pairs] +
                            [(p, False) for p in diff_pairs]):
        ctx = (world.fact_paragraph([i, j]) + "\n") if condition in ("context", "both") else ""
        prompt = fs + ctx + (f"Q: Do {world.names[i]} and {world.names[j]} "
                             f"live in the same city? A:")
        ans = pick(model, tok, prompt, [" Yes", " No"])
        yes_picks.append(ans == 0)
        correct_sc.append((ans == 0) == is_same)
    # two-choice, counterbalanced: each pair asked in both option orders
    correct_tc = []
    for _ in range(8 if SMOKE else 16):
        i, j = rng.choice(ents, 2, replace=False)
        if world.city[i] == world.city[j]:
            continue
        ctx = (world.fact_paragraph([i, j]) + "\n") if condition in ("context", "both") else ""
        q = f"Q: Which of them lives in {CITIES[world.city[i]]}: "
        p1 = fs + ctx + q + f"{world.names[i]} or {world.names[j]}? A:"
        correct_tc.append(pick(model, tok, p1, [" " + world.names[i], " " + world.names[j]]) == 0)
        p2 = fs + ctx + q + f"{world.names[j]} or {world.names[i]}? A:"
        correct_tc.append(pick(model, tok, p2, [" " + world.names[j], " " + world.names[i]]) == 1)
    return (float(np.mean(correct_sc)), float(np.mean(correct_tc)),
            float(np.mean(yes_picks)))


def declarative_qa(model, tok, world, loras):
    set_lora(loras, True)
    accs_city, accs_occ = [], []
    for e in range(N_ENTS):
        p = f"Q: What city does {world.names[e]} live in? A:"
        accs_city.append(pick(model, tok, p, [" " + c for c in CITIES]) == world.city[e])
        p = f"Q: What is {world.names[e]}'s occupation? A:"
        accs_occ.append(pick(model, tok, p, [" " + o for o in OCCS]) == world.occ[e])
    return float(np.mean(accs_city)), float(np.mean(accs_occ))


@torch.no_grad()
def name_states(model, tok, world, regs, loras, lora_on=True):
    set_lora(loras, lora_on)
    texts, names, ents = [], [], []
    for e in range(N_ENTS):
        for r in regs:
            texts.append(world.sentence(e, r))
            names.append(world.names[e])
            ents.append(e)
    enc, pos = name_final_positions(tok, texts, names)
    out = model(input_ids=enc.input_ids.to(DEV),
                attention_mask=enc.attention_mask.to(DEV),
                output_hidden_states=True)
    h = out.hidden_states[-1]
    z = h[torch.arange(len(texts)), pos].cpu().numpy()
    return z, np.array(ents)


def ridge_probe(z_tr, k_tr, z_ho, k_ho, reg=10.0):
    ents = sorted(set(k_tr.tolist()))
    Y = np.stack([(k_tr == e).astype(float) for e in ents], axis=1)
    X = np.concatenate([z_tr, np.ones((len(z_tr), 1))], axis=1)
    W = np.linalg.solve(X.T @ X + reg * np.eye(X.shape[1]), X.T @ Y)
    Xh = np.concatenate([z_ho, np.ones((len(z_ho), 1))], axis=1)
    pred = np.array(ents)[np.argmax(Xh @ W, axis=1)]
    return float(np.mean(pred == k_ho))


def np_fisher(z, ks, eps=1e-6):
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + eps)
    uniq = sorted(set(ks.tolist()))
    mus = np.stack([zn[ks == k].mean(0) for k in uniq])
    within = np.mean([np.mean(np.sum((zn[ks == k] - mus[i]) ** 2, axis=1))
                      for i, k in enumerate(uniq)])
    between = np.mean(np.sum((mus - mus.mean(0)) ** 2, axis=1))
    return float(within / (between + eps))


def np_retrieval(z_tr, k_tr, z_ho, k_ho, eps=1e-6):
    zn = lambda x: x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)
    uniq = sorted(set(k_tr.tolist()))
    cents = np.stack([zn(z_tr)[k_tr == k].mean(0) for k in uniq])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + eps
    pred = np.array(uniq)[np.argmax(zn(z_ho) @ cents.T, axis=1)]
    return float(np.mean(pred == k_ho))


@torch.no_grad()
def perplexity(model, tok, loras, lora_on):
    set_lora(loras, lora_on)
    ids = tok(NATURAL_TEXT, return_tensors="pt").input_ids.to(DEV)
    loss = model(ids, labels=ids).loss
    return float(torch.exp(loss).item())


# ------------------------------------------------------------------ training
def fisher_coherence(h, ents, eps=1e-6):
    zn = F.normalize(h, dim=-1)
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
    return within / (between + eps)


def consolidate(model, tok, world, loras, seed, lam, shuffle):
    reinit_lora(loras, seed)
    set_lora(loras, True)
    rng = np.random.default_rng(777 + seed)
    shuffle_rng = np.random.default_rng(31337 + seed)
    opt = torch.optim.AdamW(lora_params(loras), lr=LR)
    for step in range(STEPS):
        ents = rng.choice(N_ENTS, BATCH_ENTS, replace=False)
        texts, names, labels = [], [], []
        for e in ents:
            regs = rng.choice(DREAM_REGISTERS, BATCH_REGS, replace=False)
            for r in regs:
                texts.append(world.sentence(int(e), int(r)))
                names.append(world.names[int(e)])
                labels.append(int(e))
        enc, pos = name_final_positions(tok, texts, names)
        input_ids = enc.input_ids.to(DEV)
        attn = enc.attention_mask.to(DEV)
        labels_lm = input_ids.masked_fill(attn == 0, -100)
        out = model(input_ids=input_ids, attention_mask=attn,
                    labels=labels_lm, output_hidden_states=True)
        L_rec = out.loss
        ent_arr = np.array(labels)
        if shuffle:
            ent_arr = shuffle_rng.permutation(ent_arr)
        h = out.hidden_states[-1][torch.arange(len(texts)), pos]
        L_coh = fisher_coherence(h, ent_arr)
        loss = (1 - lam) * L_rec + lam * L_coh
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params(loras), 1.0)
        opt.step()
        with torch.no_grad():
            for p in lora_params(loras):
                p.data.mul_(1.0 - DECAY)
        if (step + 1) % 100 == 0:
            print(f"    step {step+1}: lm={L_rec.item():.3f} coh={L_coh.item():.3f}",
                  flush=True)


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gate_only = os.environ.get("EXPE_GATE_ONLY", "") == "1"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    loras = add_lora(model, seed=0)

    results = []
    for seed in SEEDS:
        t0 = time.time()
        world = World(seed)
        # ---- gate + baselines (adapter off / zero, so condition==context is pure base)
        set_lora(loras, False)
        ppl_base = perplexity(model, tok, loras, lora_on=False)
        gate_sc, gate_tc, gate_yr = composition_eval(model, tok, world, "context", loras)
        # zero-shot weights-only (no consolidation yet, adapter zero)
        reinit_lora(loras, seed)
        zs_sc, zs_tc, _ = composition_eval(model, tok, world, "weights", loras)
        zs_city, zs_occ = declarative_qa(model, tok, world, loras)
        print(f"seed {seed}: GATE context comp: same-city={gate_sc:.3f} "
              f"(yes-rate {gate_yr:.2f}) two-choice={gate_tc:.3f} (need >=0.70) | zero-shot weights: "
              f"sc={zs_sc:.3f} tc={zs_tc:.3f} city={zs_city:.3f} occ={zs_occ:.3f} "
              f"| ppl={ppl_base:.1f} ({time.time()-t0:.0f}s)", flush=True)
        gate_ok = gate_sc >= 0.70 and gate_tc >= 0.70
        if gate_only:
            continue
        if not gate_ok:
            print(f"  GATE FAILED at {MODEL_NAME} — recording; escalate model size.",
                  flush=True)

        for arm, lam, shuf in ARMS:
            t1 = time.time()
            consolidate(model, tok, world, loras, seed, lam, shuf)
            res = {"seed": seed, "arm": arm, "lam": lam, "shuffled": shuf,
                   "model": MODEL_NAME, "gate_sc": gate_sc, "gate_tc": gate_tc,
                   "gate_ok": gate_ok, "zs_sc": zs_sc, "zs_tc": zs_tc,
                   "zs_city": zs_city, "zs_occ": zs_occ, "ppl_base": ppl_base}
            res["qa_city"], res["qa_occ"] = declarative_qa(model, tok, world, loras)
            for cond in ["context", "weights", "both"]:
                sc, tc, yr = composition_eval(model, tok, world, cond, loras)
                res[f"comp_sc_{cond}"], res[f"comp_tc_{cond}"] = sc, tc
                res[f"comp_yesrate_{cond}"] = yr
            z_tr, k_tr = name_states(model, tok, world, DREAM_REGISTERS, loras)
            z_ho, k_ho = name_states(model, tok, world, HELD_REGISTERS, loras)
            res["probe"] = ridge_probe(z_tr, k_tr, z_ho, k_ho)
            res["retrieval"] = np_retrieval(z_tr, k_tr, z_ho, k_ho)
            res["fisher"] = np_fisher(z_ho, k_ho)
            res["ppl_post"] = perplexity(model, tok, loras, lora_on=True)
            results.append(res)
            print(f"  seed={seed} {arm:>8}: qa_city={res['qa_city']:.3f} "
                  f"qa_occ={res['qa_occ']:.3f} | comp_sc w/c/b="
                  f"{res['comp_sc_weights']:.2f}/{res['comp_sc_context']:.2f}/{res['comp_sc_both']:.2f} "
                  f"comp_tc w/c/b={res['comp_tc_weights']:.2f}/{res['comp_tc_context']:.2f}/{res['comp_tc_both']:.2f} "
                  f"| probe={res['probe']:.3f} retr={res['retrieval']:.3f} "
                  f"fish={res['fisher']:.2f} ppl={res['ppl_post']:.1f} "
                  f"({time.time()-t1:.0f}s)", flush=True)
            with open(os.path.join(OUT_DIR, RESULTS_NAME), "w") as f:
                json.dump(results, f, indent=1)
    print("done.")


if __name__ == "__main__":
    main()
