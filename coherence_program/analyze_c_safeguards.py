"""Analysis for Experiment C (safeguards vs forgetting).

Reads results_b/results_c_safeguards.json and prints, per lam:
  - per-arm means with per-seed spreads for forgetting (base_attr_acc) and
    acquisition (attr_dream_acc, attr_eval_acc) plus the representational
    metrics;
  - paired deltas vs naive with a paired t-test (5 seeds -> weak power;
    t-values are descriptive, not gatekeepers);
  - the forgetting-reduction / acquisition-cost ratio per arm;
  - the cohere-minus-recon (lam 0.3 - 0.0) contrast per arm (P-C7).

Arms are readouts, not runs: shadow/combo shadow readouts are separate arms
from their live counterparts. shadow-live == naive by construction and is
dropped from the tables.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = json.load(open(os.path.join(HERE, "results_b",
                                   "results_c_safeguards.json")))

METRICS = ["base_attr_acc", "attr_dream_acc", "attr_eval_acc",
           "retrieval_eval", "probe_eval", "lm_dream", "fisher_eval"]


def arm_rows(rows):
    """Yield (arm_name, seed, lam, metric_dict) readouts."""
    for r in rows:
        v = r["variant"]
        if v in ("naive", "trust", "trust_cal"):
            yield v, r["seed"], r["lam"], r
        elif v == "decay":
            yield f"decay_eta{r['eta']:g}", r["seed"], r["lam"], r
        elif v in ("shadow", "combo", "combo_cal"):
            sh = {m: r[f"{m}_shadow"] for m in METRICS}
            yield f"{v}_shadow_b{r['beta']:g}", r["seed"], r["lam"], sh
            if v != "shadow" and r["beta"] == 0.05:  # live once per run
                yield f"{v}_live", r["seed"], r["lam"], r


ARMS = ["naive", "trust", "trust_cal", "decay_eta0.0001", "decay_eta0.001",
        "shadow_shadow_b0.05", "shadow_shadow_b0.005", "combo_live",
        "combo_shadow_b0.05", "combo_shadow_b0.005", "combo_cal_live",
        "combo_cal_shadow_b0.05", "combo_cal_shadow_b0.005"]

data = {}   # (arm, lam, metric) -> {seed: value}
for arm, seed, lam, d in arm_rows(ROWS):
    for m in METRICS:
        data.setdefault((arm, lam, m), {})[seed] = d[m]

seeds = sorted({r["seed"] for r in ROWS})


def vec(arm, lam, m):
    dd = data[(arm, lam, m)]
    return np.array([dd[s] for s in seeds])


def paired_t(d):
    d = np.asarray(d, dtype=float)
    if np.allclose(d.std(ddof=1), 0):
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


for lam in (0.0, 0.3):
    print(f"\n================ lam = {lam} ================")
    print(f"{'arm':<22}" + "".join(f"{m:>16}" for m in METRICS[:6]))
    for arm in ARMS:
        if (arm, lam, "base_attr_acc") not in data:
            continue
        cells = []
        for m in METRICS[:6]:
            v = vec(arm, lam, m)
            cells.append(f"{v.mean():.3f}[{v.min():.2f}-{v.max():.2f}]")
        print(f"{arm:<22}" + "".join(f"{c:>16}" for c in cells))

    print(f"\n  paired deltas vs naive (positive = better than naive on "
          f"forgetting; acquisition costs signed as naive - arm):")
    print(f"  {'arm':<22}{'d_forget':>20}{'cost_dream':>20}"
          f"{'cost_eval':>20}{'ratio_f/e':>10}")
    base_n = vec("naive", lam, "base_attr_acc")
    dream_n = vec("naive", lam, "attr_dream_acc")
    eval_n = vec("naive", lam, "attr_eval_acc")
    for arm in ARMS[1:]:
        if (arm, lam, "base_attr_acc") not in data:
            continue
        df = vec(arm, lam, "base_attr_acc") - base_n
        cd = dream_n - vec(arm, lam, "attr_dream_acc")
        ce = eval_n - vec(arm, lam, "attr_eval_acc")
        ratio = df.mean() / max(cd.mean() + ce.mean(), 1e-3)
        print(f"  {arm:<22}"
              f"{df.mean():+.3f} (t={paired_t(df):+.1f})".rjust(20)
              + f"{cd.mean():+.3f} (t={paired_t(cd):+.1f})".rjust(20)
              + f"{ce.mean():+.3f} (t={paired_t(ce):+.1f})".rjust(20)
              + f"{ratio:>10.1f}")

print("\n================ cohere - recon contrast per arm (P-C7) ========")
print(f"{'arm':<22}{'d_retr':>16}{'d_probe':>16}{'d_eval':>16}"
      f"{'d_base':>16}")
for arm in ARMS:
    if (arm, 0.3, "base_attr_acc") not in data:
        continue
    cells = []
    for m in ("retrieval_eval", "probe_eval", "attr_eval_acc",
              "base_attr_acc"):
        d = vec(arm, 0.3, m) - vec(arm, 0.0, m)
        cells.append(f"{d.mean():+.3f} (t={paired_t(d):+.1f})")
    print(f"{arm:<22}" + "".join(f"{c:>16}" for c in cells))

print("\nclip fractions (trust/combo):")
for r in ROWS:
    if "clip_frac" in r and r.get("beta") in (None, 0.05):
        print(f"  s{r['seed']} lam={r['lam']} {r['variant']}: "
              f"{r['clip_frac']:.2f}")
