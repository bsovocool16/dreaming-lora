"""Aggregate Track 2 results and check them against pre-registered predictions.

Produces markdown tables (mean +/- sd over seeds) for:
  A: retrieval_acc / fisher_holdout / recon_holdout by (omega, n_ctx, lam)
  B: attr_eval_acc / retrieval_eval / fisher_eval / base_attr_acc by (n_dream, lam)
and prints the P-A* / P-B* verdicts with the numbers behind them.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    with open(path) as f:
        return json.load(f)


def agg(rows, keys, metrics):
    """Group rows by keys, return {key_tuple: {metric: (mean, sd)}}."""
    out = {}
    for r in rows:
        k = tuple(r[k_] for k_ in keys)
        out.setdefault(k, []).append(r)
    return {k: {m: (float(np.mean([r[m] for r in v])),
                    float(np.std([r[m] for r in v]))) for m in metrics}
            for k, v in sorted(out.items())}


def table_a(rows):
    print("\n## Experiment A (linear): retrieval accuracy on held-out contexts\n")
    metrics = ["retrieval_acc", "retrieval_acc_raw_h", "fisher_holdout",
               "recon_holdout"]
    for omega in sorted(set(r["omega"] for r in rows)):
        sub = [r for r in rows if r["omega"] == omega]
        g = agg(sub, ["n_ctx_train", "lam"], metrics)
        lams = sorted(set(r["lam"] for r in sub))
        print(f"\n### omega = {omega} (content-dependence of task)\n")
        hdr = "| n_ctx | raw h | " + " | ".join(f"lam={l}" for l in lams) + " |"
        print(hdr)
        print("|" + "---|" * (len(lams) + 2))
        for n_ctx in sorted(set(r["n_ctx_train"] for r in sub)):
            raw = g[(n_ctx, lams[0])]["retrieval_acc_raw_h"][0]
            cells = " | ".join(
                f"{g[(n_ctx,l)]['retrieval_acc'][0]:.3f}±{g[(n_ctx,l)]['retrieval_acc'][1]:.03f}"
                for l in lams)
            print(f"| {n_ctx} | {raw:.3f} | {cells} |")
        # recon control row (lam cost)
        print(f"\nrecon_holdout at n_ctx=8: " + ", ".join(
            f"lam={l}: {g[(8,l)]['recon_holdout'][0]:.3f}" for l in lams))


def verdicts_a(rows):
    print("\n## Pre-registered verdicts — Experiment A\n")
    g = agg(rows, ["omega", "n_ctx_train", "lam"], ["retrieval_acc", "recon_holdout"])
    lams = sorted(set(r["lam"] for r in rows))
    best_lam_pos = [l for l in lams if l > 0]

    def best_acc(omega, n):
        return max((g[(omega, n, l)]["retrieval_acc"][0], l) for l in best_lam_pos)

    for omega in sorted(set(r["omega"] for r in rows)):
        r0_low = g[(omega, 2, 0.0)]["retrieval_acc"][0]
        bl, _ = best_acc(omega, 2)
        r0_hi = g[(omega, 32, 0.0)]["retrieval_acc"][0]
        bh, lh = best_acc(omega, 32)
        print(f"omega={omega}: n_ctx=2 recon-only {r0_low:.3f} vs best-composite {bl:.3f} "
              f"(gap {bl-r0_low:+.3f}); n_ctx=32 recon-only {r0_hi:.3f} vs "
              f"best-composite {bh:.3f} at lam={lh} (gap {bh-r0_hi:+.3f}) "
              f"-> P-A2 {'OPTION-B (gap<0.05)' if bh-r0_hi < 0.05 else 'coherence load-bearing'}")


def table_b(rows):
    print("\n## Experiment B (transformer): cross-context attribute transfer\n")
    metrics = ["attr_eval_acc", "attr_dream_acc", "retrieval_eval",
               "fisher_eval", "base_attr_acc", "lm_dream"]
    g = agg(rows, ["n_dream", "lam"], metrics)
    lams = sorted(set(r["lam"] for r in rows))
    for metric, label in [("attr_eval_acc", "attribute accuracy, EVAL families (never dreamed)"),
                          ("attr_dream_acc", "attribute accuracy, dreamed families"),
                          ("retrieval_eval", "entity retrieval, eval families"),
                          ("fisher_eval", "Fisher ratio, eval families (lower=better)"),
                          ("base_attr_acc", "base-entity accuracy (forgetting control)")]:
        print(f"\n### {label}\n")
        print("| n_dream | " + " | ".join(f"lam={l}" for l in lams) + " |")
        print("|" + "---|" * (len(lams) + 1))
        for nd in sorted(set(r["n_dream"] for r in rows)):
            cells = " | ".join(f"{g[(nd,l)][metric][0]:.3f}±{g[(nd,l)][metric][1]:.3f}"
                               for l in lams)
            print(f"| {nd} | {cells} |")


def verdicts_b(rows):
    print("\n## Pre-registered verdicts — Experiment B\n")
    g = agg(rows, ["n_dream", "lam"], ["attr_eval_acc", "base_attr_acc", "fisher_eval"])
    lams = sorted(set(r["lam"] for r in rows))
    pos = [l for l in lams if l > 0]
    for nd in sorted(set(r["n_dream"] for r in rows)):
        r0 = g[(nd, 0.0)]["attr_eval_acc"][0]
        best, lbest = max((g[(nd, l)]["attr_eval_acc"][0], l) for l in pos)
        forget = g[(nd, lbest)]["base_attr_acc"][0] - g[(nd, 0.0)]["base_attr_acc"][0]
        print(f"n_dream={nd}: recon-only {r0:.3f}, best composite {best:.3f} "
              f"(lam={lbest}, gap {best-r0:+.3f}, base-acc delta {forget:+.3f})")
    # P-B3: fisher vs eval acc correlation across all cells
    xs = [r["fisher_eval"] for r in rows]
    ys = [r["attr_eval_acc"] for r in rows]
    print(f"\nP-B3 fisher_eval vs attr_eval_acc correlation: "
          f"r = {np.corrcoef(xs, ys)[0,1]:.3f} (prediction: strongly negative)")


def main():
    pa = os.path.join(HERE, "results_a", "results_a.json")
    if os.path.exists(pa):
        rows = load(pa)
        table_a(rows)
        verdicts_a(rows)
    else:
        print("no results_a yet")
    for fname, label in [("results_b.json", "k_base=8"),
                         ("results_b_k2.json", "k_base=2"),
                         ("results_b_k2_power.json",
                          "k_base=2 POWERED (15 seeds, fixed lam, arms recon/cohere/shuffled)")]:
        pb = os.path.join(HERE, "results_b", fname)
        if not os.path.exists(pb):
            print(f"no {fname} yet")
            continue
        rows = load(pb)
        for r in rows:
            r.setdefault("k_base", 8)   # first-arm rows predate the field
            r.setdefault("arm", "grid")
        # powered file has arms at equal lam; keep only non-shuffled rows for
        # the lam-grid tables (the shuffled arm is analyzed separately)
        grid_rows = [r for r in rows if not r.get("shuffled")]
        print(f"\n\n# ===== Experiment B arm: {label} =====")
        table_b(grid_rows)
        verdicts_b(grid_rows)


if __name__ == "__main__":
    main()
