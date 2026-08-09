"""
Stage 46 -- Does the residual novelty excess compound over a horizon?

Generated baskets contain about 20% more never-before-bought products than real ones
(8.38% novel categories against 6.58%, 49.26% novel products against 41.48%).  Per basket
that is small.  The question for any multi-step use is whether it ACCUMULATES: a household
that keeps buying new things drifts, trip by trip, into a product set the real household
would never reach.

The measurement is the cumulative distinct-product set.  Take a household's held-out
trips in order; after T trips, how many distinct products has it bought that it never
bought in training?  Compare against the same household's generated trips.

  real      D_real(T)   new distinct products after T trips
  generated D_gen(T)    same, from generated baskets for the same trips

If the ratio is flat in T the excess is a level effect and does not compound.  If it grows,
it does, and the horizon over which rollouts stay usable is bounded.

ONE LIMITATION, stated because it makes this a LOWER bound.  Generation reads the recency
state from the REAL purchase history, so a generated novel product does not make the next
generated basket more novel.  A true rollout would feed generated purchases back into the
state and compound further.  What is measured here is the compounding of the per-basket
excess alone, with no feedback.

Writes out/horizon.json.
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch

nb = importlib.import_module("27_nested_basket")
cf = importlib.import_module("28_nested_counterfactual")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "basket_input")
OUT = os.path.join(HERE, "..", "out")


def log(m):
    print(f"[46] {m}", flush=True)


def main(a):
    dev = torch.device("cpu")
    d = nb.NestedData(IN, device=dev)
    sp, tr = d.splits["test"], d.splits["train"]
    m, _ = cf.load(a.label, d, dev)

    seen = np.zeros((d.N, d.J), dtype=bool)
    seen[tr["user"], tr["item"]] = True

    # held-out trips per household, in chronological order
    by_hh = {}
    for i in range(sp["n_baskets"]):
        u = int(sp["user"][sp["starts"][i]])
        by_hh.setdefault(u, []).append((int(sp["day"][sp["starts"][i]]), i))
    hh = [u for u, v in by_hh.items() if len(v) >= a.min_trips]
    for u in hh:
        by_hh[u].sort()
    log(f"{len(hh):,} households with >= {a.min_trips} held-out trips")

    rng = np.random.default_rng(a.seed)
    hh = list(rng.permutation(hh))[:a.n_households]
    trips = np.array([i for u in hh for _, i in by_hh[u][:a.horizon]])
    log(f"{len(trips):,} trips across {len(hh):,} households, horizon {a.horizon}")

    g = cf.generate_baskets(m, d, dev, n_trips=len(trips), seed=a.seed, sweeps=4,
                            use_ctx=True, with_units=False, trips=trips)
    gen = {int(t): np.asarray(b) for t, b in zip(g.trips, g)}

    R = np.zeros((len(hh), a.horizon))
    G = np.zeros((len(hh), a.horizon))
    for k, u in enumerate(hh):
        accR, accG = set(), set()
        for T, (_, i) in enumerate(by_hh[u][:a.horizon]):
            for j in sp["item"][sp["starts"][i]:sp["ends"][i]]:
                if not seen[u, j]:
                    accR.add(int(j))
            for j in gen.get(i, []):
                if not seen[u, int(j)]:
                    accG.add(int(j))
            R[k, T], G[k, T] = len(accR), len(accG)

    log("")
    log("  cumulative NEW distinct products (never bought in training), per household")
    log(f"  {'after T trips':>14s} {'real':>9s} {'generated':>11s} {'ratio':>8s}")
    tab = []
    for T in a.report:
        if T > a.horizon:
            continue
        r, q = R[:, T - 1].mean(), G[:, T - 1].mean()
        tab.append({"T": T, "real": float(r), "gen": float(q),
                    "ratio": float(q / max(r, 1e-9))})
        log(f"  {T:14d} {r:9.2f} {q:11.2f} {q/max(r,1e-9):8.3f}x")

    ratios = [t["ratio"] for t in tab]
    log("")
    if max(ratios) - min(ratios) < 0.05:
        log(f"  The ratio is flat in T ({min(ratios):.3f}-{max(ratios):.3f}): the excess is")
        log("  a LEVEL effect and does not compound over this horizon.")
    else:
        log(f"  The ratio moves {ratios[0]:.3f} -> {ratios[-1]:.3f} across the horizon,")
        log("  so the per-basket excess does accumulate.")
    log("")
    log("  LOWER BOUND: generation reads recency from the REAL history, so a generated")
    log("  novel product does not make the next basket more novel.  A true rollout would")
    log("  feed generated purchases back and compound further than this.")

    res = {"label": a.label, "n_households": len(hh), "horizon": a.horizon,
           "cumulative_new": tab,
           "ratio_flat": bool(max(ratios) - min(ratios) < 0.05)}
    with open(os.path.join(OUT, "horizon.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log("wrote out/horizon.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="ps_nested")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--min-trips", type=int, default=12)
    p.add_argument("--n-households", type=int, default=600)
    p.add_argument("--report", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10, 12])
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
