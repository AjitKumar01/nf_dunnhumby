"""Complete-the-basket in the simulated environment, where the true conditional is known.

On dunnhumby our model ranks WORSE than popularity at completing a basket (MRR 0.0036 against
0.0467), and zeroing the product-varying part of the season/store/recency embeddings recovers
12x of that.  But real data cannot say what the CEILING is: the best achievable ranking is
unknown, so a bad score cannot be separated from a hard task.

Here the truth is the s25 mission mixture -- outside every model's family -- and every subset
is enumerated, so the exact conditional P(j | S) is available and can be ranked itself.  That
gives the ceiling, and every model's distance from it.

    P(j | S) proportional to P(S union {j}),   so ranking needs no normaliser at all --
    the same property that makes this the one task our model can serve exactly on real data.

    truth        the generating mixture; the achievable ceiling
    energy       ours
    dpp          det(L_S): can rank, but its conditional is repulsive by construction
    bernoulli    independent items: cannot use the basket at all
    multinom     size law x independent draws: also cannot use the basket
    popularity   marginal frequency, the floor
    co-purchase  item-item counts from the training baskets, what a real recommender uses

bernoulli, multinom and popularity are basket-blind by construction, so the gap between them
and the models that CAN condition is what the interaction is worth.

Run:  python3 s26_recsim.py
"""
import json
import math
import os

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

import s25_fair as A                              # truth, models, fitting, enumeration

J = A.J
POW = (2 ** torch.arange(J, dtype=torch.int64))


def log(m):
    print(f"[rec] {m}", flush=True)


def sub_index(items):
    """Index of a subset within the non-empty enumeration (empty set is dropped, so -1)."""
    return int(sum(1 << int(j) for j in items)) - 1


def metrics(ranks, ks=(1, 2, 3, 5)):
    r = np.asarray(ranks, dtype=float)
    out = {f"R@{k}": float((r <= k).mean()) for k in ks}
    out["MRR"] = float((1.0 / r).mean())
    out["median"] = float(np.median(r))
    return out


def main():
    T = A.T
    tr = A.sample_counts(0)
    log("fitting entrants on the mission-mixture truth")
    models = {}
    for tag, ctor, pool in (("energy", lambda: A.Energy(8, cap=2.5), 2.0),
                            ("dpp", A.DPPM, 0.0),
                            ("bernoulli", A.BernoulliM, 0.0),
                            ("multinom", A.Multinom, 0.0)):
        models[tag] = A.fit(ctor(), tr, tag, pool=pool)

    # ---- held-out baskets, and the two non-model baselines from the SAME training draw ----
    rng = np.random.default_rng(7)
    te = A.sample_counts(1)
    pop = np.zeros(J)
    co = np.zeros((J, J))
    for t, c in tr.items():
        nz = torch.nonzero(c > 0).flatten()
        for i in nz.tolist():
            w = float(c[i])
            mem = torch.nonzero(A.MASK_NE[i]).flatten().tolist()
            for x in mem:
                pop[x] += w
                for y in mem:
                    if x != y:
                        co[x, y] += w
    pop = pop / pop.sum()

    cases = []
    for t, c in te.items():
        nz = torch.nonzero(c > 0).flatten().tolist()
        for i in rng.choice(nz, size=min(400, len(nz)), replace=False):
            mem = torch.nonzero(A.MASK_NE[int(i)]).flatten().tolist()
            if len(mem) < 2:
                continue
            hid = int(rng.choice(mem))
            rest = [x for x in mem if x != hid]
            cases.append((t, rest, hid))
    log(f"{len(cases):,} held-out cases, one item removed from each basket\n")

    def rank_by(score_fn):
        out = []
        for t, rest, hid in cases:
            cand = [j for j in range(J) if j not in rest]
            sc = score_fn(t, rest, cand)
            s_hid = sc[cand.index(hid)]
            out.append(int(sum(1 for v in sc if v > s_hid)) + 1)
        return out

    def model_score(lp_cache):
        def f(t, rest, cand):
            lp = lp_cache[t]
            return [float(lp[sub_index(rest + [j])]) for j in cand]
        return f

    res = {}
    with torch.no_grad():
        caches = {"truth": {t: A.truth_logp(t) for t in te}}
        for k, m in models.items():
            caches[k] = {t: m.logp(T["dlp"][t]) for t in te}

    log(f"{'model':>14}{'R@1':>8}{'R@2':>8}{'R@3':>8}{'R@5':>8}{'MRR':>9}{'median':>8}")
    for name in ("truth", "energy", "dpp", "bernoulli", "multinom"):
        r = rank_by(model_score(caches[name]))
        mm = metrics(r)
        res[name] = mm
        log(f"{name:>14}" + "".join(f"{100*mm[f'R@{k}']:7.1f}%" for k in (1, 2, 3, 5))
            + f"{mm['MRR']:9.4f}{mm['median']:8.1f}")
    for name, fn in (("popularity", lambda t, rest, cand: [pop[j] for j in cand]),
                     ("co-purchase", lambda t, rest, cand:
                      [sum(co[j, k] for k in rest) / max(pop[j], 1e-12) for j in cand])):
        r = rank_by(fn)
        mm = metrics(r)
        res[name] = mm
        log(f"{name:>14}" + "".join(f"{100*mm[f'R@{k}']:7.1f}%" for k in (1, 2, 3, 5))
            + f"{mm['MRR']:9.4f}{mm['median']:8.1f}")

    json.dump(res, open(os.path.join(A.OUT, "v3_recsim.json"), "w"), indent=2)
    log("")
    log("truth is the ceiling: the exact conditional of the generating process.  bernoulli,")
    log("multinom and popularity cannot use the basket at all, so the gap to them is what")
    log("conditioning on what is already in the cart is worth.")


if __name__ == "__main__":
    main()
