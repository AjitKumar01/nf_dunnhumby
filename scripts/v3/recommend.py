"""Complete-the-basket: hold out one item, rank the assortment, see where it lands.

This is the use case where our model is at its strongest and has never been measured.  Ranking
needs NO normaliser: adding product j to a partial basket S changes the energy by

    delta(j) = b_ijt + sum_{k in S} phi_j.phi_k - rho_c * n_c(S) - [rho_0(n+1) - rho_0(n)]

and log Z is identical for every candidate, so it cancels in the comparison.  Exact, one pass
over the assortment, no sampling -- unlike every likelihood number in this project.

The comparison that matters is OURS WITH phi against OURS WITHOUT.  Everything else is held
fixed, so the difference is exactly what the interaction contributes to knowing which product
completes a basket.  If phi adds nothing here it adds nothing anywhere, whatever the
co-occurrence metric says.

    popularity     global purchase frequency; the floor
    ours, b only   the full contextual item value, interaction ablated
    ours, full     + sum_{k in S} phi_j.phi_k
    ndpp           P(j | S) by the Schur complement, L_jj - L_jS L_S^-1 L_Sj -- exact and
                   cheap, since inverting the n x n L_S costs nothing at n ~ 8
    multinom       its fitted item weights (no interaction, so it cannot use the basket)

Shopper is omitted: its conditional depends on the running mean of alpha over items ALREADY
chosen, so a score depends on the assumed order of the partial basket, and there is no
order-free way to ask it this question.

Metrics are recall@k and MRR over held-out items, with the rank taken among the trip's own
assortment (~5,300 products), not the full catalogue.

Run:  V3_AFFINITY=1 python3 recommend.py
"""
import argparse
import os

import numpy as np
import torch
from torch.nn.functional import softplus

from data import build
from features import Features
from fit import Batcher
from ragged import RaggedModel


def log(m):
    print(f"[rec] {m}", flush=True)


def metrics(ranks, ks=(5, 10, 20, 50, 100)):
    r = np.asarray(ranks, dtype=float)
    out = {f"recall@{k}": float((r <= k).mean()) for k in ks}
    out["MRR"] = float((1.0 / r).mean())
    out["median rank"] = float(np.median(r))
    return out


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    F = Features(J, S, 712)
    Bt = Batcher(D, F, a.nmax)
    m = RaggedModel(J=J, N=N, C=C, K=32, Kz=a.Kz, nmax=a.nmax, R=a.R, S=S, Kp=8)
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = blob["model"] if isinstance(blob, dict) and blob.get("format") == 2 else blob
    miss, _ = m.load_state_dict(sd, strict=False)
    assert not [k for k in miss if k != "cat_of"], f"missing {miss}"
    with torch.no_grad():
        co = torch.zeros(J, dtype=torch.long)
        co[torch.as_tensor(D["line_item"], dtype=torch.long)] = \
            torch.as_tensor(D["line_cat"], dtype=torch.long)
        m.cat_of.copy_(co)
    # Shrink the PRODUCT-VARYING residual of the contextual embeddings, keeping their mean.
    #
    # A shift common to every product cannot change a RANKING -- it is constant within a
    # trip -- so the mean direction (75% of mu's energy, 64% of zeta's) is not what costs us
    # 34x in MRR.  The damage is the residual, which is meant to say WHICH products move with
    # season/store/recency and evidently says it badly.  Centring removes the mean and keeps
    # the residual, which is exactly backwards: it destroyed E[n] (1.00) while leaving the
    # harmful part in place.
    if a.ctx_shrink < 1.0:
        with torch.no_grad():
            for nm in ("mu", "zeta", "psi"):
                P = getattr(m, nm)
                mu_ = P.mean(0, keepdim=True)
                P.copy_(mu_ + a.ctx_shrink * (P - mu_))
    m.double().eval()
    log(f"{os.path.basename(a.ckpt)} (iter {blob.get('iter','?')}), "
        f"ctx residual x {a.ctx_shrink}")

    # TRAINING trips only.  Counting all line_item leaks the test baskets into the
    # baseline, which is the comparison's own thumb on the scale.
    _tr = D["trip_split"] == 0
    _lp = D["line_ptr"]
    _keep = np.zeros(len(D["line_item"]), bool)
    for t in np.flatnonzero(_tr):
        _keep[int(_lp[t]):int(_lp[t + 1])] = True
    pop = np.bincount(D["line_item"][_keep], minlength=J).astype(float)
    pop = torch.as_tensor(pop / max(pop.sum(), 1.0))

    # Item-item co-purchase, the baseline a real recommender would actually use: score a
    # candidate by how often it appears with the items already in the basket.  Omitting it
    # would have compared us only against a popularity prior, which is not the bar.
    from collections import defaultdict
    co = defaultdict(float)
    for t in np.flatnonzero(_tr):
        it = np.unique(D["line_item"][int(_lp[t]):int(_lp[t + 1])])
        if len(it) < 2 or len(it) > 40:
            continue
        for x in range(len(it)):
            for y in range(len(it)):
                if x != y:
                    co[(int(it[x]), int(it[y]))] += 1.0
    log(f"co-purchase table: {len(co):,} ordered pairs from training trips")
    cnt_j = np.bincount(D["line_item"][_keep], minlength=J).astype(float) + 1.0

    rng = np.random.default_rng(0)
    idx = np.flatnonzero(D["trip_split"] == 2)
    ptr = D["line_ptr"]
    ok = [t for t in idx if 2 <= int(ptr[t + 1]) - int(ptr[t]) <= a.nmax]
    trips = np.sort(np.array(ok)[rng.choice(len(ok), size=min(a.n_trips, len(ok)),
                                            replace=False)])
    log(f"{len(trips)} test trips with >= 2 lines; one item held out at random from each")

    R = {k: [] for k in ("popularity", "co-purchase", "lam only", "lam + taste",
                         "ours b only", "ours full", "multinom")}
    r0 = None
    for k0 in range(0, len(trips), a.chunk):
        sub = trips[k0:k0 + a.chunk]
        ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(sub)
        m.house, m.ctx = hh, ctx
        with torch.no_grad():
            bflat = m.b_flat(ix)                      # value of every assortment slot
            if r0 is None:
                r0 = m.rho_0()
        for b in range(ix.B):
            sel = (ix.item_trip == b)
            items = ix.item[sel]
            bv = bflat[sel]
            basket = LI[LT == b]
            if len(basket) < 2:
                continue
            hidden = int(basket[rng.integers(len(basket))])
            rest = torch.as_tensor([int(x) for x in basket if int(x) != hidden],
                                   dtype=torch.long)
            if len(rest) == 0:
                continue
            pos = (items == hidden).nonzero().flatten()
            if len(pos) == 0:
                continue                              # held-out item outside the assortment
            p = int(pos[0])

            # --- interaction: sum_{k in rest} phi_j . phi_k, for every candidate at once ---
            Ssum = m.phi[rest].sum(0)
            pair = m.phi[items] @ Ssum
            # a candidate already in the basket must not be credited for pairing with itself
            inb = torch.zeros(len(items), dtype=torch.bool)
            inb[torch.isin(items, rest)] = True
            pair = pair - inb.double() * (m.phi[items] * m.phi[items]).sum(-1)
            # --- category and size terms, both constant across j within a category/size ---
            n_now = len(rest)
            dr0 = r0[min(n_now + 1, m.nmax)] - r0[min(n_now, m.nmax)]

            # decompose b to find where the ranking is lost: lam_j is the item's own
            # intercept and should behave like popularity; everything else is contextual
            # (household taste, price, promotion, season, store, recency).
            # co-purchase score: sum over basket members of count(j, k), normalised by j's
            # own frequency so it is a lift rather than a popularity echo
            cs = torch.zeros(len(items), dtype=torch.float64)
            _rest = [int(x) for x in rest]
            for _i, _j in enumerate(items.tolist()):
                v = 0.0
                for _k in _rest:
                    v += co.get((_j, _k), 0.0)
                cs[_i] = v / cnt_j[_j]
            cand = {"popularity": pop[items].clamp_min(1e-12).log(),
                    "co-purchase": cs,
                    "lam only": m.lam[items],
                    # theta_c(), not theta: b_at centres theta over households to fix the
                    # gauge, so scoring with the raw tensor would not be the model's own b.
                    "lam + taste": m.lam[items]
                    + (m.theta_c()[hh[b]] * m.alpha[items]).sum(-1),
                    "ours b only": bv,
                    "ours full": bv + pair - dr0,
                    "multinom": bv}
            for name, sc in cand.items():
                sc = sc.clone()
                sc[inb] = -float("inf")               # cannot recommend what is already in
                R[name].append(int((sc > sc[p]).sum()) + 1)

    log("")
    hdr = f"{'model':>14}" + "".join(f"{f'R@{k}':>9}" for k in (5, 10, 20, 50, 100)) \
        + f"{'MRR':>8}{'median':>8}"
    log(hdr)
    for name in ("popularity", "co-purchase", "lam only", "lam + taste",
                 "ours b only", "ours full"):
        if not R[name]:
            continue
        mm = metrics(R[name])
        log(f"{name:>14}" + "".join(f"{100*mm[f'recall@{k}']:8.1f}%"
                                    for k in (5, 10, 20, 50, 100))
            + f"{mm['MRR']:8.4f}{mm['median rank']:8.0f}")
    if R["ours b only"] and R["ours full"]:
        a_, b_ = np.array(R["ours b only"]), np.array(R["ours full"])
        log("")
        log(f"  interaction changes the rank on {100*float((a_ != b_).mean()):.1f}% of holdouts; "
            f"better on {100*float((b_ < a_).mean()):.1f}%, worse on {100*float((b_ > a_).mean()):.1f}%")
        log(f"  mean rank {a_.mean():.1f} -> {b_.mean():.1f}   "
            f"median {np.median(a_):.0f} -> {np.median(b_):.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="../../out/v3_run68_best.pt")
    p.add_argument("--n-trips", type=int, default=768)
    p.add_argument("--chunk", type=int, default=24)
    p.add_argument("--Kz", type=int, default=32)
    p.add_argument("--nmax", type=int, default=120)
    p.add_argument("--R", type=int, default=23)
    p.add_argument("--ctx-shrink", type=float, default=1.0)
    main(p.parse_args())
