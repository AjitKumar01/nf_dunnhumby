"""
Baselines that produce a properly normalised P(S) on the same support.

The comparison has to be like for like or it means nothing, so every model here obeys the
same three rules as version 3:

  * the support is the store's assortment from data.py -- the same set of products
  * the probability is over the SET, exactly normalised, not a per-item score or a top-k
  * it is conditioned on a non-empty basket, and the conditioning is exact

That rules out most of what is usually reported for basket data (top-k recall, next-item
accuracy, per-line likelihood against sampled negatives) and leaves models whose normaliser
is tractable in closed form.  Three are implemented.

FREQUENCY.  Independent purchase with pi_j the empirical rate at which product j is bought
at that store in training.  No fitted parameters.  This is the floor a model has to clear
to have done anything.

INDEPENDENT BERNOULLI.  pi_j = sigmoid(b_j), with the same broad covariate families as
version 3 -- intercept, household taste, price, display, mailer, seasonality, store.  This
is a separately parameterised baseline, not an exact nested ablation: the current main
model also constrains the price factors, separates common and relative price, and applies
identifiability/pooling transforms.  An exact nested comparison must use RaggedModel itself
with phi and rho_c zeroed and frozen (fit.py --zero-phi --zero-rho-c).

LOW-RANK DPP.  P(S) proportional to det(L_S) with L = V V' + diag(d).  The normaliser is
det(L + I), exact, and computable in O(J k^2) by the identity

    det(I + D + V V') = det(I + D) * det(I_k + V' (I + D)^{-1} V)

This is the strongest principled competitor and the interesting one: a DPP models
REPULSION natively and cannot represent complementarity, which is the exact complement of
version 3's Gaussian latent, whose interaction is positive semi-definite and so bounded in
how much repulsion it can express.  Whichever wins, the direction is informative.

The empty-set conditioning is the same identity in both families: det(L_empty) = 1, so
P(empty) = 1 / det(L + I) and the conditional normaliser is det(L + I) - 1, exactly as
version 3's is Z - 1.

Reported per basket AND per line, because the two families disagree about what a trip is
and quoting only one invites the wrong comparison.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from data import build
from features import Features
from ragged import RaggedIndex, esp_bucketed, poly_tree, seg_max

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[bas] {m}", flush=True)


class Batches:
    """Assortment slots and purchased lines per trip, with the same features version 3 uses."""

    def __init__(self, D, F):
        self.D, self.F = D, F
        self.C = int(D["n_cat"])
        self.ptr, self.items, self.lptr = (D["store_cat_ptr"], D["store_items"],
                                           D["line_ptr"])

    def make(self, trips):
        D, C = self.D, self.C
        it_l, slot_trip, row_of, row_trip, row_cat = [], [], [], [], []
        nrow = 0
        for bi, t in enumerate(trips):
            s = int(D["trip_store"][t]) * C
            for c in range(C):
                lo, hi = int(self.ptr[s + c]), int(self.ptr[s + c + 1])
                if hi <= lo:
                    continue
                it_l.append(self.items[lo:hi])
                slot_trip.append(np.full(hi - lo, bi, np.int64))
                row_of.append(np.full(hi - lo, nrow, np.int64))
                row_trip.append(bi)
                row_cat.append(c)
                nrow += 1
        item = torch.as_tensor(np.concatenate(it_l), dtype=torch.long)
        st = torch.as_tensor(np.concatenate(slot_trip), dtype=torch.long)
        # Keep the category rows as part of the batch contract.  The main model caps the
        # number of products selected from one category, so a baseline that discards this
        # structure is normalized on a different support even when it sees the same items.
        rix = RaggedIndex(item, np.concatenate(row_of), np.asarray(row_trip, np.int64),
                          np.asarray(row_cat, np.int64), len(trips))
        if not torch.equal(rix.item, item) or not torch.equal(rix.item_trip, st):
            raise RuntimeError("baseline assortment and ragged category index disagree")
        store = torch.as_tensor(D["trip_store"][trips], dtype=torch.long)
        day = torch.as_tensor(D["trip_day"][trips], dtype=torch.long)
        week = torch.as_tensor(D["trip_week"][trips], dtype=torch.long)
        house = torch.as_tensor(D["trip_user"][trips], dtype=torch.long)
        dlp, dsp, mlr = self.F.gather(item, store[st], day[st], week[st])
        ctx = dict(dlp=dlp.double(), disp=dsp.double(), mail=mlr.double(),
                   week=(week[st] - 1) % 52, store=store[st])
        li, lt = [], []
        for bi, t in enumerate(trips):
            a, b = int(self.lptr[t]), int(self.lptr[t + 1])
            li.append(D["line_item"][a:b])
            lt.append(np.full(b - a, bi, np.int64))
        LI = torch.as_tensor(np.concatenate(li), dtype=torch.long)
        LT = torch.as_tensor(np.concatenate(lt), dtype=torch.long)
        dl2, ds2, ml2 = self.F.gather(LI, store[LT], day[LT], week[LT])
        lctx = dict(dlp=dl2.double(), disp=ds2.double(), mail=ml2.double(),
                    week=(week[LT] - 1) % 52, store=store[LT])
        # position of each purchased line within its trip's slot block
        off = torch.cat([torch.zeros(1, dtype=torch.long),
                         torch.cumsum(torch.bincount(st, minlength=len(trips)), 0)])
        pos = {}
        for k in range(len(item)):
            pos[(int(st[k]), int(item[k]))] = k
        lslot = torch.as_tensor([pos[(int(a), int(b))] for a, b in zip(LT, LI)],
                                dtype=torch.long)
        return dict(item=item, st=st, ctx=ctx, house=house, B=len(trips),
                    li=LI, lt=LT, lctx=lctx, lslot=lslot, off=off, rix=rix)


class LinearIndex(torch.nn.Module):
    """A common additive utility layer shared by all external basket baselines.

    It covers the same observed covariates as the no-recency main runs, but deliberately
    remains the baseline parameterisation.  It must not be called byte-for-byte identical
    to the current RaggedModel index; that model has additional price constraints and
    centring/pooling transforms.
    """

    def __init__(self, J, N, S, K=32, Kp=8, Kt=8, Ks=4, seed=0,
                 taste_init=0.3):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lam = torch.nn.Parameter(torch.zeros(J))
        self.alpha = torch.nn.Parameter(torch.randn(J, K, generator=g) * taste_init)
        self.theta = torch.nn.Parameter(torch.randn(N, K, generator=g) * taste_init)
        self.gamma = torch.nn.Parameter(torch.randn(N, Kp, generator=g) * 0.1)
        self.beta = torch.nn.Parameter(torch.randn(J, Kp, generator=g) * 0.1)
        self.w_dsp = torch.nn.Parameter(torch.zeros(J))
        self.w_mlr = torch.nn.Parameter(torch.zeros(J))
        self.mu = torch.nn.Parameter(torch.randn(J, Kt, generator=g) * 0.1)
        self.delta = torch.nn.Parameter(torch.randn(52, Kt, generator=g) * 0.1)
        self.zeta = torch.nn.Parameter(torch.randn(J, Ks, generator=g) * 0.1)
        self.xi = torch.nn.Parameter(torch.randn(S, Ks, generator=g) * 0.1)

    def forward(self, it, trip, house, c):
        hh = house[trip]
        b = self.lam[it] + (self.theta[hh] * self.alpha[it]).sum(-1)
        b = b - (self.gamma[hh] * self.beta[it]).sum(-1) * c["dlp"]
        b = b + self.w_dsp[it] * c["disp"] + self.w_mlr[it] * c["mail"]
        b = b + (self.mu[it] * self.delta[c["week"]]).sum(-1)
        b = b + (self.zeta[it] * self.xi[c["store"]]).sum(-1)
        return b


class Bernoulli(torch.nn.Module):
    """Independent purchase, exactly normalized on a bounded non-empty support.

    Conditional on ``1 <= |S| <= nmax``, an independent Bernoulli model has

        P(S) = prod_{j in S} odds_j / sum_{r=1}^{nmax} e_r(odds).

    The factors ``prod_j (1-p_j)`` cancel.  Computing the bounded denominator matters for
    stores with few training visits: Laplace smoothing can otherwise assign material mass
    above the version-4 limit even though no observed basket is that large.
    """

    def __init__(self, J, N, S, **kw):
        super().__init__()
        self.idx = LinearIndex(J, N, S, **kw)

    def loglik(self, d, nmax=120, category_cap=120):
        b_slot = self.idx(d["item"], d["st"], d["house"], d["ctx"])
        b_line = self.idx(d["li"], d["lt"], d["house"], d["lctx"])
        n = torch.bincount(d["lt"], minlength=d["B"])
        if bool((n <= 0).any()) or int(n.max()) > nmax:
            raise ValueError("Bernoulli basket lies outside its bounded non-empty support")
        pos = torch.zeros(d["B"], dtype=b_slot.dtype, device=b_slot.device).index_add_(
            0, d["lt"], b_line)

        # Shift every trip so all odds passed to the polynomial recursion are <= 1.
        # The degree-r coefficient then receives exp(r*M) on the log scale.
        M = seg_max(b_slot, d["st"], d["B"])
        odds = torch.exp(b_slot - M[d["st"]])
        ix = d["rix"]
        degree = int(nmax)
        per_row_degree = min(int(category_cap), degree)
        e = esp_bucketed(odds, ix.row_of, ix.n_rows, per_row_degree,
                         ix.row_size, ix.item_pos, parallel=True)
        G = torch.zeros(d["B"] * ix.Cpad, degree + 1, dtype=b_slot.dtype,
                        device=b_slot.device)
        G[:, 0] = 1.0
        G[:, :per_row_degree + 1] = G[:, :per_row_degree + 1].index_copy(
            0, ix.flat_slot, e)
        A = poly_tree(G.view(d["B"], ix.Cpad, degree + 1), degree)
        degrees = torch.arange(degree + 1, dtype=b_slot.dtype, device=b_slot.device)
        log_terms = torch.log(A.clamp_min(1e-300)) + M[:, None] * degrees[None, :]
        log_norm = torch.logsumexp(log_terms[:, 1:], dim=1)
        return pos - log_norm


class DPP(torch.nn.Module):
    """Low-rank determinantal point process.  L = V V' + diag(d), exact normaliser."""

    def __init__(self, J, N, S, rank=16, interaction_init=0.1, **kw):
        super().__init__()
        self.idx = LinearIndex(J, N, S, **kw)
        g = torch.Generator().manual_seed(kw.get("seed", 0) + 1)
        self.V = torch.nn.Parameter(torch.randn(J, rank, generator=g) * interaction_init)
        self.rank = rank

    def loglik(self, d):
        q_slot = self.idx(d["item"], d["st"], d["house"], d["ctx"])
        out = []
        for b in range(d["B"]):
            m = d["st"] == b
            dg = torch.exp(q_slot[m])                       # diag of L, positive
            Vb = self.V[d["item"][m]]
            # log det(I + D + V V') = sum log(1+d) + logdet(I_k + V'(I+D)^-1 V)
            s = 1.0 + dg
            A = torch.eye(self.rank, dtype=dg.dtype) + Vb.T @ (Vb / s.unsqueeze(-1))
            log_norm = torch.log(s).sum() + torch.linalg.slogdet(A)[1]
            sl = d["lslot"][d["lt"] == b] - int(d["off"][b])
            Vs, ds = Vb[sl], dg[sl]
            L_S = Vs @ Vs.T + torch.diag(ds)
            log_num = torch.linalg.slogdet(L_S)[1]
            # P(empty) = 1/det(L+I), so the non-empty normaliser is det(L+I) - 1
            out.append(log_num - (log_norm + torch.log1p(-torch.exp(-log_norm))))
        return torch.stack(out)


def freq_loglik(D, Bt, trips, prior=1.0):
    """Empirical purchase rate per (store, product), Laplace-smoothed.  No fitted
    parameters; the floor any model must clear."""
    tr = np.flatnonzero(D["trip_split"] == 0)
    lp = D["line_ptr"]
    S, J = int(D["n_store"]), int(D["n_item"])
    cnt = np.zeros((S, J), np.float32)
    vis = np.zeros(S, np.float32)
    for t in tr:
        s = int(D["trip_store"][t])
        vis[s] += 1
        cnt[s, D["line_item"][int(lp[t]):int(lp[t + 1])]] += 1
    tot = 0.0
    nb = nl = 0
    for k in range(0, len(trips), 64):
        d = Bt.make(trips[k:k + 64])
        st, item = d["st"].numpy(), d["item"].numpy()
        store = D["trip_store"][trips[k:k + 64]][st]
        pi = (cnt[store, item] + prior) / (vis[store][:, None].squeeze() + 2 * prior)
        pi = np.clip(pi, 1e-9, 1 - 1e-9)
        lb = np.log(pi) - np.log1p(-pi)
        pos = np.zeros(d["B"])
        np.add.at(pos, d["lt"].numpy(), lb[d["lslot"].numpy()])
        norm = np.zeros(d["B"])
        np.add.at(norm, st, -np.log1p(-pi))
        lp_e = -norm
        tot += float((pos - norm - np.log1p(-np.exp(np.minimum(lp_e, -1e-9)))).sum())
        nb += d["B"]
        nl += len(d["li"])
    return tot / nb, tot / nl


def run(name, model, Bt, tr, va, a):
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    # Cosine decay, for the same reason the main model needed it: at a constant step size
    # the training loss stops falling well before an epoch and then oscillates, and a
    # comparison between models all stuck that way measures the optimiser, not the models.
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters,
                                                        eta_min=a.lr * 0.02)
             if a.cosine else None)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        d = Bt.make(tr[rng.choice(len(tr), size=a.batch, replace=False)])
        loss = -model.loglik(d).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if sched is not None:
            sched.step()
        if it % max(1, a.iters // 4) == 0:
            vb, vl = evaluate(model, Bt, va[:a.n_val])
            ep = it * a.batch / len(tr)
            log(f"   {name:10s} it {it:4d} ep {ep:5.3f}  train {float(loss):8.3f}  "
                f"val/basket {vb:8.3f}  val/line {vl:7.4f}  "
                f"{(time.time()-t0)/60:.1f} min")
    return evaluate(model, Bt, va[:a.n_val])


@torch.no_grad()
def evaluate(model, Bt, trips, chunk=64):
    tot, nb, nl = 0.0, 0, 0
    for k in range(0, len(trips), chunk):
        d = Bt.make(trips[k:k + chunk])
        tot += float(model.loglik(d).sum())
        nb += d["B"]
        nl += len(d["li"])
    return tot / nb, tot / nl


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    F = Features(J, S, 712)
    Bt = Batches(D, F)
    tr = np.flatnonzero(D["trip_split"] == 0)
    va = np.flatnonzero(D["trip_split"] == 1)
    log(f"{len(tr):,} training trips, {len(va):,} validation; "
        f"evaluating on {min(a.n_val, len(va)):,}")

    res = {}
    fb, fl = freq_loglik(D, Bt, va[:a.n_val])
    res["frequency"] = dict(val_per_basket=fb, val_per_line=fl, n_par=0)
    log(f"   {'frequency':10s} (no fitted parameters)      "
        f"val/basket {fb:8.3f}  val/line {fl:7.4f}")

    for name, mk in (("bernoulli", lambda: Bernoulli(J, N, S, K=a.K, Kp=a.Kp, seed=0)),
                     ("dpp", lambda: DPP(J, N, S, rank=a.rank, K=a.K, Kp=a.Kp, seed=0))):
        if name in a.skip:
            continue
        m = mk()
        npar = sum(p.numel() for p in m.parameters())
        log(f"   {name}: {npar:,} parameters")
        vb, vl = run(name, m, Bt, tr, va, a)
        res[name] = dict(val_per_basket=vb, val_per_line=vl, n_par=npar)

    log("")
    log(f"  {'model':12s} {'val/basket':>11s} {'val/line':>9s} {'params':>10s}")
    for k, v in res.items():
        log(f"  {k:12s} {v['val_per_basket']:11.3f} {v['val_per_line']:9.4f} "
            f"{v['n_par']:10,d}")
    json.dump(res, open(os.path.join(OUT, "v3_baselines.json"), "w"), indent=2)
    log("wrote out/v3_baselines.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--Kp", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--cosine", type=int, default=1)
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--skip", nargs="*", default=[])
    main(p.parse_args())
