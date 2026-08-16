"""
Three more baselines: nonsymmetric DPP, SHOPPER, and a BEMB-style multinomial set model.

Same rules as baselines.py -- exactly normalised P(S) over the store's assortment,
conditioned on non-empty -- so the numbers are comparable with each other and with
version 3's SET component.

NONSYMMETRIC DPP (Gartrell et al. 2019, 2021).  The important addition, and the strongest
competitor to version 3's central claim.  A symmetric DPP scores det(L_S) with L positive
semi-definite, which represents REPULSION only; that is why the symmetric variant finished
below a frequency baseline.  A nonsymmetric kernel

    L = D + V V' + B C B',      C block-diagonal with 2x2 blocks [[0, l], [-l, 0]]

adds a skew-symmetric part, and a skew part is exactly what lets a DPP express ATTRACTION
as well.  The normaliser stays exact, by the same determinant identity applied to the
stacked low-rank factor W = [V, B] with middle matrix M = blockdiag(I, C):

    det(I + D + W M W') = det(I + D) . det(I + M W'(I + D)^{-1} W)

If this matches version 3, the case for the enumerated within-category term is much weaker,
because a nonsymmetric DPP reaches both signs of interaction with an exact normaliser and no
Monte Carlo at all.

SHOPPER (Ruiz, Athey, Blei 2020).  The model this whole line of work descends from.  It is
sequential: items are chosen one at a time, each from a softmax over what is left, ending
with a checkout item, and the interaction enters as rho_c'(mean of alpha over the items so
far).  Its likelihood is over ORDERED baskets, so scoring a SET requires summing over
orderings:

    P(S) = sum over the n! orderings of P(ordering)
         = n! * E_{pi uniform}[ P(pi) ],  estimated by sampling orderings

That estimator is unbiased for P(S) and therefore biased LOW for log P(S) by Jensen, so
SHOPPER's number here is a lower bound on what it would score with the sum done exactly.
The bias is reported alongside it rather than hidden: it falls as the number of sampled
orderings rises, and both are printed.

BEMB-STYLE MULTINOMIAL.  BEMB models which item is chosen on a purchase occasion, not which
SET is bought, so it has no set likelihood of its own.  The faithful adaptation is a
conditional draw of n distinct items with weights w_j -- P(S | n) proportional to
prod_{j in S} w_j, normalised by the elementary symmetric polynomial e_n(w) -- with P(n)
from the empirical size distribution.  Worth stating plainly: that is exactly version 3
with phi = 0, rho_c = 0 and rho_0 set to the empirical size law, so it is a NESTED SPECIAL
CASE and the gap to it is precisely what the interaction buys once size is given away for
free.

Writes out/v3_baselines2.json.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from baselines import Batches, LinearIndex, evaluate
from data import build
from features import Features

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[bs2] {m}", flush=True)


class NDPP(torch.nn.Module):
    """Nonsymmetric low-rank DPP: L = diag(d) + V V' + B C B', C skew-symmetric."""

    def __init__(self, J, N, S, rank=16, srank=8, seed=0, **kw):
        super().__init__()
        self.idx = LinearIndex(J, N, S, seed=seed, **kw)
        g = torch.Generator().manual_seed(seed + 2)
        self.V = torch.nn.Parameter(torch.randn(J, rank, generator=g) * 0.1)
        self.B = torch.nn.Parameter(torch.randn(J, 2 * srank, generator=g) * 0.1)
        self.lam = torch.nn.Parameter(torch.ones(srank) * 0.5)
        self.rank, self.srank = rank, srank

    def _C(self):
        """Block-diagonal skew-symmetric middle matrix."""
        C = torch.zeros(2 * self.srank, 2 * self.srank, dtype=self.lam.dtype)
        for i in range(self.srank):
            C[2 * i, 2 * i + 1] = self.lam[i]
            C[2 * i + 1, 2 * i] = -self.lam[i]
        return C

    def loglik(self, d):
        q = self.idx(d["item"], d["st"], d["house"], d["ctx"])
        C = self._C()
        k, m = self.rank, 2 * self.srank
        M = torch.zeros(k + m, k + m, dtype=q.dtype)
        M[:k, :k] = torch.eye(k, dtype=q.dtype)
        M[k:, k:] = C
        out = []
        for b in range(d["B"]):
            msk = d["st"] == b
            dg = torch.exp(q[msk].clamp(-12, 6))
            W = torch.cat([self.V[d["item"][msk]], self.B[d["item"][msk]]], dim=1)
            s = 1.0 + dg
            A = torch.eye(k + m, dtype=q.dtype) + M @ (W / s.unsqueeze(-1)).T @ W
            log_norm = torch.log(s).sum() + torch.linalg.slogdet(A)[1]
            sl = d["lslot"][d["lt"] == b] - int(d["off"][b])
            Ws, ds = W[sl], dg[sl]
            L_S = Ws @ M @ Ws.T + torch.diag(ds)
            log_num = torch.linalg.slogdet(L_S)[1]
            out.append(log_num - (log_norm + torch.log1p(-torch.exp(-log_norm))))
        return torch.stack(out)


def esp_tree(W, nmax):
    """e_0..e_nmax over a padded [B, P] weight matrix, by a balanced product tree.

    The generating polynomial is prod_j (1 + w_j u) and e_n is its degree-n coefficient.
    The sequential recursion needs P steps -- 5,420 at a dunnhumby store -- which in torch
    means 5,420 kernel launches per batch and was 12x slower than every other baseline.
    Multiplying the degree-1 factors pairwise in a tree needs log2(P) ~ 13 rounds instead,
    at somewhat more arithmetic but far fewer launches.  Padded slots carry w = 0, whose
    factor is (1, 0), the identity -- so no masking is needed.
    """
    B, P = W.shape
    poly = torch.stack([torch.ones_like(W), W], dim=-1)          # [B, P, 2]
    while poly.shape[1] > 1:
        n, d = poly.shape[1], poly.shape[-1]
        if n % 2:                                                 # pad with the identity
            idp = torch.zeros(B, 1, d, dtype=W.dtype, device=W.device)
            idp[:, :, 0] = 1.0
            poly = torch.cat([poly, idp], dim=1)
            n += 1
        a, b = poly[:, 0::2], poly[:, 1::2]
        nd = min(2 * d - 1, nmax + 1)
        out = torch.zeros(B, n // 2, nd, dtype=W.dtype, device=W.device)
        for r in range(min(d, nd)):
            take = min(d, nd - r)
            out[..., r:r + take] = out[..., r:r + take] + a[..., :take] * b[..., r:r + 1]
        poly = out
    return poly[:, 0, :]


class Multinomial(torch.nn.Module):
    """BEMB-style: n distinct items drawn with weights w, size from the empirical law.

    P(S) = P(n) * prod_{j in S} w_j / e_n(w),  e_n the elementary symmetric polynomial.
    """

    def __init__(self, J, N, S, size_law, seed=0, **kw):
        super().__init__()
        self.idx = LinearIndex(J, N, S, seed=seed, **kw)
        self.register_buffer("log_pn", torch.log(torch.as_tensor(size_law) + 1e-12))

    def loglik(self, d):
        w = self.idx(d["item"], d["st"], d["house"], d["ctx"])
        wl = self.idx(d["li"], d["lt"], d["house"], d["lctx"])
        B = d["B"]
        cnt = torch.bincount(d["st"], minlength=B)
        P = int(cnt.max())
        off = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(cnt, 0)[:-1]])
        pos = torch.arange(len(w)) - off[d["st"]]
        M = torch.full((B,), -float("inf"), dtype=w.dtype).index_reduce_(
            0, d["st"], w, "amax", include_self=True)
        W = torch.zeros(B * P, dtype=w.dtype).index_copy(
            0, d["st"] * P + pos, torch.exp(w - M[d["st"]])).view(B, P)
        n = torch.bincount(d["lt"], minlength=B)
        e = esp_tree(W, int(n.max()))                                  # [B, nmax+1]
        num = torch.zeros(B, dtype=w.dtype).index_add_(0, d["lt"], wl)
        le = torch.log(e.gather(1, n.unsqueeze(1)).squeeze(1).clamp_min(1e-300))
        return (self.log_pn[n.clamp(max=len(self.log_pn) - 1)]
                + num - (le + n.to(w.dtype) * M))


class Shopper(torch.nn.Module):
    """Sequential choice with an interaction on the running mean of alpha, plus checkout.

    Set probability by averaging over sampled orderings: P(S) = n! E_pi[P(pi)].  Unbiased
    for P(S), so biased low for log P(S); the script reports the estimate at two ordering
    counts so the size of that bias is visible.
    """

    def __init__(self, J, N, S, K=32, Ki=16, seed=0, **kw):
        super().__init__()
        self.idx = LinearIndex(J, N, S, K=K, seed=seed, **kw)
        g = torch.Generator().manual_seed(seed + 3)
        self.rho = torch.nn.Parameter(torch.randn(J, Ki, generator=g) * 0.1)
        self.alpha_i = torch.nn.Parameter(torch.randn(J, Ki, generator=g) * 0.1)
        self.checkout = torch.nn.Parameter(torch.zeros(1))
        self.Ki = Ki

    def loglik(self, d, n_orders=4, gen=None):
        psi = self.idx(d["item"], d["st"], d["house"], d["ctx"])
        out = []
        for b in range(d["B"]):
            msk = d["st"] == b
            ps, it = psi[msk], d["item"][msk]
            rho, al = self.rho[it], self.alpha_i[it]
            sl = (d["lslot"][d["lt"] == b] - int(d["off"][b])).tolist()
            n = len(sl)
            per = []
            for _ in range(n_orders):
                order = list(np.random.permutation(sl))
                alive = torch.ones(len(ps), dtype=torch.bool)
                run = torch.zeros(self.Ki, dtype=ps.dtype)
                tot = 0.0
                for i, j in enumerate(order):
                    u = ps + (rho @ run if i else torch.zeros_like(ps))
                    u = torch.cat([u.masked_fill(~alive, -1e30), self.checkout])
                    tot = tot + torch.log_softmax(u, 0)[j]
                    run = (run * i + al[j]) / (i + 1)
                    alive[j] = False
                u = ps + rho @ run
                u = torch.cat([u.masked_fill(~alive, -1e30), self.checkout])
                tot = tot + torch.log_softmax(u, 0)[-1]      # checkout ends the basket
                per.append(tot)
            lp = torch.logsumexp(torch.stack(per), 0) - math.log(n_orders)
            out.append(lp + float(math.lgamma(n + 1)))       # + log n!
        return torch.stack(out)


def size_law(D, nmax=120):
    tr = np.flatnonzero(D["trip_split"] == 0)
    n = np.clip(D["trip_nlines"][tr], 0, nmax)
    c = np.bincount(n, minlength=nmax + 1).astype(np.float64) + 0.5
    return c / c.sum()


def run(name, model, Bt, tr, va, a, te=None, **kw):
    # Save, and be able to resume.
    #
    # Only SHOPPER wrote a checkpoint, so when it turned out that 12,000 iterations still
    # left every baseline descending -- 0.12 nats/1000 for SHOPPER, 0.43 for the DPP --
    # their training could not be continued and had to be thrown away and restarted.  Any
    # model that might need a longer run must be resumable, which is all of them.
    ck = os.path.join(OUT, f"v3_bl_{name}.pt")
    if a.resume and os.path.exists(ck):
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        log(f"   {name}: resumed from {os.path.basename(ck)}")
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
        loss = -model.loglik(d, **kw).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if sched is not None:
            sched.step()
        # Log on a fixed cadence, not iters//3.  At --iters 60000 that was one line every
        # 20,000 iterations, so a run that might take 40 hours showed nothing for the first
        # 13 -- no way to tell whether it was on track or to revise the plan.
        if it % a.log_every == 0 or it == a.iters:
            vb, vl = ev(model, Bt, va[:min(128,len(va))], **kw)
            _el = time.time() - t0
            log(f"   {name}: it {it}/{a.iters}  {_el/it:.3f} s/it  "
                f"projected {_el/it*a.iters/3600:.1f} h  val/bskt {vb:.3f}")
            ep = it * a.batch / len(tr)
            log(f"   {name:10s} it {it:4d} ep {ep:5.3f}  train {float(loss):9.3f}  "
                f"val/basket {vb:9.3f}  val/line {vl:7.4f}  {(time.time()-t0)/60:.1f} min")
            os.makedirs(OUT, exist_ok=True)
            torch.save(model.state_dict(), ck)
    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), ck)
    vb, vl = ev(model, Bt, va, **kw)
    if te is not None:
        tb, tl = ev(model, Bt, te, **kw)
        log(f"   {name}: valid/basket {vb:9.3f}  test/basket {tb:9.3f}  "
            f"valid/line {vl:7.3f}  test/line {tl:7.3f}")
        return vb, vl, tb, tl
    return vb, vl


@torch.no_grad()
def ev(model, Bt, trips, chunk=32, **kw):
    tot, nb, nl = 0.0, 0, 0
    for k in range(0, len(trips), chunk):
        d = Bt.make(trips[k:k + chunk])
        tot += float(model.loglik(d, **kw).sum())
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
    # SAMPLE the held-out sets, do not slice them.  Trips are stored in time order, so
    # `va[:256]` is week 83 alone and the old test slice was week 91 alone -- and week 91 is
    # an extreme outlier for the main model (+273% on E[n] against +2% for week 99).  Every
    # baseline number published before this compared one week against one week.  TEST is
    # scored too: nothing here had ever been evaluated on weeks 91+.
    _rng = np.random.default_rng(0)
    def _samp(split):
        idx = np.flatnonzero(D["trip_split"] == split)
        return np.sort(idx[_rng.choice(len(idx), size=min(a.n_val, len(idx)),
                                       replace=False)])
    va, te = _samp(1), _samp(2)
    log(f"{len(tr):,} training trips; evaluating on {len(va):,} validation and "
        f"{len(te):,} test trips, sampled across all weeks of each")
    res = {}

    if "shopper" not in a.skip:
        m = Shopper(J, N, S, K=a.K, Kp=a.Kp)
        vb, vl, tb, tl = run("shopper", m, Bt, tr, va, a, te=te, n_orders=a.orders)
        # P(S) = n! E_pi[P(pi)] is unbiased for P(S), so log of it is biased LOW.  The
        # bias falls with the number of sampled orderings; report a ladder rather than
        # quote one point and hope.
        torch.save(m.state_dict(), os.path.join(OUT, "v3_shopper.pt"))
        # The ladder must be run on the SAME trips as the headline number, and far enough
        # out to show whether it has converged.  An earlier version stopped at 256 on a
        # 192-trip subset and read 0.198 / 0.041 / 0.077 -- steps that do not shrink
        # monotonically, which is noise rather than convergence.
        ladder = {}
        for no in a.ladder:
            ladder[no] = ev(m, Bt, va, n_orders=no)[0]
            log(f"   shopper, {no:4d} orderings on {a.n_val} trips: {ladder[no]:8.3f}")
        # the headline is the ladder's converged rung, not the cheap training-time
        # evaluation: they differed by 0.25 nats last time and the json carried the worse one
        vb = ladder[max(ladder)]
        res["shopper"] = dict(val_per_basket=vb, val_per_line=vl,
                            test_per_basket=tb, test_per_line=tl,
                              ladder={str(k): v for k, v in ladder.items()},
                              orders=a.orders,
                              n_par=sum(p.numel() for p in m.parameters()))

    if "ndpp" not in a.skip:
        m = NDPP(J, N, S, rank=a.rank, srank=a.srank, K=a.K, Kp=a.Kp)
        vb, vl, tb, tl = run("ndpp", m, Bt, tr, va, a, te=te)
        res["ndpp"] = dict(val_per_basket=vb, val_per_line=vl,
                            test_per_basket=tb, test_per_line=tl,
                           n_par=sum(p.numel() for p in m.parameters()))
    if "multinomial" not in a.skip:
        m = Multinomial(J, N, S, size_law(D), K=a.K, Kp=a.Kp)
        vb, vl, tb, tl = run("multinom", m, Bt, tr, va, a, te=te)
        res["multinomial"] = dict(val_per_basket=vb, val_per_line=vl,
                            test_per_basket=tb, test_per_line=tl,
                                  n_par=sum(p.numel() for p in m.parameters()))

    log("")
    log(f"  {'model':14s} {'val/basket':>11s} {'val/line':>9s} {'params':>10s}")
    for k, v in res.items():
        log(f"  {k:14s} {v['val_per_basket']:11.3f} {v.get('test_per_basket',float('nan')):11.3f} {v['val_per_line']:9.4f} "
            f"{v['n_par']:10,d}")
    tag = a.tag or "all"
    json.dump(res, open(os.path.join(OUT, f"v3_baselines2_{tag}.json"), "w"),
              indent=2)
    log(f"wrote out/v3_baselines2_{tag}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--Kp", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--srank", type=int, default=8)
    p.add_argument("--orders", type=int, default=4)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--resume", type=int, default=0)
    p.add_argument("--cosine", type=int, default=1)
    p.add_argument("--n-val", type=int, default=256)
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--tag", default="")
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--ladder", type=int, nargs="*",
                   default=[8, 32, 128, 512, 2048])
    main(p.parse_args())
