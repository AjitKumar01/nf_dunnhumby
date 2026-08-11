"""
Fit version 3 to dunnhumby.

Starts with a TIMING PROBE, not a run.  The specification projected 2-8 hours per fit and
flagged that the previous projection of this kind was out by 18x; the corrected assortment
then grew the per-trip cost by about 1.6x on top.  So the first thing this does is measure
the cost of a hundred iterations and print the implied wall clock, before committing to
anything.  Pass --probe-only to stop there.

The objective is the likelihood of section 16: for each trip, the energy of the observed
basket minus log(Z - 1), where Z is the normaliser over every subset of the store's
assortment and the -1 conditions on the basket being non-empty.  There are no component
weights to choose.

Held-out likelihood is reported per basket and per line.  Per basket is the quantity the
model defines; per line is the one that can be compared against a model with a different
notion of a trip, and both are printed so neither can be quoted selectively.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from data import build
from features import Features
from ragged import RaggedIndex, RaggedModel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[fit] {m}", flush=True)


class Batcher:
    """Builds the ragged index and the per-slot features for a set of trips."""

    def __init__(self, D, F, nmax):
        self.D, self.F, self.nmax = D, F, nmax
        self.C = int(D["n_cat"])
        self.ptr = D["store_cat_ptr"]
        self.items = D["store_items"]
        self.lptr = D["line_ptr"]

    def make(self, trips):
        D, C = self.D, self.C
        it_l, row_of, row_trip, row_cat = [], [], [], []
        nrow = 0
        for bi, t in enumerate(trips):
            s = int(D["trip_store"][t]) * C
            for c in range(C):
                lo, hi = int(self.ptr[s + c]), int(self.ptr[s + c + 1])
                if hi <= lo:
                    continue
                it_l.append(self.items[lo:hi])
                row_of.append(np.full(hi - lo, nrow, np.int64))
                row_trip.append(bi)
                row_cat.append(c)
                nrow += 1
        item = np.concatenate(it_l)
        ix = RaggedIndex(item, np.concatenate(row_of),
                         np.array(row_trip, np.int64), np.array(row_cat, np.int64),
                         len(trips))
        store = torch.as_tensor(D["trip_store"][trips], dtype=torch.long)
        day = torch.as_tensor(D["trip_day"][trips], dtype=torch.long)
        week = torch.as_tensor(D["trip_week"][trips], dtype=torch.long)
        st_i, dy_i, wk_i = store[ix.item_trip], day[ix.item_trip], week[ix.item_trip]
        dlp, disp, mail = self.F.gather(ix.item, st_i, dy_i, wk_i)
        # week-of-year, per the spec: (WEEK_NO - 1) mod 52.  Clamping instead, as an earlier
        # version did, collapsed 54.6% of trips onto one seasonal parameter.
        user = torch.as_tensor(D["trip_user"][trips], dtype=torch.long)
        ctx = dict(dlp=dlp.double(), disp=disp.double(), mail=mail.double(),
                   week=(wk_i - 1) % 52, store=st_i,
                   rec=self.F.recency(ix.item, user[ix.item_trip], dy_i))
        li, lt, lc, lu = [], [], [], []
        for bi, t in enumerate(trips):
            a, b = int(self.lptr[t]), int(self.lptr[t + 1])
            li.append(D["line_item"][a:b])
            lc.append(D["line_cat"][a:b])
            lu.append(D["line_units"][a:b])
            lt.append(np.full(b - a, bi, np.int64))
        LI = torch.as_tensor(np.concatenate(li), dtype=torch.long)
        LT = torch.as_tensor(np.concatenate(lt), dtype=torch.long)
        # the SAME features, gathered at the purchased lines, so energy() and log_Z score
        # each product identically
        dlp_l, disp_l, mail_l = self.F.gather(LI, store[LT], day[LT], week[LT])
        lctx = dict(dlp=dlp_l.double(), disp=disp_l.double(), mail=mail_l.double(),
                    week=(week[LT] - 1) % 52, store=store[LT],
                    rec=self.F.recency(LI, user[LT], day[LT]))
        house = torch.as_tensor(D["trip_user"][trips], dtype=torch.long)
        return (ix, ctx, lctx, house,
                LI, LT, torch.as_tensor(np.concatenate(lc), dtype=torch.long),
                torch.as_tensor(np.concatenate(lu), dtype=torch.long))


def evaluate(m, B, trips, draws, gen, chunk=48, use_units=True):
    """Returns (set per basket, set per line, units per basket, total per basket).

    The SET component is reported apart from the units component because the baselines
    model sets only; quoting a total against them would be comparing different objects."""
    tot_s, tot_u, n_b, n_l = 0.0, 0.0, 0, 0
    for k in range(0, len(trips), chunk):
        sub = trips[k:k + chunk]
        ix, ctx, lctx, hh, li, lt, lc, lq = B.make(sub)
        m.house, m.ctx = hh, ctx
        with torch.no_grad():
            ll = m.loglik(ix, li, lt, lc, n_draws=draws, generator=gen, line_ctx=lctx)
            tot_s += float(ll.sum())
            if use_units:
                tot_u += float(m.units_loglik(li, lt, lq, lctx, ix.B).sum())
        n_b += len(sub)
        n_l += len(li)
    return tot_s / n_b, tot_s / n_l, tot_u / n_b, (tot_s + tot_u) / n_b


def main(a):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(a.seed)
    D = build()
    J, N, C, S = int(D["n_item"]), int(D["n_user"]), int(D["n_cat"]), int(D["n_store"])
    F = Features(J, S, 712)
    B = Batcher(D, F, a.nmax)

    tr = np.flatnonzero(D["trip_split"] == 0)
    va = np.flatnonzero(D["trip_split"] == 1)

    def in_support(idx):
        """A basket the normaliser does not sum over must not be scored by the energy.
        Clamping n to n_max, as an earlier version did, silently relabels such a basket as
        in-support; dropping it is the honest treatment and the count is reported."""
        lp, lc_ = D["line_ptr"], D["line_cat"]
        keep = np.ones(len(idx), bool)
        for i, t in enumerate(idx):
            lo, hi = int(lp[t]), int(lp[t + 1])
            if hi - lo > a.nmax or (hi > lo and np.bincount(lc_[lo:hi]).max() > a.R):
                keep[i] = False
        return keep

    kt, kv = in_support(tr), in_support(va)
    log(f"support (n_max={a.nmax}, R={a.R}): dropping {int((~kt).sum()):,} of {len(tr):,} "
        f"training ({(~kt).mean():.2%}) and {int((~kv).sum()):,} of {len(va):,} "
        f"validation ({(~kv).mean():.2%}) trips that lie outside it")
    tr, va = tr[kt], va[kv]
    log(f"{len(tr):,} training trips, {len(va):,} validation")

    m = RaggedModel(J=J, N=N, C=C, K=a.K, Kz=a.Kz, nmax=a.nmax, R=a.R, seed=a.seed,
                    S=S, Kp=a.Kp)
    npar = sum(p.numel() for p in m.parameters())
    log(f"parameters: {npar:,}  (K={a.K}, Kz={a.Kz}, Kp={a.Kp}, nmax={a.nmax}, R={a.R})")

    if a.resume:
        m.load_state_dict(torch.load(a.resume))
        log(f"resumed from {a.resume}")
    opt = torch.optim.Adam(m.parameters(), lr=a.lr, weight_decay=a.wd)
    # Cosine decay.  The previous run's training loss stopped falling at about a third of
    # an epoch and then oscillated, with held-out likelihood swinging over 1.4 nats and no
    # trend -- the signature of a step size too large for the stage, not of a model at
    # capacity.  A constant learning rate was the only schedule used until now.
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters,
                                                        eta_min=a.lr * a.lr_floor)
             if a.cosine else None)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)

    log("")
    log(f"timing probe: {a.probe} iterations at batch {a.batch}, {a.draws} draws")
    t0 = time.time()
    hist, ess_hist, n_skip = [], [], 0
    m.project(a.phi_max)
    for it in range(1, a.iters + 1):
        sub = tr[rng.choice(len(tr), size=a.batch, replace=False)]
        ix, ctx, lctx, hh, li, lt, lc, lq = B.make(sub)
        m.house, m.ctx = hh, ctx
        ll, ess = m.loglik(ix, li, lt, lc, n_draws=a.draws, generator=gen,
                           return_ess=True, line_ctx=lctx,
                           units=lq if a.units else None)
        loss = -ll.mean()
        # ESS GATE.  log Z is estimated by importance sampling; where the sampler has
        # collapsed the estimate is unreliable and biased DOWNWARD, which the objective
        # rewards (section 17).  The diverged run had ESS 0.016 on exactly the trips whose
        # energy had run away.  A batch below the floor carries no usable gradient, so it
        # is skipped rather than followed.
        e_bar = float(ess.mean())
        if e_bar < a.ess_floor:
            n_skip += 1
            opt.zero_grad()
        else:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip)
            opt.step()
            m.project(a.phi_max)           # keep lambda_max(Lambda) in the stable regime
        if sched is not None:
            sched.step()
        hist.append(float(loss))
        ess_hist.append(e_bar)
        if it == a.probe:
            dt = time.time() - t0
            per = dt / a.probe
            log(f"  {a.probe} iterations in {dt:.1f}s = {per:.3f}s/iteration")
            log(f"  slots per batch {ix.item.numel():,}, rows {ix.n_rows:,}, "
                f"Cpad {ix.Cpad}")
            log(f"  implied wall clock: {per * a.iters / 3600:.2f} h for {a.iters:,} "
                f"iterations")
            log(f"  loss {np.mean(hist[-20:]):.3f}   ESS {np.mean(ess_hist[-20:]):.3f}"
                f"   |phi| {float(m.phi.norm(dim=1).mean()):.3f}   skipped {n_skip}")
            if a.probe_only:
                json.dump(dict(sec_per_iter=per, iters=a.iters, n_par=npar,
                               slots=int(ix.item.numel()), ess=float(ess.mean())),
                          open(os.path.join(OUT, "v3_probe.json"), "w"), indent=2)
                log("  wrote out/v3_probe.json; stopping (--probe-only)")
                return
        if it % a.eval_every == 0:
            # lambda_max BEFORE evaluate(): evaluate reassigns m.house/m.ctx to the
            # validation chunks, and ix here is the training batch.  Calling it after
            # indexes a 40-trip batch into a 32-trip household tensor.
            lam_max = m.lambda_max(ix)
            vb, vl, vu, vt = evaluate(m, B, va[:a.n_val], a.draws * 2, gen, use_units=a.units)
            m.house, m.ctx = hh, ctx
            ep = it * a.batch / max(len(tr), 1)
            log(f"  it {it:5d} ep {ep:5.3f}  train {np.mean(hist[-a.eval_every:]):8.3f}  "
                f"set/bskt {vb:8.3f}  units/bskt {vu:7.3f}  total {vt:8.3f}  "
                f"ESS {np.mean(ess_hist[-a.eval_every:]):.3f}  "
                f"|phi| {float(m.phi.norm(dim=1).mean()):.3f}  "
                f"lam_max {lam_max:.3f}  skip {n_skip}  {(time.time()-t0)/60:.1f} min")
            if vb > 0:
                log("  ABORT: held-out log-likelihood is positive, which is impossible. "
                    "The objective is being maximised through a defect, not a fit.")
                return
            torch.save(m.state_dict(), os.path.join(OUT, f"v3_{a.label}.pt"))
    vb, vl, vu, vt = evaluate(m, B, va[:a.n_val], a.draws * 4, gen, use_units=a.units)
    log(f"final  set/basket {vb:.4f}  set/line {vl:.4f}  units/basket {vu:.4f}  total/basket {vt:.4f}")
    torch.save(m.state_dict(), os.path.join(OUT, f"v3_{a.label}.pt"))
    json.dump(dict(set_per_basket=vb, set_per_line=vl, units_per_basket=vu,
                   total_per_basket=vt, n_par=npar, iters=a.iters),
              open(os.path.join(OUT, f"v3_{a.label}.json"), "w"), indent=2)
    log(f"wrote out/v3_{a.label}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="run1")
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--Kz", type=int, default=12)
    p.add_argument("--Kp", type=int, default=8)
    p.add_argument("--nmax", type=int, default=60)
    p.add_argument("--R", type=int, default=4)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--n-val", type=int, default=768)
    p.add_argument("--probe", type=int, default=25)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--cosine", type=int, default=1)
    p.add_argument("--lr-floor", type=float, default=0.02)
    p.add_argument("--units", type=int, default=1)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--phi-max", type=float, default=0.25)   # 0.35 measures lambda_max 0.67
    p.add_argument("--ess-floor", type=float, default=0.30)
    p.add_argument("--clip", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
