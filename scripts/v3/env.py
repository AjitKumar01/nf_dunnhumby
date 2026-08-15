"""A retail environment: set prices, watch shoppers, collect margin.

The fitted basket model supplies the shopper.  This wraps it in the loop a markdown or
coupon policy needs -- reset, step, reward -- and nothing more.

Three design decisions worth stating, because each one could have been made dishonestly.

TRIP OCCASIONS ARE REPLAYED, NOT INVENTED.  Who shopped, on which day, at which store is
taken from the data.  Only what they put in the basket is simulated.  The model was never
given a theory of when people shop, so inventing arrivals would be putting a number into
the state that the data cannot support -- the provenance rule the spec calls Axiom O.  A
policy acting here changes what a real shopping trip contained, not how many trips happened.

COSTS ARE A DECLARED ASSUMPTION.  dunnhumby has prices and quantities but no cost of goods,
so unit cost is set as `(1 - margin0) x the product's own base price` and margin0 is an
argument, printed in the header of every run.  It is not measured and is not presented as
measured.  It matters because it creates the markdown trade-off: cutting price sells more
units but earns less on each, and with no cost term the trade-off is weaker than reality.

THE ACTION IS A LOG-PRICE SHIFT, which is exactly how price enters the model (b_j carries
-(gamma_h . beta_j) * dlp).  So an action of -0.105 is a 10% cut, applied on top of whatever
the real price deviation was that day, and the shopper's response is the model's own price
term rather than anything bolted on afterwards.

Reward is gross margin: sum over purchased lines of units x (price - cost).

Run the self-check:  python3 env.py --weeks 4
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from data import build
from features import Features
from fit import Batcher
from ragged import RaggedModel

HERE = os.path.dirname(os.path.abspath(__file__))
BI = os.path.join(HERE, "..", "..", "basket_input")


def log(m):
    print(f"[env] {m}", flush=True)


class RetailEnv:
    """One store, one week per step.

    obs    (store, week, n_trips, mean price deviation currently in force)
    action log-price shift, broadcast over products: scalar, [n_cat], or [n_item]
    reward gross margin over every basket generated that week
    """

    def __init__(self, model, D, F, Bt, margin0=0.30, split=1, device=None):
        self.m, self.D, self.F, self.Bt = model, D, F, Bt
        self.margin0 = margin0
        self.logp = torch.from_numpy(
            np.load(os.path.join(BI, "log_price.npy")).astype(np.float64))
        self.cat_of = torch.as_tensor(D["line_cat"], dtype=torch.long)
        # trips available to replay, grouped by (store, week)
        keep = np.flatnonzero(D["trip_split"] == split)
        self.trips = keep
        self.key = (D["trip_store"][keep].astype(np.int64) * 200
                    + D["trip_week"][keep].astype(np.int64))
        self.store = self.week = None

    # ---- gym-ish surface ---------------------------------------------------------
    def weeks_available(self, store):
        w = self.D["trip_week"][self.trips][self.D["trip_store"][self.trips] == store]
        return sorted(set(int(x) for x in w))

    def reset(self, store, week):
        self.store, self.week = int(store), int(week)
        self._action = None
        return self._obs()

    def _trips_now(self):
        sel = self.key == (self.store * 200 + self.week)
        return self.trips[sel]

    def _obs(self):
        t = self._trips_now()
        return dict(store=self.store, week=self.week, n_trips=len(t))

    @torch.no_grad()
    def step(self, action, generator=None, n_draws=32):
        """Apply the log-price shift, generate this week's baskets, return margin."""
        trips = self._trips_now()
        if len(trips) == 0:
            self.week += 1
            return self._obs(), 0.0, True, dict(n_trips=0, units=0, revenue=0.0)

        ix, ctx, lctx, hh, LI, LT, LC, LU = self.Bt.make(trips)
        self.m.house, self.m.ctx = hh, ctx
        shift = self._broadcast(action, ix)
        ctx = dict(ctx)
        ctx["dlp"] = ctx["dlp"] + shift
        self.m.ctx = ctx

        baskets = self.m.sample(ix, n_draws=n_draws, generator=generator)

        day = torch.as_tensor(self.D["trip_day"][trips], dtype=torch.long)
        rev = cost = 0.0
        n_units = n_lines = 0
        for b, items in enumerate(baskets):
            if not items:
                continue
            it = torch.as_tensor(items, dtype=torch.long)
            d = day[b].expand(len(it))
            base = torch.exp(self.logp[it, d])                      # observed shelf price
            sh = self._per_item_shift(action, it)
            price = base * torch.exp(sh)                            # the policy's price
            units = self._units(it, b, trips, price, base, generator)
            rev += float((price * units).sum())
            cost += float((base * (1.0 - self.margin0) * units).sum())
            n_units += int(units.sum())
            n_lines += len(it)
        self.week += 1
        done = self.week not in self.weeks_available(self.store)
        info = dict(n_trips=len(trips), lines=n_lines, units=n_units,
                    revenue=rev, cost=cost)
        return self._obs(), rev - cost, done, info

    # ---- internals ---------------------------------------------------------------
    def _broadcast(self, action, ix):
        a = torch.as_tensor(action, dtype=torch.float64)
        if a.ndim == 0:
            return a.expand(ix.item.shape[0])
        if a.numel() == int(self.D["n_cat"]):
            return a[ix.row_cat[ix.row_of]]
        return a[ix.item]

    def _per_item_shift(self, action, items):
        a = torch.as_tensor(action, dtype=torch.float64)
        if a.ndim == 0:
            return a.expand(len(items))
        if a.numel() == int(self.D["n_cat"]):
            raise NotImplementedError("category actions need the item->cat map at sample time")
        return a[items]

    def _units(self, items, b, trips, price, base, generator):
        """Units per line from the fitted shifted-negative-binomial, at the policy's price."""
        m = self.m
        hh = torch.as_tensor(self.D["trip_user"][trips][b], dtype=torch.long)
        from torch.nn.functional import softplus
        dlp = torch.log(price) - torch.log(base) + 0.0            # shift only
        z = m.a_q[items] - (softplus(m.gamma_q[hh]) * softplus(m.beta_q[items])).sum(-1) * dlp
        mu = torch.exp(z.clamp(-6.0, 4.0))
        r = softplus(m.log_r) + 1e-6
        p = r / (r + mu)
        # NB(r, p) by Gamma-Poisson mixture; units are 1 + k, matching units_loglik
        lam = torch._standard_gamma(r.expand(mu.shape)) * (1.0 - p) / p
        k = torch.poisson(lam, generator=generator)
        return 1.0 + k


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    F = Features(J, S, 712)
    Bt = Batcher(D, F, 120)
    m = RaggedModel(J=J, N=N, C=C, K=32, Kz=12, nmax=120, R=23, S=S, Kp=8)
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    m.double().eval()
    env = RetailEnv(m, D, F, Bt, margin0=a.margin0)
    log(f"checkpoint {os.path.basename(a.ckpt)}   assumed baseline margin {a.margin0:.0%} "
        f"(declared, not measured)")

    # busiest store, so a week has enough trips to say anything
    st = np.bincount(D["trip_store"][env.trips]).argmax()
    wks = env.weeks_available(st)[: a.weeks]
    log(f"store {st}, weeks {wks[0]}..{wks[-1]}")

    for name, act in (("no change", 0.0), ("10% cut", math.log(0.90)),
                      ("10% rise", math.log(1.10))):
        env.reset(st, wks[0])
        g = torch.Generator().manual_seed(0)
        tot_r = tot_u = tot_l = 0
        t0 = time.time()
        for _ in range(len(wks)):
            _, rew, done, info = env.step(act, generator=g)
            tot_r += rew
            tot_u += info["units"]
            tot_l += info["lines"]
            if done:
                break
        dt = time.time() - t0
        log(f"  {name:10s} margin {tot_r:9.2f}   units {tot_u:6d}   lines {tot_l:5d}"
            f"   {dt:5.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(HERE, "..", "..", "out", "v3_run18.pt"))
    p.add_argument("--weeks", type=int, default=4)
    p.add_argument("--margin0", type=float, default=0.30)
    main(p.parse_args())
