"""
The basket-size response to price, on the fitted model and real held-out trips.

This measures the one quantity the baselines cannot produce.  In the BEMB-style
multinomial, basket size is drawn from `log_pn`, a buffer built by counting training
baskets; it is indexed by n alone and never sees a price.  So a markdown can change WHICH
products go in the basket, but the number of them is settled before any price is consulted
-- dE[n]/dlog p is identically zero, by construction rather than by fit.  For a markdown or
coupon policy that is the whole mechanism of interest, so it is worth measuring directly
rather than asserting.

Three things are computed, in increasing order of what they claim:

  1. E[n] against the observed mean basket size.  A calibration check: if the model's size
     law is wrong, nothing downstream of it means anything.

  2. Proposition 1 under a UNIFORM shift, where it reduces to dE[n]/de = Var(n).  Both
     sides are computed here by genuinely different routes -- the right-hand side from the
     size distribution, the left by re-running the full normaliser with every b_j shifted
     -- so agreement is evidence the proposition holds on the FITTED model with real
     assortments, not only in the synthetic verifier where it was first checked.

  3. The price response itself, as an elasticity and as the answer to "cut every shelf
     price 10%, how much bigger is the basket".

Common random numbers throughout: every evaluation draws its Gaussians from a generator
re-seeded to the same value, so the perturbed and unperturbed estimates share their noise
and the difference between them is a response rather than two independent errors.  Without
this the per-trip Monte Carlo spread (median 0.04 nats, worst 2.3) would swamp the effect.

Run:  python3 price_response.py --ckpt ../../out/v3_run11.pt --n-trips 256
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v3"))

from data import build                                             # noqa: E402
from features import Features                                      # noqa: E402
from fit import Batcher                                            # noqa: E402
from ragged import RaggedModel                                     # noqa: E402


def log(m):
    print(f"[prc] {m}", flush=True)


def moments(m, ix, ctx, seed, draws, dlp_shift=0.0, b_shift=0.0, zh=None):
    """(E[n], Var(n)) with the price and index shifts applied, on a COMMON proposal.

    zh is the proposal centre, located once at the unperturbed parameters and passed to
    every call.  Relocating it per perturbation makes each difference carry a change of
    proposal alongside the effect; see size_dist.
    """
    c = dict(ctx)
    c["dlp"] = ctx["dlp"] + dlp_shift
    keep = m.ctx
    m.ctx = c
    with torch.no_grad():
        m.lam += b_shift
    try:
        g = torch.Generator().manual_seed(seed)
        e, v = m.size_moments(ix, n_draws=draws, generator=g, z_fixed=zh)
    finally:
        m.ctx = keep
        with torch.no_grad():
            m.lam -= b_shift
    return e, v


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    F = Features(int(D["n_item"]), int(D["n_store"]), 712)   # day panel width, per fit.py
    Bt = Batcher(D, F, a.nmax)

    m = RaggedModel(J=int(D["n_item"]), N=int(D["n_user"]), C=int(D["n_cat"]),
                    K=a.K, Kz=a.Kz, nmax=a.nmax, R=a.R, S=int(D["n_store"]), Kp=a.Kp)
    sd = torch.load(a.ckpt, map_location="cpu")
    m.load_state_dict(sd)
    m.double().eval()
    log(f"loaded {os.path.basename(a.ckpt)}")

    va = np.flatnonzero(D["trip_split"] == 1)[: a.n_trips]
    ix, ctx, lctx, house, LI, LT, LC, LU = Bt.make(va)
    m.house = house
    m.ctx = ctx
    obs = np.bincount(LT.numpy(), minlength=len(va))
    log(f"{len(va)} held-out trips, mean observed basket {obs.mean():.3f} lines")

    # the proposal centre, located ONCE and shared by every evaluation below
    g = torch.Generator().manual_seed(a.seed)
    _, zh = m.size_dist(ix, n_draws=8, generator=g, return_mode=True)

    # ---- 1. calibration ---------------------------------------------------------------
    e0, v0 = moments(m, ix, ctx, a.seed, a.draws, zh=zh)
    log("")
    log(f"1. calibration      E[n] {float(e0.mean()):7.3f}   observed {obs.mean():7.3f}"
        f"   gap {float(e0.mean()) - obs.mean():+.3f}")
    log(f"                    sd[n] {float(v0.sqrt().mean()):6.3f}   observed "
        f"{obs.std():7.3f}")

    # ---- 2. Proposition 1 under a uniform shift ---------------------------------------
    # b_j -> b_j + e for every j gives d'x = n, so Cov(n, d'x) = Var(n).
    eps = a.eps
    ep, _ = moments(m, ix, ctx, a.seed, a.draws, b_shift=+eps, zh=zh)
    en, _ = moments(m, ix, ctx, a.seed, a.draws, b_shift=-eps, zh=zh)
    fd = (ep - en) / (2 * eps)
    log("")
    log(f"2. Proposition 1, uniform shift (eps {eps})")
    log(f"   dE[n]/de  central difference   {float(fd.mean()):8.4f}")
    log(f"   Var(n)    from the size law    {float(v0.mean()):8.4f}")
    rel = float((fd - v0).abs().mean() / v0.mean())
    log(f"   mean abs disagreement          {float((fd - v0).abs().mean()):8.4f}"
        f"   ({rel:.2%} of Var(n))")
    bad = int(((fd - v0).abs() > 0.25 * v0).sum())
    log(f"   trips off by more than 25%     {bad} / {len(va)}")

    # ---- 3. the price response --------------------------------------------------------
    d0 = a.delta
    ep, _ = moments(m, ix, ctx, a.seed, a.draws, dlp_shift=+d0, zh=zh)
    en, _ = moments(m, ix, ctx, a.seed, a.draws, dlp_shift=-d0, zh=zh)
    slope = (ep - en) / (2 * d0)                       # dE[n] / dlog p
    cut = math.log(1 - a.cut)
    ec, _ = moments(m, ix, ctx, a.seed, a.draws, dlp_shift=cut, zh=zh)
    log("")
    log(f"3. price response (log-price shift +-{d0})")
    log(f"   dE[n]/dlog p                   {float(slope.mean()):+8.4f} lines")
    log(f"   elasticity  (dlog E[n]/dlog p) {float(slope.mean()) / float(e0.mean()):+8.4f}")
    log(f"   {a.cut:.0%} cut on every shelf price:  E[n] {float(e0.mean()):.3f}"
        f" -> {float(ec.mean()):.3f}   {float(ec.mean() - e0.mean()):+.3f} lines"
        f"  ({float((ec.mean() - e0.mean()) / e0.mean()):+.2%})")
    pos = int((ec > e0).sum())
    log(f"   trips whose basket grows       {pos} / {len(va)}  ({pos / len(va):.1%})")

    # ---- the same question, asked of the baseline -------------------------------------
    from baselines2 import size_law
    pn = size_law(D)
    nn = np.arange(len(pn))
    log("")
    log("   BEMB multinomial, same 10% cut:")
    log(f"     E[n] {float((pn * nn).sum()):.3f} -> {float((pn * nn).sum()):.3f}"
        f"   {0.0:+.3f} lines   (log_pn is a buffer indexed by n; price is not an input)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="../../out/v3_run11.pt")
    p.add_argument("--n-trips", type=int, default=256)
    p.add_argument("--draws", type=int, default=128)
    p.add_argument("--eps", type=float, default=0.005)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--cut", type=float, default=0.10)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--Kz", type=int, default=12)
    p.add_argument("--Kp", type=int, default=8)
    p.add_argument("--nmax", type=int, default=120)
    p.add_argument("--R", type=int, default=23)
    p.add_argument("--seed", type=int, default=7)
    main(p.parse_args())
