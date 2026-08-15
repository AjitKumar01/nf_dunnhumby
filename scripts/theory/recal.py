"""Recalibrate the size law on held-out data, without disturbing anything else.

Three attempts to fix basket-size calibration during training failed the same way: the
penalty matched the size law on the TRAINING batch and the held-out size law drifted anyway
(E[n] 18.7 against an observed 7.3, median 9.9 against 4.0).  An in-sample penalty cannot
close an out-of-sample gap, and each attempt also cost likelihood.

The structure of the model offers a cleaner route.  P(n | z) is proportional to
exp(-rho_0(n)) A_n(z): rho_0 is a free function of size alone, 120 parameters, and it enters
NOTHING else.  So it can be refit on held-out data to match the observed size law while
every item parameter -- and therefore the whole price response, which travels through b_j --
is left exactly as trained.  This is recalibration in the Platt sense, not retraining.

The split is honest: rho_0 is refit on the validation weeks (83-90) and every number is
reported on the TEST weeks (91+), which are untouched by both training and this fit.

Run:  python3 recal.py --ckpt ../../out/v3_run18.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v3"))

from data import build                                             # noqa: E402
from features import Features                                      # noqa: E402
from fit import Batcher, evaluate                                  # noqa: E402
from ragged import RaggedModel                                     # noqa: E402


def log(m):
    print(f"[cal] {m}", flush=True)


def size_stats(m, Bt, trips, draws, chunk=32, seed=0):
    """Held-out E[n], its median, and the observed counterpart."""
    es = []
    for k in range(0, len(trips), chunk):
        ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(trips[k:k + chunk])
        m.house, m.ctx = hh, ctx
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            e, _ = m.size_moments(ix, n_draws=draws, generator=g)
        es.append(e.numpy())
    return np.concatenate(es)


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    F = Features(J, S, 712)
    Bt = Batcher(D, F, 120)
    m = RaggedModel(J=J, N=N, C=C, K=32, Kz=12, nmax=120, R=23, S=S, Kp=8)
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    m.double().eval()
    log(f"checkpoint {os.path.basename(a.ckpt)}")

    cal = np.flatnonzero(D["trip_split"] == 1)[: a.n_cal]
    tst = np.flatnonzero(D["trip_split"] == 2)[: a.n_test]
    log(f"rho_0 refit on {len(cal)} validation trips; everything reported on "
        f"{len(tst)} TEST trips (weeks 91+), untouched by training and by this fit")

    obs_t = D["trip_nlines"][tst]
    before = size_stats(m, Bt, tst, a.draws)
    gen = torch.Generator().manual_seed(0)
    vb0, vl0, vu0, vt0 = evaluate(m, Bt, tst, a.draws, gen, use_units=True)
    log("")
    log(f"BEFORE   E[n] mean {before.mean():6.2f}  median {np.median(before):6.2f}"
        f"   observed mean {obs_t.mean():5.2f}  median {np.median(obs_t):5.2f}")
    log(f"         test set/basket {vb0:.4f}")

    # ---- refit rho_0 alone -------------------------------------------------------------
    for p in m.parameters():
        p.requires_grad_(False)
    m.rho_0_free.requires_grad_(True)
    opt = torch.optim.Adam([m.rho_0_free], lr=a.lr)
    rng = np.random.default_rng(0)
    emp = np.bincount(np.clip(D["trip_nlines"][cal], 1, 120), minlength=121)[1:] + 0.5
    emp = torch.as_tensor(emp / emp.sum())
    for it in range(1, a.iters + 1):
        sub = cal[rng.choice(len(cal), size=a.batch, replace=False)]
        ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(sub)
        m.house, m.ctx = hh, ctx
        g = torch.Generator().manual_seed(it)
        pn = m.size_dist(ix, n_draws=a.draws, generator=g, grad=True)
        pbar = pn.mean(0).clamp_min(1e-12)
        pbar = pbar / pbar.sum()
        loss = -(emp[: pbar.shape[0]] * pbar.log()).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % max(1, a.iters // 5) == 0:
            log(f"  it {it:4d}  cross-entropy to the observed size law {float(loss):.4f}")

    after = size_stats(m, Bt, tst, a.draws)
    gen = torch.Generator().manual_seed(0)
    vb1, vl1, vu1, vt1 = evaluate(m, Bt, tst, a.draws, gen, use_units=True)
    log("")
    log(f"AFTER    E[n] mean {after.mean():6.2f}  median {np.median(after):6.2f}"
        f"   observed mean {obs_t.mean():5.2f}  median {np.median(obs_t):5.2f}")
    log(f"         test set/basket {vb1:.4f}   ({vb1 - vb0:+.4f} vs before)")
    out = a.ckpt.replace(".pt", "_cal.pt")
    torch.save(m.state_dict(), out)
    log(f"wrote {os.path.basename(out)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="../../out/v3_run18.pt")
    p.add_argument("--n-cal", type=int, default=4096)
    p.add_argument("--n-test", type=int, default=512)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.05)
    main(p.parse_args())
