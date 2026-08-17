"""Does the model generalise from the validation weeks to the test weeks?

Every checkpoint this project has selected was chosen on VALIDATION likelihood, and until
distcheck was pointed at the test split nothing in the training loop had ever looked at weeks
91+.  The first look was alarming: run63's best checkpoint gives a per-trip E[n] of 7.72
against an observed 7.18 on validation, and 22.51 against an observed 6.94 on test -- the same
weights, the same code, the same draw count.

That is a 224% error on data the model was never selected against, versus 8% on data it was.
If it holds across the lineage it means the validation likelihood -- the number every decision
this session rested on -- is not measuring what it was taken to measure.

Variance is reported as the POPULATION variance, E[Var(n|trip)] + Var[E(n|trip)], because that
is what the observed variance of basket sizes measures.  The two terms are printed separately:
a model can be wrong about the spread within a trip, or about how much trips differ from each
other, and those are different faults.  The second is where the test failure shows -- Var[E]
runs 73 on validation and 1,122 on test, with per-trip E[n] reaching the nmax ceiling.

Run:  V3_AFFINITY=1 python3 testsplit.py
"""
import argparse
import os

import numpy as np
import torch

from data import build
from features import Features
from fit import Batcher
from ragged import RaggedModel


def log(m):
    print(f"[ts] {m}", flush=True)


def load_any(path, m, J, D):
    """run39-era bare state_dict, or the format-2 blob from fit.save_ckpt."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    it = None
    if isinstance(sd, dict) and sd.get("format") == 2:
        it = sd.get("iter")
        sd = sd["model"]
    missing, _ = m.load_state_dict(sd, strict=False)
    fitted = [k for k in missing if k != "cat_of"]
    if fitted:
        raise SystemExit(f"{os.path.basename(path)} missing fitted parameters: {fitted}")
    with torch.no_grad():
        co = torch.zeros(J, dtype=torch.long)
        co[torch.as_tensor(D["line_item"], dtype=torch.long)] = \
            torch.as_tensor(D["line_cat"], dtype=torch.long)
        m.cat_of.copy_(co)
    return it


def measure(m, Bt, D, trips, draws, chunk):
    ptr = D["line_ptr"]
    obs = np.array([int(ptr[t + 1]) - int(ptr[t]) for t in trips])
    obs = obs[(obs >= 1) & (obs <= 120)]
    Es, Vs = [], []
    for k in range(0, len(trips), chunk):
        ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(trips[k:k + chunk])
        m.house, m.ctx = hh, ctx
        with torch.no_grad():
            pn = m.size_dist(ix, n_draws=draws,
                             generator=torch.Generator().manual_seed(0))
            if isinstance(pn, tuple):
                pn = pn[0]
        p = pn.numpy()
        nn = np.arange(1, p.shape[1] + 1)
        e = (p * nn).sum(1)
        Es.append(e)
        Vs.append((p * nn ** 2).sum(1) - e ** 2)
    E = np.concatenate(Es)
    V = np.concatenate(Vs)
    return dict(obs_mean=obs.mean(), obs_var=obs.var(), e=E.mean(),
                v_within=V.mean(), v_spread=E.var(), v_pop=V.mean() + E.var(),
                e_max=E.max(), at_cap=float((E > 100).mean()))


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    F = Features(J, S, 712)
    Bt = Batcher(D, F, a.nmax)
    m = RaggedModel(J=J, N=N, C=C, K=32, Kz=12, nmax=a.nmax, R=a.R, S=S, Kp=8)

    # SAMPLE across each split, do not slice it.  Trips are stored in time order, so
    # [:n] takes the earliest trips only -- for validation that is week 83 alone and for
    # test week 91 alone.  Every earlier reading from this script compared those two single
    # weeks and was reported as a validation-vs-test result.  Week 91 turns out to be an
    # extreme outlier (+273% against +2% for week 99), so the split conclusion was an
    # artefact of which trips the slice happened to reach.
    _rng = np.random.default_rng(0)
    def _samp(split):
        idx = np.flatnonzero(D["trip_split"] == split)
        return np.sort(idx[_rng.choice(len(idx), size=min(a.n_trips, len(idx)),
                                       replace=False)])
    val = _samp(1)
    tst = _samp(2)
    log(f"{len(val)} validation trips (weeks 83-90), {len(tst)} test trips (weeks 91+), "
        f"{a.draws} draws")
    log("")
    log(f"{'checkpoint':>22}{'split':>6}{'E[n]':>8}{'obs':>7}{'err':>8}"
        f"{'var pop':>9}{'obs':>7}{'  = within + spread':>21}{'E[n]>100':>9}")

    for name in a.ckpts.split(","):
        path = os.path.join("..", "..", "out", name)
        if not os.path.exists(path):
            log(f"{name:>22}  (absent)")
            continue
        it = load_any(path, m, J, D)
        m.double().eval()
        tag = name.replace("v3_", "").replace("_best.pt", "").replace(".pt", "")
        for sp, trips in (("val", val), ("test", tst)):
            r = measure(m, Bt, D, trips, a.draws, a.chunk)
            err = 100.0 * (r["e"] - r["obs_mean"]) / max(r["obs_mean"], 1e-9)
            log(f"{tag:>22}{sp:>6}{r['e']:8.2f}{r['obs_mean']:7.2f}{err:+7.0f}%"
                f"{r['v_pop']:9.1f}{r['obs_var']:7.1f}"
                f"{r['v_within']:11.1f} +{r['v_spread']:8.1f}{100*r['at_cap']:8.1f}%")
    log("")
    log("err is the per-trip E[n] against the observed mean.  'E[n]>100' is the share of "
        "trips whose\npredicted size is near the nmax ceiling of 120 -- the failure is "
        "concentrated there, not spread evenly.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", default="v3_run39_best.pt,v3_run57_best.pt,v3_run60_best.pt,"
                                      "v3_run62_best.pt,v3_run63_best.pt")
    p.add_argument("--n-trips", type=int, default=384)
    p.add_argument("--draws", type=int, default=32)
    p.add_argument("--chunk", type=int, default=24)
    p.add_argument("--nmax", type=int, default=120)
    p.add_argument("--R", type=int, default=23)
    main(p.parse_args())
