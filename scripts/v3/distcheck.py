"""Compare the basket-size LAW, not two of its moments.

The training objective already optimises the whole size distribution -- fit.py builds a
cross-entropy against the empirical law -- but the acceptance goals read only the mean
(within 25%) and the variance (within 40%).  So the model is trained on a distribution and
graded on its summary, and those bands are wide enough not to identify it.  Measured on the
16,948 held-out baskets (mean 7.82, var 81.4):

    the observed law itself     E-goal PASS  var-goal PASS   KL   0.000   TV 0.000
    negative binomial (mu,var)  E-goal PASS  var-goal PASS   KL   0.035   TV 0.113
    geometric (mean only)       E-goal PASS  var-goal PASS   KL   0.039   TV 0.114
    two spikes at mean +- sd    E-goal PASS  var-goal PASS   KL 554.353   TV 0.807

A law placing every basket on one of two sizes passes both goals with 81% of its mass in the
wrong place.  This project has made the same mistake repeatedly in other guises -- a mean
standing in for a tail -- most expensively when the pi-weighted budget read 0.232 against a
true lambda_max of 2.202.

Nothing here touches training.  It loads a checkpoint and reports; fit.py is unmodified, so
past runs can be scored on the same footing as new ones.

    KL(data || model)  in nats, the same functional the loss already uses, and directly
                       comparable to the set/basket numbers beside it.  Asymmetric in the
                       useful direction: it punishes the model for putting little mass where
                       real baskets are.  A negative binomial fitted to both moments scores
                       0.035, so < 0.05 means "as good as a well-fitted NB".
    TV                 half the L1 distance: the fraction of probability mass misplaced.
    KS                 max gap between the CDFs, reported as an EFFECT SIZE.  Deliberately
                       no p-value: at n = 16,948 the 5% critical value is
                       1.36/sqrt(16948) = 0.0104, so any deviation past 1% "rejects" and the
                       test carries no information at this sample size.

Also re-does sampler-agrees distributionally.  The sampled and analytic size laws are the
same distribution reached two ways, so comparing their MEANS (9.3 vs 9.4) is weaker than
comparing the laws.  That goal is the only one that ever caught a failure the others
structurally could not -- 8.65 analytic against 25.51 sampled -- because every other number
is derived from the size law and so agrees with itself when the size law is wrong.

Run:  V3_AFFINITY=1 python3 distcheck.py --ckpt ../../out/v3_run60_best.pt
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
    print(f"[dc] {m}", flush=True)


def compare(p_model, p_obs, k):
    """KL(obs || model), total variation, KS statistic, and the two moments."""
    m = p_obs > 0
    kl = float((p_obs[m] * np.log(p_obs[m] / np.clip(p_model[m], 1e-300, None))).sum())
    tv = float(0.5 * np.abs(p_model - p_obs).sum())
    ks = float(np.abs(np.cumsum(p_model) - np.cumsum(p_obs)).max())
    e = float((k * p_model).sum())
    v = float((k * k * p_model).sum() - e * e)
    return kl, tv, ks, e, v


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    F = Features(J, S, 712)
    Bt = Batcher(D, F, a.nmax)
    m = RaggedModel(J=J, N=N, C=C, K=32, Kz=12, nmax=a.nmax, R=a.R, S=S, Kp=8)
    # Older checkpoints (run39 and before) predate the cat_of buffer, so a strict load
    # fails on them.  cat_of is a derived product -> category map, not a fitted parameter,
    # so it is rebuilt from the data exactly as fit.py does rather than left at zeros --
    # the point of a standalone evaluator is that it scores OLD runs on the same footing.
    sd = torch.load(a.ckpt, map_location="cpu")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    with torch.no_grad():
        _co = torch.zeros(J, dtype=torch.long)
        _co[torch.as_tensor(D["line_item"], dtype=torch.long)] = \
            torch.as_tensor(D["line_cat"], dtype=torch.long)
        m.cat_of.copy_(_co)
    fitted = [k for k in missing if k != "cat_of"]
    if fitted:
        raise SystemExit(f"checkpoint is missing FITTED parameters, not just buffers: "
                         f"{fitted}")
    if unexpected:
        log(f"note: checkpoint carries keys this model does not use: {unexpected}")
    m.double().eval()
    log(f"checkpoint {os.path.basename(a.ckpt)}"
        + ("  (cat_of rebuilt from data)" if "cat_of" in missing else ""))

    # TEST weeks (91+), untouched by training and by any mask or calibration
    tst = np.flatnonzero(D["trip_split"] == 2)[: a.n_trips]
    ptr = D["line_ptr"]
    obs = np.array([int(ptr[t + 1]) - int(ptr[t]) for t in tst])
    obs = obs[(obs >= 1) & (obs <= a.nmax)]
    kk = np.arange(1, a.nmax + 1)
    p_obs = np.bincount(obs, minlength=a.nmax + 1)[1:].astype(float)
    p_obs /= p_obs.sum()
    log(f"{len(obs):,} held-out baskets, mean {obs.mean():.2f}  var {obs.var():.1f}")

    # ---- analytic size law, at two draw counts so convergence is visible ---------------
    log("")
    log(f"{'source':>22}{'KL':>9}{'TV':>8}{'KS':>8}{'mean':>8}{'var':>8}")
    log(f"{'observed':>22}{0.0:9.3f}{0.0:8.3f}{0.0:8.3f}{obs.mean():8.2f}{obs.var():8.1f}")

    laws = {}
    for nd in (a.draws, a.draws * 8):
        acc = np.zeros(a.nmax)
        n_b = 0
        for k in range(0, len(tst), a.chunk):
            ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(tst[k:k + a.chunk])
            m.house, m.ctx = hh, ctx
            with torch.no_grad():
                pn = m.size_dist(ix, n_draws=nd,
                                 generator=torch.Generator().manual_seed(0))
                if isinstance(pn, tuple):
                    pn = pn[0]
            pn = pn.numpy()
            acc[: pn.shape[1]] += pn.sum(0)
            n_b += pn.shape[0]
        law = acc / max(acc.sum(), 1e-300)
        laws[nd] = law
        kl, tv, ks, e, v = compare(law, p_obs, kk)
        log(f"{'analytic @' + str(nd) + ' draws':>22}{kl:9.3f}{tv:8.3f}{ks:8.3f}{e:8.2f}{v:8.1f}")

    # ---- the same law reached by GENERATION, which is the independent witness ----------
    sizes = []
    for rep in range(a.reps):
        for k in range(0, min(len(tst), a.n_gen), a.chunk):
            ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(tst[k:k + a.chunk])
            m.house, m.ctx = hh, ctx
            g = torch.Generator().manual_seed(rep)
            with torch.no_grad():
                for b in m.sample(ix, n_draws=a.draws, generator=g):
                    sizes.append(len(b))
    sizes = np.array([s for s in sizes if 1 <= s <= a.nmax])
    if len(sizes):
        p_gen = np.bincount(sizes, minlength=a.nmax + 1)[1:].astype(float)
        p_gen /= p_gen.sum()
        kl, tv, ks, e, v = compare(p_gen, p_obs, kk)
        log(f"{'sampled':>22}{kl:9.3f}{tv:8.3f}{ks:8.3f}{e:8.2f}{v:8.1f}")
        k2, t2, s2, _, _ = compare(p_gen, laws[a.draws], kk)
        log("")
        log(f"sampler vs analytic AS LAWS: KL {k2:.3f}  TV {t2:.3f}  KS {s2:.3f}   "
            f"(the goal line compares only their means)")

    log("")
    log("reference points on this data: a negative binomial matched to both moments scores "
        "KL 0.035 / TV 0.113;\na two-point law matched to both moments scores KL 554 / "
        "TV 0.807 and passes both existing goals.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="../../out/v3_run60_best.pt")
    p.add_argument("--n-trips", type=int, default=384)
    p.add_argument("--n-gen", type=int, default=192)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--chunk", type=int, default=24)
    p.add_argument("--draws", type=int, default=32)
    p.add_argument("--nmax", type=int, default=120)
    p.add_argument("--R", type=int, default=23)
    main(p.parse_args())
