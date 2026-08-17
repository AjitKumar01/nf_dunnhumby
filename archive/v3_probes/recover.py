"""
Synthetic recovery: fit the version-3 model to data it generated, and see what comes back.

This is the experiment that has to pass before a dunnhumby fit is worth its 2-8 hours.  Data
is drawn EXACTLY from the model by the three-level sampler of Eq. 18b, so any failure to
recover belongs to the estimator rather than to misspecification.

WHAT IS SCORED, and why each is scored the way it is:

  phi   only its Gram matrix phi phi' is identified, since the model is invariant to
        rotating phi.  Reported as the correlation of the off-diagonal entries (is the
        co-purchase STRUCTURE right) and the relative Frobenius error (is the SCALE right).
  rho_c directly identified -- it is a coefficient on an observable count.  Correlation and
        root mean squared error against truth.
  rho_0 directly identified once rho_0(0) = 0 pins the scale.  Reported as the implied
        basket-size distribution, since that is what it controls and what section 14 says
        must be free.
  lam   identified up to nothing; reported as correlation.

THREE FITS, because the comparison is the point:

  full        the model as specified
  no-size     rho_0 held at zero -- the purely-Gaussian-latent model section 14.4 rejects
  no-phi      phi held at zero -- no co-purchase structure, categories independent given z

If `full` does not beat `no-size` on held-out likelihood and on the basket-size
distribution, the size potential is not earning its place and section 14's argument is
wrong about this data.

Writes out/v3_recovery.json.
"""
import argparse
import json
import os
import time

import math

import numpy as np
import torch

from core import Model, sample

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[rec] {m}", flush=True)


def size_coeffs_at_zero(T, house):
    """log A_n at z = 0, the combinatorial part of the size law before rho_0 tilts it.

    Needed to CALIBRATE rho_0 rather than guess it: since P(n|z) is proportional to
    exp(-rho_0(n)) A_n(z), setting rho_0(n) = log A_n(0) - log target(n) makes the size law
    equal `target` at z = 0, up to the z-mixture.  Three earlier attempts to reach a
    dunnhumby-like dispersion by guessing a quadratic rho_0 topped out near 0.93, because a
    quadratic cannot make the size law bimodal and conditioning on a non-empty basket
    removes the low mode.  Solving for rho_0 instead is the fix.
    """
    from core import esp_dense, poly_mul_trunc
    with torch.no_grad():
        bt = T.b_tilde(house)
        M = bt.amax(dim=-1, keepdim=True)
        w = torch.exp(bt - M).view(-1, T.C, T.P)
        e = esp_dense(w, T.R)
        r = torch.arange(T.R + 1, dtype=w.dtype)
        G = torch.exp(-T.rho_c.view(1, T.C, 1) * r * (r - 1) / 2.0) * e
        A = G[:, 0, :]
        for c in range(1, T.C):
            A = poly_mul_trunc(A, G[:, c, :], T.nmax)
        n = torch.arange(A.shape[-1], dtype=w.dtype)
        return (torch.log(A.clamp_min(1e-300)) + n * M).mean(0)


def bimodal_target(nmax, w_lo=0.55, lo=2.0, hi=14.0):
    """A deliberately two-humped size law: a small top-up trip or a large stock-up one.
    This is the shape section 14 says a Gaussian latent cannot reach without going
    critical, so it is the shape the size potential has to be tested against."""
    n = torch.arange(nmax + 1, dtype=torch.float64)
    lg = torch.lgamma(n + 1)
    p = (w_lo * torch.exp(-lo + n * math.log(lo) - lg)
         + (1 - w_lo) * torch.exp(-hi + n * math.log(hi) - lg))
    return p / p.sum()


def make_truth(a, g):
    T = Model(J=a.C * a.P, N=a.N, C=a.C, P=a.P, K=a.K, Kz=a.Kz, nmax=a.nmax,
              R=a.R, seed=a.seed)
    with torch.no_grad():
        T.lam.normal_(a.b_loc, 0.7, generator=g)
        T.alpha.normal_(0.0, 0.6, generator=g)
        T.theta.normal_(0.0, 0.6, generator=g)
        T.phi.normal_(0.0, a.phi_scale, generator=g)
        T.rho_c.normal_(0.4, 0.5, generator=g)          # mostly substitution, both signs
        if a.rho0 == "quadratic":
            n = torch.arange(1, a.nmax + 1, dtype=torch.float64)
            T.rho_0_free.copy_(-0.55 * torch.log1p(n) + 0.035 * n * (n - 1) / 2.0)
        elif a.rho0 == "bimodal":
            h = torch.arange(min(a.N, 64))
            logA = size_coeffs_at_zero(T, h)
            tgt = torch.log(bimodal_target(a.nmax).clamp_min(1e-12))
            r0 = logA[: a.nmax + 1] - tgt
            r0 = r0 - r0[0]                       # rho_0(0) = 0 fixes the scale
            T.rho_0_free.copy_(r0[1:])
        elif a.rho0 == "zero":
            T.rho_0_free.zero_()
    return T


def generate(T, a, g):
    houses, S = [], []
    t0 = time.time()
    for k in range(0, a.n_baskets, a.gen_batch):
        m = min(a.gen_batch, a.n_baskets - k)
        h = torch.randint(0, T.N, (m,), generator=g)
        s, _ = sample(T, h, n_draws=a.gen_draws, generator=g)
        houses.append(h)
        S.append(s)
        if (k + m) % 500 == 0:
            log(f"   generated {k + m}/{a.n_baskets}  ({time.time() - t0:.0f}s)")
    h = torch.cat(houses)
    s = torch.cat(S)
    keep = s.sum(-1) > 0                     # the model conditions on a non-empty basket
    return h[keep], s[keep]


def gram_scores(A, B):
    G1, G2 = A @ A.T, B @ B.T
    off = ~np.eye(len(G1), dtype=bool)
    if G2[off].std() < 1e-12:
        return float("nan"), float(np.linalg.norm(G1 - G2) / max(np.linalg.norm(G2), 1e-12))
    return (float(np.corrcoef(G1[off], G2[off])[0, 1]),
            float(np.linalg.norm(G1 - G2) / np.linalg.norm(G2)))


def size_hist(S, nmax):
    n = S.sum(-1).long().clamp(max=nmax).numpy()
    return np.bincount(n, minlength=nmax + 1) / len(n)


def fit(name, a, T, h_tr, S_tr, h_te, S_te, g):
    m = Model(J=T.J, N=T.N, C=T.C, P=T.P, K=a.K, Kz=a.Kz, nmax=a.nmax, R=a.R, seed=1)
    if name == "no-size":
        m.rho_0_free.requires_grad_(False)
    if name == "no-phi":
        m.phi.requires_grad_(False)
        with torch.no_grad():
            m.phi.zero_()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for it in range(1, a.iters + 1):
        idx = rng.choice(len(h_tr), size=min(a.batch, len(h_tr)), replace=False)
        ll, ess = m.loglik(h_tr[idx], S_tr[idx], n_draws=a.draws, generator=g,
                           return_ess=True)
        loss = -ll.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 5.0)
        opt.step()
        if it % max(1, a.iters // 5) == 0:
            with torch.no_grad():
                te = m.loglik(h_te, S_te, n_draws=a.draws * 2, generator=g).mean()
            log(f"   {name:8s} it {it:4d}  train {float(loss):8.3f}  "
                f"held-out {float(te):8.3f}  ESS {float(ess.mean()):.3f}")
    with torch.no_grad():
        te = float(m.loglik(h_te, S_te, n_draws=a.draws * 4, generator=g).mean())
    return m, te, time.time() - t0


def main(a):
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(a.seed)
    T = make_truth(a, g)
    log(f"truth: {T.J} products in {T.C} categories, {T.N} households, "
        f"K={a.K}, Kz={a.Kz}, nmax={a.nmax}, R={a.R}")
    h, S = generate(T, a, g)
    n_tr = int(0.8 * len(h))
    h_tr, S_tr, h_te, S_te = h[:n_tr], S[:n_tr], h[n_tr:], S[n_tr:]
    sizes = S.sum(-1)
    log(f"generated {len(h):,} non-empty baskets; size mean {float(sizes.mean()):.3f}  "
        f"var {float(sizes.var()):.3f}  dispersion {float(sizes.var()/sizes.mean()):.3f}")
    log(f"train {len(h_tr):,}  held out {len(h_te):,}")

    res = {"config": vars(a),
           "truth_size": dict(mean=float(sizes.mean()), var=float(sizes.var()),
                              dispersion=float(sizes.var() / sizes.mean()))}
    true_hist = size_hist(S, a.nmax)
    for name in ("full", "no-size", "no-phi"):
        m, te, secs = fit(name, a, T, h_tr, S_tr, h_te, S_te, g)
        with torch.no_grad():
            hh = torch.randint(0, T.N, (a.n_gen_eval,), generator=g)
            Sg, _ = sample(m, hh, n_draws=a.gen_draws, generator=g)
        gh = size_hist(Sg, a.nmax)
        tvd = float(0.5 * np.abs(gh - true_hist).sum())
        pc, pf = gram_scores(m.phi.detach().numpy(), T.phi.detach().numpy())
        rc_t, rc_h = T.rho_c.detach().numpy(), m.rho_c.detach().numpy()
        r0_t, r0_h = T.rho_0().detach().numpy(), m.rho_0().detach().numpy()
        lam_c = float(np.corrcoef(m.lam.detach().numpy(), T.lam.detach().numpy())[0, 1])
        res[name] = dict(
            held_out=te, secs=secs, size_tvd=tvd,
            phi_gram_corr=pc, phi_gram_rel_err=pf,
            rho_c_corr=float(np.corrcoef(rc_h, rc_t)[0, 1]),
            rho_c_rmse=float(np.sqrt(((rc_h - rc_t) ** 2).mean())),
            rho_0_corr=float(np.corrcoef(r0_h[1:], r0_t[1:])[0, 1]),
            lam_corr=lam_c)
        log(f"  {name:8s} held-out {te:8.4f}  size TVD {tvd:.4f}  "
            f"phi gram corr {pc:+.4f} rel err {pf:.4f}  "
            f"rho_c corr {res[name]['rho_c_corr']:+.4f}  "
            f"rho_0 corr {res[name]['rho_0_corr']:+.4f}  lam corr {lam_c:+.4f}")
        log("")

    log(f"  {'fit':10s} {'held-out':>10s} {'size TVD':>9s} {'phi corr':>9s} "
        f"{'rho_c corr':>11s} {'rho_0 corr':>11s}")
    for name in ("full", "no-size", "no-phi"):
        r = res[name]
        log(f"  {name:10s} {r['held_out']:10.4f} {r['size_tvd']:9.4f} "
            f"{r['phi_gram_corr']:+9.4f} {r['rho_c_corr']:+11.4f} "
            f"{r['rho_0_corr']:+11.4f}")
    d1 = res["full"]["held_out"] - res["no-size"]["held_out"]
    d2 = res["full"]["held_out"] - res["no-phi"]["held_out"]
    log("")
    log(f"  size potential is worth {d1:+.4f} nats per basket held out")
    log(f"  co-purchase phi is worth {d2:+.4f} nats per basket held out")
    with open(os.path.join(OUT, "v3_recovery.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    log("wrote out/v3_recovery.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=int, default=8)
    p.add_argument("--P", type=int, default=6)
    p.add_argument("--N", type=int, default=40)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--Kz", type=int, default=2)
    p.add_argument("--nmax", type=int, default=15)
    p.add_argument("--R", type=int, default=3)
    p.add_argument("--phi-scale", type=float, default=0.30)
    p.add_argument("--b-loc", type=float, default=-1.3)
    p.add_argument("--n-baskets", type=int, default=2500)
    p.add_argument("--gen-batch", type=int, default=250)
    p.add_argument("--gen-draws", type=int, default=64)
    p.add_argument("--n-gen-eval", type=int, default=1500)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--draws", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--rho0", choices=("bimodal", "quadratic", "zero"), default="bimodal")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
