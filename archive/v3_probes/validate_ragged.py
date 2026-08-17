"""
Validate the ragged kernel against the dense one, and then against brute force.

The dense implementation in core.py is already checked against explicit enumeration of all
2^J subsets (validate.py, six checks).  So the cheapest way to trust the ragged kernel is to
run both on the SAME instance with the SAME parameters and the SAME Gaussian draws and
require them to agree.  They compute the same quantity by different routes -- the dense one
by the O(N R) elementary-symmetric recursion, the ragged one by power sums and Newton's
identities -- so agreement is evidence about both, and any disagreement localises to the
step where they differ.

Three settings are run, because the ragged kernel's weak point is known in advance:
Newton's identities subtract quantities of similar size, so a row whose weight is dominated
by one product loses precision.  `--spread` widens the spread of product values to provoke
exactly that, and the script reports the measured cancellation alongside the error rather
than assuming the regime is safe.

Also checked: that a genuinely RAGGED instance -- categories of different sizes, which the
dense kernel cannot express -- reproduces brute-force enumeration directly.

Run:  python3 validate_ragged.py
"""
import argparse
import itertools
import math

import numpy as np
import torch

from core import Model
from ragged import RaggedIndex, RaggedModel, cancellation, log_f_ragged, seg_max


def log(m):
    print(f"[rag] {m}", flush=True)


def dense_to_ragged(m, B, C, P):
    """Index for B trips that all see the same C categories of P products."""
    item = np.tile(np.arange(C * P), B)
    row_of = np.repeat(np.arange(B * C), P)
    row_trip = np.repeat(np.arange(B), C)
    row_cat = np.tile(np.arange(C), B)
    return RaggedIndex(item, row_of, row_trip, row_cat, B)


def copy_params(src, dst):
    with torch.no_grad():
        for n in ("lam", "alpha", "theta", "phi", "rho_c", "rho_0_free"):
            getattr(dst, n).copy_(getattr(src, n))


def main(a):
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(a.seed)

    for spread in (0.6, 1.5, a.spread):
        C, P, B = a.C, a.P, a.B
        d = Model(J=C * P, N=8, C=C, P=P, K=4, Kz=a.Kz, nmax=a.nmax, R=a.R, seed=a.seed)
        with torch.no_grad():
            d.lam.normal_(-1.0, spread, generator=g)
            d.phi.normal_(0.0, 0.30, generator=g)
            d.rho_c.normal_(0.3, 0.5, generator=g)
            d.rho_0_free.normal_(0.0, 0.3, generator=g)
        r = RaggedModel(J=C * P, N=8, C=C, K=4, Kz=a.Kz, nmax=a.nmax, R=a.R, seed=a.seed)
        copy_params(d, r)
        house = torch.randint(0, 8, (B,), generator=g)
        r.house = house
        ix = dense_to_ragged(r, B, C, P)

        z = torch.randn(B, a.D, a.Kz, generator=g)
        lf_d = d.log_f(z, d.b_tilde(house))
        lf_r = log_f_ragged(r, z, ix)
        err = float((lf_d - lf_r).abs().max())

        # how close did Newton's identities come to cancelling?
        phi_i = r.phi[ix.item]
        bt = r.b_flat(ix) - 0.5 * (phi_i ** 2).sum(-1)
        proj = (z[ix.item_trip] * phi_i.unsqueeze(1)).sum(-1)
        logw = (bt.unsqueeze(1) + proj).transpose(0, 1)
        M = seg_max(logw, ix.item_trip, ix.B)
        w = torch.exp(logw - M.index_select(-1, ix.item_trip))
        canc = cancellation(w, ix.row_of, ix.n_rows)
        log(f"spread {spread:.2f}: log f max abs err {err:.3e}   "
            f"cancellation min {float(canc.min()):.2e} median "
            f"{float(canc.median()):.2e}   level {float(lf_d.abs().mean()):.2f}")

    # ---- energy, log Z and the likelihood, on the same draws -------------------------
    C, P, B = a.C, a.P, a.B
    d = Model(J=C * P, N=8, C=C, P=P, K=4, Kz=a.Kz, nmax=a.nmax, R=a.R, seed=a.seed)
    with torch.no_grad():
        d.lam.normal_(-1.0, 0.7, generator=g)
        d.phi.normal_(0.0, 0.30, generator=g)
        d.rho_c.normal_(0.3, 0.5, generator=g)
        d.rho_0_free.normal_(0.0, 0.3, generator=g)
    r = RaggedModel(J=C * P, N=8, C=C, K=4, Kz=a.Kz, nmax=a.nmax, R=a.R, seed=a.seed)
    copy_params(d, r)
    house = torch.randint(0, 8, (B,), generator=g)
    r.house = house
    ix = dense_to_ragged(r, B, C, P)

    S = (torch.rand(B, C * P, generator=g) < 0.12).to(torch.float64)
    S[:, 0] = 1.0                                    # never empty
    li, lt = S.nonzero(as_tuple=True)[1], S.nonzero(as_tuple=True)[0]
    lc = li // P
    e_d = d.energy(house, S)
    e_r = r.energy(li, lt, lc, B)
    log("")
    log(f"energy: max abs err {float((e_d - e_r).abs().max()):.3e}")

    gz1 = torch.Generator().manual_seed(99)
    gz2 = torch.Generator().manual_seed(99)
    lz_d = d.log_Z(house, n_draws=a.D, generator=gz1)
    # same proposal on both sides, so this isolates the KERNEL rather than the
    # proposal: the ragged model defaults to an identity covariance now
    lz_r = r.log_Z(ix, n_draws=a.D, generator=gz2, mode_steps=10, laplace=True)
    log(f"log Z:  max abs err {float((lz_d - lz_r).abs().max()):.3e}   "
        f"level {float(lz_d.mean()):.4f}")

    # ---- a genuinely ragged instance, against brute force ----------------------------
    sizes = [1, 2, 5, 3]
    J = sum(sizes)
    cats = []
    o = 0
    for k, s in enumerate(sizes):
        cats.append(np.arange(o, o + s))
        o += s
    rm = RaggedModel(J=J, N=2, C=len(sizes), K=3, Kz=2, nmax=J, R=min(4, max(sizes)),
                     seed=1)
    with torch.no_grad():
        rm.lam.normal_(-0.7, 0.8, generator=g)
        rm.phi.normal_(0.0, 0.35, generator=g)
        rm.rho_c.normal_(0.0, 0.6, generator=g)
        rm.rho_0_free.normal_(0.0, 0.4, generator=g)
    rm.house = torch.zeros(1, dtype=torch.long)
    item = np.concatenate(cats)
    row_of = np.concatenate([np.full(len(c), k) for k, c in enumerate(cats)])
    ixr = RaggedIndex(item, row_of, np.arange(len(sizes)) * 0, np.arange(len(sizes)), 1)

    lam = rm.lam.detach()
    phi = rm.phi.detach()
    th = rm.theta.detach()[0]
    al = rm.alpha.detach()
    rc = rm.rho_c.detach()
    r0 = rm.rho_0().detach()
    tot = []
    for bits in itertools.product([0, 1], repeat=J):
        idx = [j for j, v in enumerate(bits) if v]
        E = sum(float(lam[j] + th @ al[j]) for j in idx)
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                E += float(phi[idx[x]] @ phi[idx[y]])
        for k, c in enumerate(cats):
            nc = sum(1 for j in idx if j in set(c.tolist()))
            E -= float(rc[k]) * nc * (nc - 1) / 2.0
        E -= float(r0[len(idx)])
        tot.append(E)
    true_lz = float(torch.logsumexp(torch.tensor(tot), 0))
    gz = torch.Generator().manual_seed(5)
    lz = float(rm.log_Z(ixr, n_draws=a.big_draws, generator=gz,
                        mode_steps=10, laplace=True)[0])
    log("")
    log(f"ragged instance, category sizes {sizes} (dense cannot express this):")
    log(f"  brute force over {2**J:,} subsets  log Z {true_lz:+.8f}")
    log(f"  ragged kernel                      log Z {lz:+.8f}   err {lz - true_lz:+.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=int, default=6)
    p.add_argument("--P", type=int, default=5)
    p.add_argument("--B", type=int, default=12)
    p.add_argument("--Kz", type=int, default=2)
    p.add_argument("--nmax", type=int, default=18)
    p.add_argument("--R", type=int, default=4)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--big-draws", type=int, default=16384)
    p.add_argument("--spread", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
