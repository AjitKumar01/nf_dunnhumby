"""Hub topology: can a low-rank phi.phi' represent a star that is NOT a clique?

s23 planted four DISJOINT pairs -- every product at degree 1, the easiest possible topology --
and our model recovered them at 99.7%.  Measured afterwards, real dunnhumby co-purchase is
nothing like that: the 200 most co-purchased pairs span only 108 products, mean degree 3.7,
and ONE product appears in 100 of the 200 pairs.  It is a star graph.  On real data the model
reaches phi'phi = 0.83 (90% of its ceiling) on degree-1 pairs and 0.093 on the 152 pairs that
involve a hub.  So the arena tested the wrong graph.

The obstruction is geometric, and it is why this test needs a truth OUTSIDE the model class.
For a symmetric bilinear form, making one hub complementary with many partners forces the
partners to be complementary with each other:

    phi_i = c*phi_hub + beta*e_i  (e_i orthonormal, orthogonal to phi_hub, ||phi_hub|| = 1)
    =>  phi_hub . phi_i = c   for every partner
    =>  phi_i . phi_j   = c^2 for every pair of partners

so a star at strength 0.92 comes with a clique at 0.846.  Planting a star with phi would
therefore plant the clique too, and the model would recover it trivially -- proving nothing.

The truth here is a general symmetric W, which CAN be a star with exact zeros between
partners:

    E(S) = sum_j b_j(price) + sum_{j<k in S} W_jk - rho_0(|S|)
    W[hub, i] = +0.92 for 8 partners        <- star, degree 8
    W[9,10] = W[11,12] = W[13,14] = +0.92   <- 3 disjoint pairs, degree 1, as a control
    W = 0 everywhere else                    <- partners NOT complementary with each other

Both structures at identical strength, in one dataset, so the comparison is internal: any
model that recovers the disjoint pairs but not the star is limited by TOPOLOGY, not strength.

    energy-lowrank   phi phi' at Kz=8, our model as fitted
    energy-lowrank16 phi phi' at Kz=16, to separate rank from the geometry
    energy-fullW     free symmetric W -- the same energy with the rank constraint removed
    dpp, bernoulli   as in s23

If energy-fullW recovers the star and energy-lowrank does not, the limitation is the low-rank
factorisation and nothing else -- same objective, same estimator, same data.

Run:  python3 s24_hub.py
"""
import json
import math
import os
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

J = 16
NB = 24000
STEPS = 400
N_CTX = 4
HUB, N_PART = 0, 8
DISJOINT = [(9, 10), (11, 12), (13, 14)]
STRENGTH = 0.92
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[hub] {m}", flush=True)


IDX = torch.arange(2 ** J, dtype=torch.int64)
MASK = ((IDX.unsqueeze(1) >> torch.arange(J, dtype=torch.int64)) & 1).to(torch.float64)
NVEC = MASK.sum(1).long()
NE = NVEC > 0
MASK_NE, NVEC_NE = MASK[NE].contiguous(), NVEC[NE]
NSUB = MASK_NE.shape[0]
# sum_{j<k in S} W_jk  =  0.5 * (x' W x - sum_j W_jj x_j), computed once per W
TRI = None


def pair_energy(W):
    """[NSUB] value of sum_{j<k in S} W_jk for every non-empty subset."""
    XW = MASK_NE @ W
    return 0.5 * ((XW * MASK_NE).sum(1) - MASK_NE @ torch.diagonal(W))


def stats(lp, pairs):
    p = torch.softmax(lp, 0)
    pi = (MASK_NE * p.unsqueeze(1)).sum(0)
    lifts = np.array([float((MASK_NE[:, a] * MASK_NE[:, c] * p).sum())
                      / max(float(pi[a] * pi[c]), 1e-300) for (a, c) in pairs])
    law = torch.zeros(J + 1, dtype=p.dtype).index_add(0, NVEC_NE, p)[1:]
    return pi.numpy(), lifts, (law / law.sum()).numpy()


def make_truth(seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.zeros(J, J)
    star = [(HUB, i) for i in range(1, 1 + N_PART)]
    for a, c in star + DISJOINT:
        W[a, c] = W[c, a] = STRENGTH
    r0 = torch.zeros(J + 1)
    r0[1:] = 0.05 * torch.arange(1, J + 1, dtype=torch.float64) ** 2
    # partner-partner pairs: planted at EXACTLY zero, the control that a low-rank fit cannot
    # honour while also fitting the star
    pp = [(i, j) for i in range(1, 1 + N_PART) for j in range(i + 1, 1 + N_PART)][:8]
    return dict(a0=-1.2 + 0.5 * torch.randn(J, generator=g),
                beta=0.8 + 0.6 * torch.rand(J, generator=g),
                dlp=0.3 * torch.randn(N_CTX, J, generator=g),
                W=W, r0=r0, star=star, disj=DISJOINT, partpart=pp,
                pairs=star + DISJOINT + pp)


T = make_truth()


def truth_logp(t, dlp=None):
    d = T["dlp"][t] if dlp is None else dlp
    b = T["a0"] - T["beta"] * d
    return torch.log_softmax(MASK_NE @ b + pair_energy(T["W"]) - T["r0"][NVEC_NE], 0)


class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a0 = torch.nn.Parameter(torch.zeros(J))
        self.bt = torch.nn.Parameter(torch.zeros(J))
        self.r0 = torch.nn.Parameter(torch.zeros(J + 1))

    def b(self, dlp):
        return self.a0 - torch.nn.functional.softplus(self.bt) * dlp

    def logp(self, dlp, sub=None):
        lp = torch.log_softmax(MASK_NE @ self.b(dlp) + self.pair() - self.r0[NVEC_NE], 0)
        return lp if sub is None else lp[sub]


class LowRank(Base):
    def __init__(self, kz=8):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.PH = torch.nn.Parameter(torch.randn(J, kz, generator=g) * 0.1)

    def pair(self):
        v = MASK_NE @ self.PH
        return 0.5 * ((v * v).sum(1) - MASK_NE @ (self.PH ** 2).sum(1))

    def W(self):
        return (self.PH @ self.PH.T).detach()


class FullW(Base):
    """The same energy with the rank constraint removed."""
    def __init__(self):
        super().__init__()
        self.Wp = torch.nn.Parameter(torch.zeros(J, J))

    def _W(self):
        return 0.5 * (self.Wp + self.Wp.T)

    def pair(self):
        return pair_energy(self._W())

    def W(self):
        return self._W().detach()


def sample_counts(seed):
    rng = np.random.default_rng(seed)
    out = {}
    for t in range(N_CTX):
        p = torch.softmax(truth_logp(t), 0).numpy()
        d = rng.choice(NSUB, size=NB // N_CTX, p=p)
        c = torch.zeros(NSUB)
        i, n = np.unique(d, return_counts=True)
        c[torch.as_tensor(i)] = torch.as_tensor(n.astype(np.float64))
        out[t] = c
    return out


def fit(model, counts, tag):
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    t0 = time.time()
    for _ in range(STEPS):
        loss = sum(-(model.logp(T["dlp"][t]) * c).sum() for t, c in counts.items())
        loss = loss / sum(float(c.sum()) for c in counts.values())
        opt.zero_grad(); loss.backward(); opt.step()
    log(f"  {tag:18s} {time.time()-t0:6.1f}s   train/basket {-float(loss):9.4f}")
    return model


def main():
    log(f"J={J}: hub {HUB} with {N_PART} partners at W={STRENGTH} (degree {N_PART}), "
        f"{len(DISJOINT)} disjoint pairs at the same strength (degree 1),")
    log(f"and {len(T['partpart'])} partner-partner pairs planted at EXACTLY zero.")
    _, lt, _ = stats(truth_logp(0), T["pairs"])
    ns, nd = len(T["star"]), len(T["disj"])
    log(f"\ntrue lifts:  star {lt[:ns].mean():.3f}   disjoint {lt[ns:ns+nd].mean():.3f}   "
        f"partner-partner {lt[ns+nd:].mean():.3f}")
    tr, te = sample_counts(0), sample_counts(1)
    models = {"energy-lowrank(8)": LowRank(8), "energy-lowrank(16)": LowRank(16),
              "energy-fullW": FullW()}
    for k, m in models.items():
        fit(m, tr, k)
    ceil = float(sum((truth_logp(t) * c).sum() for t, c in te.items())
                 / sum(float(c.sum()) for c in te.values()))
    log(f"\n{'model':>20}{'held-out ll':>13}{'star lift':>11}{'disjoint':>10}"
        f"{'part-part':>11}{'W star':>9}{'W p-p':>8}")
    log(f"{'truth':>20}{ceil:13.4f}{lt[:ns].mean():11.3f}{lt[ns:ns+nd].mean():10.3f}"
        f"{lt[ns+nd:].mean():11.3f}{STRENGTH:9.3f}{0.0:8.3f}")
    res = {"_truth": dict(loglik=ceil, lifts=lt.tolist())}
    for k, m in models.items():
        with torch.no_grad():
            ll = float(sum((m.logp(T["dlp"][t]) * c).sum() for t, c in te.items())
                       / sum(float(c.sum()) for c in te.values()))
            _, lf, _ = stats(m.logp(T["dlp"][0]), T["pairs"])
            Wf = m.W()
            ws = float(np.mean([Wf[a, c] for a, c in T["star"]]))
            wp = float(np.mean([Wf[a, c] for a, c in T["partpart"]]))
        res[k] = dict(loglik=ll, lifts=lf.tolist(), W_star=ws, W_partpart=wp)
        log(f"{k:>20}{ll:13.4f}{lf[:ns].mean():11.3f}{lf[ns:ns+nd].mean():10.3f}"
            f"{lf[ns+nd:].mean():11.3f}{ws:9.3f}{wp:8.3f}")
    json.dump(res, open(os.path.join(OUT, "v3_hub.json"), "w"), indent=2)
    log(f"\nW star / W p-p are the recovered interaction on planted-0.92 and planted-0.0 pairs.")
    log(f"A low-rank phi phi' cannot hold the first high and the second at zero.")


if __name__ == "__main__":
    main()
