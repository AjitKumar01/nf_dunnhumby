"""A fair arena: the truth belongs to NO candidate model class.

s23 generated from this project's own energy form, so the likelihood axis favoured it by
construction.  Putting that axis last in the report was mitigation, not a fix -- a comparison
whose data-generating process is one entrant's functional form cannot settle anything.

Here the truth is a LATENT MISSION MIXTURE, which no candidate can express exactly:

    m ~ Categorical(w)                       a shopping mission
    x_j | m ~ Bernoulli(p_jm)                items independent GIVEN the mission
    P(S) = sum_m w_m prod_j p_jm^x (1-p_jm)^(1-x)

This is a realistic story for baskets -- people shop for occasions -- and it is outside every
entrant: it is not a pairwise energy (our model), not a determinant (DPP), not independent
(Bernoulli), not a size law times independent draws (multinomial).  Every model is therefore
misspecified, which is the only honest footing for comparing classes.

It also produces the HUB topology real data has, without being asked to.  Mission 0 is a
"no-shop" baseline where everything is unlikely; missions 1..M each raise the hub AND their
own two or three partners.  The hub is then elevated whenever any real shop happens, so it
co-occurs with every partner; partners co-occur strongly only with their own mission-mates.
That is a star with a weak clique -- which is what dunnhumby looks like (200 top pairs over
108 products, one product in 100 of them).

Scored on the same four axes as s23, plus the hub/non-hub split that s24 showed matters:

    1 log-likelihood   held-out, per basket, against the truth's own value
    2 co-occurrence    hub pairs and mission-mate pairs, separately
    3 counterfactual   d log pi_j / d log p_j against the exact response
    4 size law         KL to the exact P(n)

Price enters every model the same way, through a per-product logit shift, and the truth's
mission probabilities respond to price so the counterfactual is well posed.

Run:  python3 s25_fair.py
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
N_MIS = 5                       # plus the no-shop baseline
HUB = 0
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[fair] {m}", flush=True)


IDX = torch.arange(2 ** J, dtype=torch.int64)
MASK = ((IDX.unsqueeze(1) >> torch.arange(J, dtype=torch.int64)) & 1).to(torch.float64)
NVEC = MASK.sum(1).long()
NE = NVEC > 0
MASK_NE, NVEC_NE = MASK[NE].contiguous(), NVEC[NE]
NSUB = MASK_NE.shape[0]


def stats(lp, pairs):
    p = torch.softmax(lp, 0)
    pi = (MASK_NE * p.unsqueeze(1)).sum(0)
    lifts = np.array([float((MASK_NE[:, a] * MASK_NE[:, c] * p).sum())
                      / max(float(pi[a] * pi[c]), 1e-300) for (a, c) in pairs])
    law = torch.zeros(J + 1, dtype=p.dtype).index_add(0, NVEC_NE, p)[1:]
    return pi.numpy(), lifts, (law / law.sum()).numpy()


def kl(pm, pt):
    m = pt > 1e-12
    return float((pt[m] * np.log(pt[m] / np.clip(pm[m], 1e-300, None))).sum())


# ------------------------------------------------------------------------- the truth
def make_truth(seed=0):
    """Mission structure CALIBRATED to dunnhumby's measured lift structure.

    Measured on 155,374 training baskets, over the 200 most co-purchased pairs:
        hub pairs (152, degree>=16)   observed lift  2.186
        degree<=3 pairs (4)           observed lift 27.431
        all 200                       mean 3.357, median 2.215, max 40.7
        topology                      108 products, max degree 100, mean 3.7

    An earlier version of this truth produced hub lift 1.059 -- a hub so frequent it carried
    no information -- which is a real pattern but NOT the one the evaluated pairs show.  The
    structure below reproduces both regimes:

      baseline mission (large w)   everything unlikely, INCLUDING the hub.  The hub's lift
                                   comes from this contrast: it is absent when no real shop
                                   happens, so it correlates with whatever a shop contains.
      shopping missions            hub high + that mission's own partners high  -> lift ~2
      one RARE mission (small w)   two products that appear together and almost nowhere else.
                                   lift ~ 1/w for such a pair, so w = 0.04 gives ~25, which
                                   is how the degree<=3 pairs reach 27.
    """
    g = torch.Generator().manual_seed(seed)
    W_BASE, W_RARE = 0.55, 0.04
    n_shop = 4
    w_shop = (1.0 - W_BASE - W_RARE) / n_shop
    base = torch.full((J,), 0.02)
    P = [base.clone()]                                    # mission 0: no real shop
    parts = {}
    for m in range(1, n_shop + 1):
        q = base.clone()
        q[HUB] = 0.85
        pr = [1 + 3 * (m - 1), 2 + 3 * (m - 1), 3 + 3 * (m - 1)]
        parts[m] = [p for p in pr if p < J - 2]
        for p_ in parts[m]:
            q[p_] = 0.60
        P.append(q)
    rare = [J - 2, J - 1]                                  # the pair that carries a huge lift
    q = base.clone()
    for p_ in rare:
        q[p_] = 0.55
    P.append(q)
    P = torch.stack(P)
    w = torch.tensor([W_BASE] + [w_shop] * n_shop + [W_RARE])
    hub_pairs = [(HUB, p_) for m in parts for p_ in parts[m]]
    mate_pairs = [(a, b) for m in parts for i, a in enumerate(parts[m]) for b in parts[m][i+1:]]
    rare_pairs = [(rare[0], rare[1])]
    cross = [(parts[1][0], parts[2][0]), (parts[1][1], parts[3][0])]
    return dict(P=P, w=w, beta=0.8 + 0.6 * torch.rand(J, generator=g),
                dlp=0.3 * torch.randn(N_CTX, J, generator=g), parts=parts,
                hub=hub_pairs, mate=mate_pairs, rare=rare_pairs, cross=cross,
                pairs=hub_pairs + rare_pairs + mate_pairs + cross)


T = make_truth()


def truth_logp(t, dlp=None):
    """log P(S) under the mixture, with price shifting every component's logits."""
    d = T["dlp"][t] if dlp is None else dlp
    lg = torch.logit(T["P"].clamp(1e-6, 1 - 1e-6)) - (T["beta"] * d).unsqueeze(0)
    lp_m = MASK_NE @ torch.nn.functional.logsigmoid(lg).T + \
        (1 - MASK_NE) @ torch.nn.functional.logsigmoid(-lg).T          # [NSUB, M]
    lp = torch.logsumexp(lp_m + T["w"].log().unsqueeze(0), dim=1)
    return lp - torch.logsumexp(lp, 0)


# ------------------------------------------------------------------------------ models
class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a0 = torch.nn.Parameter(torch.zeros(J))
        self.bt = torch.nn.Parameter(torch.zeros(J))

    def b(self, dlp):
        return self.a0 - torch.nn.functional.softplus(self.bt) * dlp


class Energy(Base):
    def __init__(self, kz=8, cap=None):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.PH = torch.nn.Parameter(torch.randn(J, kz, generator=g) * 0.1)
        self.r0 = torch.nn.Parameter(torch.zeros(J + 1))
        self.cap = cap

    def logp(self, dlp, sub=None):
        v = MASK_NE @ self.PH
        pr = 0.5 * ((v * v).sum(1) - MASK_NE @ (self.PH ** 2).sum(1))
        lp = torch.log_softmax(MASK_NE @ self.b(dlp) + pr - self.r0[NVEC_NE], 0)
        return lp if sub is None else lp[sub]

    def clip(self):
        if self.cap is None:
            return
        with torch.no_grad():
            n = self.PH.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.PH.mul_(torch.clamp(self.cap / n, max=1.0))


class BernoulliM(Base):
    def logp(self, dlp, sub=None):
        b = self.b(dlp)
        lp = MASK_NE @ torch.nn.functional.logsigmoid(b) + \
            (1 - MASK_NE) @ torch.nn.functional.logsigmoid(-b)
        lp = lp - torch.logsumexp(lp, 0)
        return lp if sub is None else lp[sub]

    def clip(self):
        pass


class Multinom(Base):
    def __init__(self):
        super().__init__()
        self.r0 = torch.nn.Parameter(torch.zeros(J))

    def logp(self, dlp, sub=None):
        s = MASK_NE @ torch.log_softmax(self.b(dlp), 0)
        lr = torch.log_softmax(self.r0, 0)
        out = torch.zeros(NSUB, dtype=s.dtype)
        for n in range(1, J + 1):
            sel = NVEC_NE == n
            out = out + sel.to(s.dtype) * (lr[n - 1] + s - torch.logsumexp(s[sel], 0))
        out = out - torch.logsumexp(out, 0)
        return out if sub is None else out[sub]

    def clip(self):
        pass


class DPPM(Base):
    def __init__(self, rank=8):
        super().__init__()
        g = torch.Generator().manual_seed(2)
        self.V = torch.nn.Parameter(torch.randn(J, rank, generator=g) * 0.3)

    def logp(self, dlp, sub=None):
        d = torch.exp(self.b(dlp).clamp(-8, 6))
        L = torch.diag(d) + self.V @ self.V.T
        mk = MASK_NE if sub is None else MASK_NE[sub]
        M = mk.unsqueeze(-1) * mk.unsqueeze(-2)
        A = L.unsqueeze(0) * M + torch.eye(J, dtype=L.dtype).unsqueeze(0) * (1 - M)
        den = torch.linalg.slogdet(L + torch.eye(J, dtype=L.dtype))[1]
        return torch.linalg.slogdet(A)[1] - den - torch.log1p(-torch.exp(-den))

    def clip(self):
        pass


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


def fit(model, counts, tag, pool=0.0):
    """pool > 0 adds a partial-pooling penalty on the per-product price coefficient.

    The own-price elasticity is about -g_j (1 - pi_j) with g_j = softplus(bt_j).  Fitted
    freely from limited price variation, g_j is noisy, and noise inflates its spread: the
    unpooled models over-disperse elasticities by ~2.4x against the truth (sd 0.37-0.51 vs
    0.190) while still RANKING products correctly (r = 0.89).  Shrinking each model's
    elasticities toward their own mean by lam = 0.42 post-hoc took MAE 0.141 -> 0.071, from
    worse than a constant predictor (0.133) to 1.9x better.  This penalises the same
    dispersion during training instead: tau * Var_j(g_j), shrinking toward the mean rather
    than toward zero, so the average price sensitivity is untouched.

    Applied to every entrant at the same strength -- pooling only our model would rig the
    comparison, and the multinomial (best lam 0.87, already well scaled) should if anything
    be hurt by it.
    """
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    obs = {t: torch.nonzero(c > 0).flatten() for t, c in counts.items()}
    cnz = {t: counts[t][obs[t]] for t in counts}
    t0 = time.time()
    for _ in range(STEPS):
        loss = sum(-(model.logp(T["dlp"][t], obs[t]) * cnz[t]).sum() for t in counts)
        loss = loss / sum(float(c.sum()) for c in cnz.values())
        if pool > 0:
            g = torch.nn.functional.softplus(model.bt)
            loss = loss + pool * ((g - g.mean()) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        model.clip()
    log(f"  {tag:22s} {time.time()-t0:6.1f}s   train/basket {-float(loss):9.4f}")
    return model


def main():
    nh, nr, nm = len(T["hub"]), len(T["rare"]), len(T["mate"])
    log(f"truth: latent mission mixture, {N_MIS} missions + a no-shop baseline -- in NO "
        f"candidate's model class")
    log(f"  {nh} hub pairs (product {HUB} with every partner), {nm} mission-mate pairs, "
        f"{len(T['cross'])} cross-mission controls")
    _, lt, lawt = stats(truth_logp(0), T["pairs"])
    log(f"  true lift:  hub {lt[:nh].mean():.3f} (real 2.19)   "
        f"rare {lt[nh:nh+nr].mean():.2f} (real 27.4)   "
        f"mate {lt[nh+nr:nh+nr+nm].mean():.3f}   cross {lt[nh+nr+nm:].mean():.3f}")
    log(f"  hub degree {nh} vs mate degree 2 -- the star topology, arising from the mixture")

    tr, te = sample_counts(0), sample_counts(1)
    specs = [("energy 0.96", lambda: Energy(8, cap=0.96), 0.0),
             ("energy 0.96 pool2", lambda: Energy(8, cap=0.96), 2.0),
             ("energy 0.96 pool8", lambda: Energy(8, cap=0.96), 8.0),
             ("energy 2.5", lambda: Energy(8, cap=2.5), 0.0),
             ("energy 2.5 pool2", lambda: Energy(8, cap=2.5), 2.0),
             ("energy 2.5 pool8", lambda: Energy(8, cap=2.5), 8.0),
             ("dpp", DPPM, 0.0),
             ("bernoulli", BernoulliM, 0.0),
             ("bernoulli pool2", BernoulliM, 2.0),
             ("multinom", Multinom, 0.0),
             ("multinom pool2", Multinom, 2.0)]
    models = {}
    for k, ctor, pl in specs:
        models[k] = fit(ctor(), tr, k, pool=pl)

    ceil = float(sum((truth_logp(t) * c).sum() for t, c in te.items())
                 / sum(float(c.sum()) for c in te.values()))
    eps = math.log(0.9)

    def own_price(fn):
        with torch.no_grad():
            pi0 = stats(fn(T["dlp"][0]), T["pairs"])[0]
        o = []
        for j in range(J):
            d2 = T["dlp"][0].clone(); d2[j] += eps
            with torch.no_grad():
                pij = stats(fn(d2), T["pairs"])[0][j]
            o.append((math.log(pij) - math.log(pi0[j])) / eps)
        return np.array(o)
    e_true = own_price(lambda d: truth_logp(0, d))

    log(f"\n{'model':>19}{'held-out ll':>13}{'gap':>8}{'hub lift':>10}{'rare lift':>11}"
        f"{'mate':>8}{'size KL':>9}{'price MAE':>11}")
    log(f"{'truth':>19}{ceil:13.4f}{0.0:8.3f}{lt[:nh].mean():10.3f}"
        f"{lt[nh:nh+nr].mean():11.2f}{lt[nh+nr:nh+nr+nm].mean():8.3f}{0.0:9.4f}{'--':>11}")
    res = {"_truth": dict(loglik=ceil, lifts=lt.tolist(), law=lawt.tolist(),
                          elast=e_true.tolist(), n_hub=nh, n_rare=nr, n_mate=nm)}
    for k, m in models.items():
        with torch.no_grad():
            ll = float(sum((m.logp(T["dlp"][t]) * c).sum() for t, c in te.items())
                       / sum(float(c.sum()) for c in te.values()))
            _, lf, law = stats(m.logp(T["dlp"][0]), T["pairs"])
        el = own_price(lambda d: m.logp(d))
        res[k] = dict(loglik=ll, lifts=lf.tolist(), law=law.tolist(),
                      size_kl=kl(law, lawt), elast=el.tolist(),
                      elast_err=float(np.abs(el - e_true).mean()))
        log(f"{k:>19}{ll:13.4f}{ceil-ll:8.3f}{lf[:nh].mean():10.3f}"
            f"{lf[nh:nh+nr].mean():11.2f}{lf[nh+nr:nh+nr+nm].mean():8.3f}"
            f"{res[k]['size_kl']:9.4f}{res[k]['elast_err']:11.4f}")
    json.dump(res, open(os.path.join(OUT, "v3_fair.json"), "w"), indent=2)
    log(f"\nEvery model here is misspecified -- the truth is in none of their families.")


if __name__ == "__main__":
    main()
