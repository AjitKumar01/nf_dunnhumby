"""A simulated retail environment with known ground truth, scoring every model class on four axes.

Why synthetic.  On dunnhumby nothing can be checked against truth: observed pair lift confounds
preference with availability, the correct counterfactual response is unknown, and log Z must be
estimated so a score mixes model class with estimator quality.  Here J = 16, so all 2^16 subsets
enumerate and P(S), every marginal, every pair lift and the exact price response are closed-form.

The generating process IS this project's energy model, which favours it on likelihood by
construction.  That is stated rather than hidden, and it is why the other three axes carry the
weight: a DPP is not merely a worse fit here, it is structurally incapable of positive
correlation at any parameter setting, and the co-occurrence panel shows that directly.

    truth   E(S) = sum_j b_j(price) + sum_{j<k in S} phi_j.phi_k - rho_0(|S|)
            4 complementary pairs at phi'phi = +0.92  (lift 2.5, grocery's level)
            2 substitutable pairs at phi'phi = -0.92  (lift 0.4)
            b_j = a_j - beta_j*dlp_j with beta_j > 0, so price counterfactuals are well posed
            rho_0 quadratic, giving an overdispersed size law

    models  energy     pairwise interaction + free rho_0          (the model under test)
            dpp        P(S) ~ det(L_S), L = diag(d) + VV'         (log-submodular by design)
            bernoulli  independent items                          (no interaction at all)
            multinom   size law x independent draws               (size right, pairs by accident)

Each model uses its own EXACT path -- enumeration where the normaliser is a subset sum, closed
forms where the model provides them (a DPP's marginals come from K = L(L+I)^-1, its size law
from K's eigenvalues).  No model is approximated, so differences are model class alone.

SHOPPER IS EXCLUDED, deliberately.  Its set probability needs a sum over n! orderings; at J=16
that is 2e13 terms, so it could only be estimated while every other model here is exact, and a
comparison in which one entrant alone carries Monte Carlo bias measures the estimator, not the
model.  Shopper is compared on real data instead, where all models are estimated alike.

Run:  python3 s23_arena.py
"""
import json
import math
import os
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

J = 16
KZ = 8
NB = 24000
STEPS = 400
N_CTX = 4
RANK, SUB = 4, 2
TRUE_T = 0.92
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[arena] {m}", flush=True)


IDX = torch.arange(2 ** J, dtype=torch.int64)
MASK = ((IDX.unsqueeze(1) >> torch.arange(J, dtype=torch.int64)) & 1).to(torch.float64)
NVEC = MASK.sum(1).long()
NE = NVEC > 0
MASK_NE, NVEC_NE = MASK[NE].contiguous(), NVEC[NE]
NSUB = MASK_NE.shape[0]
SUBLIST = [torch.nonzero(MASK_NE[i]).flatten() for i in range(0, 0)]   # built lazily


def energy_terms(b, PH, r0):
    v = MASK_NE @ PH
    return MASK_NE @ b + 0.5 * ((v * v).sum(1) - MASK_NE @ (PH ** 2).sum(1)) - r0[NVEC_NE]


def stats_from_logp(lp, pairs):
    """pi, pair lifts and the size law, from a log-prob over every non-empty subset."""
    p = torch.softmax(lp, 0)
    pi = (MASK_NE * p.unsqueeze(1)).sum(0)
    lifts = np.array([float((MASK_NE[:, a] * MASK_NE[:, c] * p).sum())
                      / max(float(pi[a] * pi[c]), 1e-300) for (a, c) in pairs])
    law = torch.zeros(J + 1, dtype=p.dtype).index_add(0, NVEC_NE, p)[1:]
    return pi.numpy(), lifts, (law / law.sum()).numpy()


def kl(pm, pt):
    m = pt > 1e-12
    return float((pt[m] * np.log(pt[m] / np.clip(pm[m], 1e-300, None))).sum())


# ---------------------------------------------------------------------------------- truth
def make_truth(seed=0):
    g = torch.Generator().manual_seed(seed)
    v = math.sqrt(TRUE_T)
    comp = [(2 * i, 2 * i + 1) for i in range(RANK)]
    subs = [(8 + 2 * i, 9 + 2 * i) for i in range(SUB)]
    PH = torch.zeros(J, KZ)
    for i, (a, c) in enumerate(comp):
        PH[a, i] = v; PH[c, i] = v
    for i, (a, c) in enumerate(subs):
        PH[a, RANK + i] = v; PH[c, RANK + i] = -v
    r0 = torch.zeros(J + 1)
    r0[1:] = 0.05 * torch.arange(1, J + 1, dtype=torch.float64) ** 2
    return dict(a0=-1.2 + 0.5 * torch.randn(J, generator=g),
                beta=0.8 + 0.6 * torch.rand(J, generator=g),
                dlp=0.3 * torch.randn(N_CTX, J, generator=g),
                PH=PH, r0=r0, comp=comp, subs=subs, pairs=comp + subs)


T = make_truth()


def truth_logp(t, dlp=None):
    d = T["dlp"][t] if dlp is None else dlp
    return torch.log_softmax(energy_terms(T["a0"] - T["beta"] * d, T["PH"], T["r0"]), 0)


# --------------------------------------------------------------------------------- models
class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a0 = torch.nn.Parameter(torch.zeros(J))
        self.bt = torch.nn.Parameter(torch.zeros(J))

    def b(self, dlp):
        return self.a0 - torch.nn.functional.softplus(self.bt) * dlp


class Energy(Base):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.PH = torch.nn.Parameter(torch.randn(J, KZ, generator=g) * 0.1)
        self.r0 = torch.nn.Parameter(torch.zeros(J + 1))

    def logp(self, dlp, sub=None):
        lp = torch.log_softmax(energy_terms(self.b(dlp), self.PH, self.r0), 0)
        return lp if sub is None else lp[sub]


class BernoulliM(Base):
    def logp(self, dlp, sub=None):
        b = self.b(dlp)
        lp = MASK_NE @ torch.nn.functional.logsigmoid(b) + \
            (1 - MASK_NE) @ torch.nn.functional.logsigmoid(-b)
        lp = lp - torch.logsumexp(lp, 0)
        return lp if sub is None else lp[sub]


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


class DPPM(Base):
    """det(L_S) for every subset, via one batched slogdet on padded principal submatrices.

    At J = 16 that is [65535, 16, 16] = 134 MB, which fits; at J = 20 it would be 3.2 GB,
    which is why the arena is J = 16.
    """
    def __init__(self, rank=8):
        super().__init__()
        g = torch.Generator().manual_seed(2)
        self.V = torch.nn.Parameter(torch.randn(J, rank, generator=g) * 0.3)

    def logp(self, dlp, sub=None):
        # det(L_S) over EVERY subset costs 65,535 backward-differentiated determinants per
        # step -- ~90 minutes for one fit.  Training needs only the observed subsets, and the
        # normaliser is closed form: det(L+I) = det(diag(d)+VV'+I), a single J x J determinant.
        # Full enumeration is then needed once, at evaluation.
        d = torch.exp(self.b(dlp).clamp(-8, 6))
        L = torch.diag(d) + self.V @ self.V.T
        mk = MASK_NE if sub is None else MASK_NE[sub]
        M = mk.unsqueeze(-1) * mk.unsqueeze(-2)
        A = L.unsqueeze(0) * M + torch.eye(J, dtype=L.dtype).unsqueeze(0) * (1 - M)
        num = torch.linalg.slogdet(A)[1]
        den = torch.linalg.slogdet(L + torch.eye(J, dtype=L.dtype))[1]
        # condition on non-empty: subtract log(1 - P(empty)), P(empty) = 1/det(L+I)
        return num - den - torch.log1p(-torch.exp(-den))


# ------------------------------------------------------------------------------- fit/score
def sample_counts(seed):
    rng = np.random.default_rng(seed)
    out = {}
    per = NB // N_CTX
    for t in range(N_CTX):
        p = torch.softmax(truth_logp(t), 0).numpy()
        d = rng.choice(NSUB, size=per, p=p)
        c = torch.zeros(NSUB)
        i, n = np.unique(d, return_counts=True)
        c[torch.as_tensor(i)] = torch.as_tensor(n.astype(np.float64))
        out[t] = c
    return out


def fit(model, counts, tag):
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    t0 = time.time()
    obs = {t: torch.nonzero(c > 0).flatten() for t, c in counts.items()}
    cnz = {t: counts[t][obs[t]] for t in counts}
    log(f"  {tag:10s} training on {int(np.mean([len(o) for o in obs.values()])):,} distinct "
        f"subsets per context of {NSUB:,}")
    for _ in range(STEPS):
        loss = sum(-(model.logp(T["dlp"][t], obs[t]) * cnz[t]).sum() for t in counts)
        loss = loss / sum(float(c.sum()) for c in cnz.values())
        opt.zero_grad(); loss.backward(); opt.step()
    log(f"  {tag:10s} {time.time()-t0:6.1f}s   train/basket {-float(loss):9.4f}")
    return model


def main():
    log(f"J = {J}, {2**J:,} subsets enumerated, {NB:,} baskets over {N_CTX} price contexts")
    _, lt, lawt = stats_from_logp(truth_logp(0), T["pairs"])
    log(f"planted lifts: complements {lt[:RANK].mean():.3f}  substitutes {lt[RANK:].mean():.3f}")
    tr, te = sample_counts(0), sample_counts(1)
    models = dict(energy=Energy(), dpp=DPPM(), bernoulli=BernoulliM(), multinom=Multinom())
    for k, m in models.items():
        fit(m, tr, k)

    res = {}
    # truth's own held-out score is the ceiling any model can reach
    ceil = float(sum((truth_logp(t) * c).sum() for t, c in te.items())
                 / sum(float(c.sum()) for c in te.values()))
    res["_truth"] = dict(loglik=ceil, lifts=lt.tolist(), law=lawt.tolist(), size_kl=0.0)
    log(f"\n{'model':>10}{'held-out ll':>13}{'gap':>8}{'comp lift':>11}{'sub lift':>10}"
        f"{'size KL':>9}{'own-price':>11}")
    log(f"{'truth':>10}{ceil:13.4f}{0.0:8.3f}{lt[:RANK].mean():11.3f}{lt[RANK:].mean():10.3f}"
        f"{0.0:9.4f}{'--':>11}")

    # exact own-price response of the TRUTH: d log pi_j / d log p_j at a 10% cut
    eps = math.log(0.9)
    def own_price(fn):
        base = fn(T["dlp"][0])
        pi0 = stats_from_logp(base, T["pairs"])[0]
        out = []
        for j in range(J):
            d2 = T["dlp"][0].clone(); d2[j] += eps
            pi1 = stats_from_logp(fn(d2), T["pairs"])[0]
            out.append((math.log(pi1[j]) - math.log(pi0[j])) / eps)
        return np.array(out)
    e_true = own_price(lambda d: truth_logp(0, d))

    for k, m in models.items():
        with torch.no_grad():
            ll = float(sum((m.logp(T["dlp"][t]) * c).sum() for t, c in te.items())
                       / sum(float(c.sum()) for c in te.values()))
            _, lf, law = stats_from_logp(m.logp(T["dlp"][0]), T["pairs"])
            el = own_price(lambda d: m.logp(d))
        res[k] = dict(loglik=ll, lifts=lf.tolist(), law=law.tolist(),
                      size_kl=kl(law, lawt), elast=el.tolist(),
                      elast_err=float(np.abs(el - e_true).mean()))
        log(f"{k:>10}{ll:13.4f}{ceil-ll:8.3f}{lf[:RANK].mean():11.3f}{lf[RANK:].mean():10.3f}"
            f"{kl(law,lawt):9.4f}{res[k]['elast_err']:11.4f}")
    res["_truth"]["elast"] = e_true.tolist()
    res["_meta"] = dict(J=J, NB=NB, steps=STEPS, n_ctx=N_CTX, rank=RANK, sub=SUB,
                        true_phiphi=TRUE_T)
    json.dump(res, open(os.path.join(OUT, "v3_arena.json"), "w"), indent=2)
    log(f"\nwrote {os.path.join(OUT, 'v3_arena.json')}")


if __name__ == "__main__":
    main()
