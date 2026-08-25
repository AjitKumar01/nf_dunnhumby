"""Does the model ruin its own ranking when the contextual terms are REAL and learnable?

s25/s26 gave every model a b of only an intercept plus price, and ours recommended at 98% of
the achievable ceiling.  The real model's b also carries season, store and recency:

    b_j = lam_j + theta_h.alpha_j - gamma.beta_j*dlp + mu_j.delta_week + zeta_j.xi_store
          + psi_j.rec_j

and on dunnhumby those terms are what destroys ranking -- lam+taste scores MRR 0.0511, above
popularity's 0.0467, and adding season/store/recency drops it to 0.0015.  So the arena
validated the part that works and was silent about the part that does not.

That leaves two possibilities the real data cannot distinguish:

  (a) the effects are REAL but the model fits them badly -> a fitting problem, fixable
  (b) the effects are weak or absent and the model is fitting noise -> the terms should
      not be there at all, and deleting them is right in principle

Here they are planted as genuinely informative and genuinely learnable.  Mission weights
depend on week and store, so which products are likely really does move with both; recency
raises the probability of re-buying within a mission.  Every model gets the same terms with
the same parameterisation as the real one.  If ours still ruins its ranking with the effects
present and identifiable, that is (a) and the fault is ours.  If it recovers them and ranks
well, the real-data failure is (b) -- the terms are fitting noise -- and the answer is a
smaller model, not a better fit.

Run:  python3 s27_ctx.py
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
STEPS = 500
N_WEEK, N_STORE = 4, 2
N_CTX = N_WEEK * N_STORE
N_MIS = 4
HUB = 0
DW = 3                      # width of the season / store factors
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[ctx] {m}", flush=True)


IDX = torch.arange(2 ** J, dtype=torch.int64)
MASK = ((IDX.unsqueeze(1) >> torch.arange(J, dtype=torch.int64)) & 1).to(torch.float64)
NVEC = MASK.sum(1).long()
NE = NVEC > 0
MASK_NE, NVEC_NE = MASK[NE].contiguous(), NVEC[NE]
NSUB = MASK_NE.shape[0]


def make_truth(seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.full((J,), 0.03)
    P, parts = [base.clone()], {}
    for m in range(1, N_MIS + 1):
        q = base.clone()
        q[HUB] = 0.80
        pr = [1 + 3 * (m - 1), 2 + 3 * (m - 1), 3 + 3 * (m - 1)]
        parts[m] = [p for p in pr if p < J]
        for p_ in parts[m]:
            q[p_] = 0.60
        P.append(q)
    P = torch.stack(P)
    # mission weights genuinely depend on week AND store, so season and store really do
    # change WHICH products are likely -- not merely how many
    wl = torch.randn(N_WEEK, N_MIS + 1, generator=g) * 0.9
    sl = torch.randn(N_STORE, N_MIS + 1, generator=g) * 0.7
    W = torch.stack([torch.softmax(wl[w] + sl[s], 0)
                     for w in range(N_WEEK) for s in range(N_STORE)])   # [N_CTX, M]
    ctx_week = torch.tensor([w for w in range(N_WEEK) for _ in range(N_STORE)])
    ctx_store = torch.tensor([s for _ in range(N_WEEK) for s in range(N_STORE)])
    # recency: per (context, product).  Higher recency raises the log-odds of re-buying,
    # which is a real, learnable, product-specific contextual effect.
    rec = torch.rand(N_CTX, J, generator=g)
    rec_w = 0.9 + 0.4 * torch.rand(J, generator=g)
    hub_pairs = [(HUB, p_) for m in parts for p_ in parts[m]]
    mate = [(a, b) for m in parts for i, a in enumerate(parts[m]) for b in parts[m][i + 1:]]
    return dict(P=P, W=W, week=ctx_week, store=ctx_store, rec=rec, rec_w=rec_w,
                beta=0.8 + 0.6 * torch.rand(J, generator=g),
                dlp=0.3 * torch.randn(N_CTX, J, generator=g),
                parts=parts, hub=hub_pairs, mate=mate, pairs=hub_pairs + mate)


T = make_truth()


def truth_logp(t, dlp=None):
    d = T["dlp"][t] if dlp is None else dlp
    lg = torch.logit(T["P"].clamp(1e-6, 1 - 1e-6))
    lg = lg - (T["beta"] * d).unsqueeze(0) + (T["rec_w"] * T["rec"][t]).unsqueeze(0)
    lpm = MASK_NE @ torch.nn.functional.logsigmoid(lg).T + \
        (1 - MASK_NE) @ torch.nn.functional.logsigmoid(-lg).T
    lp = torch.logsumexp(lpm + T["W"][t].log().unsqueeze(0), dim=1)
    return lp - torch.logsumexp(lp, 0)


class Model(torch.nn.Module):
    """Ours, with the real b: intercept, price, season, store, recency, plus phi and rho_0."""

    def __init__(self, kz=8, cap=2.5, ctx=True):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.a0 = torch.nn.Parameter(torch.zeros(J))
        self.bt = torch.nn.Parameter(torch.zeros(J))
        self.PH = torch.nn.Parameter(torch.randn(J, kz, generator=g) * 0.1)
        self.r0 = torch.nn.Parameter(torch.zeros(J + 1))
        self.ctx = ctx
        self.mu = torch.nn.Parameter(torch.randn(J, DW, generator=g) * 0.1)
        self.delta = torch.nn.Parameter(torch.randn(N_WEEK, DW, generator=g) * 0.1)
        self.zeta = torch.nn.Parameter(torch.randn(J, DW, generator=g) * 0.1)
        self.xi = torch.nn.Parameter(torch.randn(N_STORE, DW, generator=g) * 0.1)
        self.psi = torch.nn.Parameter(torch.zeros(J))
        self.cap = cap

    def b(self, t, dlp):
        v = self.a0 - torch.nn.functional.softplus(self.bt) * dlp
        if self.ctx:
            v = v + (self.mu * self.delta[T["week"][t]]).sum(-1)
            v = v + (self.zeta * self.xi[T["store"][t]]).sum(-1)
            v = v + self.psi * T["rec"][t]
        return v

    def logp(self, t, dlp=None, sub=None):
        d = T["dlp"][t] if dlp is None else dlp
        vv = MASK_NE @ self.PH
        pr = 0.5 * ((vv * vv).sum(1) - MASK_NE @ (self.PH ** 2).sum(1))
        lp = torch.log_softmax(MASK_NE @ self.b(t, d) + pr - self.r0[NVEC_NE], 0)
        return lp if sub is None else lp[sub]

    def clip(self):
        with torch.no_grad():
            n = self.PH.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.PH.mul_(torch.clamp(self.cap / n, max=1.0))


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
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    obs = {t: torch.nonzero(c > 0).flatten() for t, c in counts.items()}
    cnz = {t: counts[t][obs[t]] for t in counts}
    t0 = time.time()
    for _ in range(STEPS):
        loss = sum(-(model.logp(t, sub=obs[t]) * cnz[t]).sum() for t in counts)
        loss = loss / sum(float(c.sum()) for c in cnz.values())
        if pool > 0:
            g = torch.nn.functional.softplus(model.bt)
            loss = loss + pool * ((g - g.mean()) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); model.clip()
    log(f"  {tag:22s} {time.time()-t0:6.1f}s  train/basket {-float(loss):9.4f}")
    return model


def sub_index(items):
    return int(sum(1 << int(j) for j in items)) - 1


def main():
    log(f"J={J}, {N_WEEK} weeks x {N_STORE} stores; mission weights depend on BOTH, so "
        f"season and store really change which products are likely")
    log(f"recency is per (context, product) with a positive learnable weight\n")
    tr, te = sample_counts(0), sample_counts(1)
    models = {"ours, full b": Model(ctx=True), "ours, no ctx terms": Model(ctx=False)}
    for k, m in models.items():
        fit(m, tr, k, pool=2.0)

    rng = np.random.default_rng(7)
    cases = []
    for t, c in te.items():
        nz = torch.nonzero(c > 0).flatten().tolist()
        for i in rng.choice(nz, size=min(300, len(nz)), replace=False):
            mem = torch.nonzero(MASK_NE[int(i)]).flatten().tolist()
            if len(mem) < 2:
                continue
            hid = int(rng.choice(mem))
            cases.append((t, [x for x in mem if x != hid], hid))
    log(f"\n{len(cases):,} held-out cases")

    def rank_with(fn):
        out = []
        for t, rest, hid in cases:
            cand = [j for j in range(J) if j not in rest]
            sc = fn(t, rest, cand)
            out.append(int(sum(1 for v in sc if v > sc[cand.index(hid)])) + 1)
        return np.array(out, dtype=float)

    with torch.no_grad():
        caches = {"truth": {t: truth_logp(t) for t in te}}
        for k, m in models.items():
            caches[k] = {t: m.logp(t) for t in te}

    def mk(cache):
        return lambda t, rest, cand: [float(cache[t][sub_index(rest + [j])]) for j in cand]

    log(f"{'model':>22}{'R@1':>8}{'R@2':>8}{'R@3':>8}{'MRR':>9}{'median':>8}")
    res = {}
    for name in ("truth", "ours, full b", "ours, no ctx terms"):
        r = rank_with(mk(caches[name]))
        res[name] = dict(MRR=float((1 / r).mean()),
                         **{f"R@{k}": float((r <= k).mean()) for k in (1, 2, 3)})
        log(f"{name:>22}" + "".join(f"{100*res[name][f'R@{k}']:7.1f}%" for k in (1, 2, 3))
            + f"{res[name]['MRR']:9.4f}{np.median(r):8.1f}")
    json.dump(res, open(os.path.join(OUT, "v3_ctxsim.json"), "w"), indent=2)
    log("")
    log("If 'full b' matches 'no ctx terms', the terms are harmless when the effects are real.")
    log("If it is much worse, the fault is ours even with a learnable signal present.")


if __name__ == "__main__":
    main()
