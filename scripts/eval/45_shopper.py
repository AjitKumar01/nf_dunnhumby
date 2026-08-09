"""
Stage 45 -- SHOPPER as a baseline, and a two-way comparison that gives neither model
home advantage.

SHOPPER (Ruiz, Athey & Blei) ships as CUDA C++ (`emb.cu`) and this machine has no NVIDIA
GPU, so the released code cannot be run.  This reimplements the model in PyTorch from the
paper and the parameter names in its source (lambda, theta, alpha, rho, gamma, beta,
delta).  Three things differ from ours by design, and they are the point of the
comparison:

  1. SEQUENTIAL, NOT SET.  Items are placed one at a time and each sees the mean of alpha
     over the items ALREADY placed.  The unordered likelihood sums over permutations;
     SHOPPER samples one, and so do we.  Ours conditions on the whole basket minus the
     item, which is a set statement and needs no ordering.
  2. UNTIED rho.  The effect of k on j is rho_j . alpha_k, which need not equal
     rho_k . alpha_j.  That expresses complements whose attributes sit far apart, which a
     symmetric interaction cannot -- at the cost of admitting no potential function, hence
     no joint distribution over baskets and no valid Gibbs sampler.
  3. CATALOGUE-WIDE CHOICE.  The softmax runs over all 5,455 products, not the purchased
     product's category.

What is NOT reproduced: SHOPPER is Bayesian and maximises an ELBO with per-parameter
variational means and standard deviations.  This fits the same model by MAP with an L2
penalty, which is the mode of the same posterior.  So the comparison is of MODELS, not of
inference procedures, and the SHOPPER column carries no uncertainty of its own.

THE METRIC PROBLEM.  Our conditional normalises over a category; SHOPPER's over the
catalogue.  Scoring only on ours would flatter us.  So both models are scored on BOTH,
exactly, with 5,455 products being small enough to sum:

  within-category   chance -log 47   = -3.85
  full catalogue    chance -log 3087 = -8.04   (products the store stocks)

Writes out/shopper.json.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# the model lives in ../model; add it to the path so `27_nested_basket` and
# `28_nested_counterfactual` resolve by their bare module names.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
nb = importlib.import_module("27_nested_basket")
cf = importlib.import_module("28_nested_counterfactual")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "..", "basket_input")
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[45] {m}", flush=True)


class Shopper(nn.Module):
    """Psi(j) = lambda_j + theta_u.alpha_j + rho_j.mean(alpha of items already placed)
                - (gamma_u.beta_j) dlogp + delta_w.mu_j"""

    def __init__(self, d, K=64, Kp=8, Kt=8, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        e = lambda n, k, s=0.05: nn.Parameter(torch.randn(n, k, generator=g) * s)
        self.d, self.K = d, K
        self.lam = nn.Parameter(torch.zeros(d.J))
        self.alpha, self.rho = e(d.J, K), e(d.J, K)          # UNTIED
        self.theta = e(d.N, K)
        self.gamma, self.beta = e(d.N, Kp, 0.1), e(d.J, Kp, 0.1)
        self.mu, self.delta = e(d.J, Kt, 0.02), e(52, Kt, 0.02)

    def psi(self, users, cand, ctx, dlogp, weeks):
        s = self.lam[cand]
        s = s + torch.einsum("bk,bmk->bm", self.theta[users], self.alpha[cand])
        s = s + (self.rho[cand] * ctx.unsqueeze(1)).sum(-1)
        s = s - (self.gamma[users].unsqueeze(1) * self.beta[cand]).sum(-1) * dlogp
        s = s + (self.mu[cand] * self.delta[weeks].unsqueeze(1)).sum(-1)
        return s

    def l2(self):
        return sum((p ** 2).sum() for p in
                   [self.alpha, self.rho, self.theta, self.mu, self.delta])

    def l2p(self):
        return (self.gamma ** 2).sum() + (self.beta ** 2).sum()


def prefix_ctx(model, d, sp, rows, owner, lens, rng, device):
    """Mean of alpha over the items placed BEFORE this one, under one random order."""
    item = sp["item"][rows]
    A = model.alpha[torch.as_tensor(item, device=device)]
    B = len(rows)
    ctx = torch.zeros(B, model.K, device=device)
    off = np.concatenate([[0], np.cumsum(lens)])
    for b in range(len(lens)):
        sl = slice(off[b], off[b + 1])
        n = off[b + 1] - off[b]
        if n < 2:
            continue
        perm = rng.permutation(n)
        run = torch.zeros(model.K, device=device)
        for pos, q in enumerate(perm):
            if pos:
                ctx[off[b] + q] = run / pos
            run = run + A[off[b] + q]
    return ctx


def main(a):
    dev = torch.device("cpu")
    d = nb.NestedData(IN, device=dev)
    tr, sp = d.splits["train"], d.splits["test"]
    rng = np.random.default_rng(a.seed)
    m = Shopper(d, K=a.K, seed=a.seed).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    log(f"SHOPPER reimplementation: K={a.K}, untied rho, prefix context, "
        f"catalogue-wide softmax with {a.n_neg} sampled negatives")

    for it in range(1, a.iters + 1):
        b = rng.integers(0, tr["n_baskets"], size=a.batch)
        starts, ends = tr["starts"][b], tr["ends"][b]
        lens = ends - starts
        rows = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
        owner = np.repeat(np.arange(len(b)), lens)
        item, user = tr["item"][rows], tr["user"][rows]
        day, week = tr["day"][rows], tr["week"][rows]
        B = len(rows)
        neg = rng.choice(d.J, size=(B, a.n_neg), p=d.neg_p).astype(np.int64)
        cand = np.concatenate([item[:, None], neg], 1)
        ctx = prefix_ctx(m, d, tr, rows, owner, lens, rng, dev)
        dl = d.log_price_dev[torch.as_tensor(cand),
                             torch.as_tensor(np.repeat(day[:, None], cand.shape[1], 1))]
        s = m.psi(torch.as_tensor(user), torch.as_tensor(cand), ctx, dl,
                  torch.as_tensor(week))
        loss = -torch.log_softmax(s, 1)[:, 0].mean() \
            + (a.l2 * m.l2() + a.l2p * m.l2p()) / 1500.0
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0:
            log(f"  it {it:5d}  train loss {float(loss):.4f}")

    # ------------------------------------------------------------------ scoring
    log("")
    log("scoring both models on BOTH metrics, identical rows")
    bidx = rng.choice(sp["n_baskets"], size=min(a.n_baskets, sp["n_baskets"]),
                      replace=False)
    rows = np.concatenate([np.arange(sp["starts"][i], sp["ends"][i]) for i in bidx])
    lens = np.array([sp["ends"][i] - sp["starts"][i] for i in bidx])
    owner = np.repeat(np.arange(len(bidx)), lens)
    user, item = sp["user"][rows], sp["item"][rows]
    day, week, store = sp["day"][rows], sp["week"][rows], sp["store"][rows]
    B = len(rows)
    res = {"n_rows": int(B), "models": {}}

    def cat_block():
        cats = d.item_cat_np[item]
        cand = d.cat_items_np[cats]
        av = torch.as_tensor(d.cat_mask_np[cats] > 0)
        av &= d.carried[torch.as_tensor(cand), torch.as_tensor(store).unsqueeze(1)]
        kp = owner.astype(np.int64) * d.J + item
        kc = owner[:, None].astype(np.int64) * d.J + cand
        av &= torch.as_tensor(~np.isin(kc, kp))
        tgt = torch.as_tensor(d.item_pos_np[item])
        av[torch.arange(B), tgt] = True
        return cand, av, tgt

    cand_c, av_c, tgt_c = cat_block()
    allj = np.tile(np.arange(d.J), (B, 1))
    av_f = d.carried[torch.as_tensor(allj), torch.as_tensor(store).unsqueeze(1)].clone()
    kp = owner.astype(np.int64) * d.J + item
    av_f &= torch.as_tensor(~np.isin(owner[:, None].astype(np.int64) * d.J + allj, kp))
    av_f[torch.arange(B), torch.as_tensor(item)] = True
    log(f"  {B:,} rows; within-category {float(av_c.float().sum(1).mean()):.1f} "
        f"products, full catalogue {float(av_f.float().sum(1).mean()):.0f}")

    @torch.no_grad()
    def score(fn, cand, av, tgt, chunk=256):
        out = []
        for s0 in range(0, B, chunk):
            sl = slice(s0, min(s0 + chunk, B))
            u = fn(sl, cand[sl] if cand.ndim == 2 else cand)
            u = u.masked_fill(~av[sl], -1e9)
            ar = torch.arange(u.shape[0])
            out.append(torch.log_softmax(u, 1)[ar, tgt[sl]])
        return torch.cat(out).numpy()

    # SHOPPER, prefix context under one random order (its own likelihood)
    ctx_s = prefix_ctx(m, d, sp, rows, owner, lens, np.random.default_rng(a.seed), dev)

    def shop_fn(sl, cnd):
        c = torch.as_tensor(cnd)
        dl = d.log_price_dev[c, torch.as_tensor(
            np.repeat(day[sl][:, None], c.shape[1], 1))]
        return m.psi(torch.as_tensor(user[sl]), c, ctx_s[sl], dl,
                     torch.as_tensor(week[sl]))

    for nm, cnd, av, tgt in [("within-category", cand_c, av_c, tgt_c),
                             ("full catalogue", allj, av_f,
                              torch.as_tensor(item))]:
        lp = score(shop_fn, cnd, av, tgt)
        res["models"].setdefault("SHOPPER", {})[nm] = float(lp.mean())
        log(f"  SHOPPER      {nm:16s} {lp.mean():+.4f}")

    # ours, leave-one-out context (its own likelihood)
    for lab in a.labels:
        if not os.path.exists(os.path.join(OUT, f"{lab}_nested.pt")):
            continue
        om, _ = cf.load(lab, d, dev)
        A = om.alpha.detach()
        ctx_o = torch.zeros(B, om.K)
        for k in range(len(bidx)):
            q = np.flatnonzero(owner == k)
            if len(q) > 1:
                tot = A[torch.as_tensor(item[q])].sum(0)
                for z, rr in enumerate(q):
                    ctx_o[rr] = (tot - A[item[q[z]]]) / (len(q) - 1)

        def our_fn(sl, cnd, _m=om):
            c = torch.as_tensor(cnd)
            dr = np.repeat(day[sl][:, None], c.shape[1], 1)
            st = torch.as_tensor(d.state(
                np.repeat(user[sl][:, None], c.shape[1], 1).ravel(),
                cnd.ravel() if cnd.ndim == 2 else np.tile(cnd, (len(dr), 1)).ravel(),
                dr.ravel()).reshape(len(dr), c.shape[1], nb.N_STATE_FEATURES))
            dl = d.log_price_dev[c, torch.as_tensor(dr)]
            return _m.item_utility(torch.as_tensor(user[sl]), c, ctx_o[sl], dl, st,
                                   torch.as_tensor(week[sl]),
                                   torch.as_tensor(store[sl]))

        for nm, cnd, av, tgt in [("within-category", cand_c, av_c, tgt_c),
                                 ("full catalogue", allj, av_f,
                                  torch.as_tensor(item))]:
            lp = score(our_fn, cnd, av, tgt)
            res["models"].setdefault(lab, {})[nm] = float(lp.mean())
            log(f"  {lab:12s} {nm:16s} {lp.mean():+.4f}")

    with open(os.path.join(OUT, "shopper.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log("wrote out/shopper.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", default=["ps_nested"])
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--n-neg", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--l2", type=float, default=1e-2)
    p.add_argument("--l2p", type=float, default=1e-4)
    p.add_argument("--n-baskets", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
