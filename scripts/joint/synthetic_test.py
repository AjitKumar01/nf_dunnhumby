"""
The decisive experiment: on data generated FROM the model with known parameters, does
maximum likelihood recover them better than the pseudo-likelihood the current model uses?

This is the cheapest way to answer the question, and it can only go two ways.  If the joint
objective does not beat the pseudo-likelihood at recovering parameters that generated the
data, the direction is answered negatively for a few minutes of compute rather than after a
dunnhumby refit.

Design.  Small enough that everything is exact and fast, structured like the real model:

    b_ij = lambda_j + theta_i' alpha_j
    E(S) = sum_{j in S} b_ij + (lam/2)(||sum alpha||^2 - sum ||alpha||^2),  lam = 1/(n-1)

Baskets are drawn EXACTLY from P(S|K) by sampler.py, which was validated against enumerated
probabilities to within sampling noise.  So the data really does come from the model, and
any failure to recover is the estimator's fault rather than misspecification.

What is scored.  Only alpha's Gram matrix G = alpha alpha' is identified, since the model is
invariant to rotating alpha and theta together, so recovery is measured on G -- which is
rotation-invariant by construction.  Two numbers, because they fail differently: correlation
of the off-diagonal entries (is the STRUCTURE right) and relative Frobenius error (is the
SCALE right).

Writes out/synthetic_recovery.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ragged import RaggedNormaliser                                    # noqa: E402
from sampler import sample_basket                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[syn] {m}", flush=True)


def make_truth(a):
    rng = np.random.default_rng(a.seed)
    cats = [np.arange(c * a.per_cat, (c + 1) * a.per_cat) for c in range(a.n_cat)]
    J = a.n_cat * a.per_cat
    alpha = rng.normal(0.0, 1.0, (J, a.K))
    alpha /= np.linalg.norm(alpha, axis=1, keepdims=True)
    alpha *= a.alpha_norm
    return dict(cats=cats, J=J, per_cat=a.per_cat,
                lam_j=rng.normal(0.0, 0.7, J), alpha=alpha,
                theta=rng.normal(0.0, 0.6, (a.n_house, a.K)), rng=rng)


def generate(a, T):
    rng, data, t0 = T["rng"], [], time.time()
    for v in range(a.n_baskets):
        i = int(rng.integers(0, a.n_house))
        chosen = rng.choice(a.n_cat, size=int(rng.integers(a.min_cat, a.max_cat + 1)),
                            replace=False)
        ks = [int(rng.integers(1, a.max_k + 1)) for _ in chosen]
        if sum(ks) < 2:
            ks[0] = 2
        lam = 1.0 / (sum(ks) - 1)
        cc, cp = [], []
        for c, k in zip(chosen, ks):
            idx = T["cats"][c]
            b = T["lam_j"][idx] + T["alpha"][idx] @ T["theta"][i]
            cc.append(b - (lam / 2) * (T["alpha"][idx] ** 2).sum(1))
            cp.append(T["alpha"][idx])
        S, _ = sample_basket(cc, cp, ks, lam, rng, n_prop=a.sampler_draws)
        items = [int(T["cats"][c][j]) for c, sub in zip(chosen, S) for j in sub]
        data.append(dict(house=i, cats=[int(c) for c in chosen], ks=ks,
                         items=items, lam=lam))
        if (v + 1) % 300 == 0:
            log(f"   generated {v + 1}/{a.n_baskets}  ({time.time() - t0:.0f}s)")
    return data


class Model(torch.nn.Module):
    def __init__(self, J, n_house, K, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lam_j = torch.nn.Parameter(torch.zeros(J))
        self.alpha = torch.nn.Parameter(torch.randn(J, K, generator=g) * 0.3)
        self.theta = torch.nn.Parameter(torch.randn(n_house, K, generator=g) * 0.3)


def energy(m, batch, T):
    out = []
    for d in batch:
        idx = torch.as_tensor(d["items"])
        b = m.lam_j[idx] + m.alpha[idx] @ m.theta[d["house"]]
        A = m.alpha[idx]
        s = A.sum(0)
        out.append(b.sum() + d["lam"] * (((s * s).sum() - (A * A).sum()) / 2.0))
    return torch.stack(out)


def joint_loss(m, batch, T, norm):
    row_id, row_k, row_b, lam, item_idx = [], [], [], [], []
    nr = 0
    for bi, d in enumerate(batch):
        lam.append(d["lam"])
        for c, k in zip(d["cats"], d["ks"]):
            idx = T["cats"][c]
            item_idx.append(idx)
            row_id += [nr] * len(idx)
            row_k.append(k)
            row_b.append(bi)
            nr += 1
    idx = torch.as_tensor(np.concatenate(item_idx))
    row_id = torch.tensor(row_id); row_k = torch.tensor(row_k)
    row_b = torch.tensor(row_b); lam = torch.tensor(lam)
    hh = torch.tensor([d["house"] for d in batch])
    h_item = hh[row_b[row_id]]
    b = m.lam_j[idx] + (m.alpha[idx] * m.theta[h_item]).sum(-1)
    cj = b - (lam[row_b[row_id]] / 2.0) * (m.alpha[idx] ** 2).sum(-1)
    lz, _, ess = norm.log_z(cj, m.alpha[idx], row_id, row_k, row_b, nr,
                            len(batch), lam)
    return -(energy(m, batch, T) - lz).mean(), ess.mean()


def pl_loss(m, batch, T):
    tot, cnt = 0.0, 0
    pc = T["per_cat"]
    for d in batch:
        items = d["items"]
        A_all = m.alpha[torch.as_tensor(items)]
        ssum = A_all.sum(0)
        n = len(items)
        for pos, j in enumerate(items):
            idx = torch.as_tensor(T["cats"][j // pc])
            ctx = (ssum - A_all[pos]) / max(n - 1, 1)
            u = m.lam_j[idx] + m.alpha[idx] @ m.theta[d["house"]] + m.alpha[idx] @ ctx
            mates = [q for q in items if q != j and q // pc == j // pc]
            if mates:
                keep = ~torch.isin(idx, torch.as_tensor(mates))
                idx, u = idx[keep], u[keep]
            tot = tot + torch.log_softmax(u, 0)[int((idx == j).nonzero()[0])]
            cnt += 1
    return -tot / cnt


def gram_scores(A_hat, A_true):
    G1, G2 = A_hat @ A_hat.T, A_true @ A_true.T
    off = ~np.eye(len(G1), dtype=bool)
    return (float(np.corrcoef(G1[off], G2[off])[0, 1]),
            float(np.linalg.norm(G1 - G2) / np.linalg.norm(G2)))


def main(a):
    torch.set_default_dtype(torch.float64)
    T = make_truth(a)
    log(f"truth: {T['J']} items in {a.n_cat} categories, K={a.K}, "
        f"{a.n_house} households, ||alpha||={a.alpha_norm}")
    data = generate(a, T)
    log(f"generated {len(data):,} baskets, mean size "
        f"{np.mean([len(d['items']) for d in data]):.2f}")

    res = {"config": vars(a)}
    norm = RaggedNormaliser(n_draws=a.draws, mode_steps=25, kmax=max(a.max_k, 2))
    for name in ("joint", "pseudo"):
        m = Model(T["J"], a.n_house, a.K, seed=1)
        opt = torch.optim.Adam(m.parameters(), lr=a.lr)
        rng = np.random.default_rng(0)
        t0 = time.time()
        for it in range(1, a.iters + 1):
            batch = [data[s] for s in rng.choice(len(data), size=a.batch, replace=False)]
            if name == "joint":
                loss, ess = joint_loss(m, batch, T, norm)
            else:
                loss, ess = pl_loss(m, batch, T), torch.tensor(float("nan"))
            opt.zero_grad(); loss.backward(); opt.step()
            if it % max(1, a.iters // 5) == 0:
                c, f = gram_scores(m.alpha.detach().numpy(), T["alpha"])
                extra = f"  ESS {float(ess):.2f}" if name == "joint" else ""
                log(f"   {name:7s} it {it:4d}  loss {float(loss):8.4f}  "
                    f"gram corr {c:+.4f}  rel err {f:.4f}{extra}")
        c, f = gram_scores(m.alpha.detach().numpy(), T["alpha"])
        res[name] = {"gram_corr": c, "gram_rel_err": f, "secs": time.time() - t0}

    j, p = res["joint"], res["pseudo"]
    log("")
    log(f"  {'objective':10s} {'gram corr':>10s} {'rel err':>9s} {'secs':>7s}")
    log(f"  {'joint':10s} {j['gram_corr']:+10.4f} {j['gram_rel_err']:9.4f} {j['secs']:7.0f}")
    log(f"  {'pseudo':10s} {p['gram_corr']:+10.4f} {p['gram_rel_err']:9.4f} {p['secs']:7.0f}")
    better = j["gram_corr"] > p["gram_corr"] and j["gram_rel_err"] < p["gram_rel_err"]
    res["joint_better"] = bool(better)
    log("")
    log(f"  -> the joint objective {'DOES' if better else 'DOES NOT'} recover the truth "
        f"better on data it generated")
    with open(os.path.join(OUT, "synthetic_recovery.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    log("wrote out/synthetic_recovery.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-cat", type=int, default=12)
    p.add_argument("--per-cat", type=int, default=8)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--n-house", type=int, default=60)
    p.add_argument("--n-baskets", type=int, default=1200)
    p.add_argument("--min-cat", type=int, default=2)
    p.add_argument("--max-cat", type=int, default=4)
    p.add_argument("--max-k", type=int, default=2)
    p.add_argument("--alpha-norm", type=float, default=1.2)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--draws", type=int, default=48)
    p.add_argument("--sampler-draws", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
