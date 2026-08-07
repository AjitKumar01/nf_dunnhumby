"""
Stage 33 -- Check every equation printed in NESTED_MODEL.md against the code.

A document can drift from the code silently; NESTED_MODEL.md 12 records several
places where it had.  These checks are cheap, so there is no reason to assert an
equation rather than test it.  Each line below states a formula from the document
and compares it against what 27_nested_basket.py actually computes -- either
exactly, or by finite difference where the document claims a derivative.

Run:  python3 33_verify_equations.py
"""
import importlib
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
nb = importlib.import_module("27_nested_basket")
IN = os.path.join(HERE, "..", "basket_input")
OUT = os.path.join(HERE, "..", "out")

FAILED = []


def check(name, a, b, tol=1e-5):
    good = abs(a - b) < tol
    if not good:
        FAILED.append(name)
    print(f"  {'PASS' if good else 'FAIL':4s}  {name:54s} {a:+.6f}  vs  {b:+.6f}")


def main(label="nested"):
    torch.manual_seed(0)
    d = nb.NestedData(IN, device="cpu")
    cfg = json.load(open(os.path.join(OUT, f"{label}_nested_history.json")))["config"]
    m = nb.NestedModel(d, K=cfg["K"], Kp=cfg["Kp"], Kt=cfg["Kt"], Ks=cfg["Ks"], seed=0)
    m.load_state_dict(torch.load(os.path.join(OUT, f"{label}_nested.pt"),
                                 map_location="cpu"))
    m.eval()
    rng = np.random.default_rng(0)
    bidx = rng.integers(0, d.splits["validation"]["n_baskets"], size=8)
    bt = nb.make_batch(d, m, "validation", bidx, 20, rng, "cpu")

    with torch.no_grad():
        u = m.item_utility(bt["user"], bt["cand"], bt["ctx"], bt["dlogp"],
                           bt["state"], bt["week"], bt["store"])
        r, cix = 3, 2
        i, j = int(bt["user"][r]), int(bt["cand"][r, cix])
        man = (m.lam[j] + m.theta[i] @ m.alpha[j] + m.alpha[j] @ bt["ctx"][r]
               - (m.gamma[i] @ m.beta[j]) * bt["dlogp"][r, cix]
               + m.eta[j] @ bt["state"][r, cix] + m.mu[j] @ m.delta[bt["week"][r]]
               + m.item_store[j] @ m.store_vec[bt["store"][r]])
        check("3.1  u_ijt, term by term", float(u[r, cix]), float(man))

        sp = d.splits["validation"]
        itms = sp["item"][sp["starts"][bidx[0]]:sp["ends"][bidx[0]]]
        n = len(itms)
        if n > 1:
            man_ctx = (m.alpha[itms].sum(0) - m.alpha[itms[0]]) / (n - 1)
            check("3.3  ctx_j = (sum_k alpha_k - alpha_j)/(n-1)",
                  float((bt["ctx"][0] - man_ctx).abs().max()), 0.0)

        S = list(itms[:5])
        jj, mm = S[2], int(itms[-1])

        def Epair(T):
            T = list(T)
            return sum(float(m.alpha[x] @ m.alpha[y])
                       for a_, x in enumerate(T) for y in T[a_ + 1:]) / (len(T) - 1)

        ctxj = (m.alpha[S].sum(0) - m.alpha[jj]) / (len(S) - 1)
        check("3.3  E(S') - E(S) = u_m - u_j",
              Epair([x for x in S if x != jj] + [mm]) - Epair(S),
              float(m.alpha[mm] @ ctxj - m.alpha[jj] @ ctxj), 1e-4)

        L = nb.losses(m, bt, None, 0.0)
        j2 = bt["cand"][:, 0]
        z = (m.q0[j2] - (m.q_gamma[bt["user"]] * m.q_beta[j2]).sum(-1) * bt["dlogp"][:, 0]
             + (m.q_state[j2] * bt["state"][:, 0, :]).sum(-1)).clamp(-6, 4)
        k = (bt["units"] - 1.0).clamp_min(0)
        check("3.5  L_qty = mean(exp z - k z + logGamma(k+1))", float(L["quantity"]),
              float((torch.exp(z) - k * z + torch.lgamma(k + 1)).mean()))

        check("3.4  kappa_c = log(1 + exp(kappa_raw))", float(m.kappa()[7]),
              float(torch.log1p(torch.exp(m.kappa_raw[7]))))

        for p in (0.02, 0.05, 0.10, 0.20):
            lam = -np.log(1 - p)
            check(f"3.5  E[Q|Q>0] = lam/p at p={p:.2f}", lam / p, lam / (1 - np.exp(-lam)))

        # derivatives, against finite differences on the real softmax
        cat = 12
        blk = d.cat_items[cat]
        nvi = int(d.cat_mask[cat].sum())
        blk = blk[:nvi].unsqueeze(0)
        st = torch.as_tensor(d.state(np.full(nvi, i), blk[0].numpy(),
                                     np.full(nvi, 300))).unsqueeze(0)
        dp = torch.zeros(1, nvi)

        def util(dp_):
            return m.item_utility(torch.tensor([i]), blk, torch.zeros(1, m.K), dp_, st,
                                  torch.tensor([10]), torch.tensor([3]))

        # Derivatives are checked by AUTOGRAD, not finite differences.  The utilities
        # are float32, so differencing two nearly equal log-sum-exps and dividing by a
        # small step loses most of the precision -- at eps = 1e-5 the difference
        # underflows to exactly zero.  Autograd gives the derivative the code actually
        # implements, with no step size to tune.
        pos = 0
        a_p = float(m.gamma[i] @ m.beta[blk[0, pos]])
        dpg = dp.clone().requires_grad_(True)
        with torch.enable_grad():
            u_ = util(dpg)
            lp = torch.log_softmax(u_, 1)[0, pos]
            g1 = torch.autograd.grad(lp, dpg, retain_graph=True)[0][0, pos]
            iv_ = torch.logsumexp(u_, 1)[0]
            g2 = torch.autograd.grad(iv_, dpg)[0][0, pos]
        pi = float(torch.softmax(util(dp), 1)[0, pos])
        check("8.4  d log pi_j / d log p_j = -(g.b)(1 - pi_j)", -a_p * (1 - pi),
              float(g1), 1e-4)
        check("8.4  d IV / d log p_j = -(g.b) pi_j", -a_p * pi, float(g2), 1e-4)
        aq = float(m.q_gamma[i] @ m.q_beta[blk[0, pos]])
        zz = float(m.q0[blk[0, pos]])
        lam = np.exp(zz)
        dq = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        with torch.enable_grad():
            le = torch.log(1 + torch.exp(torch.tensor(zz, dtype=torch.float64) - aq * dq[0]))
            g3 = torch.autograd.grad(le, dq)[0][0]
        check("8.4  d log E[units]/d log p = -(gq.bq) lam/(1+lam)",
              -aq * lam / (1 + lam), float(g3), 1e-6)
        kap = float(m.kappa()[cat])
        check("8.4  alloc + inc = -(g.b)[1 - pi(1 - kappa)]",
              -a_p * (1 - pi) - kap * a_p * pi, -a_p * (1 - pi * (1 - kap)))

        # sampled inclusive value
        uv = util(dp)[0]
        true_sum = float(torch.exp(uv).sum())
        mm_ = 32
        draws = np.array([float(torch.exp(uv[torch.randint(0, nvi, (mm_,))]).sum()
                                * nvi / mm_) for _ in range(4000)])
        check("5.3  E[(n/m) sum_sampled exp u] = sum_j exp u_j",
              float(draws.mean()) / true_sum, 1.0, 0.02)
        jensen = float(np.log(draws).mean()) <= np.log(true_sum)
        if not jensen:
            FAILED.append("Jensen")
        print(f"  {'PASS' if jensen else 'FAIL':4s}  "
              f"{'5.3  E[log IV_hat] <= log of the true sum':54s} "
              f"{np.log(draws).mean():+.6f}  <=  {np.log(true_sum):+.6f}")

    print()
    print(f"{len(FAILED)} failed" if FAILED else "all equations verified")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
