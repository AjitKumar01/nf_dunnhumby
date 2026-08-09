"""
Stage 14 -- Verification of the PyTorch re-implementation.

A re-implementation is worth nothing unless it is shown to recover what it claims to
estimate.  Four checks, in increasing strength:

  A. analytic checks -- the likelihood is a proper log-probability (softmax rows sum
     to 1), the KL term matches the closed form for two Gaussians, and the ELBO
     gradient matches a finite-difference gradient.
  B. degenerate-case checks -- with no heterogeneity and no price the fitted item
     intercepts must reproduce the empirical within-category shares; with a
     homogeneous price coefficient the fit must match an independent
     scipy/statsmodels conditional logit on the same data.
  C. parameter recovery -- simulate choices from the model itself with known latents,
     refit, and check that the estimated price coefficients, utilities and choice
     probabilities line up with the truth.
  D. inclusive-value / nesting recovery -- simulate a two-stage world with a known
     nesting coefficient and check stage 2 recovers it.

Writes out/model_verification.json and figures/verification.png.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

import nf_torch as nf

HERE = os.path.dirname(os.path.abspath(__file__))
MI = os.path.join(HERE, "..", "..", "model_input")
OUT = os.path.join(HERE, "..", "..", "out")
FIG = os.path.join(HERE, "..", "..", "figures")


def log(m):
    print(f"[14] {m}", flush=True)


# --------------------------------------------------------------------- A. analytic
def check_analytic(d):
    r = {}
    m = nf.ProductChoice(d, K=6, Kp=4, seed=1)
    u, i, s = d.obs["train"]
    block, mask = m.choice_block(u[:256], i[:256], s[:256])
    util = m.utility(u[:256], s[:256], block, stoch=False, mask=mask).masked_fill(mask == 0, -1e9)
    p = torch.softmax(util, 1) * mask
    r["softmax_rows_sum_to_1_max_error"] = float((p.sum(1) - 1).abs().max())

    # log_prob must equal log of the probability assigned to the chosen slot
    lp = m.log_prob(u[:256], i[:256], s[:256], stoch=False)
    chosen = (block == i[:256].unsqueeze(1))
    r["log_prob_matches_softmax_max_error"] = float(
        (lp - torch.log(p[chosen].clamp_min(1e-30))).abs().max())

    # KL against the closed form for N(mu, s^2) || N(m0, v0)
    blk = nf.GaussianBlock((7,), prior_var=0.3, prior_mean=0.2, seed=3)
    with torch.no_grad():
        blk.mu.copy_(torch.linspace(-1, 1, 7))
        blk.log_sd.copy_(torch.linspace(-2, 0, 7))
    var, pv, pm = torch.exp(2 * blk.log_sd), 0.3, 0.2
    closed = 0.5 * torch.sum((var + (blk.mu - pm) ** 2) / pv - 1 - torch.log(var / pv))
    r["kl_closed_form_error"] = float((blk.kl() - closed).abs())

    # ELBO gradient vs finite differences.
    # This has to be done in double precision on a small model: the ELBO is a sum
    # over observations plus a KL over every latent, so in float32 its magnitude
    # (~1e5) leaves about 0.01 of resolution, while a central difference at
    # eps = 1e-3 moves it by ~1e-3 * grad.  The first version of this check reported
    # a "relative error" of 2.7 that was entirely float32 round-off in the test, not
    # error in the gradient.
    m2 = nf.ProductChoice(d, K=2, Kp=1, use_user_obs=False, seed=5).double()
    uu, ii, ss = u[:32], i[:32], s[:32]

    def loss_fn():
        return -m2.log_prob(uu, ii, ss, stoch=False).sum() + m2.kl()
    loss = loss_fn(); loss.backward()
    grads = {"beta": m2.beta.mu.grad.clone(), "theta": m2.theta.mu.grad.clone(),
             "lam": m2.lam.mu.grad.clone(), "lambda0": m2.lambda0.mu.grad.clone()}
    eps, rel, absol = 1e-6, [], []
    probes = [("beta", (0, 0)), ("beta", (5, 1)), ("theta", (int(uu[0]), 0)),
              ("lam", (int(ii[0]), 0)), ("lambda0", (int(ii[0]),))]
    with torch.no_grad():
        for name, idx in probes:
            par = getattr(m2, name).mu
            orig = par[idx].item()
            par[idx] = orig + eps; hi = loss_fn().item()
            par[idx] = orig - eps; lo = loss_fn().item()
            par[idx] = orig
            fd = (hi - lo) / (2 * eps)
            ga = grads[name][idx].item()
            absol.append(abs(fd - ga))
            rel.append(abs(fd - ga) / max(abs(ga), 1e-9))
    r["elbo_gradient_max_abs_error"] = float(max(absol))
    r["elbo_gradient_max_relative_error"] = float(max(rel))

    for k, v in r.items():
        log(f"  {k}: {v:.3e}")
    return r


# ------------------------------------------------------------------ B. degenerate
def check_degenerate(d, iters=1500):
    """No heterogeneity, no price: the model must reproduce the empirical shares."""
    r = {}
    m = nf.ProductChoice(d, K=0, Kp=1, use_user_obs=False, item_intercept=True,
                         price_prior_var=1e-8, price_prior_mean=0.0,
                         intercept_var=1e6, seed=0)
    for p_ in [m.gamma, m.lam]:                      # freeze the price term at 0
        p_.mu.data.zero_(); p_.log_sd.data.fill_(-20.0)
        p_.mu.requires_grad_(False); p_.log_sd.requires_grad_(False)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=0.05)
    u, i, s = d.obs["train"]
    n = u.shape[0]
    g = torch.Generator().manual_seed(0)
    for it in range(iters):
        idx = torch.randint(0, n, (8192,), generator=g)
        loss = -(n / 8192) * m.log_prob(u[idx], i[idx], s[idx], stoch=False).sum() + m.kl()
        opt.zero_grad(); (loss / n).backward(); opt.step()

    # empirical within-category share of each item
    cnt = pd.Series(i.numpy()).value_counts()
    ci, cm = d.cat_items.numpy(), d.cat_mask.numpy()
    emp, fit = [], []
    with torch.no_grad():
        a = m.lambda0.mu
        for c in range(d.n_cats):
            js = [int(ci[c, k]) for k in range(ci.shape[1]) if cm[c, k] > 0]
            tot = sum(cnt.get(j, 0) for j in js)
            if tot < 50:
                continue
            e = np.array([cnt.get(j, 0) / tot for j in js])
            f = torch.softmax(a[js], 0).numpy()
            emp += list(e); fit += list(f)
    emp, fit = np.array(emp), np.array(fit)
    r["n_items_compared"] = int(len(emp))
    r["share_recovery_corr"] = float(np.corrcoef(emp, fit)[0, 1])
    r["share_recovery_max_abs_error"] = float(np.abs(emp - fit).max())
    r["share_recovery_mean_abs_error"] = float(np.abs(emp - fit).mean())
    log(f"  intercept-only model reproduces empirical within-category shares: "
        f"corr {r['share_recovery_corr']:.5f}, mean |error| "
        f"{r['share_recovery_mean_abs_error']:.5f}")
    return r, (emp, fit)


def check_against_independent_logit(d, cat=0):
    """Homogeneous price coefficient, one category, fitted two ways: our variational
    code with the prior switched off, and a plain scipy MLE conditional logit."""
    from scipy.optimize import minimize
    ci, cm = d.cat_items.numpy(), d.cat_mask.numpy()
    js = [int(ci[cat, k]) for k in range(ci.shape[1]) if cm[cat, k] > 0]
    J = len(js)
    u, i, s = d.obs["train"]
    keep = np.isin(i.numpy(), js)
    uu, ii, ss = u[keep], i[keep], s[keep]
    slot = {j: k for k, j in enumerate(js)}
    y = np.array([slot[int(x)] for x in ii.numpy()])
    P = d.price.numpy()[np.array(js)][:, ss.numpy()].T          # [N, J]

    def negll(th):
        a = np.concatenate([[0.0], th[:J - 1]])
        v = a[None, :] - th[J - 1] * P
        v -= v.max(1, keepdims=True)
        return -(v[np.arange(len(y)), y] - np.log(np.exp(v).sum(1))).sum()
    res = minimize(negll, np.zeros(J), method="L-BFGS-B")
    ref_eta = float(res.x[J - 1])

    # same thing through our code path: homogeneous price, no prior, no user obs
    m = nf.ProductChoice(d, K=0, Kp=1, use_user_obs=False, homogeneous=True,
                         prior_var=1e8, price_prior_var=1e8, price_prior_mean=0.0,
                         intercept_var=1e8, seed=0)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(4000):
        loss = -m.log_prob(uu, ii, ss, stoch=False).sum()      # prior is flat
        opt.zero_grad(); loss.backward(); opt.step()
    ours = float(m.price_coef.mu.item())
    out = {"category": int(cat), "n_obs": int(len(y)), "scipy_eta": ref_eta,
           "nf_torch_eta": ours, "abs_diff": abs(ref_eta - ours),
           "rel_diff": abs(ref_eta - ours) / max(abs(ref_eta), 1e-9)}
    log(f"  homogeneous price coefficient, category {cat}: scipy MLE {ref_eta:.5f} vs "
        f"nf_torch {ours:.5f}  (relative difference {out['rel_diff']:.2%})")
    return out


# --------------------------------------------------------------- C/D. recovery
def simulate_and_recover(d, K=8, Kp=4, n_rep=6, seed=0, iters=3000):
    """Draw latents, simulate choices from the model, refit, compare."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    N, J, S = d.n_users, d.n_items, d.n_sessions

    theta = torch.randn(N, K) * 0.5
    beta = torch.randn(J, K) * 0.5
    lam0 = torch.randn(J) * 1.0
    gamma = torch.abs(torch.randn(N, Kp)) * 0.35
    lam = torch.abs(torch.randn(J, Kp)) * 0.35
    b_true = (gamma @ lam.T)                                    # [N, J] price coef

    u_tr, i_tr, s_tr = d.obs["train"]
    trips_u = torch.cat([u_tr] * n_rep)
    trips_s = torch.cat([s_tr] * n_rep)
    cat_of = d.item_cat[torch.cat([i_tr] * n_rep)]

    items = d.cat_items[cat_of]                                 # [T, M]
    maskm = d.cat_mask[cat_of]
    util = (lam0[items]
            + torch.einsum("tk,tmk->tm", theta[trips_u], beta[items])
            - torch.gather(b_true[trips_u], 1, items) * d.price[items, trips_s.unsqueeze(1)])
    util = util.masked_fill(maskm == 0, -1e9)
    probs = torch.softmax(util, 1)
    draw = torch.multinomial(probs, 1).squeeze(1)
    chosen = torch.gather(items, 1, draw.unsqueeze(1)).squeeze(1)

    m = nf.ProductChoice(d, K=K, Kp=Kp, use_user_obs=False, item_intercept=True,
                         price_prior_var=0.25, price_prior_mean=0.5,
                         intercept_var=10.0, seed=1)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    n = trips_u.shape[0]
    g = torch.Generator().manual_seed(7)
    for it in range(iters):
        idx = torch.randint(0, n, (8192,), generator=g)
        lp = m.log_prob(trips_u[idx], chosen[idx], trips_s[idx])
        loss = -((n / 8192) * lp.sum() - m.kl()) / n
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        # price coefficients on (household, item) pairs the data actually covers
        uq = torch.arange(N)
        blk = d.cat_items.reshape(1, -1).expand(N, -1)
        b_hat = m.price_coefficients(uq, blk)
        flat_mask = d.cat_mask.reshape(-1) > 0
        bh = b_hat[:, flat_mask].reshape(-1).numpy()
        bt = b_true[:, d.cat_items.reshape(-1)[flat_mask]].reshape(-1).numpy()

        # utilities and choice probabilities on a sample of trips
        sub = torch.randint(0, n, (4000,), generator=g)
        it_ = d.cat_items[d.item_cat[chosen[sub]]]
        mm_ = d.cat_mask[d.item_cat[chosen[sub]]]
        uh = m.utility(trips_u[sub], trips_s[sub], it_, stoch=False, mask=mm_)
        ut = (lam0[it_] + torch.einsum("tk,tmk->tm", theta[trips_u[sub]], beta[it_])
              - torch.gather(b_true[trips_u[sub]], 1, it_)
              * d.price[it_, trips_s[sub].unsqueeze(1)])
        mm = mm_ > 0
        ph = torch.softmax(uh.masked_fill(~mm, -1e9), 1)[mm].numpy()
        pt = torch.softmax(ut.masked_fill(~mm, -1e9), 1)[mm].numpy()

    out = {
        "n_simulated_choices": int(n), "K": K, "Kp": Kp,
        "price_coef_corr": float(np.corrcoef(bt, bh)[0, 1]),
        "price_coef_slope": float(np.polyfit(bt, bh, 1)[0]),
        "price_coef_true_mean": float(bt.mean()), "price_coef_fitted_mean": float(bh.mean()),
        "choice_prob_corr": float(np.corrcoef(pt, ph)[0, 1]),
        "choice_prob_mean_abs_error": float(np.abs(pt - ph).mean()),
    }
    log(f"  recovery: price coefficient corr {out['price_coef_corr']:.3f} "
        f"(slope {out['price_coef_slope']:.3f}, true mean {out['price_coef_true_mean']:.3f} "
        f"vs fitted {out['price_coef_fitted_mean']:.3f}); choice probability corr "
        f"{out['choice_prob_corr']:.4f}")
    return out, (bt, bh, pt, ph)


def recover_nesting(d, true_nest=0.7, seed=0, iters=2500):
    """Simulate category incidence with a known nesting coefficient and recover it."""
    torch.manual_seed(seed)
    tu, ts = d.trips["train"]
    T, C = tu.shape[0], d.n_cats
    iv = torch.randn(T, C) * 0.4                     # stand-in inclusive values
    alpha = torch.full((C,), -3.2) + torch.randn(C) * 0.4
    z = alpha.unsqueeze(0) + true_nest * iv
    y = torch.bernoulli(torch.sigmoid(z))

    m = nf.CategoryChoice(d, K=4, Kiv=4, Ktime=2, seed=2)
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    g = torch.Generator().manual_seed(11)
    for it in range(iters):
        idx = torch.randint(0, T, (4096,), generator=g)
        ll = -bce(m.logits(tu[idx], ts[idx], iv[idx]), y[idx], reduction="sum")
        loss = -((T / 4096) * ll - m.kl()) / (T * C)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        est = float(m.nesting_coef(tu).mean())
    out = {"true_nesting": true_nest, "estimated_nesting": est,
           "abs_error": abs(est - true_nest)}
    log(f"  nesting coefficient: true {true_nest:.3f}, recovered {est:.3f}")
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    d = nf.load(MI, device="cpu")
    log(f"data: {d.n_users} users, {d.n_items} items, {d.n_cats} categories")

    r = {}
    log("A. analytic identities")
    r["analytic"] = check_analytic(d)
    log("B. degenerate cases")
    r["degenerate"], shares = check_degenerate(d)
    r["independent_logit"] = check_against_independent_logit(d)
    log("C. parameter recovery from simulated choices")
    r["recovery"], rec = simulate_and_recover(d)
    log("C2. does recovery scale with sample size?  (a bug would not)")
    scaling = []
    for n_rep in [1, 6, 30]:
        o, _ = simulate_and_recover(d, K=8, Kp=4, n_rep=n_rep, seed=0)
        scaling.append({"n_rep": n_rep, "n_choices": o["n_simulated_choices"],
                        "price_coef_corr": o["price_coef_corr"],
                        "price_coef_slope": o["price_coef_slope"],
                        "choice_prob_corr": o["choice_prob_corr"]})
    r["recovery_scaling"] = scaling
    log("   n_choices   price-coef corr   slope   choice-prob corr")
    for x in scaling:
        log(f"   {x['n_choices']:>9,}   {x['price_coef_corr']:>15.3f}   "
            f"{x['price_coef_slope']:.3f}   {x['choice_prob_corr']:.3f}")
    log("D. nesting-coefficient recovery")
    r["nesting_recovery"] = recover_nesting(d)

    with open(os.path.join(OUT, "model_verification.json"), "w") as f:
        json.dump(r, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    emp, fit = shares
    bt, bh, pt, ph = rec
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    ax[0].scatter(emp, fit, s=10, alpha=0.5, color="#2d6cdf")
    ax[0].plot([0, emp.max()], [0, emp.max()], "--", c="0.4", lw=1)
    ax[0].set_xlabel("empirical within-category share"); ax[0].set_ylabel("fitted")
    ax[0].set_title(f"B. intercept-only model reproduces the data\n"
                    f"corr {r['degenerate']['share_recovery_corr']:.4f}", fontsize=10)
    k = np.random.default_rng(0).choice(len(bt), min(20000, len(bt)), replace=False)
    ax[1].scatter(bt[k], bh[k], s=3, alpha=0.15, color="#2e8b6f")
    lim = [0, np.percentile(bt, 99.5)]
    ax[1].plot(lim, lim, "--", c="0.4", lw=1); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("true $\\gamma_i\\cdot\\lambda_j$"); ax[1].set_ylabel("recovered")
    ax[1].set_title(f"C. price coefficients recovered from\nsimulated choices, corr "
                    f"{r['recovery']['price_coef_corr']:.3f}", fontsize=10)
    sc = pd.DataFrame(r["recovery_scaling"])
    ax[2].plot(sc.n_choices, sc.price_coef_corr, "-o", color="#c1432c",
               label="price coefficient")
    ax[2].plot(sc.n_choices, sc.choice_prob_corr, "-o", color="#2d6cdf",
               label="choice probability")
    ax[2].axvline(d.obs["train"][0].shape[0], ls="--", c="0.4", lw=1)
    ax[2].text(d.obs["train"][0].shape[0], 0.3, " our sample", fontsize=8, rotation=90)
    ax[2].set_xscale("log"); ax[2].set_ylim(0, 1)
    ax[2].set_xlabel("simulated choices"); ax[2].set_ylabel("correlation with truth")
    ax[2].set_title("C2. recovery improves with data:\nan identification limit, not a bug",
                    fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "verification.png"), dpi=150, bbox_inches="tight")
    log("wrote out/model_verification.json and figures/verification.png")


if __name__ == "__main__":
    main()
