"""
Stage 28 -- What-if questions and basket generation from the nested model.

Stage 20 built a simulator on the flat model and had to admit two things: it could
only emit a contextual bandit (no state), and it could not generate a basket at all --
with no incidence layer there was nothing to say whether a household buys from a
category, only which item it would pick if it did.  The nested model fixes both, and
this script is what that buys.

Three things, in the order they have to be established.

1. IS THE PRICE RESPONSE REAL?  Refit on a scrambled price panel and check the
   coefficient collapses.  Same test as 26, now covering both price channels: the
   incidence/allocation coefficient gamma.beta and the quantity one q_gamma.q_beta.

2. WHERE DOES A PRICE CUT GO?  This is the question the flat model could not ask.
   With the nest, a 1% price cut on item j moves demand through three channels:

     allocation   j takes share from the rest of its category      d log pi_j
     incidence    the category becomes more attractive (via IV)    kappa * d IV
     quantity     buyers of j buy more units                       d log E[units]

   Total elasticity is the sum, and the split is what decides whether a promotion
   grows the category or just cannibalises it.  kappa is estimated, so the split is
   measured rather than assumed: kappa = 1 means IV cancels and the category does not
   expand at all beyond what the items sum to; kappa > 1 means it does.

3. CAN IT GENERATE A BASKET?  Roll the three layers forward -- incidence per
   category, then items, then units -- and compare generated baskets against held-out
   ones on size, category span, units per item and category mix.  A model that cannot
   reproduce the shape of a real basket has no business simulating a policy.

Writes out/nested_counterfactual.json and figures/nested_counterfactual.png.
"""
import argparse
import importlib
import json
import os

import numpy as np
import pandas as pd
import torch

nb = importlib.import_module("27_nested_basket")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "basket_input")
OUT = os.path.join(HERE, "..", "out")
FIG = os.path.join(HERE, "..", "figures")

PAL = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b",
       "green": "#2a9d8f", "amber": "#e9c46a"}


def log(m):
    print(f"[28] {m}", flush=True)


def load(label, d, dev):
    cfg = json.load(open(os.path.join(OUT, f"{label}_nested_history.json")))["config"]
    m = nb.NestedModel(d, K=cfg["K"], Kp=cfg["Kp"], Kt=cfg["Kt"], Ks=cfg["Ks"],
                       seed=cfg["seed"], use_nest=not cfg["no_nest"],
                       use_quantity=not cfg["no_quantity"],
                       use_store=not cfg["no_store"]).to(dev)
    m.load_state_dict(torch.load(os.path.join(OUT, f"{label}_nested.pt"),
                                 map_location=dev))
    m.eval()
    return m, cfg


@torch.no_grad()
def elasticity_decomposition(m, d, dev, n_trips=3000, seed=0):
    """Split the own-price elasticity into allocation, incidence and quantity.

    Evaluated on real held-out trips so the shares P(j|c) and the category mix are the
    ones the data actually produces, not a uniform average over the catalogue.
    """
    sp = d.splits["test"]
    rng = np.random.default_rng(seed)
    bsel = rng.choice(sp["n_baskets"], size=min(n_trips, sp["n_baskets"]), replace=False)
    rows = np.concatenate([np.arange(sp["starts"][i], sp["ends"][i]) for i in bsel])
    rows = rows[rng.random(len(rows)) < 0.5][:20000]
    user, item = sp["user"][rows], sp["item"][rows]
    day, week = sp["day"][rows], sp["week"][rows]
    store = sp["store"][rows]
    cat = sp["cat"][rows]

    ut = torch.as_tensor(user, device=dev)
    jt = torch.as_tensor(item, device=dev)
    # alpha_j . beta_j price coefficient for the chosen items
    alpha_p = (m.gamma[ut] * m.beta[jt]).sum(-1)                     # [n]
    kappa = m.kappa()[torch.as_tensor(cat, device=dev)] if m.use_nest else None

    # share of item j inside its category, at this trip
    ct = torch.as_tensor(cat, device=dev)
    blk, msk = d.cat_items[ct], d.cat_mask[ct].clone()
    Mx = blk.shape[1]
    blk_np = blk.cpu().numpy()
    day_r = np.repeat(day[:, None], Mx, 1)
    user_r = np.repeat(user[:, None], Mx, 1)
    store_r = np.repeat(store[:, None], Mx, 1)
    st = d.state(user_r.ravel(), blk_np.ravel(), day_r.ravel()).reshape(
        len(rows), Mx, nb.N_STATE_FEATURES)
    dlogp = d.log_price_dev[blk, torch.as_tensor(day_r, device=dev)]
    if m.use_store:
        msk = msk * d.carried[blk, torch.as_tensor(store_r, device=dev)].float()
    # Context zeroed: the elasticity is evaluated at the point of category choice, so
    # it is the pre-basket utility that matters.  This means the reported allocation
    # channel excludes any interaction effect -- a price cut that changes what else
    # lands in the basket is not counted here.  See NESTED_MODEL.md 9.
    u = m.item_utility(ut, blk, torch.zeros(len(rows), m.K, device=dev), dlogp,
                       torch.as_tensor(st, device=dev),
                       torch.as_tensor(week, device=dev),
                       torch.as_tensor(store, device=dev))
    u = u.masked_fill(msk == 0, -1e9)
    pi = torch.softmax(u, dim=1)
    # Padding slots hold item id 0, so a plain `blk == j` match also fires on every
    # pad slot whenever the chosen item happens to be item 0.  Restrict to real slots
    # and take the first hit, then gather by position.
    is_j = (blk == jt.unsqueeze(1)) & (msk > 0)
    pos = is_j.float().argmax(dim=1)
    pi_j = torch.gather(pi, 1, pos.unsqueeze(1)).squeeze(1).clamp(1e-6, 1 - 1e-6)
    dlogp_j = torch.gather(dlogp, 1, pos.unsqueeze(1)).squeeze(1)

    # d log pi_j / d log p_j = -alpha_j (1 - pi_j)          allocation
    e_alloc = -alpha_p * (1.0 - pi_j)
    # d IV / d log p_j = -alpha_j pi_j ; incidence multiplies by kappa
    e_inc = -(kappa * alpha_p * pi_j) if m.use_nest else torch.zeros_like(e_alloc)
    # quantity: d log E[units] / d log p_j
    if m.use_quantity:
        z = (m.q0[jt] - (m.q_gamma[ut] * m.q_beta[jt]).sum(-1) * dlogp_j).clamp(-6, 4)
        lam = torch.exp(z)
        aq = (m.q_gamma[ut] * m.q_beta[jt]).sum(-1)
        # E[units] = 1 + lam, so d log E / d log p = -aq * lam / (1 + lam)
        e_qty = -aq * lam / (1.0 + lam)
    else:
        e_qty = torch.zeros_like(e_alloc)
    tot = e_alloc + e_inc + e_qty
    # Medians do not decompose additively, so the shares are computed from means --
    # reporting median components against a median total does not sum to 100%.
    mean = lambda x: float(x.mean())
    f = lambda x: float(x.median())
    tm = mean(tot)
    return {
        "n": int(len(rows)),
        "allocation": mean(e_alloc), "incidence": mean(e_inc), "quantity": mean(e_qty),
        "total": tm, "median_total": f(tot),
        "share_allocation": mean(e_alloc) / tm if tm else float("nan"),
        "share_incidence": mean(e_inc) / tm if tm else float("nan"),
        "share_quantity": mean(e_qty) / tm if tm else float("nan"),
        "median_kappa": float(m.kappa().median()) if m.use_nest else float("nan"),
        "share_categories_kappa_above_1": float((m.kappa() > 1).float().mean())
        if m.use_nest else float("nan"),
    }


@torch.no_grad()
def generate(m, d, dev, n_trips=4000, seed=0, n_cat_eval=24):
    """Generate baskets and compare their shape with held-out real ones."""
    sp = d.splits["test"]
    rng = np.random.default_rng(seed)
    bsel = rng.choice(sp["n_baskets"], size=min(n_trips, sp["n_baskets"]), replace=False)
    real_sizes, real_cats, real_units = [], [], []
    for i in bsel:
        r = np.arange(sp["starts"][i], sp["ends"][i])
        real_sizes.append(len(r)); real_cats.append(len(set(sp["cat"][r].tolist())))
        real_units.append(float(sp["units"][r].sum()))

    gen_sizes, gen_cats, gen_units = [], [], []
    for a in range(0, len(bsel), 128):
        b = bsel[a:a + 128]
        first = sp["starts"][b]
        user, day = sp["user"][first], sp["day"][first]
        week, store = sp["week"][first], sp["store"][first]
        T = len(b)
        # sample a subset of categories to keep the cost bounded, then scale the
        # generated counts back up by C / n_cat_eval
        cats = rng.choice(d.C, size=(T, n_cat_eval), replace=True)
        n_items = np.zeros(T); n_c = np.zeros(T); n_u = np.zeros(T)
        for k in range(n_cat_eval):
            ct = torch.as_tensor(cats[:, k], device=dev)
            blk, msk = d.cat_items[ct], d.cat_mask[ct].clone()
            Mx = blk.shape[1]
            blk_np = blk.cpu().numpy()
            day_r = np.repeat(day[:, None], Mx, 1)
            user_r = np.repeat(user[:, None], Mx, 1)
            store_r = np.repeat(store[:, None], Mx, 1)
            st = d.state(user_r.ravel(), blk_np.ravel(), day_r.ravel()).reshape(
                T, Mx, nb.N_STATE_FEATURES)
            dlogp = d.log_price_dev[blk, torch.as_tensor(day_r, device=dev)]
            if m.use_store:
                msk = msk * d.carried[blk, torch.as_tensor(store_r, device=dev)].float()
            u = m.item_utility(torch.as_tensor(user, device=dev), blk,
                               # Context zeroed: baskets are generated category by
                               # category, so no basket exists yet to condition on.
                               # A sequential generator that built the basket
                               # incrementally could use it; this one cannot.
                               torch.zeros(T, m.K, device=dev), dlogp,
                               torch.as_tensor(st, device=dev),
                               torch.as_tensor(week, device=dev),
                               torch.as_tensor(store, device=dev))
            u = u.masked_fill(msk == 0, -1e9)
            iv = torch.logsumexp(u, dim=1)
            ok = msk.sum(1) > 0
            # same frozen per-category reference the model trained against
            iv = torch.where(ok, iv - m.iv_ref[ct], torch.zeros_like(iv))
            cst = d.state(user, blk_np[:, 0], day)
            lin = (m.c0[ct] + (m.c_user[torch.as_tensor(user, device=dev)] * m.c_cat[ct]).sum(-1)
                   + (m.c_state[ct] * torch.as_tensor(cst, device=dev)).sum(-1)
                   + m.kappa()[ct] * iv)
            # Recover the Poisson rate from the fitted Bernoulli head.  The model is
            # a Poisson-multinomial (see NESTED_MODEL.md 3): Q_ic ~ Poisson(Lambda)
            # allocated across items, and the incidence head fits P(Q > 0), so
            # Lambda = -log(1 - p).  Drawing one item per category instead -- as an
            # earlier version did -- forced generated items and categories to coincide,
            # which contradicts the 56%-of-baskets finding this rebuild is founded on.
            # Two separate draws, because they answer two separate questions and the
            # data only ever taught the model one of them per head:
            #   buy?     Bernoulli from the incidence head
            #   how many distinct items, GIVEN a purchase?  the breadth head
            # Recovering picks from P(buy) alone via -log(1-p) implies
            # E[picks | bought] of ~1.01-1.10 for realistic incidence rates, against a
            # real 1.284.  That single line was the whole generation shortfall.
            p_buy = torch.sigmoid(lin).clamp(1e-6, 1 - 1e-6)
            bought = (torch.rand(T, device=dev) < p_buy) & ok
            pi = torch.softmax(u, dim=1) * (msk > 0).float()
            pi = pi / pi.sum(1, keepdim=True).clamp_min(1e-9)
            if getattr(m, "use_breadth", False):
                pdev = ((dlogp * (msk > 0).float()).sum(1)
                        / (msk > 0).float().sum(1).clamp_min(1))
                zb = (m.b0[ct] + m.b_user[torch.as_tensor(user, device=dev)]
                      - m.b_price[ct] * pdev).clamp(-6, 3)
                distinct = (1.0 + torch.poisson(torch.exp(zb))) * bought.float()
            else:
                distinct = bought.float()
            # cap at the number of items the store actually stocks
            distinct = torch.minimum(distinct, (msk > 0).float().sum(1))
            Q = distinct
            # Units must come from the quantity head.  An earlier version accumulated
            # Q directly, which silently gave every purchased item exactly one unit --
            # generated units per item 1.007 against a real 1.348, even though the
            # head itself predicts 1.389.  The model was calibrated; the sampler
            # simply never called it.
            if m.use_quantity:
                # expected units per picked item, averaged over the category's items
                zj = (m.q0.unsqueeze(0).expand(T, -1).gather(1, blk)
                      - (m.q_gamma[torch.as_tensor(user, device=dev)].unsqueeze(1)
                         * m.q_beta[blk]).sum(-1) * dlogp).clamp(-6, 4)
                eu = (1.0 + torch.exp(zj))                      # E[units | picked]
                eu_bar = (eu * pi).sum(1)                       # weighted by choice prob
            else:
                eu_bar = torch.ones(T, device=dev)
            n_items += distinct.cpu().numpy()
            n_c += bought.float().cpu().numpy()
            n_u += (distinct * eu_bar).cpu().numpy()
        scale = d.C / n_cat_eval
        gen_sizes.extend(n_items * scale); gen_cats.extend(n_c * scale)
        gen_units.extend(n_u * scale)

    def stat(x):
        x = np.asarray(x, dtype=float)
        return {"mean": float(x.mean()), "median": float(np.median(x)),
                "p90": float(np.quantile(x, .9))}
    return {"real_items": stat(real_sizes), "generated_items": stat(gen_sizes),
            "real_categories": stat(real_cats), "generated_categories": stat(gen_cats),
            "real_units": stat(real_units), "generated_units": stat(gen_units),
            "trips": int(len(bsel))}


def main(a):
    os.makedirs(FIG, exist_ok=True)
    dev = torch.device(a.device)
    d = nb.NestedData(IN, device=dev)
    res = {}

    log("1. structural placebo and fitted price coefficients")
    for lb in a.labels:
        if not os.path.exists(os.path.join(OUT, f"{lb}_nested.pt")):
            log(f"   {lb}: no checkpoint, skipping"); continue
        m, cfg = load(lb, d, dev)
        h = json.load(open(os.path.join(OUT, f"{lb}_nested_history.json")))
        pc = float((m.gamma @ m.beta.T).median())
        qc = float((m.q_gamma @ m.q_beta.T).median()) if m.use_quantity else float("nan")
        res[lb] = {"placebo_price": cfg.get("placebo_price", "none"),
                   "nest": not cfg["no_nest"], "quantity": not cfg["no_quantity"],
                   "store": not cfg["no_store"],
                   "price_coef": pc, "quantity_price_coef": qc,
                   "median_kappa": float(m.kappa().median()) if m.use_nest else float("nan"),
                   "test_item": h["test_item"], "test_top1": h["test_top1"],
                   "test_quantity_nll": h["test_quantity_nll"],
                   "test_incidence_nll": h["test_incidence_nll"]}
        r = res[lb]
        log(f"   {lb:16s} placebo={r['placebo_price']:8s} price {pc:+.3f}  "
            f"qprice {qc:+.3f}  kappa {r['median_kappa']:.3f}  "
            f"item {r['test_item']:.4f}  top1 {r['test_top1']:.3f}")

    base = a.labels[0]
    if base in res:
        for lb in a.labels[1:]:
            if lb in res and res[lb]["placebo_price"] != "none":
                res[lb]["retained"] = res[lb]["price_coef"] / res[base]["price_coef"] \
                    if res[base]["price_coef"] else float("nan")
                log(f"   -> {lb} retains {res[lb]['retained']:.1%} of the price coefficient")

        m, _ = load(base, d, dev)
        log("")
        log("2. where does a price cut go?")
        dec = elasticity_decomposition(m, d, dev, seed=a.seed)
        res["decomposition"] = dec
        log(f"   median own-price elasticity {dec['total']:+.4f}, split as")
        log(f"     allocation (share within category)  {dec['allocation']:+.4f}")
        log(f"     incidence  (category expands)       {dec['incidence']:+.4f}  "
            f"({dec['share_incidence']:.0%})")
        log(f"     quantity   (units per buyer)        {dec['quantity']:+.4f}  "
            f"({dec['share_quantity']:.0%})")
        log(f"   median kappa {dec['median_kappa']:.3f}; "
            f"{dec['share_categories_kappa_above_1']:.0%} of categories above 1")

        log("")
        log("3. can it generate a basket?")
        gen = generate(m, d, dev, seed=a.seed)
        res["generation"] = gen
        for k in ["items", "categories", "units"]:
            rr, gg = gen[f"real_{k}"], gen[f"generated_{k}"]
            log(f"   {k:11s} real mean {rr['mean']:6.2f} median {rr['median']:5.1f}   "
                f"generated mean {gg['mean']:6.2f} median {gg['median']:5.1f}")

    with open(os.path.join(OUT, "nested_counterfactual.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    have = [l for l in a.labels if l in res]
    vals = [res[l]["price_coef"] for l in have]
    cols = [PAL["blue"] if res[l]["placebo_price"] == "none" else PAL["red"] for l in have]
    ax.barh(range(len(have)), vals, color=cols)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(have)))
    ax.set_yticklabels([f"{l}\n({res[l]['placebo_price']})" for l in have], fontsize=8)
    ax.set_xlabel("fitted price coefficient")
    ax.set_title("Structural placebo\nred = prices scrambled before fitting", fontsize=10)
    ax.grid(axis="x", alpha=.3)

    ax = axes[1]
    dec = res.get("decomposition")
    if dec:
        parts = ["allocation", "incidence", "quantity"]
        v = [dec[p] for p in parts]
        ax.bar(range(3), v, color=[PAL["blue"], PAL["green"], PAL["amber"]])
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(range(3)); ax.set_xticklabels(parts, fontsize=9)
        ax.set_ylabel("contribution to own-price elasticity")
        ax.set_title(f"Where a price cut goes\ntotal {dec['total']:+.3f}, "
                     f"kappa {dec['median_kappa']:.2f}", fontsize=10)
        for i, x in enumerate(v):
            ax.text(i, x, f"{x:+.3f}", ha="center",
                    va="bottom" if x >= 0 else "top", fontsize=9)
        ax.grid(axis="y", alpha=.3)

    ax = axes[2]
    gen = res.get("generation")
    if gen:
        ks = ["items", "categories", "units"]
        rv = [gen[f"real_{k}"]["mean"] for k in ks]
        gv = [gen[f"generated_{k}"]["mean"] for k in ks]
        x = np.arange(3)
        ax.bar(x - .2, rv, width=.38, color=PAL["grey"], label="real held-out")
        ax.bar(x + .2, gv, width=.38, color=PAL["blue"], label="generated")
        ax.set_xticks(x); ax.set_xticklabels(ks, fontsize=9)
        ax.set_ylabel("mean per basket")
        ax.set_title("Generated baskets vs real ones", fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)

    fig.suptitle("Nested basket model: is the price causal, where does it go, "
                 "and can it generate?", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "nested_counterfactual.png"), dpi=150,
                bbox_inches="tight")
    log("")
    log("wrote out/nested_counterfactual.json and figures/nested_counterfactual.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", default=["nested", "nested_pl"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
