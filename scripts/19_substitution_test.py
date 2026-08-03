"""
Stage 19 -- Does the substitution kernel learn real product similarity?

The paper's sec. 6.4.1 claim is that the model infers, without ever being told, that
products sharing a class substitute more strongly than products that do not.  On this
panel the paper's own specification fails that test: within-subclass cross elasticity
exceeds across-subclass in 62% of categories for `nf` against 58% for a homogeneous
logit, gaps +0.005 and +0.006.  That is not a modelling result, it is a coin flip --
and the reason is structural.  Stage 1 as written makes u_ijt depend on item j's own
price only, so at the household level d log P(j) / d p_k = alpha_k P(k) for every j.
Substitution is proportional to market share and to nothing else.  Only aggregation
over heterogeneous households can bend that, and here it barely does.

The kernel psi_j . psi_k gives item j's utility a direct response to competitor k's
price.  psi is estimated from price movements alone; SUB_COMMODITY_DESC, BRAND and
MANUFACTURER are never shown to any stage of the model.  So this script is a genuine
out-of-sample test of learned structure, not a restatement of an input.

Three questions:

  1. is psi_j . psi_k larger for item pairs sharing a sub-commodity, brand or
     manufacturer than for pairs that do not, comparing only *within* a category so
     category composition cannot drive it;
  2. does the model-implied cross elasticity now actually vary across j -- i.e. is
     IIA broken in the fitted model, not just in principle;
  3. with the price split, does the regular-price coefficient differ from the
     promotional one?  If it does, the pooled elasticity everyone quotes is a blend
     of two different behaviours, and mostly the promotional one (the EDA finds base
     prices move in ~7% of item-weeks against ~18% for promotion depth).

Writes out/substitution_test.json and figures/substitution_test.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import nf_torch as nf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
FIG = os.path.join(HERE, "..", "figures")
MI = os.path.join(HERE, "..", "model_input")


def log(m):
    print(f"[19] {m}", flush=True)


def load_model(label, d, dev):
    """Rebuild stage 1 from its checkpoint.

    05_train_nf.py saves a bare state_dict and puts the configuration in
    {label}_history.json, so the flags have to come from there -- building the
    module with the wrong Ks or price_split would silently load a different model.
    """
    cfg = json.load(open(os.path.join(OUT, f"{label}_history.json")))["config"]
    m = nf.ProductChoice(
        d, K=cfg["K"], Kp=cfg["Kp"], use_user_obs=not cfg["no_user_obs"],
        use_item_obs=cfg["item_obs"], extras=cfg["extras"],
        homogeneous=cfg["homogeneous"], prior_var=cfg["prior_var"],
        intercept_var=cfg.get("intercept_var"),
        price_prior_var=cfg.get("price_prior_var"),
        price_prior_mean=cfg.get("price_prior_mean", 0.5),
        scale_prior=not cfg.get("no_scale_prior", False),
        pool_across_categories=not cfg.get("no_pool", False),
        Ks=cfg.get("Ks", 0), sub_prior_var=cfg.get("sub_prior_var", 0.05),
        price_split=cfg.get("price_split", False)).to(dev)
    nf.load_stage1_state(m, os.path.join(OUT, f"{label}_stage1.pt"), dev)
    m.eval()
    return m, cfg


def pair_frame(items_meta, iid):
    """Every ordered within-category item pair, with its shared-attribute flags."""
    rows = []
    meta = items_meta.copy()
    meta["item_id"] = meta.PRODUCT_ID.map(iid)
    meta = meta.dropna(subset=["item_id"])
    meta["item_id"] = meta.item_id.astype(int)
    for cat, g in meta.groupby("COMMODITY_DESC"):
        ids = g.item_id.to_numpy()
        if len(ids) < 2:
            continue
        sub = g.set_index("item_id").SUB_COMMODITY_DESC.to_dict()
        brd = g.set_index("item_id").BRAND.to_dict()
        mfr = g.set_index("item_id").MANUFACTURER.to_dict()
        for j in ids:
            for k in ids:
                if j == k:
                    continue
                rows.append({"cat": cat, "j": j, "k": k,
                             "same_sub": int(sub[j] == sub[k]),
                             "same_brand": int(brd[j] == brd[k]),
                             "same_mfr": int(mfr[j] == mfr[k])})
    return pd.DataFrame(rows)


def within_category_gap(df, value, flag):
    """Mean `value` when `flag` is 1 minus when it is 0, averaged over categories.

    Averaging category by category matters: categories differ hugely in the scale of
    psi and in how many pairs they contribute, so pooling would let a handful of big
    categories decide the answer.
    """
    per = []
    for cat, g in df.groupby("cat"):
        a, b = g[g[flag] == 1][value], g[g[flag] == 0][value]
        if len(a) == 0 or len(b) == 0:
            continue
        per.append({"cat": cat, "inside": a.mean(), "outside": b.mean(),
                    "gap": a.mean() - b.mean(), "n_inside": len(a), "n_outside": len(b)})
    p = pd.DataFrame(per)
    if not len(p):
        return {"categories": 0}
    return {"categories": int(len(p)),
            "mean_inside": float(p.inside.mean()),
            "mean_outside": float(p.outside.mean()),
            "mean_gap": float(p.gap.mean()),
            "median_gap": float(p.gap.median()),
            "share_categories_higher_inside": float((p.gap > 0).mean()),
            "_per_cat": p}


def main(a):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    dev = torch.device(a.device)
    d = nf.load(MI, device=dev)
    items_meta = pd.read_parquet(os.path.join(DATA, "items.parquet"))
    iid = pd.read_csv(os.path.join(MI, "id_maps", "items.csv")).set_index("PRODUCT_ID").item_id

    pairs = pair_frame(items_meta, iid)
    log(f"{len(pairs):,} ordered within-category item pairs across "
        f"{pairs.cat.nunique()} categories")
    log(f"  share sharing a sub-commodity {pairs.same_sub.mean():.3f}, "
        f"brand {pairs.same_brand.mean():.3f}, manufacturer {pairs.same_mfr.mean():.3f}")

    res = {"n_pairs": int(len(pairs)),
           "share_same_sub": float(pairs.same_sub.mean())}

    per_cat_store = {}
    for label in a.labels:
        path = os.path.join(OUT, f"{label}_stage1.pt")
        if not os.path.exists(path):
            log(f"  {label}: no checkpoint, skipping")
            continue
        m, cfg = load_model(label, d, dev)
        Ks = getattr(m, "Ks", 0)
        r = {"Ks": int(Ks), "price_split": bool(getattr(m, "price_split", False))}

        # ---- 1. does psi encode similarity the model was never given?
        if Ks > 0:
            psi = m.psi.mu.detach()
            sim = (psi[pairs.j.to_numpy()] * psi[pairs.k.to_numpy()]).sum(-1).cpu().numpy()
            df = pairs.assign(sim=sim)
            for flag, name in [("same_sub", "sub_commodity"), ("same_brand", "brand"),
                               ("same_mfr", "manufacturer")]:
                g = within_category_gap(df, "sim", flag)
                per = g.pop("_per_cat", None)
                r[f"psi_similarity_{name}"] = g
                if flag == "same_sub" and per is not None:
                    per_cat_store[label] = per
                log(f"  {label:10s} psi.psi {name:13s}: inside {g.get('mean_inside', float('nan')):+.4f} "
                    f"outside {g.get('mean_outside', float('nan')):+.4f}  gap "
                    f"{g.get('mean_gap', float('nan')):+.4f}  higher inside in "
                    f"{g.get('share_categories_higher_inside', float('nan')):.1%} of categories")

            # ---- 2. is IIA actually broken in the *fitted* model?
            # d u_j / d log p_k = psi_j . psi_k for j != k.  Under IIA this is zero
            # for every j, so the spread across j is exactly the departure.
            sd_by_k = df.groupby(["cat", "k"]).sim.std()
            r["iia_departure"] = {
                "mean_sd_of_du_dlogp_across_j": float(sd_by_k.mean()),
                "median_abs_psi_dot": float(np.abs(df.sim).median()),
                "p90_abs_psi_dot": float(np.abs(df.sim).quantile(0.9)),
            }
            log(f"  {label:10s} IIA departure: sd of du_j/dlog p_k across j = "
                f"{sd_by_k.mean():.4f}  (IIA implies 0)")
        else:
            r["psi_similarity_sub_commodity"] = {"note": "no kernel; IIA by construction"}
            r["iia_departure"] = {"mean_sd_of_du_dlogp_across_j": 0.0}
            log(f"  {label:10s} no substitution kernel -- IIA holds by construction")

        # ---- 3. regular price vs promotional price
        if getattr(m, "price_split", False):
            ab = (m.gamma.mu.detach() @ m.lam.mu.detach().T)          # [N, J]
            ap = (m.gamma_promo.mu.detach() @ m.lam_promo.mu.detach().T)
            r["price_split_coefficients"] = {
                "median_alpha_base": float(ab.median()),
                "median_alpha_promo": float(ap.median()),
                "ratio_promo_over_base": float(ap.median() / ab.median())
                if float(ab.median()) != 0 else float("nan"),
                "corr_across_items": float(np.corrcoef(
                    ab.mean(0).cpu().numpy(), ap.mean(0).cpu().numpy())[0, 1]),
            }
            s = r["price_split_coefficients"]
            log(f"  {label:10s} price split: alpha_base {s['median_alpha_base']:.3f}  "
                f"alpha_promo {s['median_alpha_promo']:.3f}  "
                f"ratio {s['ratio_promo_over_base']:.2f}  "
                f"corr across items {s['corr_across_items']:+.3f}")
        res[label] = r

    with open(os.path.join(OUT, "substitution_test.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)

    # ------------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = [l for l in a.labels if l in res and res[l].get("Ks", 0) > 0]
    fig, axes = plt.subplots(1, 2 + int(bool(have)), figsize=(6 + 5 * (1 + bool(have)), 4.6))

    ax = axes[0]
    # Only models that *have* a kernel can answer this.  Showing 0.00 for the others
    # would read as "measured zero" when it is "not applicable" -- under IIA every
    # pair substitutes identically, so the inside/outside comparison is undefined.
    names, vals = [], []
    for l in a.labels:
        if l not in res or res[l].get("Ks", 0) == 0:
            continue
        g = res[l].get("psi_similarity_sub_commodity", {})
        names.append(l)
        vals.append(g.get("share_categories_higher_inside", 0.0))
    ax.barh(range(len(names)), vals,
            color=["#2d6cdf" if v > 0.5 else "#9aa5b1" for v in vals])
    ax.axvline(0.5, ls="--", c="k", lw=1, label="coin flip")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of categories where same-sub-commodity pairs\nsubstitute more")
    ax.set_title("Does the model learn product similarity\nit was never shown?\n"
                 "(kernel models only; IIA models cannot be asked)", fontsize=9)
    ax.legend(fontsize=8)
    for i, v in enumerate(vals):
        ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    names, vals = [], []
    for l in a.labels:
        if l not in res:
            continue
        names.append(l)
        vals.append(res[l].get("iia_departure", {}).get("mean_sd_of_du_dlogp_across_j", 0.0))
    ax.barh(range(len(names)), vals,
            color=["#2d6cdf" if v > 0 else "#9aa5b1" for v in vals])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("spread of $\\partial u_j/\\partial \\log p_k$ across $j$")
    ax.set_title("Is IIA actually broken?\n(the paper's stage 1 sits at exactly 0)", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.4f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    if have:
        ax = axes[2]
        l = have[-1]
        p = per_cat_store.get(l)
        if p is not None and len(p):
            p = p.sort_values("gap")
            ax.barh(range(len(p)), p.gap,
                    color=["#2d6cdf" if v > 0 else "#d1495b" for v in p.gap])
            ax.axvline(0, c="k", lw=1)
            ax.set_yticks([])
            ax.set_ylabel(f"{len(p)} categories")
            ax.set_xlabel("$\\psi_j\\!\\cdot\\!\\psi_k$ gap, same sub-commodity $-$ different")
            ax.set_title(f"Per category, {l}\nblue = learned the grouping", fontsize=10)
            ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Substitution kernel: does it recover structure the model never saw?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "substitution_test.png"), dpi=150, bbox_inches="tight")
    log("wrote out/substitution_test.json and figures/substitution_test.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+",
                   default=["nf", "nf_split", "nf_ks", "nf_sub"])
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
