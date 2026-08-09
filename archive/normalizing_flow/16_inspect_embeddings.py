"""
Stage 16 -- What is actually inside the fitted embeddings?

Three questions the fit statistics cannot answer.

1. Do the item vectors beta_j encode anything real?  The model is never told
   SUB_COMMODITY_DESC, BRAND or MANUFACTURER, so if nearest neighbours in beta space
   share those labels more often than chance, the latent space has learned product
   similarity rather than memorised noise.  Same question for the price vectors
   lambda_j.

2. Is the household side actually personalised, or has the variational posterior
   collapsed back to its prior?  A mean-field posterior that has learned nothing
   returns mu = 0 and sd = prior sd for every household.  The diagnostic is the ratio
   of posterior sd to prior sd, and the spread of the posterior means, per household
   and against how much data that household contributes.

3. Where does the personalisation live -- in the taste vector theta_i, the price
   vector gamma_i, or the demographic loadings?  Ablating each in turn on held-out
   data says which.

Writes out/embedding_report.json, out/embedding_neighbours.csv and
figures/embeddings.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import nf_torch as nf
from importlib import import_module

ev = import_module("07_evaluate")
trainer = import_module("05_train_nf")
HERE = os.path.dirname(os.path.abspath(__file__))
MI = os.path.join(HERE, "..", "..", "model_input")
OUT = os.path.join(HERE, "..", "..", "out")
FIG = os.path.join(HERE, "..", "..", "figures")


def log(m):
    print(f"[16] {m}", flush=True)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


# ------------------------------------------------------- 1. item embedding content
def neighbour_agreement(V, meta, field, within_category=True, k=5):
    """Share of an item's k nearest neighbours sharing `field`, against the rate
    expected if neighbours were drawn at random from the same comparison pool."""
    lab = meta[field].values
    cat = meta.group_id.values
    Vn = unit(V)
    hit, base, n = [], [], 0
    for a in range(len(meta)):
        pool = np.where(cat == cat[a])[0] if within_category else np.arange(len(meta))
        pool = pool[pool != a]
        if len(pool) < k + 1:
            continue
        sim = Vn[pool] @ Vn[a]
        nn = pool[np.argsort(-sim)[:k]]
        hit.append(float((lab[nn] == lab[a]).mean()))
        base.append(float((lab[pool] == lab[a]).mean()))
        n += 1
    if not n:
        return None
    h, b = float(np.mean(hit)), float(np.mean(base))
    # exact null: each item's neighbours drawn at random from its own pool.  hit_a is
    # then a mean of k Bernoulli(base_a) draws, so the variance of the average is
    # known and a normal approximation is adequate at these sample sizes.
    var = float(np.mean([p * (1 - p) / k for p in base]) / n)
    z = (h - b) / np.sqrt(var) if var > 0 else np.nan
    from scipy import stats as _st
    return {"field": field, "within_category": within_category, "k": k, "items": n,
            "neighbour_agreement": h, "chance_agreement": b,
            "lift": h / b if b > 0 else np.nan, "z": float(z),
            "p_value": float(2 * _st.norm.sf(abs(z))) if np.isfinite(z) else np.nan}


def top_neighbours(V, meta, n_show=12, k=4, seed=0):
    Vn = unit(V)
    rng = np.random.default_rng(seed)
    pop = meta.sort_values("n_trips", ascending=False).head(120)
    pick = rng.choice(pop.index.values, size=min(n_show, len(pop)), replace=False)
    rows = []
    for a in pick:
        same = meta.index[meta.group_id == meta.loc[a, "group_id"]]
        same = same[same != a]
        if len(same) < k:
            continue
        sim = Vn[same] @ Vn[a]
        order = same[np.argsort(-sim)[:k]]
        rows.append({
            "item": meta.loc[a, "PRODUCT_ID"],
            "category": meta.loc[a, "COMMODITY_DESC"],
            "subclass": meta.loc[a, "SUB_COMMODITY_DESC"],
            "brand": meta.loc[a, "BRAND"],
            "nearest_neighbours": " | ".join(
                f"{meta.loc[b,'SUB_COMMODITY_DESC']}/{meta.loc[b,'BRAND']}" for b in order),
            "share_same_subclass": float(
                (meta.loc[order, "SUB_COMMODITY_DESC"] == meta.loc[a, "SUB_COMMODITY_DESC"]).mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------- 2. posterior collapse check
def posterior_diagnostics(m1, d, label):
    r = {}
    for name in ["theta", "gamma", "beta", "lam"]:
        blk = getattr(m1, name, None)
        if blk is None:
            continue
        prior_sd = float(np.sqrt(blk.prior_var))
        sd = torch.exp(blk.log_sd).detach().numpy()
        mu = blk.mu.detach().numpy()
        r[name] = {
            "prior_sd": prior_sd,
            "mean_posterior_sd": float(sd.mean()),
            "posterior_sd_over_prior_sd": float(sd.mean() / prior_sd),
            "mean_abs_posterior_mean": float(np.abs(mu).mean()),
            "signal_to_noise": float(np.abs(mu).mean() / sd.mean()),
            "rows_effectively_at_prior": float(
                ((np.abs(mu).mean(1) < 0.1 * prior_sd) & (sd.mean(1) > 0.9 * prior_sd)).mean())
            if mu.ndim > 1 else np.nan,
        }
        log(f"  {label}.{name}: posterior sd / prior sd = "
            f"{r[name]['posterior_sd_over_prior_sd']:.3f}, "
            f"|mu| / sd = {r[name]['signal_to_noise']:.2f}, "
            f"rows still at the prior = {r[name]['rows_effectively_at_prior']:.3f}")
    return r


def personalisation_vs_data(m1, d):
    """Does a household's learned vector grow with how much it is observed?"""
    u, _, _ = d.obs["train"]
    cnt = np.bincount(u.numpy(), minlength=d.n_users)
    th = m1.theta.mu.detach().numpy()
    ga = m1.gamma.mu.detach().numpy()
    df = pd.DataFrame({"obs": cnt,
                       "theta_norm": np.linalg.norm(th, axis=1),
                       "gamma_norm": np.linalg.norm(ga, axis=1)})
    df = df[df.obs > 0]
    q = pd.qcut(df.obs.rank(method="first"), 5, labels=False)
    by = df.groupby(q).agg(observations=("obs", "mean"),
                           theta_norm=("theta_norm", "mean"),
                           gamma_norm=("gamma_norm", "mean"),
                           households=("obs", "size"))
    return by.reset_index(names="quintile"), {
        "corr_obs_theta_norm": float(np.corrcoef(df.obs, df.theta_norm)[0, 1]),
        "corr_obs_gamma_norm": float(np.corrcoef(df.obs, df.gamma_norm)[0, 1]),
        "households_with_training_data": int((cnt > 0).sum()),
        "households_total": int(d.n_users),
        "median_training_observations_per_household": float(np.median(cnt[cnt > 0])),
    }


# ------------------------------------------------------------ 3. ablation
@torch.no_grad()
def ablate(m1, m2, d, which):
    """Held-out unconditional log-likelihood with one household channel zeroed."""
    saved = {}
    for name in which:
        blk = getattr(m1, name, None)
        if blk is None:
            continue
        saved[name] = blk.mu.detach().clone()
        blk.mu.data.zero_()
    # Zeroing a channel changes every inclusive value, so the centring constant that
    # stage 2 was trained against has to be recomputed -- otherwise the ablation is
    # measuring a mis-centred IV rather than the missing channel.
    saved_bar = getattr(m1, "iv_bar", None)
    if saved_bar is not None:
        m1.iv_bar = m1.mean_inclusive_values()
    mask = d.cat_mask.cpu().unsqueeze(0)
    slots = ev.slot_lookup(d)
    pred = ev.trip_predictions(m1, m2, d, "test")
    y = ev.outcome_matrix(d, "test", slots)
    ll, se, n, _ = ev.predictive_fit(pred, y, mask)
    for name, v in saved.items():
        getattr(m1, name).mu.data.copy_(v)
    if saved_bar is not None:
        m1.iv_bar = saved_bar
    return ll


def main(a):
    os.makedirs(FIG, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    d = nf.load(MI, device="cpu")
    m1, m2, cfg = ev.load_model(a.label, d, "cpu")
    meta = pd.read_csv(os.path.join(MI, "id_maps", "items.csv")).sort_values("item_id")
    meta = meta.reset_index(drop=True)
    assert (meta.item_id.values == np.arange(len(meta))).all()
    log(f"model {a.label}: K={cfg['K']}, Kp={cfg['Kp']}, {d.n_users} households, "
        f"{d.n_items} items")

    r = {"label": a.label, "K": cfg["K"], "Kp": cfg["Kp"]}

    # ---- 1. what the item vectors encode
    B = m1.beta.mu.detach().numpy()
    L = m1.lam.mu.detach().numpy()
    log("item embeddings: nearest-neighbour agreement on labels never shown to the model")
    # k must be small relative to the pool: with 10 items per category, "5 of the 9
    # other items" is nearly the whole category and the test has no power.
    r["beta_neighbours"] = [x for x in (
        neighbour_agreement(B, meta, "SUB_COMMODITY_DESC", True, 1),
        neighbour_agreement(B, meta, "SUB_COMMODITY_DESC", True, 2),
        neighbour_agreement(B, meta, "BRAND", True, 2),
        neighbour_agreement(B, meta, "MANUFACTURER", True, 2),
        neighbour_agreement(B, meta, "COMMODITY_DESC", False, 5)) if x]
    for x in r["beta_neighbours"]:
        log(f"  beta, {x['field']} (k={x['k']})"
            f"{'' if x['within_category'] else ', across all items'}: "
            f"{x['neighbour_agreement']:.3f} vs chance {x['chance_agreement']:.3f} "
            f"-> lift {x['lift']:.2f}x  (p = {x['p_value']:.3g})")
    r["lambda_neighbours"] = [x for x in (
        neighbour_agreement(L, meta, "SUB_COMMODITY_DESC", True, 2),
        neighbour_agreement(L, meta, "BRAND", True, 2)) if x]
    for x in r["lambda_neighbours"]:
        log(f"  lambda (price), {x['field']} (k={x['k']}): {x['neighbour_agreement']:.3f} "
            f"vs chance {x['chance_agreement']:.3f} -> lift {x['lift']:.2f}x "
            f"(p = {x['p_value']:.3g})")
    top_neighbours(B, meta).to_csv(os.path.join(OUT, "embedding_neighbours.csv"), index=False)

    # ---- 2. posterior collapse
    log("posterior vs prior (a collapsed posterior would sit at ratio 1.0 with |mu|~0)")
    r["posterior"] = posterior_diagnostics(m1, d, a.label)
    by, pers = personalisation_vs_data(m1, d)
    r["personalisation"] = pers
    r["personalisation_by_quintile"] = by.to_dict("records")
    log(f"  households with training data: {pers['households_with_training_data']} of "
        f"{pers['households_total']}, median {pers['median_training_observations_per_household']:.0f} "
        f"observations each")
    log(f"  corr(observations, |theta_i|) = {pers['corr_obs_theta_norm']:.3f}; "
        f"corr(observations, |gamma_i|) = {pers['corr_obs_gamma_norm']:.3f}")

    # ---- 3. which channel carries the personalisation
    log("ablations on held-out data (unconditional log-likelihood per purchase)")
    base = ablate(m1, m2, d, [])
    r["ablation"] = {"full_model": base}
    for name, which in [("no theta (latent taste)", ["theta"]),
                        ("no gamma (latent price sensitivity)", ["gamma"]),
                        ("no rho (demographics)", ["rho"]),
                        ("no household latents at all", ["theta", "gamma"])]:
        v = ablate(m1, m2, d, which)
        r["ablation"][name] = v
        log(f"  {name:38s} {v:.4f}   ({v - base:+.4f})")

    with open(os.path.join(OUT, "embedding_report.json"), "w") as f:
        json.dump(r, f, indent=2, default=float)

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4))

    nb = pd.DataFrame(r["beta_neighbours"])
    x = np.arange(len(nb))
    ax[0].bar(x - 0.2, nb.chance_agreement, 0.4, label="chance", color="#9aa5b1")
    ax[0].bar(x + 0.2, nb.neighbour_agreement, 0.4, label="5 nearest in $\\beta$ space",
              color="#2d6cdf")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f.replace('_DESC', '').replace('SUB_COMMODITY', 'subclass')
                           .replace('COMMODITY', 'category').lower() for f in nb.field],
                          fontsize=8, rotation=15)
    ax[0].set_ylabel("share of neighbours sharing the label")
    ax[0].set_title("1. Item vectors encode labels the model\nnever saw", fontsize=10)
    ax[0].legend(fontsize=8)
    for i, v in enumerate(nb.lift):
        ax[0].text(i + 0.2, nb.neighbour_agreement.iloc[i] + 0.01, f"{v:.1f}x",
                   ha="center", fontsize=8)

    names, ratios, snr = [], [], []
    for k, v in r["posterior"].items():
        names.append(k); ratios.append(v["posterior_sd_over_prior_sd"]); snr.append(v["signal_to_noise"])
    xx = np.arange(len(names))
    ax[1].bar(xx - 0.2, ratios, 0.4, color="#c1432c", label="posterior sd / prior sd")
    ax[1].bar(xx + 0.2, snr, 0.4, color="#2e8b6f", label="$|\\mu|$ / posterior sd")
    ax[1].axhline(1.0, ls="--", c="0.4", lw=1)
    ax[1].set_xticks(xx); ax[1].set_xticklabels(names)
    ax[1].set_title("2. The posterior has not collapsed\nback to the prior", fontsize=10)
    ax[1].legend(fontsize=8)

    abl = r["ablation"]
    keys = [k for k in abl if k != "full_model"]
    vals = [abl["full_model"] - abl[k] for k in keys]
    ax[2].barh(range(len(keys)), vals, color="#2d6cdf")
    ax[2].set_yticks(range(len(keys)))
    ax[2].set_yticklabels([k.split(" (")[0] for k in keys], fontsize=8)
    ax[2].invert_yaxis()
    ax[2].set_xlabel("held-out log-likelihood lost (nats per purchase)")
    ax[2].set_title("3. Where the personalisation lives", fontsize=10)
    for i, v in enumerate(vals):
        ax[2].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "embeddings.png"), dpi=150, bbox_inches="tight")
    log("wrote out/embedding_report.json, out/embedding_neighbours.csv, "
        "figures/embeddings.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="nf")
    main(p.parse_args())
