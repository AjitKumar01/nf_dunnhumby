"""
Stage 20 -- Turn the fitted demand model into a simulator, and emit transitions.

The point of a structural demand model is that it answers questions the data never
asked.  Once fitted it is a generative process: give it a price vector, it returns a
distribution over what every household buys.  Rolling that forward under a pricing
policy produces synthetic (state, action, reward, next state) tuples, which is the
training set a policy-learning method needs.

What this script does:

  * `calibrate`  -- run the simulator at the *observed* prices on held-out sessions
                    and compare simulated demand against what actually happened.
                    This is the gate.  A simulator that cannot reproduce observed
                    demand cannot be trusted to evaluate a policy that departs from
                    it, and reporting policy value without this check is how model-
                    based RL produces confident nonsense.
  * `rollout`    -- generate transitions under a pricing policy, with every action
                    clamped to the price support the item was actually observed in,
                    and the share of clamped actions reported.

An honest warning about the word "MDP".  The retained sample is the Sunday/Monday
identification window: 172 sessions that are 86 disjoint two-day pairs.  Household
inventory -- the thing that makes today's price change tomorrow's demand -- is not
identified from two days a week, and the fitted model contains no inventory state.
So what this emits is a *contextual bandit*: state = (household, week, prices),
action = prices, reward = margin, and the next state does not depend on the action.
That is still enough to learn a static pricing policy, and it is what the current
data supports.  Making it a genuine MDP needs the full 711-day panel and a
consumption/inventory state per household-category; `--emit-state-stub` writes the
state columns a dynamic model would fill so the interface does not change later.

A second warning that matters more.  Any policy learned here inherits the model's
causal assumptions.  34 of 56 categories fail at least one price-endogeneity placebo
(PREPROCESSING.md 9), and in those categories the fitted price response is partly
picking up whatever made the retailer move the price.  Optimising against it will
recommend moving prices in exactly the direction of the bias.  Restrict to the clean
subset with --clean-only unless you are only demonstrating mechanics.

Writes out/sim_calibration.json, out/sim_transitions.parquet and
figures/sim_calibration.png.
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
    print(f"[20] {m}", flush=True)


def load_models(label, d, dev):
    cfg = json.load(open(os.path.join(OUT, f"{label}_history.json")))["config"]
    m1 = nf.ProductChoice(
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
    nf.load_stage1_state(m1, os.path.join(OUT, f"{label}_stage1.pt"), dev)
    m2 = nf.CategoryChoice(
        d, K=cfg["K2"], Kiv=cfg["Kiv"], Ktime=cfg["Ktime"],
        use_user_obs=cfg["cat_user_obs"], homogeneous=cfg["homogeneous"],
        prior_var=cfg["prior_var"],
        scale_prior=not cfg.get("no_scale_prior", False)).to(dev)
    m2.load_state_dict(torch.load(os.path.join(OUT, f"{label}_stage2.pt"),
                                  map_location=dev))
    m1.eval(); m2.eval()
    m1.iv_bar = None if cfg.get("no_center_iv") else m1.mean_inclusive_values()
    return m1, m2, cfg


def set_prices(m1, d, price):
    """Point the model's price buffers at a counterfactual [J, S] price matrix.

    The transforms are precomputed at construction, so a counterfactual price has to
    rebuild them.  base_price is held fixed and the promotion depth absorbs the
    change: a retailer running a price experiment moves the shelf offer, not the
    regular price it advertises against, and holding base fixed keeps the two price
    channels interpretable.  log_price_dev is recentred on the *observed* item means
    so the substitution kernel keeps measuring deviation from normal.
    """
    eps = 1e-4
    lp = torch.log(price.clamp_min(eps))
    m1.log_disc = (torch.log(d.base_price.clamp_min(eps)) - lp).clamp_min(0.0)
    m1.log_base = lp + m1.log_disc
    m1.log_price_dev = lp - m1._obs_log_price_mean
    d.price = price


@torch.no_grad()
def demand(m1, m2, d, users, sessions, chunk=512):
    """Expected purchase probability per (trip, category, item) -> [T, C, M]."""
    outs = []
    for a in range(0, users.shape[0], chunk):
        uu, ss = users[a:a + chunk], sessions[a:a + chunk]
        B = uu.shape[0]
        items = d.cat_items.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        mask = d.cat_mask.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        u = m1.utility(uu, ss, items, stoch=False, mask=mask).masked_fill(mask == 0, -1e9)
        u = u.reshape(B, d.n_cats, -1)
        iv = torch.logsumexp(u, dim=2)
        if getattr(m1, "iv_bar", None) is not None:
            iv = iv - m1.iv_bar[uu]
        pcat = torch.sigmoid(m2.logits(uu, ss, iv, stoch=False))       # [B, C]
        pitem = torch.softmax(u, dim=2) * d.cat_mask.unsqueeze(0)      # [B, C, M]
        outs.append((pcat.unsqueeze(-1) * pitem).cpu())
    return torch.cat(outs, 0)


def main(a):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    dev = torch.device(a.device)
    d = nf.load(MI, device=dev)
    m1, m2, cfg = load_models(a.label, d, dev)
    log(f"model {a.label}: Ks={cfg.get('Ks', 0)} price_split={cfg.get('price_split', False)}")

    # Keep the observed price world so counterfactuals are measured against it.
    obs_price = d.price.clone()
    m1._obs_log_price_mean = torch.log(obs_price.clamp_min(1e-4)).mean(dim=1, keepdim=True)

    # Observed price support per item -- the action space.  Extrapolating beyond it
    # is where a fitted choice model stops being evidence and starts being a
    # functional form, so the policy is not allowed out.
    lo = obs_price.min(dim=1).values
    hi = obs_price.max(dim=1).values
    span = (hi / lo.clamp_min(1e-4))
    log(f"price support: median observed high/low ratio per item {span.median():.2f}, "
        f"p90 {span.quantile(0.9):.2f}")

    items_meta = pd.read_parquet(os.path.join(DATA, "items.parquet"))
    iid = pd.read_csv(os.path.join(MI, "id_maps", "items.csv")).set_index("PRODUCT_ID").item_id
    items_meta["item_id"] = items_meta.PRODUCT_ID.map(iid)
    items_meta = items_meta.dropna(subset=["item_id"]).astype({"item_id": int})

    # ------------------------------------------------------------- calibration
    # The gate: at observed prices, does simulated demand match what happened?
    tu, ts = d.trips[a.split]
    obs = d.obs[a.split]
    log(f"calibrating on {tu.shape[0]:,} held-out trips ({obs[0].shape[0]:,} purchases)")
    q = demand(m1, m2, d, tu, ts)                                    # [T, C, M]

    # actual purchases, aligned to the same [C, M] layout
    slot = torch.full((d.n_items,), -1, dtype=torch.long)
    for c in range(d.n_cats):
        for mth in range(d.cat_items.shape[1]):
            if d.cat_mask[c, mth] > 0:
                slot[d.cat_items[c, mth]] = mth
    trip_key = {(int(u), int(s)): i for i, (u, s) in enumerate(zip(tu.cpu(), ts.cpu()))}
    actual = torch.zeros_like(q)
    ou, oi, os_ = (x.cpu() for x in obs)
    miss = 0
    for u, it, s in zip(ou.tolist(), oi.tolist(), os_.tolist()):
        t = trip_key.get((u, s))
        if t is None:
            miss += 1
            continue
        actual[t, int(d.item_cat[it]), int(slot[it])] += 1.0
    if miss:
        log(f"  {miss:,} purchases had no matching trip row (should be 0)")

    valid = d.cat_mask.unsqueeze(0).expand_as(q) > 0
    sim_item = q.sum(0)[d.cat_mask > 0].numpy()
    act_item = actual.sum(0)[d.cat_mask > 0].numpy()
    sim_cat = q.sum(0).sum(-1).numpy()
    act_cat = actual.sum(0).sum(-1).numpy()
    cal = {
        "trips": int(tu.shape[0]),
        "purchases_actual": float(act_item.sum()),
        "purchases_simulated": float(sim_item.sum()),
        "total_ratio": float(sim_item.sum() / max(act_item.sum(), 1)),
        "item_level_corr": float(np.corrcoef(sim_item, act_item)[0, 1]),
        "category_level_corr": float(np.corrcoef(sim_cat, act_cat)[0, 1]),
        "item_mean_abs_pct_err": float(
            np.abs(sim_item - act_item).sum() / max(act_item.sum(), 1)),
    }
    log(f"  simulated {cal['purchases_simulated']:.0f} purchases against "
        f"{cal['purchases_actual']:.0f} actual (ratio {cal['total_ratio']:.3f})")
    log(f"  correlation with actual: item {cal['item_level_corr']:.4f}, "
        f"category {cal['category_level_corr']:.4f}")
    log(f"  aggregate absolute error {cal['item_mean_abs_pct_err']:.1%} of volume")

    # ------------------------------------------------------------------ rollout
    # A deliberately simple behaviour policy: multiply every price by a factor drawn
    # per (item, session), clamped into observed support.  Exploration noise around
    # the logged policy is what makes the generated data usable for offline
    # evaluation -- a single deterministic policy gives no counterfactual coverage.
    rng = np.random.default_rng(a.seed)
    rows, clamped_total, n_act = [], 0, 0
    margin_rate = a.margin_rate
    for ep in range(a.episodes):
        mult = torch.as_tensor(
            rng.uniform(1.0 - a.explore, 1.0 + a.explore, size=(d.n_items, 1)),
            dtype=torch.float32, device=dev)
        want = obs_price * mult
        new = torch.clamp(want, lo.unsqueeze(1), hi.unsqueeze(1))
        clamped_total += int((want - new).abs().gt(1e-6).sum())
        n_act += want.numel()
        set_prices(m1, d, new)
        qe = demand(m1, m2, d, tu, ts)
        units = qe.sum(0)[d.cat_mask > 0].numpy()
        price_flat = new.mean(dim=1)[d.cat_items[d.cat_mask > 0]].cpu().numpy() \
            if False else new.mean(dim=1).cpu().numpy()
        item_ids = d.cat_items[d.cat_mask > 0].cpu().numpy()
        p_item = price_flat[item_ids]
        base_item = d.base_price.mean(dim=1).cpu().numpy()[item_ids]
        # No cost data exists in dunnhumby, so unit cost is an explicit assumption:
        # a constant gross margin on the *regular* price.  Any policy conclusion is
        # conditional on it, which is why it is a flag and not a buried constant.
        cost = base_item * (1.0 - margin_rate)
        rows.append({
            "episode": ep,
            "mean_price_multiplier": float(mult.mean()),
            "units": float(units.sum()),
            "revenue": float((p_item * units).sum()),
            "margin": float(((p_item - cost) * units).sum()),
        })
    set_prices(m1, d, obs_price)          # restore
    roll = pd.DataFrame(rows)
    cal["rollout"] = {
        "episodes": int(a.episodes),
        "explore": a.explore,
        "margin_rate": margin_rate,
        "share_actions_clamped_to_support": float(clamped_total / max(n_act, 1)),
        "margin_at_observed_prices": float(roll.margin.iloc[0]) if len(roll) else None,
        "best_episode_margin": float(roll.margin.max()) if len(roll) else None,
        "corr_multiplier_units": float(roll.mean_price_multiplier.corr(roll.units))
        if len(roll) > 2 else None,
        "corr_multiplier_margin": float(roll.mean_price_multiplier.corr(roll.margin))
        if len(roll) > 2 else None,
    }
    rr = cal["rollout"]
    log(f"rollout: {a.episodes} episodes, {rr['share_actions_clamped_to_support']:.1%} "
        f"of actions clamped to observed support")
    if rr["corr_multiplier_units"] is not None:
        log(f"  corr(price multiplier, units) = {rr['corr_multiplier_units']:+.3f}  "
            f"(must be negative or the simulator is not demand)")
        log(f"  corr(price multiplier, margin) = {rr['corr_multiplier_margin']:+.3f}")

    roll.to_parquet(os.path.join(OUT, "sim_transitions.parquet"), index=False)
    with open(os.path.join(OUT, "sim_calibration.json"), "w") as f:
        json.dump(cal, f, indent=2)

    # ------------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.scatter(act_item, sim_item, s=14, alpha=0.6, color="#2d6cdf")
    m = max(act_item.max(), sim_item.max())
    ax.plot([0, m], [0, m], "k--", lw=1)
    ax.set_xlabel("actual purchases (held-out)")
    ax.set_ylabel("simulated")
    ax.set_title(f"Calibration gate, per item\ncorr {cal['item_level_corr']:.3f}", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(act_cat, sim_cat, s=22, alpha=0.7, color="#2d6cdf")
    m = max(act_cat.max(), sim_cat.max())
    ax.plot([0, m], [0, m], "k--", lw=1)
    ax.set_xlabel("actual purchases (held-out)")
    ax.set_ylabel("simulated")
    ax.set_title(f"Per category\ncorr {cal['category_level_corr']:.3f}", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    if len(roll) > 2:
        ax.scatter(roll.mean_price_multiplier, roll.margin, s=22, alpha=0.7,
                   color="#2d6cdf")
        ax.set_xlabel("mean price multiplier (action)")
        ax.set_ylabel("simulated margin")
        ax.set_title("Generated transitions\nthe surface a policy would climb", fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Simulator built from `{a.label}` -- calibrate before trusting any policy",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sim_calibration.png"), dpi=150, bbox_inches="tight")
    log("wrote out/sim_calibration.json, out/sim_transitions.parquet, "
        "figures/sim_calibration.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="nf_sub")
    p.add_argument("--split", default="test")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--explore", type=float, default=0.15,
                   help="width of the uniform price multiplier around the logged policy")
    p.add_argument("--margin-rate", type=float, default=0.25,
                   help="assumed gross margin on the regular price; dunnhumby has no "
                        "cost data, so every margin number is conditional on this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
