"""
Stage 11 -- Placebo tests for price endogeneity (paper sec. 5).

The identifying assumption is that, conditional on a category-week control and a
weekday control, the Sunday->Monday price change is unrelated to anything else that
moves demand between those two days.  It is not testable directly, but it has a sharp
implication: if we relocate each price change to a week that in reality had *no*
price change, the relocated change should have no effect on demand.  Estimated price
coefficients should collapse towards zero and the p-values should be uniform.

Four tests per category, as in the paper:

    single-UPC forward / backward   one item's price series is shifted and given its
                                    own coefficient; the rest keep real prices
    all-items forward / backward    every item in the category is shifted

plus the unshifted fit, which should give a significantly negative coefficient.

Shift construction, following the paper: "we move each week with a price change
forward to the first week that had no price changes in the real data".  Here the
unit is the pair-week (Sunday of week w, Monday of week w+1) and a "change" is a
Sunday->Monday move of at least 1c.  Each change pair-week is swapped with the
nearest later (forward) or earlier (backward) pair-week that really had no change.
That preserves the set of price levels and the number of changes while destroying
the correspondence between when prices really moved and when the placebo says they
moved.

Model: the paper's baseline multinomial logit, fitted per category by maximum
likelihood, with the outside good normalised to zero utility:

    U_ijt = alpha_j + eta * price_jt + D_i . beta_j
            + g_w * (category purchase rate in that pair-week) + g_d * 1[Monday]
    U_i0t = 0

Standard errors are the usual inverse observed information.

Writes out/placebo_tests.csv, out/placebo_summary.json, figures/placebo.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
MI = os.path.join(HERE, "..", "..", "model_input")
OUT = os.path.join(HERE, "..", "..", "out")
FIG = os.path.join(HERE, "..", "..", "figures")

CHANGE_TOL = 0.01


def log(m):
    print(f"[11] {m}", flush=True)


# ------------------------------------------------------------------ price shifts
def rebuild_path(price_sun, price_mon, mode="forward", rng=None):
    """Relocate every price change to a pair-week that really had none.

    The real path is a step function: a level in force on a pair-week, and a set of
    pair-weeks at which the level steps to a new value between Sunday and Monday.
    The placebo keeps the *sequence of levels* and the *number of steps* and moves
    only the timing, so the whole path is rebuilt -- no pair-week keeps its real
    price by default.  Swapping pairs instead (the obvious first implementation)
    leaves every unswapped week at its real price and the test then has real price
    variation in it, which is exactly the contamination the paper warns about when
    it rejects a naive one-week shift.

    mode: "forward"  -> each step moves to the next pair-week that had no step
          "backward" -> to the previous one
          "random"   -> steps are scattered uniformly over the step-free weeks
                        (a fully decorrelated benchmark for what a null looks like)
    """
    T = len(price_sun)
    is_change = np.abs(price_mon - price_sun) >= CHANGE_TOL
    changes = np.where(is_change)[0]
    quiet = np.where(~is_change)[0]
    if len(changes) == 0 or len(quiet) == 0:
        return price_sun.copy(), price_mon.copy(), 0

    # levels: the level in force at the start, then the level after each step
    level0 = float(price_sun[0])
    new_levels = [float(price_mon[t]) for t in changes]

    # choose where the steps now happen
    chosen, used = [], set()
    seq = changes if mode != "backward" else changes[::-1]
    if mode == "random":
        pick = rng.choice(quiet, size=min(len(changes), len(quiet)), replace=False)
        chosen = sorted(int(x) for x in pick)
    else:
        for t in seq:
            cand = [q for q in quiet if (q > t if mode == "forward" else q < t)
                    and q not in used]
            if not cand:
                continue
            q = int(min(cand) if mode == "forward" else max(cand))
            used.add(q)
            chosen.append(q)
        chosen = sorted(chosen)

    if not chosen:
        return price_sun.copy(), price_mon.copy(), 0

    # walk the calendar and rebuild
    ps = np.empty(T); pm = np.empty(T)
    level = level0
    k = 0
    step_at = {t: i for i, t in enumerate(chosen)}
    for t in range(T):
        ps[t] = level
        if t in step_at:
            idx = step_at[t]
            level = new_levels[idx] if idx < len(new_levels) else level
            k += 1
        pm[t] = level
    return ps, pm, len(chosen)


def shift_diagnostics(real_s, real_m, plac_s, plac_m):
    """How much real price signal survives in the placebo series?

    Levels are demeaned *within item* first: pooling items whose price levels differ
    by dollars would put the correlation near 1 whatever the placebo did, and the
    question is whether an item's placebo price still tracks its own real price
    over time.
    """
    d_real = (real_m - real_s).ravel()
    d_plac = (plac_m - plac_s).ravel()
    lv_real = (real_s - real_s.mean(axis=1, keepdims=True)).ravel()
    lv_plac = (plac_s - plac_s.mean(axis=1, keepdims=True)).ravel()

    def corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    both = (np.abs(d_real) >= CHANGE_TOL) & (np.abs(d_plac) >= CHANGE_TOL)
    return {"corr_delta": corr(d_real, d_plac), "corr_level": corr(lv_real, lv_plac),
            "share_weeks_change_in_both": float(both.mean())}


# ------------------------------------------------------------------- the logit
def fit_logit(price_it, chosen, demo, pw_ctrl, monday, sess_pair, sess_is_mon,
              own_coef_item=None, ridge=1e-6, iters=200, week_fe=None, cluster=None):
    """Multinomial logit over J inside goods plus an outside good.

    price_it   [T, J]  price faced on that trip
    chosen     [T]     index of the chosen item, or J for "bought nothing"
    demo       [T, D]  household demographics
    pw_ctrl    [T]     category purchase rate in that pair-week
    monday     [T]     1 if the trip is on a Monday
    own_coef_item      if not None, that item column gets its own price coefficient
    """
    T, J = price_it.shape
    D = demo.shape[1]
    n_eta = 1 if own_coef_item is None else 2
    n_fe = 0 if week_fe is None else int(week_fe.max()) + 1

    def unpack(theta):
        i = 0
        alpha = theta[i:i + J]; i += J
        eta = theta[i:i + n_eta]; i += n_eta
        beta = theta[i:i + J * D].reshape(J, D); i += J * D
        gw = theta[i]; i += 1
        gd = theta[i]; i += 1
        fe = theta[i:i + n_fe]; i += n_fe
        return alpha, eta, beta, gw, gd, fe

    eta_map = torch.zeros(J, dtype=torch.long)
    if own_coef_item is not None:
        eta_map[own_coef_item] = 1

    def negll(theta):
        alpha, eta, beta, gw, gd, fe = unpack(theta)
        u = alpha.unsqueeze(0) + price_it * eta[eta_map].unsqueeze(0)
        u = u + demo @ beta.T
        u = u + (gw * pw_ctrl + gd * monday).unsqueeze(1)
        if n_fe:
            u = u + fe[week_fe].unsqueeze(1)
        u = torch.cat([u, torch.zeros(T, 1, dtype=u.dtype)], dim=1)
        return -(torch.log_softmax(u, dim=1)[torch.arange(T), chosen]).sum() \
            + ridge * (theta ** 2).sum()

    n_par = J + n_eta + J * D + 2 + n_fe
    theta = torch.zeros(n_par, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        theta[:J] = -3.0                      # inside goods are rare
    opt = torch.optim.LBFGS([theta], max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        loss = negll(theta)
        loss.backward()
        return loss
    opt.step(closure)

    th = theta.detach()
    H = torch.autograd.functional.hessian(negll, th)
    idx = J + (n_eta - 1)                      # the coefficient under test
    try:
        Hinv = torch.linalg.inv(H)
        se = float(torch.sqrt(torch.clamp(torch.diagonal(Hinv), min=0))[idx])
    except Exception:
        Hinv, se = None, float("nan")

    # Cluster-robust standard error.  Trips are not independent: the same household
    # appears dozens of times, and its taste for an item persists across trips.
    # Treating them as independent understates the variance of every coefficient and
    # is enough on its own to make a placebo look significant.
    se_cl = float("nan")
    if cluster is not None and Hinv is not None:
        with torch.no_grad():
            alpha, eta, beta, gw, gd, fe = unpack(th)
            u = alpha.unsqueeze(0) + price_it * eta[eta_map].unsqueeze(0)
            u = u + demo @ beta.T + (gw * pw_ctrl + gd * monday).unsqueeze(1)
            if n_fe:
                u = u + fe[week_fe].unsqueeze(1)
            u = torch.cat([u, torch.zeros(T, 1, dtype=u.dtype)], dim=1)
            P = torch.softmax(u, dim=1)[:, :J]
            Y = torch.zeros(T, J, dtype=u.dtype)
            inside = chosen < J
            Y[torch.arange(T)[inside], chosen[inside]] = 1.0
            R = Y - P                                        # [T, J]
            blocks = [R]                                     # alpha
            if n_eta == 1:
                blocks.append((R * price_it).sum(1, keepdim=True))
            else:
                m0 = (eta_map == 0)
                blocks.append(torch.stack([(R[:, m0] * price_it[:, m0]).sum(1),
                                           (R[:, ~m0] * price_it[:, ~m0]).sum(1)], 1))
            blocks.append((R.unsqueeze(2) * demo.unsqueeze(1)).reshape(T, J * D))
            rsum = R.sum(1, keepdim=True)
            blocks.append(rsum * pw_ctrl.unsqueeze(1))
            blocks.append(rsum * monday.unsqueeze(1))
            if n_fe:
                fe_blk = torch.zeros(T, n_fe, dtype=u.dtype)
                fe_blk[torch.arange(T), week_fe] = rsum.squeeze(1)
                blocks.append(fe_blk)
            S = torch.cat(blocks, dim=1)                     # [T, n_par]
            cl = torch.as_tensor(cluster, dtype=torch.long)
            G = int(cl.max()) + 1
            Sg = torch.zeros(G, S.shape[1], dtype=S.dtype)
            Sg.index_add_(0, cl, S)
            meat = Sg.T @ Sg
            V = Hinv @ meat @ Hinv
            adj = G / max(G - 1, 1)
            se_cl = float(torch.sqrt(torch.clamp(torch.diagonal(V)[idx] * adj, min=0)))
    return {"coef": float(th[idx]), "se_iid": se,
            "se": se_cl if np.isfinite(se_cl) else se,
            "se_clustered": se_cl,
            "negll": float(negll(th)), "converged": bool(torch.isfinite(th).all())}


# ------------------------------------------------------------------------ data
def build(args):
    items = pd.read_csv(os.path.join(MI, "id_maps", "items.csv"))
    sess = pd.read_csv(os.path.join(MI, "id_maps", "sessions.csv"))
    obs = pd.read_csv(os.path.join(MI, "id_maps", "observations.csv"))
    price = pd.read_csv(os.path.join(MI, "item_sess_price.tsv"), sep="\t", header=None,
                        names=["item_id", "session_id", "price"])
    ou = pd.read_csv(os.path.join(MI, "obsUser.tsv"), sep="\t", header=None)
    demo = ou.set_index(0)
    # a compact demographic vector, as in the paper (age, income, marital, homeowner)
    demo = demo[[1, 2, 6, 8]].rename(columns={1: "age", 2: "income", 6: "single", 8: "homeowner"})

    trips = obs[["user_id", "session_id", "pair_week", "weekday"]].drop_duplicates(
        ["user_id", "session_id"]).reset_index(drop=True)
    P = price.pivot(index="item_id", columns="session_id", values="price").sort_index()
    sess = sess.sort_values("session_id")
    sun = sess[sess.weekday == 0].set_index("pair_week").session_id
    mon = sess[sess.weekday == 1].set_index("pair_week").session_id
    pair_weeks = np.array(sorted(set(sun.index) & set(mon.index)))
    return items, sess, obs, trips, P, demo, sun, mon, pair_weeks


def run_category(cat, items, sess, obs, trips, P, demo, sun, mon, pair_weeks,
                 week_fe=True, seed=0):
    its = items[items.group_id == cat].sort_values("item_id")
    J = len(its)
    if J < 2:
        return []
    item_ids = its.item_id.values
    rng = np.random.default_rng(seed + int(cat))

    tr = trips.copy()
    ch = obs[obs.item_id.isin(item_ids)][["user_id", "session_id", "item_id"]]
    slot = {int(j): k for k, j in enumerate(item_ids)}
    tr = tr.merge(ch, on=["user_id", "session_id"], how="left")
    chosen = tr.item_id.map(slot).fillna(J).astype(int).values
    T = len(tr)

    rate = pd.Series(chosen < J).groupby(tr.pair_week.values).transform("mean").values
    monday = (tr.weekday == 1).astype(float).values
    Dm = demo.reindex(tr.user_id.values).fillna(0.0).values

    p_sun = P.loc[item_ids, sun.loc[pair_weeks].values].values      # [J, W]
    p_mon = P.loc[item_ids, mon.loc[pair_weeks].values].values
    pw_index = {w: i for i, w in enumerate(pair_weeks)}
    t_w = np.array([pw_index[w] for w in tr.pair_week.values])
    t_is_mon = monday.astype(bool)

    def price_matrix(ps, pm):
        return torch.tensor(np.where(t_is_mon[:, None], pm.T[t_w], ps.T[t_w]),
                            dtype=torch.float64)

    base = dict(chosen=torch.tensor(chosen, dtype=torch.long),
                demo=torch.tensor(Dm, dtype=torch.float64),
                pw_ctrl=torch.tensor(rate, dtype=torch.float64),
                monday=torch.tensor(monday, dtype=torch.float64),
                sess_pair=t_w, sess_is_mon=t_is_mon,
                week_fe=torch.tensor(t_w, dtype=torch.long) if week_fe else None,
                cluster=pd.factorize(tr.user_id.values)[0])

    n_changes = (np.abs(p_mon - p_sun) >= CHANGE_TOL).sum(1)
    focal = int(np.argmax(n_changes))

    out = []
    res = fit_logit(price_matrix(p_sun, p_mon), **base)
    out.append({"group_id": cat, "test": "actual (all items)", **res,
                "n_moved": 0, "corr_delta": 1.0, "corr_level": 1.0})
    res = fit_logit(price_matrix(p_sun, p_mon), own_coef_item=focal, **base)
    out.append({"group_id": cat, "test": "actual (single UPC)", **res,
                "n_moved": 0, "corr_delta": 1.0, "corr_level": 1.0})

    for mode in ["forward", "backward", "random"]:
        # single UPC relocated, with its own coefficient
        ps, pm = p_sun.copy(), p_mon.copy()
        ps[focal], pm[focal], nm = rebuild_path(p_sun[focal], p_mon[focal], mode, rng)
        dg = shift_diagnostics(p_sun[focal:focal + 1], p_mon[focal:focal + 1],
                               ps[focal:focal + 1], pm[focal:focal + 1])
        res = fit_logit(price_matrix(ps, pm), own_coef_item=focal, **base)
        out.append({"group_id": cat, "test": f"single UPC {mode}", **res,
                    "n_moved": nm, **dg})

        # every item relocated, one shared coefficient
        ps, pm = p_sun.copy(), p_mon.copy()
        tot = 0
        for k in range(J):
            ps[k], pm[k], nm = rebuild_path(p_sun[k], p_mon[k], mode, rng)
            tot += nm
        dg = shift_diagnostics(p_sun, p_mon, ps, pm)
        res = fit_logit(price_matrix(ps, pm), **base)
        out.append({"group_id": cat, "test": f"all items {mode}", **res,
                    "n_moved": tot, **dg})

    for o in out:
        o["t"] = o["coef"] / o["se"] if o["se"] and np.isfinite(o["se"]) and o["se"] > 0 else np.nan
        o["p"] = float(2 * stats.norm.sf(abs(o["t"]))) if np.isfinite(o["t"]) else np.nan
        o["t_iid"] = (o["coef"] / o["se_iid"]
                      if o["se_iid"] and np.isfinite(o["se_iid"]) and o["se_iid"] > 0 else np.nan)
        o["p_iid"] = (float(2 * stats.norm.sf(abs(o["t_iid"])))
                      if np.isfinite(o["t_iid"]) else np.nan)
        o["n_items"] = J
        o["n_trips"] = T
        o["n_purchases"] = int((chosen < J).sum())
    return out


def main(a):
    os.makedirs(FIG, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    items, sess, obs, trips, P, demo, sun, mon, pair_weeks = build(a)
    cats = sorted(items.group_id.unique())
    if a.limit:
        cats = cats[:a.limit]
    log(f"{len(cats)} categories, {len(trips):,} trips, {len(pair_weeks)} pair-weeks")

    rows = []
    for n, c in enumerate(cats, 1):
        rows += run_category(c, items, sess, obs, trips, P, demo, sun, mon, pair_weeks,
                             week_fe=not a.no_week_fe)
        if n % 5 == 0 or n == len(cats):
            log(f"  {n}/{len(cats)} categories fitted")
    df = pd.DataFrame(rows)
    names = pd.read_csv(os.path.join(MI, "id_maps", "categories.csv"))
    df = df.merge(names, on="group_id", how="left")
    df.to_csv(os.path.join(OUT, f"placebo_tests{a.tag}.csv"), index=False)

    # ------------------------------------------------------------------ summary
    summ = {}
    for test, g in df.groupby("test"):
        g = g[np.isfinite(g.p)]
        ks = stats.kstest(g.p, "uniform")
        summ[test] = {
            "categories": int(len(g)),
            "median_coef": float(g.coef.median()),
            "mean_coef": float(g.coef.mean()),
            "share_p_below_01": float((g.p < 0.01).mean()),
            "share_p_below_05": float((g.p < 0.05).mean()),
            "share_p_below_01_iid_se": float((g.p_iid < 0.01).mean()),
            "mean_se_inflation_from_clustering": float((g.se_clustered / g.se_iid).mean()),
            "share_negative_and_sig05": float(((g.p < 0.05) & (g.coef < 0)).mean()),
            "median_corr_delta_with_real": float(g.corr_delta.median()),
            "median_corr_level_with_real": float(g.corr_level.median()),
            "ks_stat_vs_uniform": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        }
    placebo_tests = [t for t in summ if not t.startswith("actual")]
    fails = df[df.test.isin(placebo_tests) & (df.p < 0.01)]
    n_fail = fails.group_id.nunique()
    summ["_overall"] = {
        "categories": int(df.group_id.nunique()),
        "categories_failing_any_placebo_at_1pct": int(n_fail),
        "categories_failing_backward_only": int(
            df[df.test.str.contains("backward") & (df.p < 0.01)].group_id.nunique()),
        "failing_category_names": sorted(fails.COMMODITY_DESC.dropna().unique().tolist()),
    }
    with open(os.path.join(OUT, f"placebo_summary{a.tag}.json"), "w") as f:
        json.dump(summ, f, indent=2)

    print("\n" + pd.DataFrame(summ).T.drop(index="_overall").round(4).to_string())
    print(f"\ncategories failing at least one placebo at 1%: {n_fail} of {df.group_id.nunique()}")
    print("  " + ", ".join(summ["_overall"]["failing_category_names"]))

    # ------------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["actual (all items)", "all items forward", "all items backward",
             "all items random", "single UPC forward", "single UPC random"]
    order = [o for o in order if o in set(df.test)]
    fig, axes = plt.subplots(2, len(order), figsize=(3.0 * len(order), 6.4), sharey="row")
    for k, test in enumerate(order):
        g = df[(df.test == test) & np.isfinite(df.p)]
        real = test.startswith("actual")
        col = "#c1432c" if real else "#2d6cdf"
        ax = axes[0, k]
        ax.hist(g.p, bins=np.linspace(0, 1, 11), color=col, alpha=0.85,
                edgecolor="white")
        ax.axhline(len(g) / 10, ls="--", c="0.35", lw=1)
        ax.set_title(test, fontsize=9)
        ax.set_xlabel("p-value")
        if k == 0:
            ax.set_ylabel("categories")
        ax.text(0.97, 0.94, f"KS p={summ[test]['ks_pvalue']:.2f}\n"
                            f"{summ[test]['share_p_below_01']:.0%} below 1%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax = axes[1, k]
        lim = np.nanpercentile(np.abs(df.coef), 97)
        ax.hist(np.clip(g.coef, -lim, lim), bins=21, color=col, alpha=0.85,
                edgecolor="white")
        ax.axvline(0, ls="--", c="0.35", lw=1)
        ax.set_xlabel("price coefficient")
        if k == 0:
            ax.set_ylabel("categories")
        ax.text(0.03, 0.94, f"median\n{g.coef.median():.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8)
    fig.suptitle("Placebo tests: real price series (red) versus price changes relocated "
                 "to weeks that had none (blue)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"placebo{a.tag}.png"), dpi=150, bbox_inches="tight")
    log(f"wrote out/placebo_tests{a.tag}.csv, out/placebo_summary{a.tag}.json, figures/placebo{a.tag}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="fit only the first N categories")
    p.add_argument("--no-week-fe", action="store_true",
                   help="use the paper's single pseudo-week covariate instead of full "
                        "pair-week fixed effects")
    p.add_argument("--tag", default="", help="suffix for the output files")
    main(p.parse_args())
