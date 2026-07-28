"""
Stage 5 -- Fit the two stages of the Nested Factorization model.

    python 05_train_nf.py --label nf              # full model, paper spec
    python 05_train_nf.py --label logit --homogeneous   # no heterogeneity baseline
    python 05_train_nf.py --label nf_promo --extras display mailer coupon

Stage 1 is trained on the product-choice observations (a line per category a
household actually bought from).  Stage 2 is trained on every (trip, category)
cell, with the inclusive value from stage 1 plugged in where the price term sits.
Held-out selection follows the paper: hyperparameters are scored on validation
observations that fall in *price-change* weeks, not on the validation set at large.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import nf_torch as nf

HERE = os.path.dirname(os.path.abspath(__file__))
MI = os.path.join(HERE, "..", "model_input")
OUT = os.path.join(HERE, "..", "out")


def set_indir(path):
    """Point the module at a different model_input directory."""
    global MI
    MI = path


def log(m):
    print(f"[05] {m}", flush=True)


def pick_device(name):
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------- stage 1
def train_stage1(d, cfg, dev):
    m = nf.ProductChoice(d, K=cfg.K, Kp=cfg.Kp, use_user_obs=not cfg.no_user_obs,
                         use_item_obs=cfg.item_obs, extras=cfg.extras,
                         homogeneous=cfg.homogeneous, prior_var=cfg.prior_var,
                         intercept_var=cfg.intercept_var, price_prior_var=cfg.price_prior_var,
                         price_prior_mean=cfg.price_prior_mean,
                         scale_prior=not cfg.no_scale_prior,
                         pool_across_categories=not cfg.no_pool, seed=cfg.seed).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr)
    uu, ii, ss = d.obs["train"]
    n = uu.shape[0]
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)

    val_u, val_i, val_s = d.obs["validation"]
    price_week_mask = validation_price_mask(d, val_i, val_s)
    log(f"stage 1: {n:,} training observations, {int(price_week_mask.sum()):,} of "
        f"{val_i.shape[0]:,} validation observations in price-change weeks")

    best, best_it, hist = -np.inf, -1, []
    t0 = time.time()
    for it in range(1, cfg.iters + 1):
        idx = torch.randint(0, n, (cfg.batch,), generator=g).to(dev)
        lp = m.log_prob(uu[idx], ii[idx], ss[idx])
        elbo = (n / cfg.batch) * lp.sum() - m.kl()
        opt.zero_grad()
        (-elbo / n).backward()
        opt.step()
        if it % cfg.eval_every == 0 or it == cfg.iters:
            with torch.no_grad():
                vlp = batched_logprob(m, val_u, val_i, val_s)
                v_all = float(vlp.mean())
                v_price = float(vlp[price_week_mask].mean()) if bool(price_week_mask.any()) else np.nan
            hist.append({"iter": it, "elbo_per_obs": float((elbo / n).detach()),
                         "val_loglik": v_all, "val_loglik_price_weeks": v_price,
                         "secs": time.time() - t0})
            crit = v_price if not np.isnan(v_price) else v_all
            if crit > best:
                best, best_it = crit, it
                torch.save(m.state_dict(), os.path.join(OUT, f"{cfg.label}_stage1.pt"))
            with torch.no_grad():
                blk, _ = m.choice_block(val_u[:2048], val_i[:2048], val_s[:2048])
                bmean = float(m.price_coefficients(val_u[:2048], blk).mean())
            hist[-1]["mean_price_coef"] = bmean
            log(f"  it {it:6d}  elbo/obs {elbo/n: .4f}  val {v_all: .4f}  "
                f"val(price wks) {v_price: .4f}  b {bmean: .3f}"
                f"{'  *' if best_it == it else ''}")
    m.load_state_dict(torch.load(os.path.join(OUT, f"{cfg.label}_stage1.pt")))
    return m, hist


def batched_logprob(m, u, i, s, chunk=20000):
    out = []
    for a in range(0, u.shape[0], chunk):
        out.append(m.log_prob(u[a:a + chunk], i[a:a + chunk], s[a:a + chunk], stoch=False))
    return torch.cat(out)


def validation_price_mask(d, items, sessions):
    """True where the item's price moved across the Sunday/Monday boundary."""
    ev = pd.read_csv(os.path.join(MI, "events.csv"))
    sess = pd.read_csv(os.path.join(MI, "id_maps", "sessions.csv"))
    key = ev.loc[ev.own_price_change == 1, ["item_id", "pair_week"]]
    pw = sess.set_index("session_id").pair_week
    it = items.detach().cpu().numpy()
    pws = pw.reindex(sessions.detach().cpu().numpy()).values
    flag = pd.MultiIndex.from_arrays([it, pws]).isin(
        pd.MultiIndex.from_frame(key))
    return torch.as_tensor(flag, device=items.device)


# ------------------------------------------------------------------- stage 2
def build_stage2_cells(d, split):
    """Every (trip, category) cell with a 0/1 purchase outcome."""
    tu, ts = d.trips[split]
    C = d.n_cats
    users = tu.repeat_interleave(C)
    sessions = ts.repeat_interleave(C)
    cats = torch.arange(C, device=tu.device).repeat(tu.shape[0])

    ou, oi, os_ = d.obs[split]
    oc = d.item_cat[oi]
    trip_key = {}
    tk = (tu.to(torch.int64) * d.n_sessions + ts.to(torch.int64)).cpu().numpy()
    for pos, k in enumerate(tk):
        trip_key[int(k)] = pos
    ok = (ou.to(torch.int64) * d.n_sessions + os_.to(torch.int64)).cpu().numpy()
    rows = np.array([trip_key[int(k)] for k in ok], dtype=np.int64)
    y = torch.zeros(tu.shape[0] * C, device=tu.device)
    y[torch.as_tensor(rows, device=tu.device) * C + oc] = 1.0
    return users, sessions, cats, y, tu, ts


def train_stage2(d, m1, cfg, dev):
    log("stage 2: computing inclusive values from the stage-1 posterior means ...")
    iv = {}
    for split in ["train", "validation", "test"]:
        tu, ts = d.trips[split]
        iv[split] = m1.inclusive_values(tu, ts)
    if not cfg.no_center_iv:
        iv_bar = m1.mean_inclusive_values()
        for split in ["train", "validation", "test"]:
            tu, _ = d.trips[split]
            iv[split] = iv[split] - iv_bar[tu]
        log(f"  inclusive values centred by the household-category mean over all "
            f"sessions (sd of the remaining variation: {float(iv['train'].std()):.4f})")
    log(f"  IV computed for {sum(v.shape[0] for v in iv.values()):,} trips "
        f"x {d.n_cats} categories")

    m = nf.CategoryChoice(d, K=cfg.K2, Kiv=cfg.Kiv, Ktime=cfg.Ktime,
                          use_user_obs=cfg.cat_user_obs, homogeneous=cfg.homogeneous,
                          prior_var=cfg.prior_var, scale_prior=not cfg.no_scale_prior,
                          seed=cfg.seed).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=cfg.lr2)

    ytr = stage2_targets(d, "train")
    yva = stage2_targets(d, "validation")
    tu, ts = d.trips["train"]
    vu, vs = d.trips["validation"]
    n = tu.shape[0]
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    best, hist = -np.inf, []
    t0 = time.time()
    for it in range(1, cfg.iters2 + 1):
        idx = torch.randint(0, n, (cfg.batch2,), generator=g).to(dev)
        z = m.logits(tu[idx], ts[idx], iv["train"][idx])
        ll = -bce(z, ytr[idx], reduction="sum")
        elbo = (n / cfg.batch2) * ll - m.kl()
        opt.zero_grad()
        (-elbo / (n * d.n_cats)).backward()
        opt.step()
        if it % cfg.eval_every2 == 0 or it == cfg.iters2:
            with torch.no_grad():
                zv = m.logits(vu, vs, iv["validation"], stoch=False)
                v = float(-bce(zv, yva, reduction="mean"))
            hist.append({"iter": it, "elbo_per_cell": float((elbo / (n * d.n_cats)).detach()),
                         "val_loglik": v, "secs": time.time() - t0})
            if v > best:
                best = v
                torch.save(m.state_dict(), os.path.join(OUT, f"{cfg.label}_stage2.pt"))
            with torch.no_grad():
                nmean = float(m.nesting_coef(vu[:2048]).mean())
            hist[-1]["mean_nesting_coef"] = nmean
            log(f"  it {it:6d}  elbo/cell {elbo/(n*d.n_cats): .5f}  "
                f"val loglik/cell {v: .5f}  nest {nmean: .3f}")
    m.load_state_dict(torch.load(os.path.join(OUT, f"{cfg.label}_stage2.pt")))
    return m, iv, hist


def stage2_targets(d, split):
    """[T, C] 0/1 matrix: did this trip buy from this category."""
    tu, ts = d.trips[split]
    C, S = d.n_cats, d.n_sessions
    key = (tu.to(torch.int64) * S + ts.to(torch.int64)).cpu().numpy()
    pos = {int(k): i for i, k in enumerate(key)}
    ou, oi, os_ = d.obs[split]
    ok = (ou.to(torch.int64) * S + os_.to(torch.int64)).cpu().numpy()
    rows = np.fromiter((pos[int(k)] for k in ok), dtype=np.int64, count=len(ok))
    y = torch.zeros((tu.shape[0], C), device=tu.device)
    y[torch.as_tensor(rows, device=tu.device), d.item_cat[oi]] = 1.0
    return y


def main(cfg):
    os.makedirs(OUT, exist_ok=True)
    if getattr(cfg, "indir", None):
        set_indir(os.path.join(HERE, "..", cfg.indir))
    dev = pick_device(cfg.device)
    log(f"device: {dev}   label: {cfg.label}   extras: {list(cfg.extras)}   input: {MI}")
    d = nf.load(MI, device=dev, extras=cfg.extras)
    log(f"data: {d.n_users} users, {d.n_items} items, {d.n_cats} categories, "
        f"{d.n_sessions} sessions, {d.n_periods} pair-weeks")

    m1, h1 = train_stage1(d, cfg, dev)
    if cfg.stage1_only:
        with open(os.path.join(OUT, f"{cfg.label}_history.json"), "w") as f:
            json.dump({"config": vars(cfg), "stage1": h1, "stage2": []}, f, indent=2)
        log(f"stage-1 only; best validation (price weeks) "
            f"{max(h['val_loglik_price_weeks'] for h in h1): .4f}")
        return
    m2, iv, h2 = train_stage2(d, m1, cfg, dev)

    torch.save({"iv": {k: v.cpu() for k, v in iv.items()}}, os.path.join(OUT, f"{cfg.label}_iv.pt"))
    with open(os.path.join(OUT, f"{cfg.label}_history.json"), "w") as f:
        json.dump({"config": vars(cfg), "stage1": h1, "stage2": h2}, f, indent=2)
    log(f"saved {cfg.label}_stage1.pt / {cfg.label}_stage2.pt / {cfg.label}_iv.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="nf")
    p.add_argument("--indir", default="", help="model input directory (default model_input)")
    # cpu, not auto: on Apple silicon "auto" selects MPS, whose float32
    # reductions differ enough to move the reported log-likelihoods.  Pass
    # --device mps or --device cuda explicitly to opt in.
    p.add_argument("--device", default="cpu")
    # Product stage.  The paper selects K=80, Kp=20 on its own (much larger) panel;
    # 06_hyperparam_sweep.py selects K=40 here, and K=80 overfits this sample --
    # validation falls from -1.75 to -2.78 (see REPORT.md).  The defaults below are
    # therefore the *selected* configuration, so that a bare
    #     python3 05_train_nf.py
    # reproduces the numbers in the reports.  run_all.sh passes them explicitly too.
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--Kp", type=int, default=20)
    p.add_argument("--no-user-obs", action="store_true")
    p.add_argument("--item-obs", action="store_true")
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=100)
    # paper's selected category-stage hyperparameters: K=40, Kiv=40, time dim 10, lr 0.01
    p.add_argument("--K2", type=int, default=40)
    p.add_argument("--Kiv", type=int, default=40)
    p.add_argument("--Ktime", type=int, default=10)
    p.add_argument("--cat-user-obs", action="store_true")
    p.add_argument("--lr2", type=float, default=0.01)
    p.add_argument("--iters2", type=int, default=3000)
    p.add_argument("--batch2", type=int, default=2000)
    p.add_argument("--eval-every2", type=int, default=250)
    p.add_argument("--prior-var", type=float, default=1.0)
    p.add_argument("--homogeneous", action="store_true",
                   help="no latent heterogeneity: the paper's multinomial/nested logit baseline")
    p.add_argument("--extras", nargs="*", default=[],
                   choices=["display", "mailer", "coupon"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage1-only", action="store_true")
    p.add_argument("--no-pool", action="store_true",
                   help="give every (household, category) its own latent vectors, i.e. "
                        "estimate each category in isolation")
    p.add_argument("--price-prior-var", type=float, default=0.25)
    p.add_argument("--no-center-iv", action="store_true",
                   help="feed the raw inclusive value to stage 2 instead of the "
                        "household-category-centred one")
    p.add_argument("--price-prior-mean", type=float, default=0.5,
                   help="prior mean of the price coefficient gamma_i . lambda_j; the "
                        "bilinear term is stuck at zero if both factors start there")
    p.add_argument("--no-scale-prior", action="store_true",
                   help="use the paper's flat N(0,1) prior on every factor instead of "
                        "scaling it as 1/sqrt(K)")
    p.add_argument("--intercept-var", type=float, default=10.0,
                   help="prior variance for the item intercepts; pass 0 to fall back "
                        "to --prior-var")
    main(p.parse_args())
