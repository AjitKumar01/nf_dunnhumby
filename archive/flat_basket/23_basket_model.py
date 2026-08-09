"""
Stage 23 -- A basket model with product interactions, household state and price.

Structure.  A basket is a set, not a single choice.  For household i shopping on day
t, each item j in the basket is scored against the whole catalogue:

    s_ijt = lambda_j                      how popular the item is at all
          + theta_i . alpha_j             household taste x item embedding
          + rho_j   . alpha_bar(context)  how item j responds to the rest of the basket
          - (gamma_i . beta_j) * dlogp_jt price, deviation from the item's own normal
          + eta_j   . state_ijt           how long since this household last bought
                                          something from this sub-commodity

    P(j | i, t, context) = softmax over {j} union {negatives}

Three things this buys that the paper's model cannot express.

1. Multiple items per category.  There is no within-category softmax and no unit
   demand assumption, so nothing has to be thrown away: 132 of 307 categories fail
   the paper's 15% multi-item cutoff, and 56% of baskets contain more than one item
   from some category.

2. Interaction.  `rho_j . alpha_bar(context)` makes item j's attractiveness depend on
   what is already in the basket.  Positive means complement, negative means the
   shopper has already satisfied that need.  This is a genuine product-to-product
   channel, not the market-share proportionality the within-category softmax implies.

3. State.  `eta_j . state_ijt` gives the model a memory.  The repurchase hazard varies
   4.3x with recency (0.149 at 7-14 days against 0.035 after 84), and a model without
   state must attribute all of that to a fixed household taste.

Why the embedding should now be meaningful.  alpha_j is pushed together for items
that appear in the same baskets and are bought by the same households.  Items in one
sub-commodity are 7.88x more likely to share a basket than chance, so the gradient
has something real to fit -- which is exactly what was missing when the model only
ever saw one item per category and alpha_j could only encode market share.

Negative sampling follows the usual unigram^0.75 draw.  Batching is by *basket*, not
by row, so the context mean is computed exactly rather than from a stale copy.

Writes out/<label>_basket.pt and out/<label>_basket_history.json.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "..", "basket_input")
OUT = os.path.join(HERE, "..", "..", "out")

N_STATE_FEATURES = 4


def log(m):
    print(f"[23] {m}", flush=True)


class BasketData:
    """Arrays plus the vectorised state lookup described in 22_basket_data.py."""

    def __init__(self, indir, device="cpu", placebo_price="none",
                 placebo_seed=0):
        self.meta = json.load(open(os.path.join(indir, "meta.json")))
        bk = pd.read_parquet(os.path.join(indir, "baskets.parquet"))
        self.items = pd.read_parquet(os.path.join(indir, "items.parquet"))
        lpd = np.load(os.path.join(indir, "log_price_dev.npy"))
        # A structural placebo: scramble the price panel before fitting.  If the
        # fitted price coefficient survives a scrambled panel, it was never
        # measuring price.  "permute" reorders each item's own days, keeping its
        # exact price distribution; "swap" hands each item another item's series.
        if placebo_price != "none":
            rng = np.random.default_rng(placebo_seed)
            if placebo_price == "permute":
                for j in range(lpd.shape[0]):
                    lpd[j] = rng.permutation(lpd[j])
            elif placebo_price == "swap":
                lpd = lpd[rng.permutation(lpd.shape[0])]
            else:
                raise ValueError(f"unknown placebo_price {placebo_price!r}")
            log(f"PLACEBO: price panel scrambled with '{placebo_price}' -- any "
                f"surviving price coefficient is spurious by construction")
        self.log_price_dev = torch.as_tensor(lpd, device=device)
        z = np.load(os.path.join(indir, "state.npz"))
        self.keys = z["keys"]
        self.sub_gap = z["sub_gap"].astype(np.float32)
        self.item_sub = z["item_sub"].astype(np.int64)
        self.stride = int(self.meta["day_stride"])
        self.S = int(self.meta["n_subs"])
        self.J = int(self.meta["n_items"])
        self.N = int(self.meta["n_users"])
        self.device = device

        # Rows sorted by basket so a basket's items are contiguous; then an index of
        # where each basket starts, which is what lets a batch be a set of baskets.
        bk = bk.sort_values(["split", "BASKET_ID", "item_id"]).reset_index(drop=True)
        self.splits = {}
        for sp, g in bk.groupby("split", sort=False):
            g = g.reset_index(drop=True)
            codes, _ = pd.factorize(g.BASKET_ID)
            starts = np.flatnonzero(np.r_[True, np.diff(codes) != 0])
            ends = np.r_[starts[1:], len(g)]
            self.splits[sp] = {
                "user": g.user_id.to_numpy(np.int64),
                "item": g.item_id.to_numpy(np.int64),
                "day": g.DAY.to_numpy(np.int64),
                # week-of-year, so seasonality generalises to held-out weeks
                "week": (g.WEEK_NO.to_numpy(np.int64) - 1) % 52,
                "starts": starts.astype(np.int64),
                "ends": ends.astype(np.int64),
                "n_baskets": len(starts),
                "n_rows": len(g),
            }
        # Negative-sampling distribution: unigram^0.75 over training purchases.
        cnt = np.bincount(self.splits["train"]["item"], minlength=self.J).astype(np.float64)
        p = np.power(np.maximum(cnt, 1.0), 0.75)
        self.neg_p = p / p.sum()
        log(f"data: {self.N:,} households, {self.J:,} items, {self.S:,} sub-commodities; "
            + ", ".join(f"{k} {v['n_rows']:,} rows / {v['n_baskets']:,} baskets"
                        for k, v in self.splits.items()))

    def state(self, user, item, day):
        """State features for aligned arrays of (user, item, day).

        Returns [n, 4]: never-bought flag, a fast decay, a decay on the
        sub-commodity's own repurchase cadence, and a slow log term.  Together they
        can represent the humped hazard the EDA found -- low right after a purchase,
        peaking around a week or two, decaying after.
        """
        sub = self.item_sub[item]
        group = user.astype(np.int64) * self.S + sub
        key = group * self.stride + day
        idx = np.searchsorted(self.keys, key, side="left")
        prev = idx - 1
        ok = prev >= 0
        prev_clipped = np.clip(prev, 0, len(self.keys) - 1)
        prev_key = self.keys[prev_clipped]
        # The previous key must belong to the same (household, sub-commodity) group,
        # otherwise it is some other household's purchase sitting next to it.
        same = ok & ((prev_key // self.stride) == group)
        since = np.where(same, day - (prev_key % self.stride), 0).astype(np.float32)
        gap = self.sub_gap[sub]
        never = (~same).astype(np.float32)
        f = np.stack([
            never,
            np.where(same, np.exp(-since / 7.0), 0.0),
            np.where(same, np.exp(-since / gap), 0.0),
            np.where(same, np.log1p(since) / np.log(100.0), 0.0),
        ], axis=1).astype(np.float32)
        return f


class BasketModel(nn.Module):
    def __init__(self, data: BasketData, K=32, Kp=8, use_context=True,
                 use_state=True, use_price=True, use_taste=True, seed=0,
                 tie_context=False, Kt=8, n_weeks=1):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        J, N = data.J, data.N
        self.K, self.Kp = K, Kp
        self.use_context, self.use_state = use_context, use_state
        self.use_price, self.use_taste = use_price, use_taste
        self.tie_context = tie_context

        def emb(n, k, sd=0.05):
            return nn.Parameter(torch.randn(n, k, generator=g) * sd)

        self.lam = nn.Parameter(torch.zeros(J))
        self.alpha = emb(J, K)                      # item embedding -- the thing tested
        self.theta = emb(N, K) if use_taste else None
        # tie_context makes the interaction symmetric: rho_j = alpha_j, so the
        # context term is alpha_j . alpha_bar(basket).  With a free rho the
        # co-purchase signal lands in rho and alpha never has to represent it -- which
        # is exactly what happened on the first fit, where ablating the context term
        # *improved* sub-commodity recovery (0.117 against 0.058).  Tying forces the
        # 7.88x within-sub-commodity co-occurrence found in the EDA to be expressed
        # in alpha itself, which is the embedding the requirement is about.
        self.rho = None if (tie_context or not use_context) else emb(J, K)
        self.gamma = emb(N, Kp, 0.1) if use_price else None
        self.beta = emb(J, Kp, 0.1) if use_price else None
        self.eta = emb(J, N_STATE_FEATURES, 0.01) if use_state else None
        # mu_j . delta_w: seasonality.  Without it the price coefficient absorbs any
        # week-frequency co-movement of price and demand -- the placebo battery
        # measures that at 11.3% of the coefficient (25_basket_placebo.py).
        self.mu = emb(J, Kt, 0.02) if Kt > 0 else None
        self.delta = emb(n_weeks, Kt, 0.02) if Kt > 0 else None

    def score(self, users, items, ctx, dlogp, state, weeks=None):
        """users [B], items [B, M], ctx [B, K], dlogp [B, M], state [B, M, F]."""
        s = self.lam[items]
        a = self.alpha[items]                                   # [B, M, K]
        if self.theta is not None:
            s = s + torch.einsum("bk,bmk->bm", self.theta[users], a)
        if self.rho is not None:
            # rho_j . alpha_bar(context): item j's response to the rest of the basket
            s = s + (self.rho[items] * ctx.unsqueeze(1)).sum(-1)
        elif self.use_context and self.tie_context:
            # symmetric version: alpha_j . alpha_bar(context)
            s = s + (a * ctx.unsqueeze(1)).sum(-1)
        if self.gamma is not None:
            s = s - (self.gamma[users].unsqueeze(1) * self.beta[items]).sum(-1) * dlogp
        if self.eta is not None:
            s = s + (self.eta[items] * state).sum(-1)
        if self.mu is not None and weeks is not None:
            s = s + (self.mu[items] * self.delta[weeks].unsqueeze(1)).sum(-1)
        return s

    def l2(self):
        """Penalty on the representation parameters, excluding price.

        The price block gets its own coefficient because it is doing a different job.
        Everything else is there to fit the basket well and benefits from heavy
        shrinkage; gamma and beta exist to measure a causal price response, and
        shrinking them biases that response toward zero.  Penalising all 500k
        parameters at one rate drove the median gamma.beta from +0.89 at l2=1e-4 to
        +0.08 at l2=1e-2 -- an order of magnitude below the reduced-form elasticity
        of 0.84 that 25_basket_placebo.py estimates from the same data.  A model that
        shrunk is useless for what-if questions no matter how well it ranks items.
        """
        t = 0.0
        for p in [self.alpha, self.theta, self.rho, self.eta, self.mu, self.delta]:
            if p is not None:
                t = t + (p ** 2).sum()
        return t

    def l2_price(self):
        t = 0.0
        for p in [self.gamma, self.beta]:
            if p is not None:
                t = t + (p ** 2).sum()
        return t


def make_batch(d: BasketData, split, bidx, n_neg, rng, model_K, device, alpha_detached):
    """Assemble one batch of whole baskets: positives, negatives, context, state."""
    sp = d.splits[split]
    starts, ends = sp["starts"][bidx], sp["ends"][bidx]
    lens = ends - starts
    rows = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    user = sp["user"][rows]
    item = sp["item"][rows]
    day = sp["day"][rows]
    week = sp["week"][rows]
    B = len(rows)

    # Context: mean alpha over the *other* items of the same basket.  Computed from
    # the basket sums, so it is exact for the batch rather than an approximation.
    owner = np.repeat(np.arange(len(bidx)), lens)
    a_all = alpha_detached[item]                                    # [B, K]
    sums = torch.zeros(len(bidx), model_K, device=device, dtype=a_all.dtype)
    sums.index_add_(0, torch.as_tensor(owner, device=device), a_all)
    n_in = torch.as_tensor(lens, device=device, dtype=a_all.dtype)[torch.as_tensor(owner, device=device)]
    ctx = (sums[torch.as_tensor(owner, device=device)] - a_all) / (n_in - 1).clamp_min(1).unsqueeze(1)
    # A single-item basket has no context; zero is the neutral value.
    ctx = torch.where((n_in > 1).unsqueeze(1), ctx, torch.zeros_like(ctx))

    neg = rng.choice(d.J, size=(B, n_neg), p=d.neg_p).astype(np.int64)
    cand = np.concatenate([item[:, None], neg], axis=1)             # [B, 1+n_neg]
    day_rep = np.repeat(day[:, None], cand.shape[1], axis=1)
    user_rep = np.repeat(user[:, None], cand.shape[1], axis=1)
    st = d.state(user_rep.ravel(), cand.ravel(), day_rep.ravel()).reshape(
        B, cand.shape[1], N_STATE_FEATURES)
    dlogp = d.log_price_dev[torch.as_tensor(cand, device=device),
                            torch.as_tensor(day_rep, device=device)]
    return (torch.as_tensor(user, device=device),
            torch.as_tensor(cand, device=device),
            ctx, dlogp,
            torch.as_tensor(st, device=device),
            torch.as_tensor(week, device=device))


@torch.no_grad()
def knn_purity_probe(model, labels, k=10, n_sample=1500, seed=0):
    """Share of an item's k nearest neighbours (cosine on alpha) sharing its label.

    Subsampled so it costs a fraction of a second; the full version lives in
    24_embedding_eval.py.
    """
    E = model.alpha.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(E), min(n_sample, len(E)), replace=False)
    U = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
    sim = U[sel] @ U.T
    sim[np.arange(len(sel)), sel] = -np.inf
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    return float((labels[idx] == labels[sel][:, None]).mean())


@torch.no_grad()
def evaluate(model, d, split, n_neg, rng, device, max_baskets=4000):
    sp = d.splits[split]
    nb = min(sp["n_baskets"], max_baskets)
    idx = rng.choice(sp["n_baskets"], size=nb, replace=False)
    tot, cnt, hits = 0.0, 0, 0
    alpha_det = model.alpha.detach()
    for a in range(0, nb, 256):
        b = idx[a:a + 256]
        users, cand, ctx, dlogp, st, wk = make_batch(d, split, b, n_neg, rng, model.K,
                                                    device, alpha_det)
        s = model.score(users, cand, ctx, dlogp, st, wk)
        lp = torch.log_softmax(s, dim=1)[:, 0]
        tot += float(lp.sum()); cnt += lp.shape[0]
        hits += int((s.argmax(1) == 0).sum())
    return tot / max(cnt, 1), hits / max(cnt, 1)


def main(a):
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    d = BasketData(IN, device=dev, placebo_price=a.placebo_price,
                   placebo_seed=a.seed)
    n_weeks = 52          # week-of-year index; see BasketData.splits
    model = BasketModel(d, K=a.K, Kp=a.Kp, use_context=not a.no_context,
                        use_state=not a.no_state, use_price=not a.no_price,
                        use_taste=not a.no_taste, seed=a.seed,
                        tie_context=a.tie_context,
                        Kt=0 if a.no_season else a.Kt, n_weeks=n_weeks).to(dev)
    sub_labels = d.items.sort_values("item_id").sub_id.to_numpy()
    n_par = sum(p.numel() for p in model.parameters())
    log(f"model {a.label}: K={a.K} Kp={a.Kp} context={not a.no_context} "
        f"tied={a.tie_context} state={not a.no_state} price={not a.no_price} "
        f"taste={not a.no_taste} Kt={0 if a.no_season else a.Kt} "
        f"placebo={a.placebo_price} -> {n_par:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    # Cosine decay to lr * final_lr_frac.  Without it the validation sequence bounces
    # by ~0.03 nats between evaluations and "best checkpoint" is largely a lottery:
    # two runs differing only in the price penalty were checkpointed 5,500 iterations
    # apart, which is what made the embedding look like it traded off against the
    # price coefficient when it was really just a different draw.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.iters, eta_min=a.lr * a.final_lr_frac) if a.lr_decay else None
    rng = np.random.default_rng(a.seed)
    ev_rng = np.random.default_rng(12345)      # fixed, so evals are comparable
    sp = d.splits["train"]
    hist, best, best_it = [], -1e9, -1
    t0 = time.time()
    for it in range(1, a.iters + 1):
        bidx = rng.integers(0, sp["n_baskets"], size=a.batch_baskets)
        users, cand, ctx, dlogp, st, wk = make_batch(d, "train", bidx, a.n_neg, rng,
                                                    model.K, dev, model.alpha.detach())
        s = model.score(users, cand, ctx, dlogp, st, wk)
        loss = (-torch.log_softmax(s, dim=1)[:, 0].mean()
                + (a.l2 * model.l2() + a.l2_price * model.l2_price()) / cand.shape[0])
        opt.zero_grad(); loss.backward(); opt.step()
        if sched is not None:
            sched.step()

        if it % a.eval_every == 0 or it == a.iters:
            vll, vacc = evaluate(model, d, "validation", a.n_neg, ev_rng, dev)
            # Diagnostic only.  Selection stays on validation log-likelihood, so the
            # "the model is never shown sub-commodity" claim is untouched -- this just
            # makes it visible whether ranking and embedding peak together.
            pur = knn_purity_probe(model, sub_labels)
            pcoef = float((model.gamma.detach() @ model.beta.detach().T).median()) \
                if model.gamma is not None else float("nan")
            hist.append({"iter": it, "train_loss": float(loss.detach()), "val_loglik": vll,
                         "val_top1": vacc, "knn_purity": pur, "median_price_coef": pcoef,
                         "secs": time.time() - t0})
            star = ""
            if vll > best:
                best, best_it = vll, it
                torch.save(model.state_dict(), os.path.join(OUT, f"{a.label}_basket.pt"))
                star = " *"
            log(f"  it {it:5d}  loss {float(loss.detach()):.4f}  val loglik {vll:.4f}  "
                f"top1 {vacc:.3f}  purity {pur:.3f}  price {pcoef:+.3f}{star}")

    # Reload the best checkpoint before the final report, so the numbers describe the
    # model that is actually saved rather than the last (possibly overfit) step.
    model.load_state_dict(torch.load(os.path.join(OUT, f"{a.label}_basket.pt"),
                                     map_location=dev))
    tll, tacc = evaluate(model, d, "test", a.n_neg, np.random.default_rng(999), dev,
                         max_baskets=8000)
    log(f"best validation {best:.4f} at iteration {best_it}; test loglik {tll:.4f}, "
        f"top1 {tacc:.3f}")
    with open(os.path.join(OUT, f"{a.label}_basket_history.json"), "w") as f:
        json.dump({"config": vars(a), "history": hist, "n_parameters": n_par,
                   "best_val_loglik": best, "best_iter": best_it,
                   "test_loglik": tll, "test_top1": tacc}, f, indent=2)
    log(f"saved {a.label}_basket.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="basket")
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--Kp", type=int, default=8)
    p.add_argument("--n-neg", type=int, default=20)
    p.add_argument("--batch-baskets", type=int, default=256)
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--lr-decay", action="store_true",
                   help="cosine-decay the learning rate; makes the end of training "
                        "stable so the saved checkpoint is not a lottery")
    p.add_argument("--final-lr-frac", type=float, default=0.05)
    p.add_argument("--l2-price", type=float, default=1e-4,
                   help="separate penalty on gamma and beta; heavy shrinkage here "
                        "biases the price response toward zero and destroys any "
                        "counterfactual use of the model")
    p.add_argument("--no-context", action="store_true", help="ablate product interaction")
    p.add_argument("--tie-context", action="store_true",
                   help="use alpha_j itself as the interaction coefficient instead of a "
                        "free rho_j, so co-purchase structure is forced into the "
                        "embedding the sub-commodity test reads")
    p.add_argument("--no-state", action="store_true", help="ablate household state")
    p.add_argument("--no-price", action="store_true")
    p.add_argument("--no-taste", action="store_true", help="ablate household taste vectors")
    p.add_argument("--Kt", type=int, default=8,
                   help="rank of the item x week seasonality term mu_j . delta_w")
    p.add_argument("--no-season", action="store_true", help="ablate seasonality")
    p.add_argument("--placebo-price", default="none",
                   choices=["none", "permute", "swap"],
                   help="scramble the price panel before fitting; a price coefficient "
                        "that survives this was never measuring price")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
