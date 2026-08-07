"""
Stage 27 -- Nested basket model: category incidence, item choice, quantity, stores.

Stage 23 dropped three things it should not have, and this puts them back.

1. THE NEST.  23 models "which item", never "does this household buy from this
   category at all".  That is not a simplification, it is the loss of the question a
   retailer actually asks: cut Tide's price and does total detergent volume grow, or
   does Tide just take share from Gain?  23 cannot distinguish those.  Unit demand
   failing kills the within-category *softmax*; it says nothing about whether
   incidence is a separate decision from allocation, and 23 conflated the two.

2. QUANTITY.  23 treats purchase as binary.  22.3% of (basket, item) rows buy more
   than one unit and those rows carry 42.6% of all units, so binary throws away
   nearly half the volume.  Worse, it throws away one of the two channels a price cut
   works through: the units-per-buyer elasticity is -0.22, on top of the extra buyers.

3. STORES.  23 pools prices to chain level and ignores assortment.  15.8% of
   store-item-weeks sit more than a cent from the chain price, which is measurement
   error in the regressor the elasticity is read off; and the median store carries
   63% of the catalogue, so scoring an item a store never stocked as a *rejected*
   alternative is a specification error -- it was not there to reject.

How the nest and multiple items coexist
---------------------------------------
The paper gets its nest from a within-category softmax plus an outside good, which is
exactly what unit demand buys it.  Drop unit demand and that construction is gone, but
the nest is not: it survives as a Poisson-multinomial factorisation.

    total units from category c    Q_ict ~ Poisson(exp(a_ic + kappa_c * IV_ict))
    allocation across its items    multinomial(softmax_j u_ijt)
    IV_ict = log sum_{j in c, stocked at s} exp(u_ijt)

Poisson-multinomial factorises, so this is equivalent to independent counts

    q_ijt ~ Poisson(exp(a_ic + (kappa_c - 1) * IV_ict + u_ijt))

and `kappa` is a nesting coefficient with the paper's interpretation:

    kappa = 1   IV cancels; category volume is whatever the items sum to (IIA)
    kappa = 0   category volume is fixed; a price cut only moves share
    kappa > 1   the category expands more than proportionally

That is the parameter that answers the question in (1), it is estimated rather than
assumed, and it is compatible with buying three yogurts.

Fitting.  A full Poisson likelihood needs every item on every trip (5,455 x 199,347),
so the three heads are trained jointly on sampled data:

  item head       softmax over {chosen} u {negatives}.  Negatives are drawn from the
                  WHOLE catalogue (unigram^0.75) and then masked: an item the trip's
                  store does not carry gets -1e9, so the effective competitor count is
                  below n_neg and varies by store
  quantity head   units - 1 ~ Poisson, on chosen items only, with its own price
                  coefficient, so the quantity margin gets its own elasticity
  incidence head  per (trip, category): bought or not, against IV computed over the
                  category's stocked items.  Categories are sampled, positives and
                  negatives, so this costs ~4 categories x ~29 items per trip

Writes out/<label>_nested.pt and out/<label>_nested_history.json.
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
IN = os.path.join(HERE, "..", os.environ.get("NF_BASKET_INPUT", "basket_input"))
OUT = os.path.join(HERE, "..", "out")

N_STATE_FEATURES = 4


def log(m):
    print(f"[27] {m}", flush=True)


class NestedData:
    def __init__(self, indir, device="cpu", placebo_price="none",
                 placebo_seed=0):
        self.meta = json.load(open(os.path.join(indir, "meta.json")))
        bk = pd.read_parquet(os.path.join(indir, "baskets.parquet"))
        self.items = pd.read_parquet(os.path.join(indir, "items.parquet"))
        self.J = int(self.meta["n_items"]); self.N = int(self.meta["n_users"])
        self.S = int(self.meta["n_subs"]); self.C = int(self.meta["n_commodities"])
        self.n_stores = int(self.meta["n_stores"])
        self.stride = int(self.meta["day_stride"])
        self.device = device

        lpd = np.load(os.path.join(indir, "log_price_dev.npy"))
        # Structural placebo: scramble the price panel before fitting.  A price
        # coefficient that survives this was never measuring price.
        if placebo_price != "none":
            r = np.random.default_rng(placebo_seed)
            if placebo_price == "permute":
                for j in range(lpd.shape[0]):
                    lpd[j] = r.permutation(lpd[j])
            elif placebo_price == "swap":
                lpd = lpd[r.permutation(lpd.shape[0])]
            log(f"PLACEBO: price panel scrambled with '{placebo_price}'")
        self.log_price_dev = torch.as_tensor(lpd, device=device)
        z = np.load(os.path.join(indir, "state.npz"))
        self.keys = z["keys"]; self.sub_gap = z["sub_gap"].astype(np.float32)
        self.item_sub = z["item_sub"].astype(np.int64)
        self.cat_keys = z["cat_keys"]; self.cat_gap = z["cat_gap"].astype(np.float32)

        # --- stores: sparse price deviations, dense availability
        sz = np.load(os.path.join(indir, "store_price.npz"))
        self.carried = torch.as_tensor(sz["carried"], device=device)      # [J, nS] bool
        # sparse (item, store, week) -> log price deviation, as a dict-free lookup on
        # a sorted key array, same trick the state uses
        k = (sz["item"].astype(np.int64) * self.n_stores + sz["store"]) * 128 + sz["week"]
        o = np.argsort(k)
        self.sp_keys = k[o]
        dev_arr = sz["dev"][o]
        if placebo_price != "none":
            # the store-level deviation is price too; leaving it real would leak
            dev_arr = np.random.default_rng(placebo_seed + 1).permutation(dev_arr)
        self.sp_dev = torch.as_tensor(dev_arr, device=device)

        it_cat = self.items.sort_values("item_id").cat_id.to_numpy()
        self.item_cat = torch.as_tensor(it_cat, device=device)
        # padded [C, Mmax] item block per category, for the inclusive value
        cmax = int(np.bincount(it_cat).max())
        ci = np.zeros((self.C, cmax), dtype=np.int64)
        cm = np.zeros((self.C, cmax), dtype=np.float32)
        for c in range(self.C):
            js = np.flatnonzero(it_cat == c)
            ci[c, :len(js)] = js; cm[c, :len(js)] = 1.0
        self.cat_items = torch.as_tensor(ci, device=device)
        self.cat_mask = torch.as_tensor(cm, device=device)
        self.cat_items_np = ci                       # for in-category negatives
        self.cat_nvalid_np = cm.sum(1).astype(np.int64)
        self.item_cat_np = it_cat

        bk = bk.sort_values(["split", "BASKET_ID", "item_id"]).reset_index(drop=True)
        self.splits = {}
        for sp, g in bk.groupby("split", sort=False):
            g = g.reset_index(drop=True)
            codes, _ = pd.factorize(g.BASKET_ID)
            starts = np.flatnonzero(np.r_[True, np.diff(codes) != 0])
            ends = np.r_[starts[1:], len(g)]
            self.splits[sp] = {
                "user": g.user_id.to_numpy(np.int64), "item": g.item_id.to_numpy(np.int64),
                "day": g.DAY.to_numpy(np.int64), "units": g.units.to_numpy(np.float32),
                "store": g.store_id.to_numpy(np.int64),
                "week": (g.WEEK_NO.to_numpy(np.int64) - 1) % 52,
                "raw_week": g.WEEK_NO.to_numpy(np.int64),
                "cat": it_cat[g.item_id.to_numpy()],
                "starts": starts.astype(np.int64), "ends": ends.astype(np.int64),
                "n_baskets": len(starts), "n_rows": len(g)}
        cnt = np.bincount(self.splits["train"]["item"], minlength=self.J).astype(np.float64)
        pr = np.power(np.maximum(cnt, 1.0), 0.75)
        self.neg_p = pr / pr.sum()
        log(f"data: {self.N:,} households, {self.J:,} items, {self.C} categories, "
            f"{self.n_stores} stores; "
            + ", ".join(f"{k} {v['n_rows']:,} rows" for k, v in self.splits.items()))
        log(f"      units>1 on {self.meta['share_rows_units_gt1']:.1%} of rows carrying "
            f"{self.meta['share_units_in_multi_rows']:.1%} of units; "
            f"assortment {self.meta['assortment_share']:.1%} of the item x store grid")

    def store_dev(self, item, store, week):
        """Store-level log price deviation; 0 where that store-week was not observed."""
        k = (item * self.n_stores + store) * 128 + week
        idx = np.searchsorted(self.sp_keys, k)
        ok = (idx < len(self.sp_keys))
        idx = np.clip(idx, 0, max(len(self.sp_keys) - 1, 0))
        hit = ok & (self.sp_keys[idx] == k)
        out = torch.zeros(len(k), dtype=torch.float32, device=self.device)
        if hit.any():
            out[torch.as_tensor(np.flatnonzero(hit), device=self.device)] = \
                self.sp_dev[torch.as_tensor(idx[hit], device=self.device)]
        return out

    def cat_state(self, user, cat, day):
        """Days since this household last bought ANYTHING from this category.

        Same construction as `state`, one level up: strictly-before lookup on a sorted
        key array, so nothing after `day` can enter.  Replaces the old proxy, which read
        the sub-commodity of whichever item in the category had the lowest PRODUCT_ID.
        """
        group = user.astype(np.int64) * self.C + cat
        key = group * self.stride + day
        idx = np.searchsorted(self.cat_keys, key, side="left")
        prev = np.clip(idx - 1, 0, len(self.cat_keys) - 1)
        prev_key = self.cat_keys[prev]
        same = (idx - 1 >= 0) & ((prev_key // self.stride) == group)
        since = np.where(same, day - (prev_key % self.stride), 0).astype(np.float32)
        gap = self.cat_gap[cat]
        return np.stack([
            (~same).astype(np.float32),
            np.where(same, np.exp(-since / 7.0), 0.0),
            np.where(same, np.exp(-since / gap), 0.0),
            np.where(same, np.log1p(since) / np.log(100.0), 0.0)], axis=1).astype(np.float32)

    def state(self, user, item, day):
        sub = self.item_sub[item]
        group = user.astype(np.int64) * self.S + sub
        key = group * self.stride + day
        idx = np.searchsorted(self.keys, key, side="left")
        prev = np.clip(idx - 1, 0, len(self.keys) - 1)
        prev_key = self.keys[prev]
        same = (idx - 1 >= 0) & ((prev_key // self.stride) == group)
        since = np.where(same, day - (prev_key % self.stride), 0).astype(np.float32)
        gap = self.sub_gap[sub]
        return np.stack([
            (~same).astype(np.float32),
            np.where(same, np.exp(-since / 7.0), 0.0),
            np.where(same, np.exp(-since / gap), 0.0),
            np.where(same, np.log1p(since) / np.log(100.0), 0.0)], axis=1).astype(np.float32)


class NestedModel(nn.Module):
    def __init__(self, d: NestedData, K=64, Kp=8, Kt=8, Ks=4, n_weeks=52, seed=0,
                 use_nest=True, use_quantity=True, use_store=True,
                 use_store_price=True, avail_only=False, use_state=True,
                 use_breadth=True, use_context=True, ctx_agg="mean",
                 learn_ctx_scale=False, use_cat_context=False,
                 use_cat_pair=False, untie_rho=False, prefix_context=False,
                 neg_in_cat=0.0, item_loss="softmax"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        J, N, C = d.J, d.N, d.C
        self.K, self.Kp, self.d = K, Kp, d
        self.use_nest, self.use_quantity, self.use_store = use_nest, use_quantity, use_store
        self.use_breadth = use_breadth
        # The tied interaction alpha_j . alpha_bar(basket) carried over from the flat
        # model, where BASKET_MODEL.md 2 shows tying is what forces co-purchase
        # structure into the embedding the sub-commodity test reads.  It was
        # hard-wired here and therefore never ablated in the nested model -- so the
        # claim that it is load-bearing rested entirely on the flat model's evidence.
        self.use_context = use_context
        # How the basket context is pooled.  "mean" divides by (n-1), which makes
        # E(S) a family indexed by basket size and -- measured in NESTED_MODEL.md
        # 8.6c -- dilutes a strong pairwise affinity by 1/(n-1), so the implied joint
        # has weak co-occurrence even though alpha encodes it well (spearman +0.58
        # against the real lift).  "sum" drops the divisor: E(S) becomes a single
        # pairwise energy over all basket sizes, and the coupling per pair is
        # undiluted.  The swap identity holds either way.
        self.ctx_agg = ctx_agg
        # A learnable scalar on the interaction lets the model choose its strength
        # rather than inheriting whatever the pooling implies.
        self.ctx_scale = nn.Parameter(torch.ones(1)) if learn_ctx_scale else None
        # Which CATEGORIES land in a basket is decided by C independent Bernoulli
        # draws, because the incidence head is given a zero context.  95% of item
        # pairs are cross-category, so 95% of co-purchase structure is set by
        # independent coin flips -- measured in NESTED_MODEL.md 8.6e: within-category
        # pairs reach a rank correlation of +0.290 against real lift, cross-category
        # pairs -0.094.  A category-level context fixes the level the item context
        # cannot reach: same leave-one-out construction, one layer up.
        self.use_cat_context = use_cat_context
        self.cat_ctx_scale = nn.Parameter(torch.ones(1)) if use_cat_context else None
        # A FULL pairwise matrix over categories.  For items this is impossible --
        # 5,455^2 is 30M parameters -- which is why the item interaction is a dot
        # product of embeddings, and why it can only express similarity: alpha is
        # also carrying taste and is graded on sub-commodity clustering.  Categories
        # are different.  C = 188 means C(C-1)/2 = 17,578 free parameters, so the
        # pairwise log-odds can simply be LEARNED, with no embedding, no tying and no
        # similarity constraint.  It can represent complementarity -- two categories
        # that co-occur while being nothing alike -- which no dot product on a shared
        # embedding can.  And 95% of item pairs are cross-category, so this is the
        # level where the co-occurrence gap lives (NESTED_MODEL.md 8.6e).
        self.use_cat_pair = use_cat_pair
        self.cat_pair = nn.Parameter(torch.zeros(C, C)) if use_cat_pair else None
        # SHOPPER (Ruiz, Athey & Blei 2020, eq. 4) writes the interaction as
        # rho_c . mean(alpha of items already in the basket) with rho UNTIED from
        # alpha.  The factorisation is deliberately asymmetric: the effect of k on j is
        # rho_j . alpha_k, which need not equal rho_k . alpha_j, and which may be
        # NEGATIVE.  That is what lets taco shells and beans be complements while their
        # attributes sit far apart, and what lets two brands of the same yogurt repel
        # as substitutes.  The tied form alpha_j . alpha_k can do neither: it is
        # symmetric and large exactly when two items are alike, which is why
        # NESTED_MODEL.md 8.6e found every variant getting the AMOUNT of clustering
        # right and none getting the PATTERN right.
        self.untie_rho = untie_rho
        # SHOPPER is a sequential model: the context at step i is the mean over the
        # i-1 items already placed, and the unordered likelihood sums over
        # permutations (eq. 6).  With prefix_context a random permutation is drawn per
        # basket and each row sees only the items before it -- one unbiased sample from
        # that sum -- so training matches sequential generation instead of the
        # leave-one-out full-basket context.
        self.prefix_context = prefix_context
        self.neg_in_cat = neg_in_cat
        self.item_loss = item_loss

        def emb(n, k, sd=0.05):
            return nn.Parameter(torch.randn(n, k, generator=g) * sd)

        # --- item layer (as stage 23, tied interaction)
        self.lam = nn.Parameter(torch.zeros(J))
        self.alpha = emb(J, K)
        self.rho = emb(J, K) if untie_rho else None
        self.theta = emb(N, K)
        self.gamma = emb(N, Kp, 0.1); self.beta = emb(J, Kp, 0.1)
        self.use_state = use_state
        self.eta = emb(J, N_STATE_FEATURES, 0.01) if use_state else None
        self.mu = emb(J, Kt, 0.02); self.delta = emb(n_weeks, Kt, 0.02)
        # store x item affinity: format and assortment differ, and a low-rank term
        # keeps it to 115*Ks + J*Ks parameters instead of 115*5455
        self.use_store_price = use_store_price
        keep_aff = use_store and not avail_only
        self.store_vec = emb(d.n_stores, Ks, 0.02) if keep_aff else None
        self.item_store = emb(J, Ks, 0.02) if keep_aff else None

        # --- quantity head: units - 1 ~ Poisson(exp(.)), own price coefficient so the
        # quantity margin is not forced to share the incidence one
        if use_quantity:
            self.q0 = nn.Parameter(torch.zeros(J))
            self.q_gamma = emb(N, Kp, 0.05); self.q_beta = emb(J, Kp, 0.05)
            self.q_state = emb(J, N_STATE_FEATURES, 0.01)

        # --- incidence head: per category, with the nesting coefficient on IV
        # Breadth: given the category is bought, how many *distinct* items?  The
        # incidence head only ever sees 0/1, so nothing in the model knows this.
        # Recovering a Poisson rate from P(buy) via -log(1-p) reproduces the
        # probability but implies E[picks | bought] of about 1.01-1.10 for realistic
        # incidence rates -- against a real 1.284.  That gap is the whole generation
        # shortfall, and no amount of sampler fixing closes it: the quantity is simply
        # not in the likelihood.  breadth - 1 ~ Poisson, per category, with household
        # and price terms because a big shopper and a promotion both widen a basket.
        if use_breadth:
            self.b0 = nn.Parameter(torch.zeros(C))
            self.b_user = nn.Parameter(torch.zeros(N))
            self.b_price = nn.Parameter(torch.zeros(C))
        if use_nest:
            self.register_buffer("iv_ref", torch.zeros(C))
            self.register_buffer("iv_ref_set", torch.zeros(1))
            self.c0 = nn.Parameter(torch.zeros(C))
            self.c_user = emb(N, K, 0.05); self.c_cat = emb(C, K, 0.05)
            # kappa parameterised as softplus so it stays positive; init near 1, which
            # is the "IV cancels" point, so the data has to move it either way
            self.kappa_raw = nn.Parameter(torch.full((C,), 0.5413))   # softplus -> 1.0
            self.c_state = emb(C, N_STATE_FEATURES, 0.01)

    def kappa(self):
        return nn.functional.softplus(self.kappa_raw)

    def cat_pair_sym(self):
        """Symmetric, zero-diagonal: W[c,c'] is one number per unordered pair."""
        w = 0.5 * (self.cat_pair + self.cat_pair.T)
        return w - torch.diag_embed(torch.diagonal(w))

    def item_utility(self, users, items, ctx, dlogp, state, weeks, stores):
        s = self.lam[items]
        a = self.alpha[items]
        s = s + torch.einsum("bk,bmk->bm", self.theta[users], a)
        if self.use_context:
            r = self.rho[items] if self.rho is not None else a
            inter = (r * ctx.unsqueeze(1)).sum(-1)
            s = s + (inter * self.ctx_scale if self.ctx_scale is not None else inter)
        s = s - (self.gamma[users].unsqueeze(1) * self.beta[items]).sum(-1) * dlogp
        if self.eta is not None:
            s = s + (self.eta[items] * state).sum(-1)
        s = s + (self.mu[items] * self.delta[weeks].unsqueeze(1)).sum(-1)
        if self.store_vec is not None:
            s = s + (self.item_store[items] * self.store_vec[stores].unsqueeze(1)).sum(-1)
        return s

    def l2_repr(self):
        ps = [self.alpha, self.theta, self.eta, self.mu, self.delta,
              self.store_vec, self.item_store]
        if self.use_nest:
            ps += [self.c_user, self.c_cat, self.c_state]
        if self.use_quantity:
            ps += [self.q_state]
        return sum((p ** 2).sum() for p in ps if p is not None)

    def l2_rho(self):
        """Penalised separately: untied rho has J*K free parameters with nothing
        anchoring it, and the flat model's attempt (BASKET_MODEL.md 2) lost 0.17 nats
        of fit without one."""
        return (self.rho ** 2).sum() if self.rho is not None else 0.0

    def l2_cat_pair(self):
        """Penalised separately from the representation block.

        Unregularised, the pairwise matrix absorbs signal that belongs to the
        inclusive value: kappa fell 0.663 -> 0.597 and the price response recovered
        from generated data dropped 82% -> 70%.  Price reaches incidence only through
        kappa * IV, so anything that explains category co-occurrence more cheaply than
        IV does will take that job and take the price response with it.
        """
        return (self.cat_pair ** 2).sum() if self.cat_pair is not None else 0.0

    def l2_price(self):
        ps = [self.gamma, self.beta]
        if self.use_quantity:
            ps += [self.q_gamma, self.q_beta]
        return sum((p ** 2).sum() for p in ps if p is not None)


def make_batch(d, model, split, bidx, n_neg, rng, device, clean_neg=True):
    """Positives, store-aware negatives, context, prices, state, units."""
    sp = d.splits[split]
    starts, ends = sp["starts"][bidx], sp["ends"][bidx]
    lens = ends - starts
    rows = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    user, item = sp["user"][rows], sp["item"][rows]
    day, week = sp["day"][rows], sp["week"][rows]
    store, units = sp["store"][rows], sp["units"][rows]
    B = len(rows)

    owner = np.repeat(np.arange(len(bidx)), lens)
    ow = torch.as_tensor(owner, device=device)
    a_all = model.alpha.detach()[item]
    sums = torch.zeros(len(bidx), model.K, device=device, dtype=a_all.dtype)
    sums.index_add_(0, ow, a_all)
    n_in = torch.as_tensor(lens, device=device, dtype=a_all.dtype)[ow]
    if getattr(model, "prefix_context", False):
        # SHOPPER eq. 6: draw one permutation per basket and let each row see only the
        # items BEFORE it.  Vectorised as a cumulative sum along a random order.
        key = rng.random(B)
        order = np.lexsort((key, owner))            # random within basket
        pos = np.empty(B, dtype=np.int64)
        pos[order] = np.arange(B) - np.repeat(
            np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
        a_ord = a_all[torch.as_tensor(order, device=device)]
        csum = torch.cumsum(a_ord, 0)
        base = torch.zeros_like(csum)
        starts_c = np.concatenate([[0], np.cumsum(lens)[:-1]])
        base_idx = torch.as_tensor(np.repeat(starts_c, lens), device=device)
        # prefix sum within basket = cumsum up to here minus cumsum before the basket
        before = torch.cat([torch.zeros(1, model.K, device=device, dtype=csum.dtype),
                            csum])[base_idx]
        pre_ord = torch.cat([torch.zeros(1, model.K, device=device, dtype=csum.dtype),
                             csum])[torch.arange(B, device=device)] - before
        ctx_ord = pre_ord / torch.as_tensor(pos[order], device=device,
                                            dtype=csum.dtype).clamp_min(1).unsqueeze(1)
        ctx_ord = torch.where(
            torch.as_tensor(pos[order] > 0, device=device).unsqueeze(1),
            ctx_ord, torch.zeros_like(ctx_ord))
        ctx = torch.zeros_like(ctx_ord)
        ctx[torch.as_tensor(order, device=device)] = ctx_ord
    else:
        ctx = sums[ow] - a_all
        if getattr(model, "ctx_agg", "mean") == "mean":
            ctx = ctx / (n_in - 1).clamp_min(1).unsqueeze(1)
        ctx = torch.where((n_in > 1).unsqueeze(1), ctx, torch.zeros_like(ctx))

    neg = rng.choice(d.J, size=(B, n_neg), p=d.neg_p).astype(np.int64)
    negq = d.neg_p[neg].astype(np.float64)      # proposal density, per column
    # Harder negatives.  Drawn from the whole catalogue, a decoy is usually separable
    # by lambda_j + theta_i.alpha_j alone, so the softmax rarely has to ask "does this
    # item fit THIS basket" -- which may be why no interaction parameterisation moved
    # the per-pair co-occurrence rank correlation off zero (NESTED_MODEL.md 8.6f).
    # Replacing a fraction with items from the TRUE item's own category removes
    # popularity and broad taste as discriminators and leaves the basket context as
    # the signal that can separate them.
    frac = getattr(model, "neg_in_cat", 0.0)
    if frac > 0:
        kk = max(1, int(round(n_neg * frac)))
        cpos = d.item_cat_np[item]
        nv = d.cat_nvalid_np[cpos]
        pick = (rng.random((B, kk)) * nv[:, None]).astype(np.int64)
        neg[:, :kk] = d.cat_items_np[cpos[:, None], pick]
        # these columns come from a DIFFERENT proposal -- uniform over the category's
        # stocked items -- so they carry their own q for the logQ correction
        negq[:, :kk] = 1.0 / np.maximum(nv[:, None], 1)
    cand = np.concatenate([item[:, None], neg], axis=1)
    M = cand.shape[1]
    day_r = np.repeat(day[:, None], M, 1)
    user_r = np.repeat(user[:, None], M, 1)
    store_r = np.repeat(store[:, None], M, 1)
    rw = np.repeat(sp["raw_week"][rows][:, None], M, 1)

    st = d.state(user_r.ravel(), cand.ravel(), day_r.ravel()).reshape(B, M, N_STATE_FEATURES)
    ci = torch.as_tensor(cand, device=device)
    dlogp = d.log_price_dev[ci, torch.as_tensor(day_r, device=device)]
    if model.use_store and model.use_store_price:
        dlogp = dlogp + d.store_dev(cand.ravel(), store_r.ravel(), rw.ravel()).reshape(B, M)
    # An item the store does not stock was not available to reject, so it must not
    # sit in the choice set.  The chosen item is stocked by construction.
    avail = torch.ones(B, M, dtype=torch.bool, device=device)
    if model.use_store:
        avail = d.carried[ci, torch.as_tensor(store_r, device=device)].clone()
    # Two kinds of bad negative, both measured on this data:
    #   1.2%  a "negative" that IS the true item -- the loss can never be achieved
    #  17.8%  a "negative" that is another item of the SAME basket.  The conditional is
    #         explicitly given the rest of the basket, so those items are positives; the
    #         loss was pushing down products that are genuinely in the cart.
    # Both are masked out of the denominator.
    key_pos = owner.astype(np.int64) * d.J + item
    key_neg = owner[:, None].astype(np.int64) * d.J + neg
    bad = np.isin(key_neg, key_pos)
    if clean_neg:
        avail[:, 1:] &= torch.as_tensor(~bad, device=device)
    avail[:, 0] = True

    # logQ correction.  Negatives are drawn from a proposal q, not from the model, so a
    # plain softmax over {true} u {sampled} estimates a distribution tilted by q.  Here q
    # spans 3.5e-06 to 6.3e-03 -- 7.5 nats -- and lambda_j was absorbing it
    # (corr(lambda_j, log q) = -0.51).  Subtracting log(K q(m)) from each sampled logit
    # makes sum_m exp(u_m - log(K q_m)) an unbiased estimator of sum_{m != j} exp(u_m),
    # which is the denominator the softmax is supposed to have.
    logq = torch.zeros(B, M, device=device)
    logq[:, 1:] = torch.log(torch.as_tensor(negq, device=device,
                                            dtype=torch.float32).clamp_min(1e-12)
                            * float(n_neg))
    return dict(
        user=torch.as_tensor(user, device=device), cand=ci, ctx=ctx, dlogp=dlogp,
        state=torch.as_tensor(st, device=device),
        week=torch.as_tensor(week, device=device),
        store=torch.as_tensor(store, device=device),
        units=torch.as_tensor(units, device=device), avail=avail,
        logq=logq if clean_neg else None)


def incidence_batch(d, model, split, bidx, n_cat, rng, device, iv_cap=32):
    """Sampled (trip, category) pairs with the inclusive value over stocked items."""
    sp = d.splits[split]
    # Categories are sampled UNIFORMLY, and this is the third attempt at getting it
    # right, so the reasoning is worth recording.
    #
    # Case-control sampling (a few bought, a few not) is tempting because the true
    # base rate is only 3.37% -- 6.3 of 188 categories -- so uniform sampling sees few
    # positives.  But it biases the fit in two separate ways:
    #
    #   1. the intercept, which the standard offset log(pi_1/pi_0) does correct;
    #   2. the term kappa * (IV - ref), whose *mean differs between the sample and the
    #      population* because bought categories have higher inclusive values.  The
    #      offset does not touch this, c0 absorbs the training-sample average, and at
    #      generation a uniform sample sits ~0.9 lower in (IV - ref).  With kappa
    #      around 0.8 that is a 0.73 drop in the logit, which emptied the generated
    #      baskets: 4.0 categories against a real 6.5, median basket 0 items.
    #
    # Uniform sampling removes both at the source and needs no correction at all.
    # The cost is more categories per trip to see enough positives, hence the higher
    # n_cat default for this sampler.
    cats, ys, trip, off, brd = [], [], [], [], []
    bought_sets = []
    for ti, i in enumerate(bidx):
        rows_i = sp["cat"][sp["starts"][i]:sp["ends"][i]]
        cnt = {}
        for c in rows_i.tolist():
            cnt[c] = cnt.get(c, 0) + 1
        bought_sets.append(sorted(cnt))
        # without replacement: with it, 48% of trips duplicated a category and
        # double-counted it in the incidence loss
        for c in rng.choice(d.C, size=min(n_cat, d.C), replace=False).tolist():
            cats.append(c); ys.append(1.0 if c in cnt else 0.0)
            trip.append(ti); off.append(0.0)
            brd.append(float(cnt.get(c, 0)))     # distinct items, 0 when not bought
    if not cats:
        return None
    cats = np.asarray(cats, dtype=np.int64); trip = np.asarray(trip)
    P = len(cats)
    first = sp["starts"][bidx][trip]
    user, day = sp["user"][first], sp["day"][first]
    week, store, rw = sp["week"][first], sp["store"][first], sp["raw_week"][first]

    ct = torch.as_tensor(cats, device=device)
    blk, msk = d.cat_items[ct], d.cat_mask[ct].clone()
    # Every category is padded to the largest one (225 items) although the median has
    # 15, so a dense block spends most of its work on padding and costs 5.4x the item
    # head.  Sample `iv_cap` of each category's items instead and scale the sum by
    # n_c / m: that is an unbiased estimator of sum_j exp(u_j), which is what the
    # inclusive value needs, at a fixed cost per category.
    n_valid = msk.sum(1)
    if iv_cap and blk.shape[1] > iv_cap:
        m = torch.clamp(n_valid, max=iv_cap)
        r = torch.rand(len(cats), iv_cap, device=device)
        pick = (r * n_valid.clamp_min(1).unsqueeze(1)).long().clamp_max(
            blk.shape[1] - 1)                                    # [P, iv_cap]
        blk = torch.gather(blk, 1, pick)
        msk = torch.gather(msk, 1, pick)
        iv_scale = torch.log(n_valid.clamp_min(1) / m.clamp_min(1))
    else:
        iv_scale = torch.zeros(len(cats), device=device)
    Mx = blk.shape[1]
    blk_np = blk.cpu().numpy()
    day_r = np.repeat(day[:, None], Mx, 1)
    user_r = np.repeat(user[:, None], Mx, 1)
    store_r = np.repeat(store[:, None], Mx, 1)
    rw_r = np.repeat(rw[:, None], Mx, 1)
    st = d.state(user_r.ravel(), blk_np.ravel(), day_r.ravel()).reshape(P, Mx, N_STATE_FEATURES)
    dlogp = d.log_price_dev[blk, torch.as_tensor(day_r, device=device)]
    if model.use_store:
        if model.use_store_price:
            dlogp = dlogp + d.store_dev(blk_np.ravel(), store_r.ravel(),
                                        rw_r.ravel()).reshape(P, Mx)
        msk = msk * d.carried[blk, torch.as_tensor(store_r, device=device)].float()
    # Incidence is decided before the basket exists, so there is no context yet: the
    # inclusive value has to be what the category is worth on arrival, not after the
    # basket is known.  Feeding a context here would be circular.  Note the
    # consequence -- the interaction term is inert for the incidence head, and so for
    # the incidence channel of the elasticity decomposition, which is why removing it
    # barely moves anything outside item ranking (NESTED_MODEL.md 8.1).
    ctx = torch.zeros(P, model.K, device=device)
    u = model.item_utility(torch.as_tensor(user, device=device), blk, ctx, dlogp,
                           torch.as_tensor(st, device=device),
                           torch.as_tensor(week, device=device),
                           torch.as_tensor(store, device=device))
    any_stocked = msk.sum(1) > 0
    iv = torch.logsumexp(u.masked_fill(msk == 0, -1e9), dim=1) + iv_scale
    # A category the store stocks nothing from has IV = -1e9, which would dominate the
    # incidence logit; zero it and let the category intercept explain it, which is
    # what it is: not on offer.  Centring happens in losses(), against the frozen
    # per-category reference, so training and generation share one constant.
    iv = torch.where(any_stocked, iv, torch.zeros_like(iv))
    cst = d.cat_state(user, cats, day)
    # mean price deviation across the category's stocked items, for the breadth head
    pdev = (dlogp * (msk > 0).float()).sum(1) / (msk > 0).float().sum(1).clamp_min(1)
    # leave-one-out mean of the category embedding over the OTHER categories this
    # trip bought.  Same construction as the item context, one level up: for a
    # category the trip bought, itself is removed; for one it did not, nothing is.
    # sum of the learned pairwise log-odds against the OTHER categories bought
    pair_term = torch.zeros(P, device=device)
    if getattr(model, "use_cat_pair", False):
        Bmat = torch.zeros(len(bidx), model.d.C, device=device)
        for ti, b in enumerate(bought_sets):
            if b:
                Bmat[ti, torch.as_tensor(b, device=device)] = 1.0
        W = model.cat_pair_sym()
        pair_term = (Bmat @ W)[torch.as_tensor(trip, device=device), ct]

    cat_ctx = torch.zeros(P, model.K, device=device)
    if getattr(model, "use_cat_context", False):
        cc = model.c_cat.detach()
        sums = torch.stack([cc[b].sum(0) if b else torch.zeros(model.K, device=device)
                            for b in bought_sets])
        nb_ = torch.as_tensor([len(b) for b in bought_sets], device=device,
                              dtype=cc.dtype)
        tt = torch.as_tensor(trip, device=device)
        yv = torch.as_tensor(np.asarray(ys, dtype=np.float32), device=device)
        num = sums[tt] - cc[ct] * yv.unsqueeze(1)
        den = (nb_[tt] - yv).clamp_min(1.0).unsqueeze(1)
        cat_ctx = torch.where((nb_[tt] - yv > 0).unsqueeze(1), num / den,
                              torch.zeros_like(num))
    return dict(cat=ct, user=torch.as_tensor(user, device=device), iv=iv,
                cat_ctx=cat_ctx, pair_term=pair_term,
                state=torch.as_tensor(cst, device=device),
                offset=torch.as_tensor(np.asarray(off, dtype=np.float32), device=device),
                breadth=torch.as_tensor(np.asarray(brd, dtype=np.float32), device=device),
                pdev=pdev,
                y=torch.as_tensor(np.asarray(ys, dtype=np.float32), device=device))


@torch.no_grad()
def uniform_iv(d, model, split, bidx, n_cat, rng, device, iv_cap=32):
    """Inclusive values for categories drawn UNIFORMLY, ignoring what was bought.

    Used only to set the frozen per-category IV reference.  incidence_batch cannot be
    reused for this: it is case-control sampled by design, so its category mix is not
    the population's.
    """
    sp = d.splits[split]
    T = len(bidx)
    cats = rng.integers(0, d.C, size=(T, n_cat)).ravel()
    trip = np.repeat(np.arange(T), n_cat)
    first = sp["starts"][bidx][trip]
    user, day = sp["user"][first], sp["day"][first]
    week, store, rw = sp["week"][first], sp["store"][first], sp["raw_week"][first]
    ct = torch.as_tensor(cats, device=device)
    blk, msk = d.cat_items[ct], d.cat_mask[ct].clone()
    n_valid = msk.sum(1)
    if iv_cap and blk.shape[1] > iv_cap:
        m_ = torch.clamp(n_valid, max=iv_cap)
        r = torch.rand(len(cats), iv_cap, device=device)
        pick = (r * n_valid.clamp_min(1).unsqueeze(1)).long().clamp_max(blk.shape[1] - 1)
        blk, msk = torch.gather(blk, 1, pick), torch.gather(msk, 1, pick)
        iv_scale = torch.log(n_valid.clamp_min(1) / m_.clamp_min(1))
    else:
        iv_scale = torch.zeros(len(cats), device=device)
    Mx = blk.shape[1]
    blk_np = blk.cpu().numpy()
    day_r = np.repeat(day[:, None], Mx, 1)
    user_r = np.repeat(user[:, None], Mx, 1)
    store_r = np.repeat(store[:, None], Mx, 1)
    rw_r = np.repeat(rw[:, None], Mx, 1)
    st = d.state(user_r.ravel(), blk_np.ravel(), day_r.ravel()).reshape(
        len(cats), Mx, N_STATE_FEATURES)
    dlogp = d.log_price_dev[blk, torch.as_tensor(day_r, device=device)]
    if model.use_store:
        if model.use_store_price:
            dlogp = dlogp + d.store_dev(blk_np.ravel(), store_r.ravel(),
                                        rw_r.ravel()).reshape(len(cats), Mx)
        msk = msk * d.carried[blk, torch.as_tensor(store_r, device=device)].float()
    u = model.item_utility(torch.as_tensor(user, device=device), blk,
                           torch.zeros(len(cats), model.K, device=device), dlogp,
                           torch.as_tensor(st, device=device),
                           torch.as_tensor(week, device=device),
                           torch.as_tensor(store, device=device))
    ok = msk.sum(1) > 0
    iv = torch.logsumexp(u.masked_fill(msk == 0, -1e9), dim=1) + iv_scale
    iv = torch.where(ok, iv, torch.zeros_like(iv))
    return {"cat": ct[ok], "iv": iv[ok]}


def losses(model, bt, ib, iv_center):
    out = {}
    u = model.item_utility(bt["user"], bt["cand"], bt["ctx"], bt["dlogp"],
                           bt["state"], bt["week"], bt["store"])
    if bt.get("logq") is not None:
        u = u - bt["logq"]
    u = u.masked_fill(~bt["avail"], -1e9)
    if getattr(model, "item_loss", "softmax") == "ove":
        # Titsias' one-vs-each bound, which is what SHOPPER optimises by default
        # (`-likelihood 1` in shopper-src): log softmax_i >= sum_{i'} log sigmoid(u_i - u_i').
        #
        # This is a different LEARNING SIGNAL, not a different model.  In a sampled
        # softmax the normaliser is shared, so once lambda_j + theta_i.alpha_j
        # separates the decoys the probability saturates and every negative's gradient
        # goes to zero together -- the loss stops asking anything of the interaction
        # term.  One-vs-each makes the true item beat each negative INDIVIDUALLY, and
        # the gradient on a negative is 1 - sigmoid(u_i - u_i'), which stays large for
        # any competitor that is not yet clearly beaten.  That is the pressure the
        # interaction needs and never got.
        d_ = u[:, :1] - u[:, 1:]
        ok = bt["avail"][:, 1:]
        lse = torch.nn.functional.logsigmoid(d_)
        out["item"] = -(lse * ok).sum(1).div(ok.sum(1).clamp_min(1)).mean()
    else:
        out["item"] = -torch.log_softmax(u, dim=1)[:, 0].mean()

    if model.use_quantity:
        # units - 1 ~ Poisson(exp(z)), with its own price coefficient so the quantity
        # margin is not forced to share the incidence one.
        j = bt["cand"][:, 0]
        z = (model.q0[j]
             - (model.q_gamma[bt["user"]] * model.q_beta[j]).sum(-1) * bt["dlogp"][:, 0]
             + (model.q_state[j] * bt["state"][:, 0, :]).sum(-1)).clamp(-6.0, 4.0)
        k = (bt["units"] - 1.0).clamp_min(0)
        out["quantity"] = (torch.exp(z) - k * z + torch.lgamma(k + 1)).mean()

    if model.use_nest and ib is not None:
        kap = model.kappa()[ib["cat"]]
        lin = (model.c0[ib["cat"]]
               + (model.c_user[ib["user"]] * model.c_cat[ib["cat"]]).sum(-1)
               + (model.c_state[ib["cat"]] * ib["state"]).sum(-1)
               + kap * (ib["iv"] - model.iv_ref[ib["cat"]]))
        if model.cat_pair is not None and "pair_term" in ib:
            lin = lin + ib["pair_term"]
        if model.cat_ctx_scale is not None and "cat_ctx" in ib:
            lin = lin + model.cat_ctx_scale * (
                model.c_cat[ib["cat"]] * ib["cat_ctx"]).sum(-1)
        out["incidence"] = nn.functional.binary_cross_entropy_with_logits(lin, ib["y"])

        if model.use_breadth:
            # breadth - 1 ~ Poisson, on the categories actually bought
            pos = ib["y"] > 0
            if bool(pos.any()):
                zb = (model.b0[ib["cat"][pos]]
                      + model.b_user[ib["user"][pos]]
                      - model.b_price[ib["cat"][pos]] * ib["pdev"][pos]).clamp(-6.0, 3.0)
                kb = (ib["breadth"][pos] - 1.0).clamp_min(0)
                out["breadth"] = (torch.exp(zb) - kb * zb + torch.lgamma(kb + 1)).mean()
    return out


@torch.no_grad()
def evaluate(model, d, split, n_neg, rng, device, iv_center, max_baskets=3000,
             iv_cap=32):
    sp = d.splits[split]
    nb = min(sp["n_baskets"], max_baskets)
    idx = rng.choice(sp["n_baskets"], size=nb, replace=False)
    s_item = s_q = s_c = s_b = 0.0
    n_i = n_q = n_c = n_b = hits = 0
    for a in range(0, nb, 256):
        b = idx[a:a + 256]
        # clean_neg=False on purpose: evaluation keeps the ORIGINAL 21-way choice set
        # and no logQ reweighting, because every baseline is scored on that same set.
        # Both fixes change what the model LEARNS, not how it is measured.
        bt = make_batch(d, model, split, b, n_neg, rng, device, clean_neg=False)
        u = model.item_utility(bt["user"], bt["cand"], bt["ctx"], bt["dlogp"],
                               bt["state"], bt["week"], bt["store"])
        u = u.masked_fill(~bt["avail"], -1e9)
        lp = torch.log_softmax(u, dim=1)[:, 0]
        s_item += float(lp.sum()); n_i += lp.shape[0]
        hits += int((u.argmax(1) == 0).sum())
        L = losses(model, bt, None, iv_center)
        if "quantity" in L:
            s_q += float(L["quantity"]) * lp.shape[0]; n_q += lp.shape[0]
        if model.use_nest:
            ib = incidence_batch(d, model, split, b, 16, rng, device, iv_cap)
            if ib is not None:
                Lc = losses(model, bt, ib, iv_center)
                s_c += float(Lc["incidence"]) * ib["y"].shape[0]; n_c += ib["y"].shape[0]
                if "breadth" in Lc:
                    nb_ = int((ib["y"] > 0).sum())
                    s_b += float(Lc["breadth"]) * nb_; n_b += nb_
    return (s_item / max(n_i, 1), hits / max(n_i, 1),
            s_q / max(n_q, 1), s_c / max(n_c, 1), s_b / max(n_b, 1))


def main(a):
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    d = NestedData(IN, device=dev, placebo_price=a.placebo_price,
                   placebo_seed=a.seed)
    model = NestedModel(d, K=a.K, Kp=a.Kp, Kt=a.Kt, Ks=a.Ks, seed=a.seed,
                        use_nest=not a.no_nest, use_quantity=not a.no_quantity,
                        use_store=not a.no_store,
                        use_store_price=not (a.no_store_price or a.avail_only),
                        avail_only=a.avail_only, use_state=not a.no_state,
                        use_breadth=not a.no_breadth,
                        use_context=not a.no_context, ctx_agg=a.ctx_agg,
                        learn_ctx_scale=a.learn_ctx_scale,
                        use_cat_context=a.cat_context,
                        use_cat_pair=a.cat_pair, untie_rho=a.untie_rho,
                        prefix_context=a.prefix_context,
                        neg_in_cat=a.neg_in_cat, item_loss=a.item_loss).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"model {a.label}: K={a.K} nest={not a.no_nest} quantity={not a.no_quantity} "
        f"context={not a.no_context} store={not a.no_store} "
        f"avail_only={a.avail_only} -> {n_par:,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters,
                                                      eta_min=a.lr * 0.05)
    rng = np.random.default_rng(a.seed)
    ev = np.random.default_rng(12345)
    sp = d.splits["train"]
    hist, best, best_it = [], -1e9, -1
    t0 = time.time()
    for it in range(1, a.iters + 1):
        bidx = rng.integers(0, sp["n_baskets"], size=a.batch_baskets)
        bt = make_batch(d, model, "train", bidx, a.n_neg, rng, dev)
        ib = incidence_batch(d, model, "train", bidx, a.n_cat, rng, dev, a.iv_cap) \
            if model.use_nest else None
        # Freeze the per-category IV reference once the utilities have settled.  Any
        # fixed constant is absorbed by c0, so this does not change the fit -- it only
        # guarantees training and generation subtract the same thing.
        if model.use_nest and it == a.iv_warmup:
            with torch.no_grad():
                acc = torch.zeros(d.C, device=dev)
                cnt = torch.zeros(d.C, device=dev)
                # The reference must come from a UNIFORM sample of categories.  Taking
                # it from incidence_batch instead uses the case-control sample, which
                # over-represents categories the trip bought and whose IV is therefore
                # higher; the frozen reference then sits above the IV of a typical
                # category, (IV - ref) is negative at generation time, and most
                # generated baskets come out empty.  That is what happened on the
                # first attempt: kappa 1.411 -> 0.763 and 4.0 categories against 6.5.
                for _ in range(8):
                    bb = rng.integers(0, sp["n_baskets"], size=a.batch_baskets)
                    w = uniform_iv(d, model, "train", bb, a.iv_ref_cats, rng, dev,
                                   a.iv_cap)
                    if w is None:
                        continue
                    acc.index_add_(0, w["cat"], w["iv"])
                    cnt.index_add_(0, w["cat"], torch.ones_like(w["iv"]))
                model.iv_ref.copy_(torch.where(cnt > 0, acc / cnt.clamp_min(1),
                                               torch.zeros_like(acc)))
                model.iv_ref_set.fill_(1.0)
            log(f"  froze IV reference at iteration {it}: "
                f"mean {float(model.iv_ref.mean()):.3f}, "
                f"{int((cnt > 0).sum())} of {d.C} categories observed")
        L = losses(model, bt, ib, a.iv_center)
        loss = (L["item"] + a.w_quantity * L.get("quantity", 0.0)
                + a.w_incidence * L.get("incidence", 0.0)
                + a.w_breadth * L.get("breadth", 0.0)
                # divided by a FIXED constant, not bt["cand"].shape[0]: the row count
                # varies 1157-1797 between batches, which made the effective weight
                # decay fluctuate by 1.55x within a single run.
                + (a.l2 * model.l2_repr() + a.l2_price * model.l2_price()
                   + a.l2_cat_pair * model.l2_cat_pair()
                   + a.l2_rho * model.l2_rho())
                / a.l2_denom)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if it % a.eval_every == 0 or it == a.iters:
            vi, vacc, vq, vc, vb = evaluate(model, d, "validation", a.n_neg, ev, dev,
                                            a.iv_center, iv_cap=a.iv_cap)
            kap = float(model.kappa().detach().median()) if model.use_nest else float("nan")
            brd = float(1.0 + torch.exp(model.b0.detach()).mean()) \
                if model.use_breadth else float("nan")
            pc = float((model.gamma.detach() @ model.beta.detach().T).median())
            qc = float((model.q_gamma.detach() @ model.q_beta.detach().T).median()) \
                if model.use_quantity else float("nan")
            hist.append({"iter": it, "val_item": vi, "val_top1": vacc,
                         "val_quantity_nll": vq, "val_incidence_nll": vc,
                         "val_breadth_nll": vb,
                         "kappa_median": kap, "price_coef": pc,
                         "quantity_price_coef": qc, "secs": time.time() - t0})
            # breadth is in the selection score now.  It was trained but had no
            # held-out number, so checkpointing was blind to the head that decides how
            # many DIFFERENT products a category contributes -- the one that fixed
            # generated basket size (6.94 -> 8.27 items).
            score = vi - a.w_quantity * vq - a.w_incidence * vc - a.w_breadth * vb
            star = ""
            if score > best:
                best, best_it = score, it
                torch.save(model.state_dict(), os.path.join(OUT, f"{a.label}_nested.pt"))
                star = " *"
            log(f"  it {it:5d}  item {vi:.4f}  top1 {vacc:.3f}  qNLL {vq:.4f}  "
                f"cNLL {vc:.4f}  bNLL {vb:.4f}  kappa {kap:.3f}  breadth {brd:.3f}  "
                f"price {pc:+.3f}{star}")

    model.load_state_dict(torch.load(os.path.join(OUT, f"{a.label}_nested.pt"),
                                     map_location=dev))
    ti, tacc, tq, tc, tb = evaluate(model, d, "test", a.n_neg,
                                    np.random.default_rng(999), dev, a.iv_center, 6000,
                                    iv_cap=a.iv_cap)
    log(f"best at {best_it}; TEST item {ti:.4f} top1 {tacc:.3f} qNLL {tq:.4f} cNLL {tc:.4f}")
    with open(os.path.join(OUT, f"{a.label}_nested_history.json"), "w") as f:
        json.dump({"config": vars(a), "history": hist, "n_parameters": n_par,
                   "best_iter": best_it, "test_item": ti, "test_top1": tacc,
                   "test_quantity_nll": tq, "test_incidence_nll": tc,
                   "test_breadth_nll": tb}, f, indent=2)
    log(f"saved {a.label}_nested.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="nested")
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--Kp", type=int, default=8)
    p.add_argument("--Kt", type=int, default=8)
    p.add_argument("--Ks", type=int, default=4)
    p.add_argument("--n-neg", type=int, default=20)
    p.add_argument("--n-cat", type=int, default=16,
                   help="categories sampled uniformly per trip for the "
                        "incidence head")
    p.add_argument("--iv-ref-cats", type=int, default=24,
                   help="categories sampled uniformly per trip when freezing the "
                        "per-category IV reference")
    p.add_argument("--iv-warmup", type=int, default=500,
                   help="iteration at which the per-category IV reference is frozen")
    p.add_argument("--iv-cap", type=int, default=32,
                   help="items sampled per category for the inclusive value; 0 uses "
                        "the full padded block, which is 5x slower for no gain")
    p.add_argument("--batch-baskets", type=int, default=192)
    p.add_argument("--iters", type=int, default=12000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--l2", type=float, default=1e-2)
    p.add_argument("--l2-denom", type=float, default=1500.0,
                   help="fixed divisor for the L2 terms; was the batch row count, "
                        "which fluctuates 1.55x and made regularisation batch-dependent")
    p.add_argument("--l2-price", type=float, default=1e-4)
    p.add_argument("--l2-cat-pair", type=float, default=1e-2,
                   help="L2 on the category-pair matrix, separated from the "
                        "representation block: too weak and it crowds out kappa*IV, "
                        "which is the only channel price reaches incidence through")
    p.add_argument("--w-quantity", type=float, default=1.0)
    p.add_argument("--w-incidence", type=float, default=1.0)
    p.add_argument("--iv-center", type=float, default=0.0,
                   help="extra shift on the (already batch-centred) inclusive value")
    p.add_argument("--placebo-price", default="none",
                   choices=["none", "permute", "swap"])
    p.add_argument("--no-nest", action="store_true")
    p.add_argument("--no-quantity", action="store_true")
    p.add_argument("--w-breadth", type=float, default=1.0)
    p.add_argument("--no-breadth", action="store_true",
                   help="ablate the breadth head; generation then produces ~1.09 "
                        "distinct items per purchased category against a real 1.284")
    p.add_argument("--ctx-agg", default="mean", choices=["mean", "sum"],
                   help="pool the basket context by mean (divides by n-1) or by sum. "
                        "sum makes E(S) one pairwise energy over all basket sizes and "
                        "stops a strong pair being diluted by 1/(n-1)")
    p.add_argument("--item-loss", default="softmax", choices=["softmax", "ove"],
                   help="'ove' is Titsias' one-vs-each bound, SHOPPER's default. "
                        "Evaluation always uses the sampled softmax, so item "
                        "log-likelihood stays comparable across this flag")
    p.add_argument("--neg-in-cat", type=float, default=0.0,
                   help="fraction of negatives drawn from the TRUE item's own "
                        "category, so popularity and taste cannot separate them and "
                        "only the basket context can. NOTE: this changes the negative "
                        "distribution, so item log-likelihood is no longer comparable "
                        "across settings; use 31_benchmark.py, which fixes its own")
    p.add_argument("--untie-rho", action="store_true",
                   help="SHOPPER's asymmetric interaction rho_j . alpha_k instead of "
                        "the tied alpha_j . alpha_k, so complements and substitutes "
                        "are representable")
    p.add_argument("--l2-rho", type=float, default=1e-2,
                   help="L2 on rho, separated from the representation block")
    p.add_argument("--prefix-context", action="store_true",
                   help="context is the mean over a random PREFIX of the basket "
                        "(SHOPPER eq. 6, one permutation sample) rather than "
                        "leave-one-out over all of it; matches sequential generation")
    p.add_argument("--cat-pair", action="store_true",
                   help="learn the full C x C matrix of category-pair log-odds in the "
                        "incidence head; 17,578 free parameters, no embedding, so it "
                        "can express complementarity rather than only similarity")
    p.add_argument("--cat-context", action="store_true",
                   help="give the incidence head a leave-one-out context over the "
                        "other categories in the basket, so which categories co-occur "
                        "is modelled rather than left to independent Bernoulli draws")
    p.add_argument("--learn-ctx-scale", action="store_true",
                   help="a learnable scalar on the interaction term")
    p.add_argument("--no-context", action="store_true",
                   help="ablate the tied within-basket interaction alpha_j . alpha_bar")
    p.add_argument("--no-state", action="store_true",
                   help="ablate the household recency state")
    p.add_argument("--no-store", action="store_true")
    p.add_argument("--no-store-price", action="store_true",
                   help="keep availability and store affinity, drop store-level prices")
    p.add_argument("--avail-only", action="store_true",
                   help="availability mask only: no store prices, no store affinity. "
                        "Isolates how much of the store gain is just a smaller choice "
                        "set rather than real information")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
