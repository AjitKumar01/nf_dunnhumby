"""
Nested Factorization in PyTorch, reading the same input files as src/bemb_loc.

Why a re-implementation rather than the authors' C++:
  * the shipped bemb_loc binary needs GSL and, more importantly, it stores prices
    as an item x session matrix -- prices common to all users.  The paper's
    *category* stage feeds the inclusive value IV_ict into that same slot, and
    IV varies across users.  That stage was run with the user-varying ("TTFM")
    build, which is not in this repository.  Stage 2 therefore needs new code
    whatever we do.
  * dunnhumby's extra signals (display, mailer, household coupon eligibility) are
    additional time-varying attributes; bemb_loc has exactly one such slot (price).

The model follows the paper exactly.

Product choice, conditional on buying from category c (paper eq. 4-6):

    u_ijt = lambda0_j + theta_i . beta_j + W_i . rho_j + sigma_i . X_j
            - (gamma_i . lambda_j) * price_jt
            [+ (gammaD_i . lambdaD_j) * display_jt
             + (gammaM_i . lambdaM_j) * mailer_jt
             + (gammaC_i . lambdaC_j) * coupon_ijt ]      <- dunnhumby extensions
    P(j | category purchase) = softmax over the items of the category

Category choice (paper eq. 7-9):

    IV_ict = log sum_j exp(u_ijt)
    u_ict  = vartheta_i . beta_c + W_i . rho_c + psi_i . X_c
             - (phi_i . lambda_c) * IV_ict + mu_c . delta_t + w_c,weekday
    P(buy from c) = sigmoid(u_ict)

Inference is mean-field variational Bayes with the reparameterisation trick and
stochastic gradients, as in paper sec. 3.2.4 / app. 8.2: every latent is an
independent Gaussian with its own mean and variance, the prior is N(0, s2), and
the ELBO is maximised with Adam.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- io
def read_tsv(path, cols, dtypes):
    import pandas as pd
    return pd.read_csv(path, sep="\t", header=None, names=cols, dtype=dtypes)


@dataclass
class NFData:
    """Everything the two stages need, already on the target device."""
    n_users: int
    n_items: int
    n_cats: int
    n_sessions: int
    n_periods: int          # pair-weeks (the paper's week trend index)
    n_weekdays: int
    item_cat: torch.Tensor      # [J] category of each item
    cat_items: torch.Tensor     # [C, Jmax] padded item ids
    cat_mask: torch.Tensor      # [C, Jmax] 1 where the slot is a real item
    price: torch.Tensor         # [J, S]  loyalty price -- what the shopper faces
    base_price: torch.Tensor    # [J, S]  regular posted price, before promotion
    promo_depth: torch.Tensor   # [J, S]  1 - price/base_price, in [0, 1)
    W: torch.Tensor             # [N, UC] user observables
    X: torch.Tensor             # [J, IC] item observables
    sess_period: torch.Tensor   # [S]
    sess_weekday: torch.Tensor  # [S]
    obs: dict = field(default_factory=dict)   # split -> (user, item, session) tensors
    trips: dict = field(default_factory=dict)  # split -> (user, session) tensors
    extras: dict = field(default_factory=dict)  # name -> [J, S] tensor
    coupon: torch.Tensor = None                # sparse [N, J, S] as intervals


def load(indir, device="cpu", extras=()):
    import os
    import pandas as pd

    it = pd.read_csv(os.path.join(indir, "itemGroup.tsv"), sep="\t", header=None,
                     names=["item_id", "group_id"])
    n_items = int(it.item_id.max()) + 1
    n_cats = int(it.group_id.max()) + 1
    item_cat = np.zeros(n_items, dtype=np.int64)
    item_cat[it.item_id.values] = it.group_id.values

    jmax = int(np.bincount(item_cat).max())
    cat_items = np.zeros((n_cats, jmax), dtype=np.int64)
    cat_mask = np.zeros((n_cats, jmax), dtype=np.float32)
    for c in range(n_cats):
        js = np.where(item_cat == c)[0]
        cat_items[c, :len(js)] = js
        cat_mask[c, :len(js)] = 1.0

    sd = pd.read_csv(os.path.join(indir, "sess_days.tsv"), sep="\t", header=None,
                     names=["session_id", "day_id", "weekday_id", "hour"])
    n_sessions = int(sd.session_id.max()) + 1
    sess_period = np.zeros(n_sessions, dtype=np.int64)
    sess_weekday = np.zeros(n_sessions, dtype=np.int64)
    sess_period[sd.session_id.values] = sd.day_id.values
    sess_weekday[sd.session_id.values] = sd.weekday_id.values

    def read_is(fname):
        d = pd.read_csv(os.path.join(indir, fname), sep="\t", header=None,
                        names=["item_id", "session_id", "v"])
        m = np.zeros((n_items, n_sessions), dtype=np.float32)
        m[d.item_id.values, d.session_id.values] = d.v.values
        return m

    price = read_is("item_sess_price.tsv")
    # Written by 03_make_model_inputs.py.  Older input directories predate them, so
    # fall back to "no promotion ever" rather than failing -- the nf models never
    # touch these, and a substitution model run against an old directory should say
    # so loudly at fit time instead of crashing here.
    if os.path.exists(os.path.join(indir, "item_sess_base_price.tsv")):
        base_price = read_is("item_sess_base_price.tsv")
        promo_depth = read_is("item_sess_promo_depth.tsv")
    else:
        base_price, promo_depth = price.copy(), np.zeros_like(price)

    ou = pd.read_csv(os.path.join(indir, "obsUser.tsv"), sep="\t", header=None)
    n_users = int(ou[0].max()) + 1
    W = np.zeros((n_users, ou.shape[1] - 1), dtype=np.float32)
    W[ou[0].values.astype(int)] = ou.iloc[:, 1:].values
    oi = pd.read_csv(os.path.join(indir, "obsItem.tsv"), sep="\t", header=None)
    X = np.zeros((n_items, oi.shape[1] - 1), dtype=np.float32)
    X[oi[0].values.astype(int)] = oi.iloc[:, 1:].values

    d = NFData(
        n_users=n_users, n_items=n_items, n_cats=n_cats, n_sessions=n_sessions,
        n_periods=int(sess_period.max()) + 1, n_weekdays=int(sess_weekday.max()) + 1,
        item_cat=torch.as_tensor(item_cat, device=device),
        cat_items=torch.as_tensor(cat_items, device=device),
        cat_mask=torch.as_tensor(cat_mask, device=device),
        price=torch.as_tensor(price, device=device),
        base_price=torch.as_tensor(base_price, device=device),
        promo_depth=torch.as_tensor(promo_depth, device=device),
        W=torch.as_tensor(W, device=device),
        X=torch.as_tensor(X, device=device),
        sess_period=torch.as_tensor(sess_period, device=device),
        sess_weekday=torch.as_tensor(sess_weekday, device=device),
    )

    for name, fname in [("display", "item_sess_display.tsv"), ("mailer", "item_sess_mailer.tsv")]:
        if name in extras:
            path = os.path.join(indir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"extra '{name}' requested but {path} is missing; "
                                        "run 04_extras.py first")
            d.extras[name] = torch.as_tensor(read_is(fname), device=device)

    if "coupon" in extras:
        path = os.path.join(indir, "coupon_campaigns.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"extra 'coupon' requested but {path} is missing; "
                                    "run 04_extras.py first")
        z = np.load(path)
        d.coupon = {k: torch.as_tensor(z[k].astype(np.float32), device=device)
                    for k in ["U", "P", "S", "w"]}

    for split, fname in [("train", "train.tsv"), ("validation", "validation.tsv"), ("test", "test.tsv")]:
        o = pd.read_csv(os.path.join(indir, fname), sep="\t", header=None,
                        names=["user_id", "item_id", "session_id", "units"])
        d.obs[split] = tuple(torch.as_tensor(o[k].values, device=device)
                             for k in ["user_id", "item_id", "session_id"])
        t = o[["user_id", "session_id"]].drop_duplicates()
        d.trips[split] = tuple(torch.as_tensor(t[k].values, device=device)
                               for k in ["user_id", "session_id"])
    return d


def coupon_weight(coupon, users, items, sessions):
    """Coupon eligibility weight for a batch of (user, item, session).

    Eligibility factors as household-in-campaign x product-in-campaign x
    day-in-window, so it is evaluated as a max over the ~21 campaigns rather than
    materialising a user x item x session tensor.

    users [B], items [B, M], sessions [B]  ->  [B, M]
    """
    if coupon is None:
        return None
    U, P, S, w = coupon["U"], coupon["P"], coupon["S"], coupon["w"]
    a = U[users].unsqueeze(1) * S[sessions].unsqueeze(1)   # [B, 1, Kc]
    b = P[items]                                           # [B, M, Kc]
    return ((a * b) * w).amax(dim=-1)


# ------------------------------------------------------------------- variational
class GaussianBlock(nn.Module):
    """Mean-field Gaussian variational factor for one block of latents."""

    def __init__(self, shape, prior_var=1.0, prior_mean=0.0, init_sd=0.1, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.mu = nn.Parameter(torch.randn(shape, generator=g) * init_sd + prior_mean)
        self.log_sd = nn.Parameter(torch.full(shape, np.log(init_sd), dtype=torch.float32))
        self.prior_var = prior_var
        self.prior_mean = prior_mean

    def sample(self, stochastic=True):
        if not stochastic:
            return self.mu
        return self.mu + torch.exp(self.log_sd) * torch.randn_like(self.mu)

    def kl(self):
        var = torch.exp(2 * self.log_sd)
        pv = self.prior_var
        return 0.5 * torch.sum(
            (var + (self.mu - self.prior_mean) ** 2) / pv - 1.0 - torch.log(var / pv))


def load_stage1_state(model, path, map_location="cpu"):
    """Load a stage-1 checkpoint, tolerating buffers that used to be persisted.

    Checkpoints written before the price buffers were marked non-persistent carry
    log_base / log_disc / log_price_dev as saved tensors.  They are recomputed from
    the data at construction, so drop anything the module no longer expects -- but
    keep strict=True for everything else, so a genuinely missing parameter is still
    an error rather than a silently half-initialised model.
    """
    import torch as _t
    sd = _t.load(path, map_location=map_location)
    want = set(model.state_dict().keys())
    model.load_state_dict({k: v for k, v in sd.items() if k in want}, strict=True)
    return model


# ----------------------------------------------------------------- stage 1 model
class ProductChoice(nn.Module):
    """Choice of item within a category, conditional on buying from the category."""

    def __init__(self, data: NFData, K=80, Kp=20, use_user_obs=True, use_item_obs=False,
                 extras=(), item_intercept=True, prior_var=1.0, intercept_var=None,
                 price_prior_var=None, price_prior_mean=0.5, homogeneous=False,
                 scale_prior=True, pool_across_categories=True, seed=0,
                 Ks=0, sub_prior_var=0.05, price_split=False):
        super().__init__()
        self.d = data
        self.K, self.Kp = (0, Kp) if homogeneous else (K, Kp)
        self.homogeneous = homogeneous
        # Ks > 0 adds the substitution kernel; price_split separates the regular
        # price from the promotional cut.  Both default off so `nf` is unchanged.
        self.Ks = int(Ks)
        self.price_split = bool(price_split)
        # pool_across_categories=False gives every (household, category) its own
        # latent vectors, so nothing is shared across categories.  That is the
        # category-by-category benchmark the paper argues against.
        self.pool = pool_across_categories
        self.nrow = data.n_users if pool_across_categories else data.n_users * data.n_cats
        self.extras = tuple(extras)
        N, J = data.n_users, data.n_items
        price_prior_var = prior_var if price_prior_var is None else price_prior_var

        def blk(shape, **kw):
            kw.setdefault("prior_var", prior_var)
            return GaussianBlock(shape, seed=seed, **kw)

        def factor_var(target, dim):
            """Prior variance per factor so the inner product has variance `target`.

            Var(theta.beta) = dim * s^4 with equal factor variances, so s^2 =
            sqrt(target/dim).  Without this the prior on the utility widens as
            sqrt(K) and larger K is barely regularised at all -- which is what
            makes K=80 unusable on a sample this size.
            """
            return float(np.sqrt(target / max(dim, 1))) if scale_prior else target

        NR = self.nrow
        self.lambda0 = blk((J,), prior_var=intercept_var or prior_var) if item_intercept else None
        if self.K > 0:
            v = factor_var(prior_var, self.K)
            self.theta = blk((NR, self.K), prior_var=v)
            self.beta = blk((J, self.K), prior_var=v)
        self.rho = blk((J, data.W.shape[1])) if use_user_obs else None
        self.sigma = blk((NR, data.X.shape[1])) if use_item_obs else None

        # Price.  A bilinear term whose two factors both start at zero has zero
        # gradient in both, so gamma_i . lambda_j stays at zero and the model
        # simply learns no price response.  bemb_loc exposes '-meangamma' and
        # '-meanbeta' for exactly this; here both factors get prior mean
        # sqrt(price_prior_mean / Kp), so the coefficient starts at
        # price_prior_mean (positive: demand slopes down) and the data moves it.
        pm = float(np.sqrt(max(price_prior_mean, 0.0) / max(Kp, 1)))
        if homogeneous:
            self.price_coef = blk((1,), prior_var=price_prior_var,
                                  prior_mean=price_prior_mean)
        else:
            v = factor_var(price_prior_var, Kp)
            self.gamma = blk((NR, Kp), prior_var=v, prior_mean=pm)
            self.lam = blk((J, Kp), prior_var=v, prior_mean=pm)
        # Change 3 -- split the own-price effect into the regular posted price and
        # the promotional cut.  log price = log(base) + log(1 - depth), so writing
        # the utility as
        #     -alpha_b * log(base)  +  alpha_p * (-log(1 - depth))
        # nests the pooled model exactly: alpha_b == alpha_p reproduces
        # -alpha * log(price).  Whether the two differ is then a testable question
        # rather than an assumption, and it matters because the EDA shows the base
        # price moves in only ~7% of item-weeks against ~18% for promotion depth --
        # so the pooled coefficient is mostly a *promotional* elasticity, and any
        # counterfactual about regular prices is extrapolating.
        if self.price_split and not homogeneous:
            v = factor_var(price_prior_var, Kp)
            self.gamma_promo = blk((NR, Kp), prior_var=v, prior_mean=pm)
            self.lam_promo = blk((J, Kp), prior_var=v, prior_mean=pm)
        elif self.price_split and homogeneous:
            self.promo_coef = blk((1,), prior_var=price_prior_var,
                                  prior_mean=price_prior_mean)

        # Change 1 -- the substitution kernel.  Stage 1 as the paper writes it makes
        # item j's utility depend on j's own price only, so at the household level
        # d log P(j) / d p_k = alpha_k P(k): identical for every j, i.e. IIA.  Items
        # substitute in proportion to market share and nothing else, which is why
        # the paper's own sub-commodity test fails to replicate here.  psi_j . psi_k
        # lets j's utility respond directly to k's price, so the strength of
        # substitution is learned rather than implied by shares.  psi is never shown
        # SUB_COMMODITY_DESC, so that test stays honest.
        #
        # The prior is deliberately tight (sub_prior_var defaults to 0.05): the
        # null of IIA should be given up only where the data insists.
        if self.Ks > 0:
            v = factor_var(sub_prior_var, self.Ks)
            self.psi = blk((J, self.Ks), prior_var=v)

        # Precomputed [J, S] price transforms, so the hot path does no logs.
        #
        # Derive both from price and base_price rather than reading promo_depth back
        # off disk, so that log_base - log_disc == log(price) *exactly*.  Taking the
        # stored depth instead leaves a residual: it is clipped at zero, and on the
        # 34 cells (0.035%) where the weekly modal base came out below the weekly
        # modal loyalty price the clip breaks the identity by up to log 2.  Clamping
        # the discount at zero here treats those cells as "no promotion", which is
        # the right reading, and keeps the split an exact reparameterisation of the
        # pooled log-price model everywhere else.
        eps = 1e-4
        lp = torch.log(data.price.clamp_min(eps))
        log_disc = (torch.log(data.base_price.clamp_min(eps)) - lp).clamp_min(0.0)
        # persistent=False: these are deterministic functions of the price panel, not
        # fitted quantities.  Saving them would bloat every checkpoint and, worse,
        # make checkpoints trained before this change unloadable -- the module would
        # demand keys the old state_dict has never heard of.
        self.register_buffer("log_disc", log_disc, persistent=False)      # >= 0
        self.register_buffer("log_base", lp + log_disc, persistent=False)
        # The kernel reads *deviations*, not levels: a competitor's price level is
        # a fixed effect that the item intercept already carries, and only movement
        # identifies substitution.  Centring within item also keeps the term at zero
        # in a week where nothing moved.
        self.register_buffer("log_price_dev", lp - lp.mean(dim=1, keepdim=True),
                             persistent=False)

        for e in self.extras:
            if homogeneous:
                setattr(self, f"{e}_coef", blk((1,), prior_var=price_prior_var,
                                               prior_mean=price_prior_mean))
            else:
                v = factor_var(price_prior_var, Kp)
                setattr(self, f"g_{e}", blk((NR, Kp), prior_var=v, prior_mean=pm))
                setattr(self, f"l_{e}", blk((J, Kp), prior_var=v, prior_mean=pm))

    # ---- utilities for a set of (user, session) rows against a padded item block
    def utility(self, users, sessions, items, stoch=True, draws=None, mask=None):
        """users [B], sessions [B], items [B, M] -> utilities [B, M].

        `mask` [B, M] marks the real slots in a padded block.  It is required once
        the substitution kernel is on: the kernel sums over competitors, and the
        padding (slot 0 of `cat_items`) would otherwise be counted as a real rival.
        Callers pass blocks in two different layouts -- one category per row, and
        the whole `cat_items` grid flattened to [B, C*Jmax] -- so the kernel groups
        by each slot's own category rather than assuming either shape.
        """
        d = self.d
        s = draws if draws is not None else {}
        if self.Ks > 0 and mask is None:
            raise ValueError(
                "utility() needs `mask` when the substitution kernel is active: the "
                "competitor sum cannot tell real items from block padding without it")

        def get(name):
            if name in s:
                return s[name]
            blk = getattr(self, name, None)
            return None if blk is None else blk.sample(stoch)

        # row index into the per-household blocks: the household itself when
        # pooling, otherwise the (household, category) cell
        rows = (users.unsqueeze(1).expand_as(items) if self.pool
                else users.unsqueeze(1) * d.n_cats + d.item_cat[items])

        u = torch.zeros(items.shape, dtype=torch.float32, device=items.device)
        if self.lambda0 is not None:
            u = u + get("lambda0")[items]
        if self.K > 0:
            th = get("theta")[rows]                       # [B, M, K]
            be = get("beta")[items]                       # [B, M, K]
            u = u + (th * be).sum(-1)
        if self.rho is not None:
            u = u + torch.einsum("bw,bmw->bm", d.W[users], get("rho")[items])
        if self.sigma is not None:
            u = u + (get("sigma")[rows] * d.X[items]).sum(-1)

        sess_col = sessions.unsqueeze(1)
        if self.price_split:
            # log(price) = log(base) - log_disc, so equal coefficients here reproduce
            # the pooled -alpha*log(price) exactly; the split only lets them differ.
            lb = self.log_base[items, sess_col]            # [B, M]
            ld = self.log_disc[items, sess_col]            # [B, M] >= 0
            if self.homogeneous:
                u = u - get("price_coef") * lb + get("promo_coef") * ld
            else:
                u = u - (get("gamma")[rows] * get("lam")[items]).sum(-1) * lb
                u = u + (get("gamma_promo")[rows] * get("lam_promo")[items]).sum(-1) * ld
        else:
            p = d.price[items, sess_col]                   # [B, M]
            if self.homogeneous:
                u = u - get("price_coef") * p
            else:
                u = u - (get("gamma")[rows] * get("lam")[items]).sum(-1) * p

        # Substitution kernel: item j's utility responds to every *other* item's
        # price move in the same category, with strength psi_j . psi_k.
        #   sum_{k != j} (psi_j . psi_k) g_k  =  psi_j . (sum_k psi_k g_k) - |psi_j|^2 g_j
        # so the whole thing is two contractions rather than a loop over pairs.
        if self.Ks > 0:
            psi = get("psi")[items]                        # [B, M, Ks]
            g = self.log_price_dev[items, sess_col] * mask  # [B, M]
            # Substitution is a within-category story, but callers may hand over the
            # whole grid, so accumulate psi_k * g_k into the slot's own category and
            # read it back per slot.  Padding contributes nothing because g is
            # already masked to zero.
            slot_cat = d.item_cat[items]                   # [B, M]
            B, M = items.shape
            agg = torch.zeros(B, d.n_cats, self.Ks, dtype=psi.dtype, device=psi.device)
            agg.scatter_add_(1, slot_cat.unsqueeze(-1).expand(B, M, self.Ks),
                             psi * g.unsqueeze(-1))        # [B, C, Ks]
            agg_slot = torch.gather(agg, 1, slot_cat.unsqueeze(-1).expand(B, M, self.Ks))
            cross = (psi * agg_slot).sum(-1) - (psi * psi).sum(-1) * g
            u = u + cross * mask

        for e in self.extras:
            if e == "coupon":
                v = coupon_weight(d.coupon, users, items, sessions)
            else:
                v = d.extras[e][items, sessions.unsqueeze(1)]
            if self.homogeneous:
                u = u + get(f"{e}_coef") * v
            else:
                u = u + (get(f"g_{e}")[rows] * get(f"l_{e}")[items]).sum(-1) * v
        return u

    def choice_block(self, users, items_chosen, sessions):
        """Padded item block for the chosen items' categories."""
        c = self.d.item_cat[items_chosen]
        return self.d.cat_items[c], self.d.cat_mask[c]

    def log_prob(self, users, items, sessions, stoch=True):
        block, mask = self.choice_block(users, items, sessions)
        u = self.utility(users, sessions, block, stoch=stoch, mask=mask)
        u = u.masked_fill(mask == 0, -1e9)
        logp = torch.log_softmax(u, dim=1)
        pos = (block == items.unsqueeze(1))
        return logp[pos]

    def price_coefficients(self, users, items):
        """gamma_i . lambda_j -- the price coefficient, for a batch of (user, items)."""
        if self.homogeneous:
            return self.price_coef.mu.reshape(1, 1).expand(users.shape[0], items.shape[1])
        rows = (users.unsqueeze(1).expand_as(items) if self.pool
                else users.unsqueeze(1) * self.d.n_cats + self.d.item_cat[items])
        return (self.gamma.mu[rows] * self.lam.mu[items]).sum(-1)

    def kl(self):
        return sum(m.kl() for m in self.modules() if isinstance(m, GaussianBlock))

    @torch.no_grad()
    def mean_inclusive_values(self, chunk=16):
        """[N, C] average of IV_ict over every session.

        Subtracting this from IV leaves only the part that moves with prices.  The
        level of IV_ict is a household's affinity for the category, which is exactly
        what vartheta_i . beta_c already represents in the category-stage utility;
        leaving both in makes the nesting coefficient a contrast between two
        collinear terms and it is routinely estimated with the wrong sign.  After
        centring, the nesting coefficient is identified from the Sunday->Monday
        price variation alone -- the same variation the whole design rests on.
        """
        d = self.d
        users = torch.arange(d.n_users, device=d.price.device)
        total = torch.zeros((d.n_users, d.n_cats), device=d.price.device)
        for s0 in range(0, d.n_sessions, chunk):
            for s in range(s0, min(s0 + chunk, d.n_sessions)):
                ss = torch.full_like(users, s)
                total += self.inclusive_values(users, ss)
        return total / d.n_sessions

    @torch.no_grad()
    def inclusive_values(self, users, sessions, chunk=4096):
        """IV_ict for every category, for the given (user, session) rows -> [B, C]."""
        d = self.d
        out = []
        for a in range(0, users.shape[0], chunk):
            uu, ss = users[a:a + chunk], sessions[a:a + chunk]
            B = uu.shape[0]
            items = d.cat_items.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
            mask = d.cat_mask.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
            u = self.utility(uu, ss, items, stoch=False, mask=mask)
            u = u.masked_fill(mask == 0, -1e9)
            u = u.reshape(B, d.n_cats, -1)
            out.append(torch.logsumexp(u, dim=2))
        return torch.cat(out, 0)


# ----------------------------------------------------------------- stage 2 model
class CategoryChoice(nn.Module):
    """Binary choice of whether to buy anything from each category on a trip."""

    def __init__(self, data: NFData, K=40, Kiv=40, Ktime=10, use_user_obs=False,
                 prior_var=1.0, homogeneous=False, scale_prior=True, seed=0):
        super().__init__()
        self.d = data
        self.K = 0 if homogeneous else K
        self.homogeneous = homogeneous
        N, C = data.n_users, data.n_cats

        def blk(shape, **kw):
            kw.setdefault("prior_var", prior_var)
            return GaussianBlock(shape, seed=seed, **kw)

        def factor_var(target, dim):
            return float(np.sqrt(target / max(dim, 1))) if scale_prior else target

        self.alpha_c = blk((C,), prior_var=10.0)
        if self.K > 0:
            v = factor_var(prior_var, self.K)
            self.vartheta = blk((N, self.K), prior_var=v)
            self.beta_c = blk((C, self.K), prior_var=v)
        self.rho_c = blk((C, data.W.shape[1])) if use_user_obs else None
        if homogeneous:
            self.iv_coef = blk((1,), prior_mean=1.0)
        else:
            v = factor_var(prior_var, Kiv)
            # The paper writes the inclusive-value term with a minus sign (eq. 8)
            # because it is fed through the code path built for price; the fitted
            # coefficient then comes out negative.  Here it is written with a plus,
            # so phi_i . lambda_c is the nesting coefficient directly: 1 collapses to
            # a plain logit, 0 means no substitution across the nest boundary.
            # Prior means are set so the product starts near 1.
            m0 = float(1.0 / np.sqrt(max(Kiv, 1)))
            self.phi = blk((N, Kiv), prior_var=v, prior_mean=m0)
            self.lam_c = blk((C, Kiv), prior_var=v, prior_mean=m0)
        self.mu_c = blk((C, Ktime))
        self.delta_t = blk((data.n_periods, Ktime), prior_var=0.01)
        self.w_ct = blk((C, data.n_weekdays))

    def nesting_coef(self, users):
        """phi_i . lambda_c -- the nesting coefficient, [B, C]."""
        if self.homogeneous:
            return self.iv_coef.mu.expand(users.shape[0], self.d.n_cats)
        return self.phi.mu[users] @ self.lam_c.mu.T

    def logits(self, users, sessions, iv, stoch=True):
        """users [B], sessions [B], iv [B, C] -> logits [B, C]."""
        d = self.d
        z = self.alpha_c.sample(stoch).unsqueeze(0).expand(users.shape[0], -1).clone()
        if self.K > 0:
            z = z + self.vartheta.sample(stoch)[users] @ self.beta_c.sample(stoch).T
        if self.rho_c is not None:
            z = z + d.W[users] @ self.rho_c.sample(stoch).T
        if self.homogeneous:
            z = z + self.iv_coef.sample(stoch) * iv
        else:
            z = z + (self.phi.sample(stoch)[users] @ self.lam_c.sample(stoch).T) * iv
        z = z + self.delta_t.sample(stoch)[d.sess_period[sessions]] @ self.mu_c.sample(stoch).T
        z = z + self.w_ct.sample(stoch)[:, d.sess_weekday[sessions]].T
        return z

    def kl(self):
        return sum(m.kl() for m in self.modules() if isinstance(m, GaussianBlock))
