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
IN = os.path.join(HERE, "..", "..", os.environ.get("NF_BASKET_INPUT", "basket_input"))
OUT = os.path.join(HERE, "..", "..", "out")
FIG = os.path.join(HERE, "..", "..", "figures")

PAL = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b",
       "green": "#2a9d8f", "amber": "#e9c46a"}


class _Baskets(list):
    """Generated baskets, carrying the trip index each one came from."""
    def __init__(self, items, trips):
        super().__init__(items)
        self.trips = np.asarray(trips)


DEBUG_PASS1 = None


def log(m):
    print(f"[28] {m}", flush=True)


def load(label, d, dev):
    """Rebuild a checkpoint's architecture from its saved config.

    Every flag that changes the model must be passed here.  Seven were not, and the
    failure was silent for the one that matters: a model trained with --no-context loaded
    with use_context=True, so every downstream evaluation of the context ablation fed it
    a basket-interaction term it had never been trained to use.  Three other ablations
    (--no-state, --no-breadth, --avail-only) crashed on a state_dict key mismatch, which
    is why they have no generator_eval artefact.
    """
    cfg = json.load(open(os.path.join(OUT, f"{label}_nested_history.json")))["config"]
    m = nb.NestedModel(d, K=cfg["K"], Kp=cfg["Kp"], Kt=cfg["Kt"], Ks=cfg["Ks"],
                       seed=cfg["seed"], use_nest=not cfg["no_nest"],
                       use_quantity=not cfg["no_quantity"],
                       use_store=not cfg["no_store"],
                       use_store_price=not (cfg["no_store_price"] or cfg["avail_only"]),
                       avail_only=cfg["avail_only"],
                       use_state=not cfg["no_state"],
                       use_breadth=not cfg["no_breadth"],
                       use_context=not cfg["no_context"],
                       ctx_agg=cfg.get("ctx_agg", "mean"),
                       learn_ctx_scale=cfg.get("learn_ctx_scale", False),
                       use_cat_context=cfg.get("cat_context", False),
                       use_cat_pair=cfg.get("cat_pair", False),
                       untie_rho=cfg.get("untie_rho", False),
                       prefix_context=cfg.get("prefix_context", False),
                       neg_in_cat=cfg.get("neg_in_cat", 0.0),
                       item_loss=cfg.get("item_loss", "softmax"),
                       use_persist=not cfg.get("no_persist", False),
                       use_promo=cfg.get("use_promo", False),
                       use_nb=cfg.get("nb_units", False)).to(dev)
    # strict=False so checkpoints predating a newly added parameter still load.  Any
    # such parameter is initialised at zero, so the loaded model reproduces the older one
    # exactly rather than silently inheriting a random value.
    sd = torch.load(os.path.join(OUT, f"{label}_nested.pt"), map_location=dev)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"{label}: checkpoint has parameters the model lacks: {unexpected}")
    for k in missing:
        p_ = dict(m.named_parameters()).get(k)
        if p_ is not None:
            torch.nn.init.zeros_(p_)
    m.eval()
    # Temperature fitted on validation during training.  Generation samples from
    # softmax(u), which is scale-sensitive: an uncalibrated model at sd(u)=2.15 instead
    # of 1.24 produces a far too peaked choice distribution.
    m.temperature = float(json.load(open(
        os.path.join(OUT, f"{label}_nested_history.json"))).get("temperature", 1.0))
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
            ref = m.iv_ref[ct] if getattr(m, 'iv_ref', None) is not None \
                else torch.zeros_like(iv)
            iv = torch.where(ok, iv - ref, torch.zeros_like(iv))
            # cat_state, not state: 27 trains the incidence head on the category's own
            # recency, so generating with the old first-item proxy is a train/generate
            # mismatch.  It cost 14% of generated basket size when they disagreed.
            cst = d.cat_state(user, cats[:, k], day)
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
            # cloglog, matching Eq. 9 of the specification.  This was sigmoid, left
            # behind when the incidence link changed: at the 3.25% base rate the two
            # agree to 2%, so the NUMBER of categories looked right, but for a category
            # the household uses heavily eta is high and cloglog is up to 1.3x sigmoid --
            # so generation systematically under-picked exactly the familiar categories.
            p_buy = (-torch.expm1(-torch.exp(lin.clamp(-12.0, 4.0)))).clamp(1e-6, 1 - 1e-6)
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


def _cat_cache(m, d, dev, cats, user, day, store):
    """Per-category (stocked items, state, price deviation) for one trip, built once.

    Both the sequential sampler and the Gibbs sweep need utilities for a single
    category at a time; recomputing all C per draw is 188x the work.
    """
    out = {}
    for c in cats:
        jj = d.cat_items[c][d.cat_mask[c] > 0]
        if m.use_store:
            jj = jj[d.carried[jj, store]]
        n = len(jj)
        if n == 0:
            out[c] = (jj, None, None, 0)
            continue
        stc = torch.as_tensor(
            d.state(np.full(n, user), jj.cpu().numpy(), np.full(n, day)), device=dev)
        dpc = d.log_price_dev[jj, torch.full((n,), day, device=dev, dtype=torch.long)]
        out[c] = (jj, stc, dpc, n)
    return out


@torch.no_grad()
def generate_baskets(m, d, dev, n_trips=300, seed=0, sweeps=2, use_ctx=True,
                     with_units=False, trips=None, cat_sweeps=2,
                     item_temp=1.0, require_nonempty=False, max_tries=20):
    """Emit ACTUAL baskets -- item ids, not counts -- and measure co-purchase structure.

    `generate` above accumulates expected counts and never samples an item, which is why
    the question "where does the basket context come from at generation time" never
    arose there: there is no basket.  That is fine for checking marginals and useless
    for anything downstream that needs item ids, an MDP included.

    Here a basket is materialised in two stages.

      pass 1   for every category, draw incidence from the Bernoulli head and breadth
               from the breadth head, then sample that many DISTINCT items from
               softmax(u) over the category's stocked items.  The context is zero,
               because at this point no basket exists.

      pass 2   Gibbs sweeps.  NESTED_MODEL.md 3.3 shows the item head's conditionals are
               those of P(S) prop exp(E(S)) at fixed basket size, so resampling one slot
               at a time from its own category, with the context recomputed from the
               current draft, targets exactly that joint.  Category composition and
               basket size are held fixed, which is the move E(S) is defined over.

    `use_ctx=False` skips pass 2 and reproduces what the model emits today, so the two
    can be compared on the same trips.

    TWO CORRECTIONS, both off by default so every existing artefact reproduces.

    `require_nonempty` implements the n >= 1 conditioning of spec Eq. 8.  Training divides
    by 1 - prod_c P(y_c = 0); generation did not, and emitted an empty basket 4.42% of the
    time against a real 0%.  Redrawing the composition until something is bought is exact
    rejection sampling from the conditioned law, and closes 39% of the basket-size
    shortfall on its own.

    `item_temp` sharpens ONLY the within-category item draw -- pass 1 and the Gibbs sweeps
    -- and deliberately leaves the inclusive value alone.  Scaling `m.item_utility`
    globally, as the temperature sweep in 42_limitations does, also scales IV, which feeds
    incidence, so basket size inflated 7.55 -> 12.39 items and confounded the result.  IV
    is computed from the untempered utilities here, so the two effects separate.
    """
    sp = d.splits["test"]
    rng = np.random.default_rng(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    # NOTE: when `trips` is None this is a RANDOM subset, so the k-th returned basket
    # belongs to trip bsel[k], NOT to trip k.  Attributing them positionally compares a
    # generated basket against a different household's history, which makes an accurate
    # generator look badly wrong -- it cost a long investigation before being caught.
    # `.trips` is attached to the returned list so callers cannot get this wrong.
    bsel = (rng.choice(sp["n_baskets"], size=min(n_trips, sp["n_baskets"]), replace=False)
            if trips is None else np.asarray(trips))
    A = m.alpha.detach()

    def cat_utilities(user, day, week, store, ctx):
        """u and stocked-mask for EVERY category, for one trip.  [C, Mx]."""
        blk, msk = d.cat_items, d.cat_mask.clone()
        Mx = blk.shape[1]
        blk_np = blk.cpu().numpy()
        rep = lambda v: np.full((d.C, Mx), v)
        st = d.state(rep(user).ravel(), blk_np.ravel(), rep(day).ravel()).reshape(
            d.C, Mx, nb.N_STATE_FEATURES)
        dlogp = d.log_price_dev[blk, torch.as_tensor(rep(day), device=dev)]
        if m.use_store:
            msk = msk * d.carried[blk, torch.as_tensor(rep(store), device=dev)].float()
        u = m.item_utility(torch.full((d.C,), user, device=dev, dtype=torch.long), blk,
                           ctx.unsqueeze(0).expand(d.C, -1), dlogp,
                           torch.as_tensor(st, device=dev),
                           torch.full((d.C,), week, device=dev, dtype=torch.long),
                           torch.full((d.C,), store, device=dev, dtype=torch.long))
        return u.masked_fill(msk == 0, -1e9), msk, blk

    out = []
    for i in bsel:
        first = sp["starts"][i]
        user, day = int(sp["user"][first]), int(sp["day"][first])
        week, store = int(sp["week"][first]), int(sp["store"][first])
        zero = torch.zeros(m.K, device=dev)
        u, msk, blk = cat_utilities(user, day, week, store, zero)
        iv = torch.logsumexp(u, dim=1)
        ok = msk.sum(1) > 0
        ref = m.iv_ref if getattr(m, 'iv_ref', None) is not None \
            else torch.zeros_like(iv)
        iv = torch.where(ok, iv - ref, torch.zeros_like(iv))
        cst = d.cat_state(np.full(d.C, user), np.arange(d.C), np.full(d.C, day))
        allc = torch.arange(d.C, device=dev)
        lin = (m.c0 + (m.c_user[user].unsqueeze(0) * m.c_cat).sum(-1)
               + (m.c_state * torch.as_tensor(cst, device=dev)).sum(-1)
               + m.kappa() * iv)
        # the household-category habit term, added to the incidence head in training and
        # previously missing here -- a second train/generate mismatch alongside the link
        if getattr(m, "c_hab", None) is not None:
            lin = lin + m.c_hab * d.hh_cat[user]
        # Pass 0 -- WHICH CATEGORIES.  Independent Bernoulli draws leave every
        # cross-category pair at lift ~1, and 95% of item pairs are cross-category
        # (NESTED_MODEL.md 8.6e).  With a category context the indicator vector is a
        # C-dimensional binary field whose single-site conditionals the incidence head
        # supplies, so Gibbs over it is the same move used for items, one level up.
        # cloglog, matching Eq. 9 of the specification (see the note above).
        # spec Eq. 8 conditions on n >= 1: training divides by 1 - prod_c P(y_c = 0),
        # generation did not.  Redrawing until something is bought is exact rejection
        # sampling from the conditioned law.  Only the draw is retried; `lin` above is
        # deterministic given the trip.
        p_ent = (-torch.expm1(-torch.exp(lin.clamp(-12.0, 4.0)))).clamp(1e-6, 1 - 1e-6)
        for _try in range(max_tries if require_nonempty else 1):
            bought = (torch.rand(d.C, generator=g) < p_ent) & ok
            if not require_nonempty or bool(bought.any()):
                break
        has_pair = getattr(m, "cat_pair", None) is not None
        if (has_pair or getattr(m, "cat_ctx_scale", None) is not None) and cat_sweeps:
            W = m.cat_pair_sym() if has_pair else None
            cc = m.c_cat.detach() if m.cat_ctx_scale is not None else m.c_cat.detach()
            S = cc[bought].sum(0) if int(bought.sum()) else torch.zeros(m.K, device=dev)
            nB = float(bought.sum())
            order = torch.arange(d.C, device=dev)
            for _ in range(cat_sweeps):
                for c in order.tolist():
                    if not bool(ok[c]):
                        continue
                    inb = float(bought[c])
                    den = max(nB - inb, 1.0)
                    ctx = (S - cc[c] * inb) / den
                    lg = lin[c]
                    if m.cat_ctx_scale is not None:
                        lg = lg + m.cat_ctx_scale * (cc[c] * ctx).sum()
                    if W is not None:
                        lg = lg + (W[c] * bought.float()).sum() - W[c, c] * inb
                    pb = (-torch.expm1(-torch.exp(lg.clamp(-12.0, 4.0)))
                          ).clamp(1e-6, 1 - 1e-6)
                    new_b = bool(torch.rand(1, generator=g) < pb)
                    if new_b != bool(bought[c]):
                        S = S + cc[c] if new_b else S - cc[c]
                        nB = nB + 1 if new_b else nB - 1
                        bought[c] = new_b
        pdev = torch.zeros(d.C, device=dev)
        if getattr(m, "use_breadth", False):
            zb = (m.b0 + m.b_user[user] - m.b_price * pdev).clamp(-6, 3)
            k = (1.0 + torch.poisson(torch.exp(zb), generator=g)) * bought.float()
        else:
            k = bought.float()
        k = torch.minimum(k, (msk > 0).float().sum(1)).long()

        slots = []                                   # (category, item_id)
        if getattr(m, "prefix_context", False):
            # SHOPPER's generative process: items are placed one at a time and each
            # sees the mean of alpha over the items already placed.  This is exactly
            # the context the model was TRAINED on under --prefix-context, so
            # training and generation agree with no Gibbs and no compatibility
            # assumption.  Slot order is randomised because the data has none.
            todo = [c for c in torch.nonzero(k > 0).flatten().tolist()
                    for _ in range(int(k[c]))]
            perm = torch.randperm(len(todo), generator=g).tolist()
            # Cache each needed category's items, state and price ONCE.  Calling
            # cat_utilities per slot recomputes all C categories for one draw, which
            # is 188x the work and does not finish on the full test set.
            cache = _cat_cache(m, d, dev, sorted(set(todo)), user, day, store)
            run = torch.zeros(m.K, device=dev)
            for si, ti in enumerate(perm):
                c = todo[ti]
                jj, stc, dpc, nnj = cache[c]
                if not nnj:
                    continue
                ctx = run / si if si else torch.zeros(m.K, device=dev)
                uc = m.item_utility(
                    torch.full((1,), user, device=dev, dtype=torch.long),
                    jj.unsqueeze(0), ctx.unsqueeze(0), dpc.unsqueeze(0),
                    stc.unsqueeze(0),
                    torch.full((1,), week, device=dev, dtype=torch.long),
                    torch.full((1,), store, device=dev, dtype=torch.long))[0]
                q = int(torch.multinomial(torch.softmax(uc / item_temp, 0), 1,
                                          generator=g))
                j = int(jj[q])
                slots.append((c, j))
                run = run + A[j]
        else:
            for c in torch.nonzero(k > 0).flatten().tolist():
                # item_temp sharpens the item draw only.  `iv` above was taken from the
                # untempered `u`, so incidence is unaffected and the two channels separate.
                pi = torch.softmax(u[c] / item_temp, 0) * (msk[c] > 0).float()
                if float(pi.sum()) <= 0:
                    continue
                pick = torch.multinomial(pi / pi.sum(), int(k[c]), replacement=False,
                                         generator=g)
                if DEBUG_PASS1 is not None:
                    DEBUG_PASS1.append((user, c, (pi / pi.sum()).detach().cpu().numpy(),
                                        [int(blk[c, q]) for q in pick.tolist()],
                                        int(k[c])))
                slots += [(c, int(blk[c, q])) for q in pick.tolist()]

        # ---- pass 2: Gibbs sweeps with the context recomputed from the draft
        #
        # Only the category being resampled is recomputed.  Recomputing all C per slot
        # is ~42k utility evaluations and as many state lookups for one draw, which is
        # 188x the work for no extra information: every other category is frozen.
        if use_ctx and len(slots) > 1 and not getattr(m, "prefix_context", False):
            cache = _cat_cache(m, d, dev, sorted({c for c, _ in slots}), user, day, store)
            tot = A[[j for _, j in slots]].sum(0)
            n_slots = len(slots)
            for _ in range(sweeps):
                for si in range(n_slots):
                    c, cur = slots[si]
                    jj, stc, dpc, nnj = cache[c]
                    if not nnj:
                        continue
                    ctx = tot - A[cur]
                    if getattr(m, "ctx_agg", "mean") == "mean":
                        ctx = ctx / (n_slots - 1)
                    uc = m.item_utility(
                        torch.full((1,), user, device=dev, dtype=torch.long),
                        jj.unsqueeze(0), ctx.unsqueeze(0), dpc.unsqueeze(0),
                        stc.unsqueeze(0),
                        torch.full((1,), week, device=dev, dtype=torch.long),
                        torch.full((1,), store, device=dev, dtype=torch.long))[0]
                    # A_-j excludes whatever occupies the OTHER slots: the basket is a
                    # set (spec Eq. 3), so without this a category can be given the same
                    # product twice and the chain leaves S(K).
                    others = {jj_ for si2, (c2, jj_) in enumerate(slots)
                              if si2 != si and c2 == c}
                    if others:
                        blocked = torch.as_tensor(
                            np.isin(jj.cpu().numpy(), np.fromiter(others, dtype=np.int64)),
                            device=dev)
                        uc = uc.masked_fill(blocked, -1e9)
                    q = int(torch.multinomial(torch.softmax(uc / item_temp, 0), 1,
                                              generator=g))
                    new_j = int(jj[q])
                    tot = tot - A[cur] + A[new_j]      # keep the running sum exact
                    slots[si] = (c, new_j)
        ids = [j for _, j in slots]
        if not with_units:
            out.append(ids)
            continue
        # units for the chosen items, from the quantity head -- the same draw the
        # likelihood was fitted on, units - 1 ~ Poisson(exp(z))
        if not ids or not m.use_quantity:
            out.append((ids, [1] * len(ids)))
            continue
        jt = torch.as_tensor(ids, device=dev)
        stq = torch.as_tensor(d.state(np.full(len(ids), user), np.asarray(ids),
                                      np.full(len(ids), day)), device=dev)
        dpq = d.log_price_dev[jt, torch.full((len(ids),), day, device=dev,
                                             dtype=torch.long)]
        if m.use_store and m.use_store_price:
            dpq = dpq + d.store_dev(np.asarray(ids), np.full(len(ids), store),
                                    np.full(len(ids), sp["raw_week"][first]))
        zq = (m.q0[jt]
              - (m.q_gamma[user].unsqueeze(0) * m.q_beta[jt]).sum(-1) * dpq
              + (m.q_state[jt] * stq).sum(-1)).clamp(-6, 4)
        lam_q = torch.exp(zq)
        if getattr(m, "q_disp", None) is not None:
            # NB2 as a gamma-Poisson mixture: draw the rate from Gamma(r, r/L), then a
            # Poisson at that rate.  Same mean L, variance L + L^2/r, and it reduces to
            # the Poisson branch as r grows.
            r_ = torch.nn.functional.softplus(m.q_disp)
            gam = torch.distributions.Gamma(r_.expand_as(lam_q),
                                            (r_ / lam_q.clamp_min(1e-9)))
            lam_q = gam.sample()
        u_units = (1 + torch.poisson(lam_q, generator=g)).long().tolist()
        out.append((ids, u_units))
    return _Baskets(out, bsel)


@torch.no_grad()
def co_purchase_check(m, d, dev, n_trips=300, seeds=5, sweeps=4):
    """Do generated baskets carry co-purchase structure, and how much of the real kind?

    Two measures, both computed inside a basket and averaged over its item pairs:

      alpha.alpha    mean dot product between the embeddings of two items in the same
                     basket.  This is the quantity the interaction term acts on, so it
                     is the direct read on whether the term reached the output.
      same-sub       share of within-basket item pairs sharing a SUB_COMMODITY, a label
                     the model never sees.  A held-out check on the same question.

    Compared across real held-out baskets, generation with the context zeroed (what this
    script emitted before), and generation with Gibbs sweeps (NESTED_MODEL.md 3.3).
    """
    items = pd.read_parquet(os.path.join(IN, "items.parquet")).sort_values("item_id")
    sub = torch.as_tensor(items.sub_id.to_numpy(), device=dev)
    A = m.alpha.detach()
    sp = d.splits["test"]

    def stat(baskets):
        pr, ss, sz = [], [], []
        for b in baskets:
            sz.append(len(b))
            if len(b) < 2:
                continue
            v = A[b]
            G = v @ v.T
            n = len(b)
            pr.append(float((G.sum() - G.diag().sum()) / (n * (n - 1))))
            s = sub[b]
            ss.append(float((s.unsqueeze(0) == s.unsqueeze(1)).float().sum() - n)
                      / (n * (n - 1)))
        return float(np.mean(sz)), float(np.mean(pr)), float(np.mean(ss))

    def agg(fn):
        R = np.array([fn(100 + s) for s in range(seeds)])
        return {"items": R[:, 0].mean(), "items_sd": R[:, 0].std(),
                "alpha_dot": R[:, 1].mean(), "alpha_dot_sd": R[:, 1].std(),
                "same_sub_pair_share": R[:, 2].mean(),
                "same_sub_pair_share_sd": R[:, 2].std()}

    def real_fn(s):
        bs = np.random.default_rng(s).choice(sp["n_baskets"], size=n_trips, replace=False)
        return stat([sp["item"][sp["starts"][i]:sp["ends"][i]].tolist() for i in bs])

    out = {"n_trips_per_seed": n_trips, "seeds": seeds, "sweeps": sweeps,
           "real": agg(real_fn),
           "generated_context_zeroed": agg(
               lambda s: stat(generate_baskets(m, d, dev, n_trips, s, use_ctx=False))),
           "generated_gibbs": agg(
               lambda s: stat(generate_baskets(m, d, dev, n_trips, s, sweeps=sweeps,
                                               use_ctx=True)))}
    for k in ("alpha_dot", "same_sub_pair_share"):
        for g in ("generated_gibbs", "generated_context_zeroed"):
            out[g][f"share_of_real_{k}"] = out[g][k] / out["real"][k]
    return out


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

        log("")
        log("4. do generated baskets carry co-purchase structure?")
        cp = co_purchase_check(m, d, dev, sweeps=a.gibbs_sweeps)
        res["co_purchase"] = cp
        for k, lab in [("real", "real held-out"),
                       ("generated_context_zeroed", "generated, context zeroed"),
                       ("generated_gibbs", f"generated, {a.gibbs_sweeps} Gibbs sweeps")]:
            v = cp[k]
            log(f"   {lab:32s} items {v['items']:5.2f}  "
                f"alpha.alpha {v['alpha_dot']:+.4f}+-{v['alpha_dot_sd']:.4f}  "
                f"same-sub pairs {v['same_sub_pair_share']:.4f}"
                f"+-{v['same_sub_pair_share_sd']:.4f}")
        gz, gg = cp["generated_context_zeroed"], cp["generated_gibbs"]
        log(f"   -> Gibbs lifts alpha.alpha {gz['alpha_dot']:.4f} -> "
            f"{gg['alpha_dot']:.4f} ({gg['alpha_dot'] / gz['alpha_dot']:.2f}x), "
            f"reaching {gg['share_of_real_alpha_dot']:.0%} of the real level")

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
    p.add_argument("--gibbs-sweeps", type=int, default=4,
                   help="Gibbs sweeps over a draft basket when generating actual "
                        "item ids; 0 reproduces the zeroed context")
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
