"""
Fit version 3 to dunnhumby.

Starts with a TIMING PROBE, not a run.  The specification projected 2-8 hours per fit and
flagged that the previous projection of this kind was out by 18x; the corrected assortment
then grew the per-trip cost by about 1.6x on top.  So the first thing this does is measure
the cost of a hundred iterations and print the implied wall clock, before committing to
anything.  Pass --probe-only to stop there.

The objective is the likelihood of section 16: for each trip, the energy of the observed
basket minus log(Z - 1), where Z is the normaliser over every subset of the store's
assortment and the -1 conditions on the basket being non-empty.  There are no component
weights to choose.

Held-out likelihood is reported per basket and per line.  Per basket is the quantity the
model defines; per line is the one that can be compared against a model with a different
notion of a trip, and both are printed so neither can be quoted selectively.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.nn.functional import softplus

from data import build
from features import Features
from ragged import (RaggedIndex, RaggedModel, set_quad, sobol_grid, size_band_scales,
                    sparse_prepare, log_f_sparse,
                    sobol_mixture_grid)

HERE = os.path.dirname(os.path.abspath(__file__))
from paths import OUT


def log(m):
    print(f"[fit] {m}", flush=True)


class Batcher:
    """Builds the ragged index and the per-slot features for a set of trips."""

    def __init__(self, D, F, nmax, price_ref="trip"):
        self.D, self.F, self.nmax = D, F, nmax
        self.price_ref = price_ref
        self.sub_of = None
        if price_ref == "subcommodity":
            # The reference group's WIDTH sets the cross-price magnitude, because
            # cross-elast ~ gb (kappa - 1) (n_riv / n_ref).  As purchases actually
            # experience them the affinity rows have median 128 products (mean 388) while
            # sub-commodities have median 16 -- 8x narrower, and exactly the grouping the
            # data's rival price is measured over.  Referencing the affinity row got the
            # SIGN right (-0.1621 -> +0.0440) but only a third of the target +0.1351.
            import pandas as pd
            from features import BI as _BI          # resolved from the module dir, not CWD
            _it = pd.read_parquet(os.path.join(_BI, "items.parquet"))
            self.sub_of = torch.zeros(int(D["n_item"]), dtype=torch.long)
            _ii = _it["item_id"].astype(int).values
            _ss = _it["sub_id"].astype(int).values
            _ok = _ii < int(D["n_item"])
            self.sub_of[torch.as_tensor(_ii[_ok])] = torch.as_tensor(_ss[_ok])
            self.n_sub = int(self.sub_of.max()) + 1
        self.C = int(D["n_cat"])
        self.ptr = D["store_cat_ptr"]
        self.items = D["store_items"]
        self.lptr = D["line_ptr"]

    def make(self, trips):
        D, C = self.D, self.C
        it_l, row_of, row_trip, row_cat = [], [], [], []
        nrow = 0
        for bi, t in enumerate(trips):
            s = int(D["trip_store"][t]) * C
            for c in range(C):
                lo, hi = int(self.ptr[s + c]), int(self.ptr[s + c + 1])
                if hi <= lo:
                    continue
                it_l.append(self.items[lo:hi])
                row_of.append(np.full(hi - lo, nrow, np.int64))
                row_trip.append(bi)
                row_cat.append(c)
                nrow += 1
        item = np.concatenate(it_l)
        ix = RaggedIndex(item, np.concatenate(row_of),
                         np.array(row_trip, np.int64), np.array(row_cat, np.int64),
                         len(trips))
        store = torch.as_tensor(D["trip_store"][trips], dtype=torch.long)
        day = torch.as_tensor(D["trip_day"][trips], dtype=torch.long)
        week = torch.as_tensor(D["trip_week"][trips], dtype=torch.long)
        st_i, dy_i, wk_i = store[ix.item_trip], day[ix.item_trip], week[ix.item_trip]
        dlp, disp, mail = self.F.gather(ix.item, st_i, dy_i, wk_i)
        # week-of-year, per the spec: (WEEK_NO - 1) mod 52.  Clamping instead, as an earlier
        # version did, collapsed 54.6% of trips onto one seasonal parameter.
        user = torch.as_tensor(D["trip_user"][trips], dtype=torch.long)
        # Per-trip mean price deviation over the ASSORTMENT.  It is the reference for
        # splitting dlp into a common level and an idiosyncratic deviation, and it must be
        # the same number whether b_at is called on assortment slots or on purchased lines
        # -- so it is computed once here, from the assortment, and carried in both views.
        #
        # WHICH reference decides whether the model can express substitution at all.
        # b_j = ... - gb_j [dbar + kappa (dlp_j - dbar)], so a rival's price rise moves b_j
        # by gb_j (kappa - 1) * d_dbar -- strongly POSITIVE at kappa = 35.6 (gb(kappa-1) =
        # 0.535).  Averaged over the whole assortment, though, one rival moves dbar by
        # d / 5,292, so the channel is diluted ~5,000x and the size effect swamps it: the
        # fitted cross-price elasticity is -0.116 where the data (item-week panel, rival =
        # mean log price within the sub-commodity) shows +0.1351.
        #
        # Referencing the RAGGED ROW instead -- the store's own category, median 9 products
        # -- moves the reference by d/n_c, giving 0.535 * (n_riv/n_c) ~ +0.18 for three
        # rivals in a nine-product category.  That is the right order, and it is the right
        # economics: a shopper judges a price against close alternatives, not against the
        # whole store.  Nothing in the normaliser changes; dbar is a feature.
        #
        # Pairwise phi cannot do this job.  d pi_k / d b_j = Cov(1_j, 1_k), so an
        # interaction's leverage on a marginal is second order in pi_j pi_k ~ 1e-3:
        # measured, driving phi_j.phi_k to -0.64 moved the cross-price elasticity by 0.014
        # against the 0.25 required, with lambda_max never exceeding 0.216.
        if self.price_ref == "subcommodity":
            _key = ix.item_trip * self.n_sub + self.sub_of[ix.item]
            _ng = ix.B * self.n_sub
            _sb = torch.zeros(_ng, dtype=torch.float64).index_add_(0, _key, dlp.double())
            _sc = torch.zeros(_ng, dtype=torch.float64).index_add_(
                0, _key, torch.ones_like(dlp, dtype=torch.float64))
            _subbar = _sb / _sc.clamp_min(1.0)
            _dbar = _subbar[_key]                 # per SLOT
        elif self.price_ref == "category":
            _rb = torch.zeros(ix.n_rows, dtype=torch.float64).index_add_(
                0, ix.row_of, dlp.double())
            _rc = torch.zeros(ix.n_rows, dtype=torch.float64).index_add_(
                0, ix.row_of, torch.ones_like(dlp, dtype=torch.float64))
            _rowbar = _rb / _rc.clamp_min(1.0)
            _dbar = _rowbar[ix.row_of]            # per SLOT, not per trip
        else:
            _dbar = torch.zeros(ix.B, dtype=torch.float64).index_add_(
                0, ix.item_trip, dlp.double())
            _dcnt = torch.zeros(ix.B, dtype=torch.float64).index_add_(
                0, ix.item_trip, torch.ones_like(dlp, dtype=torch.float64))
            _dbar = _dbar / _dcnt.clamp_min(1.0)
        ctx = dict(dlp_bar=_dbar, dlp=dlp.double(), disp=disp.double(), mail=mail.double(),
                   week=(wk_i - 1) % 52, store=st_i,
                   rec=self.F.recency(ix.item, user[ix.item_trip], dy_i))
        li, lt, lc, lu = [], [], [], []
        for bi, t in enumerate(trips):
            a, b = int(self.lptr[t]), int(self.lptr[t + 1])
            li.append(D["line_item"][a:b])
            lc.append(D["line_cat"][a:b])
            lu.append(D["line_units"][a:b])
            lt.append(np.full(b - a, bi, np.int64))
        LI = torch.as_tensor(np.concatenate(li), dtype=torch.long)
        LT = torch.as_tensor(np.concatenate(lt), dtype=torch.long)
        # the SAME features, gathered at the purchased lines, so energy() and log_Z score
        # each product identically
        dlp_l, disp_l, mail_l = self.F.gather(LI, store[LT], day[LT], week[LT])
        if self.price_ref == "subcommodity":
            _dbar_l = _subbar[LT * self.n_sub + self.sub_of[LI]]
        elif self.price_ref == "category":
            # each purchased line sits in the row (trip, category); look up that row's
            # reference so energy() and log_Z score the product identically.
            _slot_row = torch.full((ix.B, int(ix.row_cat.max()) + 1), -1, dtype=torch.long)
            _slot_row[ix.row_trip, ix.row_cat] = torch.arange(ix.n_rows)
            _lrow = _slot_row[LT, torch.as_tensor(np.concatenate(lc), dtype=torch.long)]
            _dbar_l = _rowbar[_lrow.clamp_min(0)]
            _dbar_l = torch.where(_lrow >= 0, _dbar_l, torch.zeros_like(_dbar_l))
        else:
            _dbar_l = _dbar
        lctx = dict(dlp_bar=_dbar_l, dlp=dlp_l.double(), disp=disp_l.double(), mail=mail_l.double(),
                    week=(week[LT] - 1) % 52, store=store[LT],
                    rec=self.F.recency(LI, user[LT], day[LT]))
        house = torch.as_tensor(D["trip_user"][trips], dtype=torch.long)
        return (ix, ctx, lctx, house,
                LI, LT, torch.as_tensor(np.concatenate(lc), dtype=torch.long),
                torch.as_tensor(np.concatenate(lu), dtype=torch.long))


def popularity_logits(D, trips):
    """Exposure-corrected product incidence initializer on the training window.

    A zero lam makes the first model nearly uniform over about 5,000 available products,
    even though one pass over the training data already identifies their marginal rates.
    This is initialization, not target leakage: only training trips and the assortments in
    which each item was available enter the estimate.
    """
    J, C, S = (int(D[k]) for k in ("n_item", "n_cat", "n_store"))
    count = np.zeros(J, dtype=np.float64)
    for t in trips:
        lo, hi = int(D["line_ptr"][t]), int(D["line_ptr"][t + 1])
        count[D["line_item"][lo:hi]] += 1.0
    exposure = np.zeros(J, dtype=np.float64)
    store_n = np.bincount(D["trip_store"][trips], minlength=S)
    ptr, items = D["store_cat_ptr"], D["store_items"]
    for store in range(S):
        lo, hi = int(ptr[store * C]), int(ptr[(store + 1) * C])
        exposure[items[lo:hi]] += store_n[store]
    seen = exposure > 0
    value = np.empty(J, dtype=np.float64)
    value[seen] = np.log((count[seen] + 0.5) / (exposure[seen] + 1.0))
    value[~seen] = np.median(value[seen])
    value -= value.mean()       # common shifts are exactly a linear rho_0 gauge direction
    return torch.as_tensor(value, dtype=torch.float64)


def _size_coeffs(m, z, ix):
    """mean_b log A_n(z) at the given z -- the combinatorial part of the size law, before
    rho_0 tilts it."""
    from ragged import esp_bucketed, poly_mul_trunc, seg_max
    phi_i = m.phi[ix.item]
    bt = m.b_flat(ix) - 0.5 * (phi_i ** 2).sum(-1)
    proj = (z[ix.item_trip] * phi_i.unsqueeze(1)).sum(-1)
    logw = (bt.unsqueeze(1) + proj).transpose(0, 1)
    M = seg_max(logw, ix.item_trip, ix.B)
    w = torch.exp(logw - M.index_select(-1, ix.item_trip))
    e = esp_bucketed(w, ix.row_of, ix.n_rows, m.R, ix.row_size, ix.item_pos)
    r = torch.arange(m.R + 1, dtype=w.dtype)
    G = torch.exp(-m.rho_c[ix.row_cat].unsqueeze(-1) * m.pair_feature(r)).unsqueeze(0) * e
    Gp = torch.zeros(1, ix.B * ix.Cpad, m.R + 1, dtype=w.dtype)
    Gp[:, :, 0] = 1.0
    Gp = Gp.index_copy(1, ix.flat_slot, G).view(1, ix.B, ix.Cpad, m.R + 1)
    A = Gp[:, :, 0, :]
    for c in range(1, ix.Cpad):
        A = poly_mul_trunc(A, Gp[:, :, c, :], m.nmax)
    n_ax = torch.arange(A.shape[-1], dtype=w.dtype)
    return (torch.log(A.clamp_min(1e-300)) + n_ax * M.unsqueeze(-1))[0].mean(0)


def initialize_size_potential(model, data, training, batcher, nmax):
    """Initialize rho_0 against the empirical training size law at reference contexts."""
    n_train = data["trip_nlines"][training]
    count = np.bincount(np.clip(n_train, 0, nmax), minlength=nmax + 1) + 0.5
    target = torch.log(torch.as_tensor(count / count.sum(), dtype=model.lam.dtype))
    sub = training[np.random.default_rng(0).choice(len(training), size=64, replace=False)]
    ix, ctx, _, house, *_ = batcher.make(sub)
    model.house, model.ctx = house, ctx
    z = torch.zeros(ix.B, 1, model.Kz, dtype=model.lam.dtype)
    with torch.no_grad():
        coeff = _size_coeffs(model, z, ix)
        rho = coeff[:nmax + 1] - target
        model.rho_0_free.copy_((rho - rho[0])[1:])
    return float((count / count.sum() * np.arange(nmax + 1)).sum())


def transfer_multinomial_nonprice(model, checkpoint):
    """Gauge-map shared fitted additive blocks from a Multinomial checkpoint."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = blob.get("model", blob) if isinstance(blob, dict) else blob
    required = ("idx.lam", "idx.alpha", "idx.theta", "idx.w_dsp", "idx.w_mlr",
                "idx.mu", "idx.delta", "idx.zeta", "idx.xi")
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"multinomial utility checkpoint is missing {missing}")
    with torch.no_grad():
        model.alpha.copy_(state["idx.alpha"])
        model.theta.copy_(state["idx.theta"])
        model.mu.copy_(state["idx.mu"])
        week_mean = state["idx.delta"].mean(0)
        nw = state["idx.delta"].shape[0]
        model.delta[:nw].copy_(state["idx.delta"])
        if model.delta.shape[0] > nw:
            model.delta[nw:].copy_(week_mean)
        model.zeta.copy_(state["idx.zeta"])
        model.xi.copy_(state["idx.xi"])
        model.w_dsp.copy_(state["idx.w_dsp"])
        model.w_mlr.copy_(state["idx.w_mlr"])
        model.lam.copy_(state["idx.lam"]
                        + state["idx.alpha"] @ state["idx.theta"].mean(0)
                        + state["idx.mu"] @ week_mean
                        + state["idx.zeta"] @ state["idx.xi"].mean(0))


def calibrate_size_ipf(model, data, training, batcher, nmax, steps=6,
                       n_trips=256, chunk=24, damp=0.5):
    """Fit the aggregate size law by deterministic iterative proportional fitting.

    For a common size tilt, ``P(n|x)`` is multiplied by ``exp(-Delta rho(n))``.  Updating
    ``rho(n) += log[p_model(n)/p_target(n)]`` is therefore the exact one-table IPF update;
    averaging context-specific normalized laws makes repeated damped updates necessary.
    Composition is invariant to rho_0, so this calibration cannot undo a utility transfer.
    """
    if steps <= 0 or n_trips <= 0 or not 0 < damp <= 1:
        raise ValueError("size IPF requires positive steps/trips and damp in (0,1]")
    count = np.bincount(np.clip(data["trip_nlines"][training], 1, nmax),
                        minlength=nmax + 1)[1:].astype(np.float64) + 0.5
    target = torch.as_tensor(count / count.sum(), dtype=model.lam.dtype)
    sample = training[np.random.default_rng(1729).choice(
        len(training), size=min(n_trips, len(training)), replace=False)]
    history = []
    for iteration in range(steps):
        total = torch.zeros(nmax, dtype=model.lam.dtype)
        seen = 0
        for start in range(0, len(sample), chunk):
            sub = sample[start:start + chunk]
            ix, ctx, _, house, *_ = batcher.make(sub)
            model.house, model.ctx = house, ctx
            with torch.no_grad():
                _, pn = model.log_Z(ix, drop_empty=True, return_size=True)
            total += pn.sum(0)
            seen += ix.B
        pbar = (total / seen).clamp_min(1e-300)
        pbar = pbar / pbar.sum()
        grid = torch.arange(1, nmax + 1, dtype=pbar.dtype)
        history.append(dict(step=iteration,
                            mean=float((pbar * grid).sum()),
                            kl=float((target * (target.log() - pbar.log())).sum()),
                            max_log_ratio=float((pbar.log() - target.log()).abs().max())))
        delta = pbar.log() - target.log()
        delta = delta - delta[0]                 # fix size-one as the non-empty gauge
        with torch.no_grad():
            model.rho_0_free.add_(damp * delta)
    return history


GOALS = [
    # name              how it is measured                     pass band
    ("logZ-converged",  "log Z at train draws vs 16x",          "|gap| < 1.0 nat"),
    ("E[n]-converged",  "E[n] at train draws vs 8x",            "within 10%"),
    ("E[n]-calibrated", "held-out E[n] vs observed",            "within 25%"),
    ("var-calibrated",  "held-out Var(n) vs observed",          "within 40%"),
    ("sampler-agrees",  "sampled basket size vs analytic E[n]", "within 25%"),
    ("elasticity",      "proxy vs the data's -0.121",           "within 30%"),
    ("data-kept",       "trips dropped this window",            "< 2%"),
]


def check_goals(vals):
    """Pre-declared acceptance bands, evaluated every checkpoint.

    Set BEFORE the run and printed as PASS/FAIL, so a good loss cannot stand in for a usable
    model.  The one that matters most is sampler-agrees: the size law and the sampler are the
    same distribution reached two ways, so they must return the same mean.  On the commodity
    partition they did (11.69 vs 11.41).  Under affinity rho_c they diverge -- 8.65 analytic
    against 25.51 sampled after a full epoch -- and nothing else in the log could see it,
    because every other number is derived from the size law and agrees with itself.
    """
    out = []
    for name, _, _ in GOALS:
        v = vals.get(name)
        out.append(f"{name}={'PASS' if v else 'FAIL' if v is not None else '--'}")
    return "  ".join(out)


def rec_eval(m, B, trips, seed=0, chunk=24, return_ranks=False):
    """Complete-the-basket MRR and median rank, at every checkpoint.

    This is the task metric, and it costs almost nothing: ranking candidates needs NO
    normaliser, because log Z is identical for every candidate and cancels.  One b_flat pass
    plus one matrix-vector product per trip -- no sampling, unlike every other number here.

    It earns its place because nothing else in the log can see a ranking failure.  Measured on
    run68, the model scored MRR 0.0036 against a popularity baseline's 0.0467 -- WORSE than
    ranking by raw frequency -- while every pre-declared goal, the normaliser check and the
    distributional KL all looked unremarkable.  They score the joint distribution or its
    moments; none of them scores the ordering.

    The holdout is drawn with a fixed seed so the same items are hidden at every checkpoint
    and the series is comparable across a run.
    """
    rng = np.random.default_rng(seed)
    # save and restore the model's batch context: this runs INSIDE the checkpoint block,
    # before the normaliser check, and leaving m.ctx pointing at the recommendation batch
    # makes the next b_flat mismatch its index (127,999 slots against 128,546).
    _sh, _sc = m.house, m.ctx
    ranks = []
    for k in range(0, len(trips), chunk):
        ix, ctx, lctx, hh, LI, LT, LC, LU = B.make(trips[k:k + chunk])
        m.house, m.ctx = hh, ctx
        with torch.no_grad():
            bf = m.b_flat(ix)
        # The task reveals all but one item and asks which candidate completes the basket.
        # The final size is therefore fixed.  Candidate j has exact conditional score
        # E(rest union {j})-E(rest): b_j + phi_j' sum_rest phi - rho_c Delta g(n_c).
        # No normalizer appears because it is common to every candidate.  The former +6
        # "pinning" approximation differentiated the unconditional log Z, still mixed over
        # final sizes, and made MRR depend on the size model as well as the ranking model.
        for b in range(ix.B):
            sel = ix.item_trip == b
            items, bv = ix.item[sel], bf[sel]
            basket = LI[LT == b]
            if len(basket) < 2:
                continue
            hid = int(basket[rng.integers(len(basket))])
            rest = torch.as_tensor([int(x) for x in basket if int(x) != hid],
                                   dtype=torch.long)
            if len(rest) == 0:
                continue
            pos = (items == hid).nonzero().flatten()
            if len(pos) == 0:
                continue
            with torch.no_grad():
                rest_count = torch.bincount(m.cat_of[rest], minlength=m.C)
                count_before = rest_count[m.cat_of[items]]
                # Under a banded pair scale the interaction enters as (1/s(n)) * pair, and
                # the completion task fixes the final size n = |rest| + 1, so s is a
                # constant here -- but it must be the SAME constant the energy uses or the
                # ranking is scoring a different law than the one being fitted.
                _s = 1.0
                if getattr(m, "size_bands", None):
                    _n = min(len(rest) + 1, m.nmax)
                    _s = float(m.pair_scale_of_n()[_n])
                sc = (bv + (m.phi[items] @ m.phi[rest].sum(0)) / _s
                      - m.rho_c[m.cat_of[items]] * m.pair_increment(count_before))
            inb = torch.zeros(len(items), dtype=torch.bool)
            inb[torch.isin(items, rest)] = True
            sc = sc.clone()
            sc[inb] = -float("inf")
            ranks.append(int((sc > sc[int(pos[0])]).sum()) + 1)
    m.house, m.ctx = _sh, _sc
    if return_ranks:
        return np.asarray(ranks, dtype=float)
    if not ranks:
        return float("nan"), float("nan")
    r = np.asarray(ranks, dtype=float)
    return float((1.0 / r).mean()), float(np.median(r))


def save_ckpt(path, m, opt, sched, it, rng, gen, best_vb, best_it, lz_strikes,
              cum_iter=None):
    """Everything needed to continue training, not just the weights.

    Saving only m.state_dict() makes --resume a warm INITIALISATION rather than a
    continuation: Adam's first and second moment estimates restart at zero, so the first
    steps after a resume are effectively unscaled, and CosineAnnealingLR restarts at the
    full learning rate however far through the schedule the run had got.  Both produce a
    transient that is easy to mistake for the model doing something.

    The two RNG streams are carried as well, so a resumed run draws the batches and the
    proposal noise it would have drawn had it never stopped -- otherwise "resume" silently
    replays the same trips from the start of the stream.

    Written under a temporary name and renamed, because torch.save is not atomic and an
    eval that is interrupted mid-write leaves a truncated file where the best checkpoint
    used to be.
    """
    blob = dict(
        format=2,
        # WHICH PARAMETERISATION the stored gamma/beta are in.  Under --price-soft they
        # are the price coefficients themselves; otherwise they are softplus pre-images.
        # The tensors look identical, so a loader that guesses wrong reads softplus(0.0207)
        # = 0.7036 as the coefficient -- 34x too large.  That is not a small error: it
        # dropped run409's MRR from 0.0705 to 0.0044 in eval_mrr_cutoffs.py while the
        # training log, which had the flag, reported the model working normally.  Recorded
        # for the same reason the data partition is.
        model_flags=dict(price_soft=int(bool(getattr(m, "price_soft", False))),
                         poly_degree=int(getattr(m, "poly_degree", 0) or 0),
                         # What dlp is measured against.  Same trap as price_soft: the
                         # weights look identical, but scoring a category-referenced model
                         # against a trip mean silently deletes its substitution channel.
                         price_ref=str(getattr(m, "price_ref", "trip"))),
        # How log Z was integrated.  Carried in the checkpoint so an eval cannot score
        # this model with a different normaliser than the one it was trained against --
        # recommend_pi.py hardcoded smolyak_grid(4, 8) regardless of the checkpoint.
        # The data partition is chosen by the V3_PARTITION / V3_AFFINITY environment
        # variables and was recorded NOWHERE, so run97 (trained under V3_AFFINITY=1,
        # 280 categories) could not be re-evaluated at all against the default build's
        # 188 -- it failed on a rho_c shape mismatch with no hint as to why.  Record it.
        data=dict(partition=os.environ.get("V3_PARTITION", ""),
                  affinity=os.environ.get("V3_AFFINITY", "0"),
                  n_cat=int(m.rho_c.shape[0]), n_item=int(m.lam.shape[0]),
                  nmax=int(m.nmax), R=int(m.R),
                  rho_pair_cap=int(m.rho_pair_cap)),
        quad=dict(Kz=m.Kz, quad_q=getattr(m, "_quad_q", 0),
                  qmc_n=getattr(m, "_qmc_n", 0), qmc_seed=getattr(m, "_qmc_seed", 0),
                  reps=getattr(m, "_qmc_reps", 1),
                  probe=getattr(m, "_quad_probe", 0),
                  steps=getattr(m, "_quad_steps", 2), chunk=getattr(m, "_quad_chunk", 8),
                  size_bands=getattr(m, "_qmc_size_bands", 0),
                  size_steps=getattr(m, "_qmc_size_steps", 2),
                  mode_logtol=getattr(m, "_qmc_mode_logtol", 8.0),
                  mode_sep=getattr(m, "_qmc_mode_sep", 1.0),
                  mix_n=getattr(m, "_qmc_mix_n", 0)),
        model=m.state_dict(),
        opt=opt.state_dict(),
        sched=sched.state_dict() if sched is not None else None,
        iter=it,
        # Iterations of training this MODEL has had, across every lineage that
        # produced it -- not this process's counter.  They diverge whenever a
        # resume cannot restore `it`: run61 loaded run60's weights from a
        # pre-format-2 file, restarted at 1, and silently dropped run60's 2,300
        # iterations (0.351 epochs).  Every run after inherited the offset, so
        # run63 reported 0.472 epochs against a true 0.823.
        cum_iter=int(cum_iter if cum_iter is not None else it),
        best_vb=best_vb,
        best_it=best_it,
        lz_strikes=lz_strikes,
        rng_np=rng.bit_generator.state,
        rng_torch=gen.get_state(),
    )
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load_ckpt(path, m):
    """Load a checkpoint of either format; returns the extra state or None.

    format 2 is the dict written by save_ckpt.  Anything else is a bare state_dict from
    run60 and earlier, which carries weights only -- those resume as before, and the log
    says so rather than implying a continuation that cannot happen.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and blob.get("format") == 2:
        missing, _ = m.load_state_dict(blob["model"], strict=False)
        # Tells the caller whether these weights are ALREADY unconstrained, so the warm
        # start is not applied a second time (softplus(softplus(x)) is not a warm start).
        blob["_ckpt_price_soft"] = bool(
            (blob.get("model_flags") or {}).get("price_soft", 0))
        return blob, [k for k in missing if k != "cat_of"]
    missing, _ = m.load_state_dict(blob, strict=False)
    return None, [k for k in missing if k != "cat_of"]


def optimizer_parameter_groups(model, lr, lam_lr_scale=1.0, price_lr_scale=1.0,
                               kappa_lr_scale=1.0):
    """Build Adam groups with an independently controlled product-intercept rate.

    ``lam`` has a qualitatively different information set from the other parameters: its
    iteration-zero value is an exposure-corrected incidence estimate over the *entire*
    training split, whereas an optimiser update sees only one small basket minibatch.
    Giving Adam the same rate for both makes its per-coordinate normalisation turn the
    noisy rare-product gradients into full-sized intercept steps.  Preserve the global
    rate for structural parameters and scale only this already-estimated block.

    The group metadata is saved in Adam's state dict.  ``--fresh-sched`` uses ``lr_scale``
    when it resets a continuation, so it cannot silently restore lam to the main rate.

    ``price_lr_scale`` exists for the same reason, one level down.  Under the softplus
    constraint the trained parameter is the PRE-image: gamma = -3.9186, and the quantity
    that enters the utility is softplus(gamma) = 0.0195.  An Adam step of lr moves the raw
    parameter by ~lr and the effective coefficient by only lr * sigma(gamma) = 0.0195 lr.
    Dropping the softplus makes gamma itself the coefficient, so the SAME lr now moves the
    coefficient 51x further -- 10% of its own value per iteration.  Adam normalises the
    gradient, so this is not visible as a gradient change; run407 diverged in 500 steps
    with E[n] pinned at n_max and the price sign flipped.  Rescaling this group restores
    the effective step while keeping what the reparameterisation was for: no positivity
    floor, and no per-product freezing as sigma -> 0.
    """
    if lam_lr_scale < 0 or price_lr_scale < 0:
        raise ValueError("lr scales must be non-negative")
    named = [(name, value) for name, value in model.named_parameters()
             if value.requires_grad]
    price = [v for n, v in named if n in ("gamma", "beta")] if price_lr_scale != 1.0 else []
    # price_kappa is the SAME failure one level up, with the sign reversed.  It splits the
    # price response into an aggregate part (governed by gamma.beta) and an idiosyncratic
    # part scaled by kappa, and its natural scale is ~40 -- so lr = 0.002 moves it 0.005%
    # per step.  Measured: it travelled 10.12 -> 19.59 over 25,000 iterations and would
    # need ~50,000 more to reach its own likelihood optimum at 40-60, where the fitted
    # own-price elasticity (-0.71 to -1.00) finally brackets the -0.7725 the data shows.
    # Left at the structural rate the block is not converged, it is merely slow.
    kappa = [v for n, v in named if n == "price_kappa"] if kappa_lr_scale != 1.0 else []
    _price_ids = {id(v) for v in price} | {id(v) for v in kappa}
    lam = [v for n, v in named if n == "lam" and id(v) not in _price_ids]
    other = [v for n, v in named if n != "lam" and id(v) not in _price_ids]
    if lam_lr_scale == 1.0 and not price and not kappa:
        return [dict(params=other + lam, lr=lr, lr_scale=1.0, group_name="main")]
    groups = []
    if lam_lr_scale == 1.0:
        other = other + lam
        lam = []
    if other:
        groups.append(dict(params=other, lr=lr, lr_scale=1.0, group_name="main"))
    if lam:
        groups.append(dict(params=lam, lr=lr * lam_lr_scale,
                           lr_scale=lam_lr_scale, group_name="lam"))
    if price:
        groups.append(dict(params=price, lr=lr * price_lr_scale,
                           lr_scale=price_lr_scale, group_name="price"))
    if kappa:
        groups.append(dict(params=kappa, lr=lr * kappa_lr_scale,
                           lr_scale=kappa_lr_scale, group_name="kappa"))
    return groups


def conditional_composition_ce(m, ix, li, lt, bflat, per_trip_out=False):
    """Cross-entropy of the model's OWN leave-one-out conditional, as an AUXILIARY term.

    Why.  SHOPPER's loss is a sequential softmax, so every purchased item is a 5,455-way
    classification against all alternatives: 24 baskets yield ~192 dense discriminative
    gradients.  The version-4 set likelihood yields 24.  Matched on UPDATES -- which is the
    comparison protocol -- that is ~8x less learning signal per step from identical data,
    and it shows up exactly where we lose: composition (-52.97 vs SHOPPER's implied
    -52.15), not size (-3.03 vs -3.02).

    This supplies the same signal without touching the law.  The model already defines the
    leave-one-out conditional, and it is what MRR ranks on:

        score(j | S minus j) = b_j + phi_j . sum_{k in S, k != j} phi_k - rho_c(j) n_c(S minus j)

    A softmax over the trip's assortment against the purchased item is therefore a
    COMPOSITE LIKELIHOOD of this same model -- consistent, not a surrogate for a different
    one.  The reported metric stays the exact set likelihood with the full normaliser; this
    only shapes the path taken to reach it.

    Vectorised per trip: one [n_bought x n_slots] matmul against phi, negligible beside the
    ESP normaliser it rides along with.
    """
    rho_c = m.rho_c
    J = m.lam.shape[0]
    slot_of = torch.full((J,), -1, dtype=torch.long, device=ix.item.device)
    total = bflat.new_zeros(())
    per_trip = bflat.new_zeros(ix.B)
    n_lines = 0
    for t in range(ix.B):
        sel = ix.item_trip == t
        if not bool(sel.any()):
            continue
        items_t = ix.item[sel]
        b_t, phi_t = bflat[sel], m.phi[items_t]
        cat_t = m.cat_of[items_t]
        bought = li[lt == t]
        if bought.numel() == 0:
            continue
        # position of each purchased item inside THIS trip's assortment.  ix.item is ordered
        # by category row, not by item id, so a scatter map is required -- searchsorted
        # would silently return wrong positions.
        slot_of[items_t] = torch.arange(items_t.numel(), device=items_t.device)
        pos = slot_of[bought]
        slot_of[items_t] = -1
        keep = pos >= 0
        if not bool(keep.any()):
            continue
        bought, pos = bought[keep], pos[keep]
        phi_b = m.phi[bought]
        v = phi_b.sum(0)
        cat_b = m.cat_of[bought]
        nc = torch.bincount(cat_b, minlength=rho_c.numel()).to(b_t.dtype)
        U = v.unsqueeze(0) - phi_b                          # [n_b, Kz] leave-one-out
        # final size is |bought|, so s(n) is one constant for this trip
        _s = 1.0
        if getattr(m, "size_bands", None):
            _s = float(m.pair_scale_of_n()[min(int(bought.numel()), m.nmax)])
        sc = b_t.unsqueeze(0) + (U @ phi_t.T) / _s          # [n_b, S_t]
        # -rho_c(c) * n_c, with the held-out item removed from its own category count
        sc = sc - (rho_c[cat_t] * nc[cat_t]).unsqueeze(0)
        sc = sc + (cat_t.unsqueeze(0) == cat_b.unsqueeze(1)).to(sc.dtype) * rho_c[cat_t].unsqueeze(0)
        per_trip[t] = -torch.nn.functional.cross_entropy(sc, pos, reduction="sum")
        total = total - per_trip[t]
        n_lines += int(pos.numel())
    if per_trip_out:
        return per_trip
    return total / max(n_lines, 1)


def observed_composition_loglik(joint_loglik, size_prob, line_trip):
    """Return log P(S | |S|, x) from the joint set likelihood and model size law."""
    n = torch.bincount(line_trip, minlength=joint_loglik.shape[0])
    if bool((n <= 0).any()) or int(n.max()) > size_prob.shape[1]:
        raise ValueError("observed basket lies outside the returned non-empty size law")
    row = torch.arange(joint_loglik.shape[0], device=joint_loglik.device)
    log_size = torch.log(size_prob[row, n - 1].clamp_min(1e-300))
    return joint_loglik - log_size


def observed_factored_loglik(joint_loglik, internal_size_prob, line_trip,
                             external_size_logprob):
    """log P(S||S|,x) + log P_size(|S|), on one complete support."""
    comp = observed_composition_loglik(joint_loglik, internal_size_prob, line_trip)
    n = torch.bincount(line_trip, minlength=joint_loglik.shape[0])
    if int(n.max()) > external_size_logprob.numel():
        raise ValueError("observed basket lies outside external size support")
    return comp + external_size_logprob[n - 1]


def evaluate(m, B, trips, draws, gen, chunk=48, use_units=True,
             return_decomposition=False):
    """Returns (set per basket, set per line, units per basket, total per basket).

    The SET component is reported apart from the units component because the baselines
    model sets only; quoting a total against them would be comparing different objects."""
    tot_s, tot_u, tot_size, n_b, n_l = 0.0, 0.0, 0.0, 0, 0
    for k in range(0, len(trips), chunk):
        sub = trips[k:k + chunk]
        ix, ctx, lctx, hh, li, lt, lc, lq = B.make(sub)
        m.house, m.ctx = hh, ctx
        with torch.no_grad():
            if return_decomposition:
                ll, pn = m.loglik(ix, li, lt, lc, n_draws=draws, generator=gen,
                                  line_ctx=lctx, return_size=True)
                n = torch.bincount(lt, minlength=ix.B)
                tot_size += float(torch.log(
                    pn[torch.arange(ix.B), n - 1].clamp_min(1e-300)).sum())
            else:
                ll = m.loglik(ix, li, lt, lc, n_draws=draws, generator=gen,
                              line_ctx=lctx)
            tot_s += float(ll.sum())
            if use_units:
                tot_u += float(m.units_loglik(li, lt, lq, lctx, ix.B).sum())
        n_b += len(sub)
        n_l += len(li)
    base = (tot_s / n_b, tot_s / n_l, tot_u / n_b, (tot_s + tot_u) / n_b)
    if return_decomposition:
        return base + (tot_size / n_b, (tot_s - tot_size) / n_b)
    return base


def calibrate_poly_degree(m, a, B, D, tr, va, log):
    """Pick the per-row ESP truncation degree, on the model that will actually train.

    Must run AFTER any --resume/--warm-start: the safe degree depends on rho_c, which
    is ~0 on a fresh model (every degree agrees, so anything looks fine) and -0.34 on a
    trained one (degree 40 already returns garbage).  Calibrating before the load chose
    96 for a checkpoint whose true safe ceiling was 32.
    """
    if a.poly_degree == 0:
        return
    if True:
        # The degree cap must be DERIVED from the data, not read off one dataset.  32 happens
        # to be safe here because the largest single-affinity-row count over all 198,690 trips
        # is 26 -- but a catalogue with bigger categories would be silently truncated by a
        # hardcoded constant.  So:
        #   1. floor the cap above the largest row count actually present, with margin;
        #   2. CALIBRATE against the uncapped polynomial on real trips and pick the smallest
        #      degree meeting --poly-degree-tol;
        #   3. keep the measured error in the log and the checkpoint.
        # --poly-degree -1 calibrates; a positive value forces that degree but still verifies.
        _lp, _lc = D["line_ptr"], D["line_cat"]
        _worst = 0
        for _t in np.concatenate([tr, va]):
            _lo, _hi = int(_lp[_t]), int(_lp[_t + 1])
            if _hi > _lo:
                _worst = max(_worst, int(np.bincount(_lc[_lo:_hi]).max()))
        # HARD floor: a row that actually contains _worst items needs degree _worst to have
        # a non-zero coefficient there, else an observed basket gets probability zero.
        # Anything ABOVE that is a judgement about unobserved tail mass, and the calibration
        # below measures that directly rather than guessing a multiplier.
        _floor = min(a.R, _worst)
        _ixc, _cc, _, _hhc, _, _, _, _ = B.make(va[:16])
        _oh, _oc = m.house, m.ctx
        m.house, m.ctx = _hhc, _cc
        # CALIBRATE UPWARD, never against the untruncated polynomial.  exp(-rho_c C(n,2))
        # at rho_c = -0.34 and n = 120 is 10^1045 against float64's 10^308, so degree a.R
        # is precisely the value that overflows -- using it as the reference compares
        # everything to NaN, and NaN <= tol is False, so the loop falls through and returns
        # a.R itself.  Worse, degrees just below overflow are FINITE AND MEANINGLESS: at
        # degree 64 sum_j pi_j = 120.00 = n_max ("every product certain") when the truth is
        # 7.6.  The floor is the largest per-category count actually present, which is the
        # smallest degree that can give an observed basket non-zero probability; go up from
        # there and stop at the first degree that moves the answer.
        with torch.no_grad():
            _z = torch.zeros(_ixc.B, 1, a.Kz, dtype=m.lam.dtype)
            _cands = ([a.poly_degree] if a.poly_degree > 0 else
                      # Above the floor there is no accuracy gain on OBSERVED data, only
                      # unobserved tail mass -- and exp(-rho_c C(n,2)) grows explosively in
                      # n, so reaching for headroom is how the recursion loses precision.
                      # Stay within 1.5x the floor.
                      sorted({_floor} | {d for d in (32, 40, 48, 64, 96)
                                         if _floor <= d <= int(1.5 * _floor)}))
            _base, _chosen, _err, _table = None, int(_floor), 0.0, []
            for _d in _cands:
                _v = log_f_sparse(m, _z, _ixc, sparse_prepare(m, _ixc, degree=_d), True)
                if not bool(torch.isfinite(_v).all()):
                    _table.append((_d, float("nan")))
                    break
                _val = float(_v.mean())
                _table.append((_d, _val))
                if a.poly_degree > 0:
                    _chosen, _err = _d, 0.0
                    break
                if _base is None:
                    _base = _val
                else:
                    _e = abs(_val - _base) / max(abs(_base), 1e-9)
                    if _e > a.poly_degree_tol:
                        break
                    _err = _e
                _chosen = _d
        m.house, m.ctx = _oh, _oc
        if _chosen < _floor:
            raise SystemExit(
                f"--poly-degree {_chosen} < largest observed single-row count {_worst}: an "
                f"observed basket would get probability zero.  Use at least {_worst}, or "
                f"--poly-degree -1 to calibrate automatically.")
        m.poly_degree = m._poly_degree = int(_chosen)
        log(f"  per-row polynomial degree {_chosen} (support unchanged at 1..{a.nmax}). "
            f"largest observed single-row count {_worst}, floor {_floor}; calibrated "
            f"upward, relative |d log f| vs the floor = {_err:.2e} on 16 real trips")
        log("    log f by degree: " + "  ".join(
            f"d{_d}:{'NaN' if _v != _v else f'{_v:.4f}'}" for _d, _v in _table))
        if _err > max(a.poly_degree_tol, 1e-9):
            log(f"  WARNING: degree {_chosen} exceeds the {a.poly_degree_tol:g} tolerance")


def main(a):
    # Subnormal arithmetic runs one to two orders of magnitude slower on CPU, and the ESP
    # coefficients underflow into that range as soon as the mode iteration wanders.
    torch.set_flush_denormal(True)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(a.seed)
    if os.environ.get("V3_DETECT_ANOMALY", "0") == "1":
        torch.autograd.set_detect_anomaly(True)
        log("autograd anomaly detection enabled")
    if not 0.0 <= a.phi_step_scale <= 1.0:
        raise SystemExit("--phi-step-scale must lie in [0,1]")
    if not 0.0 <= a.rho_c_step_scale <= 1.0:
        raise SystemExit("--rho-c-step-scale must lie in [0,1]")
    if a.lam_lr_scale < 0.0:
        raise SystemExit("--lam-lr-scale must be non-negative")
    if a.pi_project_every < 0:
        raise SystemExit("--pi-project-every must be non-negative")
    if a.qmc_refresh_every < 0 or a.qmc_eval_n < 0:
        raise SystemExit("--qmc-refresh-every and --qmc-eval-n must be non-negative")
    if a.resume and a.warm_start:
        raise SystemExit("--resume and --warm-start are mutually exclusive")
    if sum(bool(x) for x in (a.interaction_stage, a.composition_stage, a.size_stage)) > 1:
        raise SystemExit("--interaction-stage, --composition-stage, and --size-stage "
                         "are mutually exclusive")
    if a.composition_stage and (a.cd or a.pseudo or a.adapt_draws != 1):
        raise SystemExit("--composition-stage requires the joint likelihood and adapt-draws=1")
    if a.multinomial_utility_start and not a.warm_start:
        raise SystemExit("--multinomial-utility-start requires a full-model --warm-start")
    if a.factored_size and not a.allow_factored_ablation:
        raise SystemExit(
            "--factored-size changes the version-4 joint law and is retired from normal "
            "training. Reproducing that diagnostic ablation requires the explicit "
            "--allow-factored-ablation 1 acknowledgement.")
    if a.factored_size and (a.size_stage or a.size_ipf_steps or a.reinit_rho0_after_warm):
        raise SystemExit("--factored-size replaces rho_0 calibration and is incompatible "
                         "with size-stage, size-IPF, and reinit-rho0-after-warm")
    if a.size_ipf_steps < 0 or a.size_ipf_trips <= 0 or not 0 < a.size_ipf_damp <= 1:
        raise SystemExit("size IPF requires steps >= 0, trips > 0, and damp in (0,1]")
    D = build()
    J, N, C, S = int(D["n_item"]), int(D["n_user"]), int(D["n_cat"]), int(D["n_store"])
    if a.require_version4:
        failures = []
        if os.environ.get("V3_AFFINITY", "0") != "1" or C != 280:
            failures.append("V3_AFFINITY=1 with the 280-category affinity partition")
        if a.factored_size:
            failures.append("the original joint size law (--factored-size 0)")
        if a.warm_start:
            failures.append("a genuinely fresh lineage (no --warm-start)")
        # A format-2 continuation preserves the optimizer, scheduler and both RNG streams;
        # it is the same fresh lineage, not a warm start.  The old guard rejected every
        # legitimate crash recovery, which forced operators either to drop the guard or to
        # restart weeks of work.  Admit only a self-describing checkpoint that already
        # carries the complete version-4 universe and estimator contract.
        if a.resume:
            try:
                _resume_guard = torch.load(a.resume, map_location="cpu", weights_only=False)
            except Exception as exc:
                failures.append(f"a readable continuation checkpoint ({exc})")
                _resume_guard = None
            if not isinstance(_resume_guard, dict) or _resume_guard.get("format") != 2:
                failures.append("a provenance-complete format-2 continuation checkpoint")
            else:
                _rd = _resume_guard.get("data", {})
                _rq = _resume_guard.get("quad", {})
                _rm = _resume_guard.get("model", {})
                if (_rd.get("affinity") != "1" or int(_rd.get("n_cat", -1)) != 280
                        or int(_rd.get("n_item", -1)) != 5455):
                    failures.append("a continuation from the affinity-280, 5,455-product universe")
                if (int(_rd.get("nmax", -1)) < int(D["trip_nlines"].max())
                        or int(_rd.get("R", -1)) < int(_rd.get("nmax", -1))
                        or int(_rd.get("rho_pair_cap", -1)) != int(_rd.get("nmax", -2))):
                    failures.append("a continuation with complete unsaturated size/category support")
                if int(_rq.get("Kz", -1)) < 32 or int(_rq.get("qmc_n", 0)) <= 0:
                    failures.append("a continuation trained with the audited rank/QMC normalizer")
                _factored = _rm.get("factored_size_enabled")
                if _factored is None or bool(torch.as_tensor(_factored).item()):
                    failures.append("a continuation of the original non-factored joint law")
        if a.phi_mask or a.phi_topk > 0:
            failures.append("all 5,455 products carrying interaction embeddings")
        if a.Kz < 32:
            failures.append("interaction rank Kz >= 32")
        observed_max = int(D["trip_nlines"].max())
        if a.nmax < observed_max or a.R < a.nmax:
            failures.append(
                f"complete basket/category support (nmax >= {observed_max}, R >= nmax)")
        if a.qmc_n <= 0:
            failures.append("the audited positive-weight QMC normalizer (--qmc-n > 0)")
        if failures:
            raise SystemExit("--require-version4 invariant failure: " + "; ".join(failures))
        log("version-4 experiment guard: PASS (fresh-lineage, affinity-280, original joint law, "
            "full catalogue/rank/support, QMC normalizer)")
    F = Features(J, S, 712)
    B = Batcher(D, F, a.nmax, price_ref=a.price_ref)

    tr = np.flatnonzero(D["trip_split"] == 0)
    va = np.flatnonzero(D["trip_split"] == 1)

    def in_support(idx):
        """A basket the normaliser does not sum over must not be scored by the energy.
        Clamping n to n_max, as an earlier version did, silently relabels such a basket as
        in-support; dropping it is the honest treatment and the count is reported."""
        lp, lc_ = D["line_ptr"], D["line_cat"]
        keep = np.ones(len(idx), bool)
        for i, t in enumerate(idx):
            lo, hi = int(lp[t]), int(lp[t + 1])
            if hi - lo > a.nmax or (hi > lo and np.bincount(lc_[lo:hi]).max() > a.R):
                keep[i] = False
        return keep

    kt, kv = in_support(tr), in_support(va)
    log(f"support (n_max={a.nmax}, R={a.R}): dropping {int((~kt).sum()):,} of {len(tr):,} "
        f"training ({(~kt).mean():.2%}) and {int((~kv).sum()):,} of {len(va):,} "
        f"validation ({(~kv).mean():.2%}) trips that lie outside it")
    tr, va = tr[kt], va[kv]
    # Validation must be a fixed RANDOM sample, not a prefix.  va[:128] happens to contain
    # no pathological trip -- its worst model E[n] is 16.5, where a random sample reaches
    # 117 -- so the between-trip term of Var(n) read 5.9 instead of 88.3 and the logged
    # Var(n) came out 59.7 against a true 117-185.  Every eval below indexes va, so
    # permuting it once here makes the slice representative without changing any call site.
    # Third time today a statistic has hidden behind a slice that excludes its own tail.
    va = va[np.random.default_rng(12345).permutation(len(va))]
    log(f"{len(tr):,} training trips, {len(va):,} validation")
    # Record the configuration.  Recovering whether run9 used cosine decay meant grepping a
    # session transcript for the launch command, because neither the log nor the checkpoint
    # carried it -- a comparison between runs is not checkable if the runs are not labelled.
    cfg = {k: v for k, v in sorted(vars(a).items())}
    log("config: " + "  ".join(f"{k}={v}" for k, v in cfg.items()))
    if not a.factored_size and not a.composition_stage:
        log("theory invariant: original version-4 joint law; full log Z = "
            "log sum_n exp[-rho_0(n)] Z_n(x), price-responsive size retained")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"v3_{a.label}.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)

    m = RaggedModel(J=J, N=N, C=C, K=a.K, Kz=a.Kz, nmax=a.nmax, R=a.R, seed=a.seed,
                    S=S, Kp=a.Kp, phi_init=a.phi_init, taste_init=a.taste_init)
    npar = sum(p.numel() for p in m.parameters())
    log(f"parameters: {npar:,}  (K={a.K}, Kz={a.Kz}, Kp={a.Kp}, nmax={a.nmax}, R={a.R})")

    # Empirical size law over sizes 1..nmax, for the calibration penalty below.
    _c = np.bincount(np.clip(D["trip_nlines"][tr], 0, a.nmax), minlength=a.nmax + 1) + 0.5
    emp_pn = torch.as_tensor((_c / _c.sum())[1:], dtype=torch.float64)
    emp_pn = emp_pn / emp_pn.sum()
    _n = np.clip(D["trip_nlines"][tr], 1, a.nmax)
    emp_var = float(_n.var())
    log(f"empirical size law: mean {_n.mean():.2f}  var {emp_var:.1f}")

    with torch.no_grad():                      # product -> category, for the conditional
        _co = torch.zeros(J, dtype=torch.long)
        _co[torch.as_tensor(D["line_item"], dtype=torch.long)] = \
            torch.as_tensor(D["line_cat"], dtype=torch.long)
        m.cat_of.copy_(_co)
    if a.init_popularity and not a.resume:
        with torch.no_grad():
            _pop = popularity_logits(D, tr).to(dtype=m.lam.dtype, device=m.lam.device)
            m.lam.copy_(_pop)
        log(f"lam initialised from training incidence/exposure: sd {float(m.lam.std()):.3f}, "
            f"range {float(m.lam.min()):.2f}..{float(m.lam.max()):.2f}")
    if a.objective == "composite":
        m.rho_0_free.requires_grad_(False)     # gets exactly zero gradient anyway
        log(f"  COMPOSITE objective: normaliser-free leave-one-out conditional, "
            f"0.063 s/step vs 2.292 (36.6x). rho_0 FROZEN at the empirical size law "
            f"(zero gradient under this objective; costs ~0.008 nats). "
            f"Evaluation still uses the exact normaliser.")
    # Now that the weights are final (fresh, warm-started or resumed), choose the
    # truncation degree against THEIR rho_c.
    calibrate_poly_degree(m, a, B, D, tr, va, log)
    m.price_ref = a.price_ref
    if a.price_ref != "trip":
        _grp = {"category": "the store's own category (median 128 products as purchases "
                            "experience it)",
                "subcommodity": "the item's sub-commodity (median 16), the same grouping "
                                "the data's rival price is measured over"}[a.price_ref]
        log(f"  price reference: {a.price_ref} -- dlp is measured against {_grp}, not the "
            f"whole assortment (median 5,292).  Substitution enters ONLY here: a rival's "
            f"rise moves b_j by gb(kappa-1) d_dbar, and the reference's WIDTH sets the "
            f"magnitude.  Measured before refitting, cross-price elasticity is -0.1621 "
            f"(trip), +0.0702 (category), +0.4757 (subcommodity); the data shows +0.1351")
    if a.price_soft:
        # The reparameterisation is gamma' = softplus(gamma), so the warm start is only
        # identity if it is applied to the weights the run will actually TRAIN.  Doing it
        # here would convert the FRESH init and then have --resume/--warm-start overwrite
        # gamma with the checkpoint's raw pre-softplus values while price_soft stayed set:
        # price_g() would return raw gamma, which is below -3 for 100% of products, i.e. a
        # large negative price coefficient instead of ~0.01.  That is run406's initial eval
        # of -270,161.  The conversion is deferred to just before the optimiser is
        # built, after every checkpoint load path has run.
        m.price_soft = True
        log(f"  price: UNCONSTRAINED bilinear + hinge penalty (weight {a.price_hinge_w}). "
            f"softplus saturated 100% of gamma below -3, block gradient 5.6e-4 vs 0.05-2.2 "
            f"elsewhere; warm start applied after checkpoint load so step 0 is identical")
    # degree calibration happens after the checkpoint load; see below.
    if a.phi_pool == "mean":
        m.size_bands = size_band_scales(a.nmax, a.size_bands, pool="mean")
        m._phi_pool, m._size_bands_n = a.phi_pool, a.size_bands
        log(f"  MEAN-pooled interaction: (1/s) sum_{{j<k}} phi_j.phi_k over "
            f"{len(m.size_bands)} size bands "
            f"{[(lo, hi, round(s, 2)) for lo, hi, s in m.size_bands]}")
        log(f"  energy, normaliser, MRR and the conditional all use this same s(n); "
            f"verified against exact enumeration to 2e-5 nats")
    if a.qmc_n > 0:
        if a.qmc_reps < 2:
            raise SystemExit("QMC training requires --qmc-reps >= 2 so log Z error is observable")
        # DETERMINISTIC log Z at ANY rank, with NO product mask.
        #
        # Smolyak (below) costs O(Kz^q), which is what pinned Kz at 4, and a fixed grid
        # cannot follow the integrand's mode, which is what forced the 20-product mask.
        # Adaptive scrambled-Sobol costs qmc_n nodes at every Kz, and _log_Z_adaptive
        # shifts it to the mode.  A FIXED seed makes it a deterministic rule -- the same
        # nodes every step -- so it adds no gradient variance, unlike the importance
        # sampler it replaces.  See ragged.sobol_grid for the measurements.
        def _set_qmc_rule(nodes, seed):
            nodes, seed = int(nodes), int(seed)
            mix_n = (int(a.qmc_mix_n) if nodes == int(a.qmc_n) and a.qmc_mix_n > 0
                     else 2 * nodes)
            description = set_quad(
                m, qmc_n=nodes, qmc_seed=seed, Kz=a.Kz,
                probe=a.quad_probe, steps=a.quad_steps, chunk=a.quad_chunk,
                qmc_reps=a.qmc_reps, size_bands=a.qmc_size_bands,
                size_steps=a.qmc_size_steps, mode_logtol=a.qmc_mode_logtol,
                mode_sep=a.qmc_mode_sep, mix_n=mix_n)
            m._qmc_n, m._qmc_seed, m._quad_probe = nodes, seed, a.quad_probe
            m._qmc_reps = a.qmc_reps
            m._quad_steps, m._quad_chunk = a.quad_steps, a.quad_chunk
            m._qmc_size_bands, m._qmc_size_steps = a.qmc_size_bands, a.qmc_size_steps
            m._qmc_mode_logtol, m._qmc_mode_sep = a.qmc_mode_logtol, a.qmc_mode_sep
            m._qmc_mix_n = mix_n
            return description

        desc = _set_qmc_rule(a.qmc_n, a.qmc_seed)
        log(f"  log Z: {desc} (replaces {a.draws} sampled draws)")
        log(f"  verified in this kernel vs exact enumeration over all 2^J-1 subsets, "
            f"every product carrying phi at ||phi_j||=0.96: -0.00006 nats at Kz=128, "
            f"+0.00035 at Kz=512")
    elif a.quad_q > 0:
        # DETERMINISTIC log Z.  f(z) was always exact (the ESP recursion is a closed
        # form); only the outer E_z was sampled, and importance sampling cannot do
        # that integral -- verified against exact enumeration, 4096 draws are wrong
        # by 8-36 nats.  A Smolyak grid does it to 0.0009 nats at Kz=2, q=6.
        desc = set_quad(m, quad_q=a.quad_q, Kz=a.Kz)
        m._quad_q = a.quad_q
        log(f"  log Z: {desc} (replaces {a.draws} sampled draws)")
    if a.no_rec:
        with torch.no_grad():
            m.psi.zero_()
        m.psi.requires_grad_(False)
        log("  --no-rec: psi zeroed and frozen (recency removed from b)")

    if a.init_rho0:
        # Initialise the size potential at the empirical basket-size law.
        #
        # P(n | z) is proportional to exp(-rho_0(n)) A_n(z), so setting
        # rho_0(n) = log A_n(0) - log target(n) makes the size law equal `target` at z = 0.
        # The BEMB-style multinomial baseline is handed the empirical marginal size law.
        # This initialisation matches it only at the reference context used below: with
        # phi=rho_c=0 the model has P(n|x) proportional to exp(-rho_0(n))*e_n(exp b(x)),
        # so no single context-free rho_0 can impose the same P(n) for every x.  It is still
        # a far better initial value than zero, but it must not be described as an exact
        # equivalence to the externally factored multinomial baseline.
        _size_mean = initialize_size_potential(m, D, tr, B, a.nmax)
        log(f"rho_0 initialised at the empirical size law "
            f"(mean {_size_mean:.2f} lines)")
    resume_blob = None
    warm_blob = None
    if a.warm_start:
        # A mature nested model already identifies the expensive high-dimensional utility,
        # size, and units blocks.  Reusing those WEIGHTS avoids spending full-QMC updates
        # relearning an additive model.  This is deliberately not --resume: optimiser and
        # RNG state start fresh because the interaction parameters are a new stage.
        _phi_fresh = m.phi.detach().clone()
        warm_blob, _miss = load_ckpt(a.warm_start, m)
        _NEW_OK = {"price_kappa", "factored_size_enabled", "factored_size_log_p"}
        _miss = [k for k in _miss if k not in _NEW_OK]
        if _miss:
            raise SystemExit(f"warm-start checkpoint is missing fitted parameters: {_miss}")
        if warm_blob is not None and warm_blob.get("data"):
            _wd = warm_blob["data"]
            if int(_wd.get("n_item", J)) != J or int(_wd.get("n_cat", C)) != C:
                raise SystemExit(
                    f"warm-start support mismatch: checkpoint has "
                    f"J={_wd.get('n_item')} C={_wd.get('n_cat')}, data has J={J} C={C}")
        if a.reinit_interactions:
            with torch.no_grad():
                m.phi.copy_(_phi_fresh)
                m.rho_c.zero_()
            log(f"warm-started additive weights from {os.path.basename(a.warm_start)}; "
                f"phi reinitialised at scale {a.phi_init:g}, rho_c reset to zero")
        else:
            log(f"warm-started weights from {os.path.basename(a.warm_start)}; "
                "optimizer, scheduler, and RNG start fresh")
    if a.resume:
        resume_blob, _miss = load_ckpt(a.resume, m)
        # Parameters introduced after a checkpoint was written, whose fresh initialisation
        # reproduces the OLD model exactly, may be absent -- resuming then starts from the
        # previous behaviour rather than a different one.  price_kappa initialises to
        # softplus(0.5413) = 1.0, which is the un-split price term.  Anything else missing
        # is a genuine mismatch and still fails.
        _NEW_OK = {"price_kappa", "factored_size_enabled", "factored_size_log_p"}
        _miss = [k for k in _miss if k not in _NEW_OK]
        if _miss:
            raise SystemExit(f"resume checkpoint is missing fitted parameters: {_miss}")
        if resume_blob is None:
            log(f"resumed WEIGHTS ONLY from {os.path.basename(a.resume)} "
                f"(pre-format-2 checkpoint: no optimiser, schedule or RNG state to restore)")
        else:
            log(f"resumed from {os.path.basename(a.resume)} at iteration "
                f"{resume_blob['iter']} -- optimiser, schedule and RNG restored")
    if a.kappa_init > 0 and hasattr(m, "price_kappa"):
        # kappa is SLOW: measured, it moves 1.41 units per 1,000 iterations even at 20x the
        # structural rate, because its gradient is small and sign-noisy over 24-trip
        # minibatches, so Adam's averaging cancels most of it.  Where it STARTS therefore
        # decides where it ends.  It is also identified from the data without the model:
        # the item-week panel with item fixed effects gives an own-price elasticity of
        # -0.7725 (controlling for display and mailer, which the model carries separately),
        # and a sweep on the fitted model puts the likelihood's OWN optimum at kappa 40-60,
        # spanning elasticity -0.71 to -1.00.  Data and likelihood agree, so start at the
        # value they agree on instead of waiting ~13,500 iterations to walk there.
        with torch.no_grad():
            _k0 = float(torch.nn.functional.softplus(m.price_kappa))
            m.price_kappa.fill_(float(np.log(np.expm1(a.kappa_init))))
        log(f"  kappa initialised {_k0:.2f} -> {a.kappa_init:.2f} "
            f"(data-implied own-price elasticity -0.7725; likelihood optimum 40-60)")
    if a.price_soft and resume_blob is not None and resume_blob.get("_ckpt_price_soft"):
        log("  price warm start SKIPPED: checkpoint already stores unconstrained "
            "gamma/beta (applying softplus again would not be a warm start)")
    elif a.price_soft:
        # Now that gamma/beta hold the weights this run starts from -- fresh init, warm
        # start, or resume -- map them through the constraint so the model is unchanged at
        # step 0 while the gradient is free.  Verified |dloss| = 0.00e+00 on run404's best.
        with torch.no_grad():
            _g0, _b0 = m.gamma.detach().clone(), m.beta.detach().clone()
            m.gamma.copy_(torch.nn.functional.softplus(m.gamma))
            m.beta.copy_(torch.nn.functional.softplus(m.beta))
            # DESATURATE whatever we were handed.  A checkpoint trained long under the
            # constraint arrives with gamma ratcheted deep into softplus saturation --
            # measured, mean -4.72 and min -5.88 after 38,500 iterations, i.e. effective
            # coefficients of 0.0097 and 0.0028.  Unconstrained, an Adam step of lr moves
            # those by lr regardless of their size, so the smallest cross ZERO within a few
            # hundred iterations, the price coefficient changes sign, and E[n] runs to
            # n_max.  Stage 2 must not depend on stage 1 having been floored: it can see the
            # saturation in the weights it loads, so it repairs it here.  The likelihood
            # cost is tiny (these coefficients are ~0 by construction) and project_price
            # puts the aggregate straight back on target at the next projection.
            _sat = 0
            if a.price_soft_floor > 0:
                _sat = int((m.gamma < a.price_soft_floor).sum()
                           + (m.beta < a.price_soft_floor).sum())
                m.gamma.clamp_(min=a.price_soft_floor)
                m.beta.clamp_(min=a.price_soft_floor)
        if a.price_soft_floor > 0:
            log(f"  price warm start: desaturated {_sat:,} coefficients below "
                f"{a.price_soft_floor:g} (an Adam step of lr moves them by lr regardless "
                f"of size, so saturated ones cross zero and flip the price sign)")
        log(f"  price warm start: gamma {_g0.mean():+.4f} -> {m.gamma.mean():+.4f}, "
            f"beta {_b0.mean():+.4f} -> {m.beta.mean():+.4f} "
            f"(softplus applied to the loaded weights, identity in the likelihood)")

    if a.multinomial_utility_start:
        try:
            transfer_multinomial_nonprice(m, a.multinomial_utility_start)
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(f"cannot transfer multinomial utilities: {exc}") from exc
        log(f"gauge-mapped taste/season/store/promotion utilities from "
            f"{os.path.basename(a.multinomial_utility_start)}; main price and interaction "
            "blocks retained")
    if a.factored_size:
        with torch.no_grad():
            m.factored_size_enabled.fill_(True)
            m.factored_size_log_p.copy_(emp_pn.clamp_min(1e-300).log())
        m.rho_0_free.requires_grad_(False)
        log("factored size objective: exact conditional composition plus the smoothed "
            "empirical training size law; rho_0 frozen because it cancels identically")
    elif bool(m.factored_size_enabled):
        m.rho_0_free.requires_grad_(False)
        log("factored size objective restored from checkpoint; rho_0 remains frozen")
    if a.reinit_rho0_after_warm:
        _size_mean = initialize_size_potential(m, D, tr, B, a.nmax)
        log(f"rho_0 reinitialised after utility transfer at empirical size law "
            f"(mean {_size_mean:.2f} lines)")
    if a.size_ipf_steps:
        _ipf = calibrate_size_ipf(
            m, D, tr, B, a.nmax, steps=a.size_ipf_steps,
            n_trips=a.size_ipf_trips, chunk=a.batch, damp=a.size_ipf_damp)
        for _row in _ipf:
            log(f"  size IPF {_row['step'] + 1}/{a.size_ipf_steps}: "
                f"E[n] {_row['mean']:.2f}, KL(target||model) {_row['kl']:.4f}, "
                f"max |log ratio| {_row['max_log_ratio']:.2f}")
    if a.freeze_rho0:
        # Freeze rho_0 at the empirical size law.
        #
        # Pseudo-likelihood identifies rho_0 only through its DIFFERENCES, since that is all
        # a conditional contains -- very weak information about a 120-parameter function
        # whose level sets E[n].  Left free it ran E[n] to 62.5 against an observed 7.2 with
        # the exact full-sum estimator, so the failure is the method, not the sampling.  The
        # empirical size law is correct by counting and has nothing for a fit to improve, so
        # each estimator is used where it is strong: counting for the size law, conditionals
        # for the interaction.
        m.rho_0_free.requires_grad_(False)
        log("rho_0 frozen at the empirical size law")
    if a.zero_phi:
        # A genuinely nested no-latent-interaction fit.  Zeroing phi only while scoring a
        # fitted full model is not a fair baseline: the remaining item/context utilities
        # have never been allowed to absorb the signal previously carried by phi.  Apply
        # this after resume, then remove phi from the optimiser so it remains exactly zero.
        with torch.no_grad():
            m.phi.zero_()
        m.phi.requires_grad_(False)
        m._exact_additive = True
        log("phi zeroed and frozen: fitting the nested no-latent-interaction model")
    if a.zero_rho_c:
        # The second interaction block is the within-affinity-category pair potential.
        # Freeze it too for the exact additive-utility nested model.
        with torch.no_grad():
            m.rho_c.zero_()
        m.rho_c.requires_grad_(False)
        log("rho_c zeroed and frozen: fitting without category-count interactions")
    if a.interaction_stage:
        if a.zero_phi or a.zero_rho_c:
            raise SystemExit("--interaction-stage cannot be combined with zeroed interactions")
        _live = {"phi", "rho_c"} if bool(m.factored_size_enabled) \
            else {"phi", "rho_c", "rho_0_free"}
        for _name, _parameter in m.named_parameters():
            _parameter.requires_grad_(_name in _live)
        log("  interaction stage: training "
            + ("phi and rho_c only; factored size law fixed"
               if bool(m.factored_size_enabled)
               else "phi, rho_c, and rho_0 only")
            + "; mature utility and units blocks frozen")
    elif a.composition_stage:
        _frozen = {"rho_0_free", "a_q", "gamma_q", "beta_q", "log_r"}
        for _name, _parameter in m.named_parameters():
            if _name in _frozen:
                _parameter.requires_grad_(False)
        log("  composition stage: optimizing log P(S | |S|, x); "
            "size potential and units block frozen")
    elif a.size_stage:
        for _name, _parameter in m.named_parameters():
            _parameter.requires_grad_(_name == "rho_0_free")
        log("  size stage: training rho_0 only; composition and units fixed exactly")
    opt = torch.optim.Adam(
        optimizer_parameter_groups(m, a.lr, a.lam_lr_scale,
                                   a.price_lr_scale if a.price_soft else 1.0,
                                   a.kappa_lr_scale),
        lr=a.lr, weight_decay=a.wd)
    if a.price_soft and a.price_lr_scale != 1.0:
        log(f"  separate price learning rate: {a.lr * a.price_lr_scale:g} "
            f"({a.price_lr_scale:g}x structural rate {a.lr:g}) -- gamma/beta are now the "
            f"coefficients themselves, not softplus pre-images, so an unscaled step is "
            f"51x the constrained one and diverges")
    if a.kappa_lr_scale != 1.0 and hasattr(m, "price_kappa"):
        log(f"  separate kappa learning rate: {a.lr * a.kappa_lr_scale:g} "
            f"({a.kappa_lr_scale:g}x structural rate {a.lr:g}) -- kappa's natural scale is "
            f"~40, so the structural rate moves it 0.005% per step and it cannot reach "
            f"its optimum inside a run")
    if a.lam_lr_scale != 1.0 and m.lam.requires_grad:
        log(f"  separate lam learning rate: {a.lr * a.lam_lr_scale:g} "
            f"({a.lam_lr_scale:g}x structural rate {a.lr:g})")
    # Learning-rate schedule.  A mature checkpoint needs about one tenth of the rate that
    # efficiently fits an immature one, but run103 used the mature rate from iteration 200
    # onward and consequently spent 7,200 updates doing roughly one quarter the useful
    # work.  Relative MultiStep milestones make that staging explicit for continuations:
    # e.g. --lr .002 --lr-milestones 2000,6000 --lr-gamma .5 gives .002 -> .001 -> .0005.
    # They take precedence over cosine.  The scheduler counter is relative to THIS process,
    # which is exactly what a fresh schedule on a resumed checkpoint requires.
    try:
        lr_milestones = tuple(int(x) for x in a.lr_milestones.split(",") if x.strip())
    except ValueError as exc:
        raise SystemExit("--lr-milestones must be comma-separated positive integers") from exc
    if any(x <= 0 for x in lr_milestones) or tuple(sorted(lr_milestones)) != lr_milestones:
        raise SystemExit("--lr-milestones must be positive and sorted")

    def make_scheduler():
        if lr_milestones:
            return torch.optim.lr_scheduler.MultiStepLR(
                opt, milestones=list(lr_milestones), gamma=a.lr_gamma)
        return (torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=a.iters, eta_min=a.lr * a.lr_floor)
                if a.cosine else None)

    sched = make_scheduler()
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    it0 = 0                                   # iterations already done, for a continuation
    cum_base = (int(warm_blob.get("cum_iter", warm_blob.get("iter", 0)))
                if warm_blob is not None else 0)  # ... including lineages before this one
    if warm_blob is not None:
        log(f"  warm-start lineage carries {cum_base:,} prior utility-training updates; "
            "this stage begins at iteration 0 with fresh Adam moments")
    if resume_blob is not None:
        try:
            opt.load_state_dict(resume_blob["opt"])
            if a.price_soft:
                # gamma/beta are no longer the same coordinates they were when these
                # moments were accumulated (the gradient is 43x larger unconstrained), so
                # the carried exp_avg/exp_avg_sq are about a function that no longer
                # exists.  Clear just those two; everything else resumes untouched.
                for _p in (m.gamma, m.beta):
                    opt.state.pop(_p, None)
                log("  price warm start: cleared Adam moments for gamma/beta "
                    "(reparameterised; carried moments describe the old coordinates)")
        except ValueError:
            # The parameter set changed since the checkpoint (see _NEW_OK above), so Adam's
            # saved moments no longer line up.  Weights ARE restored; only the moment
            # estimates start fresh, which costs a few hundred iterations of warm-up.
            log("  optimiser state not restored: the parameter set changed since this "
                "checkpoint (weights restored, Adam moments start fresh)")
        if sched is not None and resume_blob.get("sched") is not None:
            if a.fresh_sched:
                # A finished cosine run resumes at its FLOOR (2% of lr), so continuing it
                # trains at 2% rate and "converges" trivially.  opt.load_state_dict above
                # also restores param_groups['lr'], so the schedule must be reset AND the
                # rate put back, or the fresh schedule anneals from the floor to the floor.
                for _g in opt.param_groups:
                    _group_lr = a.lr * float(_g.get("lr_scale", 1.0))
                    _g["lr"] = _group_lr
                    _g["initial_lr"] = _group_lr
                sched = make_scheduler()
                _sd = (f"MultiStep milestones {list(lr_milestones)}, gamma {a.lr_gamma:g}"
                       if lr_milestones else f"cosine over {a.iters} iterations")
                log(f"  --fresh-sched: lr reset to {a.lr:g}, {_sd}; "
                    "optimiser moments and RNG still carried")
            else:
                sched.load_state_dict(resume_blob["sched"])
        rng.bit_generator.state = resume_blob["rng_np"]
        gen.set_state(resume_blob["rng_torch"])
        it0 = int(resume_blob["iter"])
        cum_base = int(resume_blob.get("cum_iter", it0))
        _lr_desc = "/".join(
            f"{g.get('group_name', k)}={g['lr']:.6g}" for k, g in enumerate(opt.param_groups))
        log(f"  continuing from iteration {it0}: lr {_lr_desc} "
            f"(a fresh schedule would start at {a.lr:.6f}), "
            f"Adam state for {len(opt.state)} tensors")
        if it0 >= a.iters:
            raise SystemExit(f"--iters {a.iters} is at or below the resumed iteration "
                             f"{it0}; nothing left to run")

    # Apply the exact projection that will constrain training BEFORE scoring/saving
    # iteration zero.  Previously the score described the unprojected random phi while the
    # "iteration-zero best" file contained projected weights when the operator cap bound.
    phi_cap = a.phi_max
    if a.phi_deg:
        _dg = np.load(a.phi_deg)
        if _dg.shape[0] != m.phi.shape[0]:
            raise SystemExit(f"degree file covers {_dg.shape[0]} products, model has "
                             f"{m.phi.shape[0]}")
        _sc = np.minimum(np.sqrt(_dg), a.phi_deg_cap)
        phi_cap = (a.phi_max * torch.as_tensor(_sc, dtype=m.phi.dtype)).unsqueeze(1)
        log(f"degree-aware phi cap: {float(phi_cap.min()):.2f}..{float(phi_cap.max()):.2f}, "
            f"{int((_sc > 1.0).sum())} products above the base, "
            f"{int((_sc >= a.phi_deg_cap).sum())} at the ceiling")
    m.project(phi_cap, op_max=a.phi_op_max)

    # Training needs a low-variance gradient, not a publication-precision likelihood on
    # every noisy minibatch.  A larger independent rule is used for iteration zero and
    # checkpoints; refreshed small rules are installed at the top of each training step.
    # Both rules use the same exact importance identity and therefore target the same Z.
    _qmc_eval_n = int(a.qmc_eval_n) if a.qmc_eval_n > 0 else int(a.qmc_n)
    if a.qmc_n > 0 and _qmc_eval_n < a.qmc_n:
        raise SystemExit("--qmc-eval-n must be at least --qmc-n")
    if a.qmc_n > 0 and a.qmc_refresh_every > 0:
        log(f"  stochastic RQMC training: refresh every {a.qmc_refresh_every} update(s); "
            f"{a.qmc_n} training nodes, {_qmc_eval_n} fixed checkpoint nodes")
    if a.qmc_n > 0 and _qmc_eval_n > a.qmc_n:
        _set_qmc_rule(_qmc_eval_n, a.qmc_seed + 2_000_003)

    _initial_eval = None
    _initial_rec = None
    # The mask MUST be applied before the initial evaluation.  It used to sit ~80 lines
    # later, so iteration zero was scored with phi on all 5,455 products: sparse_prepare
    # then treats ~127,000 slots as z-dependent instead of 720, and the eval goes from
    # 0.47 s per 48 trips to 44.63 s -- 95x, about six minutes of apparent "startup stall"
    # for a number that is wrong anyway, since it describes an unmasked model.
    phi_mask = None
    if a.phi_init_file:
        _pi = np.load(a.phi_init_file)
        if _pi.shape != tuple(m.phi.shape):
            raise SystemExit(f"--phi-init-file has shape {_pi.shape}, model phi is "
                             f"{tuple(m.phi.shape)} -- wrong mask or rank")
        with torch.no_grad():
            m.phi.copy_(torch.as_tensor(_pi, dtype=m.phi.dtype))
        _n = m.phi.norm(dim=1)
        log(f"phi initialised from {os.path.basename(a.phi_init_file)}: "
            f"{int((_n > 0).sum())} active rows, |phi_j| median "
            f"{float(_n[_n > 0].median()):.3f} max {float(_n.max()):.3f} "
            f"(spectral placement, not the 0.03 saddle seed)")
    if a.phi_mask:
        _mk = np.load(a.phi_mask)
        if _mk.shape[0] != m.phi.shape[0]:
            raise SystemExit(f"mask covers {_mk.shape[0]} products, model has "
                             f"{m.phi.shape[0]} -- wrong partition or catalogue")
        phi_mask = torch.as_tensor(_mk, dtype=m.phi.dtype).unsqueeze(1)
        with torch.no_grad():
            m.phi.mul_(phi_mask)          # applied at init too, not only after each step
        log(f"phi restricted to {int(_mk.sum())} of {_mk.shape[0]} products "
            f"({100.0*_mk.sum()/_mk.shape[0]:.2f}%) from {os.path.basename(a.phi_mask)}")

    if a.eval_initial:
        # Baseline scripts historically reported only post-training values while fit.py's
        # first visible point was iteration 100 or 200.  Log iteration zero explicitly on
        # the exact validation slice so a future comparison cannot mistake initialization
        # for a trained result.  A private generator leaves the training stream unchanged.
        _eval_gen = torch.Generator().manual_seed(a.seed + 9001)
        _vb0, _vl0, _vu0, _vt0, _vsz0, _vco0 = evaluate(
            m, B, va[:a.n_val], a.draws * 4, _eval_gen, use_units=a.units,
            return_decomposition=True)
        _initial_eval = (_vb0, _vl0, _vu0, _vt0, _vsz0, _vco0)
        if a.n_rec > 0:
            _initial_rec = rec_eval(m, B, va[:a.n_rec])
        log(f"initial it {it0:6d}  set/basket {_vb0:.4f}  set/line {_vl0:.4f}  "
            f"size/basket {_vsz0:.4f}  comp/basket {_vco0:.4f}  "
            f"units/basket {_vu0:.4f}  total/basket {_vt0:.4f}"
            + (f"  MRR {_initial_rec[0]:.4f}(med {_initial_rec[1]:.0f})"
               if _initial_rec is not None else ""))

    log("")
    log(f"timing probe: {a.probe} iterations at batch {a.batch}, {a.draws} draws")
    t0 = time.time()
    hist, ess_hist, emin_hist, en_hist, enmax_hist, ce_hist = [], [], [], [], [], []
    qmc_se_hist, qmc_se_max_hist = [], []
    qmc_mode2_hist = []
    el_hist = []
    n_skip = n_drop = n_redo = n_qbad = n_qretry = n_gradbad = 0
    # Carry the best-so-far across a resume.  Starting it at -inf would let the first eval
    # of the continuation overwrite v3_<label>_best.pt with a WORSE model purely because it
    # is the first one this process has seen.
    e_ema = v_ema = None
    n_bang = 0
    best_vb, best_it = -1e18, -1
    best_mrr, best_mrr_it = -1e18, -1
    # A nested ablation starts a different statistical model.  Carrying the full model's
    # best score would prevent the ablation from ever writing its own best checkpoint.
    if (resume_blob is not None and resume_blob.get("best_vb") is not None
            and not a.zero_phi and not a.zero_rho_c):
        best_vb, best_it = float(resume_blob["best_vb"]), int(resume_blob["best_it"])
        log(f"  best-so-far carried over: set/basket {best_vb:.4f} at iteration {best_it}")
    # Start on the SAFE side.  lam_seen begins high so the mixture is on from iteration 1
    # and is switched off only once lambda_max has been MEASURED below the threshold.
    # Starting at 0 meant the first eval_every iterations always ran the single proposal --
    # and lambda_max reached 2.74 inside 100 iterations, which is precisely where that
    # proposal biases log Z low and drives phi upward.  By the first eval the runaway had
    # already happened: E[n] read 13.5 at 16 draws against 32.2 at 128.
    lam_seen = float("inf")
    obs_mean = float(np.mean(D["trip_nlines"][tr]))
    _fit_units = bool(a.units and not a.interaction_stage and
                      not a.composition_stage and not a.size_stage)
    lz_strikes = int(resume_blob['lz_strikes']) if resume_blob else 0
    # Iteration zero is a real candidate checkpoint.  Merely logging it while initialising
    # best_vb at -inf made a regressing probe label its first post-update weights "best".
    # That is exactly what happened in run115: -56.551 was saved as best even though the
    # untouched -56.410 model was better.  Seed best selection here and persist the actual
    # initial weights before the first optimiser step can overwrite them.
    if _initial_eval is not None and _initial_eval[0] > best_vb:
        best_vb, best_it = float(_initial_eval[0]), it0
        save_ckpt(os.path.join(OUT, f"v3_{a.label}_best.pt"), m, opt, sched,
                  it0, rng, gen, best_vb, best_it, lz_strikes, cum_iter=cum_base)
        json.dump(dict(iter=it0, set_per_basket=best_vb,
                       epoch=it0 * a.batch / max(len(tr), 1)),
                  open(os.path.join(OUT, f"v3_{a.label}_best.json"), "w"), indent=2)
        log(f"  iteration-zero checkpoint is current best: set/basket {best_vb:.4f}")
    if _initial_rec is not None and _initial_rec[0] > best_mrr:
        best_mrr, best_mrr_it = float(_initial_rec[0]), it0
        save_ckpt(os.path.join(OUT, f"v3_{a.label}_bestmrr.pt"), m, opt, sched,
                  it0, rng, gen, best_vb, best_it, lz_strikes, cum_iter=cum_base)
        json.dump(dict(iter=it0, mrr=best_mrr,
                       epoch=it0 * a.batch / max(len(tr), 1)),
                  open(os.path.join(OUT, f"v3_{a.label}_bestmrr.json"), "w"), indent=2)
        log(f"  iteration-zero checkpoint is current best MRR: {best_mrr:.4f}")
    # empirical per-product price response, standardised once with its reliability weights
    _bt = None
    if a.beta_cal_w > 0 and a.beta_target and os.path.exists(a.beta_target):
        _npz = np.load(a.beta_target)
        _tw = torch.as_tensor(_npz["weight"], dtype=torch.float64)
        _tt = torch.as_tensor(_npz["target"], dtype=torch.float64)
        _sw = _tw.sum().clamp_min(1e-9)
        _tm = (_tw * _tt).sum() / _sw
        _ts = (((_tw * (_tt - _tm) ** 2).sum() / _sw).clamp_min(1e-12)).sqrt()
        _bt = dict(weight=_tw, z=(_tt - _tm) / _ts)
        log(f"  beta calibration: {int((_tw > 0).sum()):,} products with a measured price "
            f"response, weight {a.beta_cal_w}")
    _pn_cache = [None]
    for it in range(it0 + 1, a.iters + 1):
        if a.qmc_n > 0 and a.qmc_refresh_every > 0:
            refresh_block = (it - it0 - 1) // int(a.qmc_refresh_every)
            _set_qmc_rule(a.qmc_n, a.qmc_seed + 1_000_003 * refresh_block)
        sub = tr[rng.choice(len(tr), size=a.batch, replace=False)]
        ix, ctx, lctx, hh, li, lt, lc, lq = B.make(sub)
        m.house, m.ctx = hh, ctx
        # Switch the proposal on the LAST measured lambda_max.  Below the threshold the
        # single Gaussian is exact at 8 draws; above it, it biases log Z low and the fit
        # chases phi to its cap.  The mixture needs roughly double the draws to match at the
        # low end, so the count rises with it.
        # Always on.  Switching on a measured lambda_max cannot work: it triples between
        # evals (0.575 at iteration 50, 1.869 at 100), so any threshold rule acts on stale
        # information and turns the mixture off exactly when it starts being needed.  The
        # asymmetry settles it -- the mixture costs +0.0056 nats of bias where it is
        # unnecessary against the single proposal's +0.0003, five thousandths of a nat,
        # while being wrong the other way ends the run.
        # Anisotropic proposal: widen along Lambda's top eigenvector only.  The isotropic
        # mixture was measured to DIVERGE on real data (12.03 -> 12.30 -> 12.47 with sd
        # growing 0.015 -> 0.47) because widening all Kz = 12 dimensions costs 2^12 in
        # volume; along one direction it costs 2, and the same checkpoint converges to
        # 11.972 +/- 0.0006.  It needs no extra draws.
        _mix = None
        _nd = a.draws
        if a.cd:
            # Contrastive divergence with an EXACT sampler.
            #
            # d log P(S)/d theta = dE(S)/d theta - E_model[dE/d theta], and the second term
            # needs samples from the model, not an estimate of log Z.  Corollary 3's sampler
            # provides them -- validated at 1.011 on size and 0.98 on inclusion probabilities
            # -- so no persistence is needed and the gradient is unbiased.  This avoids the
            # failure that ended runs 41-53: importance sampling biases log Z low and the
            # bias becomes a gradient pushing phi to its cap, while pseudo-likelihood avoids
            # Z but cannot pin the level.
            e_data = m.energy(li, lt, lc, ix.B, lctx)
            with torch.no_grad():
                # The sampler's stage 1 picks z by importance resampling, so CD inherits a
                # weakened form of the same z-integral dependence that biased log Z.  If the
                # climb in ||phi|| is that residual bias, more draws HERE should slow it; if
                # it is not, the pressure is in the objective and no estimator removes it.
                s_slot, s_trip = m.sample_slots(ix, n_draws=a.cd_draws or a.draws,
                                                generator=gen, mode_steps=a.mode_steps)
            if s_slot.numel() > 0:
                s_ctx = {k: (v[s_slot] if torch.is_tensor(v) and v.shape[0] == ix.item.shape[0]
                             else v) for k, v in ctx.items()}
                e_model = m.energy(ix.item[s_slot], s_trip,
                                   ix.row_cat[ix.row_of[s_slot]], ix.B, s_ctx)
            else:
                e_model = torch.zeros_like(e_data)
            ll = e_data - e_model
            with torch.no_grad():
                _, ess, pn = m.loglik(ix, li, lt, lc, n_draws=_nd, generator=gen,
                                      return_ess=True, return_size=True, line_ctx=lctx,
                                      mode_steps=a.mode_steps, mix_scales=_mix,
                                      aniso=a.aniso, units=None)
            if _fit_units:
                ll = ll + m.units_loglik(li, lt, lq, lctx, ix.B)
        elif a.objective == "composite":
            # COMPOSITE LIKELIHOOD (Besag 1975; Lindsay 1988): a consistent estimator of the
            # SAME version-4 law that needs no normaliser.  The objective is the model's own
            # leave-one-out conditional -- the quantity MRR ranks on -- verified to 3.6e-15
            # against an independent implementation of the scoring formula.
            #
            # Measured: 0.063 s/step against 2.292 for the full set likelihood, 36.6x.  What
            # it cannot identify is rho_0: the conditional holds |S| fixed, so the size law
            # gets EXACTLY zero gradient (verified, 0.000e+00).  That is why rho_0 is frozen
            # at its empirical initialisation here -- measured to cost 0.008 nats, since the
            # size component moved only 0.008 over 2,400 full-objective updates while
            # composition moved ~1.5.
            #
            # The reported metric is unchanged: evaluation still uses the exact normaliser.
            # PER TRIP, at basket scale: sum_{j in S} log P(j | S\{j}).  Returning a
            # line-averaged scalar instead put ll ~8 where the set likelihood is ~56, and
            # every penalty in this file is tuned to the latter -- the fit diverged to -3e7.
            ll = conditional_composition_ce(m, ix, li, lt, m.b_flat(ix), per_trip_out=True)
            ess = torch.ones(ix.B, dtype=ll.dtype, device=ll.device)
            # pn (the size law) needs the normaliser, which this objective avoids.  Rather
            # than special-case a dozen downstream consumers, refresh it on the amortised
            # schedule -- WITH gradient, so the elasticity constraint still trains gamma and
            # beta -- and reuse the cached copy in between.  At elast_every=20 that is
            # 2.29/20 = 0.11 s/step on top of 0.063.
            if (it % max(a.elast_every, 1)) == 0 or _pn_cache[0] is None:
                _, _e2, pn = m.loglik(ix, li, lt, lc, n_draws=_nd, generator=gen,
                                      return_ess=True, return_size=True, line_ctx=lctx,
                                      mode_steps=a.mode_steps, mix_scales=_mix,
                                      aniso=a.aniso, units=None)
                _pn_cache[0] = pn.detach()
            else:
                pn = _pn_cache[0][: ix.B] if _pn_cache[0].shape[0] >= ix.B else _pn_cache[0]
                if pn.shape[0] != ix.B:
                    pn = pn[:1].expand(ix.B, -1)
            if _fit_units:
                ll = ll + m.units_loglik(li, lt, lq, lctx, ix.B)
        elif a.pseudo:
            # Fit by pseudo-likelihood: no Z, no draws, no proposal.  ESS and pn are still
            # wanted for the goals line, so the normaliser is evaluated WITHOUT gradients.
            ll = m.pseudo_loglik(ix, li, lt, lc, ix.B, line_ctx=lctx,
                                 neg_per_trip=a.neg_per_trip, generator=gen)
            with torch.no_grad():
                _, ess, pn = m.loglik(ix, li, lt, lc, n_draws=_nd, generator=gen,
                                      return_ess=True, return_size=True, line_ctx=lctx,
                                      mode_steps=a.mode_steps, mix_scales=_mix,
                                      aniso=a.aniso, units=None)
            if _fit_units:
                ll = ll + m.units_loglik(li, lt, lq, lctx, ix.B)
        else:
            ll, ess, pn = m.loglik(ix, li, lt, lc, n_draws=_nd, generator=gen,
                           return_ess=True, return_size=True, line_ctx=lctx,
                           mode_steps=a.mode_steps, mix_scales=_mix, aniso=a.aniso,
                                 antithetic=a.antithetic > 0,
                           units=lq if _fit_units else None)

        # Deterministic hard-batch refinement.  A rare broad-shell trip can have a much
        # larger finite-rule error than the other 23 trips.  The old response was to throw
        # away the entire update after paying for it; run110's first probe discarded 8% of
        # batches that way.  Re-evaluate only when the observable per-trip replicate SE
        # crosses the guard, using an independent fixed rule with more nodes.  This remains
        # common-random-number deterministic across optimiser steps and, because it happens
        # before ``loss`` is assembled, every size/elasticity penalty uses the refined pn.
        _first_qs = getattr(m, "_last_qmc_logz_se", None)
        _retry = (not a.cd and not a.pseudo and m.quad_a is not None
                  and a.qmc_retry_n > int(getattr(m, "_qmc_n", 0))
                  and a.qmc_step_se > 0 and _first_qs is not None
                  and bool(torch.isfinite(_first_qs).all())
                  and float(_first_qs.max()) > a.qmc_step_se)
        if _retry:
            bad_q = (_first_qs > a.qmc_step_se).nonzero().flatten()
            old_rule = m.quad_a
            old_mix_rule = getattr(m, "quad_mix_a", None)
            retry_seed = int(getattr(m, "_qmc_seed", 0)) + 1_000_003
            first_mode_count = getattr(m, "_last_qmc_mode_count", None)
            try:
                m.quad_a = sobol_grid(
                    m.Kz, a.qmc_retry_n, seed=retry_seed,
                    replicates=m.quad_replicates)
                if getattr(m, "quad_size_bands", 0):
                    m.quad_mix_a = sobol_mixture_grid(
                        m.Kz, 2 * a.qmc_retry_n, seed=retry_seed,
                        replicates=m.quad_replicates, components=2)
                # Build only the flagged trips.  Re-running all 24 made a two-trip retry
                # cost more than the base update and defeated the point of adaptation.
                hard_sub = sub[bad_q.detach().cpu().numpy()]
                hix, hctx, hlctx, hhh, hli, hlt, hlc, hlq = B.make(hard_sub)
                m.house, m.ctx = hhh, hctx
                hll, hess, hpn = m.loglik(
                    hix, hli, hlt, hlc, n_draws=_nd, generator=gen,
                    return_ess=True, return_size=True, line_ctx=hlctx,
                    mode_steps=a.mode_steps, mix_scales=_mix, aniso=a.aniso,
                    antithetic=a.antithetic > 0,
                    units=hlq if _fit_units else None)
                # index_copy keeps the accepted rows' original graph and routes the hard
                # rows' gradients through the refined graph.
                ll = ll.index_copy(0, bad_q, hll)
                ess = ess.index_copy(0, bad_q, hess)
                pn = pn.index_copy(0, bad_q, hpn)
                retry_qs = m._last_qmc_logz_se
                m._last_qmc_logz_se = _first_qs.index_copy(0, bad_q, retry_qs)
                if first_mode_count is not None and m._last_qmc_mode_count is not None:
                    m._last_qmc_mode_count = first_mode_count.index_copy(
                        0, bad_q, m._last_qmc_mode_count)
                n_qretry += int(bad_q.numel())
            finally:
                m.quad_a = old_rule
                m.quad_mix_a = old_mix_rule
                m.house, m.ctx = hh, ctx
        _objective_ll = (observed_composition_loglik(ll, pn, lt)
                         if a.composition_stage else ll)
        loss = -_objective_ll.mean()
        if a.price_soft and a.price_hinge_w > 0 and getattr(m, "_last_gb", None) is not None:
            # non-positive own-price response as a PENALTY, not a reparameterisation:
            # b_j -= (gamma.beta) * dlogp, so the response is -(gamma.beta) and we penalise
            # gamma.beta < 0.  Unlike softplus this has a live gradient everywhere.
            loss = loss + a.price_hinge_w * torch.relu(-m._last_gb).pow(2).mean()
        # Keep the unmasked data term so the ESS gate can swap it out WITHOUT discarding the
        # penalties added below.  The gate used to rebuild loss from scratch as
        # -ll[keep].mean(), which silently dropped every one of them: --pool-ctx,
        # --pool-beta, --elast-w and --pool-prod never reached backward().  run75 came out
        # BIT-IDENTICAL to run74 in every logged column, which is how this surfaced.  Only
        # size_kl survived, because the gate re-added it by hand; it is added once here and
        # must NOT be re-added there.
        _data_term = loss
        # AFTER _data_term, not before.  _data_term is what the ESS gate subtracts and
        # replaces with the masked likelihood; anything folded into it is silently removed
        # from backward().  Adding this above that line made armF come out BIT-IDENTICAL to
        # armD -- the exact signature the comment below records for run75 vs run74.
        if a.comp_ce_w > 0:
            loss = loss + a.comp_ce_w * conditional_composition_ce(
                m, ix, li, lt, m.b_flat(ix))
        # Size-law calibration penalty.
        #
        # The measured failure is a chain with one root: the fitted size law puts too much
        # mass on large baskets (P(1) 0.143 against an empirical 0.196), E[n] then runs away
        # on a minority of trips (94.5 expected items where 16 were bought), and since
        # lambda_max <~ ||phi||^2 E[n] those trips break section 14's stability condition.
        # Above lambda_max = 1 the map z <- grad log f is no longer a contraction, so the
        # mode iteration in log_Z diverges, the proposal is centred wrong, and ESS collapses
        # to 0.08 -- 1.3 useful draws out of 16.  E[n] and log Z instability correlate at
        # +0.919 across trips.
        #
        # Capping ||phi|| cannot fix this: the bound is a sum over thousands of products, so
        # the cap constrains each term and not the total.  Damping the mode iteration cannot
        # either -- its Jacobian eigenvalues are 1 - t + t*lambda, above 1 for every t > 0
        # once lambda > 1.  What does work is holding E[n] down, and the size law is the
        # thing that sets it.  The cross-entropy to the empirical law is the direct lever,
        # and pn rides along on the normaliser's own draws, so it is free.
        # Match the size law's SPREAD, not only its shape.
        #
        # The cross-entropy alone matches the marginal size law on the training batch, and
        # that is what it did -- while the held-out environment generated 11.41 items per
        # basket against 6.31 observed.  Two separate faults hid behind one number.  The
        # marginal can look right while per-trip E[n] is far too high, because the batch
        # average washes out the tail; and dE[n]/de = Var(n) (Proposition 1) means an
        # over-dispersed size law is also an over-elastic one -- at Var(n) = 96.5 a 10%
        # price cut tripled basket contents.  So Var(n) is pinned to the empirical variance
        # directly, on the same free pn.
        ce = None
        if a.size_kl > 0 and pn is not None and not a.composition_stage \
                and not bool(m.factored_size_enabled):
            if pn is None: pn = None
            pbar = pn.mean(0).clamp_min(1e-12)
            pbar = pbar / pbar.sum()
            ce = -(emp_pn[: pbar.shape[0]] * pbar.log()).sum()
            nax = torch.arange(1, pn.shape[1] + 1, dtype=pn.dtype)
            e_tr = (pn * nax).sum(1)
            v_tr = (pn * nax ** 2).sum(1) - e_tr ** 2
            ce = ce + a.var_w * (torch.log1p(v_tr.mean()) - math.log1p(emp_var)) ** 2
            # PER-TRIP size calibration.  Everything above matches an AVERAGE -- the
            # cross-entropy and the reverse KL both act on pbar = pn.mean(0) -- and the
            # failure is not in the average.  Measured on run80 at iter 1200, 480 validation
            # trips: the 117 trips whose normaliser gap exceeds 1 nat have model E[n] 36.4
            # against 14.6 for the other 363, a 2.49x separation, while every other per-trip
            # quantity is flat (logsumexp b 1.01x, assortment 1.01x, categories 1.04x, and
            # phi mass identical to three decimals -- the interaction plays no part).  So a
            # minority of trips with runaway E[n] destroys the estimator, and the batch mean
            # cannot see them: it is the fifth time in this file a mean has stood in for a
            # tail.  e_tr is the per-trip E[n], already computed on the line above and until
            # now used only through v_tr.mean().
            if a.en_w > 0:
                _obs = torch.zeros_like(e_tr).index_add_(
                    0, lt, torch.ones(lt.shape[0], dtype=e_tr.dtype))
                ce = ce + a.en_w * ((e_tr - _obs) ** 2).mean()
            # REVERSE KL, because the cross-entropy above is structurally blind to the
            # failure it was written to prevent.
            #
            # -sum_n emp(n) log p(n) weights every size by its EMPIRICAL frequency, so it
            # punishes the model for MISSING data mass and never for inventing mass where
            # the data has none.  Measured on run68 (train, 240 trips), the whole E[n]
            # miscalibration is one fat tail that cross-entropy cannot see:
            #
            #     n        model P   emp P   model n*P   emp n*P    excess
            #     1-20      0.7624  0.9000       5.399     5.136    +0.263
            #     21-50     0.0935  0.0917       2.510     2.754    -0.244
            #     51-120    0.0442  0.0083       5.194     0.525    +4.669   <- 99.6% of it
            #
            # That bucket carries 0.83% of the empirical mass, hence 0.83% of the CE weight,
            # while contributing 40% of model E[n].  So E[n] ran 1.56x in-sample on the best
            # checkpoint in the project and no amount of tuning size_kl could touch it --
            # the penalty has no gradient there to tune.  It also feeds the aborts: the tail
            # inflates Z, but log Z is estimated from 16 draws and biased DOWNWARD worst
            # exactly where the tail is heavy, so the tail grows nearly free, Var(n) rises,
            # ESS collapses, and the normaliser check fires.
            #
            # KL(model || emp) = sum_n p(n) log(p(n)/emp(n)) is mode-SEEKING: its gradient
            # is log(p/emp) + 1, which is large precisely where the model has mass the data
            # does not.  emp is smoothed because it is exactly zero at some sizes.
            if a.rkl_w > 0:
                _ep = emp_pn[: pbar.shape[0]] + a.rkl_eps
                _ep = _ep / _ep.sum()
                ce = ce + a.rkl_w * (pbar * (pbar.log() - _ep.log())).sum()
            loss = loss + a.size_kl * ce
        # Match the aggregate price elasticity the DATA shows.
        #
        # Regressing log(basket size) on the mean price deviation faced over 40,000
        # training trips gives d log n / d log p = -0.121: a 10% cut grows the basket about
        # 1.3%.  The fitted model gave -11.5 -- a 10% cut tripled basket contents, ~95x too
        # elastic.  Nothing in the likelihood objects, because prices move little enough
        # that the price term barely enters log P(basket); the elasticity was free to drift
        # and did.  Constraining the SIGN through softplus fixed the direction and let the
        # magnitude grow 24x (median gamma.beta 0.074 -> 1.81) -- I checked the sign
        # afterwards and never checked the size.
        #
        # Proposition 1 makes the quantity available in closed form: for a uniform shift,
        # dE[n]/d log p = -(gamma.beta) Var(n), so the elasticity is that over E[n].  Both
        # moments come from pn, which the normaliser already returns.
        # MEASURE always, PENALISE only when asked.  Both used to sit behind elast_w > 0, so
        # --elast-w 0 logged 'elast +nan' and blinded one of the three things this model
        # exists to do.  The quantity is closed form from pn, which the normaliser already
        # returned -- a few tensor ops, no extra pass -- and when elast_w is 0 it never
        # enters loss, so gradients are bit-identical either way.
        # Under the composite objective pn is unavailable (no normaliser was computed).
        # The elasticity constraint still matters -- it is what makes the price
        # counterfactual meaningful -- so it is AMORTISED: every --elast-every steps one
        # exact normaliser pass supplies pn and the penalty is applied then.  At
        # elast_every=20 that is 2.29/20 = 0.11 s/step on top of 0.063, still ~12x faster
        # than the full objective.
        if pn is None:
            elast = float("nan")
        else:
            gb = (m.price_g()[hh][ix.item_trip]
                  * m.price_b()[ix.item]).sum(-1).mean()
            nax = torch.arange(1, pn.shape[1] + 1, dtype=pn.dtype)
            e_b = (pn * nax).sum(1)
            v_b = (pn * nax ** 2).sum(1) - e_b ** 2
            elast = (torch.zeros((), dtype=gb.dtype, device=gb.device)
                     if bool(m.factored_size_enabled)
                     else -(gb * v_b.mean() / e_b.mean().clamp_min(1e-6)))
            if (a.elast_w > 0 and not a.composition_stage
                    and not bool(m.factored_size_enabled)):
                loss = loss + a.elast_w * (elast - a.elast_target) ** 2
        # PARTIAL POOLING on the per-product price coefficient.
        #
        # The own-price elasticity is about -g_j (1 - pi_j) with g_j the product's price
        # coefficient.  Fitted freely from limited price variation, g_j is noisy, and noise
        # inflates its SPREAD without improving its ranking.  Measured on the fair arena, the
        # unpooled models over-disperse elasticities ~2.4x against the truth (sd 0.37-0.51 vs
        # 0.190) while still ranking products correctly (r = 0.890) -- and scored WORSE than a
        # constant predictor (MAE 0.141 against 0.133).  Shrinking toward the mean fixed it:
        #     arena, cap 0.96:  MAE 0.1410 -> 0.0989
        #     arena, cap 2.5 :  MAE 0.2326 -> 0.1001
        # at a cost of 0.003 nats of likelihood and no change to any pair lift.
        #
        # Toward the MEAN, not toward zero: the average price sensitivity is identified by
        # the elasticity target and must not be shrunk, only its dispersion across products.
        if a.pool_beta > 0:
            _g = m.price_b()
            loss = loss + a.pool_beta * ((_g - _g.mean(0, keepdim=True)) ** 2).mean()
        # PARTIAL POOLING on the contextual item embeddings, for the same reason.
        #
        # In the arena, where season/store/recency effects are REAL and learnable, these terms
        # HELP: MRR 0.4940 -> 0.5003, 98.9% of the achievable ceiling, and train likelihood
        # -5.99 -> -5.82.  So they are not intrinsically harmful.  The real-data failure is
        # identification: mu is 5455xR here against 16x3 in the arena -- about a thousand
        # times more contextual parameters for six times the data -- so the residuals fit
        # noise, which costs 34x in ranking MRR while the likelihood barely notices.
        #
        # Toward the MEAN, not toward zero: the mean carries basket size (centring mu
        # collapsed E[n] to 1.00) and only the dispersion across products is noise.  This is
        # the same treatment that took beta's price MAE from 0.141 to 0.099 for 0.003 nats.
        if a.pool_ctx > 0:
            for _p in (m.mu, m.zeta, m.psi):
                loss = loss + a.pool_ctx * ((_p - _p.mean(0, keepdim=True)) ** 2).mean()
        # CALIBRATE beta's CROSS-PRODUCT PATTERN against the measured price response.
        #
        # price_kappa gives the model capacity for a per-product price response, but
        # capacity is not information: after kappa grew 1.0 -> 8.3 the correlation between
        # softplus(beta_j) and the empirical per-product response was still ~0 (-0.019 ->
        # -0.051), so kappa was amplifying a coefficient carrying no signal.  That lands
        # squarely on the WITHIN-TRIP relative b that ranking reads, and MRR fell 3x
        # (0.0267 -> 0.0084) while the likelihood improved.
        #
        # The pull is on STANDARDISED values, so it constrains the ordering and spread
        # across products and leaves the LEVEL to --elast-w.  The two are then orthogonal:
        # elast_w fixes how strongly price acts overall, this fixes which products it acts
        # on.  Reliability weights are min(purchases, 5000); products without an estimate
        # get zero weight.  The empirical slopes are confounded by promotions and
        # seasonality, so this is a weak pull toward a pattern, not a fit to a truth.
        if a.beta_cal_w > 0 and _bt is not None:
            _bj = (m.price_g().mean(0) * m.price_b()).sum(-1)      # [J]
            _cw = _bt["weight"]
            _cs = _cw.sum().clamp_min(1e-9)
            _bm = (_cw * _bj).sum() / _cs
            _bsd = (((_cw * (_bj - _bm) ** 2).sum() / _cs).clamp_min(1e-12)).sqrt()
            _bz = (_bj - _bm) / _bsd
            loss = loss + a.beta_cal_w * ((_cw * (_bz - _bt["z"]) ** 2).sum() / _cs)
        # RIDGE ON THE PRODUCT, not on the factors.  This is the degree fix.
        #
        # A squared penalty on each factor of a bilinear term becomes a NUCLEAR-norm penalty
        # on their product, which is degree 1, while lam pays a ridge, degree 2.  Verified
        # numerically by minimising over the free scale with no algebra assumed: bilinear
        # cost/||v|| is flat for ||v|| in 50..400 while lam cost/||v||^2 is flat at 5e-6.
        # The coefficient depends on which penalties are actually live: 2.204e-4 (crossover
        # ||v|| = 44) if pool_ctx is active, 7.28e-5 = wd*sqrt(W) (crossover ||v|| = 14.6)
        # with weight decay alone, which is the regime that really ran, since the ESS gate
        # discarded pool_ctx.  Measured intercept is ||c|| = 244, far above either.  So the
        # bilinear
        # route is structurally cheaper and wins every time, which is why raising pool_ctx
        # never worked -- it moves the coefficient, not the exponent, and the optimiser
        # evades it anyway by shrinking mu and growing delta.
        #
        # Gauge-fixing delta killed only the CONSTANT channel mu'delta_bar.  Ranking happens
        # WITHIN a trip, where the week is fixed, so mu_j'delta_w is still a per-product
        # offset that acts exactly like an intercept even when it averages to zero over
        # weeks -- and it still had the cheap linear route.  run74: season 0.042 -> 1.407
        # while MRR fell 0.0780 -> 0.0270.
        #
        # ||P Q'||_F^2 = tr((P'P)(Q'Q)) -- a K x K trace, so the J x W (or J x N) offset
        # matrix is never formed.  Divided by its entry count this is a ridge of strength
        # pool_prod per realised offset entry; pool_prod = (wd/2)*J*W = 1.45 makes it exactly
        # the ridge lam already pays.  Per unit of PER-PRODUCT offset that is W times dearer
        # than lam, which is correct: an effect that varies by week spends 53 numbers to say
        # what lam says with one, and should need correspondingly more evidence.
        #
        # All THREE bilinear pairs, not just the seasonal one.  Penalising mu.delta alone
        # would leave theta.alpha as an untaxed escape route and the mass would simply move
        # there -- the same way it moved out of the constant channel once that was closed.
        if a.pool_prod > 0:
            for _P, _Q in ((m.mu, m.delta_c()), (m.zeta, m.xi_c()), (m.alpha, m.theta_c())):
                loss = loss + a.pool_prod * torch.trace(
                    (_P.T @ _P) @ (_Q.T @ _Q)) / (_P.shape[0] * _Q.shape[0])
        # ESS GATE.  log Z is estimated by importance sampling; where the sampler has
        # collapsed the estimate is unreliable and biased DOWNWARD, which the objective
        # rewards (section 17).  The diverged run had ESS 0.016 on exactly the trips whose
        # energy had run away.  A batch below the floor carries no usable gradient, so it
        # is skipped rather than followed.
        # Gate on the WORST trip, not the batch mean.  Measured on a trained checkpoint:
        # per-trip log Z standard deviation over independent seeds ran to 2.28 nats on one
        # trip against a median of 0.041 -- a 56x spread -- and that trip reported ESS
        # 0.891, comfortably above any floor.  A batch mean of 0.95 hides it completely,
        # so a few trips were contributing most of the gradient noise unnoticed.
        # Drop the offending TRIPS, do not throw away the batch.
        #
        # Gating the whole batch on its worst trip was the wrong remedy for the right
        # diagnosis.  A batch of 24 nearly always contains one trip whose importance
        # weights have collapsed, so the gate fired on 250 of the first 400 batches -- 62%
        # of the compute discarded, and 23 healthy trips thrown out for every bad one.
        # Masking keeps their gradient and removes only the noise.  The batch is still
        # skipped outright if too little of it survives to be worth a step.
        # Re-estimate the hard trips instead of discarding them.
        #
        # lambda_max < 1 was enforced for five runs to protect the estimator, and it cost
        # the interaction the model exists for.  The premise does not survive measurement:
        # at lambda_max ~ 2.5 plain importance sampling converged to log Z 10.0075 across
        # 8 -> 2048 draws with ESS 0.998.  It is fine there on ordinary trips; roughly 7%
        # break, and they are the same tail found this morning.  Crippling phi for every
        # trip to protect a minority is the wrong trade -- give that minority more draws.
        e_bar, e_min = float(ess.mean().detach()), float(ess.min().detach())
        # A finite deterministic QMC rule can fail coherently: all nodes can agree inside
        # the local basin while missing a remote large-basket mode.  ESS alone did not stop
        # run100's iteration 4,200 -> 4,400 collapse; E[n] jumped 7.4 -> 90.1 and the mean
        # replicate SE jumped to 0.71, yet several bad updates were applied before the
        # 200-iteration checkpoint guard ran.  Refuse such a batch before either Adam or
        # the rho_0 feedback projections can mutate the model.  The limits are deliberately
        # loose relative to a healthy repaired run (max SE about 0.007, E[n] about 7.8).
        _qmc_bad = False
        if m.quad_a is not None:
            _qs = getattr(m, "_last_qmc_logz_se", None)
            if a.qmc_step_se > 0:
                _qmc_bad = (_qs is None or not bool(torch.isfinite(_qs).all())
                            or float(_qs.max()) > a.qmc_step_se)
            if a.qmc_en_max > 0 and pn is not None:
                _nn = torch.arange(1, pn.shape[1] + 1, dtype=pn.dtype,
                                   device=pn.device)
                _en_step = float((pn.detach() * _nn).sum(1).mean())
                _qmc_bad = (_qmc_bad or not math.isfinite(_en_step)
                            or _en_step > a.qmc_en_max * obs_mean)
            if _qmc_bad:
                n_qbad += 1
        if a.adapt_draws > 1:
            with torch.no_grad():
                bad = (ess < a.ess_floor_min).nonzero().flatten()
            if bad.numel() > 0:
                ll2, ess2 = m.loglik(ix, li, lt, lc, n_draws=a.draws * a.adapt_draws,
                                     generator=gen, return_ess=True, line_ctx=lctx,
                                     mode_steps=a.mode_steps, mix_scales=_mix, aniso=a.aniso,
                                 antithetic=a.antithetic > 0,
                                     units=lq if _fit_units else None)
                ll = torch.where((ess < a.ess_floor_min), ll2, ll)
                ess = torch.where((ess < a.ess_floor_min), ess2, ess)
                n_redo += int(bad.numel())
        keep = ess >= a.ess_floor_min
        n_drop += int((~keep).sum())
        if (_qmc_bad or int(keep.sum()) < max(2, int(a.min_keep * a.batch))
                or e_bar < a.ess_floor):
            n_skip += 1
            opt.zero_grad()
        else:
            # Swap in the ESS-masked data term; every penalty above is preserved.
            loss = loss - _data_term + (-_objective_ll[keep].mean())
            opt.zero_grad()
            loss.backward()
            # clip_grad_norm_ defaults to error_if_nonfinite=False.  One NaN gradient then
            # makes the total norm NaN and scales every otherwise healthy gradient by NaN;
            # Adam consequently destroys the entire checkpoint before the post-step
            # projections can diagnose the originating block.
            _bad_grad = [name for name, value in m.named_parameters()
                         if value.grad is not None
                         and not bool(torch.isfinite(value.grad).all())]
            if _bad_grad:
                # The positive linear-space ESP forward can be finite while the derivative
                # of log(A_n) reaches 1/A_n ~ 1e308 for a numerically irrelevant degree.
                # Its exact final score is finite, but an intermediate 0*inf in the raw
                # polynomial adjoint is not representable.  Never apply a clipped or
                # partially finite gradient: reject this minibatch and advance to the next
                # independently scrambled rule, just as the estimator-SE gate does.
                n_skip += 1
                n_qbad += 1
                n_gradbad += 1
                if n_gradbad <= 10 or n_gradbad % 100 == 0:
                    log(f"  skipped non-finite polynomial gradient at iteration {it} "
                        f"({', '.join(_bad_grad)}); finite loss {float(loss.detach()):.4f}")
                opt.zero_grad(set_to_none=True)
                if sched is not None:
                    sched.step()
                continue
            torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip,
                                           error_if_nonfinite=True)
            # Adam is deliberately scale-invariant to multiplying a parameter's gradient,
            # so gradient scaling cannot slow a sensitive block.  Interpolate the actual
            # Adam update instead, retaining its mature first/second moments.  This is
            # needed for rho_c: a change of only -0.0208 crossed a quadratic category-count
            # phase boundary and moved one validation trip from E[n]=23 to 104 in 200
            # updates.  A separate optimiser param group would make the saved one-group
            # Adam state unrestorable; post-step interpolation changes only the step size.
            _phi_before = (m.phi.detach().clone()
                           if a.phi_step_scale < 1.0 else None)
            _rhoc_before = (m.rho_c.detach().clone()
                            if a.rho_c_step_scale < 1.0 else None)
            opt.step()
            # Keep the CONSTRAINED price block out of softplus saturation.  gamma ratchets
            # ever more negative because softplus'(x) = sigma(x) is the very factor that
            # would push it back -- 0.047 at x = -3 but 0.0028 at -5.9.  Measured, 38,500
            # iterations of stage 1 reached gamma mean -4.72, min -5.88 (effective
            # coefficients 0.0097 and 0.0028); unconstraining THAT in stage 2 removes the
            # damping, |g_gamma| jumps to 1.68, and Adam steps of 2.5e-5 cross zero within
            # 500 iterations -- the price sign flips and E[n] runs to n_max.  Applied here,
            # after EVERY optimiser step, because it is a projection: the phi/rho_c site
            # below is gated and does not run on every iteration.  Meaningless under
            # --price-soft, where gamma IS the coefficient rather than its pre-image.
            if a.gamma_floor is not None and not getattr(m, "price_soft", False):
                with torch.no_grad():
                    m.gamma.clamp_(min=a.gamma_floor)
                    m.beta.clamp_(min=a.gamma_floor)
            with torch.no_grad():
                if _phi_before is not None:
                    m.phi.copy_(_phi_before + a.phi_step_scale * (m.phi - _phi_before))
                if _rhoc_before is not None:
                    m.rho_c.copy_(
                        _rhoc_before + a.rho_c_step_scale * (m.rho_c - _rhoc_before))
            # --- rho_0: fix the gauge, then floor the curvature ------------------------
            #
            # Adding Delta to every b_j multiplies each n-subset by e^{n Delta}, which is
            # IDENTICAL to rho_0(n) -> rho_0(n) - n Delta.  So rho_0's linear component and
            # lam's mean are one flat direction, and rho_0 duly collapsed onto it: measured
            # first differences run 7.59 -> 4.22 on run80 and 23.73 -> 22.44 on run68, i.e.
            # very nearly a straight line.  Centring lam and passing the level to rho_0 is
            # the exact reparameterisation (b unchanged) that pins it.
            if a.lam_centre and not a.composition_stage and not a.size_stage:
                with torch.no_grad():
                    _mu = m.lam.mean().clone()
                    m.lam.sub_(_mu)
                    _n = torch.arange(1, m.rho_0_free.shape[0] + 1,
                                      dtype=m.rho_0_free.dtype, device=m.rho_0_free.device)
                    m.rho_0_free.sub_(_mu * _n)
            # A straight rho_0 has NO curvature, and curvature is the only thing bounding
            # basket size: Var(n|trip) ~ 1/rho_0''.  Measured curvature was NEGATIVE across
            # the whole range where baskets live (n=1..40; median -0.034 on run80, -0.014 on
            # run68), turning positive only past n=50 where there is no data.  The term whose
            # job is to confine size was providing anti-confinement.
            #
            # That is what makes the model unstable, via Proposition 1: dE[n]/dDelta =
            # Var(n).  b is only 0.185 higher on held-out trips, but Var(n) ~ 86 amplifies
            # that into +15.95 items (predicted 15.97, observed 15.95 -- exact), so held-out
            # E[n] hits 18.6 against 7.4 observed and the normaliser becomes unestimable.
            # Flooring rho_0'' >= c makes Var <= 1/c a guarantee: c = 0.09 caps the
            # amplifier at 11, so the model's own generalisation error can move E[n] by at
            # most ~2 items.
            # Keep phi inside the region where log Z is computable at all.
            #
            # c = max_u sum_j max(phi_j'u, 0) sets where the latent target keeps its mass
            # (||z|| ~ c) and hence log Z ~ c^2/2.  The proposal is a mode-centred Gaussian
            # with sd capped at 4.47 and the prior's typical radius is sqrt(Kz) = 5.66, so
            # at the measured c = 74 (run84) the sampler was drawing at radius ~5 while the
            # mass sat beyond 100.  log Z was really ~2.7e3, not the ~10.7 reported.
            # This supersedes --lam-project (lambda_max was computed from pi at the MODE,
            # where E[n|zhat] = 2.8 against a marginal 17.7, so it never bound) and
            # --gap-project (phi is uncorrelated with the gap: +0.009, identical phi mass in
            # blown-up and healthy trips).
            # SUPERSEDED by --phi-norm-max.  c is a worst case over directions the mass
            # never visits: it is a max over u of a sum over ALL products, so it grows with
            # the catalogue (measured 233.3 at 5,455 products) even though the integrand
            # sits at ||z*|| = 0.07.  Rescaling all of phi by c_max/c therefore crushes the
            # complementarity to nothing for a reason that does not exist.  Kept only so
            # old commands reproduce; --phi-norm-max is the constraint that binds.
            if a.c_max != 0 and not a.size_stage:
                _cm = a.c_max if a.c_max > 0 else math.sqrt(m.Kz)
                _c = m.phi_radius()
                if _c > _cm:
                    with torch.no_grad():
                        m.phi.mul_(_cm / _c)
            # The bound that REPLACES c is --phi-max, which m.project already applies per
            # product every step -- no new knob, and two knobs for one constraint is how
            # this codebase has been bitten before.
            #
            # The mode of log f(z) - ||z||^2/2 solves the mean-field equation z* = Phi'pi(z*),
            # so ||z*|| = ||sum_j pi_j phi_j|| <= (max_j ||phi_j||) * sum_j pi_j = phi_max E[n].
            # sum_j pi_j = E[n] is pinned near 8 by the size law whatever the catalogue size,
            # so this bound does NOT scale with the number of products -- and spreading the
            # same E[n] over more products shrinks ||z*|| further (measured ||z*|| = 1.91 at
            # J=20 down to 0.07 at all 5,455).  What breaks the integral is multimodality,
            # driven by phi_max alone: against exact enumeration with 12 products sharing one
            # phi direction so nothing cancels, phi_max=0.96 is exact to 0.0001 nats and 1.40
            # to 0.0031, failing only past 2.0 (+0.064).  Grocery needs 0.96.
            # Hold lam's spread below the measured cliff.
            #
            # lam sd has risen monotonically in EVERY run regardless of configuration
            # (0.183->0.612 run74, 0.092->0.491 run81, 0.111->0.590 run83), and the
            # normaliser gap is flat until lam sd ~0.6 and then explodes: measured on a
            # trained checkpoint, scaling lam's spread gave gap 0.006 at sd 0.43, 0.022 at
            # 0.64, 0.673 at 0.86 and 67.1 at 1.29.  run83 crossed sd 0.590 and the gap
            # jumped +0.818 -> +2.373 in one eval, which is why every run dies near
            # iteration 1400-1600.  The incentive is in the likelihood itself -- importance
            # sampling always UNDERSTATES log Z (Jensen), ll = E - log Z, and the
            # understatement grows with the spread of the energies -- so no penalty on the
            # size law or the contextual terms can reach it.  A projection can.
            # MRR peaked at 0.0872 with lam sd 0.380, so 0.45 sits above the useful range
            # and below the cliff.  Applied AFTER the centring so the mean stays at zero.
            if a.lam_sd_max > 0 and not a.size_stage:
                with torch.no_grad():
                    _sd = m.lam.std()
                    if float(_sd) > a.lam_sd_max:
                        m.lam.mul_(a.lam_sd_max / _sd.clamp_min(1e-12))
            if pn is not None and (a.rho0_curv > 0 and not a.composition_stage
                    and not bool(m.factored_size_enabled)):
                with torch.no_grad():
                    _r = m.rho_0()
                    _d1 = _r[1:] - _r[:-1]
                    _d2 = (_d1[1:] - _d1[:-1]).clamp_min(a.rho0_curv)
                    _d1 = torch.cat([_d1[:1], _d1[:1] + torch.cumsum(_d2, 0)])
                    m.rho_0_free.copy_(torch.cumsum(_d1, 0))
            # Cap lambda_max, not ||phi||.
            #
            # Proposition 3 needs lambda_max(Lambda) < 1 for the mode to be unique and for
            # the fixed-point iteration inside log_Z to contract.  A fixed cap on ||phi||
            # does not deliver that: Lambda = sum_j pi_j(1-pi_j) phi_j phi_j', so
            # lambda_max <~ ||phi||^2 E[n], and the cap bounds each term while E[n] scales
            # the sum.  Run 12 showed the failure exactly -- ||phi|| FELL from 0.173 to
            # 0.155 while lambda_max rose to 2.596, because E[n] had run away to ~108.
            # Above 1 the iteration diverges, the estimate degrades, and the arithmetic
            # drifts into denormals: iterations 400-600 took 41.6 minutes against 8.
            #
            # Inverting the bound gives the cap that actually holds the condition, with
            # E[n] read off the size law the normaliser already returns.
            # Drive the cap off the WORST kept trip, not the batch mean.
            #
            # lambda_max is a per-trip quantity and section 14 requires it below 1 on every
            # trip, not on average.  Feeding the cap a batch mean let a calibrated-looking
            # E[n] of 9.0 sit alongside individual trips near 68, and lambda_max reached
            # 1.776 while the cap -- computed as sqrt(0.5/9.0) = 0.236, above the 0.20
            # ceiling -- never engaged at all.  That is the third time in this file that a
            # mean stood in for a worst case; the ESS gate had the same defect.
            #
            # Trips rejected by the ESS gate contribute no gradient, so they need not
            # satisfy the constraint and must not be allowed to clamp phi for everyone
            # else.  The maximum is taken over the kept trips only.
            if pn is None:
                en_all = None
            else:
                en_all = (pn.detach() * torch.arange(1, pn.shape[1] + 1,
                                                     dtype=pn.dtype)).sum(1)
            # A QUANTILE of the kept trips, not their maximum.
            #
            # The cap was being set by whichever trip had the largest E[n], and a handful
            # expect 50+ items: measured max E[n] per batch of 53.3, 45.9, 58.3 forced
            # ||phi|| <= 0.09 for the whole model.  Interaction strength is ||phi||^2, so
            # the pairwise term collapsed to 0.0045 -- and with it went the thing the model
            # exists for.  Simulated co-occurrence came out at 8% of the observed rate, with
            # 168 of the 200 commonest real pairs never generated at all.
            #
            # Those trips are already excluded from the LOSS by the ESS gate; letting them
            # still dictate phi for every other trip was the mistake.  A high quantile keeps
            # the constraint honest for the bulk while refusing to let the tail set it.
            # Quantile over ALL trips, not the kept ones.  Taking it over the survivors
            # made the constraint self-relaxing: dropping a hard trip lowered en_max, which
            # raised the phi budget, which grew phi, which broke more trips.  Measured at
            # iteration 800 of run25 -- lambda_max 11.0, ESS 0.674, 1,642 trips dropped and
            # ||phi|| pinned at its ceiling.
            en_max = float(torch.quantile(en_all, a.lam_q)) if en_all.numel() > 3 \
                else float(en_all.max())
            # The per-item cap is now a loose ceiling; the binding constraint is the
            # global budget sum_j ||phi_j||^2, which is what lambda_max depends on.
            # pi_j(1-pi_j) per product, EXACTLY -- Corollary 2 by autograd.  Every
            # cheap proxy bound the wrong thing or bound it inconsistently; this is the
            # quantity section 14 names, at about one extra normaliser evaluation.
            # pi_exact is approximately one extra normaliser backward pass.  Recomputing it
            # on every update is wasteful when the resulting budget is far from binding
            # (run112/120 lambda_max ~0.03 against target 0.85).  It is a slowly moving
            # safety weight, not a likelihood term: refresh periodically and reuse it while
            # the every-update row/operator caps and per-batch QMC guards remain live.
            _refresh_pi = (not a.size_stage and a.pi_project_every > 0 and
                           ((it - it0 - 1) % a.pi_project_every == 0))
            if _refresh_pi:
                pslot = m.pi_exact(ix)
                with torch.no_grad():
                    w_new = torch.zeros(m.phi.shape[0], dtype=pslot.dtype)
                    cntj = torch.zeros_like(w_new)
                    w_new.index_add_(0, ix.item, pslot * (1 - pslot))
                    cntj.index_add_(0, ix.item, torch.ones_like(pslot))
                    seen = cntj > 0
                    w_new[seen] /= cntj[seen]
                    prev = getattr(m, "_pi_w", None)
                    m._pi_w = (w_new if prev is None else
                               torch.where(seen, 0.5 * prev + 0.5 * w_new, prev))
            cap = phi_cap
            # With pi weights the bound is lambda_max itself, so the budget IS the target.
            # The budget caps the TRACE sum_j pi_j(1-pi_j)||phi_j||^2, but lambda_max is
            # about trace / effective rank.  Capping the trace at lam_target assumes erank=1
            # and throws away exactly what whitening and a wide Kz were built to buy: run32
            # reached erank 121 of 128 and lambda_max 0.016 against a 0.85 target -- a
            # constraint 50x tighter than intended, which shrank ||phi|| from 0.120 to 0.001
            # in 800 iterations.  Scaling the budget by erank is what makes the width count.
            if getattr(m, "_pi_w", None) is not None:
                with torch.no_grad():
                    _sv = torch.linalg.svdvals(m.phi.detach())
                    _er = float((_sv ** 2).sum() ** 2 / (_sv ** 4).sum().clamp_min(1e-30))
                budget = a.lam_target * max(_er, 1.0) * a.budget_f
            else:
                budget = a.lam_target / max(en_max, 1.0) * float(m.phi.shape[0]) * a.budget_f
            # Proximal L1 must be scaled by the step size: prox_{lr * lambda}.  Applying
            # the raw coefficienteach step subtracted 0.02 from every norm against a learning
            # rate of 0.005, and phi was annihilated in five steps (probe showed |phi| 0.000).
            lr_now = opt.param_groups[0]["lr"]
            if not a.size_stage:
                if a.freeze_rho_c:
                    # Ablation: hold rho_c at zero so the affinity partition contributes no
                    # attraction.  If phi still climbs, the pressure is not coming from the
                    # category term.
                    with torch.no_grad():
                        m.rho_c.zero_()
                else:
                    m.project_rho_c(a.rho_c_floor)
                m.project(cap, budget=budget, thresh=a.phi_l1 * lr_now,
                          centre=a.phi_centre > 0, whiten=a.phi_whiten,
                          op_max=a.phi_op_max)
            # Sparsity is what makes log Z estimable, and it is the FRACTION of products
            # carrying phi that matters, not the norm.  Measured on run55's checkpoint at
            # ||phi|| = 0.96, the gap between 16 draws and 4096 was:
            #     1% of products   0.000 nats      20% of products   23.9 nats
            #     5% of products   1.888 nats     100% of products  128.4 nats
            # so a dense phi is unfittable at any affordable draw count while a sparse one
            # is exact at the cheapest.  A hard top-k keeps that fraction where the probe
            # measured it; an L1 penalty controls it only indirectly and drifts as phi grows.
            if not a.size_stage and 0.0 < a.phi_topk < 1.0:
                with torch.no_grad():
                    _nrm = m.phi.norm(dim=1)
                    _k = max(1, int(a.phi_topk * _nrm.shape[0]))
                    _keep = torch.topk(_nrm, _k).indices
                    _mask = torch.zeros_like(_nrm, dtype=torch.bool)
                    _mask[_keep] = True
                    m.phi.mul_(_mask.unsqueeze(1).to(m.phi.dtype))
            # A STATIC mask chosen from co-purchase, which is what --phi-topk should have
            # been.  Selecting by current norm picks products where phi is unpenalised
            # rather than useful: run57's mask drifted to rare items (median frequency rank
            # 2138 of 5455) and ZERO of the 200 most co-purchased pairs had both products
            # retained, so phi_j.phi_k was identically zero on every pair the evaluation
            # scores.  Ranking by co-purchase count instead puts 98% of those pairs inside
            # the mask at a 3% budget -- and, being fixed in advance, it can be checked
            # before the run rather than diagnosed after it.
            if not a.size_stage and phi_mask is not None:
                with torch.no_grad():
                    m.phi.mul_(phi_mask)
            # Shrink the STORE context vector toward its mean over stores.
            #
            # Measured post-hoc on run65's best checkpoint, against an observed Var(n) of
            # 114.2 and mean 8.71:
            #     as-is             E[n] 11.11  w 32.1  s 258.4  varpop 290.5
            #     xi -> its mean    E[n]  7.27  w 22.6  s 103.7  varpop 126.3
            # so it takes the population variance from 2.5x the observed value to 1.1x, and
            # the mean from +28% to -17%, the closest any configuration has come on both.
            #
            # It does NOT work through the common shift: sd(Delta) barely moves (0.304 ->
            # 0.286).  It works because the store term inflates Var(n|trip) -- w falls 32.1
            # -> 22.6 -- and s carries Var(n) SQUARED, so the between-trip spread follows.
            #
            # At full strength xi is constant across stores, which makes zeta_j . xi a
            # per-product constant that lambda_j already spans; the store term is then
            # redundant rather than merely damped.  That is a real loss of capacity and the
            # reason this is a tunable projection rather than a deletion.
            # NOTE: a hard projection of the contextual residuals to their mean (--ctx-shrink 0)
            # was tried and is CATASTROPHIC.  It improves complete-the-basket MRR 12x
            # (0.0036 -> 0.0442) but puts every product in every basket: E[n] 120.0 against an
            # observed 7.2, held-out set/basket -315.7 at shrink 0 and -121.8 at 0.5.  Those
            # residuals are load-bearing -- they SUPPRESS most of the catalogue -- and ranking
            # is blind to the level of b, so the ranking metric could not see the damage.
            # The pooling penalty below is the non-destructive form: shrink the dispersion,
            # keep the level.  --ctx-shrink is retained only to reproduce that finding.
            if not a.size_stage and a.ctx_shrink < 1.0:
                with torch.no_grad():
                    for _p in (m.mu, m.zeta, m.psi):
                        _mu = _p.mean(0, keepdim=True).clone()
                        _p.mul_(a.ctx_shrink).add_((1.0 - a.ctx_shrink) * _mu)
            if not a.size_stage and a.xi_shrink > 0:
                with torch.no_grad():
                    # take the mean BEFORE mutating: mul_(1-alpha) at alpha=1 zeroes xi, and
                    # the mean of a zeroed tensor is zero, which would delete the term rather
                    # than flatten it.
                    _xm = m.xi.mean(0, keepdim=True).clone()
                    m.xi.mul_(1.0 - a.xi_shrink).add_(a.xi_shrink * _xm)
            # Var(n) is the quantity every other failure runs through; project it too.
            if (a.var_target != 0 and pn is not None and not a.composition_stage
                    and not bool(m.factored_size_enabled)):
                _n = torch.arange(1, pn.shape[1] + 1, dtype=pn.dtype)
                _e = (pn.detach() * _n).sum(1)
                _v = float(((pn.detach() * _n ** 2).sum(1) - _e ** 2).mean())
                # Drive the projections from a SMOOTHED estimate, not this batch's.
                #
                # Both are feedback controllers on rho_0, and both were reading a 24-trip,
                # 16-draw estimate -- far noisier than the 384-trip, 32-draw figure the
                # checkpoint prints.  Whatever noise is in that reading is injected into
                # rho_0 every iteration, so rho_0 random-walks even when the model is right
                # on average.  Measured in run62, with the estimator itself converged
                # (E[n]-converged passing at every checkpoint, e.g. 8.5 against 9.1 at 8x
                # draws), E[n] still swung 8.5 -> 30.0 -> 12.9 -> 8.7 -> 13.8 across
                # consecutive checkpoints.  A converged measurement that keeps moving is the
                # parameters oscillating, not the estimate.
                #
                # Two things make it worse than ordinary noise.  project_mean divides by
                # v_now, and the model runs narrow (var 18-21 against an observed 67), so
                # the denominator is small and the step large.  And once b hits its +-0.5
                # clamp the correction no longer shrinks as the error shrinks -- a fixed
                # step applied every iteration, which is bang-bang control and settles at a
                # limit cycle rather than at the target.
                #
                # An EMA leaves the fixed point exactly where it was (it converges to the
                # same mean) and only removes the noise driving the loop.
                # The Newton step for a GLOBAL correction to the POPULATION mean divides
                # by the POPULATION variance, Var(n) = E[Var(n|trip)] + Var[E(n|trip)].
                # Passing only the within-trip term made the denominator too small -- 26
                # against 182 at run64's iteration 5000, so every step was 7x the true
                # Newton step and the controller overshot by design.
                _vpop = _v + float(_e.var())
                _em = float(_e.mean())
                if a.proj_ema > 1:
                    _beta = 1.0 / a.proj_ema
                    e_ema = _em if e_ema is None else (1 - _beta) * e_ema + _beta * _em
                    v_ema = _v if v_ema is None else (1 - _beta) * v_ema + _beta * _v
                    _em_use, _v_use = e_ema, v_ema
                else:
                    _em_use, _v_use = _em, _v
                # project_var is OFF by default: it is a one-way ratchet that decalibrates
                # the very quantity it targets.  c = 0.5*(1/v_target - 1/v_now) is clamped
                # NON-NEGATIVE (the widening direction sent E[n] to the nmax boundary within
                # two steps), so rho_0 can only ever gain positive n^2 curvature and the size
                # law can be narrowed but never widened.  Driven by a noisy v_now, any batch
                # whose estimate strays above target narrows the law permanently, and over
                # thousands of steps that ratchets monotonically down.
                #
                # Measured over 150 iterations from a common checkpoint, against an observed
                # Var(n) of 67:
                #     project_var ON,  size_kl ON    var 37.7
                #     project_var OFF, size_kl ON    var 65.3
                #     project_var OFF, size_kl OFF   var 76.8
                # The cross-entropy term calibrates the spread on its own; project_var then
                # halves it.  project_mean stays on -- it is pulling E[n] the right way
                # (14.2 with it against 17.2 without, observed 7.2) and its correction is
                # signed, so it has no ratchet.
                if a.var_project:
                    m.project_var(_v_use, emp_var if a.var_target < 0 else a.var_target,
                                  damp=a.var_damp)
                _b = m.project_mean(_em_use, obs_mean, _vpop, damp=a.var_damp)
                # Count clamp hits: if the mean correction is saturating, the controller is
                # in bang-bang and no amount of damping will let it settle.
                if abs(_b) >= 0.5 * a.var_damp - 1e-12:
                    n_bang += 1
            # The data's aggregate elasticity is -0.121 and Proposition 1 gives
            # elasticity = -(gamma.beta) Var(n) / E[n], so the mean sensitivity it implies
            # is 0.121 * E[n] / Var(n).  Projected, not penalised -- see project_price.
            if (a.elast_w > 0 and not a.size_stage
                    and not bool(m.factored_size_enabled)):
                # Target the elasticity through the model's CURRENT moments, not the
                # empirical ones.  Proposition 1 gives elasticity = -(gamma.beta) Var/E, so
                # gamma.beta = |target| * E/Var -- and using E_obs/Var_obs assumes the size
                # law is already calibrated.  It is not: at iteration 2000 of run26 the
                # model sat at Var/E = 37/17.1 = 2.16 against an empirical 10.6, so a
                # gamma.beta pinned for the empirical ratio delivered -0.040 instead of
                # -0.121.  Reading E and Var off pn each step makes the target self-correct
                # as the size law converges.
                # The Newton step for a GLOBAL correction to the POPULATION mean divides
                # by the POPULATION variance, Var(n) = E[Var(n|trip)] + Var[E(n|trip)].
                # Passing only the within-trip term made the denominator too small -- 26
                # against 182 at run64's iteration 5000, so every step was 7x the true
                # Newton step and the controller overshot by design.
                _vpop = _v + float(_e.var())
                _em = float(_e.mean()) if a.var_target != 0 else obs_mean
                _vm = _v if a.var_target != 0 else emp_var
                m.project_price(abs(a.elast_target) * max(_em, 1e-6) / max(_vm, 1e-6))
        if sched is not None:
            sched.step()
        hist.append(float(loss.detach()))
        ess_hist.append(e_bar)
        emin_hist.append(e_min)
        _qse = getattr(m, "_last_qmc_logz_se", None)
        if _qse is not None:
            qmc_se_hist.append(float(_qse.mean()))
            qmc_se_max_hist.append(float(_qse.max()))
        _qm = getattr(m, "_last_qmc_mode_count", None)
        if _qm is not None:
            qmc_mode2_hist.append(float((_qm == 2).double().mean()))
        with torch.no_grad():
            _e = (pn * torch.arange(1, pn.shape[1] + 1, dtype=pn.dtype)).sum(1)
            en_hist.append(float(_e.mean()))
            enmax_hist.append(float(_e.max()))
        if ce is not None:
            ce_hist.append(float(ce.detach()))
        if elast is not None:
            el_hist.append(float(elast.detach()))
        if it - it0 == a.probe:
            dt = time.time() - t0
            per = dt / a.probe
            log(f"  {a.probe} iterations in {dt:.1f}s = {per:.3f}s/iteration")
            log(f"  slots per batch {ix.item.numel():,}, rows {ix.n_rows:,}, "
                f"Cpad {ix.Cpad}")
            log(f"  implied wall clock: {per * a.iters / 3600:.2f} h for {a.iters:,} "
                f"iterations")
            log(f"  loss {np.mean(hist[-20:]):.3f}   ESS {np.mean(ess_hist[-20:]):.3f}"
                f"   |phi| {float(m.phi.detach().norm(dim=1).mean()):.3f}   skipped {n_skip}"
                f"   retried {n_qretry}"
                f"   ESS min {np.min(emin_hist):.3f}"
                + (f"   RQMC logZ SE mean/max {np.mean(qmc_se_hist[-20:]):.4f}/"
                   f"{np.max(qmc_se_max_hist[-20:]):.4f}"
                   if qmc_se_hist else ""))
            if a.probe_only:
                json.dump(dict(sec_per_iter=per, iters=a.iters, n_par=npar,
                               slots=int(ix.item.numel()), ess=float(ess.mean())),
                          open(os.path.join(OUT, "v3_probe.json"), "w"), indent=2)
                log("  wrote out/v3_probe.json; stopping (--probe-only)")
                return
        if it % a.eval_every == 0:
            if a.qmc_n > 0 and _qmc_eval_n > a.qmc_n:
                _set_qmc_rule(_qmc_eval_n, a.qmc_seed + 2_000_003)
            # lambda_max BEFORE evaluate(): evaluate reassigns m.house/m.ctx to the
            # validation chunks, and ix here is the training batch.  Calling it after
            # indexes a 40-trip batch into a 32-trip household tensor.
            lam_max = m.lambda_max(ix)
            lam_seen = lam_max
            # Log the quantity the QMC normaliser's validity actually rests on.
            #
            # The claim that products are free is that the mode radius ||z*|| stays small
            # however many products carry phi, because z* = Phi'pi(z*) and sum_j pi_j = E[n]
            # is pinned by the size law.  Measured offline that held (1.91 at J=20 down to
            # 0.07 at all 5,455), but offline is a fixed phi -- training moves phi, so the
            # bound is asserted every eval rather than assumed.  Accuracy against exact
            # enumeration was 0.0001 nats at ||z*|| ~ 12 and degrades past ~18, so anything
            # under 10 is comfortable and this exists to catch a drift toward that edge.
            if m.quad_a is not None:
                with torch.no_grad():
                    _pm = float(m.phi.norm(dim=1).max())
                    if bool(m.factored_size_enabled):
                        _zn = float("nan")
                    else:
                        # This diagnostic needs only the proposal mode.  Calling log Z with
                        # return_mode would evaluate every QMC node as well.
                        _zh = m._adaptive_frame(ix, True, m.quad_steps)[0][:, 0]
                        _zn = float(_zh.norm(dim=-1).max())
                _mode_text = ("fixed-n mode checked by observed normalizer below"
                              if bool(m.factored_size_enabled)
                              else f"max||z*|| {_zn:.2f} (accurate to ~12, fails ~18)")
                log(f"  QMC envelope: {_mode_text}, max||phi_j|| {_pm:.3f}, "
                    f"phi-carrying products "
                    f"{int((m.phi.norm(dim=1) > 1e-6).sum()):,}")
            # Project on the MEASURED lambda_max, not a proxy for it.
            #
            # The pi-weighted budget averages pi(1-pi) per product across the batch, but a
            # product sits in every trip's assortment and is bought in almost none, so its
            # average is tiny: measured 0.232 against a lambda_max of 2.202, ten times too
            # small, and a budget of 4.2 never bound.  lambda_max is computed per trip at
            # the mode, where the concentration that makes it large survives.  Averaging
            # destroys it -- the fourth time in this file a mean has stood in for a tail.
            # Hold phi AT the feasibility frontier, not merely below it.
            #
            # lambda_max = sum_j pi_j ||phi_j||^2 ~ E[n] <||phi||^2> is a BUDGET, and the
            # estimator is valid only while it is under 1 (measured: log Z degrades past
            # lam_max ~ 1 and collapses by 4).  The fit sits at 0.121 -- 12% of the budget --
            # because phi starts at a SADDLE (dE/dphi_j = sum_{k in S} phi_k = 0 at phi = 0)
            # and escapes only exponentially from the 0.03 seed: 0.03 -> 0.099 over 2,600
            # updates, against the ~0.93 the co-occurrence data implies.
            #
            # Measured on the run155 checkpoint with NO training, scaling phi toward the
            # frontier is worth more than the entire fitted interaction:
            #     scale 1.0  lam_max 0.121  set LL -57.0893   (fitted; phi=0 gives -57.12)
            #     scale 2.0  lam_max 0.711  set LL -57.0323   <- +0.057 nats, free
            #     scale 3.0  lam_max 1.556  set LL -57.9885   <- past the frontier
            # so the projection scales UP as well as down, with a floor well inside 1.
            if a.lam_project > 0 and a.lam_floor > 0 and 0 < lam_max < a.lam_floor:
                _up = math.sqrt(a.lam_floor / max(lam_max, 1e-9))
                _up = min(_up, a.lam_up_max)
                with torch.no_grad():
                    m.phi.mul_(_up)
                log(f"  lambda_max {lam_max:.3f} < floor {a.lam_floor}: phi scaled UP by "
                    f"{_up:.3f} (frontier is 1.0; log Z degrades past it)")
            elif a.lam_project > 0 and lam_max > a.lam_target:
                with torch.no_grad():
                    m.phi.mul_(math.sqrt(a.lam_target / lam_max))
                log(f"  lambda_max {lam_max:.3f} > {a.lam_target}: phi scaled by "
                    f"{math.sqrt(a.lam_target/lam_max):.3f}")
            # Held-out size calibration.  The E[n] logged before this was measured on the
            # TRAINING batch and read 8.1 against 7.8 all night while the held-out figure
            # was 11.41 against 6.31.  An in-sample calibration metric is not a calibration
            # metric.
            vix, vctx, _, vhh, _, vLT, _, _ = B.make(va[: min(128, a.n_val)])
            sh, sc = m.house, m.ctx
            m.house, m.ctx = vhh, vctx
            with torch.no_grad():
                ev_e, ev_v = m.size_moments(vix, n_draws=a.draws * 2,
                                            generator=torch.Generator().manual_seed(0))
                # E[n] needs the convergence check log Z gets.  At lambda_max ~ 3.5 the
                # 32-draw figure is a LOWER BOUND: on validation it read 5.41 against 7.43
                # at 384 draws, and on test it ran 13.0 -> 17.6 -> 20.4 without converging.
                # Every E[n] logged before this was understated, and the column looked calm
                # only because it was too cheap to reveal the problem.
                # Under deterministic quadrature n_draws is ignored, so the historical
                # "8x" call was an identical second full-catalogue log Z evaluation.
                if m.quad_a is not None or m.quad is not None:
                    ev_e8 = ev_e
                else:
                    ev_e8, _ = m.size_moments(
                        vix, n_draws=a.draws * 8,
                        generator=torch.Generator().manual_seed(0))
                ho_e8 = float(ev_e8.mean())
            m.house, m.ctx = sh, sc
            vobs = np.bincount(vLT.numpy(), minlength=vix.B)
            # Var(n) must be the POPULATION variance, to match what vobs.var() measures.
            #
            # This printed mean(ev_v) -- E[Var(n | trip)], the average spread WITHIN a trip --
            # and compared it against the variance of observed basket sizes ACROSS trips.
            # Different quantities.  The law of total variance supplies the missing piece:
            #     Var(n) = E[Var(n|trip)] + Var[E(n|trip)]
            # Measured on run63's best checkpoint over 384 validation trips, the two terms
            # were 28.4 and 73.4, so the model's population variance is 101.8 against an
            # observed 63.4.  The old field read 28.4 and made the size law look far too
            # NARROW when it is in fact too WIDE -- and every "too narrow" reading this
            # session, including project_var's target of emp_var, inherited that inversion.
            ho_e = float(ev_e.mean())
            ho_e_med = float(ev_e.median())
            ho_v = float(ev_v.mean() + ev_e.var())
            ho_v_within = float(ev_v.mean())        # kept, so the split stays visible
            ho_e_spread = float(ev_e.var())
            vb, vl, vu, vt, vsz, vco = evaluate(
                m, B, va[:a.n_val], a.draws * 2, gen, use_units=a.units,
                return_decomposition=True)
            m.house, m.ctx = hh, ctx
            rec_mrr, rec_med = rec_eval(m, B, va[:a.n_rec]) if a.n_rec > 0 \
                else (float('nan'), float('nan'))
            ep = it * a.batch / max(len(tr), 1)
            cum_it = cum_base + (it - it0) + a.cum_offset
            cum_ep = cum_it * a.batch / max(len(tr), 1)
            log(f"  it {it:5d} ep {ep:5.3f} cum {cum_ep:5.3f}  train {np.mean(hist[-a.eval_every:]):8.3f}  "
                f"set/bskt {vb:8.3f} (size {vsz:.3f} comp {vco:.3f})  "
                f"units/bskt {vu:7.3f}  total {vt:8.3f}  "
                f"ESS {np.mean(ess_hist[-a.eval_every:]):.3f} "
                f"(min {np.min(emin_hist[-a.eval_every:]):.3f})  "
                + (f"RQMCse {np.mean(qmc_se_hist[-a.eval_every:]):.4f}/"
                   f"{np.max(qmc_se_max_hist[-a.eval_every:]):.4f}  "
                   if qmc_se_hist else "")
                + (f"mode2 {np.mean(qmc_mode2_hist[-a.eval_every:]):.1%}  "
                   if qmc_mode2_hist else "")
                +
                f"|phi| {float(m.phi.detach().norm(dim=1).mean()):.3f} "
                f"(max {float(m.phi.norm(dim=1).max()):.2f} "
                f"zero {float((m.phi.norm(dim=1) < 1e-8).double().mean()):.0%} "
                f"erank {float((lambda sv: (sv**2).sum()**2/(sv**4).sum())(torch.linalg.svdvals(m.phi.detach()))):.0f})  "
                f"lam_max {lam_max:.3f}  E[n] {ho_e:.1f}(med {ho_e_med:.1f})/{vobs.mean():.1f} "
                f"[{ho_e8:.1f}@8x] var {ho_v:.0f}/{vobs.var():.0f} "
                f"(w{ho_v_within:.0f}+s{ho_e_spread:.0f})  "
                f"MRR {rec_mrr:.4f}(med {rec_med:.0f})  "
                # lam is the per-product intercept and the parameter ranking depends on.
                # pi_exact used to shadow it into self.__dict__ as a detached tensor, so it
                # took exactly one Adam step (|lam| <= lr) and froze, runs 29-71.  Logged so
                # a dead intercept can never again be invisible for forty runs.
                # lam is the per-product intercept and the parameter ranking depends on.
                # 'season' is the spread of the CENTRED seasonal term across products and
                # weeks -- what mu'delta legitimately contributes once it can no longer hold
                # a per-product constant.  'db' is the constant channel itself: gauge-fixed
                # to zero by delta_c(), logged as a guard that it stays there.
                f"lam sd {float(m.lam.std()):.3f}"
                f"(season {float((m.mu @ m.delta_c().T).std()):.3f} "
                f"db {float((m.mu * m.delta_c().mean(0)).sum(-1).std()):.1e})  "
                f"elast {np.mean(el_hist[-a.eval_every:]) if el_hist else float('nan'):+.3f}"
                f"/{a.elast_target:+.3f}  "
                f"skip {n_skip}  qretry {n_qretry}  qbad {n_qbad}  gbad {n_gradbad}  "
                f"drop {n_drop}  redo {n_redo}  "
                f"bang {n_bang}  "
                f"{(time.time()-t0)/60:.1f} min")
            # MACHINE-READABLE eval record, one JSON object per line.
            #
            # The human line above is ~1,200 characters and every diagnosis this run has
            # needed so far started by regex-ing it back apart -- badly, and differently
            # each time.  This writes the same scalars as data, so later analysis is a
            # query.  It also captures things the text line omits and that mattered here:
            # per-block gradient norms (which located the dead price block and the fact
            # that lambda had no available gain), the lz guard, and wall-clock, so a stall
            # can be told apart from a slowdown after the fact.
            # Divergence tripwire.  A blown-up utility block shows first as E[n] running
            # away to n_max while the data sits near 8.6; run407 reached -6.6e9 that way
            # and every later eval was wasted compute.  Stop at the first eval that shows
            # it, with the diagnosis, rather than clamping -- a clamp would hide it.
            if float(ho_e) > 0.5 * a.nmax and float(vobs.mean()) < 0.25 * a.nmax:
                log(f"DIVERGED at it {it}: model E[n] = {float(ho_e):.1f} against observed "
                    f"{float(vobs.mean()):.2f} (n_max {a.nmax}).  The utility block has run "
                    f"away; set/basket {vb:.4f}.  Stopping.")
                raise SystemExit(3)
            if a.metrics_jsonl:
                try:
                    _gn = {}
                    for _n, _p in m.named_parameters():
                        if _p.grad is not None:
                            _gn[_n] = float(_p.grad.norm())
                    _rec = dict(
                        it=int(it), cum_it=int(cum_it), epoch=float(ep), cum_epoch=float(cum_ep),
                        wall_min=float((time.time() - t0) / 60.0), unix=float(time.time()),
                        train_loss=float(np.mean(hist[-a.eval_every:])),
                        set_per_basket=float(vb), set_per_line=float(vl),
                        size_per_basket=float(vsz), comp_per_basket=float(vco),
                        units_per_basket=float(vu), total_per_basket=float(vt),
                        mrr=float(rec_mrr), mrr_median_rank=float(rec_med),
                        phi_mean=float(m.phi.detach().norm(dim=1).mean()),
                        phi_max=float(m.phi.detach().norm(dim=1).max()),
                        phi_zero_frac=float((m.phi.detach().norm(dim=1) < 1e-8).double().mean()),
                        lam_max=float(lam_max), lam_sd=float(m.lam.std()),
                        en_model=float(ho_e), en_obs=float(vobs.mean()),
                        var_model=float(ho_v), var_obs=float(vobs.var()),
                        elast=float(np.mean(el_hist[-a.eval_every:])) if el_hist else None,
                        elast_target=float(a.elast_target),
                        ess_mean=float(np.mean(ess_hist[-a.eval_every:])) if ess_hist else None,
                        ess_min=float(np.min(emin_hist[-a.eval_every:])) if emin_hist else None,
                        qmc_se_mean=float(np.mean(qmc_se_hist[-a.eval_every:])) if qmc_se_hist else None,
                        qmc_se_max=float(np.max(qmc_se_max_hist[-a.eval_every:])) if qmc_se_max_hist else None,
                        n_skip=int(n_skip), n_qretry=int(n_qretry), n_qbad=int(n_qbad),
                        n_gradbad=int(n_gradbad), n_drop=int(n_drop),
                        lr=float(opt.param_groups[0]["lr"]),
                        grad_norms=_gn,
                    )
                    with open(a.metrics_jsonl, "a") as _fh:
                        _fh.write(json.dumps(_rec) + "\n")
                except Exception as _e:      # logging must never kill a run
                    log(f"  [metrics] record skipped: {type(_e).__name__}: {_e}")
            if vb > 0:
                log("  ABORT: held-out log-likelihood is positive, which is impossible. "
                    "The objective is being maximised through a defect, not a fit.")
                return
            # A collapsing normaliser does not need to make the likelihood POSITIVE to be
            # fake, only to make it better.  run34 reported -35.07 -- nine nats past
            # anything legitimate -- while its log Z ran 7.41 at 8 draws to 76.50 at 2048
            # and was still climbing, so the 8-draw figure understated it by 69 nats and the
            # loop was rewarded one-for-one.  ESS read 0.919: every draw in the same region,
            # agreeing with each other, together missing nearly all the mass.  ESS cannot
            # see this, so log Z is compared against a high-draw reference instead.
            if m.quad_a is not None:
                _factored_n = torch.bincount(lt, minlength=ix.B) \
                    if bool(m.factored_size_enabled) else None
                with torch.no_grad():
                    if _factored_n is not None:
                        lz_lo, _ = m.log_Z_observed_size(
                            ix, _factored_n, return_ess=True)
                    else:
                        lz_lo, _ = m.log_Z(ix, drop_empty=True, return_ess=True)
                    _qse = getattr(m, "_last_qmc_logz_se", None)
                if _qse is None:
                    raise RuntimeError(
                        "QMC normaliser has one scramble and no error estimate; use "
                        "--qmc-reps >= 2")
                rep_gap = 2.0 * float(_qse.max())
                # Replicate spread measures randomised-QMC variance at N.  An independent
                # 2N rule also catches a common finite-N drift that all four N scrambles
                # could share.  Keep training's fixed rule intact after the check.
                old_rule = m.quad_a
                old_mix_rule = getattr(m, "quad_mix_a", None)
                try:
                    m.quad_a = sobol_grid(
                        m.Kz, 2 * int(m._qmc_n), seed=int(m._qmc_seed) + 1_000_003,
                        replicates=m.quad_replicates)
                    if getattr(m, "quad_size_bands", 0):
                        m.quad_mix_a = sobol_mixture_grid(
                            m.Kz, 2 * int(m.quad_mix_n),
                            seed=int(m._qmc_seed) + 1_000_003,
                            replicates=m.quad_replicates, components=2)
                    with torch.no_grad():
                        lz_hi = (m.log_Z_observed_size(ix, _factored_n)
                                 if _factored_n is not None
                                 else m.log_Z(ix, drop_empty=True))
                finally:
                    m.quad_a = old_rule
                    m.quad_mix_a = old_mix_rule
                conv_gap = float((lz_hi - lz_lo).abs().max())
                gap = max(rep_gap, conv_gap)
                _zname = "observed-size log Z_n" if _factored_n is not None else "log Z"
                log(f"  normaliser check ({_zname}): RQMC 2*SE {rep_gap:.4f} nats; "
                    f"independent N-vs-2N |gap| {conv_gap:.4f}; guard {gap:.4f} "
                    f"({m.quad_replicates} scrambles)")
            else:
                with torch.no_grad():
                    g_a = torch.Generator().manual_seed(11)
                    lz_lo = m.log_Z(ix, n_draws=_nd, generator=g_a, drop_empty=True,
                                    mix_scales=_mix, aniso=a.aniso,
                                    antithetic=a.antithetic > 0)
                    g_b = torch.Generator().manual_seed(11)
                    lz_hi = m.log_Z(ix, n_draws=_nd * 16, generator=g_b, drop_empty=True,
                                    mix_scales=_mix, aniso=a.aniso,
                                    antithetic=a.antithetic > 0)
                    gap = float((lz_hi - lz_lo).mean())
                log(f"  normaliser check: log Z at {_nd} draws vs {_nd*16} "
                    f"differs by {gap:+.3f} nats")
            # GAP-BASED phi control, replacing the lam_max projection.
            #
            # lam_max is not a safety measure in the regime where safety matters -- it is
            # NON-MONOTONE in phi and FALLS as the estimator deteriorates.  Measured on
            # run68 by scaling phi, with the gap that actually decides soundness:
            #     scale 0.70  lam_max 0.663  gap -0.007   ok
            #     scale 1.00  lam_max 1.004  gap  0.147   ok
            #     scale 1.30  lam_max 0.913  gap  0.578   ok
            #     scale 1.60  lam_max 0.596  gap  3.564   BROKEN
            # The model is sound at lam_max 1.004 and broken at 0.596.  The cause is the
            # pi(1-pi) weighting inside Lambda: as phi grows, inclusion probabilities
            # saturate, pi(1-pi) -> 0, and lam_max collapses even as the integral gets
            # harder.  So --lam-target has been regulating the wrong quantity all along.
            #
            # The gap is the honest diagnostic -- it is what the abort has always used, and
            # what has caught every real failure.  Control phi with it directly.
            #
            # gap rises roughly exponentially in the phi scale: log gap went -1.92 -> -0.55
            # -> 1.27 over scale 1.0 -> 1.3 -> 1.6, a slope of about 5.3 per unit scale.  So
            # the correction to bring gap to target is Delta_scale = log(target/gap)/5.3,
            # clamped, which is gentle because the response is steep.
            if a.gap_project > 0 and gap > a.gap_project:
                _f = math.exp(math.log(a.gap_project / max(gap, 1e-9)) / 5.3)
                _f = min(max(_f, 0.85), 1.0)
                with torch.no_grad():
                    m.phi.mul_(_f)
                log(f"  gap {gap:+.3f} > {a.gap_project}: phi scaled by {_f:.3f} "
                    f"(lam_max reads {lam_max:.3f}, which is NOT the binding quantity)")
            # sampler-vs-analytic consistency: same distribution, two routes.
            samp_n = float("nan")
            try:
                with torch.no_grad():
                    sm_ix, sm_ctx, _, sm_hh, *_ = B.make(va[:24])
                    m.house, m.ctx = sm_hh, sm_ctx
                    sm_e, sm_v = m.size_moments(
                        sm_ix, n_draws=a.draws,
                        generator=torch.Generator().manual_seed(it))
                    if bool(m.factored_size_enabled):
                        # This check is about the size law.  Under the factorisation its
                        # exact sampler is a direct categorical draw; the conditional set
                        # sampler is a separate z|n operation and must not be substituted
                        # with the old joint sampler.
                        psize = m.factored_size_log_p.exp().unsqueeze(0).expand(sm_ix.B, -1)
                        sampled_size = torch.multinomial(
                            psize, 1, generator=torch.Generator().manual_seed(it))[:, 0] + 1
                        samp_n = float(sampled_size.double().mean())
                    else:
                        bk = m.sample(sm_ix, n_draws=a.draws,
                                      generator=torch.Generator().manual_seed(it))
                        samp_n = float(np.mean([len(b) for b in bk]))
                    an_n = float(sm_e.mean())
                    # One generated basket per trip makes this a noisy sample mean.  With
                    # Var(n) near 84 and only 24 trips its standard error is about 1.9
                    # items; a fixed 25% band spuriously failed 17.5% of repeated checks on
                    # the verified run102 checkpoint.  Use the variance of the quantity
                    # actually sampled, while retaining the historical relative floor.
                    samp_tol = max(0.25 * max(an_n, 1e-9),
                                   2.0 * math.sqrt(float(sm_v.sum())) / len(sm_v))
                m.house, m.ctx = hh, ctx
            except Exception as exc:                       # never let a check kill a run
                an_n = float("nan")
                samp_tol = float("nan")
                log(f"  sampler check failed: {exc}")
            g_ok = dict((
                ("logZ-converged", abs(gap) < a.lz_gap),
                ("E[n]-converged", abs(ho_e8 - ho_e) <= 0.10 * max(ho_e, 1e-9)),
                ("E[n]-calibrated", abs(ho_e - vobs.mean()) <= 0.25 * max(vobs.mean(), 1e-9)),
                ("var-calibrated", abs(ho_v - vobs.var()) <= 0.40 * max(vobs.var(), 1e-9)),
                ("sampler-agrees", abs(samp_n - an_n) <= samp_tol),
                ("elasticity", abs((np.mean(el_hist[-a.eval_every:]) if el_hist else 0)
                                   - a.elast_target) <= 0.30 * abs(a.elast_target)),
                ("data-kept", n_drop < 0.02 * it * a.batch),
            ))
            log(f"  GOALS  {check_goals(g_ok)}   sampled {samp_n:.1f} vs analytic {an_n:.1f} "
                f"(tol {samp_tol:.1f})")
            # The gap is measured on ONE batch of 24 trips at one seed, so it is a noisy
            # statistic, and aborting on a single crossing is a one-sample test.  run60's
            # last twelve readings were
            #   0.322 0.206 0.563 0.357 0.661 0.558 0.426 0.250 0.433 0.155 0.805 1.086
            # -- noise around ~0.45 with no trend, where a genuine runaway looks like run59's
            # 0.198 0.404 0.359 0.753 0.843 1.087.  run57 likewise spiked to 0.846 and fell
            # straight back to 0.003.  Requiring consecutive violations separates the two
            # without weakening the level: a real collapse is monotone and trips every
            # checkpoint, while noise does not repeat.
            lz_strikes = lz_strikes + 1 if gap > a.lz_gap else 0
            if lz_strikes >= a.lz_strikes:
                log(f"  ABORT: the normaliser is not converged at the training draw count "
                    f"({gap:+.3f} > {a.lz_gap}). Any likelihood gain from here is the "
                    f"estimate collapsing, not the model improving.")
                return
            save_ckpt(os.path.join(OUT, f"v3_{a.label}.pt"), m, opt, sched, it,
                      rng, gen, best_vb, best_it, lz_strikes, cum_iter=cum_it)
            # Keep the BEST checkpoint, not just the last.  The file above is overwritten
            # every eval, so an aborted run leaves whatever the final passing eval held --
            # run37 happened to stop near its best by luck, not design.
            if vb > best_vb:
                best_vb, best_it = vb, it
                save_ckpt(os.path.join(OUT, f"v3_{a.label}_best.pt"), m, opt, sched,
                          it, rng, gen, best_vb, best_it, lz_strikes,
                          cum_iter=cum_it)
                json.dump(dict(iter=it, set_per_basket=vb, epoch=ep),
                          open(os.path.join(OUT, f"v3_{a.label}_best.json"), "w"), indent=2)
            # Also keep the best-RANKING checkpoint.  Held-out likelihood and held-out MRR
            # move in OPPOSITE directions here: run86 (exact normaliser) peaked at MRR
            # 0.0676 by iteration 600 and fell to 0.0615 by 1000 while set/basket improved
            # -44.65 -> -43.59; run84 did the same under a biased one, so it is a property
            # of the objective, not of the estimator.  Ranking is scale-free -- log Z
            # cancels within a trip -- so MRR measures relative b while the likelihood
            # measures absolute b plus the normaliser, and the fit trades the first for the
            # second.  Selecting on likelihood alone therefore ships a model well past its
            # ranking peak, which is the wrong choice for recommendation and coupons.
            if rec_mrr > best_mrr:
                best_mrr, best_mrr_it = rec_mrr, it
                save_ckpt(os.path.join(OUT, f"v3_{a.label}_bestmrr.pt"), m, opt, sched,
                          it, rng, gen, best_vb, best_it, lz_strikes, cum_iter=cum_it)
                json.dump(dict(iter=it, mrr=rec_mrr, set_per_basket=vb, epoch=ep),
                          open(os.path.join(OUT, f"v3_{a.label}_bestmrr.json"), "w"),
                          indent=2)
    if a.qmc_n > 0 and _qmc_eval_n > a.qmc_n:
        _set_qmc_rule(_qmc_eval_n, a.qmc_seed + 2_000_003)
    vb, vl, vu, vt, vsz, vco = evaluate(
        m, B, va[:a.n_val], a.draws * 4, gen, use_units=a.units,
        return_decomposition=True)
    log(f"final  set/basket {vb:.4f}  set/line {vl:.4f}  size/basket {vsz:.4f}  "
        f"comp/basket {vco:.4f}  units/basket {vu:.4f}  total/basket {vt:.4f}")
    save_ckpt(os.path.join(OUT, f"v3_{a.label}.pt"), m, opt, sched, a.iters,
              rng, gen, best_vb, best_it, lz_strikes,
              cum_iter=cum_base + a.iters - it0 + a.cum_offset)
    json.dump(dict(set_per_basket=vb, set_per_line=vl, units_per_basket=vu,
                   total_per_basket=vt, n_par=npar, iters=a.iters, config=cfg),
              open(os.path.join(OUT, f"v3_{a.label}.json"), "w"), indent=2)
    log(f"wrote out/v3_{a.label}.pt")
    if best_it >= 0:
        log(f"best checkpoint: iteration {best_it}, set/basket {best_vb:.4f} "
            f"-> out/v3_{a.label}_best.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="run1")
    p.add_argument("--require-version4", type=int, default=0,
                   help="fail unless the run is a fresh, affinity-280, full-catalogue, "
                        "rank>=32, complete-support original version-4 QMC experiment")
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--Kz", type=int, default=12)
    p.add_argument("--Kp", type=int, default=8)
    p.add_argument("--nmax", type=int, default=60)
    p.add_argument("--R", type=int, default=4)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--lam-lr-scale", type=float, default=1.0,
                   help="product-intercept LR divided by the structural LR; 0 freezes "
                        "the full-training popularity initialization")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--n-val", type=int, default=768)
    p.add_argument("--eval-initial", type=int, default=1,
                   help="score iteration zero/resume point before the timing probe")
    p.add_argument("--probe", type=int, default=25)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--warm-start", default="",
                   help="load fitted weights but start a new optimiser/stage at iteration 0")
    p.add_argument("--reinit-interactions", type=int, default=0,
                   help="with --warm-start, restore fresh phi and reset rho_c to zero")
    p.add_argument("--interaction-stage", type=int, default=0,
                   help="train only phi, rho_c, and rho_0; freeze a mature warm-start backbone")
    p.add_argument("--composition-stage", type=int, default=0,
                   help="optimize conditional basket composition with size and units frozen")
    p.add_argument("--factored-size", type=int, default=0,
                   help="RETIRED model-changing ablation: exact P(S||S|,x) times empirical "
                        "P(|S|); not the version-4 joint law")
    p.add_argument("--allow-factored-ablation", type=int, default=0,
                   help="explicit acknowledgement required to reproduce the retired "
                        "model-changing --factored-size ablation")
    p.add_argument("--size-stage", type=int, default=0,
                   help="train rho_0 only, preserving composition and recommendation scores")
    p.add_argument("--multinomial-utility-start", default="",
                   help="gauge-map shared non-price utility blocks from this checkpoint")
    p.add_argument("--reinit-rho0-after-warm", type=int, default=0,
                   help="reinitialize size potential after changing warm-start utilities")
    p.add_argument("--size-ipf-steps", type=int, default=0,
                   help="deterministic aggregate-size IPF updates before optimization")
    p.add_argument("--size-ipf-trips", type=int, default=256,
                   help="fixed training contexts used by pre-optimization size IPF")
    p.add_argument("--size-ipf-damp", type=float, default=0.5,
                   help="damping for each log-ratio size IPF update")
    p.add_argument("--fresh-sched", type=int, default=0)
    p.add_argument("--init-rho0", type=int, default=1)
    p.add_argument("--init-popularity", type=int, default=0,
                   help="initialize product intercepts from training incidence/exposure")
    p.add_argument("--taste-init", type=float, default=0.3,
                   help="standard deviation of initial alpha/theta factors")
    p.add_argument("--cosine", type=int, default=1)
    p.add_argument("--lr-floor", type=float, default=0.02)
    p.add_argument("--lr-milestones", default="",
                   help="relative update counts for staged LR drops; takes precedence over cosine")
    p.add_argument("--lr-gamma", type=float, default=0.5,
                   help="multiplicative LR drop at each --lr-milestones boundary")
    p.add_argument("--units", type=int, default=1)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--phi-max", type=float, default=1.20)   # 0.25 collapses ESS   # 0.35 measures lambda_max 0.67
    p.add_argument("--phi-step-scale", type=float, default=1.0,
                   help="multiply phi's actual Adam update while retaining Adam moments")
    p.add_argument("--size-kl", type=float, default=1.0)
    p.add_argument("--var-w", type=float, default=0.0)
    # Reverse KL on the size law: mode-seeking, so it removes model mass where the data
    # has none.  The forward cross-entropy (--size-kl) cannot see that failure at all.
    p.add_argument("--rkl-w", type=float, default=0.0)
    # Per-trip E[n] calibration.  The size-law penalties above all act on the batch
    # MEAN; the trips that break the normaliser are a minority with runaway per-trip E[n].
    p.add_argument("--en-w", type=float, default=0.0)
    # rho_0 curvature floor: Var(n|trip) ~ 1/rho_0'', and Var is the amplifier in
    # Proposition 1 (dE[n]/dDelta = Var(n)).  c=0.09 caps Var at ~11.
    p.add_argument("--rho0-curv", type=float, default=0.0)
    # Recency is built from purchase history, so on a TEMPORAL split (train weeks 1-82,
    # valid 83-90, test 91-102) its distribution drifts: the "never bought before"
    # indicator falls 74.7% -> 51.5%, and log1p(since)/log(100) leaves the training support
    # (max 1.000 -> 1.399).  Measured, that moves mean b by +0.337 -- 104% of the whole
    # held-out shift -- which Proposition 1 turns into 0.337*Var(n) ~ 22 extra items.
    p.add_argument("--no-rec", type=int, default=0)
    p.add_argument("--lam-sd-max", type=float, default=0.0)
    # c <= sqrt(Kz) keeps the latent mass inside the prior's typical set, where the
    # mode-centred proposal can actually reach it.  -1 means auto = sqrt(Kz).
    p.add_argument("--c-max", type=float, default=0.0)
    # Smolyak level for the deterministic normaliser.  0 = keep importance sampling.
    p.add_argument("--quad-q", type=int, default=0)
    p.add_argument("--qmc-n", type=int, default=0)
    p.add_argument("--qmc-seed", type=int, default=0)
    p.add_argument("--qmc-reps", type=int, default=4,
                   help="independent fixed Sobol scrambles within --qmc-n total nodes")
    p.add_argument("--qmc-refresh-every", type=int, default=0,
                   help="refresh randomized Sobol scrambles every N training updates; "
                        "0 retains common random numbers")
    p.add_argument("--qmc-eval-n", type=int, default=0,
                   help="fixed high-fidelity Sobol nodes for initial/checkpoint/final eval; "
                        "0 uses --qmc-n")
    p.add_argument("--qmc-step-se", type=float, default=0.0,
                   help="skip an update when any per-trip RQMC log-Z SE exceeds this; "
                        "0 disables the per-step safety gate")
    p.add_argument("--qmc-retry-n", type=int, default=0,
                   help="retry a high-SE QMC batch with this many independent nodes; "
                        "0 disables deterministic hard-batch refinement")
    p.add_argument("--qmc-en-max", type=float, default=0.0,
                   help="skip an update when batch E[n] exceeds this multiple of the "
                        "empirical mean; 0 disables the per-step safety gate")
    p.add_argument("--quad-probe", type=int, default=0,
                   help="-1 uses unit scales in the Phi frame; 0 probes all Phi directions; "
                        "n probes the top n")
    p.add_argument("--quad-steps", type=int, default=2)
    p.add_argument("--quad-chunk", type=int, default=32,
                   help="QMC nodes per autograd block; 32 is exact to roundoff versus 8 "
                        "and avoids checkpoint recomputation at the default qmc-n=32")
    p.add_argument("--qmc-size-bands", type=int, default=0,
                   help="screen coarse basket-size bands for remote latent modes; 1 enables")
    p.add_argument("--qmc-size-steps", type=int, default=2,
                   help="vectorised fixed-point steps for the size-band mode screen")
    p.add_argument("--qmc-mode-logtol", type=float, default=8.0,
                   help="retain a separated second mode when its score is within this many nats")
    p.add_argument("--qmc-mode-sep", type=float, default=1.0,
                   help="minimum Euclidean separation for a distinct proposal mode")
    p.add_argument("--qmc-mix-n", type=int, default=0,
                   help="total nodes for a two-mode mixture; 0 uses 2*--qmc-n")
    p.add_argument("--lam-centre", type=int, default=0)
    p.add_argument("--rkl-eps", type=float, default=1e-4)
    p.add_argument("--var-target", type=float, default=-1.0)
    p.add_argument("--var-damp", type=float, default=0.15)
    p.add_argument("--proj-ema", type=int, default=1)
    p.add_argument("--var-project", type=int, default=0)
    p.add_argument("--cum-offset", type=int, default=0)
    p.add_argument("--xi-shrink", type=float, default=0.0)
    p.add_argument("--pool-beta", type=float, default=0.0)
    p.add_argument("--phi-deg", default="")
    p.add_argument("--phi-op-max", type=float, default=0.0,
                   help="cap lambda_max(Phi'Phi); 0 disables the operator-norm projection")
    p.add_argument("--phi-deg-cap", type=float, default=2.5)
    p.add_argument("--gap-project", type=float, default=0.0)
    p.add_argument("--ctx-shrink", type=float, default=1.0)
    p.add_argument("--pool-ctx", type=float, default=0.0)
    # (wd/2)*J*W = 5e-6*5455*53 = 1.45: the same ridge lam pays, per offset entry.
    p.add_argument("--pool-prod", type=float, default=0.0)
    p.add_argument("--beta-cal-w", type=float, default=0.0)
    p.add_argument("--beta-target", default="../../basket_input/v3_beta_target.npz")
    p.add_argument("--n-rec", type=int, default=192)
    p.add_argument("--elast-w", type=float, default=20.0)
    p.add_argument("--elast-target", type=float, default=-0.121)
    p.add_argument("--phi-init", type=float, default=0.03)
    p.add_argument("--phi-init-file", default="",
                   help="npy [J, Kz] spectral initialisation from build by "
                        "phi_spectral_init.py. phi=0 is a saddle (dE/dphi_j = sum_k phi_k "
                        "= 0 there), so SGD escapes only exponentially: measured 0.03 -> "
                        "0.099 over 2,600 updates against the ~0.93 the data implies. "
                        "Placing phi at the rank-Kz factorisation of the empirical log-lift "
                        "beat 15,000 updates of SGD by +0.022 nats on held-out data.")
    p.add_argument("--phi-topk", type=float, default=0.0)
    p.add_argument("--phi-mask", default="")
    p.add_argument("--ess-floor", type=float, default=0.30)
    p.add_argument("--ess-floor-min", type=float, default=0.15)
    p.add_argument("--min-keep", type=float, default=0.5)
    p.add_argument("--lam-target", type=float, default=0.85)
    p.add_argument("--lam-floor", type=float, default=0.0,
                   help="scale phi UP toward this lambda_max; 0 disables. "
                        "The estimator is valid below ~1.0")
    p.add_argument("--lam-up-max", type=float, default=1.15,
                   help="max per-eval upscaling of phi, so it approaches "
                        "the frontier gradually rather than in one jump")
    p.add_argument("--lam-q", type=float, default=0.90)
    p.add_argument("--phi-l1", type=float, default=0.0,
                   help="row sparsity penalty; keep 0 to retain all catalogue products")
    p.add_argument("--phi-centre", type=int, default=0,
                   help="DEPRECATED: not a gauge. Proposition 2 -- centering adds "
                        "-(n-1) m'sum_j phi_j, which depends on basket COMPOSITION "
                        "and cannot be absorbed into rho_0, so it changes the law. "
                        "Measured -5.05e-03 nats against a 6.27e-07 orthogonal-rotation "
                        "control on run403. Default off; only a right orthogonal "
                        "rotation Phi -> Phi Q is a true gauge.")
    p.add_argument("--phi-whiten", type=float, default=0.0,
                   help="DEPRECATED: not a gauge. Corollary 1 -- altering the singular "
                        "values of Phi changes the Gram matrix W and hence the law, even "
                        "at fixed Frobenius norm. Measured +1.10e-05 nats. Declare it as "
                        "a regulariser if wanted; it is not estimator hygiene.")
    p.add_argument("--adapt-draws", type=int, default=1)
    p.add_argument("--lz-gap", type=float, default=1.0)
    p.add_argument("--lz-strikes", type=int, default=1)
    p.add_argument("--rho-c-floor", type=float, default=-1.5)
    p.add_argument("--mix-lam", type=float, default=1.0)
    p.add_argument("--aniso", type=float, default=2.0)
    p.add_argument("--antithetic", type=int, default=0)
    p.add_argument("--lam-project", type=int, default=1)
    p.add_argument("--pi-project-every", type=int, default=1,
                   help="refresh exact incidence weights for the phi budget every N updates; "
                        "0 disables the nonbinding global budget")
    p.add_argument("--pseudo", type=int, default=0)
    p.add_argument("--metrics-jsonl", default="",
                   help="append one JSON record per eval here, for later analysis")
    p.add_argument("--poly-degree-tol", type=float, default=1e-3,
                   help="max tolerated |d log Z| when auto-calibrating the degree")
    p.add_argument("--poly-degree", type=int, default=0,
                   help="per-row polynomial degree cap (0 = --R). 32 is "
                        "non-binding: largest observed single-row count is 26")
    p.add_argument("--elast-every", type=int, default=20,
                   help="under --objective composite, apply the elasticity "
                        "constraint every N steps via one exact pass")
    p.add_argument("--objective", default="full", choices=("full", "composite"),
                   help="composite = normaliser-free leave-one-out conditional "
                        "(36x faster/step); rho_0 must be frozen")
    p.add_argument("--price-soft", type=int, default=0,
                   help="unconstrained price bilinear + hinge penalty instead "
                        "of the softplus hard constraint")
    p.add_argument("--price-hinge-w", type=float, default=10.0)
    p.add_argument("--price-ref", choices=("trip", "category", "subcommodity"),
                   default="trip",
                   help="What dlp is measured against: the whole assortment (trip) or the "
                        "store's own category (category).  Substitution is only expressible "
                        "under 'category' -- see Batcher.make.")
    p.add_argument("--price-soft-floor", type=float, default=0.02,
                   help="Floor on the EFFECTIVE price coefficients when --price-soft "
                        "converts them.  Repairs a checkpoint that arrived saturated, so "
                        "stage 2 does not depend on how stage 1 was run.  0 disables.  "
                        "Measured on a stage-1 checkpoint with gamma min -5.876: 0 and "
                        "0.01 both diverged at the same iteration, 0.02 survived.")
    p.add_argument("--gamma-floor", type=float, default=-3.0,
                   help="Floor on the pre-softplus price parameters while the block is "
                        "CONSTRAINED.  softplus(-3) = 0.0486 with slope 0.047, so the "
                        "block stays responsive; without it gamma ratchets to -5.9 and "
                        "unconstraining it in stage 2 diverges.  Ignored under "
                        "--price-soft, where gamma IS the coefficient.")
    p.add_argument("--kappa-init", type=float, default=0.0,
                   help="Set softplus(price_kappa) to this value after the checkpoint "
                        "load.  0 leaves it alone.  kappa moves ~1.4 units per 1,000 "
                        "iterations, so its initial value effectively fixes it.")
    p.add_argument("--kappa-lr-scale", type=float, default=20.0,
                   help="Adam rate for price_kappa relative to --lr.  kappa ~ 40 while "
                        "gamma ~ 0.02, a 400x spread that one rate cannot serve.")
    p.add_argument("--price-lr-scale", type=float, default=0.05,
                   help="Adam rate for gamma/beta relative to --lr when --price-soft "
                        "is on.  Unconstrained, these ARE the price coefficients "
                        "(~0.02), so the structural rate moves them 10%% per step; "
                        "0.05 keeps the effective step near the constrained one "
                        "(softplus' = 0.0195) while leaving the block free to move.")
    p.add_argument("--phi-pool", default="sum", choices=("sum", "mean"))
    p.add_argument("--size-bands", type=int, default=3)
    p.add_argument("--comp-ce-w", type=float, default=0.0,
                   help="weight on the model's own leave-one-out conditional "
                        "composition cross-entropy (composite likelihood)")
    p.add_argument("--freeze-rho0", type=int, default=0)
    p.add_argument("--cd", type=int, default=0)
    p.add_argument("--cd-draws", type=int, default=0)
    p.add_argument("--neg-per-trip", type=int, default=64)
    p.add_argument("--freeze-rho-c", type=int, default=0)
    p.add_argument("--zero-phi", type=int, default=0,
                   help="zero and freeze phi after loading a checkpoint (nested baseline)")
    p.add_argument("--zero-rho-c", type=int, default=0,
                   help="zero and freeze rho_c after loading a checkpoint (nested baseline)")
    p.add_argument("--rho-c-step-scale", type=float, default=1.0,
                   help="multiply rho_c's actual Adam update; 0 freezes its learned value")
    p.add_argument("--mix-scales-lo", type=float, default=1.0)
    p.add_argument("--mix-scales-hi", type=float, default=2.0)
    p.add_argument("--budget-f", type=float, default=1.0)
    p.add_argument("--mode-steps", type=int, default=1)
    p.add_argument("--clip", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
