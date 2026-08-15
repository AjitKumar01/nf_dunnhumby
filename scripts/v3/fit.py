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
from ragged import RaggedIndex, RaggedModel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[fit] {m}", flush=True)


class Batcher:
    """Builds the ragged index and the per-slot features for a set of trips."""

    def __init__(self, D, F, nmax):
        self.D, self.F, self.nmax = D, F, nmax
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
        ctx = dict(dlp=dlp.double(), disp=disp.double(), mail=mail.double(),
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
        lctx = dict(dlp=dlp_l.double(), disp=disp_l.double(), mail=mail_l.double(),
                    week=(week[LT] - 1) % 52, store=store[LT],
                    rec=self.F.recency(LI, user[LT], day[LT]))
        house = torch.as_tensor(D["trip_user"][trips], dtype=torch.long)
        return (ix, ctx, lctx, house,
                LI, LT, torch.as_tensor(np.concatenate(lc), dtype=torch.long),
                torch.as_tensor(np.concatenate(lu), dtype=torch.long))


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
    G = torch.exp(-m.rho_c[ix.row_cat].unsqueeze(-1) * r * (r - 1) / 2.0).unsqueeze(0) * e
    Gp = torch.zeros(1, ix.B * ix.Cpad, m.R + 1, dtype=w.dtype)
    Gp[:, :, 0] = 1.0
    Gp = Gp.index_copy(1, ix.flat_slot, G).view(1, ix.B, ix.Cpad, m.R + 1)
    A = Gp[:, :, 0, :]
    for c in range(1, ix.Cpad):
        A = poly_mul_trunc(A, Gp[:, :, c, :], m.nmax)
    n_ax = torch.arange(A.shape[-1], dtype=w.dtype)
    return (torch.log(A.clamp_min(1e-300)) + n_ax * M.unsqueeze(-1))[0].mean(0)


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
        return blob, [k for k in missing if k != "cat_of"]
    missing, _ = m.load_state_dict(blob, strict=False)
    return None, [k for k in missing if k != "cat_of"]


def evaluate(m, B, trips, draws, gen, chunk=48, use_units=True):
    """Returns (set per basket, set per line, units per basket, total per basket).

    The SET component is reported apart from the units component because the baselines
    model sets only; quoting a total against them would be comparing different objects."""
    tot_s, tot_u, n_b, n_l = 0.0, 0.0, 0, 0
    for k in range(0, len(trips), chunk):
        sub = trips[k:k + chunk]
        ix, ctx, lctx, hh, li, lt, lc, lq = B.make(sub)
        m.house, m.ctx = hh, ctx
        with torch.no_grad():
            ll = m.loglik(ix, li, lt, lc, n_draws=draws, generator=gen, line_ctx=lctx)
            tot_s += float(ll.sum())
            if use_units:
                tot_u += float(m.units_loglik(li, lt, lq, lctx, ix.B).sum())
        n_b += len(sub)
        n_l += len(li)
    return tot_s / n_b, tot_s / n_l, tot_u / n_b, (tot_s + tot_u) / n_b


def main(a):
    # Subnormal arithmetic runs one to two orders of magnitude slower on CPU, and the ESP
    # coefficients underflow into that range as soon as the mode iteration wanders.
    torch.set_flush_denormal(True)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(a.seed)
    D = build()
    J, N, C, S = int(D["n_item"]), int(D["n_user"]), int(D["n_cat"]), int(D["n_store"])
    F = Features(J, S, 712)
    B = Batcher(D, F, a.nmax)

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
    log(f"{len(tr):,} training trips, {len(va):,} validation")
    # Record the configuration.  Recovering whether run9 used cosine decay meant grepping a
    # session transcript for the launch command, because neither the log nor the checkpoint
    # carried it -- a comparison between runs is not checkable if the runs are not labelled.
    cfg = {k: v for k, v in sorted(vars(a).items())}
    log("config: " + "  ".join(f"{k}={v}" for k, v in cfg.items()))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"v3_{a.label}.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)

    m = RaggedModel(J=J, N=N, C=C, K=a.K, Kz=a.Kz, nmax=a.nmax, R=a.R, seed=a.seed,
                    S=S, Kp=a.Kp, phi_init=a.phi_init)
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

    if a.init_rho0:
        # Initialise the size potential at the empirical basket-size law.
        #
        # P(n | z) is proportional to exp(-rho_0(n)) A_n(z), so setting
        # rho_0(n) = log A_n(0) - log target(n) makes the size law equal `target` at z = 0.
        # The BEMB-style multinomial baseline is exactly this model with phi = 0,
        # rho_c = 0 and rho_0 set that way, and it is HANDED the empirical law -- while
        # this model was being made to rediscover it from a zero initialisation, through
        # the normaliser, on a Monte Carlo gradient.  That is a large share of the
        # optimisation spent recovering something computable in closed form from one pass
        # over the training data.
        n_tr = D["trip_nlines"][tr]
        cnt = np.bincount(np.clip(n_tr, 0, a.nmax), minlength=a.nmax + 1) + 0.5
        tgt = torch.log(torch.as_tensor(cnt / cnt.sum()))
        with torch.no_grad():
            sub = tr[np.random.default_rng(0).choice(len(tr), size=64, replace=False)]
            ix0, ctx0, lctx0, hh0, *_ = B.make(sub)
            m.house, m.ctx = hh0, ctx0
            zz = torch.zeros(ix0.B, 1, m.Kz, dtype=torch.float64)
            from ragged import log_f_ragged           # noqa: F401  (kept local)
            lg = _size_coeffs(m, zz, ix0)             # [nmax+1], mean log A_n at z = 0
            r0 = lg[: a.nmax + 1] - tgt
            r0 = r0 - r0[0]                            # rho_0(0) = 0 fixes the scale
            m.rho_0_free.copy_(r0[1:])
        log(f"rho_0 initialised at the empirical size law "
            f"(mean {float((cnt/cnt.sum() * np.arange(a.nmax+1)).sum()):.2f} lines)")
    resume_blob = None
    if a.resume:
        resume_blob, _miss = load_ckpt(a.resume, m)
        if _miss:
            raise SystemExit(f"resume checkpoint is missing fitted parameters: {_miss}")
        if resume_blob is None:
            log(f"resumed WEIGHTS ONLY from {os.path.basename(a.resume)} "
                f"(pre-format-2 checkpoint: no optimiser, schedule or RNG state to restore)")
        else:
            log(f"resumed from {os.path.basename(a.resume)} at iteration "
                f"{resume_blob['iter']} -- optimiser, schedule and RNG restored")
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
    opt = torch.optim.Adam([p_ for p_ in m.parameters() if p_.requires_grad],
                           lr=a.lr, weight_decay=a.wd)
    # Cosine decay.  The previous run's training loss stopped falling at about a third of
    # an epoch and then oscillated, with held-out likelihood swinging over 1.4 nats and no
    # trend -- the signature of a step size too large for the stage, not of a model at
    # capacity.  A constant learning rate was the only schedule used until now.
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters,
                                                        eta_min=a.lr * a.lr_floor)
             if a.cosine else None)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    it0 = 0                                   # iterations already done, for a continuation
    cum_base = 0                              # ... including lineages before this one
    if resume_blob is not None:
        opt.load_state_dict(resume_blob["opt"])
        if sched is not None and resume_blob.get("sched") is not None:
            sched.load_state_dict(resume_blob["sched"])
        rng.bit_generator.state = resume_blob["rng_np"]
        gen.set_state(resume_blob["rng_torch"])
        it0 = int(resume_blob["iter"])
        cum_base = int(resume_blob.get("cum_iter", it0))
        log(f"  continuing from iteration {it0}: lr {opt.param_groups[0]['lr']:.6f} "
            f"(a fresh schedule would start at {a.lr:.6f}), "
            f"Adam state for {len(opt.state)} tensors")
        if it0 >= a.iters:
            raise SystemExit(f"--iters {a.iters} is at or below the resumed iteration "
                             f"{it0}; nothing left to run")

    log("")
    log(f"timing probe: {a.probe} iterations at batch {a.batch}, {a.draws} draws")
    t0 = time.time()
    hist, ess_hist, emin_hist, en_hist, enmax_hist, ce_hist = [], [], [], [], [], []
    el_hist = []
    n_skip = n_drop = n_redo = 0
    # Carry the best-so-far across a resume.  Starting it at -inf would let the first eval
    # of the continuation overwrite v3_<label>_best.pt with a WORSE model purely because it
    # is the first one this process has seen.
    e_ema = v_ema = None
    n_bang = 0
    best_vb, best_it = -1e18, -1
    if resume_blob is not None and resume_blob.get("best_vb") is not None:
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
    m.project(a.phi_max)
    lz_strikes = int(resume_blob['lz_strikes']) if resume_blob else 0
    phi_mask = None
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
    for it in range(it0 + 1, a.iters + 1):
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
            if a.units:
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
            if a.units:
                ll = ll + m.units_loglik(li, lt, lq, lctx, ix.B)
        else:
            ll, ess, pn = m.loglik(ix, li, lt, lc, n_draws=_nd, generator=gen,
                           return_ess=True, return_size=True, line_ctx=lctx,
                           mode_steps=a.mode_steps, mix_scales=_mix, aniso=a.aniso,
                                 antithetic=a.antithetic > 0,
                           units=lq if a.units else None)
        loss = -ll.mean()
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
        if a.size_kl > 0:
            pbar = pn.mean(0).clamp_min(1e-12)
            pbar = pbar / pbar.sum()
            ce = -(emp_pn[: pbar.shape[0]] * pbar.log()).sum()
            nax = torch.arange(1, pn.shape[1] + 1, dtype=pn.dtype)
            e_tr = (pn * nax).sum(1)
            v_tr = (pn * nax ** 2).sum(1) - e_tr ** 2
            ce = ce + a.var_w * (torch.log1p(v_tr.mean()) - math.log1p(emp_var)) ** 2
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
        elast = None
        if a.elast_w > 0:
            gb = (softplus(m.gamma[hh][ix.item_trip])
                  * softplus(m.beta[ix.item])).sum(-1).mean()
            nax = torch.arange(1, pn.shape[1] + 1, dtype=pn.dtype)
            e_b = (pn * nax).sum(1)
            v_b = (pn * nax ** 2).sum(1) - e_b ** 2
            elast = -(gb * v_b.mean() / e_b.mean().clamp_min(1e-6))
            loss = loss + a.elast_w * (elast - a.elast_target) ** 2
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
        e_bar, e_min = float(ess.mean()), float(ess.min())
        if a.adapt_draws > 1:
            with torch.no_grad():
                bad = (ess < a.ess_floor_min).nonzero().flatten()
            if bad.numel() > 0:
                ll2, ess2 = m.loglik(ix, li, lt, lc, n_draws=a.draws * a.adapt_draws,
                                     generator=gen, return_ess=True, line_ctx=lctx,
                                     mode_steps=a.mode_steps, mix_scales=_mix, aniso=a.aniso,
                                 antithetic=a.antithetic > 0,
                                     units=lq if a.units else None)
                ll = torch.where((ess < a.ess_floor_min), ll2, ll)
                ess = torch.where((ess < a.ess_floor_min), ess2, ess)
                n_redo += int(bad.numel())
        keep = ess >= a.ess_floor_min
        n_drop += int((~keep).sum())
        if int(keep.sum()) < max(2, int(a.min_keep * a.batch)) or e_bar < a.ess_floor:
            n_skip += 1
            opt.zero_grad()
        else:
            loss = -ll[keep].mean()
            if ce is not None:
                loss = loss + a.size_kl * ce
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip)
            opt.step()
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
            with torch.no_grad():
                pass
            pslot = m.pi_exact(ix)
            with torch.no_grad():
                w_new = torch.zeros(m.phi.shape[0], dtype=pslot.dtype)
                cntj = torch.zeros_like(w_new)
                w_new.index_add_(0, ix.item, pslot * (1 - pslot))
                cntj.index_add_(0, ix.item, torch.ones_like(pslot))
                seen = cntj > 0
                w_new[seen] /= cntj[seen]
                prev = getattr(m, "_pi_w", None)
                m._pi_w = w_new if prev is None else torch.where(seen, 0.5 * prev + 0.5 * w_new, prev)
            cap = a.phi_max
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
            if a.freeze_rho_c:
                # Ablation: hold rho_c at zero so the affinity partition contributes no
                # attraction.  If phi still climbs, the pressure is not coming from the
                # category term.
                with torch.no_grad():
                    m.rho_c.zero_()
            else:
                m.project_rho_c(a.rho_c_floor)
            m.project(cap, budget=budget, thresh=a.phi_l1 * lr_now,
                      centre=a.phi_centre > 0, whiten=a.phi_whiten)
            # Sparsity is what makes log Z estimable, and it is the FRACTION of products
            # carrying phi that matters, not the norm.  Measured on run55's checkpoint at
            # ||phi|| = 0.96, the gap between 16 draws and 4096 was:
            #     1% of products   0.000 nats      20% of products   23.9 nats
            #     5% of products   1.888 nats     100% of products  128.4 nats
            # so a dense phi is unfittable at any affordable draw count while a sparse one
            # is exact at the cheapest.  A hard top-k keeps that fraction where the probe
            # measured it; an L1 penalty controls it only indirectly and drifts as phi grows.
            if 0.0 < a.phi_topk < 1.0:
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
            if phi_mask is not None:
                with torch.no_grad():
                    m.phi.mul_(phi_mask)
            # Var(n) is the quantity every other failure runs through; project it too.
            if a.var_target != 0:
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
                _b = m.project_mean(_em_use, obs_mean, _v_use, damp=a.var_damp)
                # Count clamp hits: if the mean correction is saturating, the controller is
                # in bang-bang and no amount of damping will let it settle.
                if abs(_b) >= 0.5 * a.var_damp - 1e-12:
                    n_bang += 1
            # The data's aggregate elasticity is -0.121 and Proposition 1 gives
            # elasticity = -(gamma.beta) Var(n) / E[n], so the mean sensitivity it implies
            # is 0.121 * E[n] / Var(n).  Projected, not penalised -- see project_price.
            if a.elast_w > 0:
                # Target the elasticity through the model's CURRENT moments, not the
                # empirical ones.  Proposition 1 gives elasticity = -(gamma.beta) Var/E, so
                # gamma.beta = |target| * E/Var -- and using E_obs/Var_obs assumes the size
                # law is already calibrated.  It is not: at iteration 2000 of run26 the
                # model sat at Var/E = 37/17.1 = 2.16 against an empirical 10.6, so a
                # gamma.beta pinned for the empirical ratio delivered -0.040 instead of
                # -0.121.  Reading E and Var off pn each step makes the target self-correct
                # as the size law converges.
                _em = float(_e.mean()) if a.var_target != 0 else obs_mean
                _vm = _v if a.var_target != 0 else emp_var
                m.project_price(abs(a.elast_target) * max(_em, 1e-6) / max(_vm, 1e-6))
        if sched is not None:
            sched.step()
        hist.append(float(loss))
        ess_hist.append(e_bar)
        emin_hist.append(e_min)
        with torch.no_grad():
            _e = (pn * torch.arange(1, pn.shape[1] + 1, dtype=pn.dtype)).sum(1)
            en_hist.append(float(_e.mean()))
            enmax_hist.append(float(_e.max()))
        if ce is not None:
            ce_hist.append(float(ce))
        if elast is not None:
            el_hist.append(float(elast))
        if it == a.probe:
            dt = time.time() - t0
            per = dt / a.probe
            log(f"  {a.probe} iterations in {dt:.1f}s = {per:.3f}s/iteration")
            log(f"  slots per batch {ix.item.numel():,}, rows {ix.n_rows:,}, "
                f"Cpad {ix.Cpad}")
            log(f"  implied wall clock: {per * a.iters / 3600:.2f} h for {a.iters:,} "
                f"iterations")
            log(f"  loss {np.mean(hist[-20:]):.3f}   ESS {np.mean(ess_hist[-20:]):.3f}"
                f"   |phi| {float(m.phi.norm(dim=1).mean()):.3f}   skipped {n_skip}"
                f"   ESS min {np.min(emin_hist):.3f}")
            if a.probe_only:
                json.dump(dict(sec_per_iter=per, iters=a.iters, n_par=npar,
                               slots=int(ix.item.numel()), ess=float(ess.mean())),
                          open(os.path.join(OUT, "v3_probe.json"), "w"), indent=2)
                log("  wrote out/v3_probe.json; stopping (--probe-only)")
                return
        if it % a.eval_every == 0:
            # lambda_max BEFORE evaluate(): evaluate reassigns m.house/m.ctx to the
            # validation chunks, and ix here is the training batch.  Calling it after
            # indexes a 40-trip batch into a 32-trip household tensor.
            lam_max = m.lambda_max(ix)
            lam_seen = lam_max
            # Project on the MEASURED lambda_max, not a proxy for it.
            #
            # The pi-weighted budget averages pi(1-pi) per product across the batch, but a
            # product sits in every trip's assortment and is bought in almost none, so its
            # average is tiny: measured 0.232 against a lambda_max of 2.202, ten times too
            # small, and a budget of 4.2 never bound.  lambda_max is computed per trip at
            # the mode, where the concentration that makes it large survives.  Averaging
            # destroys it -- the fourth time in this file a mean has stood in for a tail.
            if a.lam_project > 0 and lam_max > a.lam_target:
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
                ev_e8, _ = m.size_moments(vix, n_draws=a.draws * 8,
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
            ho_v = float(ev_v.mean() + ev_e.var())
            ho_v_within = float(ev_v.mean())        # kept, so the split stays visible
            ho_e_spread = float(ev_e.var())
            vb, vl, vu, vt = evaluate(m, B, va[:a.n_val], a.draws * 2, gen, use_units=a.units)
            m.house, m.ctx = hh, ctx
            ep = it * a.batch / max(len(tr), 1)
            cum_it = cum_base + (it - it0) + a.cum_offset
            cum_ep = cum_it * a.batch / max(len(tr), 1)
            log(f"  it {it:5d} ep {ep:5.3f} cum {cum_ep:5.3f}  train {np.mean(hist[-a.eval_every:]):8.3f}  "
                f"set/bskt {vb:8.3f}  units/bskt {vu:7.3f}  total {vt:8.3f}  "
                f"ESS {np.mean(ess_hist[-a.eval_every:]):.3f} "
                f"(min {np.min(emin_hist[-a.eval_every:]):.3f})  "
                f"|phi| {float(m.phi.norm(dim=1).mean()):.3f} "
                f"(max {float(m.phi.norm(dim=1).max()):.2f} "
                f"zero {float((m.phi.norm(dim=1) < 1e-8).double().mean()):.0%} "
                f"erank {float((lambda sv: (sv**2).sum()**2/(sv**4).sum())(torch.linalg.svdvals(m.phi.detach()))):.0f})  "
                f"lam_max {lam_max:.3f}  E[n] {ho_e:.1f}/{vobs.mean():.1f} "
                f"[{ho_e8:.1f}@8x] var {ho_v:.0f}/{vobs.var():.0f} "
                f"(w{ho_v_within:.0f}+s{ho_e_spread:.0f})  "
                f"elast {np.mean(el_hist[-a.eval_every:]) if el_hist else float('nan'):+.3f}"
                f"/{a.elast_target:+.3f}  "
                f"skip {n_skip}  drop {n_drop}  redo {n_redo}  bang {n_bang}  "
                f"{(time.time()-t0)/60:.1f} min")
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
            # sampler-vs-analytic consistency: same distribution, two routes.
            samp_n = float("nan")
            try:
                with torch.no_grad():
                    sm_ix, sm_ctx, _, sm_hh, *_ = B.make(va[:24])
                    m.house, m.ctx = sm_hh, sm_ctx
                    bk = m.sample(sm_ix, n_draws=a.draws,
                                  generator=torch.Generator().manual_seed(it))
                    samp_n = float(np.mean([len(b) for b in bk]))
                    sm_e, _ = m.size_moments(sm_ix, n_draws=a.draws,
                                             generator=torch.Generator().manual_seed(it))
                    an_n = float(sm_e.mean())
                m.house, m.ctx = hh, ctx
            except Exception as exc:                       # never let a check kill a run
                an_n = float("nan")
                log(f"  sampler check failed: {exc}")
            g_ok = dict((
                ("logZ-converged", abs(gap) < a.lz_gap),
                ("E[n]-converged", abs(ho_e8 - ho_e) <= 0.10 * max(ho_e, 1e-9)),
                ("E[n]-calibrated", abs(ho_e - vobs.mean()) <= 0.25 * max(vobs.mean(), 1e-9)),
                ("var-calibrated", abs(ho_v - vobs.var()) <= 0.40 * max(vobs.var(), 1e-9)),
                ("sampler-agrees", abs(samp_n - an_n) <= 0.25 * max(an_n, 1e-9)),
                ("elasticity", abs((np.mean(el_hist[-a.eval_every:]) if el_hist else 0)
                                   - a.elast_target) <= 0.30 * abs(a.elast_target)),
                ("data-kept", n_drop < 0.02 * it * a.batch),
            ))
            log(f"  GOALS  {check_goals(g_ok)}   sampled {samp_n:.1f} vs analytic {an_n:.1f}")
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
    vb, vl, vu, vt = evaluate(m, B, va[:a.n_val], a.draws * 4, gen, use_units=a.units)
    log(f"final  set/basket {vb:.4f}  set/line {vl:.4f}  units/basket {vu:.4f}  total/basket {vt:.4f}")
    save_ckpt(os.path.join(OUT, f"v3_{a.label}.pt"), m, opt, sched, a.iters,
              rng, gen, best_vb, best_it, lz_strikes,
              cum_iter=cum_base + a.iters - it0 + a.cum_offset)
    json.dump(dict(set_per_basket=vb, set_per_line=vl, units_per_basket=vu,
                   total_per_basket=vt, n_par=npar, iters=a.iters),
              open(os.path.join(OUT, f"v3_{a.label}.json"), "w"), indent=2)
    log(f"wrote out/v3_{a.label}.pt")
    if best_it >= 0:
        log(f"best checkpoint: iteration {best_it}, set/basket {best_vb:.4f} "
            f"-> out/v3_{a.label}_best.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="run1")
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--Kz", type=int, default=12)
    p.add_argument("--Kp", type=int, default=8)
    p.add_argument("--nmax", type=int, default=60)
    p.add_argument("--R", type=int, default=4)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--n-val", type=int, default=768)
    p.add_argument("--probe", type=int, default=25)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--init-rho0", type=int, default=1)
    p.add_argument("--cosine", type=int, default=1)
    p.add_argument("--lr-floor", type=float, default=0.02)
    p.add_argument("--units", type=int, default=1)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--phi-max", type=float, default=1.20)   # 0.25 collapses ESS   # 0.35 measures lambda_max 0.67
    p.add_argument("--size-kl", type=float, default=1.0)
    p.add_argument("--var-w", type=float, default=0.0)
    p.add_argument("--var-target", type=float, default=-1.0)
    p.add_argument("--var-damp", type=float, default=0.15)
    p.add_argument("--proj-ema", type=int, default=1)
    p.add_argument("--var-project", type=int, default=0)
    p.add_argument("--cum-offset", type=int, default=0)
    p.add_argument("--elast-w", type=float, default=20.0)
    p.add_argument("--elast-target", type=float, default=-0.121)
    p.add_argument("--phi-init", type=float, default=0.03)
    p.add_argument("--phi-topk", type=float, default=0.0)
    p.add_argument("--phi-mask", default="")
    p.add_argument("--ess-floor", type=float, default=0.30)
    p.add_argument("--ess-floor-min", type=float, default=0.15)
    p.add_argument("--min-keep", type=float, default=0.5)
    p.add_argument("--lam-target", type=float, default=0.85)
    p.add_argument("--lam-q", type=float, default=0.90)
    p.add_argument("--phi-l1", type=float, default=2.0)
    p.add_argument("--phi-centre", type=int, default=1)
    p.add_argument("--phi-whiten", type=float, default=0.0)
    p.add_argument("--adapt-draws", type=int, default=1)
    p.add_argument("--lz-gap", type=float, default=1.0)
    p.add_argument("--lz-strikes", type=int, default=1)
    p.add_argument("--rho-c-floor", type=float, default=-1.5)
    p.add_argument("--mix-lam", type=float, default=1.0)
    p.add_argument("--aniso", type=float, default=2.0)
    p.add_argument("--antithetic", type=int, default=0)
    p.add_argument("--lam-project", type=int, default=1)
    p.add_argument("--pseudo", type=int, default=0)
    p.add_argument("--freeze-rho0", type=int, default=0)
    p.add_argument("--cd", type=int, default=0)
    p.add_argument("--cd-draws", type=int, default=0)
    p.add_argument("--neg-per-trip", type=int, default=64)
    p.add_argument("--freeze-rho-c", type=int, default=0)
    p.add_argument("--mix-scales-lo", type=float, default=1.0)
    p.add_argument("--mix-scales-hi", type=float, default=2.0)
    p.add_argument("--budget-f", type=float, default=1.0)
    p.add_argument("--mode-steps", type=int, default=1)
    p.add_argument("--clip", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
