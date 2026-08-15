"""
The ragged kernel: the same model as core.py, without padding the item axis.

WHY THIS EXISTS.  core.py assumes every trip sees C categories of exactly P products.  A
dunnhumby store carries a median of 18 products in a category and up to 225.  Padding every
category to 225 would waste roughly 12x the arithmetic -- the same mistake that cost an
earlier branch 17.8 hours per fit against a projected 1.

WHAT IS RAGGED AND WHAT IS NOT.  Items are kept in one flat array with a row index, so a
category of 3 products costs 3 slots.  The CATEGORY axis is padded to the batch maximum,
because it is short (at most 185 non-empty categories per store), because the convolution
across categories is a sequential scan either way, and because a missing category pads with
the identity polynomial (1, 0, 0, ...) which the convolution absorbs for free.

HOW THE ELEMENTARY SYMMETRIC POLYNOMIALS ARE FORMED.  Not by the O(N R) recursion -- that
needs a per-row loop over a ragged axis.  Instead by POWER SUMS and Newton's identities,

    p_i = sum_j w_j^i          (a scatter-add: no padding, no recursion, no loop over items)

    e_1 = p1
    e_2 = (p1^2 - p2) / 2
    e_3 = (p1^3 - 3 p1 p2 + 2 p3) / 6
    e_4 = (p1^4 - 6 p1^2 p2 + 3 p2^2 + 8 p1 p3 - 6 p4) / 24

WHERE THAT CAN GO WRONG, AND THE GUARD.  Newton's identities subtract quantities of similar
size, so when one weight dominates its row the difference cancels and precision is lost.
`cancellation` returns (p1^2 - p2)/p1^2 per row; below about 1e-8 roughly half the mantissa
has gone.  validate_ragged.py measures this on the real fitted-scale weights rather than
assuming it is safe.

SCALING.  Weights are divided by the largest weight IN THE TRIP -- not in the row -- so
every category shares one scale and the factor comes back as n*log(M) inside the final
log-sum-exp over n.  A per-row scale would not survive the convolution across categories.
"""
import math

import torch
from torch.nn.functional import softplus


def seg_sum(vals, idx, n):
    """Sum `vals` within each segment.  vals [..., T], idx [T] -> [..., n]."""
    out = torch.zeros(vals.shape[:-1] + (n,), dtype=vals.dtype, device=vals.device)
    return out.index_add_(-1, idx, vals)


def seg_max(vals, idx, n):
    out = torch.full(vals.shape[:-1] + (n,), -float("inf"),
                     dtype=vals.dtype, device=vals.device)
    return out.index_reduce_(-1, idx, vals, "amax", include_self=True)


def esp_newton(w, row_of, n_rows, R, row_size=None):
    """e_0..e_R per row from power sums.  w [..., T] -> [..., n_rows, R+1].

    TWO EXACTNESS GUARDS, both of which cost nothing and one of which is not optional.

    An elementary symmetric polynomial of non-negative weights is non-negative, and it is
    exactly ZERO when its degree exceeds the number of items in the row.  Newton's
    identities do not know that: for a single-item category, e_2 = (p1^2 - p2)/2 =
    (w^2 - w^2)/2 lands on -2.8e-17 rather than 0.  A single negative coefficient
    propagates through the 183-step convolution across categories and corrupts A_n, and
    since log takes a clamp afterwards the corruption is silent.  Measured on real
    dunnhumby batches: 349 of ~8,200 rows carried a negative e_r from the first iteration.

    So degrees above the row size are zeroed explicitly, and everything is clamped at zero.
    """
    p = [seg_sum(w ** i, row_of, n_rows) for i in range(1, R + 1)]
    e = [torch.ones_like(p[0])]
    if R >= 1:
        e.append(p[0])
    if R >= 2:
        e.append((p[0] ** 2 - p[1]) / 2)
    if R >= 3:
        e.append((p[0] ** 3 - 3 * p[0] * p[1] + 2 * p[2]) / 6)
    if R >= 4:
        e.append((p[0] ** 4 - 6 * p[0] ** 2 * p[1] + 3 * p[1] ** 2
                  + 8 * p[0] * p[2] - 6 * p[3]) / 24)
    for r in range(5, R + 1):                       # general Newton recursion beyond 4
        acc = torch.zeros_like(p[0])
        for i in range(1, r + 1):
            acc = acc + ((-1) ** (i - 1)) * e[r - i] * p[i - 1]
        e.append(acc / r)
    out = torch.stack(e, dim=-1).clamp_min(0.0)
    if row_size is not None:
        deg = torch.arange(R + 1, device=out.device)
        out = torch.where(deg <= row_size.unsqueeze(-1), out, torch.zeros_like(out))
    return out


def esp_bucketed(w, row_of, n_rows, R, row_size, item_pos, buckets=(8, 32, 96, 256)):
    """e_0..e_R per row by the STABLE O(N R) recursion, without padding everything.

    Newton's identities build e_r from power sums as an alternating sum; at order 12 or 23
    the terms reach 1e30 and cancel catastrophically -- measured log Z errors of 1e62 and
    1e264 against the dense kernel, while the e_2 cancellation diagnostic still read a
    healthy 0.6.  The recursion e_r <- e_r + w_i e_{r-1} has no subtraction at all and is
    unconditionally stable at any R, but it needs a per-row loop over items.

    So rows are BUCKETED by size and padded only to their bucket's maximum.  A padded slot
    carries weight 0, which is invisible to the recursion, so no masking is needed.  With a
    median row of 18 products and a maximum of 225, bucketing to (8, 32, 96, 256) wastes
    about 2x rather than the 25x that padding everything to 225 would cost.
    """
    lead = w.shape[:-1]
    out = torch.zeros(lead + (n_rows, R + 1), dtype=w.dtype, device=w.device)
    out[..., 0] = 1.0
    lo = 0
    for hi in buckets:
        sel_r = (row_size > lo) & (row_size <= hi)
        lo = hi
        if not bool(sel_r.any()):
            continue
        ridx = torch.nonzero(sel_r, as_tuple=True)[0]
        loc = torch.full((n_rows,), -1, dtype=torch.long, device=w.device)
        loc[ridx] = torch.arange(len(ridx), device=w.device)
        sel_i = sel_r[row_of]
        wi = w[..., sel_i]                                          # [..., T_b]
        flat = loc[row_of[sel_i]] * hi + item_pos[sel_i]            # [T_b]
        P = torch.zeros(lead + (len(ridx) * hi,), dtype=w.dtype, device=w.device)
        P = P.index_copy(-1, flat, wi).view(lead + (len(ridx), hi))
        e = [torch.ones(lead + (len(ridx),), dtype=w.dtype, device=w.device)]
        e += [torch.zeros(lead + (len(ridx),), dtype=w.dtype, device=w.device)
              for _ in range(R)]
        for i in range(hi):
            x = P[..., i]
            for r in range(min(R, i + 1), 0, -1):
                e[r] = e[r] + x * e[r - 1]
        out = out.index_copy(-2, ridx, torch.stack(e, dim=-1))
    return out


def cancellation(w, row_of, n_rows):
    """(p1^2 - p2)/p1^2 per row: how much of the leading term survives the subtraction."""
    p1 = seg_sum(w, row_of, n_rows)
    p2 = seg_sum(w ** 2, row_of, n_rows)
    return (p1 ** 2 - p2) / p1.pow(2).clamp_min(1e-300)


def poly_mul_trunc(A, G, nmax):
    """Multiply polynomials along the last axis, truncating at degree nmax."""
    out = torch.zeros(A.shape[:-1] + (nmax + 1,), dtype=A.dtype, device=A.device)
    LA, LG = A.shape[-1], G.shape[-1]
    for r in range(min(LG, nmax + 1)):
        take = min(LA, nmax + 1 - r)
        out[..., r:r + take] = out[..., r:r + take] + A[..., :take] * G[..., r:r + 1]
    return out


def poly_tree(P, nmax):
    """Product of the polynomials along axis -2, truncated at degree nmax.

    The categories of a trip contribute independent factors and the basket-size generating
    polynomial is their product.  Multiplying them one at a time takes Cpad - 1 sequential
    steps -- 182 at a dunnhumby store -- each a separate small kernel launch, and that loop
    measured 38% of a log f evaluation.  Multiplying pairwise in a balanced tree needs
    ceil(log2(Cpad)) = 8 rounds instead, at slightly more arithmetic but far fewer launches:
    261 ms -> 96 ms on a batch of 24, agreeing with the sequential result to 4.3e-15.

    Padding to an even count uses the identity polynomial (1, 0, 0, ...), so no masking is
    needed.  This is the same trick as esp_tree in the multinomial baseline, which was
    written first and should have been applied here at the same time.
    """
    while P.shape[-2] > 1:
        C, d = P.shape[-2], P.shape[-1]
        if C % 2:
            pad = torch.zeros(P.shape[:-2] + (1, d), dtype=P.dtype, device=P.device)
            pad[..., 0] = 1.0
            P = torch.cat([P, pad], dim=-2)
            C += 1
        A, B = P[..., 0::2, :], P[..., 1::2, :]
        nd = min(2 * d - 1, nmax + 1)
        out = torch.zeros(P.shape[:-2] + (C // 2, nd), dtype=P.dtype, device=P.device)
        for k in range(min(d, nd)):
            take = min(d, nd - k)
            out[..., k:k + take] = out[..., k:k + take] + A[..., :take] * B[..., k:k + 1]
        P = out
    return P[..., 0, :]


class RaggedIndex:
    """The per-batch layout.  Everything here is integer bookkeeping, no parameters.

    item      [T]        global product id of every assortment slot in the batch
    row_of    [T]        which (trip, category) row each slot belongs to
    row_trip  [n_rows]   which trip each row belongs to
    row_cat   [n_rows]   which category
    row_pos   [n_rows]   the row's position within its trip, 0..Cpad-1
    """

    def __init__(self, item, row_of, row_trip, row_cat, n_trips, device=None):
        self.item = torch.as_tensor(item, dtype=torch.long, device=device)
        self.row_of = torch.as_tensor(row_of, dtype=torch.long, device=device)
        self.row_trip = torch.as_tensor(row_trip, dtype=torch.long, device=device)
        self.row_cat = torch.as_tensor(row_cat, dtype=torch.long, device=device)
        self.n_rows = len(self.row_trip)
        self.B = n_trips
        # position of each row inside its trip
        order = torch.argsort(self.row_trip, stable=True)
        pos = torch.zeros(self.n_rows, dtype=torch.long, device=device)
        counts = torch.bincount(self.row_trip, minlength=n_trips)
        self.Cpad = int(counts.max())
        run = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                         torch.cumsum(counts, 0)[:-1]])
        pos[order] = (torch.arange(self.n_rows, device=device)
                      - run[self.row_trip[order]])
        self.row_pos = pos
        self.row_size = torch.bincount(self.row_of, minlength=self.n_rows)
        # position of each item inside its row, for the bucketed scatter
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                            torch.cumsum(self.row_size, 0)[:-1]])
        self.item_pos = (torch.arange(len(self.row_of), device=device)
                         - starts[self.row_of])
        # largest row size appearing at each slot position, for the degree cap
        sd = torch.zeros(self.Cpad, dtype=torch.long, device=device)
        sd.index_reduce_(0, self.row_pos, self.row_size, "amax", include_self=True)
        self.slot_deg = sd
        self.item_trip = self.row_trip[self.row_of]
        self.flat_slot = self.row_trip * self.Cpad + self.row_pos


def log_f_ragged(model, z, ix, drop_empty=False, return_terms=False,
                 return_parts=False):
    """log f(z) for a batch.  z [B, D, Kz] -> [B, D].

    drop_empty excludes the n = 0 term, giving f(z) - 1 directly.  That matters: the model
    conditions on a non-empty basket, so the quantity actually needed is Z - 1, and forming
    it as exp(log Z) - 1 subtracts two nearly equal numbers whenever Z is close to 1.  The
    empty basket contributes exactly 1 to f for every z (A_0 = 1, rho_0(0) = 0), so it can
    be dropped from the sum instead of subtracted afterwards -- exact, and stable however
    small Z - 1 becomes."""
    D = z.shape[1]
    phi_i = model.phi[ix.item]                                     # [T, Kz]
    bt = model.b_flat(ix) - 0.5 * (phi_i ** 2).sum(-1)             # [T]
    proj = (z[ix.item_trip] * phi_i.unsqueeze(1)).sum(-1)          # [T, D]
    logw = (bt.unsqueeze(1) + proj).transpose(0, 1)                # [D, T]
    # ONE scale per (trip, draw): a per-row scale would not survive the convolution
    M = seg_max(logw, ix.item_trip, ix.B)                          # [D, B]
    w = torch.exp(logw - M.index_select(-1, ix.item_trip))         # [D, T]
    e = esp_bucketed(w, ix.row_of, ix.n_rows, model.R, ix.row_size, ix.item_pos)
    r = torch.arange(model.R + 1, dtype=w.dtype, device=w.device)
    a = torch.exp(-model.rho_c[ix.row_cat].unsqueeze(-1) * r * (r - 1) / 2.0)
    G = a.unsqueeze(0) * e                                          # [D, n_rows, R+1]
    # scatter rows into [D, B, Cpad, R+1]; missing rows are the identity polynomial
    Gp = torch.zeros(D, ix.B * ix.Cpad, model.R + 1, dtype=w.dtype, device=w.device)
    Gp[:, :, 0] = 1.0
    Gp = Gp.index_copy(1, ix.flat_slot, G).view(D, ix.B, ix.Cpad, model.R + 1)
    # The per-slot degree cap that used to sit here claimed a 4x cut on the grounds that
    # "the median category never exceeds 3 items in any observed basket".  That reasoning
    # was wrong -- the normaliser sums over every POSSIBLE basket, so the bound is the
    # assortment's category size (up to 225 products), not the observed composition.
    # Measured, it cut 4,392 coefficients to 3,476: 1.26x, not 4x, capping 63 of 183 slots.
    # The tree supersedes it and is worth 2.7x on its own.
    A = poly_tree(Gp, model.nmax)
    n = torch.arange(A.shape[-1], dtype=w.dtype, device=w.device)
    lg = (torch.log(A.clamp_min(1e-300)) - model.rho_0()[: A.shape[-1]]
          + n * M.unsqueeze(-1))
    if return_parts:
        # Hand back the objects the sampler needs so it consumes exactly what this function
        # builds.  The sampler used to rebuild w, G and the convolution itself, and the two
        # constructions disagreed: it returned baskets of 25 where no single draw had
        # E[n|z] above 13.1.  Patching only its size stage made that worse -- size became
        # right while allocation still came from the local rebuild, and generated item
        # rates fell to 0.08x with none of the 200 commonest pairs ever appearing.  One
        # construction, consumed by every stage, is the only version that cannot drift.
        lg_out = lg[..., 1:] if drop_empty else lg
        return dict(logw=logw, M=M, w=w, G=G, Gp=Gp, A=A, lg=lg_out)
    if drop_empty:
        lg = lg[..., 1:]
    if return_terms:
        # the per-size terms, before they are summed away.  log f is a logsumexp over n, so
        # the size law is already sitting inside every normaliser evaluation; returning the
        # summands costs nothing and is what size_dist needs.
        return lg                                                   # [D, B, n]
    return torch.logsumexp(lg, dim=-1).transpose(0, 1)              # [B, D]


class RaggedModel(torch.nn.Module):
    """Same parameters and the same three quantities as core.Model, ragged over items."""

    def __init__(self, J, N, C, K=8, Kz=3, nmax=24, R=4, seed=0,
                 S=1, Kp=8, Kt=8, Ks=4, n_week=53, phi_init=0.03):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.J, self.N, self.C, self.S = J, N, C, S
        self.K, self.Kz, self.nmax, self.R = K, Kz, nmax, R
        self.lam = torch.nn.Parameter(torch.zeros(J))
        self.alpha = torch.nn.Parameter(torch.randn(J, K, generator=g) * 0.3)
        self.theta = torch.nn.Parameter(torch.randn(N, K, generator=g) * 0.3)
        # ||phi_j|| ~ phi_init * sqrt(Kz).  At the old 0.15 with Kz = 12 that was 0.52,
        # against a cap of 0.20 -- the model began 2.6x outside the region section 14
        # requires it to stay in, and the first projection yanked it back.  Since
        # lambda_max <~ ||phi||^2 E[n], starting at 0.52 with E[n] ~ 7.8 means lambda_max
        # ~ 2.1 at step zero: no unique mode, and a fixed-point iteration that diverges
        # rather than converges.  0.03 puts ||phi|| at 0.10, half the cap.
        self.phi = torch.nn.Parameter(torch.randn(J, Kz, generator=g) * phi_init)
        self.rho_c = torch.nn.Parameter(torch.zeros(C))
        self.rho_0_free = torch.nn.Parameter(torch.zeros(nmax))
        # --- conditioning blocks (Eq. 7) -------------------------------------------------
        # softplus(-2.8) = 0.060, so gamma.beta starts at about 8 * 0.060^2 = 0.029 --
        # the same magnitude the unconstrained product had at initialisation
        # softplus(-3.3) = 0.036, so gamma.beta starts near 8 * 0.036^2 = 0.010 -- the
        # value the data's aggregate elasticity of -0.121 implies, given Var(n) ~ 83.
        # -2.8 started it at 0.029 and training pushed the median to 1.81.
        self.gamma = torch.nn.Parameter(torch.randn(N, Kp, generator=g) * 0.1 - 3.3)
        self.beta = torch.nn.Parameter(torch.randn(J, Kp, generator=g) * 0.1 - 3.3)
        self.w_dsp = torch.nn.Parameter(torch.zeros(J))
        self.w_mlr = torch.nn.Parameter(torch.zeros(J))
        self.mu = torch.nn.Parameter(torch.randn(J, Kt, generator=g) * 0.1)
        self.delta = torch.nn.Parameter(torch.randn(n_week, Kt, generator=g) * 0.1)
        self.psi = torch.nn.Parameter(torch.zeros(J, 4))       # recency loading
        self.zeta = torch.nn.Parameter(torch.randn(J, Ks, generator=g) * 0.1)
        self.xi = torch.nn.Parameter(torch.randn(S, Ks, generator=g) * 0.1)
        # --- units, Eq. 19: P(q_j | j in S) = NB(q_j - 1 ; Lambda_ij, r_j) ------------
        # A separate factor, and the factorisation is DERIVED rather than assumed: if the
        # energy does not depend on q, summing q out of P(S, q) leaves exactly P(S), so
        # P(S, q) = P(S) prod_j P(q_j).  Without it the model cannot say how many of
        # anything, which makes revenue -- price times units -- uncomputable, so a
        # simulator that omits it cannot price a promotion.
        self.a_q = torch.nn.Parameter(torch.zeros(J))
        self.gamma_q = torch.nn.Parameter(torch.randn(N, Kp, generator=g) * 0.1 - 2.8)
        self.beta_q = torch.nn.Parameter(torch.randn(J, Kp, generator=g) * 0.1 - 2.8)
        self.log_r = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("cat_of", torch.zeros(J, dtype=torch.long))
        self.house = None      # [B] set per batch, with .ctx below
        self.ctx = None        # dict of per-slot features, set per batch

    def rho_0(self):
        z = torch.zeros(1, dtype=self.rho_0_free.dtype, device=self.rho_0_free.device)
        return torch.cat([z, self.rho_0_free])

    def b_at(self, it, trip, c):
        """Eq. 7 at an arbitrary set of (product, trip) pairs.

        ONE function for both the normaliser and the energy.  They previously had separate
        code paths and drifted: b_flat applied price, promotion, seasonality and store while
        energy() applied only taste, so E(S) and log Z scored the same product differently
        and the difference was free reward for the optimiser.  Nothing may compute an item
        value except through here.
        """
        hh = self.house[trip]
        b = self.lam[it] + (self.theta[hh] * self.alpha[it]).sum(-1)
        if c is None:
            return b
        # Price sensitivity is held non-negative.
        #
        # d b_j / d log p_j = -(gamma_h . beta_j), and nothing constrained that inner
        # product's sign, so the model was free to learn that a product becomes MORE
        # attractive when it gets dearer.  It did: on a fitted checkpoint a 10% cut on one
        # product gave d b_j = -0.0164, i.e. cheaper made it less likely.  A model used to
        # choose markdowns cannot have the own-price effect pointing the wrong way, whatever
        # its likelihood.  Passing both factors through softplus makes gamma.beta >= 0
        # elementwise, so the derivative is <= 0 by construction rather than by hope.
        b = b - (softplus(self.gamma[hh]) * softplus(self.beta[it])).sum(-1) * c["dlp"]
        b = b + self.w_dsp[it] * c["disp"] + self.w_mlr[it] * c["mail"]
        b = b + (self.mu[it] * self.delta[c["week"]]).sum(-1)
        b = b + (self.zeta[it] * self.xi[c["store"]]).sum(-1)
        if "rec" in c:
            b = b + (self.psi[it] * c["rec"]).sum(-1)
        return b

    def b_flat(self, ix):
        """b at every assortment slot, [T] -- the normaliser's view."""
        return self.b_at(ix.item, ix.item_trip, self.ctx)

    def log_Z(self, ix, n_draws=32, mode_steps=1, generator=None, return_ess=False,
              drop_empty=False, laplace=False, antithetic=False, return_size=False,
              z_init=None, return_mode=False, ais_steps=0, ais_hmc=1,
              mix_scales=None, aniso=0.0):
        """log Z by importance sampling from a Gaussian centred at the mode.

        The proposal covariance is the IDENTITY rather than the Laplace curvature, and the
        mode is located in ONE fixed-point step.  Measured on a trained checkpoint at 8
        draws, against a 1024-draw 12-step reference:

            mode_steps 3   |err| 0.0084   ESS 0.999   963 ms
            mode_steps 1   |err| 0.0085   ESS 0.999   438 ms
            mode_steps 0   |err| 0.0242   ESS 0.992   172 ms

        Steps two and three change nothing and cost 525 ms -- more than half the call.  The
        first step is not optional: dropping it triples the error.  A warm start from a
        previous visit's mode was tried and is not worth the machinery, since one step from
        zero already lands: at 0.05 drift it gave |err| 0.0084 against the cold 0.0085, and
        at 0.15 drift a stale mode with no correction step was WORSE than starting fresh
        (0.1440).  The contraction does the work; the starting point barely matters.  That is
        not a corner cut, it follows from the regime section 14 requires the model to
        operate in: the posterior of z is N(0, I) tilted by Lambda, and with
        lambda_max(Lambda) around 0.1 to 0.3 the exact covariance (I - Lambda)^{-1} lies
        between I and about 1.4 I.  Measured on a partly trained model at batch 24:

            10 mode steps + curvature   ESS 0.998   |log Z - ref| 0.0262   8.92 s
             3 mode steps + identity    ESS 0.998   |log Z - ref| 0.0284   1.43 s
             0 mode steps + identity    ESS 0.934   |log Z - ref| 0.1309   0.58 s

        6.2x faster for 0.002 nats.  The curvature cost 2*K_z = 24 gradient evaluations per
        call -- 65% of the total -- to fit a covariance the stability condition already
        guarantees is close to the identity.  The mode itself still matters: dropping it
        entirely costs ESS and an order of magnitude in accuracy.

        Pass laplace=True to restore the curvature; if ESS ever falls, that is the switch.
        """
        B = ix.B
        log_corr = None
        with torch.no_grad():
            # Warm start from the caller's cached mode.
            #
            # Measured on a batch of 24: log Z costs 28 ms per draw plus 755 ms that does
            # not depend on the draws at all -- 61% of a 16-draw call.  That fixed part is
            # the mode-finding passes, which run at D = 1 and so do almost no arithmetic;
            # they are dominated by launch overhead across the bucket loop and the tree
            # rounds.  Most of the compute was going into locating a 12-dimensional mode,
            # not into estimating the integral.
            #
            # A trip's mode moves slowly between the iterations that sample it, because the
            # parameters move slowly.  Starting from where it was last time lets the same
            # accuracy be reached in one step rather than three.  Starting from zero, as
            # this did, threw that away on every visit.
            if z_init is None:
                z = torch.zeros(B, 1, self.Kz, dtype=self.lam.dtype,
                                device=self.lam.device)
            else:
                z = z_init.detach().view(B, 1, self.Kz).to(self.lam.dtype)
            for _ in range(mode_steps):
                zz = z.detach().requires_grad_(True)
                with torch.enable_grad():
                    lf = log_f_ragged(self, zz, ix, drop_empty).sum()
                z = torch.autograd.grad(lf, zz)[0]
            zh = z.detach()
            if laplace:
                eps = 0.15
                curv = torch.zeros(B, self.Kz, dtype=zh.dtype, device=zh.device)
                for k in range(self.Kz):
                    d = torch.zeros(B, 1, self.Kz, dtype=zh.dtype, device=zh.device)
                    d[:, :, k] = eps
                    gs = []
                    for s in (d, -d):
                        zz = (zh + s).detach().requires_grad_(True)
                        with torch.enable_grad():
                            lf = log_f_ragged(self, zz, ix, drop_empty).sum()
                        gs.append((torch.autograd.grad(lf, zz)[0] - zz.detach())[:, 0, k])
                    curv[:, k] = -(gs[0] - gs[1]) / (2 * eps)
                sd = (1.0 / curv.clamp_min(0.05)).sqrt().clamp(0.05, 5.0)
            else:
                sd = torch.ones(B, self.Kz, dtype=zh.dtype, device=zh.device)
            if aniso > 1.0:
                # Widen the proposal ALONG Lambda's top eigenvector only.
                #
                # The target is broad where Lambda = Cov(sum_j phi_j x_j | z) is large and
                # unit-width elsewhere, and Lambda is low rank -- effective rank 1-2 in every
                # fitted model.  Widening isotropically therefore pays s^Kz in volume for
                # coverage it needs in one direction: at Kz = 12 that is 2^12 ~ 4,000x, and
                # the wide draws that occasionally land where log f is enormous made log Z
                # drift UPWARD with more draws (12.03 -> 12.30 -> 12.47) where the single
                # proposal converged to sd 0.0008.  Along one eigenvector the cost is s.
                #
                # Measured against exact enumeration at Kz = 12, bias at phi.phi = 2.0:
                # single -0.0385, isotropic mixture -0.0252, anisotropic +0.0017.  In a fit
                # at that strength the single proposal returns its cap with lift 1.000 --
                # no dependence at all -- while this recovers 1.920 against a true 2.000,
                # matching exact enumeration, with lift 2.063 against 2.078.
                with torch.no_grad():
                    ph = self.phi[ix.item]
                    bslot = self.b_flat(ix)
                    pr = (zh[:, 0][ix.item_trip] * ph).sum(-1)
                    pw = torch.sigmoid(bslot + pr - 0.5 * (ph ** 2).sum(-1))
                    wgt = (pw * (1 - pw)).unsqueeze(-1)
                    vtop = torch.zeros(B, self.Kz, dtype=zh.dtype, device=zh.device)
                    for b_ in range(B):
                        msk = ix.item_trip == b_
                        P = ph[msk]
                        L = (P * wgt[msk]).T @ P
                        u = torch.ones(self.Kz, dtype=L.dtype, device=L.device)
                        u = u / u.norm().clamp_min(1e-30)
                        for _ in range(12):
                            u2 = L @ u
                            nu = u2.norm()
                            if float(nu) < 1e-30:
                                break
                            u = u2 / nu
                        vtop[b_] = u
                e0 = torch.randn(B, n_draws, self.Kz, dtype=zh.dtype, device=zh.device,
                                 generator=generator)
                pj = (e0 * vtop.unsqueeze(1)).sum(-1, keepdim=True)
                noise = e0 + (aniso - 1.0) * pj * vtop.unsqueeze(1)
                r = (noise * vtop.unsqueeze(1)).sum(-1)
                perp2 = (noise ** 2).sum(-1) - r ** 2
                aniso_lq = -0.5 * (perp2 + (r / aniso) ** 2) - math.log(aniso)
            # The mixture is switched on by lambda_max, not applied unconditionally.
            #
            # Measured against exact enumeration on a 12-product catalogue, the single
            # Gaussian is essentially exact while the interaction is weak and badly biased
            # once it is strong; the mixture is the reverse, because half its draws land in
            # a wide component whose weights are negligible:
            #
            #   regime                  single bias    mixture bias
            #   phi.phi 0.5             -0.0015        -0.0015
            #   phi.phi 4.0             -0.4476        -0.0744
            #   real ckpt, lam ~ 0.045  +0.0003 (rmse 0.0007)   +0.0056 (rmse 0.0059)
            #
            # Applying it everywhere cost 8-30x accuracy in the regime that was already
            # fine and tripped the convergence guard at iteration 100.  It earns its place
            # only where the single proposal's bias exceeds its extra variance.
            if mix_scales is not None and len(mix_scales) > 1:
                # Defensive mixture proposal: half the draws tight, half wide, scored under
                # the MIXTURE density.
                #
                # A single Gaussian at the mode does not merely add noise as the interaction
                # strengthens -- it biases log Z LOW, and a log Z biased low makes the
                # likelihood reward larger phi, which biases it lower still.  Measured on a
                # 12-product catalogue against exact enumeration, bias by phi.phi:
                #
                #     phi.phi   N(mode,I)   mix(1,2)
                #        0.5     -0.0015     -0.0015
                #        2.0     -0.0330     -0.0138
                #        4.0     -0.4476     -0.0744
                #
                # and in a fit at phi.phi = 2.0 -- the strength real co-purchase lift needs --
                # the single proposal recovered 4.500 (its cap, lift 1.000, no dependence at
                # all) where the mixture recovered 1.951 against a true 2.000.  That is the
                # mechanism behind every run whose ||phi|| pinned to its cap and generated no
                # co-occurrence.
                ns = len(mix_scales)
                per = max(1, n_draws // ns)
                parts = [torch.randn(B, per, self.Kz, dtype=zh.dtype, device=zh.device,
                                     generator=generator) * sc for sc in mix_scales]
                noise = torch.cat(parts, dim=1)[:, :n_draws]
                comp = torch.stack([
                    (-0.5 * (noise / sc).pow(2).sum(-1)
                     - self.Kz * math.log(sc)) for sc in mix_scales])
                mix_lq = torch.logsumexp(comp, 0) - math.log(ns)
            elif antithetic:
                # Antithetic pairs: draw D/2 vectors and use both z and -z.  log f is close
                # to quadratic near the mode, so the odd-order error cancels exactly
                # between a pair and the estimator's variance falls at no extra cost.  The
                # proposal is symmetric about the mode, so the pairing is valid.
                h = torch.randn(B, (n_draws + 1) // 2, self.Kz, dtype=zh.dtype,
                                device=zh.device, generator=generator)
                noise = torch.cat([h, -h], dim=1)[:, :n_draws]
            else:
                noise = torch.randn(B, n_draws, self.Kz, dtype=zh.dtype, device=zh.device,
                                    generator=generator)
            zs = zh + noise * sd.unsqueeze(1)
            _L2P = float(math.log(2 * math.pi))
            if aniso > 1.0:
                log_q = aniso_lq - sd.log().sum(-1, keepdim=True) - 0.5 * self.Kz * _L2P
            if mix_scales is not None and len(mix_scales) > 1:
                log_q = mix_lq - sd.log().sum(-1, keepdim=True) - 0.5 * self.Kz * _L2P
            if ais_steps > 0:
                zs, log_corr = self._ais(zs, ix, drop_empty, ais_steps, ais_hmc, generator)
            L2P = float(math.log(2 * math.pi))
            log_q = (-0.5 * (noise ** 2).sum(-1) - sd.log().sum(-1, keepdim=True)
                     - 0.5 * self.Kz * L2P)
        L2P = float(math.log(2 * math.pi))
        if return_size:
            # the per-size terms are already formed inside log f; taking them here shares
            # the draws with the normaliser, so the size law costs nothing extra
            lg = log_f_ragged(self, zs, ix, drop_empty, return_terms=True)   # [D, B, n]
            lf = torch.logsumexp(lg, dim=-1).transpose(0, 1)                 # [B, D]
        else:
            lf = log_f_ragged(self, zs, ix, drop_empty)
        base = -0.5 * self.Kz * L2P - 0.5 * (zs ** 2).sum(-1)
        log_p = base + lf
        lw = log_p - log_q
        if log_corr is not None:
            lw = lw + log_corr
        lz = torch.logsumexp(lw, dim=1) - math.log(n_draws)
        pn = None
        mode_out = zh[:, 0, :].detach() if return_mode else None
        if return_size:
            tot = (base - log_q).unsqueeze(-1) + lg.permute(1, 0, 2)         # [B, D, n]
            pn = torch.softmax(tot.reshape(B, -1), dim=1).view(tot.shape).sum(1)
        ess = None
        if return_ess:
            with torch.no_grad():
                ww = torch.softmax(lw, dim=1)
                ess = 1.0 / (ww ** 2).sum(1) / n_draws   # PER TRIP, never a batch mean
        out = [lz]
        if return_ess:
            out.append(ess)
        if return_size:
            out.append(pn)
        if return_mode:
            out.append(mode_out)
        return out[0] if len(out) == 1 else tuple(out)

    def size_dist(self, ix, n_draws=64, mode_steps=3, generator=None, drop_empty=True,
                  z_fixed=None, return_mode=False, grad=False):
        """P(n) per trip, on the same importance draws log_Z uses.  [B, nmax]

        Z = E_z[sum_n e^{-rho_0(n)} A_n(z)], so the joint over (draw, size) is already
        formed inside log_f_ragged; the size law is that joint with the draw summed out,
        each draw carrying its own importance weight.  Normalising over the FLATTENED
        (draw, size) pairs and then summing over draws is what makes the weights count --
        normalising per draw first would give every draw equal say regardless of weight.

        With drop_empty (the default) index i is size i + 1, matching the conditioning on a
        non-empty basket that the likelihood uses.

        z_fixed supplies the proposal centre instead of locating it, which matters whenever
        two calls are DIFFERENCED.  Importance sampling is unbiased under any proposal, so
        the centre is irrelevant to a single estimate -- but a finite-draw difference
        between two estimates on two different proposals carries the change of proposal
        along with the effect being measured.  Checking dE[n]/de against Var(n) with the
        mode relocated at each e disagreed by 15% and would not converge as e -> 0; on a
        common proposal the same check converges as e^2, to 0.18%.  Any derivative taken by
        differencing this function must pass z_fixed.
        """
        B, L2P = ix.B, float(math.log(2 * math.pi))
        with torch.no_grad():
            if z_fixed is not None:
                zh = z_fixed
            else:
                z = torch.zeros(B, 1, self.Kz, dtype=self.lam.dtype,
                                device=self.lam.device)
                for _ in range(mode_steps):
                    zz = z.detach().requires_grad_(True)
                    with torch.enable_grad():
                        lf = log_f_ragged(self, zz, ix, drop_empty).sum()
                    z = torch.autograd.grad(lf, zz)[0]
                zh = z.detach()
            noise = torch.randn(B, n_draws, self.Kz, dtype=zh.dtype, device=zh.device,
                                generator=generator)
            zs = zh + noise                              # identity proposal, as in log_Z
            log_q = -0.5 * (noise ** 2).sum(-1) - 0.5 * self.Kz * L2P
            zs_d, log_q_d, zh_d = zs, log_q, zh
        # The proposal above is detached -- it is a sampling device, not part of the model.
        # What follows IS the model: rho_0 and the item values enter here, and recalibrating
        # rho_0 on held-out data needs to differentiate through it.  The whole function used
        # to sit under no_grad, so pn came back with no graph and the refit could not run.
        with torch.enable_grad() if grad else torch.no_grad():
            lg = log_f_ragged(self, zs_d, ix, drop_empty, return_terms=True)  # [D, B, n]
            base = -0.5 * self.Kz * L2P - 0.5 * (zs_d ** 2).sum(-1) - log_q_d  # [B, D]
            tot = base.unsqueeze(-1) + lg.permute(1, 0, 2)                   # [B, D, n]
            p = torch.softmax(tot.reshape(B, -1), dim=1).view(tot.shape).sum(1)
        return (p, zh_d) if return_mode else p

    def size_moments(self, ix, **kw):
        """(E[n], Var(n)) per trip, from size_dist."""
        p = self.size_dist(ix, **kw)
        if isinstance(p, tuple):
            p = p[0]
        n = torch.arange(1, p.shape[1] + 1, dtype=p.dtype, device=p.device)
        m1 = (p * n).sum(1)
        return m1, (p * n ** 2).sum(1) - m1 ** 2

    @torch.no_grad()
    def sample(self, ix, n_draws=64, generator=None, mode_steps=1):
        """Draw one basket per trip, exactly, by Corollary 3's three levels.

        Z = E_z[sum_n e^{-rho_0(n)} A_n(z)] and A_n is a convolution over categories, so the
        joint factors into a chain that can be sampled top down without ever touching the
        2^J subsets:

          1. z, from its posterior.  Drawn by sampling-importance-resampling on the same
             proposal the normaliser uses; with ESS at 0.998 the reweighting is close to a
             formality, but it is the only inexact step and is consistent as draws grow.
          2. the size n, from P(n | z) proportional to e^{-rho_0(n)} A_n(z).  Exact: these
             are the terms log f already sums over.
          3. the split of n across categories, then which products fill each slot.  Both
             exact, by walking the same recursions backwards -- see below.

        Everything below level 1 is exact given z, which is what makes this usable as an
        environment: a rollout is a categorical draw per level, not a Gibbs chain.

        The per-trip Python loop here is honest but slow; it is fine for validation and for
        modest rollouts, and vectorising across trips is the obvious next step.
        """
        B, L2P = ix.B, float(math.log(2 * math.pi))
        z = torch.zeros(B, 1, self.Kz, dtype=self.lam.dtype, device=self.lam.device)
        for _ in range(mode_steps):
            zz = z.detach().requires_grad_(True)
            with torch.enable_grad():
                lf = log_f_ragged(self, zz, ix, True).sum()
            z = torch.autograd.grad(lf, zz)[0]
        noise = torch.randn(B, n_draws, self.Kz, dtype=z.dtype, generator=generator)
        zs = z.detach() + noise
        log_q = -0.5 * (noise ** 2).sum(-1) - 0.5 * self.Kz * L2P
        lw = (-0.5 * self.Kz * L2P - 0.5 * (zs ** 2).sum(-1)
              + log_f_ragged(self, zs, ix, True)) - log_q            # [B, D]
        pick = torch.multinomial(torch.softmax(lw, dim=1), 1, generator=generator)
        zsel = zs.gather(1, pick.unsqueeze(-1).expand(-1, -1, self.Kz))[:, 0]  # [B, Kz]

        # Stage 2 takes the size law from log_f_ragged, not from a second construction.
        #
        # This used to rebuild the per-category polynomials and convolve them sequentially,
        # while log_f_ragged builds the same object and convolves it with poly_tree.  Two
        # implementations of one quantity, and only one of them is covered by the
        # brute-force validation.  They disagreed: with SIR-weighted E[n|z] at 7.07 and no
        # single draw exceeding 13.1, the sampler was returning baskets of 25.  Reusing the
        # validated path removes the second implementation rather than repairing it.
        # ONE construction, consumed by every stage.
        parts = log_f_ragged(self, zsel.unsqueeze(1), ix, True, return_parts=True)
        w_all = parts["w"][0]                       # [T]   per-slot weights, scaled by M
        G_all = parts["G"][0]                       # [n_rows, R+1]  a_c(r) * e_r(w_c)
        lg_all = parts["lg"][0]                     # [B, n]  index i is size i+1
        out = []
        for b in range(B):
            rows = (ix.row_trip == b).nonzero().flatten().tolist()
            # --- level 2: the size, straight from the shared terms -------------------
            row_lg = lg_all[b]
            n = int(torch.multinomial(torch.softmax(row_lg, 0), 1,
                                      generator=generator)) + 1
            # --- prefix products over THIS trip's categories, from the shared G -------
            polys = [G_all[r_] for r_ in rows]
            pref = [torch.ones(1, dtype=w_all.dtype, device=w_all.device)]
            for Gc in polys:
                pref.append(poly_mul_trunc(pref[-1].unsqueeze(0),
                                           Gc.unsqueeze(0), self.nmax)[0])
            # --- level 3: split n across categories, backwards ------------------------
            chosen = []
            left = n
            for c in range(len(polys) - 1, -1, -1):
                if left == 0:
                    break
                Gc, P = polys[c], pref[c]
                hi = min(left, Gc.shape[0] - 1)
                cand = torch.stack([
                    Gc[r] * P[left - r] if (left - r) < P.shape[0]
                    else torch.zeros((), dtype=w_all.dtype, device=w_all.device)
                    for r in range(hi + 1)])
                tot = float(cand.sum())
                if tot <= 0:
                    continue
                r_take = int(torch.multinomial(cand / cand.sum(), 1,
                                               generator=generator))
                if r_take:
                    # --- level 4: which products, from the SAME w -----------------
                    sel = (ix.row_of == rows[c]).nonzero().flatten()
                    wc = w_all[sel]
                    E = torch.zeros(len(wc) + 1, self.R + 1, dtype=wc.dtype,
                                    device=wc.device)
                    E[0, 0] = 1.0
                    for k in range(1, len(wc) + 1):
                        E[k] = E[k - 1].clone()
                        E[k, 1:] = E[k, 1:] + wc[k - 1] * E[k - 1, :-1]
                    need = r_take
                    for k in range(len(wc), 0, -1):
                        if need == 0:
                            break
                        den = float(E[k, need])
                        if den <= 0:
                            continue
                        num = float(wc[k - 1] * E[k - 1, need - 1])
                        if float(torch.rand(1, generator=generator)) < num / den:
                            chosen.append(int(ix.item[sel[k - 1]]))
                            need -= 1
                left -= r_take
            out.append(sorted(chosen))
        return out

    @torch.no_grad()
    def sample_slots(self, ix, n_draws=16, generator=None, mode_steps=1):
        """Same chain as sample(), returning ASSORTMENT SLOT indices rather than product ids.

        Contrastive divergence needs the sampled basket scored with the same covariates the
        data basket carries, and those live per slot.  Returning slots lets the caller gather
        ctx directly instead of reconstructing it from product ids.
        """
        item_of = {}
        for t_ in range(ix.B):
            sel = (ix.item_trip == t_).nonzero().flatten()
            item_of[t_] = {int(ix.item[k]): int(k) for k in sel}
        out = self.sample(ix, n_draws=n_draws, generator=generator, mode_steps=mode_steps)
        slots, trips = [], []
        for t_, bk in enumerate(out):
            for j in bk:
                k = item_of[t_].get(j)
                if k is not None:
                    slots.append(k); trips.append(t_)
        dev = self.lam.device
        return (torch.as_tensor(slots, dtype=torch.long, device=dev),
                torch.as_tensor(trips, dtype=torch.long, device=dev))


    def _ais(self, zs, ix, drop_empty, n_steps, n_hmc, generator):
        """BROKEN -- DO NOT ENABLE.  Kept only so the mistake stays on the record.

        The weight is double counted: lw in log_Z is already log p - log q with the full
        log f in it, and this adds sum_t (beta_t - beta_{t-1}) log f on top.  It also starts
        from the mode-centred proposal rather than from N(0, I), which the annealing path
        assumes at beta = 0.  Measured on run24, log Z drifts 20.3 -> 20.8 -> 22.3 as steps
        go 4 -> 8 -> 16, diverging rather than converging, while ESS falls 0.584 -> 0.255.

        The premise was wrong as well.  Plain importance sampling on the same trips gives
        10.0178 at 8 draws, 10.0072 at 64, 10.0086 at 512, 10.0075 at 2048, ESS 0.998
        throughout -- converged to four decimals across a 256x increase.  So the single
        Gaussian proposal is NOT failing at lambda_max ~ 2.5 on ordinary trips, and the
        argument for replacing it -- that a multimodal posterior was making the cheap
        sampler confidently wrong -- is not supported by measurement.  The estimator trouble
        is confined to specific trips, where it has been since the first ESS reading.
        """
        B, D = zs.shape[0], zs.shape[1]
        betas = torch.linspace(0.0, 1.0, n_steps + 1, dtype=zs.dtype)[1:]
        log_corr = torch.zeros(B, D, dtype=zs.dtype, device=zs.device)
        prev = 0.0
        eps = 0.15
        for b_t in betas:
            with torch.no_grad():
                lf = log_f_ragged(self, zs, ix, drop_empty)          # [B, D]
            log_corr = log_corr + (b_t - prev) * lf
            prev = float(b_t)
            for _ in range(n_hmc):
                zz = zs.detach().requires_grad_(True)
                with torch.enable_grad():
                    tgt = (-0.5 * (zz ** 2).sum(-1)
                           + b_t * log_f_ragged(self, zz, ix, drop_empty)).sum()
                g = torch.autograd.grad(tgt, zz)[0]
                noise = torch.randn(zs.shape, dtype=zs.dtype, device=zs.device,
                                    generator=generator)
                zs = (zs + 0.5 * eps ** 2 * g + eps * noise).detach()
        return zs, log_corr

    def pseudo_loglik(self, ix, line_item, line_trip, line_cat, B, line_ctx=None,
                      neg_per_trip=64, generator=None):
        """sum_j log P(x_j | x_-j) -- the fit that never touches Z.

        Every proposal tried for log Z failed the same way, because d log Z / d phi_j is a
        second moment under the MODEL and, sitting inside a log, its noise becomes BIAS in
        the gradient -- which is what drove ||phi|| to its cap in every run.  The conditional
        has no such problem.  Dropping or adding product j changes the energy by

            b_j + sum_{k in S, k != j} phi_j.phi_k - rho_c(c_j) * (n_{c_j} - x_j)
                - [rho_0(n + 1 - x_j) - rho_0(n - x_j)]

        so P(x_j | x_-j) is a logistic in that quantity and its normaliser is 1 + exp(.),
        exact.  No z-integral, no draws, no proposal.  Consistent for the joint, at some
        statistical efficiency: measured against exact enumeration it recovers phi.phi to
        about 8% low with ZERO seed variance, where the best proposal gave 3.148 +/- 1.85
        and lift 1.000 -- no dependence at all -- once phi.phi reached 3.

        The positives are every purchased line.  The negatives are a sample of products the
        trip did NOT buy; scoring all ~5,400 per trip is affordable only in principle, and a
        sample keeps the estimator unbiased in expectation.
        """
        dt, dev = self.lam.dtype, self.lam.device
        # --- basket sufficient statistics -------------------------------------------
        Ssum = torch.zeros(B, self.Kz, dtype=dt, device=dev).index_add_(
            0, line_trip, self.phi[line_item])                      # sum of phi over S
        n_tot = torch.bincount(line_trip, minlength=B).to(dt)
        key = line_trip * self.C + line_cat
        n_c = torch.bincount(key, minlength=B * self.C).view(B, self.C).to(dt)
        r0 = self.rho_0()

        def cond_logit(it, tr, ctx, inbasket):
            ph = self.phi[it]
            b = self.b_at(it, tr, ctx)
            # sum over k in S, k != j
            pair = (ph * (Ssum[tr] - inbasket.unsqueeze(-1) * ph)).sum(-1)
            cj = self.cat_of[it]
            nc = n_c[tr, cj] - inbasket
            n0 = n_tot[tr] - inbasket
            dr0 = r0[(n0 + 1).clamp(max=self.nmax).long()] - r0[n0.clamp(max=self.nmax).long()]
            return b + pair - self.rho_c[cj] * nc - dr0

        # EVERY assortment slot, not a sample of them.
        #
        # The bench version summed over all J = 12 products exactly, and that is what
        # recovered phi at every strength with zero seed variance.  Approximating the sum by
        # 512 draws upweighted ~10x put back exactly the variance the method was chosen to
        # remove: E[n] ran to 54.2 against an observed 7.2 and held-out likelihood went
        # backwards.  The full sum is ~127,000 slots, which log_f_ragged already evaluates
        # every iteration, so the exact estimator costs about the same and is deterministic.
        n_line = line_item.shape[0]
        one = torch.ones(n_line, dtype=dt, device=dev)
        zero_l = torch.zeros(n_line, dtype=dt, device=dev)
        sp = torch.nn.functional.softplus

        # every slot scored as if absent from the basket
        T = ix.item.shape[0]
        allc = cond_logit(ix.item, ix.item_trip, self.ctx,
                          torch.zeros(T, dtype=dt, device=dev))
        lp = torch.zeros(B, dtype=dt, device=dev).index_add_(0, ix.item_trip, -sp(allc))

        # purchased lines were counted above with the wrong conditioning: remove that term
        # and add log P(x_j = 1 | x_-j), which conditions on S \ {j}
        wrong = cond_logit(line_item, line_trip, line_ctx, zero_l)
        right = cond_logit(line_item, line_trip, line_ctx, one)
        lp = lp.index_add_(0, line_trip, sp(wrong) + right - sp(right))
        return lp



    def energy(self, line_item, line_trip, line_cat, B, line_ctx=None):
        """E(S) from the observed lines, using the SAME item values as the normaliser."""
        dt, dev = self.lam.dtype, self.lam.device
        lin = torch.zeros(B, dtype=dt, device=dev).index_add_(
            0, line_trip, self.b_at(line_item, line_trip, line_ctx))
        v = torch.zeros(B, self.Kz, dtype=dt, device=dev).index_add_(
            0, line_trip, self.phi[line_item])
        sq = torch.zeros(B, dtype=dt, device=dev).index_add_(
            0, line_trip, (self.phi[line_item] ** 2).sum(-1))
        pair = 0.5 * ((v * v).sum(-1) - sq)
        key = line_trip * self.C + line_cat
        nc = torch.bincount(key, minlength=B * self.C).view(B, self.C).to(dt)
        pen_c = (self.rho_c.unsqueeze(0) * nc * (nc - 1) / 2.0).sum(-1)
        n = torch.bincount(line_trip, minlength=B).clamp(max=self.nmax)
        return lin + pair - pen_c - self.rho_0()[n]

    def units_loglik(self, line_item, line_trip, units, line_ctx, B):
        """log P(q | S), summed per trip.  Shifted negative binomial: q - 1 ~ NB(mu, r).

        Version 2 measures the shifted POISSON as the wrong law here -- q is under-dispersed
        (0.62) while q - 1 is over-dispersed (2.35), which a one-parameter Poisson cannot do
        -- and the negative binomial cuts units-per-line total variation from 0.045 to
        0.016 for one extra parameter."""
        hh = self.house[line_trip]
        # same constraint on the units model: quantity must not rise with price
        z = self.a_q[line_item] - (softplus(self.gamma_q[hh])
                                   * softplus(self.beta_q[line_item])).sum(-1) \
            * line_ctx["dlp"]
        mu = torch.exp(z.clamp(-6.0, 4.0))
        r = torch.nn.functional.softplus(self.log_r) + 1e-6
        k = (units - 1).to(mu.dtype).clamp_min(0.0)
        ll = (torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1.0)
              + r * (torch.log(r) - torch.log(r + mu))
              + k * (torch.log(mu.clamp_min(1e-12)) - torch.log(r + mu)))
        return torch.zeros(B, dtype=mu.dtype, device=mu.device).index_add_(0, line_trip, ll)

    def loglik(self, ix, line_item, line_trip, line_cat, n_draws=32,
               generator=None, return_ess=False, line_ctx=None, units=None,
               return_size=False, z_init=None, return_mode=False, mode_steps=1,
               mix_scales=None, aniso=0.0):
        """log P(S | S non-empty) = E(S) - log(Z - 1), with log(Z - 1) computed directly
        rather than by subtracting 1 from Z."""
        out = self.log_Z(ix, n_draws=n_draws, generator=generator,
                         return_ess=return_ess, drop_empty=True,
                         return_size=return_size, z_init=z_init,
                         return_mode=return_mode, mode_steps=mode_steps,
                         mix_scales=mix_scales, aniso=aniso)
        out = list(out) if isinstance(out, tuple) else [out]
        lz1 = out.pop(0)
        ess = out.pop(0) if return_ess else None
        pn = out.pop(0) if return_size else None
        zh = out.pop(0) if return_mode else None
        ll = self.energy(line_item, line_trip, line_cat, ix.B, line_ctx) - lz1
        if units is not None:
            ll = ll + self.units_loglik(line_item, line_trip, units, line_ctx, ix.B)
        res = [ll]
        if return_ess:
            res.append(ess)
        if return_size:
            res.append(pn)
        if return_mode:
            res.append(zh)
        return res[0] if len(res) == 1 else tuple(res)

    def pi_exact(self, ix, mode_steps=2):
        """pi_j = d log(Z-1) / d b_j at the mode -- Corollary 2, by autograd.

        Every cheap stand-in for pi has failed in its own direction.  sigmoid(b) overstated
        it 135x (0.23 against a measured 0.0017) and crushed phi; a softmax of exp(b)
        normalised to E[n] got the scale right but the shape wrong, and lambda_max swung
        6.50, 0.71, 1.49, 3.29, 4.17, 1.57 across consecutive checkpoints while ||phi||
        barely moved -- a constraint that binds on some batches and not others.  This is the
        quantity itself, at roughly the cost of one extra normaliser evaluation.
        """
        zh = torch.zeros(ix.B, 1, self.Kz, dtype=self.lam.dtype, device=self.lam.device)
        for _ in range(mode_steps):
            zz = zh.detach().requires_grad_(True)
            with torch.enable_grad():
                zh = torch.autograd.grad(log_f_ragged(self, zz, ix, True).sum(), zz)[0]
        b0 = self.b_flat(ix).detach().requires_grad_(True)
        old, self.ctx = self.ctx, None
        lam0 = self._parameters.pop('lam')
        self.lam = lam0.data
        try:
            with torch.enable_grad():
                phi_i = self.phi[ix.item].detach()
                bt = b0 - 0.5 * (phi_i ** 2).sum(-1)
                proj = (zh.detach()[ix.item_trip] * phi_i.unsqueeze(1)).sum(-1)
                logw = (bt.unsqueeze(1) + proj).transpose(0, 1)
                M = seg_max(logw, ix.item_trip, ix.B)
                w = torch.exp(logw - M.index_select(-1, ix.item_trip))
                e = esp_bucketed(w, ix.row_of, ix.n_rows, self.R, ix.row_size, ix.item_pos)
                r = torch.arange(self.R + 1, dtype=w.dtype)
                a_ = torch.exp(-self.rho_c[ix.row_cat].detach().unsqueeze(-1)
                               * r * (r - 1) / 2.0)
                Gp = torch.zeros(1, ix.B * ix.Cpad, self.R + 1, dtype=w.dtype)
                Gp[:, :, 0] = 1.0
                Gp = Gp.index_copy(1, ix.flat_slot,
                                   a_.unsqueeze(0) * e).view(1, ix.B, ix.Cpad, self.R + 1)
                A = poly_tree(Gp, self.nmax)
                n_ax = torch.arange(A.shape[-1], dtype=w.dtype)
                lg = (torch.log(A.clamp_min(1e-300))
                      - self.rho_0().detach()[: A.shape[-1]] + n_ax * M.unsqueeze(-1))
                lf = torch.logsumexp(lg[..., 1:], dim=-1).sum()
            pi = torch.autograd.grad(lf, b0)[0].clamp(0, 1).detach()
        finally:
            self._parameters['lam'] = lam0
            self.ctx = old
        return pi

    def lambda_max(self, ix):
        """Top eigenvalue of Lambda = sum_j pi_j(1-pi_j) phi_j phi_j' at the mode.

        Section 14 makes lambda_max < 1 the condition under which the mode is unique, the
        fixed-point map contracts and the Laplace proposal is sound.  Capping ||phi|| bounds
        it only loosely, so the quantity itself is measured.  pi_j = d log f / d b_j comes
        from autograd, which is exact rather than an approximation of the inclusion
        probability."""
        zh = torch.zeros(ix.B, 1, self.Kz, dtype=self.lam.dtype, device=self.lam.device)
        for _ in range(8):
            zz = zh.detach().requires_grad_(True)
            with torch.enable_grad():
                zh = torch.autograd.grad(log_f_ragged(self, zz, ix, True).sum(), zz)[0]
        b0 = self.b_flat(ix).detach().requires_grad_(True)
        old, self.ctx = self.ctx, None
        lam0 = self.lam
        try:
            with torch.enable_grad():
                # re-express log f as a function of the slot values directly
                phi_i = self.phi[ix.item].detach()
                bt = b0 - 0.5 * (phi_i ** 2).sum(-1)
                proj = (zh.detach()[ix.item_trip] * phi_i.unsqueeze(1)).sum(-1)
                logw = (bt.unsqueeze(1) + proj).transpose(0, 1)
                M = seg_max(logw, ix.item_trip, ix.B)
                w = torch.exp(logw - M.index_select(-1, ix.item_trip))
                e = esp_newton(w, ix.row_of, ix.n_rows, self.R, ix.row_size)
                r = torch.arange(self.R + 1, dtype=w.dtype)
                a_ = torch.exp(-self.rho_c[ix.row_cat].detach().unsqueeze(-1)
                               * r * (r - 1) / 2.0)
                G = a_.unsqueeze(0) * e
                Gp = torch.zeros(1, ix.B * ix.Cpad, self.R + 1, dtype=w.dtype)
                Gp[:, :, 0] = 1.0
                Gp = Gp.index_copy(1, ix.flat_slot, G).view(1, ix.B, ix.Cpad, self.R + 1)
                A = poly_tree(Gp, self.nmax)
                n_ax = torch.arange(A.shape[-1], dtype=w.dtype)
                lg = (torch.log(A.clamp_min(1e-300))
                      - self.rho_0().detach()[: A.shape[-1]] + n_ax * M.unsqueeze(-1))
                lf = torch.logsumexp(lg[..., 1:], dim=-1).sum()
            pi = torch.autograd.grad(lf, b0)[0].clamp(0, 1)
        finally:
            self.ctx = old
        v = (pi * (1 - pi)).detach()
        out = 0.0
        with torch.no_grad():
            for b in range(ix.B):
                msk = ix.item_trip == b
                P = self.phi[ix.item[msk]]
                L = (P * v[msk].unsqueeze(-1)).T @ P
                # Power iteration, not eigvalsh.  Lambda is PSD and, once phi collapses
                # toward one direction, all but one eigenvalue is ~0 -- eigvalsh raises
                # "too many repeated eigenvalues" and killed run35 at iteration 400 on a
                # purely diagnostic call.  A few matvecs give the top eigenvalue and cannot
                # fail on a degenerate spectrum.
                u = torch.ones(L.shape[0], dtype=L.dtype, device=L.device)
                u = u / u.norm().clamp_min(1e-30)
                lam_b = 0.0
                for _ in range(24):
                    u2 = L @ u
                    nu = float(u2.norm())
                    if nu < 1e-30:
                        lam_b = 0.0
                        break
                    u = u2 / nu
                    lam_b = float(u @ (L @ u))
                out = max(out, lam_b)
        return out

    @torch.no_grad()
    @torch.no_grad()
    def project_mean(self, e_now, e_target, v_now, damp=0.5, b_max=0.5):
        """Put E[n] on target by adding a linear term to rho_0.

        project_var narrows the size law but does not move where it is centred.  On a
        converged checkpoint narrowing happened to pull the mean down too (11.93 -> 6.41)
        and I took that for the mechanism; during training it does not hold -- Var(n) landed
        on 72 against a 67 target while E[n] rose to 29.2.  Mean and spread need separate
        corrections.

        Proposition 1 supplies the first one exactly as it supplied the second: for a linear
        term b*n added to rho_0, dE[n]/db = -Var(n).  So b = (E[n] - target) / Var(n).
        """
        if v_now <= 0:
            return 0.0
        b = (e_now - e_target) / v_now
        b = float(min(max(b, -b_max), b_max)) * damp
        n = torch.arange(1, self.rho_0_free.shape[0] + 1,
                         dtype=self.rho_0_free.dtype, device=self.rho_0_free.device)
        self.rho_0_free += b * n
        return b

    @torch.no_grad()
    def project_var(self, v_now, v_target, damp=0.35, c_max=0.02):
        """Put Var(n) on target by adding a quadratic to rho_0.

        Every failure measured today traces to one number.  Var(n) sat near 204 against an
        observed 83, and Proposition 1 says dE[n]/de = Var(n), so a 0.178 difference in mean
        index between an ordinary trip and a "runaway" one became 0.178 * 204 = 36.3 extra
        items -- exactly the 41.6 vs 5.3 gap measured.  The tail trips were never unusual;
        they were ordinary trips amplified 204x.  The same number drives the size inflation,
        the 8x over-elasticity, the ESS collapse and the dropped trips.

        P(n) is exp(g(n)) with g = log A_n - rho_0(n), locally quadratic at its mode, so
        Var(n) = 1/k with k = rho_0'' - (log A_n)''.  Adding c n^2 to rho_0 raises k by 2c:

            1/V_new = 1/V_now + 2c   ->   c = (1/V_target - 1/V_now) / 2

        Measured on a fitted checkpoint, one step took E[n] 11.93 -> 6.41 and Var(n)
        118.5 -> 32.6, against observed 5.73 and 35.3.

        Two guards, both learned the hard way.  c is clamped non-negative: the widening
        direction sent E[n] to the nmax boundary within two steps.  And the step is damped,
        because the linear solve overshoots -- c = 0.00181 aimed at 83 and delivered 32.6.
        """
        if v_now <= 0 or v_target <= 0:
            return 0.0
        c = 0.5 * (1.0 / v_target - 1.0 / v_now)
        c = float(min(max(c, 0.0), c_max)) * damp
        if c <= 0:
            return 0.0
        n = torch.arange(1, self.rho_0_free.shape[0] + 1,
                         dtype=self.rho_0_free.dtype, device=self.rho_0_free.device)
        self.rho_0_free += c * n ** 2
        return c

    @torch.no_grad()
    def project_rho_c(self, floor=-1.5):
        """Floor the within-category ATTRACTION.

        rho_c enters as exp(-rho_c n_c(n_c-1)/2), so repulsion (rho_c > 0) shrinks the term
        and attraction grows it QUADRATICALLY in the category count.  With R = 23 and
        rho_c = -2.809 that is exp(711); float64 overflows at exp(709), and run35c went NaN.
        The spec puts no floor on rho_c, which was harmless while the partition was
        dunnhumby's commodity groups -- they rarely group true complements, so rho_c stayed
        near zero.  Affinity groups drive it hard negative and the term detonates.

        The data bounds the sensible value: a 2.5x pair lift is rho_c = -0.92, so -1.5
        (4.5x) leaves room while keeping exp(1.5 * 253) at 1e165.
        """
        self.rho_c.clamp_(min=floor)

    @torch.no_grad()
    def project_price(self, target_gb):
        """Force the mean price sensitivity onto the value the data implies.

        A soft penalty on the elasticity did not hold it: with Adam the step size is set by
        the gradient's SIGN and running scale, not its magnitude, so a penalty term with a
        huge gradient still moves the parameter by about lr per step and the likelihood
        simply out-pushes it.  Measured: with weight 20 and the target at -0.121, the
        elasticity went -0.765 at iteration 400 to -4.871 at 800 -- the wrong way, fast.

        A projection cannot be out-pushed.  gamma and beta enter only through
        softplus(gamma).softplus(beta), and at this scale softplus(x) ~ e^x, so subtracting
        a constant c from both raw tensors multiplies the product by about e^{-2c}.  One
        closed-form step therefore lands the mean on target, and it is reapplied after every
        optimiser step exactly as the phi cap is.
        """
        gb = (softplus(self.gamma).mean(0) * softplus(self.beta).mean(0)).sum()
        cur = float(gb)
        if cur <= 0 or target_gb <= 0:
            return
        c = 0.5 * math.log(cur / target_gb)
        self.gamma -= c
        self.beta -= c

    @torch.no_grad()
    def project(self, phi_max, budget=None, thresh=0.0, centre=False, whiten=0.0):
        """Shape the interaction, instead of only shrinking it.

        The old behaviour was a per-item cap on ||phi_j||, which bounds lambda_max by
        bounding every item equally.  Measured against data, that is the wrong trade.  Real
        baskets show a lift of 2.5x over independence on their commonest pairs, needing
        phi_j.phi_k ~ 0.91; a uniform cap that keeps lambda_max under 1 at E[n] = 10 allows
        phi.phi <= 0.100, a lift of 1.105x.  The fitted model supplied 1.014x.  No setting
        of a uniform cap closes a factor of nine.

        But lambda_max is bounded by sum_j pi_j(1-pi_j) ||phi_j||^2 -- a SUM over 5,455
        products.  If only a small share of products carry a large phi and the rest carry
        none, the sum stays small while those products get the coupling the data asks for.
        At E[n] = 10, phi.phi = 0.91 on the active items needs only about a tenth of them
        active to keep the sum under 1.  So:

          thresh   group soft-threshold, phi_j -> phi_j max(0, 1 - t/||phi_j||).  A
                   proximal step, not a penalty: with Adam the step size follows the
                   gradient's running scale rather than its magnitude, which is why four
                   soft penalties failed to hold anything today.
          budget   the global sum is what lambda_max actually depends on, so it is capped
                   directly and phi_max becomes a loose ceiling rather than the binding one.
          centre   62% of the fitted pair energy came from a single shared direction -- the
                   interaction re-implementing the size law that rho_0 already carries.
                   Removing the mean frees the rank for genuine pairings.
        """
        if centre:
            self.phi -= self.phi.mean(0, keepdim=True)
        if whiten > 0:
            # Flatten phi's spectrum so the latent width is actually used.
            #
            # lambda_max is the top eigenvalue of sum_j pi_j(1-pi_j) phi_j phi_j', bounded by
            # trace / EFFECTIVE rank -- not trace / Kz.  Raising Kz 12 -> 128 looked like an
            # 18x win on lambda_max when measured with isotropic phi, and delivered nothing
            # in training: after 800 iterations one direction carried 88.5% of phi and the
            # effective rank was 1.3 of 128, so lambda_max went straight back to ~trace
            # (1.152 then 5.370).  The width is only headroom while the mass is spread.
            #
            # Pushing the singular values toward their mean forces that.  Partial, at
            # strength `whiten`, so the fit keeps its preferred directions but cannot
            # collapse onto one.
            # Flatten the SHAPE, keep the SIZE.  Setting singular values toward their mean
            # lowers the Frobenius norm -- Kz*mean(s)^2 <= sum s_i^2 by Cauchy-Schwarz -- and
            # applied every step it compounds: at strength 0.30 phi fell to a mean norm of
            # 0.001 with 93% of products at exactly zero inside 400 iterations.  Rescaling
            # to the original norm separates the two effects, which is the whole point.
            U, S, Vh = torch.linalg.svd(self.phi, full_matrices=False)
            S2 = (1 - whiten) * S + whiten * S.mean()
            S2 = S2 * (S.norm() / S2.norm().clamp_min(1e-30))
            self.phi.copy_(U @ torch.diag(S2) @ Vh)
        if thresh > 0:
            n = self.phi.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.phi.mul_((1.0 - thresh / n).clamp_min(0.0))
        n = self.phi.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.phi.mul_(torch.clamp(phi_max / n, max=1.0))
        if budget is not None:
            # Weight by inclusion probability -- lambda_max is sum_j pi_j(1-pi_j)||phi_j||^2,
            # not sum_j ||phi_j||^2.  Budgeting the unweighted sum left the real quantity
            # free, and the optimiser found the gap: run27 swung lambda_max 4.34 -> 1.55
            # between two checkpoints while ||phi|| moved 0.106 -> 0.108 and its max stayed
            # pinned at the ceiling.  It was shifting phi mass onto high-probability
            # products, which costs nothing under an unweighted budget and everything under
            # the condition section 14 actually states.
            w = getattr(self, "_pi_w", None)
            if w is None:
                tot = float((self.phi ** 2).sum())
            else:
                # lambda_max <= sum_j pi_j(1-pi_j) ||phi_j||^2 exactly -- no catalogue-size
                # factor.  The unweighted form needed one because it stood in for the
                # weights; carrying it here as well made the bound 5,455x too tight and
                # crushed ||phi|| to 0.008 on the probe.
                tot = float((w.unsqueeze(-1) * self.phi ** 2).sum())
            if tot > budget:
                self.phi.mul_(math.sqrt(budget / tot))
