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
        self.item_trip = self.row_trip[self.row_of]
        self.flat_slot = self.row_trip * self.Cpad + self.row_pos


def log_f_ragged(model, z, ix, drop_empty=False):
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
    e = esp_newton(w, ix.row_of, ix.n_rows, model.R, ix.row_size)  # [D, n_rows, R+1]
    r = torch.arange(model.R + 1, dtype=w.dtype, device=w.device)
    a = torch.exp(-model.rho_c[ix.row_cat].unsqueeze(-1) * r * (r - 1) / 2.0)
    G = a.unsqueeze(0) * e                                          # [D, n_rows, R+1]
    # scatter rows into [D, B, Cpad, R+1]; missing rows are the identity polynomial
    Gp = torch.zeros(D, ix.B * ix.Cpad, model.R + 1, dtype=w.dtype, device=w.device)
    Gp[:, :, 0] = 1.0
    Gp = Gp.index_copy(1, ix.flat_slot, G).view(D, ix.B, ix.Cpad, model.R + 1)
    A = Gp[:, :, 0, :]
    for c in range(1, ix.Cpad):
        A = poly_mul_trunc(A, Gp[:, :, c, :], model.nmax)          # [D, B, nmax+1]
    n = torch.arange(A.shape[-1], dtype=w.dtype, device=w.device)
    lg = (torch.log(A.clamp_min(1e-300)) - model.rho_0()[: A.shape[-1]]
          + n * M.unsqueeze(-1))
    if drop_empty:
        lg = lg[..., 1:]
    return torch.logsumexp(lg, dim=-1).transpose(0, 1)              # [B, D]


class RaggedModel(torch.nn.Module):
    """Same parameters and the same three quantities as core.Model, ragged over items."""

    def __init__(self, J, N, C, K=8, Kz=3, nmax=24, R=4, seed=0,
                 S=1, Kp=8, Kt=8, Ks=4, n_week=53):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.J, self.N, self.C, self.S = J, N, C, S
        self.K, self.Kz, self.nmax, self.R = K, Kz, nmax, R
        self.lam = torch.nn.Parameter(torch.zeros(J))
        self.alpha = torch.nn.Parameter(torch.randn(J, K, generator=g) * 0.3)
        self.theta = torch.nn.Parameter(torch.randn(N, K, generator=g) * 0.3)
        self.phi = torch.nn.Parameter(torch.randn(J, Kz, generator=g) * 0.15)
        self.rho_c = torch.nn.Parameter(torch.zeros(C))
        self.rho_0_free = torch.nn.Parameter(torch.zeros(nmax))
        # --- conditioning blocks (Eq. 7) -------------------------------------------------
        self.gamma = torch.nn.Parameter(torch.randn(N, Kp, generator=g) * 0.1)
        self.beta = torch.nn.Parameter(torch.randn(J, Kp, generator=g) * 0.1)
        self.w_dsp = torch.nn.Parameter(torch.zeros(J))
        self.w_mlr = torch.nn.Parameter(torch.zeros(J))
        self.mu = torch.nn.Parameter(torch.randn(J, Kt, generator=g) * 0.1)
        self.delta = torch.nn.Parameter(torch.randn(n_week, Kt, generator=g) * 0.1)
        self.zeta = torch.nn.Parameter(torch.randn(J, Ks, generator=g) * 0.1)
        self.xi = torch.nn.Parameter(torch.randn(S, Ks, generator=g) * 0.1)
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
        b = b - (self.gamma[hh] * self.beta[it]).sum(-1) * c["dlp"]
        b = b + self.w_dsp[it] * c["disp"] + self.w_mlr[it] * c["mail"]
        b = b + (self.mu[it] * self.delta[c["week"]]).sum(-1)
        b = b + (self.zeta[it] * self.xi[c["store"]]).sum(-1)
        return b

    def b_flat(self, ix):
        """b at every assortment slot, [T] -- the normaliser's view."""
        return self.b_at(ix.item, ix.item_trip, self.ctx)

    def log_Z(self, ix, n_draws=32, mode_steps=10, generator=None, return_ess=False,
              drop_empty=False):
        B = ix.B
        with torch.no_grad():
            z = torch.zeros(B, 1, self.Kz, dtype=self.lam.dtype, device=self.lam.device)
            for _ in range(mode_steps):
                zz = z.detach().requires_grad_(True)
                with torch.enable_grad():
                    lf = log_f_ragged(self, zz, ix, drop_empty).sum()
                z = torch.autograd.grad(lf, zz)[0]
            zh = z.detach()
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
            noise = torch.randn(B, n_draws, self.Kz, dtype=zh.dtype, device=zh.device,
                                generator=generator)
            zs = zh + noise * sd.unsqueeze(1)
            L2P = float(math.log(2 * math.pi))
            log_q = (-0.5 * (noise ** 2).sum(-1) - sd.log().sum(-1, keepdim=True)
                     - 0.5 * self.Kz * L2P)
        L2P = float(math.log(2 * math.pi))
        log_p = (-0.5 * self.Kz * L2P - 0.5 * (zs ** 2).sum(-1)
                 + log_f_ragged(self, zs, ix, drop_empty))
        lw = log_p - log_q
        lz = torch.logsumexp(lw, dim=1) - math.log(n_draws)
        if not return_ess:
            return lz
        with torch.no_grad():
            ww = torch.softmax(lw, dim=1)
            ess = 1.0 / (ww ** 2).sum(1) / n_draws
        return lz, ess

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

    def loglik(self, ix, line_item, line_trip, line_cat, n_draws=32,
               generator=None, return_ess=False, line_ctx=None):
        """log P(S | S non-empty) = E(S) - log(Z - 1), with log(Z - 1) computed directly
        rather than by subtracting 1 from Z."""
        out = self.log_Z(ix, n_draws=n_draws, generator=generator,
                         return_ess=return_ess, drop_empty=True)
        lz1, ess = out if return_ess else (out, None)
        ll = self.energy(line_item, line_trip, line_cat, ix.B, line_ctx) - lz1
        return (ll, ess) if return_ess else ll

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
                A = Gp[:, :, 0, :]
                for c in range(1, ix.Cpad):
                    A = poly_mul_trunc(A, Gp[:, :, c, :], self.nmax)
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
                out = max(out, float(torch.linalg.eigvalsh(L).max()))
        return out

    @torch.no_grad()
    def project(self, phi_max):
        """Hard cap on ||phi_j||.  The diverged run reached a mean norm of 2.94 from an
        initialisation of 0.15, which drove the pair term to 15,437 on a 36-line basket.
        Section 14 says the model must stay where lambda_max(Lambda) < 1, and Lambda scales
        with ||phi||^2, so bounding the norm is the cheapest sufficient guard."""
        n = self.phi.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.phi.mul_(torch.clamp(phi_max / n, max=1.0))
