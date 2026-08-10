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


def esp_newton(w, row_of, n_rows, R):
    """e_0..e_R per row from power sums.  w [..., T] -> [..., n_rows, R+1]."""
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
    return torch.stack(e, dim=-1)


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
        self.item_trip = self.row_trip[self.row_of]
        self.flat_slot = self.row_trip * self.Cpad + self.row_pos


def log_f_ragged(model, z, ix):
    """log f(z) for a batch.  z [B, D, Kz] -> [B, D].  Identical mathematics to
    core.Model.log_f, with the item axis unpadded."""
    D = z.shape[1]
    phi_i = model.phi[ix.item]                                     # [T, Kz]
    bt = model.b_flat(ix) - 0.5 * (phi_i ** 2).sum(-1)             # [T]
    proj = (z[ix.item_trip] * phi_i.unsqueeze(1)).sum(-1)          # [T, D]
    logw = (bt.unsqueeze(1) + proj).transpose(0, 1)                # [D, T]
    # ONE scale per (trip, draw): a per-row scale would not survive the convolution
    M = seg_max(logw, ix.item_trip, ix.B)                          # [D, B]
    w = torch.exp(logw - M.index_select(-1, ix.item_trip))         # [D, T]
    e = esp_newton(w, ix.row_of, ix.n_rows, model.R)               # [D, n_rows, R+1]
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
    return torch.logsumexp(lg, dim=-1).transpose(0, 1)              # [B, D]


class RaggedModel(torch.nn.Module):
    """Same parameters and the same three quantities as core.Model, ragged over items."""

    def __init__(self, J, N, C, K=8, Kz=3, nmax=24, R=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.J, self.N, self.C = J, N, C
        self.K, self.Kz, self.nmax, self.R = K, Kz, nmax, R
        self.lam = torch.nn.Parameter(torch.zeros(J))
        self.alpha = torch.nn.Parameter(torch.randn(J, K, generator=g) * 0.3)
        self.theta = torch.nn.Parameter(torch.randn(N, K, generator=g) * 0.3)
        self.phi = torch.nn.Parameter(torch.randn(J, Kz, generator=g) * 0.15)
        self.rho_c = torch.nn.Parameter(torch.zeros(C))
        self.rho_0_free = torch.nn.Parameter(torch.zeros(nmax))
        self.house = None                     # [B] set per batch

    def rho_0(self):
        z = torch.zeros(1, dtype=self.rho_0_free.dtype, device=self.rho_0_free.device)
        return torch.cat([z, self.rho_0_free])

    def b_flat(self, ix):
        """b_ij at every assortment slot, [T].  Extend here for price, promotion, coupon."""
        return (self.lam[ix.item]
                + (self.theta[self.house[ix.item_trip]] * self.alpha[ix.item]).sum(-1))

    def log_Z(self, ix, n_draws=32, mode_steps=10, generator=None, return_ess=False):
        B = ix.B
        with torch.no_grad():
            z = torch.zeros(B, 1, self.Kz, dtype=self.lam.dtype, device=self.lam.device)
            for _ in range(mode_steps):
                zz = z.detach().requires_grad_(True)
                with torch.enable_grad():
                    lf = log_f_ragged(self, zz, ix).sum()
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
                        lf = log_f_ragged(self, zz, ix).sum()
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
                 + log_f_ragged(self, zs, ix))
        lw = log_p - log_q
        lz = torch.logsumexp(lw, dim=1) - math.log(n_draws)
        if not return_ess:
            return lz
        with torch.no_grad():
            ww = torch.softmax(lw, dim=1)
            ess = 1.0 / (ww ** 2).sum(1) / n_draws
        return lz, ess

    def energy(self, line_item, line_trip, line_cat, B):
        """E(S) from the observed lines.  No assortment needed: the energy is a function of
        the basket alone."""
        dt, dev = self.lam.dtype, self.lam.device
        lin = torch.zeros(B, dtype=dt, device=dev).index_add_(
            0, line_trip, self.lam[line_item]
            + (self.theta[self.house[line_trip]] * self.alpha[line_item]).sum(-1))
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
               generator=None, return_ess=False):
        out = self.log_Z(ix, n_draws=n_draws, generator=generator, return_ess=return_ess)
        lz, ess = out if return_ess else (out, None)
        lzm1 = lz + torch.log1p(-torch.exp(-lz.clamp_min(1e-6)))
        ll = self.energy(line_item, line_trip, line_cat, ix.B) - lzm1
        return (ll, ess) if return_ess else ll
