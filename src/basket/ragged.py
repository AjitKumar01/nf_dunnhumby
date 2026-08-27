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

import numpy as np
import torch
from torch.nn.functional import softplus
from torch.autograd.function import once_differentiable


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


def _poly_mul_trunc_eager(A, G, nmax):
    """Reference subtraction-free convolution used by the fused autograd operator."""
    lead = torch.broadcast_shapes(A.shape[:-1], G.shape[:-1])
    LA, LG = A.shape[-1], G.shape[-1]
    L = min(LA + LG - 1, nmax + 1)
    out = torch.zeros(lead + (L,), dtype=A.dtype, device=A.device)
    for r in range(min(LG, L)):
        take = min(LA, L - r)
        out[..., r:r + take] = (out[..., r:r + take]
                                  + A[..., :take] * G[..., r:r + 1])
    return out


class _PolyMulTrunc(torch.autograd.Function):
    """Truncated polynomial product without a graph of thousands of CopySlices.

    PyTorch's generic autograd records every sliced update in the direct convolution.
    At the full catalogue shape those CopySlices dominate both memory traffic and backward
    time.  The derivative is another positive convolution, so save only A and G and apply
    that reverse rule explicitly.  Forward uses the reference arithmetic above verbatim.
    """

    @staticmethod
    def forward(ctx, A, G, nmax):
        ctx.save_for_backward(A, G)
        ctx.nmax = int(nmax)
        return _poly_mul_trunc_eager(A, G, ctx.nmax)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        A, G = ctx.saved_tensors
        lead = torch.broadcast_shapes(A.shape[:-1], G.shape[:-1])
        LA, LG, L = A.shape[-1], G.shape[-1], grad_out.shape[-1]
        Ae = A.expand(lead + (LA,))
        Ge = G.expand(lead + (LG,))
        dA = torch.zeros_like(Ae) if ctx.needs_input_grad[0] else None
        dG = torch.zeros_like(Ge) if ctx.needs_input_grad[1] else None
        for r in range(min(LG, L)):
            take = min(LA, L - r)
            go = grad_out[..., r:r + take]
            if dA is not None:
                dA[..., :take].add_(go * Ge[..., r:r + 1])
            if dG is not None:
                dG[..., r] = (go * Ae[..., :take]).sum(-1)
        if dA is not None and dA.shape != A.shape:
            dA = dA.sum_to_size(A.shape)
        if dG is not None and dG.shape != G.shape:
            dG = dG.sum_to_size(G.shape)
        return dA, dG, None


def _poly_mul_trunc(A, G, nmax):
    return _PolyMulTrunc.apply(A, G, int(nmax))


def _esp_product_tree(P, R):
    """Product of (1 + P_i x), truncated at R, with logarithmic item depth.

    The scalar ESP recursion is optimal in arithmetic but serial in the number of items.
    A 1,774-item category therefore launches 1,774 dependent tensor operations per QMC
    block.  Polynomial multiplication is associative: a balanced tree uses about 11
    rounds, exposes parallel work within each round, and remains subtraction-free.
    """
    one = torch.ones(P.shape + (1,), dtype=P.dtype, device=P.device)
    polys = torch.cat([one, P.unsqueeze(-1)], dim=-1)               # [..., rows, item, 2]
    while polys.shape[-2] > 1:
        if polys.shape[-2] % 2:
            ident = torch.zeros(polys.shape[:-2] + (1, polys.shape[-1]),
                                dtype=P.dtype, device=P.device)
            ident[..., 0, 0] = 1.0
            polys = torch.cat([polys, ident], dim=-2)
        A, B = polys[..., 0::2, :], polys[..., 1::2, :]
        polys = _poly_mul_trunc(A, B, R)
    out = polys.squeeze(-2)
    if out.shape[-1] < R + 1:
        out = torch.nn.functional.pad(out, (0, R + 1 - out.shape[-1]))
    return out


def esp_bucketed(w, row_of, n_rows, R, row_size, item_pos, buckets=(8, 32, 96, 256),
                 parallel=False):
    """e_0..e_R per row by the STABLE O(N R) recursion, without padding everything.

    Newton's identities build e_r from power sums as an alternating sum; at order 12 or 23
    the terms reach 1e30 and cancel catastrophically -- measured log Z errors of 1e62 and
    1e264 against the dense kernel, while the e_2 cancellation diagnostic still read a
    healthy 0.6.  The recursion e_r <- e_r + w_i e_{r-1} has no subtraction at all and is
    unconditionally stable at any R, but it needs a per-row loop over items.

    So rows are BUCKETED by size and padded only to their bucket's maximum.  A padded slot
    carries weight 0, which is invisible to the recursion, so no masking is needed.  The
    historical fixed buckets ended at 256 and therefore SILENTLY returned the identity
    polynomial for larger categories.  Dunnhumby's residual affinity category has 1,774
    products, so that omitted a material part of Z.  Extend the buckets geometrically to
    the actual maximum and assert that every row was covered.
    """
    lead = w.shape[:-1]
    out = torch.zeros(lead + (n_rows, R + 1), dtype=w.dtype, device=w.device)
    out[..., 0] = 1.0
    max_size = int(row_size.max().item()) if row_size.numel() else 0
    limits = sorted({int(x) for x in buckets if int(x) > 0})
    if not limits and max_size:
        limits = [max_size]
    while limits and limits[-1] < max_size:
        limits.append(min(max_size, 2 * limits[-1]))
    covered = torch.zeros(n_rows, dtype=torch.bool, device=w.device)
    lo = 0
    for hi in limits:
        sel_r = (row_size > lo) & (row_size <= hi)
        lo = hi
        if not bool(sel_r.any()):
            continue
        covered |= sel_r
        ridx = torch.nonzero(sel_r, as_tuple=True)[0]
        loc = torch.full((n_rows,), -1, dtype=torch.long, device=w.device)
        loc[ridx] = torch.arange(len(ridx), device=w.device)
        sel_i = sel_r[row_of]
        wi = w[..., sel_i]                                          # [..., T_b]
        flat = loc[row_of[sel_i]] * hi + item_pos[sel_i]            # [T_b]
        P = torch.zeros(lead + (len(ridx) * hi,), dtype=w.dtype, device=w.device)
        P = P.index_copy(-1, flat, wi).view(lead + (len(ridx), hi))
        # The r-loop this replaces counted DOWNWARD, from min(R, i+1) to 1, precisely so
        # that e[r] += x * e[r-1] always read the PRE-update e[r-1].  That is the proof
        # that the R updates within one item are mutually independent: there is no carried
        # dependence to break, only a Python loop that was serialising them.  Held as one
        # tensor E [..., n_rows, R+1] the whole inner loop is a single shifted multiply-add.
        #
        # The item loop stays sequential -- e at item i genuinely depends on item i-1.
        #
        # Launch count, with R = 23 and item i contributing min(R, i+1) updates:
        #     hi=8      36      hi=96    1,955
        #     hi=32    483      hi=256   5,635        total 8,109 -> 392 (20.7x fewer)
        # Each op is small ([16 draws x a few thousand rows]), so this loop was bound by
        # launch overhead rather than arithmetic, which is where fewer-and-larger wins.
        # Measured at the real shape (5,436 rows, 151k slots, D=16, R=23), all three
        # bit-identical to the sequential version and to each other:
        #     sequential 0.147s    cat 0.044s    pad 0.040s    bounded-cat 0.044s
        # so the padded shift wins and is also the shortest to read.  Restoring the
        # min(R, i+1) bound bought nothing: r > i+1 is still exactly zero, so the wider
        # update multiplies zeros rather than doing wrong work, and the extra slicing costs
        # more than the arithmetic it saves.  In-place (E[..., 1:] += ...) is NOT available
        # here -- esp_bucketed sits inside the autograd path and the multiply's saved
        # tensor is the very slice the add would overwrite.
        # A row containing at most ``hi`` products has e_r == 0 for every r > hi.
        # Carrying the global R=120 axis through the 8- and 32-product buckets was pure
        # zero arithmetic and became material once complete support replaced R=23.  Work
        # only through the bucket's attainable degree, then pad the proven zeros back so
        # downstream category convolution sees the unchanged public shape.
        local_R = min(R, hi)
        if parallel and hi >= 64:
            E = _esp_product_tree(P, local_R)
        else:
            E = torch.zeros(lead + (len(ridx), local_R + 1),
                            dtype=w.dtype, device=w.device)
            E[..., 0] = 1.0
            for i in range(hi):
                x = P[..., i].unsqueeze(-1)                         # [..., n_b, 1]
                E = E + x * torch.nn.functional.pad(E[..., :-1], (1, 0))
        if local_R < R:
            E = torch.nn.functional.pad(E, (0, R - local_R))
        out = out.index_copy(-2, ridx, E)
    missing = (row_size > 0) & ~covered
    if bool(missing.any()):
        bad = torch.nonzero(missing, as_tuple=True)[0][:8].tolist()
        raise RuntimeError(f"esp_bucketed failed to cover rows {bad}; max size={max_size}")
    return out


def cancellation(w, row_of, n_rows):
    """(p1^2 - p2)/p1^2 per row: how much of the leading term survives the subtraction."""
    p1 = seg_sum(w, row_of, n_rows)
    p2 = seg_sum(w ** 2, row_of, n_rows)
    return (p1 ** 2 - p2) / p1.pow(2).clamp_min(1e-300)


def poly_mul_trunc(A, G, nmax):
    """Multiply polynomials along the last axis, truncating at degree nmax."""
    return _poly_mul_trunc(A, G, nmax)


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
        P = _poly_mul_trunc(A, B, nmax)
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


def gh_grid(d, q):
    """Dense probabilists' Gauss-Hermite grid, q^d nodes, all weights POSITIVE.

    Smolyak weights are signed, which is fine for an expectation but not for forming a
    distribution to sample from.  Stage 1 of the sampler needs p(z) proportional to
    w_p f(z_p), so it uses a dense grid instead.
    """
    import itertools
    x, w = np.polynomial.hermite_e.hermegauss(q)
    w = w / math.sqrt(2 * math.pi)
    G = np.array(list(itertools.product(*[x] * d)), dtype=np.float64).reshape(-1, d)
    xi = {round(float(v), 12): i for i, v in enumerate(x)}
    W = np.ones(len(G))
    for k in range(d):
        W *= w[[xi[round(float(v), 12)] for v in G[:, k]]]
    return torch.as_tensor(G), torch.as_tensor(W)


def sobol_grid(d, n, seed=0, replicates=1):
    """Scrambled-Sobol nodes for E_{z~N(0,I_d)}[g(z)] -- n nodes at ANY dimension d.

    Same (nodes, weights) contract as gh_grid, so _log_Z_adaptive integrates with it
    unchanged; the weights are uniform 1/n because a QMC rule is an equal-weight average.

    This is what removes the two constraints that shaped the model.

    RANK.  A tensor grid costs q^d and Smolyak costs O(d^q), so Kz was held at 4.  Sobol
    costs n regardless of d.  Verified against exact subset enumeration at J=16, all
    products at grocery strength ||phi_j||=0.96, 16384 nodes:

            Kz        c      exact     error      sd(3)     sec
            32     3.30      4.003   -0.0004    0.00086    0.15
           128     3.77      4.328   +0.0008    0.00105    0.05
           256     3.73      4.367   +0.0005    0.00239    0.09
           512     3.63      4.225   -0.0003    0.00182    0.17

    Flat to 0.0005 nats at Kz=512, and no more expensive than Kz=32.  f(z) = F(Phi z)
    depends on z only through Phi z, so the EFFECTIVE dimension is rank(Phi) however large
    Kz is, which is the regime Sobol is built for.

    PRODUCTS.  The 20-product mask was protecting against c = max_u sum_j max(phi_j'u, 0),
    which grows with the catalogue.  c is a worst case over directions the mass never
    visits.  The mode solves the mean-field equation z* = Phi' pi(z*), so

        ||z*||  =  || sum_j pi_j phi_j ||  ~  rho sqrt(sum_j pi_j^2)  <=  rho E[n]

    and sum_j pi_j = E[n] is pinned near 8 by the size law whatever the catalogue size.
    Spreading the same E[n] over more products SHRINKS sum_j pi_j^2.  Measured, every
    product carrying phi at ||phi_j||=0.96, b recalibrated so E[n] = 8:

            J       Kz         c   ||z*||   sd(4 scrambles)    4k vs 64k
           20       32      4.33     1.91           0.00474     -0.00092
          100       32     14.03     0.62           0.00525     +0.00493
          500       64     40.13     0.27           0.00200     -0.00222

    c rises 10x, ||z*|| FALLS 7x, and the error does not move.  Adding products makes the
    integral easier.  The mask was budgeting against a bound that does not bind.

    WHAT DOES BIND is the per-product norm rho.  Failure is multimodality -- Z is exactly a
    Gaussian mixture with one component per subset, centred at mu_S = sum_{j in S} phi_j --
    and it is driven by rho, not by c or J.  Adversarial check, A products sharing ONE phi
    direction so nothing cancels and ||z*|| saturates its bound, exact enumeration at J=20:

            A      rho         c   ||z*||    exact     error
           12     0.96     12.69    11.95    74.66   -0.0001
            8     1.40     13.30    12.55    71.26   -0.0031
            8     2.00     19.00    18.44   143.57   +0.0644

    Twelve aligned products drive ||z*|| to 12 and c to 12.7 and it is still exact to
    0.0001, because alignment makes the integrand UNIMODAL, just displaced, and a
    mode-shifted rule absorbs a displacement exactly.  So the constraint to enforce is
    max_j ||phi_j|| <= ~1.4 (see fit.py --phi-norm-max), which unlike c does not scale with
    the catalogue.  Grocery needs 0.96.
    """
    replicates = int(replicates)
    if replicates < 1 or n % replicates:
        raise ValueError(f"qmc nodes ({n}) must be divisible by replicates ({replicates})")
    per = n // replicates
    if per < 2 or per & (per - 1):
        raise ValueError(
            f"nodes per Sobol scramble must be a power of two >= 2; got {per} "
            f"from {n} nodes / {replicates} replicates")
    blocks = []
    for r in range(replicates):
        # Separate Owen scrambles turn one uncheckable deterministic error into replicate
        # estimates.  The seeds are fixed across optimisation steps (common random numbers),
        # so this reduces approximation variance without injecting gradient noise.
        e = torch.quasirandom.SobolEngine(d, scramble=True, seed=seed + 104729 * r)
        blocks.append(e.draw(per).double())
    u = torch.cat(blocks).clamp(1e-12, 1.0 - 1e-12)
    x = torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)
    return x, torch.full((n,), 1.0 / n, dtype=torch.float64)


def sobol_mixture_grid(d, n, seed=0, replicates=1, components=2):
    """Normal Sobol blocks for equal-allocation deterministic-mixture RQMC.

    Keeping scramble and component as separate axes is important: after the proposal
    centres are added, the flattened order remains ``[replicate, component, node]``.
    Consequently each replicate is a complete mixture estimate and its dispersion is a
    valid, observable integration-error diagnostic.
    """
    replicates, components, n = int(replicates), int(components), int(n)
    if replicates < 1 or components < 1 or n % (replicates * components):
        raise ValueError(
            f"mixture QMC nodes ({n}) must be divisible by replicates*components "
            f"({replicates}*{components})")
    per = n // (replicates * components)
    if per < 2 or per & (per - 1):
        raise ValueError(
            "nodes per Sobol mixture component must be a power of two >= 2; "
            f"got {per} from {n}/({replicates}*{components})")
    blocks = torch.empty(replicates, components, per, d, dtype=torch.float64)
    for r in range(replicates):
        for c in range(components):
            e = torch.quasirandom.SobolEngine(
                d, scramble=True, seed=seed + 104729 * r + 13007 * c)
            blocks[r, c] = e.draw(per).double()
    u = blocks.clamp(1e-12, 1.0 - 1e-12)
    return torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)


def set_quad(model, quad_q=0, qmc_n=0, qmc_seed=0, Kz=None, probe=0,
             steps=2, chunk=0, qmc_reps=1, size_bands=0, size_steps=2,
             mode_logtol=8.0, mode_sep=1.0, mix_n=0, gh_cap=0):
    """ONE place that decides how log Z is integrated, for training and every eval.

    The scripts each built their own grid, and recommend_pi.py hardcoded smolyak_grid(4, 8)
    -- which against a Kz=128 checkpoint is not merely inaccurate, the nodes are the wrong
    SHAPE.  Every serious defect in this project has come from two code paths computing the
    same quantity differently (b_flat vs energy applying different terms; evalall's lo/hi
    passes; size_dist bypassing the quadrature), so the integrator is chosen here or
    nowhere.  Returns a one-line description for the log.
    """
    d = int(model.Kz if Kz is None else Kz)
    model.quad = model.quad_a = None
    model.quad_z = None
    model.quad_mix_a = None
    model.quad_replicates = 1
    model.quad_size_bands = 0
    if qmc_n > 0:
        model.quad_a = sobol_grid(d, int(qmc_n), seed=int(qmc_seed),
                                  replicates=int(qmc_reps))
        model.quad_probe = int(probe)
        model.quad_steps = int(steps)
        model.quad_chunk = int(chunk)
        model.quad_replicates = int(qmc_reps)
        model.quad_size_bands = int(size_bands)
        model.quad_size_steps = int(size_steps)
        model.quad_mode_logtol = float(mode_logtol)
        model.quad_mode_sep = float(mode_sep)
        model.quad_mix_n = int(mix_n) if int(mix_n) > 0 else 2 * int(qmc_n)
        if model.quad_size_bands:
            model.quad_mix_a = sobol_mixture_grid(
                d, model.quad_mix_n, seed=int(qmc_seed),
                replicates=int(qmc_reps), components=2)
            return (f"size-stratified multimode scrambled-Sobol, Kz={d}, "
                    f"{int(qmc_n)} one-mode/{model.quad_mix_n} two-mode nodes in "
                    f"{int(qmc_reps)} scrambles, seed {int(qmc_seed)}, "
                    f"{int(size_steps)} vectorised size-mode steps, "
                    f"second mode within {float(mode_logtol):g} nats and "
                    f"separation {float(mode_sep):g}"
                    + (f", node chunk {int(chunk)}" if chunk else ""))
        return (f"adaptive scrambled-Sobol, Kz={d}, {int(qmc_n)} nodes in "
                f"{int(qmc_reps)} scrambles, seed {int(qmc_seed)}, "
                f"{'unit scale in the Phi frame (no curvature probes)' if int(probe) < 0 else 'curvature along ' + ('top %d Phi directions' % int(probe) if probe else 'all Phi directions')}, "
                f"{int(steps)} mode steps"
                + (f", node chunk {int(chunk)}" if chunk else ""))
    if quad_q > 0:
        model.quad = smolyak_grid(d, int(quad_q))
        # Stage 1 of sample() needs p(z) proportional to w_p f(z_p), which requires
        # POSITIVE weights.  Smolyak's are signed, so the sampler cannot use them and was
        # silently falling through to a mode-shifted importance proposal -- the one the
        # quad_z comment records as unreliable ("sampled E[n] drifts 0.33x -> 1.29x as
        # n_draws goes 8 -> 1024").  Measured here: sampled E[n] went 4.7 -> 19.8 -> 120.0
        # -> nan as phi grew, while log Z stayed converged.  quad_z was written for exactly
        # this and was never populated by anything.
        #
        # A dense Gauss-Hermite grid costs q^Kz, so it is only built when that is
        # affordable; above the cap quad_z stays None and sample() keeps the old path,
        # which is at least explicit rather than accidental.
        #
        # The cap is 1024 nodes, not the normaliser's budget: stage 1 only has to pick z
        # from a discrete approximation to its posterior, and log_f_ragged over the grid
        # costs B x P slot evaluations.  At Kz=4 that is q=5 (625 nodes); q=8 would be
        # 4,096 and made the check hang.
        # quad_z is left OFF.  A dense GH grid is the right idea -- positive weights, so it
        # can serve as a sampling distribution where Smolyak's signed weights cannot -- but
        # measured against the analytic E[n] = 10.35 at run403's iteration 5000, a q=5 grid
        # (625 nodes) sampled 5.75 while the legacy proposal gave 10.67.  It is too coarse
        # to represent p(z), and a finer grid costs q^Kz.  Enable only with a measured
        # agreement check; shipping it as-is would replace a working sampler with a biased
        # one.  gh_cap is retained so the grid can be built for that check.
        model.quad_z = None
        if int(gh_cap) > 0:
            for _qz in range(int(quad_q), 2, -1):
                if _qz ** d <= int(gh_cap):
                    break
        _zdesc = ("" if model.quad_z is None else
                  f", sampling grid GH ({len(model.quad_z[0])} nodes)")
        return (f"Smolyak, Kz={d}, q={int(quad_q)}, {len(model.quad[0])} nodes" + _zdesc)
    return "SAMPLED (no deterministic rule) -- importance sampling, known wrong by 8-36 nats"


def smolyak_grid(d, q):
    """Smolyak sparse grid for E_{z~N(0,I_d)}[g(z)], by the combination technique:

        A(q,d) = sum_{q-d+1 <= |i| <= q} (-1)^{q-|i|} C(d-1, q-|i|) (U^{i_1} x ... x U^{i_d})

    with U^i the probabilists' Gauss-Hermite rule at level i (2i-1 nodes).  Verified against
    exact enumeration at ||phi_j|| = 0.96 (phi'phi = 0.92, a grocery complement lift of 2.5):

        Kz=2, q=6:   17 nodes, error -0.005 nats     Kz=4, q=7:  201 nodes, +0.021
        Kz=3, q=7:  105 nodes, error +0.009 nats     Monte Carlo, 4096 draws: 8-36 nats

    Weights are SIGNED; the caller must sum in linear space, not by logsumexp.
    """
    import itertools
    from math import comb

    def _rule(level):
        x, w = np.polynomial.hermite_e.hermegauss(2 * level - 1)
        return x, w / math.sqrt(2 * math.pi)

    def _comps(total, k):
        if k == 1:
            yield (total,)
            return
        for first in range(1, total - k + 2):
            for rest in _comps(total - first, k - 1):
                yield (first,) + rest

    acc = {}
    for total in range(max(d, q - d + 1), q + 1):
        c = (-1) ** (q - total) * comb(d - 1, q - total)
        if c == 0:
            continue
        for idx in _comps(total, d):
            gs = [_rule(k) for k in idx]
            for combo in itertools.product(*[range(len(g[0])) for g in gs]):
                node = tuple(round(float(gs[k][0][combo[k]]), 12) for k in range(d))
                wt = c * float(np.prod([gs[k][1][combo[k]] for k in range(d)]))
                acc[node] = acc.get(node, 0.0) + wt
    nodes = np.array(list(acc.keys()), dtype=np.float64).reshape(-1, d)
    wts = np.array(list(acc.values()), dtype=np.float64)
    keep = np.abs(wts) > 1e-14
    return (torch.as_tensor(nodes[keep], dtype=torch.float64),
            torch.as_tensor(wts[keep], dtype=torch.float64))


def sparse_prepare(model, ix, degree=None, nmax=None):
    """Everything in log f that does not depend on z.

    Only phi-carrying products depend on z: for phi_j = 0 the weight is exp(b_j) whatever
    z is.  With a mask of ~20 products out of 5,455 that is 99.6% of the elementary
    symmetric polynomial work, currently recomputed once per quadrature node.  The scale is
    taken z-INDEPENDENTLY (seg_max over bt alone) so the inactive ESP can be built once;
    that is exact -- shifting the scale is compensated by the n*M term in lg -- and safe
    numerically because |phi_j'z| <= ||phi_j|| * 3.75 on the Smolyak grid.
    """
    # Per-row polynomial degree.  This is NOT the support: --R declares which trips are
    # in the data, while this only bounds how many items ONE affinity row may contribute to
    # the polynomial.  Measured over all 198,690 trips the largest single-row count is 26,
    # and the model's own posterior gives E[n_c] ~ 0.024, so a degree of 32 is non-binding:
    # max |d log Z| = 9.4e-5 nats against degree 120, which is 50x below the estimator's own
    # sd.  It is worth 1.8-2.0x end to end because poly_tree -- 59% of log f -- scales as
    # degree^2.
    R = model.R if degree is None else min(int(degree), model.R)
    if degree is None and getattr(model, "poly_degree", 0):
        R = min(int(model.poly_degree), model.R)
    nmax = model.nmax if nmax is None else min(int(nmax), model.nmax)
    phi_i = model.phi[ix.item]
    _b0 = model.b_flat(ix)
    _phi_sq = (phi_i ** 2).sum(-1)
    bt = _b0 - 0.5 * _phi_sq                                           # [T]
    act = phi_i.norm(dim=-1) > 1e-12                                   # [T]
    M0 = seg_max(bt.unsqueeze(0), ix.item_trip, ix.B)                  # [1, B]
    sh = M0[0].index_select(0, ix.item_trip)                           # [T]
    w0 = torch.where(act, torch.zeros_like(bt), torch.exp(bt - sh))
    all_active = not bool((~act).any())
    if not all_active:
        e0 = esp_bucketed(w0.unsqueeze(0), ix.row_of, ix.n_rows, R,
                          ix.row_size, ix.item_pos)                    # [1, n_rows, R+1]
    else:
        # Full-catalogue phi makes every inactive polynomial the identity.  Walking all
        # 5,455 zero weights (including the 1,774-item residual row) twice per call only to
        # rediscover [1,0,...] was a sizeable fixed cost.
        e0 = torch.zeros(1, ix.n_rows, R + 1,
                         dtype=w0.dtype, device=w0.device)
        e0[..., 0] = 1.0
    ai = torch.nonzero(act, as_tuple=True)[0]                          # active slots
    arow = ix.row_of[ai]
    urow, inv = torch.unique(arow, return_inverse=True)                # rows touched
    # Dense-by-trip projection layout.  Dunnhumby assortments contain a mean 5,135 of the
    # 5,455 catalogue products, so padding each trip to its largest assortment wastes only
    # a few percent.  A batched GEMM then computes Phi z without materialising the old
    # [active_slots, nodes, Kz] multiply (multiple GiB at B=24, Kz=32).
    atrip = ix.item_trip[ai]
    atpos = torch.zeros(len(ai), dtype=torch.long, device=ai.device)
    atmax = 0
    if len(ai) > 0:
        ao = torch.argsort(atrip * (len(ai) + 1) + torch.arange(len(ai), device=ai.device))
        ats = atrip[ao]
        atcnt = torch.bincount(ats, minlength=ix.B)
        ast = torch.zeros(ix.B, dtype=torch.long, device=ai.device)
        ast[1:] = torch.cumsum(atcnt, 0)[:-1]
        atpos[ao] = torch.arange(len(ai), device=ai.device) - ast[ats]
        atmax = int(atcnt.max().item())
    dense_active = bool(atmax and len(ai) >= 0.5 * ix.B * atmax)
    if dense_active:
        aflat_trip = atrip * atmax + atpos
        apad = torch.zeros(ix.B * atmax, model.Kz,
                           dtype=phi_i.dtype, device=phi_i.device)
        apad = apad.index_copy(0, aflat_trip, phi_i[ai]).view(ix.B, atmax, model.Kz)
    else:
        apad = None
    # position of each active slot within its row's active list
    ordv = torch.argsort(inv * (len(ai) + 1) + torch.arange(len(ai), device=ai.device))
    inv_s = inv[ordv]
    pos = torch.zeros(len(ai), dtype=torch.long, device=ai.device)
    if len(ai) > 0:
        start = torch.zeros(len(urow), dtype=torch.long, device=ai.device)
        cnt = torch.bincount(inv_s, minlength=len(urow))
        start[1:] = torch.cumsum(cnt, 0)[:-1]
        pos[ordv] = torch.arange(len(ai), device=ai.device) - start[inv_s]
    acnt = torch.bincount(inv, minlength=len(urow))
    kmax = int(pos.max().item()) + 1 if len(ai) > 0 else 0
    # The category convolution is 77% of log f (profiled: poly_tree 1.714s of 2.230s), and
    # a category containing no phi product has a z-INDEPENDENT polynomial.  So split
    #     A(z) = A_const  *  A_active(z)
    # and build A_const once over the ~250 inactive categories, leaving the per-node tree
    # to run over the handful that phi actually touches.
    r_ = torch.arange(R + 1, dtype=e0.dtype, device=e0.device)
    a_ = torch.exp(-model.rho_c[ix.row_cat].unsqueeze(-1) * model.pair_feature(r_))
    G0 = a_.unsqueeze(0) * e0                                          # [1, n_rows, R+1]
    aflat = ix.flat_slot[urow] if len(ai) > 0 else ix.flat_slot[:0]
    const_identity = len(urow) == ix.n_rows
    if const_identity:
        # Every category is z-dependent at full catalogue coverage, hence the constant
        # half is exactly the identity.  Avoid building and tree-multiplying thousands of
        # row polynomials that are immediately overwritten by identities.
        A_const = torch.zeros(1, ix.B, nmax + 1,
                              dtype=e0.dtype, device=e0.device)
        A_const[..., 0] = 1.0
    else:
        Gc = torch.zeros(1, ix.B * ix.Cpad, R + 1,
                         dtype=e0.dtype, device=e0.device)
        Gc[:, :, 0] = 1.0
        Gc = Gc.index_copy(1, ix.flat_slot, G0)
        if len(aflat) > 0:                   # active categories -> identity in const half
            Gc[:, aflat, :] = 0.0
            Gc[:, aflat, 0] = 1.0
        A_const = poly_tree(Gc.view(1, ix.B, ix.Cpad, R + 1), nmax)
    ab = (aflat // ix.Cpad) if len(aflat) > 0 else aflat
    if len(aflat) > 0:
        o = torch.argsort(ab * (len(ab) + 1) + torch.arange(len(ab), device=ab.device))
        ab_s = ab[o]
        cnt = torch.bincount(ab_s, minlength=ix.B)
        st = torch.zeros(ix.B, dtype=torch.long, device=ab.device)
        st[1:] = torch.cumsum(cnt, 0)[:-1]
        acol = torch.zeros(len(ab), dtype=torch.long, device=ab.device)
        acol[o] = torch.arange(len(ab), device=ab.device) - st[ab_s]
        cpad_a = int(cnt.max().item())
    else:
        acol, cpad_a = ab, 0
    return dict(bt=bt, b0=_b0, phi_sq=_phi_sq, M0=M0, sh=sh, e0=e0, ai=ai, urow=urow, inv=inv, pos=pos,
                acnt=acnt, kmax=kmax, a_row=a_, A_const=A_const, ab=ab, acol=acol,
                cpad_a=cpad_a, atrip=atrip, atpos=atpos, atmax=atmax, apad=apad,
                inactive_identity=all_active, const_identity=const_identity, R=R)


def size_band_scales(nmax, n_bands, pool="mean", n_ref=7.8):
    """Piecewise-constant pair scale s(n), as (lo, hi, s) covering 0..nmax.

    The interaction is  (1/s) * sum_{j<k in S} phi_j.phi_k.  With pool="sum" and one band
    s == 1 and this is exactly the current version-4 law.  With pool="mean", s = n - 1, so
    each item interacts with the MEAN of the others rather than the sum.

    Why mean pooling.  Measured on the run155 checkpoint with real b, real rho_c and real
    per-trip assortments (median 5,340 candidates), fitting ONLY the interaction on the
    leave-one-out conditional:

        form   best held-out CE   vs no interaction
        sum            7.9082           +0.0000        <- never improves, any lr
        mean           7.8827           +0.0255

    A sum gives a basket of size |R| an interaction |R| times stronger, so large baskets
    dominate the gradient and force phi toward zero rather than let it specialise; the size
    law (rho_0, rho_c) already controls size.  A mean is scale-free.  That matches where the
    deficit actually is -- composition, not size.

    Why bands.  Hubbard-Stratonovich needs exp(a||v||^2/2) = E_z exp(sqrt(a) z'v), so a
    size-dependent a = 1/s(n) means a size-dependent z-scale, and the ESP recursion derives
    every e_n from ONE weight vector.  Exact mean pooling would need one recursion per n
    (120x).  Holding s constant within a band needs one per BAND -- and because the energy
    uses the same s(n), the banded model is an EXACT law, not an approximation of one.
    Bands are geometric: basket sizes run 1..103 with median 6.4, so resolution is needed
    at small n and not at large.
    """
    nmax = int(nmax)
    if pool == "sum" or n_bands <= 1:
        return [(0, nmax, 1.0)]
    edges, lo = [], 1
    for b in range(int(n_bands)):
        hi = int(round(nmax ** ((b + 1) / float(n_bands))))
        hi = max(hi, lo)
        if b == int(n_bands) - 1:
            hi = nmax
        if hi >= lo:
            edges.append((lo, hi))
            lo = hi + 1
        if lo > nmax:
            break
    # NORMALISE so a typical basket has s = 1.
    #
    # w_j = exp(b_j - ||phi_j||^2/(2s) + phi_j'z/sqrt(s)) is EXACTLY sum pooling with
    # phi~ = phi/sqrt(s).  So an unnormalised s rescales the effective interaction, and the
    # --phi-max cap then bites as phi_max/sqrt(s): at s=10.7 that is 0.29 against the sum
    # model's 0.96, i.e. a 12x weaker interaction.  Measured, armH's phi sat at max 0.17 and
    # did not grow -- the cap, not the pooling, was binding.
    #
    # Dividing by s(E[n]) keeps the typical basket at the sum model's strength and leaves
    # only what the hypothesis is actually about: the SHAPE across sizes -- small baskets
    # coupled more strongly than large ones.  It also leaves the estimator envelope
    # unchanged, since z is scaled by 1/sqrt(s) in the same proportion.
    s_ref = max(float(n_ref) - 1.0, 1.0)
    bands = []
    for i, (a, b) in enumerate(edges):
        s = math.sqrt(max(a - 1, 1) * max(b - 1, 1)) / s_ref
        # Floor s at 0.5.  s < 1 AMPLIFIES phi by 1/sqrt(s), and the estimator was verified
        # exact to 1e-4 nats only up to a per-product norm of ~1.4, degrading past 2.0.
        # Unclamped, the smallest band reached phi_eff = 2.6x and the enumeration check went
        # from 2e-5 to 3e-3 nats.  A floor of 0.5 holds phi_eff <= phi_max/0.707 = 1.36,
        # inside the verified envelope, and costs only the very smallest baskets a little
        # extra coupling they have few pairs to use anyway.
        bands.append((a if i else 0, b, float(max(s, 0.5))))   # band 0 also carries n=0
    return bands


def log_f_banded(model, z, ix, C, bands, drop_empty=False, nmax=None,
                 detach_params=False):
    """log f(z) with a piecewise-constant pair scale.

    One ESP pass per band, keeping only that band's degrees.  The z-scale and the
    -||phi||^2/(2s) offset are the only things that change between bands, so the expensive
    z-independent cache from sparse_prepare is built once and reused.
    """
    nmax = model.nmax if nmax is None else int(nmax)
    b0 = C["b0"]
    phi_sq = C["phi_sq"]
    parts = []
    for (lo, hi, s) in bands:
        if lo > nmax:
            break
        Cb = dict(C)
        Cb["bt"] = b0 - phi_sq / (2.0 * s)
        M0 = seg_max(Cb["bt"].unsqueeze(0), ix.item_trip, ix.B)
        Cb["M0"] = M0
        Cb["sh"] = M0[0].index_select(0, ix.item_trip)
        lg = log_f_sparse(model, z / math.sqrt(s), ix, Cb, drop_empty=False,
                          return_terms=True, detach_params=detach_params, nmax=nmax)
        parts.append(lg[..., lo:min(hi, nmax) + 1])
    lg = torch.cat(parts, dim=-1)
    if drop_empty:
        lg = lg[..., 1:]
    return torch.logsumexp(lg, dim=-1).transpose(0, 1)


def _log_f_dispatch(model, z, ix, C, drop_empty=False, **kw):
    """One entry point so a banded model cannot be integrated with the sum-pooled kernel."""
    if getattr(model, "size_bands", None):
        return log_f_banded(model, z, ix, C, model.size_bands, drop_empty=drop_empty, **kw)
    return log_f_sparse(model, z, ix, C, drop_empty=drop_empty, **kw)


def log_f_sparse(model, z, ix, C, drop_empty=False, return_terms=False,
                 detach_params=False, nmax=None):
    """log f(z) using the z-independent cache from sparse_prepare.  Same value as
    log_f_ragged, with the ESP over non-phi products lifted out of the node loop."""
    D = z.shape[1]
    nmax = model.nmax if nmax is None else int(nmax)
    R, dt, dev = C.get("R", model.R), C["e0"].dtype, C["e0"].device
    ai, urow, inv, pos = C["ai"], C["urow"], C["inv"], C["pos"]
    node_M = C["M0"]
    if len(ai) > 0:
        if C["apad"] is not None:
            apad = C["apad"].detach() if detach_params else C["apad"]
            proj_pad = torch.bmm(apad, z.transpose(1, 2))               # [B, Tmax, D]
            proj = proj_pad[C["atrip"], C["atpos"]]                    # [Ta, D]
        else:
            phi_a = model.phi[ix.item[ai]]                              # [Ta, Kz]
            if detach_params:
                phi_a = phi_a.detach()
            proj = (z[ix.item_trip[ai]] * phi_a.unsqueeze(1)).sum(-1)   # [Ta, D]
        if C["inactive_identity"]:
            # With every product active there is no cached inactive polynomial whose scale
            # must remain fixed.  Re-centre the weights separately at every (node, trip),
            # exactly as log_f_ragged does.  The former z-independent scale was safe on the
            # old radius-4 Smolyak grid but overflows at a legitimate remote mode: degree-120
            # coefficients can exceed float64 even when log f itself is only O(100).
            #
            # Scaling every weight in a trip by exp(-M) scales the degree-n coefficient by
            # exp(-nM), so adding n*M below is an exact algebraic inverse, not a clamp.
            logwa = C["bt"][ai].unsqueeze(1) + proj                    # [Ta, D]
            node_M = seg_max(logwa.transpose(0, 1), C["atrip"], ix.B) # [D, B]
            wa = torch.exp(logwa - node_M[:, C["atrip"]].transpose(0, 1))
        else:
            wa = torch.exp(C["bt"][ai].unsqueeze(1)
                           - C["sh"][ai].unsqueeze(1) + proj)
        # Bucket the ACTIVE slots themselves.  The old rectangular allocation was
        # [nodes, active_rows, max_active_row_size]; with all products active, a single
        # 1,774-product residual category forced every other row to that width (34+ GiB at
        # the real batch/node shape).  This representation is proportional to the actual
        # active slots plus modest bucket padding.
        Ea = esp_bucketed(wa.transpose(0, 1), inv, len(urow), R,
                          C["acnt"], pos, parallel=True)
        # combined row polynomial = (inactive ESP) * (active ESP), truncated at R
        if C["inactive_identity"]:
            # At full-catalogue coverage e0 is exactly (1,0,...), so convolution with it
            # is the identity.  The generic loop performed R+1 sliced multiply-adds for
            # every mode, curvature and QMC node block to reproduce Ea unchanged.
            conv = Ea
        else:
            base = C["e0"][0, urow]                                     # [n_arow, R+1]
            conv = torch.zeros(D, len(urow), R + 1, dtype=dt, device=dev)
            for r in range(R + 1):
                conv[..., r:] = (conv[..., r:]
                                  + base[:, r].unsqueeze(0).unsqueeze(-1)
                                  * Ea[..., : R + 1 - r])
    if C["cpad_a"] == 0:
        A = C["A_const"][..., :nmax + 1].expand(D, -1, -1)
    else:
        Ga = C["a_row"][C["urow"]].unsqueeze(0) * conv          # [D, n_arow, R+1]
        Gz = torch.zeros(D, ix.B, C["cpad_a"], R + 1, dtype=dt, device=dev)
        Gz[:, :, :, 0] = 1.0
        Gz[:, C["ab"], C["acol"]] = Ga
        A_active = poly_tree(Gz, nmax)
        # Likewise A_const is exactly the identity when every category is active.  This
        # skips a 121-term polynomial convolution in the common all-5,455-products path.
        A = (A_active if C["const_identity"] else
             poly_mul_trunc(A_active, C["A_const"], nmax))
    n = torch.arange(A.shape[-1], dtype=dt, device=dev)
    rho0 = model.rho_0()[: A.shape[-1]]
    if detach_params:
        rho0 = rho0.detach()
    lg = (torch.log(A.clamp_min(1e-300)) - rho0
          + n * node_M.unsqueeze(-1))
    if drop_empty:
        lg = lg[..., 1:]
    if return_terms:
        return lg
    return torch.logsumexp(lg, dim=-1).transpose(0, 1)


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
    # Honour the per-row degree cap here too.
    #
    # log_f_sparse takes its degree from sparse_prepare (model.poly_degree, calibrated to
    # 48), but this dense path used model.R = 120 -- two implementations of one quantity,
    # disagreeing.  It matters because G carries exp(-rho_c r(r-1)/2) and 224 of 280 rho_c
    # are negative: at r=120 with rho_c = -0.1126 that is exp(804) = inf, so log(A_n) = inf
    # and sample()'s size draw raised "probability tensor contains inf, nan or element < 0".
    # The normaliser never saw it because its degree cap stops at 48 (exp(127), finite).
    _R = model.R
    if getattr(model, "poly_degree", 0):
        _R = min(int(model.poly_degree), model.R)
    e = esp_bucketed(w, ix.row_of, ix.n_rows, _R, ix.row_size, ix.item_pos)
    r = torch.arange(_R + 1, dtype=w.dtype, device=w.device)
    a = torch.exp(-model.rho_c[ix.row_cat].unsqueeze(-1) * model.pair_feature(r))
    G = a.unsqueeze(0) * e                                          # [D, n_rows, R+1]
    # scatter rows into [D, B, Cpad, R+1]; missing rows are the identity polynomial
    Gp = torch.zeros(D, ix.B * ix.Cpad, _R + 1, dtype=w.dtype, device=w.device)
    Gp[:, :, 0] = 1.0
    Gp = Gp.index_copy(1, ix.flat_slot, G).view(D, ix.B, ix.Cpad, _R + 1)
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
                 S=1, Kp=8, Kt=8, Ks=4, n_week=53, phi_init=0.03,
                 taste_init=0.3):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.J, self.N, self.C, self.S = J, N, C, S
        self.K, self.Kz, self.nmax, self.R = K, Kz, nmax, R
        self.lam = torch.nn.Parameter(torch.zeros(J))
        self.alpha = torch.nn.Parameter(torch.randn(J, K, generator=g) * taste_init)
        self.theta = torch.nn.Parameter(torch.randn(N, K, generator=g) * taste_init)
        # ||phi_j|| ~ phi_init * sqrt(Kz).  At the old 0.15 with Kz = 12 that was 0.52,
        # against a cap of 0.20 -- the model began 2.6x outside the region section 14
        # requires it to stay in, and the first projection yanked it back.  Since
        # lambda_max <~ ||phi||^2 E[n], starting at 0.52 with E[n] ~ 7.8 means lambda_max
        # ~ 2.1 at step zero: no unique mode, and a fixed-point iteration that diverges
        # rather than converges.  0.03 puts ||phi|| at 0.10, half the cap.
        self.phi = torch.nn.Parameter(torch.randn(J, Kz, generator=g) * phi_init)
        self.rho_c = torch.nn.Parameter(torch.zeros(C))
        # Version-4's foundational energy uses rho_c*k(k-1)/2 on the ENTIRE declared
        # support.  Saturating at the historical R=23 implementation limit changes the
        # joint law and its conditional logits.  Keep the original quadratic through
        # nmax; numerical safety is enforced by the rho_c floor.  At rho_c=-0.92 and
        # nmax=120 the largest exponent is 0.92*C(120,2)=656.9, within float64.
        self.rho_pair_cap = nmax
        self.rho_0_free = torch.nn.Parameter(torch.zeros(nmax))
        # Optional exact factorisation P(S|x)=P(S||S|,x)P_size(|S||x).  The composition
        # normaliser Z_n(x) is still supplied by this model; only the unstable coupling of
        # all n through one context-free rho_0 is replaced.  Buffers make the statistical
        # model self-describing in a checkpoint and add no fitted parameters.
        self.register_buffer("factored_size_enabled", torch.tensor(False))
        self.register_buffer("factored_size_log_p", torch.full((nmax,), -math.log(nmax)))
        # softplus(0.5413) = 1.0: starts as the un-split model, so a resumed run is unchanged
        self.price_kappa = torch.nn.Parameter(torch.tensor(0.5413))
        self.quad = None        # (nodes, weights) -> deterministic log Z
        self.quad_a = None      # positive-weight rule for ADAPTIVE (mode-centred) log Z
        self.quad_mix_a = None  # [replicate, component, node, Kz] multimode Sobol blocks
        self.quad_probe = 0     # 0 = all Phi eigendirections; n = leading n directions
        self.quad_chunk = 0     # 0 = all nodes at once; n = accumulate n at a time
        self.size_bands = None  # [(lo, hi, s)] piecewise pair scale; None = sum pooling
        self.cv_nodes = 0       # >0 -> Laplace-tilted IS instead of the node grid
        self.cv_seed = 0
        self.poly_degree = 0    # 0 = use R; else per-row polynomial degree cap
        self.price_soft = False # unconstrained price bilinear + hinge penalty
        self._last_gb = None
        self.quad_replicates = 1
        # A single zero-start mode can become blind to a remote large-basket basin.  The
        # optional size-stratified rule screens several basket-size bands together, then
        # spends QMC nodes only on the dominant basin (or two close basins).
        self.quad_size_bands = 0
        self.quad_size_steps = 2
        self.quad_mode_logtol = 8.0
        self.quad_mode_sep = 1.0
        self.quad_mix_n = 0
        self._last_qmc_logz_se = None
        self._last_qmc_mode_count = None
        self._last_qmc_mode_gap = None
        self._last_qmc_mode_sep = None
        self._last_qmc_curv_min = None
        self._last_qmc_scale_max = None
        # Mode-ascent iterations for the ADAPTIVE rule.  --mode-steps fed the sampler,
        # which is inert under quadrature, so this path silently always ran 8.  Measured
        # against exact enumeration at Kz=128, 512 nodes (error / ms):
        #     0 steps  -0.0349 / 1.9      2 steps  -0.0095 / 4.0      8 steps  -0.0039 / 10.6
        #     1 step   -0.0165 / 3.0      4 steps  -0.0052 / 6.2
        # Monotone -- the ascent is doing real work, not spinning -- but 4 buys 1.7x the
        # speed for 0.0013 nats, well inside the ~0.01 tolerance the rest of this operates at.
        self.quad_steps = 4
        self.quad_share = False  # one shift/scale for the batch (shared nodes)
        self.quad_z = None      # dense GH grid -> deterministic z sampling
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

    def pair_feature(self, count):
        k = count.clamp(max=self.rho_pair_cap)
        return k * (k - 1) / 2.0

    def pair_increment(self, count_before):
        """g(k+1)-g(k) for version-4's g(k)=choose(k,2) on declared support."""
        return torch.where(count_before < self.rho_pair_cap,
                           count_before, torch.zeros_like(count_before))

    def price_g(self):
        """Household price-sensitivity factor.  ONE definition, used everywhere.

        The coefficient is <price_g[h], price_b[j]> and it appeared in six places: b_at,
        the elasticity penalty, the beta calibration, the pooling penalty and two
        diagnostics.  Under --price-soft the parameterisation changes, and when only b_at
        was switched the elasticity penalty kept computing softplus(gamma)*softplus(beta) --
        a different quantity -- which inflated the training loss to 23,321.  Same failure
        mode this file already records for b_flat vs energy: two implementations of one
        thing, drifting apart.
        """
        return self.gamma if getattr(self, "price_soft", False) else softplus(self.gamma)

    def price_b(self):
        """Item price-sensitivity factor.  See price_g."""
        return self.beta if getattr(self, "price_soft", False) else softplus(self.beta)

    def b_at(self, it, trip, c):
        """Eq. 7 at an arbitrary set of (product, trip) pairs.

        ONE function for both the normaliser and the energy.  They previously had separate
        code paths and drifted: b_flat applied price, promotion, seasonality and store while
        energy() applied only taste, so E(S) and log Z scored the same product differently
        and the difference was free reward for the optimiser.  Nothing may compute an item
        value except through here.
        """
        hh = self.house[trip]
        b = self.lam[it] + (self.theta_c()[hh] * self.alpha[it]).sum(-1)
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
        # PRICE, split into a common level and an idiosyncratic deviation.
        #
        #     dlp_j = m + e_j        m = the trip's mean over its assortment
        #
        # m shifts every b_j equally, so Proposition 1 applies and dE[n]/dm is amplified by
        # Var(n) -- measured 10.5x, and CORRECT, since the data's own dispersion is 10.55.
        # e_j shifts b_j differentially: it moves share between products and leaves the sum
        # nearly untouched, so it is not amplified.
        #
        # One coefficient served both, so pinning the aggregate elasticity (-0.121) pinned
        # the per-product response too: gamma.beta = 0.121/10.5 = 0.0115, against a data
        # own-price response near -0.66.  The whole price effect then had to live in the
        # units model, and a coupon could not change WHETHER a product was bought.
        #
        # kappa scales the idiosyncratic part alone.  The aggregate stays -(gamma.beta)*
        # Var/E and remains pinned by --elast-w; the share response becomes
        # -(gamma.beta)*kappa and is free.  The data supports the split: basket size against
        # the common level gives -0.042, while share against relative price gives -0.078
        # with t = -154.
        # Price sensitivity: hard non-negativity (softplus) or UNCONSTRAINED + hinge penalty.
        #
        # softplus enforces a non-positive own-price response by construction, but softplus(x)
        # AND softplus'(x) both vanish as x -> -inf, so any component driven negative is
        # trapped in a flat region and cannot return.  Measured on a real checkpoint: 100% of
        # gamma and 99.9% of beta sit below -3 (softplus' < 0.047), the block's gradient norm
        # is 5.6e-4 against 0.05-2.2 for every other block, and the fitted price coefficient
        # is 0.0099 -- no price effect at all.  Tellingly, price_kappa, the one price
        # parameter NOT wrapped in softplus, holds the largest available gain in the model.
        #
        # price_soft replaces the hard constraint with an unconstrained bilinear form plus a
        # hinge penalty relu(-gamma.beta)^2 in the objective (see fit.py --price-hinge-w).
        # Same economics asymptotically, but the gradient never dies, so the block can move.
        _gb = (self.price_g()[hh] * self.price_b()[it]).sum(-1)
        self._last_gb = _gb
        if "dlp_bar" in c:
            # Trip-level references are one scalar per trip and must be gathered by trip;
            # category-level ones are already aligned with the elements being scored.
            _db = c["dlp_bar"]
            _m = _db if _db.shape[0] == it.shape[0] else _db[trip]
            b = b - _gb * (_m + softplus(self.price_kappa) * (c["dlp"] - _m))
        else:
            b = b - _gb * c["dlp"]
        b = b + self.w_dsp[it] * c["disp"] + self.w_mlr[it] * c["mail"]
        b = b + (self.mu[it] * self.delta_c()[c["week"]]).sum(-1)
        b = b + (self.zeta[it] * self.xi_c()[c["store"]]).sum(-1)
        if "rec" in c:
            b = b + (self.psi[it] * c["rec"]).sum(-1)
        return b

    # ---- GAUGE FIXING on the three bilinear terms -------------------------------------
    #
    # b contains lam_j + theta_h'alpha_j + mu_j'delta_w + zeta_j'xi_s.  For any fixed a,
    #
    #     lam_j -> lam_j + mu_j'a ,   delta_w -> delta_w - a
    #
    # leaves b EXACTLY invariant, for every product and every week.  So the likelihood is
    # flat along an 8-dimensional family and cannot decide how much per-product intercept
    # sits in lam versus in the seasonal term.  Only the penalties decide, and they are of
    # different degree.  Writing mu_j = (v_j/s).unit(delta_bar) to store an intercept v:
    #
    #     pool_ctx on mu   2.0 * ||v-vbar||^2 / (J*Kp*s^2)      falls with s
    #     wd on delta      (wd/2) * W * s^2                     rises with s
    #     minimised over s at 2*sqrt(AB) = 2.20e-4 * ||v-vbar|| -- LINEAR in ||v||
    #
    # while the same intercept in lam costs (wd/2)||v||^2 -- QUADRATIC.  Two factors of one
    # product turn squared penalties on the factors into a nuclear-norm penalty on the
    # product, which is degree 1.  Crossover at ||v|| = 44; the measured net intercept is
    # ||c|| = 244, so essentially all of it belongs in the bilinear term at the optimum, and
    # lam saturates at ||lam|| = 2.20e-4/wd = 22.0, i.e. std 0.298 REGARDLESS of ||c||.
    # run73 measured lam std 0.173 -> 0.311 flattening, against 0.298 predicted.  That is
    # why raising --pool-ctx never worked: it changes the coefficient, not the exponent, and
    # the optimiser evades it outright by shrinking mu and growing delta.
    #
    # Centring the CONTEXT side removes the flat direction and makes lam the unique owner of
    # the per-product constant.  It is gauge fixing, not regularisation: no signal is
    # deleted, and expressiveness strictly INCREASES, because the constant channel was
    # mu_j'delta_bar, confined to a <=Kp-dimensional subspace of R^J, and is now lam_j, free
    # in all J.  Applied to a running model it would need lam += mu'delta_bar at the same
    # instant to keep b invariant; from scratch both start at zero and no transfer is needed.
    #
    # theta'alpha needs this too.  It is the same shape -- a product of TWO trained tensors
    # -- so centring only delta and xi would let the intercept migrate into taste instead of
    # lam, and the fix would silently fail.  psi'rec does NOT need it: rec is fixed data, so
    # psi enters linearly, is penalised quadratically, and has no escape route.
    def theta_c(self):
        return self.theta - self.theta.mean(0, keepdim=True)

    def delta_c(self):
        return self.delta - self.delta.mean(0, keepdim=True)

    def xi_c(self):
        return self.xi - self.xi.mean(0, keepdim=True)

    def b_flat(self, ix):
        """b at every assortment slot, [T] -- the normaliser's view."""
        if getattr(self, "_b_override", None) is not None:
            return self._b_override
        return self.b_at(ix.item, ix.item_trip, self.ctx)

    def pi_quad(self, ix):
        """pi_j = P(j in S), MARGINAL over z, by autograd through the quadrature.

        Corollary 2 gives pi_j = d log(Z-1) / d b_j.  pi_exact evaluates it at the MODE,
        which is a different quantity: measured, E[n | z = zhat] was 2.8 against a marginal
        17.7, and lambda_max -- built on it -- was wrong by that factor for the whole
        project.  With log Z deterministic the marginal is just a backward pass, and
        sum_j pi_j = E[n] holds by construction rather than by hope.
        """
        b0 = self.b_flat(ix).detach().requires_grad_(True)
        self._b_override = b0
        try:
            with torch.enable_grad():
                lz = self.log_Z(ix, drop_empty=True)
            pi = torch.autograd.grad(lz.sum(), b0)[0].detach()
        finally:
            self._b_override = None
        return pi

    def _adaptive_frame(self, ix, drop_empty, mode_steps, cache=None):
        """Detached mode and active-subspace curvature frame shared by Z and sampling.

        ``sparse_prepare`` is fairly expensive at full-catalogue coverage.  Callers that
        also need the differentiable cache may pass it here; the proposal sees a detached
        view of exactly those tensors, so this removes a duplicate cache construction
        without adding a proposal-gradient path or changing the numerical rule.
        """
        B = ix.B
        if cache is None:
            with torch.no_grad():
                C_mode = sparse_prepare(self, ix)
        else:
            C_mode = {k: (v.detach() if torch.is_tensor(v) else v)
                      for k, v in cache.items()}
        z = torch.zeros(B, 1, self.Kz, dtype=self.lam.dtype, device=self.lam.device)
        for _ in range(mode_steps):
            zz = z.detach().requires_grad_(True)
            with torch.enable_grad():
                obj = (log_f_sparse(self, zz, ix, C_mode, drop_empty,
                                    detach_params=True)
                       - 0.5 * (zz ** 2).sum(-1)).sum()
                gz = torch.autograd.grad(obj, zz)[0]
            z = (zz + 0.5 * gz).detach()
        zh = z.detach()

        with torch.no_grad():
            present = torch.unique(ix.item)
            phi_u = self.phi.index_select(0, present).detach()
            gram = phi_u.transpose(0, 1) @ phi_u
            evals, Q = torch.linalg.eigh(gram)
            Q = Q.index_select(1, torch.argsort(evals, descending=True))
            sd = torch.ones(B, self.Kz, dtype=zh.dtype, device=zh.device)
            if self.quad_probe >= 0:
                n_probe = (self.Kz if self.quad_probe == 0
                           else min(self.quad_probe, self.Kz))
                eps = 0.35
                directions = Q[:, :n_probe].transpose(0, 1) * eps
                probe = torch.cat([directions, -directions], 0).unsqueeze(0)
                f0 = (log_f_sparse(self, zh, ix, C_mode, drop_empty,
                                   detach_params=True) - 0.5 * (zh ** 2).sum(-1))
                zp = zh + probe
                fpm = (log_f_sparse(self, zp, ix, C_mode, drop_empty,
                                    detach_params=True) - 0.5 * (zp ** 2).sum(-1))
                curv = ((2.0 * f0 - fpm[:, :n_probe] - fpm[:, n_probe:])
                        / (eps * eps)).clamp(1.0 / 64.0, 1.0)
                sd[:, :n_probe] = curv.rsqrt()

        if self.quad_share:
            zh = zh.mean(dim=0, keepdim=True).expand_as(zh).contiguous()
            sd = sd.mean(dim=0, keepdim=True).expand_as(sd).contiguous()
        return zh.detach(), sd.detach(), Q.detach()

    def _size_multimode_centres(self, ix, drop_empty, cache):
        """Screen the total normaliser and coarse size bands in one batched solve.

        If ``L_n(z)`` is the log contribution of baskets of size n, every band has its
        own fixed-point equation

            z = grad_z log sum_{n in band} exp L_n(z).

        Running those equations as an extra node axis makes the screen vectorised: it is
        independent of catalogue size beyond work the polynomial kernel already does.  A
        unit step is the fixed-point update itself (not gradient descent); empirically the
        remote large-basket basin that defeated the old zero-start rule is reached in two
        updates.
        """
        nfirst = 1 if drop_empty else 0
        nlast = self.nmax
        raw_bands = ((1, 4), (5, 10), (11, 20), (21, 40),
                     (41, 80), (81, nlast))
        bands = [(max(lo, nfirst), min(hi, nlast)) for lo, hi in raw_bands
                 if max(lo, nfirst) <= min(hi, nlast)]
        M = 1 + len(bands)                 # component zero is the full size sum
        B = ix.B
        C_mode = {k: (v.detach() if torch.is_tensor(v) else v)
                  for k, v in cache.items()}
        z = torch.zeros(B, M, self.Kz, dtype=self.lam.dtype, device=self.lam.device)
        scores = None
        active_modes = None
        n_steps = max(1, int(self.quad_size_steps))
        for step in range(n_steps):
            zz = z.detach().requires_grad_(True)
            with torch.enable_grad():
                lg = log_f_sparse(self, zz, ix, C_mode, drop_empty,
                                  return_terms=True, detach_params=True).permute(1, 0, 2)
                vals = [torch.logsumexp(lg[:, 0, :], dim=-1)]
                for j, (lo, hi) in enumerate(bands, start=1):
                    vals.append(torch.logsumexp(
                        lg[:, j, lo - nfirst:hi - nfirst + 1], dim=-1))
                vals = torch.stack(vals, dim=1)
                if active_modes is None:
                    # Do not backpropagate through size bands that provably cannot become
                    # a dominant Gaussian basin.  Their raw ESP coefficient can be about
                    # 1e-308; differentiating log(coef) then creates an intermediate
                    # 1e308 adjoint inside the polynomial tree and 0*inf -> NaN, even
                    # though the band's probability and the final objective are finite.
                    # This corrupted every parameter in run155 at iteration 2010.
                    #
                    # The screening threshold is not heuristic.  For a basket S,
                    # v_S=sum phi_j and the operator projection gives
                    # ||v_S||^2 <= lambda_max(Phi'Phi)|S|.  At every z,
                    #
                    #   z'v_S - ||z||^2/2 <= ||v_S||^2/2
                    #                            <= lambda_max*nmax/2.
                    #
                    # Therefore a whole size band farther below the full z=0 mass than
                    # that bound cannot overtake it anywhere.  Eight nats cover roundoff.
                    with torch.no_grad():
                        _gram = self.phi.detach().transpose(0, 1) @ self.phi.detach()
                        _lam = torch.linalg.eigvalsh(_gram)[-1].clamp_min(0.0)
                        _recoverable = 0.5 * _lam * nlast + 8.0
                        active_modes = (vals.detach()
                                        >= vals[:, :1].detach() - _recoverable)
                        active_modes[:, 0] = True
                # Detached inactive values retain their score for diagnostics/selection
                # but contribute no unstable polynomial adjoint.  Their only derivative
                # is the Gaussian -z term, so their fixed-point centre stays at zero.
                safe_vals = torch.where(active_modes, vals, vals.detach())
                obj = safe_vals - 0.5 * zz.square().sum(-1)
                mode_grad_failed = False
                gz = torch.autograd.grad(obj.sum(), zz)[0]
                if not bool(torch.isfinite(gz).all()):
                    # Proposal adaptation is detached from the statistical objective.  A
                    # zero/last finite centre still defines a full-support Gaussian and
                    # therefore leaves the positive-weight importance identity exact; it
                    # can only cost variance.  Falling back here is preferable to letting
                    # an irrelevant 1e-308 tail coefficient poison the actual likelihood
                    # gradient.  The replicate-SE/retry gate will reject the update if the
                    # unshifted proposal is not accurate enough.
                    gz = torch.zeros_like(zz)
                    mode_grad_failed = True
                    self._last_qmc_mode_grad_fallbacks = int(
                        getattr(self, "_last_qmc_mode_grad_fallbacks", 0)) + 1
            scores = obj.detach()
            z = (zz + gz).detach()          # exactly grad log f_band
            if mode_grad_failed:
                break
            # Ordinary trips contract immediately into one small cluster.  A second pass
            # then changes log Z by only O(1e-5) but costs another full size-screen kernel.
            # Stop after the first pass unless at least one trip has already produced two
            # centres far enough apart to be a genuine multimode candidate.  Failed tail
            # trips move 1.8--4.4 units on pass one and therefore still receive pass two.
            if step == 0 and n_steps > 1:
                spread = (z - z[:, :1, :]).norm(dim=-1).amax(dim=1)
                if not bool((spread >= float(self.quad_mode_sep)).any()):
                    break

        # Select one dominant basin per trip.  A second, spatially distinct basin is kept
        # only near a phase transition, where dropping either contribution could matter.
        top_id = scores.argmax(dim=1)
        gather_z = top_id[:, None, None].expand(-1, 1, self.Kz)
        top = z.gather(1, gather_z)[:, 0]
        top_score = scores.gather(1, top_id[:, None])[:, 0]
        dist = (z - top[:, None, :]).norm(dim=-1)
        eligible = ((dist >= float(self.quad_mode_sep))
                    & (scores >= top_score[:, None] - float(self.quad_mode_logtol)))
        eligible.scatter_(1, top_id[:, None], False)
        candidate = torch.where(eligible, scores, torch.full_like(scores, -torch.inf))
        second_score, second_id = candidate.max(dim=1)
        has_second = torch.isfinite(second_score)
        second = z.gather(
            1, second_id[:, None, None].expand(-1, 1, self.Kz))[:, 0]
        second = torch.where(has_second[:, None], second, top)
        second_dist = (second - top).norm(dim=-1)

        self._last_qmc_mode_count = (1 + has_second.to(torch.long)).detach()
        self._last_qmc_mode_gap = torch.where(
            has_second, top_score - second_score,
            torch.full_like(top_score, torch.inf)).detach()
        self._last_qmc_mode_sep = second_dist.detach()
        return torch.stack([top, second], dim=1).detach(), has_second.detach(), top.detach()

    def _size_proposal_frame(self, ix, centres, cache, drop_empty):
        """Unit/diagonal proposal scales in the catalogue-active Sobol frame.

        A normal distribution is invariant to an orthogonal rotation, but a *finite*
        Sobol block is not.  Its earliest coordinates have the strongest stratification.
        The ordinary adaptive rule therefore rotates nodes into the eigenframe of
        ``Phi'Phi``.  The first size-multimode implementation omitted that rotation.  At
        the failed checkpoint, one raw-coordinate scramble then hit the broad radial tail
        on every hard trip and raised log Z by 4--8 nats; merely restoring this frame makes
        the same nodes stable.

        Optional symmetric objective probes retain the cheap diagonal curvature rule used
        by ``_adaptive_frame``.  The default ``quad_probe=-1`` pays only one Kz-by-Kz
        eigendecomposition per batch and keeps unit scales.
        """
        B, M, Kz = centres.shape
        with torch.no_grad():
            present = torch.unique(ix.item)
            phi_u = self.phi.index_select(0, present).detach()
            gram = phi_u.transpose(0, 1) @ phi_u
            evals, Q = torch.linalg.eigh(gram)
            Q = Q.index_select(1, torch.argsort(evals, descending=True))
            sd = torch.ones(B, M, Kz, dtype=centres.dtype, device=centres.device)
            raw = torch.ones(B, M, Kz, dtype=centres.dtype, device=centres.device)
            if self.quad_probe >= 0:
                n_probe = (Kz if self.quad_probe == 0
                           else min(int(self.quad_probe), Kz))
                eps = 0.35
                direction = Q[:, :n_probe].transpose(0, 1) * eps
                offset = torch.cat([
                    torch.zeros(1, Kz, dtype=centres.dtype, device=centres.device),
                    direction, -direction], dim=0)
                z = centres[:, :, None, :] + offset[None, None, :, :]
                zflat = z.reshape(B, M * (1 + 2 * n_probe), Kz)
                C_mode = {k: (v.detach() if torch.is_tensor(v) else v)
                          for k, v in cache.items()}
                f = (log_f_sparse(self, zflat, ix, C_mode, drop_empty,
                                  detach_params=True)
                     - 0.5 * zflat.square().sum(-1))
                f = f.reshape(B, M, 1 + 2 * n_probe)
                curv = ((2.0 * f[:, :, :1] - f[:, :, 1:1 + n_probe]
                         - f[:, :, 1 + n_probe:]) / (eps * eps))
                # For this exact Gaussian mixture, -H log target = I-Cov(mu|z) <= I.
                # A non-positive value marks a saddle; the bounded widening is defensive
                # and its RQMC replicate spread remains observable by the caller.
                raw[:, :, :n_probe] = curv
                sd[:, :, :n_probe] = curv.clamp(1.0 / 64.0, 1.0).rsqrt()
        self._last_qmc_curv_min = raw.amin(dim=(1, 2)).detach()
        self._last_qmc_scale_max = sd.amax(dim=(1, 2)).detach()
        return Q.detach(), sd.detach()

    def _size_multimode_proposal(self, ix, drop_empty, cache):
        """Return detached proposal nodes and log(target Gaussian/proposal/node-count)."""
        centres, has_second, top = self._size_multimode_centres(ix, drop_empty, cache)
        B, Kz = ix.B, self.Kz
        if bool(has_second.any()):
            Q, sd = self._size_proposal_frame(ix, centres, cache, drop_empty)
            x = self.quad_mix_a.to(dtype=top.dtype, device=top.device)
            reps, M, per, _ = x.shape
            delta = torch.matmul(
                x.unsqueeze(0) * sd[:, None, :, None, :], Q.transpose(0, 1))
            z = centres[:, None, :, None, :] + delta
            zflat = z.reshape(B, reps * M * per, Kz)
            # Exact balance denominator for the rotated diagonal Gaussian mixture.
            # Duplicate centres on one-mode trips are harmless and pool two Sobol blocks.
            diff = zflat[:, :, None, :] - centres[:, None, :, :]
            eigcoord = torch.matmul(diff, Q)
            comp_lq = (-0.5 * (eigcoord / sd[:, None, :, :]).square().sum(-1)
                       - sd.log().sum(-1)[:, None, :]
                       - 0.5 * Kz * math.log(2.0 * math.pi))
            logq = torch.logsumexp(comp_lq, dim=-1) - math.log(M)
            logprior = (-0.5 * zflat.square().sum(-1)
                        - 0.5 * Kz * math.log(2.0 * math.pi))
            base = logprior - logq - math.log(zflat.shape[1])
            return zflat.detach(), base.detach(), top

        nodes, wts = self.quad_a
        Q, sd = self._size_proposal_frame(
            ix, top[:, None, :], cache, drop_empty)
        x = nodes.to(dtype=top.dtype, device=top.device)
        qw = wts.to(dtype=top.dtype, device=top.device)
        delta = torch.matmul(x.unsqueeze(0) * sd[:, 0, None, :], Q.transpose(0, 1))
        zflat = top[:, None, :] + delta
        # Normal constants cancel between N(z;0,I) and the rotated diagonal proposal.
        base = (-0.5 * zflat.square().sum(-1) + 0.5 * x.square().sum(-1)[None, :]
                + sd[:, 0].log().sum(-1, keepdim=True) + qw.log()[None, :])
        return zflat.detach(), base.detach(), top

    def _log_Z_size_multimode(self, ix, drop_empty, return_ess, return_size,
                              return_mode):
        """Size-stratified one/two-mode deterministic-mixture RQMC normaliser."""
        B = ix.B
        C = sparse_prepare(self, ix)
        zs, base, mode = self._size_multimode_proposal(ix, drop_empty, C)
        P = zs.shape[1]
        chunk = self.quad_chunk if self.quad_chunk > 0 else P
        if self.quad_chunk <= 0 and len(C["ai"]) * P > 2_000_000:
            chunk = min(32, P)
        log_mass = None
        node_logs = []

        def _block(zc, bc):
            if return_size:
                lg = log_f_sparse(self, zc, ix, C, drop_empty, return_terms=True)
                joint = lg.permute(1, 0, 2) + bc.unsqueeze(-1)
                node_log = torch.logsumexp(joint, dim=-1)
                return torch.logsumexp(joint, dim=1), node_log
            node_log = _log_f_dispatch(self, zc, ix, C, drop_empty) + bc
            return torch.logsumexp(node_log, dim=1), node_log

        for lo in range(0, P, chunk):
            zc, bc = zs[:, lo:lo + chunk], base[:, lo:lo + chunk]
            use_checkpoint = torch.is_grad_enabled() and chunk < P
            mass_c, node_c = (
                torch.utils.checkpoint.checkpoint(_block, zc, bc, use_reentrant=False)
                if use_checkpoint else _block(zc, bc))
            log_mass = mass_c if log_mass is None else torch.logaddexp(log_mass, mass_c)
            if return_ess:
                node_logs.append(node_c)

        if return_size:
            lz = torch.logsumexp(log_mass, dim=-1)
            pn = torch.softmax(log_mass, dim=-1)
        else:
            lz, pn = log_mass, None
        out = [lz]
        if return_ess:
            nl = torch.cat(node_logs, dim=1)
            ess = (torch.exp(2.0 * torch.logsumexp(nl, dim=1)
                             - torch.logsumexp(2.0 * nl, dim=1)) / P)
            out.append(ess.clamp(max=1.0))
            reps = int(getattr(self, "quad_replicates", 1))
            if reps > 1 and P % reps == 0:
                per = P // reps
                rep_lz = torch.logsumexp(nl.view(B, reps, per), dim=-1) + math.log(reps)
                self._last_qmc_logz_se = (
                    rep_lz.detach().std(dim=1, unbiased=True) / math.sqrt(reps))
            else:
                self._last_qmc_logz_se = None
        if return_size:
            out.append(pn)
        if return_mode:
            out.append(mode)
        return out[0] if len(out) == 1 else tuple(out)

    def _log_Z_cv(self, ix, drop_empty, return_ess=False, return_size=False,
                  return_mode=False, n_nodes=8, seed=0, reps=2):
        """log Z by Laplace-tilted importance sampling.  Exact (unbiased in Z), no mode search.

        log f(z) is the CGF of v_S = sum_{j in S} phi_j under W(S)/f(0), so a = grad log f(0)
        is E[v_S] and B = Hess log f(0) is Cov(v_S).  The quadratic
        q(z) = exp(g0 + a'z + z'Bz/2) then has closed forms for BOTH its Gaussian integral
        and its tilt, giving the identity

            Z = [int q phi_I] * E_{z ~ N(m,V)}[ f(z)/q(z) ],   V = (I-B)^{-1},  m = V a,

        with f/q = 1 to second order by construction.  Verified numerically end to end:
        the closed form against 4e6-sample MC, a and B against exact enumeration to 4e-16,
        and convergence to the enumerated log Z as nodes rise.

        The identity holds for ANY B with I - B > 0, so the mean-field
        B = Phi' diag(pi(1-pi)) Phi costs VARIANCE ONLY, never bias -- checked by forcing
        B_eff to 0, 0.5x, 1x and 1.5x the mean-field value and recovering the same answer
        (only the spread changed, optimal at 1x).

        Measured against the 8-node mode-shifted rule this replaces, at the trained model's
        own lambda_max = 0.177: sd 0.0043 vs 0.0874, i.e. 20x less variance for 20% less
        time, and better than the 128-node evaluation rule at an eighth of its cost.
        """
        B_, Kz = ix.B, self.Kz
        dt, dev = self.lam.dtype, self.lam.device
        C = sparse_prepare(self, ix)
        z0 = torch.zeros(B_, 1, Kz, dtype=dt, device=dev, requires_grad=True)
        g0 = log_f_sparse(self, z0, ix, C, drop_empty)[:, 0]            # [B]
        a = torch.autograd.grad(g0.sum(), z0, create_graph=torch.is_grad_enabled())[0][:, 0]
        # pi_j = d log f / d b_j, one backward pass; Cov(v) ~ Phi' diag(pi(1-pi)) Phi
        with torch.enable_grad():
            b0 = C["b0"].detach().requires_grad_(True)
            Cp = dict(C); Cp["bt"] = b0 - 0.5 * C["phi_sq"]
            M0 = seg_max(Cp["bt"].unsqueeze(0), ix.item_trip, ix.B)
            Cp["M0"], Cp["sh"] = M0, M0[0].index_select(0, ix.item_trip)
            gp = log_f_sparse(self, torch.zeros(B_, 1, Kz, dtype=dt, device=dev),
                              ix, Cp, drop_empty)[:, 0]
            pi = torch.autograd.grad(gp.sum(), b0)[0].detach().clamp(1e-12, 1 - 1e-12)
        w = (pi * (1.0 - pi)).unsqueeze(-1)                             # [T,1]
        phi_i = self.phi[ix.item]
        Bm = torch.zeros(B_, Kz, Kz, dtype=dt, device=dev).index_add_(
            0, ix.item_trip, (w * phi_i).unsqueeze(-1) * phi_i.unsqueeze(-2))
        Bm = 0.5 * (Bm + Bm.transpose(1, 2))
        I = torch.eye(Kz, dtype=dt, device=dev).expand(B_, Kz, Kz)
        M = I - Bm
        ev = torch.linalg.eigvalsh(M).min(dim=1).values                 # [B]
        bump = (1e-6 - ev).clamp_min(0.0)
        M = M + bump.view(B_, 1, 1) * I
        B_eff = (I - M).detach()                                        # q and tilt SHARE this
        Md = M.detach()
        L = torch.linalg.cholesky(torch.linalg.inv(Md))                 # V = M^{-1}
        m = torch.linalg.solve(Md, a.detach().unsqueeze(-1)).squeeze(-1)
        log_Zq = (g0 - 0.5 * torch.logdet(M)
                  + 0.5 * (a.unsqueeze(1) @ torch.linalg.solve(M, a.unsqueeze(-1))).view(B_))
        acc = []
        per = max(n_nodes // max(reps, 1), 1)
        for r in range(max(reps, 1)):
            e = torch.quasirandom.SobolEngine(Kz, scramble=True, seed=int(seed) + 7919 * r)
            u = e.draw(per).to(dt).clamp(1e-12, 1 - 1e-12)
            x = torch.erfinv(2 * u - 1) * math.sqrt(2.0)                # [P, Kz]
            zs = m.unsqueeze(1) + torch.einsum('bij,pj->bpi', L, x)     # [B, P, Kz]
            lf = log_f_sparse(self, zs, ix, C, drop_empty)              # [B, P]
            lq = (g0.unsqueeze(1) + (zs * a.unsqueeze(1)).sum(-1)
                  + 0.5 * torch.einsum('bpi,bij,bpj->bp', zs, B_eff, zs))
            acc.append(torch.logsumexp(lf - lq, dim=1) - math.log(per))
        lz = log_Zq + (torch.logsumexp(torch.stack(acc), 0) - math.log(max(reps, 1)))
        out = [lz]
        if return_ess:
            out.append(torch.ones(B_, dtype=dt, device=dev))
        if return_size:
            raise ValueError("size_dist under the CV rule is not implemented; use the quadrature")
        if return_mode:
            out.append(m)
        return out[0] if len(out) == 1 else tuple(out)

    def _log_Z_adaptive(self, ix, drop_empty, return_ess, return_size, return_mode,
                        mode_steps=8):
        """log Z by a positive-weight rule in a mode/curvature adapted frame.

            Z = int f(z) N(z;0,I) dz
              = (2pi)^{d/2}|S|^{1/2} E_{x~N(0,I)}[ exp(||x||^2/2) f(zh+Lx) N(zh+Lx;0,I) ]

        L = Q diag(sd), where Q orders coordinates by the eigenvectors of Phi'Phi and the
        leading scales come from finite differences of log[f(z)N(z)] at its mode.  Sobol's
        first coordinates are its best stratified, so assigning them to the learned active
        subspace is much lower variance than estimating one isotropic scale from random
        directions.  Directions orthogonal to span(Phi) retain their exact N(0,1) scale.

        The proposal construction is detached: differentiating a finite adaptive rule
        through its mode adds a large, artificial gradient path although the exact integral
        is invariant to the proposal.  The final integrand remains fully differentiable.
        """
        if getattr(self, "quad_size_bands", 0):
            return self._log_Z_size_multimode(
                ix, drop_empty, return_ess, return_size, return_mode)
        B = ix.B
        nodes, wts = self.quad_a
        if bool((wts <= 0).any()):
            raise ValueError("adaptive quadrature requires strictly positive weights")

        # Build the differentiable cache once.  Proposal construction consumes a detached
        # view of it, and the actual integral consumes the original graph.  The former code
        # called sparse_prepare twice per log Z evaluation.
        C = sparse_prepare(self, ix)
        zh, sd, Q = self._adaptive_frame(ix, drop_empty, mode_steps, cache=C)

        # Stream all
        # outputs, including the size law; the old return_size path disabled chunking and
        # was therefore precisely the path training could not run at full catalogue size.
        x = nodes.to(dtype=zh.dtype, device=zh.device)
        qw = wts.to(dtype=zh.dtype, device=zh.device)
        P = x.shape[0]
        chunk = self.quad_chunk if self.quad_chunk > 0 else P
        if self.quad_chunk <= 0 and len(C["ai"]) * P > 2_000_000:
            chunk = min(32, P)
        log_mass = None
        node_logs = []
        logdet = sd.log().sum(-1, keepdim=True)

        def _block(xc, wc):
            delta = torch.matmul(xc.unsqueeze(0) * sd.unsqueeze(1),
                                 Q.transpose(0, 1))
            zc = zh + delta
            base = (-0.5 * (zc ** 2).sum(-1) + wc.log().unsqueeze(0)
                    + 0.5 * (xc ** 2).sum(-1).unsqueeze(0) + logdet)
            if return_size:
                lg = log_f_sparse(self, zc, ix, C, drop_empty, return_terms=True)
                joint = lg.permute(1, 0, 2) + base.unsqueeze(-1)
                node_log = torch.logsumexp(joint, dim=-1)
                return torch.logsumexp(joint, dim=1), node_log
            node_log = _log_f_dispatch(self, zc, ix, C, drop_empty) + base
            return torch.logsumexp(node_log, dim=1), node_log

        for lo in range(0, P, chunk):
            xc, wc = x[lo:lo + chunk], qw[lo:lo + chunk]
            use_checkpoint = torch.is_grad_enabled() and chunk < P
            mass_c, node_c = (
                torch.utils.checkpoint.checkpoint(_block, xc, wc, use_reentrant=False)
                if use_checkpoint else _block(xc, wc))
            log_mass = mass_c if log_mass is None else torch.logaddexp(log_mass, mass_c)
            if return_ess:
                node_logs.append(node_c)

        if return_size:
            lz = torch.logsumexp(log_mass, dim=-1)
            pn = torch.softmax(log_mass, dim=-1)
        else:
            lz, pn = log_mass, None
        out = [lz]
        if return_ess:
            nl = torch.cat(node_logs, dim=1)
            ess = torch.exp(2.0 * torch.logsumexp(nl, dim=1)
                            - torch.logsumexp(2.0 * nl, dim=1)) / P
            out.append(ess.clamp(max=1.0))
            reps = int(getattr(self, "quad_replicates", 1))
            if reps > 1 and P % reps == 0:
                per = P // reps
                rep_lz = torch.logsumexp(nl.view(B, reps, per), dim=-1) + math.log(reps)
                self._last_qmc_logz_se = (
                    rep_lz.detach().std(dim=1, unbiased=True) / math.sqrt(reps))
            else:
                self._last_qmc_logz_se = None
        if return_size:
            out.append(pn)
        if return_mode:
            out.append(zh[:, 0, :])
        return out[0] if len(out) == 1 else tuple(out)

    def _log_Z_quad(self, ix, drop_empty, return_ess, return_size, return_mode):
        """log Z by DETERMINISTIC Smolyak quadrature -- no proposal, no draws, no bias.

        f(z) is already exact (the ESP/poly-tree recursion is a closed form), so the only
        approximation in the whole pipeline was the outer E_z over a Kz-dimensional
        Gaussian.  Importance sampling cannot do that integral here: verified against exact
        enumeration, 4096 draws are wrong by 8-36 nats, because f(z) ~ exp(c||z||) keeps its
        mass in a shell the mode-centred proposal never reaches.  A sparse grid does it
        deterministically -- at the strength grocery needs (phi'phi = 0.92) the error is
        0.005 nats at Kz=2 with 17 points, fewer points than the 16 draws it replaces.

        Smolyak weights are signed, so the sum is formed in linear space after factoring out
        the row max; logsumexp cannot be used.
        """
        nodes, wts = self.quad                     # [P, Kz], [P]
        B, P = ix.B, nodes.shape[0]
        zs = nodes.to(self.lam.dtype).unsqueeze(0).expand(B, P, self.Kz)
        w = wts.to(self.lam.dtype)
        # Sparse path: only phi-carrying products depend on z, so the ESP over the rest
        # and the category convolution over untouched categories are lifted out of the node
        # loop.  Verified bit-equal to log_f_ragged (max rel 2.3e-16) and gradient-equal on
        # masked-in phi (6.9e-14); the only difference is that phi_j = 0 products get no
        # gradient, and fit.py re-applies the mask every step so those are discarded anyway.
        # 83.7x at a 24-product mask, 7.0x at 400.
        _C = sparse_prepare(self, ix)
        if return_size:
            lg = log_f_sparse(self, zs, ix, _C, drop_empty, return_terms=True)   # [P, B, n]
            lf = torch.logsumexp(lg, dim=-1).transpose(0, 1)                     # [B, P]
        else:
            lf = log_f_sparse(self, zs, ix, _C, drop_empty)                      # [B, P]
        M = lf.max(dim=1, keepdim=True).values
        S = (w.unsqueeze(0) * torch.exp(lf - M)).sum(1)
        lz = M.squeeze(1) + torch.log(S.clamp_min(1e-300))
        out = [lz]
        if return_ess:
            # Deterministic: every node contributes by construction.  Reported as 1 so the
            # ESS gate -- which exists to catch a collapsed sampler -- never fires on a rule
            # that has no sampler to collapse.
            out.append(torch.ones(B, dtype=lz.dtype, device=lz.device))
        if return_size:
            t = w.view(1, P, 1) * torch.exp(lg.permute(1, 0, 2) - M.unsqueeze(-1))
            pn = t.sum(1)
            pn = pn.clamp_min(0.0)
            pn = pn / pn.sum(-1, keepdim=True).clamp_min(1e-300)
            out.append(pn)
        if return_mode:
            out.append(torch.zeros(B, self.Kz, dtype=lz.dtype, device=lz.device))
        return out[0] if len(out) == 1 else tuple(out)

    def log_Z_observed_size(self, ix, observed_n, return_ess=False, return_mode=False):
        """RQMC log Z_n(x) for only each trip's observed size.

        Under the factored objective, summing over all 120 sizes and then subtracting
        log P(n|x) is algebraically redundant: the full log Z and rho_0 cancel, leaving
        the fixed-size coefficient Z_n.  Compute that coefficient directly.  The ESP tree
        is truncated at the largest observed size in the minibatch, while the QMC nodes
        remain fixed standard-normal Sobol nodes.  This changes neither the integral nor
        its gradient and exposes replicate error for the quantity actually optimized.
        """
        if getattr(self, "quad_a", None) is None:
            raise ValueError("observed-size normalizer requires the positive Sobol rule")
        observed_n = observed_n.to(dtype=torch.long, device=self.lam.device)
        if observed_n.shape != (ix.B,) or bool((observed_n < 1).any()):
            raise ValueError("one positive observed size is required per trip")
        max_n = int(observed_n.max())
        if max_n > self.nmax:
            raise ValueError("observed size exceeds model support")
        # For fixed size n, no category can contribute more than n items.  This makes
        # complete support (R=nmax) cost only O(max observed n in this minibatch), so the
        # three formerly dropped high-within-category baskets do not impose R=120 work on
        # every ordinary batch.
        C = sparse_prepare(self, ix, degree=max_n, nmax=max_n)
        # Locate the mode of the FIXED-size target.  Reusing the full model's size-mixture
        # mode is both slower and wrong for a conditional coefficient: a rare observed n
        # can have its mass far from the mode of sum_m Z_m.  Two detached half-steps are
        # enough in the stable phi regime and cost D=1 polynomial passes truncated at n.
        C_mode = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in C.items()}
        row = torch.arange(ix.B, device=self.lam.device)
        mode = torch.zeros(ix.B, 1, self.Kz, dtype=self.lam.dtype,
                           device=self.lam.device)
        for _ in range(self.quad_steps):
            zz = mode.detach().requires_grad_(True)
            with torch.enable_grad():
                terms = log_f_sparse(
                    self, zz, ix, C_mode, True, return_terms=True,
                    detach_params=True, nmax=max_n).permute(1, 0, 2)
                target = (terms[row, 0, observed_n - 1]
                          - 0.5 * zz[:, 0].square().sum(-1)).sum()
                grad = torch.autograd.grad(target, zz)[0]
            mode = (zz + 0.5 * grad).detach()
        zh = mode[:, 0]
        # Diagonal Laplace frame in the global Phi eigenspace.  Fixed-size posteriors can
        # be much broader than N(zh,I) near a complementarity transition; that was the
        # remaining source of rare ESS=0.07 batches after the mode shift.  Probe only the
        # leading requested directions in one vectorised D=(1+2p) pass.
        with torch.no_grad():
            present = torch.unique(ix.item)
            phi_u = self.phi.index_select(0, present).detach()
            gram = phi_u.transpose(0, 1) @ phi_u
            evals, Q = torch.linalg.eigh(gram)
            Q = Q.index_select(1, torch.argsort(evals, descending=True))
            sd = torch.ones(ix.B, self.Kz, dtype=zh.dtype, device=zh.device)
            if self.quad_probe >= 0:
                nprobe = (self.Kz if self.quad_probe == 0
                          else min(int(self.quad_probe), self.Kz))
                eps = 0.35
                direction = Q[:, :nprobe].transpose(0, 1) * eps
                offset = torch.cat([
                    torch.zeros(1, self.Kz, dtype=zh.dtype, device=zh.device),
                    direction, -direction], dim=0)
                zp = zh[:, None, :] + offset[None, :, :]
                lgp = log_f_sparse(
                    self, zp, ix, C_mode, True, return_terms=True,
                    detach_params=True, nmax=max_n).permute(1, 0, 2)
                fp = lgp[row, :, observed_n - 1] - 0.5 * zp.square().sum(-1)
                curv = ((2.0 * fp[:, :1] - fp[:, 1:1 + nprobe]
                         - fp[:, 1 + nprobe:]) / (eps * eps)).clamp(1.0 / 64.0, 1.0)
                sd[:, :nprobe] = curv.rsqrt()
        self._last_qmc_curv_min = sd.square().reciprocal().amin(dim=1).detach()
        self._last_qmc_scale_max = sd.amax(dim=1).detach()
        nodes, weights = self.quad_a
        nodes = nodes.to(dtype=self.lam.dtype, device=self.lam.device)
        logw = weights.to(dtype=self.lam.dtype, device=self.lam.device).log()
        P = nodes.shape[0]
        chunk = self.quad_chunk if self.quad_chunk > 0 else P
        if self.quad_chunk <= 0 and len(C["ai"]) * P > 2_000_000:
            chunk = min(32, P)
        gathered = []
        def _block(zc):
            lg = log_f_sparse(self, zc, ix, C, True, return_terms=True,
                              nmax=max_n).permute(1, 0, 2)  # [B,node,n]
            return lg[row, :, observed_n - 1]

        for lo in range(0, P, chunk):
            x = nodes[lo:lo + chunk]
            delta = torch.matmul(x.unsqueeze(0) * sd.unsqueeze(1), Q.transpose(0, 1))
            zc = zh.unsqueeze(1) + delta
            use_checkpoint = torch.is_grad_enabled() and chunk < P
            term = (torch.utils.checkpoint.checkpoint(
                        _block, zc, use_reentrant=False)
                    if use_checkpoint else _block(zc))
            # Prior/proposal ratio for q=N(zh,I); normal constants cancel.
            base = (-0.5 * zc.square().sum(-1) + 0.5 * x.square().sum(-1).unsqueeze(0)
                    + sd.log().sum(-1, keepdim=True)
                    + logw[lo:lo + chunk].unsqueeze(0))
            gathered.append(term + base)
        node_log = torch.cat(gathered, dim=1)
        lz = torch.logsumexp(node_log, dim=1)
        reps = int(getattr(self, "quad_replicates", 1))
        if reps > 1 and P % reps == 0:
            per = P // reps
            rep_lz = torch.logsumexp(node_log.view(ix.B, reps, per), dim=-1) \
                     + math.log(reps)
            self._last_qmc_logz_se = (
                rep_lz.detach().std(dim=1, unbiased=True) / math.sqrt(reps))
        else:
            self._last_qmc_logz_se = None
        self._last_qmc_mode_count = torch.ones(ix.B, dtype=torch.long,
                                                device=self.lam.device)
        out = [lz]
        if return_ess:
            ess = (torch.exp(2.0 * torch.logsumexp(node_log, dim=1)
                             - torch.logsumexp(2.0 * node_log, dim=1)) / P)
            out.append(ess.clamp(max=1.0))
        if return_mode:
            out.append(zh)
        return out[0] if len(out) == 1 else tuple(out)

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
        if getattr(self, "_exact_additive", False):
            # If phi is identically zero, f(z) has no z dependence and E_z[f(z)] = f(0)
            # exactly.  Running a Sobol rule and constructing its adaptive proposal would
            # repeat the same polynomial once per node.  The sparse cache also recognizes
            # that every item is inactive and builds the full category polynomial once.
            # This path is used by the re-optimised nested control, not by the full model.
            C = sparse_prepare(self, ix)
            z0 = torch.zeros(ix.B, 1, self.Kz, dtype=self.lam.dtype,
                             device=self.lam.device)
            if return_size:
                terms = log_f_sparse(self, z0, ix, C, drop_empty,
                                     return_terms=True)[0]             # [B, n]
                lz = torch.logsumexp(terms, dim=-1)
                pn = torch.softmax(terms, dim=-1)
            else:
                lz = log_f_sparse(self, z0, ix, C, drop_empty).squeeze(1)
                pn = None
            one = torch.ones(ix.B, dtype=lz.dtype, device=lz.device)
            zero = torch.zeros_like(one)
            self._last_qmc_logz_se = zero.detach()
            self._last_qmc_mode_count = torch.ones(ix.B, dtype=torch.long,
                                                    device=lz.device)
            self._last_qmc_mode_gap = zero.detach()
            self._last_qmc_mode_sep = zero.detach()
            self._last_qmc_curv_min = one.detach()
            self._last_qmc_scale_max = one.detach()
            out = [lz]
            if return_ess:
                out.append(one)
            if return_size:
                out.append(pn)
            if return_mode:
                out.append(torch.zeros(ix.B, self.Kz, dtype=lz.dtype, device=lz.device))
            return out[0] if len(out) == 1 else tuple(out)
        if getattr(self, "quad_a", None) is not None:
            return self._log_Z_adaptive(ix, drop_empty, return_ess, return_size, return_mode,
                                        mode_steps=self.quad_steps)
        if getattr(self, "quad", None) is not None:
            return self._log_Z_quad(ix, drop_empty, return_ess, return_size, return_mode)
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
            # The proposal density MUST match the proposal the draws came from.  These three
            # branches used to be followed by an unconditional
            #     log_q = -0.5*(noise**2).sum(-1) - sd.log().sum(...) - 0.5*Kz*log 2pi
            # which overwrote every one of them, so aniso's and the mixture's densities were
            # computed and thrown away and every draw was scored under the plain isotropic
            # Gaussian.  Measured at 16 draws against a 512-draw reference of 10.873:
            # mix_scales(1,3) returned 853.619 (off by 843 nats) because it DREW from the
            # mixture and SCORED under a single Gaussian, and aniso 2 / 4 / 8 were
            # bit-identical because only their effect on sd survived.  An if/elif/else makes
            # the density follow the draws.
            _L2P = float(math.log(2 * math.pi))
            _sdlog = sd.log().sum(-1, keepdim=True)
            if mix_scales is not None and len(mix_scales) > 1:
                log_q = mix_lq - _sdlog - 0.5 * self.Kz * _L2P
            elif aniso > 1.0:
                log_q = aniso_lq - _sdlog - 0.5 * self.Kz * _L2P
            else:
                log_q = -0.5 * (noise ** 2).sum(-1) - _sdlog - 0.5 * self.Kz * _L2P
            if ais_steps > 0:
                # _ais is BROKEN by its own docstring: the weight is double counted and it
                # diverges (log Z 20.3 -> 20.8 -> 22.3 as steps go 4 -> 8 -> 16 while ESS
                # falls 0.584 -> 0.255).  It returned 23.968 against a true 10.873 here.
                # Refuse rather than return a number that looks like a log Z.
                raise ValueError(
                    "ais_steps > 0: _ais is known-broken (double-counted weight, diverges "
                    "with steps). It returns plausible-looking wrong values, so it is "
                    "disabled rather than silently used. See _ais.__doc__.")
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
        if bool(self.factored_size_enabled):
            if not drop_empty:
                raise ValueError("factored size law is defined conditional on a non-empty basket")
            p = self.factored_size_log_p.exp().unsqueeze(0).expand(ix.B, -1)
            if return_mode:
                z = torch.zeros(ix.B, 1, self.Kz, dtype=self.lam.dtype,
                                device=self.lam.device)
                return p, z
            return p
        B, L2P = ix.B, float(math.log(2 * math.pi))
        if (getattr(self, "quad", None) is not None
                or getattr(self, "quad_a", None) is not None) and z_fixed is None:
            # EXACT size law from the same grid the normaliser uses.  This function
            # otherwise runs the importance sampler -- three autograd mode steps through the
            # DENSE log_f, then reweighted draws -- so E[n] and Var(n) were still estimates
            # while the likelihood beside them was exact.  That is why they swung between
            # consecutive evals (var 92 -> 284 at one point) with a stable likelihood.
            # Measured: 1208 ms per batch of 24 against 128 ms for loglik.
            with torch.enable_grad() if grad else torch.no_grad():
                out = self.log_Z(ix, drop_empty=drop_empty, return_size=True,
                                 return_mode=return_mode)
            p = out[1]
            return (p, out[2].unsqueeze(1)) if return_mode else p
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
        if bool(self.factored_size_enabled):
            raise NotImplementedError(
                "factored-size simulation must draw z conditional on the external size; "
                "using the old joint sampler would be statistically wrong")
        B, L2P = ix.B, float(math.log(2 * math.pi))
        if getattr(self, "quad_a", None) is not None:
            # The adaptive positive-weight rule is also the discrete approximation to the
            # posterior of z.  Reusing its frame/nodes keeps simulation and likelihood on
            # the same approximation; the former IS branch ignored quad_a and could drift
            # with n_draws even while training/evaluation used a fixed QMC rule.
            # The frame and node evaluation use the same z-independent cache.  sample() is
            # under no_grad, so sharing it is both exact and strictly less work.
            C = sparse_prepare(self, ix)
            if getattr(self, "quad_size_bands", 0):
                zs, base, _mode = self._size_multimode_proposal(ix, True, C)
                P = zs.shape[1]
                chunk = self.quad_chunk if self.quad_chunk > 0 else P
                if self.quad_chunk <= 0 and len(C["ai"]) * P > 2_000_000:
                    chunk = min(32, P)
                log_nodes = []
                for lo in range(0, P, chunk):
                    zc = zs[:, lo:lo + chunk]
                    log_nodes.append(
                        log_f_sparse(self, zc, ix, C, True) + base[:, lo:lo + chunk])
                lw = torch.cat(log_nodes, dim=1)
                pick = torch.multinomial(torch.softmax(lw, dim=1), 1,
                                         generator=generator)
                zsel = zs.gather(
                    1, pick.unsqueeze(-1).expand(-1, -1, self.Kz))[:, 0]
            else:
                zh, sd, Q = self._adaptive_frame(ix, True, self.quad_steps, cache=C)
                qx, qw = self.quad_a
                x = qx.to(dtype=zh.dtype, device=zh.device)
                qw = qw.to(dtype=zh.dtype, device=zh.device)
                P = x.shape[0]
                chunk = self.quad_chunk if self.quad_chunk > 0 else P
                if self.quad_chunk <= 0 and len(C["ai"]) * P > 2_000_000:
                    chunk = min(32, P)
                logdet = sd.log().sum(-1, keepdim=True)
                log_nodes = []
                for lo in range(0, P, chunk):
                    xc, wc = x[lo:lo + chunk], qw[lo:lo + chunk]
                    delta = torch.matmul(xc.unsqueeze(0) * sd.unsqueeze(1),
                                         Q.transpose(0, 1))
                    zc = zh + delta
                    base = (-0.5 * (zc ** 2).sum(-1) + wc.log().unsqueeze(0)
                            + 0.5 * (xc ** 2).sum(-1).unsqueeze(0) + logdet)
                    log_nodes.append(log_f_sparse(self, zc, ix, C, True) + base)
                lw = torch.cat(log_nodes, dim=1)
                pick = torch.multinomial(torch.softmax(lw, dim=1), 1,
                                         generator=generator)
                xsel = x.index_select(0, pick[:, 0])
                delta = torch.matmul((xsel * sd).unsqueeze(1),
                                     Q.transpose(0, 1))[:, 0]
                zsel = zh[:, 0] + delta
        elif getattr(self, "quad_z", None) is not None:
            # Stage 1 on a DETERMINISTIC grid.  Sampling-importance-resampling around the
            # mode is the only inexact step in Corollary 3, and it is unreliable here:
            # measured, the sampled E[n] drifts 0.33x -> 1.29x of the model's own E[n] as
            # n_draws goes 8 -> 1024, so a rollout's basket sizes depended on a tuning knob.
            # With Kz small the posterior over z can be formed exactly on a dense
            # Gauss-Hermite grid (positive weights, unlike Smolyak) and sampled from
            # directly: p(z_p) proportional to w_p f(z_p).
            gz, gw = self.quad_z
            zs = gz.to(self.lam.dtype).unsqueeze(0).expand(B, gz.shape[0], self.Kz)
            # Use the SPARSE kernel with a shared cache, exactly as the quad_a branch does.
            # log_f_ragged walks all ~125,000 slots at every node; log_f_sparse lifts the
            # phi-free products out of the node loop, which is the whole point of a mask.
            # sample() is under no_grad, so sharing the cache is exact and strictly less
            # work.  With a 30-product mask that is 720 z-dependent slots per node, not
            # 125,000 -- the difference between a usable check and one that hangs.
            _Cz = sparse_prepare(self, ix)
            _P = zs.shape[1]
            _ch = self.quad_chunk if self.quad_chunk > 0 else _P
            if self.quad_chunk <= 0 and len(_Cz["ai"]) * _P > 2_000_000:
                _ch = max(1, min(64, _P))
            _parts = [log_f_sparse(self, zs[:, lo:lo + _ch], ix, _Cz, True)
                      for lo in range(0, _P, _ch)]
            lw = torch.cat(_parts, dim=1) + gw.to(self.lam.dtype).log().unsqueeze(0)
            pick = torch.multinomial(torch.softmax(lw, dim=1), 1, generator=generator)
            zsel = zs.gather(1, pick.unsqueeze(-1).expand(-1, -1, self.Kz))[:, 0]
        else:
            z = torch.zeros(B, 1, self.Kz, dtype=self.lam.dtype, device=self.lam.device)
            for _ in range(mode_steps):
                zz = z.detach().requires_grad_(True)
                with torch.enable_grad():
                    lf = log_f_ragged(self, zz, ix, True).sum()
                z = torch.autograd.grad(lf, zz)[0]
            noise = torch.randn(B, n_draws, self.Kz, dtype=z.dtype, device=z.device,
                                generator=generator)
            zs = z.detach() + noise
            log_q = -0.5 * (noise ** 2).sum(-1) - 0.5 * self.Kz * L2P
            lw = (-0.5 * self.Kz * L2P - 0.5 * (zs ** 2).sum(-1)
                  + log_f_ragged(self, zs, ix, True)) - log_q        # [B, D]
            pick = torch.multinomial(torch.softmax(lw, dim=1), 1, generator=generator)
            zsel = zs.gather(1, pick.unsqueeze(-1).expand(-1, -1, self.Kz))[:, 0]

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
            # RESCALE every polynomial to max 1 before multiplying.
            #
            # G = exp(-rho_c r(r-1)/2) e_r(w) is consumed here in LINEAR space, and the
            # prefix product runs over all of this trip's categories up to degree nmax.
            # With rho_c negative -- 224 of 280 rows are, min -0.1126 -- the factor grows
            # with r, and the accumulated product overflows to inf; the multinomial below
            # then raises "probability tensor contains inf, nan or element < 0".  That is
            # the sampler failure seen in run155 (sampled 11.1 vs analytic 7.1), run302
            # (120.0) and run403 (nan), on both Smolyak and QMC, while log Z stayed correct
            # because log_f works in log space with a per-trip max shift.
            #
            # Only RATIOS matter for every draw below, so scaling each array by a positive
            # constant is exact, not an approximation.
            def _norm1(v):
                mx = v.max()
                return v / mx if bool(torch.isfinite(mx)) and float(mx) > 0 else v
            polys = [_norm1(G_all[r_]) for r_ in rows]
            pref = [torch.ones(1, dtype=w_all.dtype, device=w_all.device)]
            for Gc in polys:
                pref.append(_norm1(poly_mul_trunc(pref[-1].unsqueeze(0),
                                                  Gc.unsqueeze(0), self.nmax)[0]))
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
                    # This table is built over EVERY product in the category -- up to 1,773
                    # -- so the raw recursion underflows exactly as the prefix polynomials
                    # above did.  When E[k, need] reached 0 the old code did `continue`
                    # WITHOUT decrementing need, so the walk ran out of items and returned
                    # a basket shorter than the n that was drawn: sampled E[n] was 5.83
                    # against the model's own 6.49, a 10% loss that no number of draws
                    # removed.  Normalise each row and carry its log scale, so the ratio
                    #   w_k E[k-1][need-1] / E[k][need]
                    # is corrected by exp(ls[k-1] - ls[k]) and stays exact.
                    sel = (ix.row_of == rows[c]).nonzero().flatten()
                    wc = w_all[sel]
                    E = torch.zeros(len(wc) + 1, self.R + 1, dtype=wc.dtype,
                                    device=wc.device)
                    E[0, 0] = 1.0
                    ls = torch.zeros(len(wc) + 1, dtype=wc.dtype, device=wc.device)
                    for k in range(1, len(wc) + 1):
                        row = E[k - 1].clone()
                        row[1:] = row[1:] + wc[k - 1] * E[k - 1, :-1]
                        mx = float(row.max())
                        if mx > 0 and math.isfinite(mx):
                            row = row / mx
                            ls[k] = ls[k - 1] + math.log(mx)
                        else:
                            ls[k] = ls[k - 1]
                        E[k] = row
                    need = r_take
                    for k in range(len(wc), 0, -1):
                        if need == 0:
                            break
                        den = float(E[k, need])
                        if den <= 0:
                            continue
                        num = float(wc[k - 1] * E[k - 1, need - 1]
                                    * torch.exp(ls[k - 1] - ls[k]))
                        p = num / den
                        if not math.isfinite(p):
                            continue
                        # k items remain and `need` are still required: once they are equal
                        # the walk MUST take every one of them, and rounding must not be
                        # allowed to drop one.
                        if k == need or float(torch.rand(1, generator=generator)) < p:
                            chosen.append(int(ix.item[sel[k - 1]]))
                            need -= 1
                    if need:
                        self._sample_short = getattr(self, "_sample_short", 0) + need
                left -= r_take
            if left:
                # Categories skipped for tot <= 0 leave the size draw unfulfilled; the
                # basket is then not a draw from P(S | n).  Counted, not hidden.
                self._sample_short = getattr(self, "_sample_short", 0) + left
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
            return b + pair - self.rho_c[cj] * self.pair_increment(nc) - dr0

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



    def pair_scale_of_n(self):
        """s(n) as a [nmax+1] lookup, from self.size_bands."""
        s = torch.ones(self.nmax + 1, dtype=self.lam.dtype, device=self.lam.device)
        for (lo, hi, sc) in (self.size_bands or []):
            s[lo:min(hi, self.nmax) + 1] = float(sc)
        return s

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
        # Same piecewise-constant s(n) the normaliser uses.  If these disagree the energy
        # and log Z score the same basket under different laws -- the exact failure this
        # file records for b_flat vs energy.
        if getattr(self, "size_bands", None):
            n_e = torch.bincount(line_trip, minlength=B).clamp(max=self.nmax)
            pair = pair / self.pair_scale_of_n().to(pair.dtype)[n_e]
        key = line_trip * self.C + line_cat
        nc = torch.bincount(key, minlength=B * self.C).view(B, self.C).to(dt)
        pen_c = (self.rho_c.unsqueeze(0) * self.pair_feature(nc)).sum(-1)
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
               mix_scales=None, aniso=0.0, antithetic=False):
        """log P(S | S non-empty) = E(S) - log(Z - 1), with log(Z - 1) computed directly
        rather than by subtracting 1 from Z."""
        factored = bool(self.factored_size_enabled)
        if factored and getattr(self, "quad_a", None) is not None:
            n = torch.bincount(line_trip, minlength=ix.B)
            fixed = self.log_Z_observed_size(
                ix, n, return_ess=return_ess, return_mode=return_mode)
            fixed = list(fixed) if isinstance(fixed, tuple) else [fixed]
            lzn = fixed.pop(0)
            ess = fixed.pop(0) if return_ess else None
            zh = fixed.pop(0) if return_mode else None
            ll = (self.energy(line_item, line_trip, line_cat, ix.B, line_ctx) - lzn
                  + self.factored_size_log_p[n - 1])
            if units is not None:
                ll = ll + self.units_loglik(
                    line_item, line_trip, units, line_ctx, ix.B)
            res = [ll]
            if return_ess:
                res.append(ess)
            if return_size:
                res.append(self.factored_size_log_p.exp().unsqueeze(0).expand(ix.B, -1))
            if return_mode:
                res.append(zh)
            return res[0] if len(res) == 1 else tuple(res)
        need_size = return_size or factored
        out = self.log_Z(ix, n_draws=n_draws, generator=generator,
                         return_ess=return_ess, drop_empty=True,
                         return_size=need_size, z_init=z_init,
                         return_mode=return_mode, mode_steps=mode_steps,
                         mix_scales=mix_scales, aniso=aniso,
                         antithetic=antithetic)
        out = list(out) if isinstance(out, tuple) else [out]
        lz1 = out.pop(0)
        ess = out.pop(0) if return_ess else None
        pn_internal = out.pop(0) if need_size else None
        zh = out.pop(0) if return_mode else None
        ll = self.energy(line_item, line_trip, line_cat, ix.B, line_ctx) - lz1
        if factored:
            n = torch.bincount(line_trip, minlength=ix.B)
            if bool((n <= 0).any()) or int(n.max()) > pn_internal.shape[1]:
                raise ValueError("observed basket lies outside factored size support")
            row = torch.arange(ix.B, device=line_trip.device)
            ll = (ll - pn_internal[row, n - 1].clamp_min(1e-300).log()
                  + self.factored_size_log_p[n - 1])
        if units is not None:
            ll = ll + self.units_loglik(line_item, line_trip, units, line_ctx, ix.B)
        res = [ll]
        if return_ess:
            res.append(ess)
        if return_size:
            pn = (self.factored_size_log_p.exp().unsqueeze(0).expand(ix.B, -1)
                  if factored else pn_internal)
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
                               * self.pair_feature(r))
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
            # `self.lam = lam0.data` above lands in self.__dict__, because at that moment
            # 'lam' had been popped from _parameters and .data is a plain Tensor.  Restoring
            # _parameters is NOT enough: nn.Module.__getattr__ only runs when normal lookup
            # FAILS, and the __dict__ entry means it never fails again.  Left behind, self.lam
            # is a detached tensor for the rest of the process, so b_flat contributes no
            # gradient to lam and Adam skips it -- lam froze after a single step (|lam| <= lr
            # exactly, 1740 products at exactly 0) from run29 through run71.  Drop the shadow.
            self.__dict__.pop('lam', None)
            self._parameters['lam'] = lam0
            self.ctx = old
        return pi

    def phi_radius(self, iters=200):
        """c = max_{||u||=1} sum_j max(phi_j'u, 0) -- the radius at which the latent
        target keeps its mass, and the quantity that decides whether log Z is computable.

        Along direction u the item weights are w_j = exp(b_j - ||phi_j||^2/2 + phi_j'z), so
        as ||z|| -> infinity the best subset picks up every product with a positive
        projection and

            log f(z)  ~  c ||z||,      log p(z) = log f(z) - ||z||^2/2  ~  c||z|| - ||z||^2/2

        which peaks at ||z|| ~ c and gives log Z ~ c^2/2.  The proposal is a Gaussian at the
        mode with sd capped at 4.47, and the prior's typical radius is sqrt(Kz), so unless
        c is of that order the sampler never visits the region that carries the mass.
        Measured on this project's checkpoints: c = 74.1 (run84), 59.3 (run80), 49.4
        (run68), 255.6 (run39) against sqrt(Kz) = 5.66 -- so log Z was really 1.2e3..3.3e4
        while the estimator reported ~10.7, and every likelihood ever logged here was
        computed against a normaliser that was wrong by thousands of nats.
        c is homogeneous of degree 1 in phi, so rescaling phi by c_max/c lands exactly on
        the constraint.  The objective is convex in u, so subgradient ascent converges.
        """
        with torch.no_grad():
            nz = self.phi.norm(dim=1) > 1e-9
            if not bool(nz.any()):
                return 0.0
            P = self.phi[nz]
            u = P.sum(0)
            if float(u.norm()) < 1e-12:
                u = P[0].clone()
            u = u / u.norm().clamp_min(1e-12)
            for _ in range(iters):
                g = (P * ((P @ u) > 0).to(P.dtype).unsqueeze(-1)).sum(0)
                n = g.norm()
                if float(n) < 1e-14:
                    break
                un = g / n
                if float((un - u).norm()) < 1e-12:
                    u = un
                    break
                u = un
            return float((P @ u).clamp_min(0).sum())

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
                               * self.pair_feature(r))
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
        # Squash smoothly instead of clamping.  A hard clamp makes the step INDEPENDENT of
        # the error once it binds: at |b| >= b_max the correction is a fixed b_max * damp in
        # the rho_0 slope, which with dE[n]/db = -Var(n) ~ 182 moves E[n] by 3.6 per
        # iteration whether the error is 1 or 40.  A fixed step cannot settle -- it can only
        # limit-cycle, and run64 duly oscillated E[n] between 7.2 and 50.0 with the clamp
        # binding on a rising share of iterations (`bang` 17 -> 126).
        #
        # b_max * tanh(b / b_max) is identical to b for small b (tanh x = x - x^3/3 + ...),
        # bounded by b_max exactly as before, and -- the point -- still strictly increasing
        # in the error, so the correction shrinks as the model approaches target.
        b = float(b_max * math.tanh(b / b_max)) * damp
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

        The subtract-a-constant step above is multiplicative ONLY because softplus(x) ~ e^x
        near x = -3.9.  Under price_soft the parameters ARE the coefficients (gamma =
        +0.0213 after the warm start), so subtracting c is an additive shift and any c >
        0.0213 drives gamma negative -- the price term changes sign, every utility runs
        away, and E[n] pins at n_max.  That is run407/run408, and no learning rate can
        prevent it because a projection is not an optimiser step.  In that parameterisation
        the exact analogue is a multiply, which also drops the e^x approximation.
        """
        gb = (self.price_g().mean(0) * self.price_b().mean(0)).sum()
        cur = float(gb)
        if cur <= 0 or target_gb <= 0:
            return
        if getattr(self, "price_soft", False):
            r = math.sqrt(target_gb / cur)          # gb is bilinear: scaling both by r
            self.gamma *= r                         # scales the product by exactly r^2
            self.beta *= r
            return
        c = 0.5 * math.log(cur / target_gb)
        self.gamma -= c
        self.beta -= c

    @torch.no_grad()
    def project(self, phi_max, budget=None, thresh=0.0, centre=False, whiten=0.0,
                op_max=0.0):
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
        if whiten > 0 or op_max > 0:
            if not bool(torch.isfinite(self.phi).all()):
                raise FloatingPointError("non-finite phi before spectral projection")
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
            # Work through the Kz x Kz Gram matrix instead of taking an SVD of the tall
            # J x Kz matrix.  With whitening plus an active operator cap, every singular
            # value can become exactly sqrt(op_max).  LAPACK's divide-and-conquer SVD has
            # intermittently failed to converge on that intentionally repeated spectrum
            # (run155, immediately after the iteration-2000 checkpoint), even though phi
            # was finite and Phi'Phi was 2 I to machine precision.  Symmetric eigh on the
            # 32 x 32 Gram matrix is both cheaper and stable for repeated eigenvalues.
            #
            # If Phi = U diag(S) V', changing only its singular values is equivalently
            #
            #   Phi_new = Phi V diag(S_new / S) V'.
            #
            # This avoids constructing U.  A numerically zero direction stays zero rather
            # than inventing an arbitrary left singular vector; the production matrix is
            # full rank, so this only defines safe behaviour for degenerate test cases.
            gram = self.phi.transpose(0, 1) @ self.phi
            gram = 0.5 * (gram + gram.transpose(0, 1))
            if not bool(torch.isfinite(gram).all()):
                raise FloatingPointError(
                    "non-finite Phi'Phi before spectral projection; "
                    f"max|phi|={float(self.phi.abs().max()):.6g}")
            evals, V = torch.linalg.eigh(gram)
            if not bool(torch.isfinite(evals).all()) or not bool(torch.isfinite(V).all()):
                raise FloatingPointError("symmetric eigensolver returned a non-finite spectrum")
            S = evals.clamp_min(0.0).sqrt()
            S2 = ((1 - whiten) * S + whiten * S.mean()) if whiten > 0 else S
            if whiten > 0:
                S2 = S2 * (S.norm() / S2.norm().clamp_min(1e-30))
            if op_max > 0:
                # Operator-norm projection.  For every n-item subset x,
                #
                #   pair(x) = (||Phi' x||^2 - sum_j x_j||phi_j||^2)/2
                #           <= lambda_max(Phi'Phi) * n / 2.
                #
                # A row-norm cap does not control the sum of thousands of aligned rows;
                # this does.  op_max is lambda_max(Phi'Phi), so singular values are capped
                # at sqrt(op_max).  At op_max=2 a pair with dot product 0.92 remains
                # feasible, while the size-120 clique that invalidated run100 cannot build
                # 30 units of catalogue energy in one direction.
                S2 = S2.clamp_max(math.sqrt(op_max))
            tol = torch.finfo(S.dtype).eps * max(self.phi.shape) * S.max().clamp_min(1.0)
            scale = torch.where(S > tol, S2 / S.clamp_min(tol), torch.zeros_like(S))
            projected = (self.phi @ V * scale.unsqueeze(0)) @ V.transpose(0, 1)
            if not bool(torch.isfinite(projected).all()):
                raise FloatingPointError(
                    "spectral projection produced non-finite phi; "
                    f"singular range={float(S.min()):.6g}..{float(S.max()):.6g}, "
                    f"scale range={float(scale.min()):.6g}..{float(scale.max()):.6g}")
            self.phi.copy_(projected)
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
