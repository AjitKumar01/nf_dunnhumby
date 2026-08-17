"""
Version 3 of the basket model: energy, normaliser, likelihood, exact sampler.

The model, in full (paper/version_3.html Eq. 8-9).  For a trip by household i with a store
assortment split into C categories:

    E(S) = sum_{j in S} b_ij
         + sum_{{j,k} subset S} phi_j' phi_k
         - sum_c rho_c n_c(n_c-1)/2
         - rho_0(n)

    P(S) = exp E(S) / Z,      Z = sum over ALL subsets of the assortment.

Theorem 1 turns that sum into a K_z-dimensional Gaussian integral.  Writing
b~_j = b_j - ||phi_j||^2/2 and w_j(z) = exp(b~_j + z'phi_j),

    G_c[r] = exp(-rho_c r(r-1)/2) e_r(w^(c)),     e_r the elementary symmetric polynomial
    A_n    = coefficient of u^n in prod_c ( sum_r G_c[r] u^r )
    f(z)   = sum_n exp(-rho_0(n)) A_n(z)
    Z      = E_{z ~ N(0,I)} [ f(z) ]

and conditionally on z the model is an exact three-level nest (Eq. 18b): total size, then
allocation across categories, then which products within a category.  `sample` walks down
those three levels, so it is exact given z.

SCOPE OF THIS FILE.  A DENSE assortment -- every trip sees the same C categories with P
products each.  That is what the synthetic recovery experiment needs, and it keeps the
tensor algebra readable.  A dunnhumby fit needs the ragged version (store-specific
assortments, category sizes from 1 to 182); the mathematics is identical and the
bookkeeping is not, which is why it is a separate job and not a flag.

NUMERICS.  All weights are divided by the trip's largest weight before the elementary
symmetric polynomials are formed, so nothing overflows; e_r is homogeneous of degree r, so
the scale comes back as n*log(M) inside the final log-sum-exp over n.
"""
import math

import torch

NEG = -1e30


# --------------------------------------------------------------------------- pieces


def esp_dense(w, R):
    """e_0..e_R over the last axis.  w [..., P] -> [..., R+1].

    The recursion e_r <- e_r + w_i e_{r-1} costs O(P R) and needs no masking: a zero
    weight is invisible to it, which is how padding would be handled in the ragged case.
    """
    shape = w.shape[:-1]
    e = [torch.ones(shape, dtype=w.dtype, device=w.device)]
    e += [torch.zeros(shape, dtype=w.dtype, device=w.device) for _ in range(R)]
    for i in range(w.shape[-1]):
        wi = w[..., i]
        for r in range(R, 0, -1):
            e[r] = e[r] + wi * e[r - 1]
    return torch.stack(e, dim=-1)


def poly_mul_trunc(A, G, nmax):
    """Multiply polynomials along the last axis, truncating at degree nmax.

    A [..., LA], G [..., LG] -> [..., nmax+1].  This is the step that couples categories:
    it is what enforces sum_c r_c = n in Theorem 1.
    """
    out = torch.zeros(A.shape[:-1] + (nmax + 1,), dtype=A.dtype, device=A.device)
    LA, LG = A.shape[-1], G.shape[-1]
    for r in range(min(LG, nmax + 1)):
        take = min(LA, nmax + 1 - r)
        out[..., r:r + take] = out[..., r:r + take] + A[..., :take] * G[..., r:r + 1]
    return out


class Model(torch.nn.Module):
    """Parameters and the three quantities everything else is built from: the energy of an
    observed basket, log f(z), and log Z."""

    def __init__(self, J, N, C, P, K=8, Kz=3, nmax=20, R=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.J, self.N, self.C, self.P = J, N, C, P
        self.K, self.Kz, self.nmax, self.R = K, Kz, nmax, R
        self.lam = torch.nn.Parameter(torch.zeros(J))
        self.alpha = torch.nn.Parameter(torch.randn(J, K, generator=g) * 0.3)
        self.theta = torch.nn.Parameter(torch.randn(N, K, generator=g) * 0.3)
        self.phi = torch.nn.Parameter(torch.randn(J, Kz, generator=g) * 0.15)
        self.rho_c = torch.nn.Parameter(torch.zeros(C))
        # rho_0(0) is pinned at 0: it fixes the scale, since P(empty) = 1/Z (Corollary 1)
        self.rho_0_free = torch.nn.Parameter(torch.zeros(nmax))
        self.register_buffer("cat_of", torch.arange(J) // P)

    def rho_0(self):
        return torch.cat([torch.zeros(1, dtype=self.rho_0_free.dtype,
                                      device=self.rho_0_free.device), self.rho_0_free])

    # ---------------------------------------------------------------- item values

    def b(self, house):
        """b_ij for every product, [B, J].  Extend here for price, promotion, coupon."""
        return self.lam.unsqueeze(0) + self.theta[house] @ self.alpha.T

    def b_tilde(self, house):
        return self.b(house) - 0.5 * (self.phi ** 2).sum(-1).unsqueeze(0)

    # ------------------------------------------------------------------ integrand

    def log_f(self, z, bt):
        """log f(z) for a batch of trips and draws.

        z  [B, D, Kz]      bt [B, J]        returns [B, D]
        """
        B, D = z.shape[0], z.shape[1]
        proj = torch.einsum("bdk,jk->bdj", z, self.phi)          # [B, D, J]
        logw = bt.unsqueeze(1) + proj
        M = logw.amax(dim=-1, keepdim=True)                      # per (trip, draw)
        w = torch.exp(logw - M).view(B, D, self.C, self.P)
        e = esp_dense(w, self.R)                                 # [B, D, C, R+1]
        r = torch.arange(self.R + 1, dtype=w.dtype, device=w.device)
        a = torch.exp(-self.rho_c.view(1, 1, self.C, 1) * r * (r - 1) / 2.0)
        G = a * e
        A = G[:, :, 0, :]
        for c in range(1, self.C):
            A = poly_mul_trunc(A, G[:, :, c, :], self.nmax)
        n = torch.arange(A.shape[-1], dtype=w.dtype, device=w.device)
        # f = sum_n exp(-rho_0(n)) A_n M^n, in the log domain
        return torch.logsumexp(
            torch.log(A.clamp_min(1e-300)) - self.rho_0()[: A.shape[-1]] + n * M, dim=-1)

    # ----------------------------------------------------------------- normaliser

    def log_Z(self, house, n_draws=64, mode_steps=12, generator=None, return_ess=False):
        """log Z by importance sampling from a Laplace proposal at the mode of
        F(z) = -||z||^2/2 + log f(z).

        The mode solves z = grad_z log f(z), which is the self-consistent-field equation of
        section 14.1; iterating it needs no step size.  The proposal is built under no_grad
        and treated as a constant, which is legitimate because importance sampling is
        unbiased for any fixed proposal -- so autograd through the weights gives the
        gradient of log Z without differentiating the optimiser.
        """
        bt = self.b_tilde(house)
        B = bt.shape[0]
        with torch.no_grad():
            z = torch.zeros(B, 1, self.Kz, dtype=bt.dtype, device=bt.device)
            for _ in range(mode_steps):
                zz = z.detach().requires_grad_(True)
                with torch.enable_grad():
                    lf = self.log_f(zz, bt.detach()).sum()
                z = torch.autograd.grad(lf, zz)[0]
            zh = z.detach()
            eps = 0.15                                    # diagonal curvature at the mode
            curv = torch.zeros(B, self.Kz, dtype=bt.dtype, device=bt.device)
            for k in range(self.Kz):
                d = torch.zeros(B, 1, self.Kz, dtype=bt.dtype, device=bt.device)
                d[:, :, k] = eps
                gp, gm = [], []
                for s in (d, -d):
                    zz = (zh + s).detach().requires_grad_(True)
                    with torch.enable_grad():
                        lf = self.log_f(zz, bt.detach()).sum()
                    g = torch.autograd.grad(lf, zz)[0] - zz.detach()
                    (gp if s is d else gm).append(g[:, 0, k])
                curv[:, k] = -(gp[0] - gm[0]) / (2 * eps)
            sd = (1.0 / curv.clamp_min(0.05)).sqrt().clamp(0.05, 5.0)
            noise = torch.randn(B, n_draws, self.Kz, dtype=bt.dtype,
                                device=bt.device, generator=generator)
            zs = zh + noise * sd.unsqueeze(1)
            L2P = float(math.log(2 * math.pi))
            log_q = (-0.5 * (noise ** 2).sum(-1) - sd.log().sum(-1, keepdim=True)
                     - 0.5 * self.Kz * L2P)
        L2P = float(math.log(2 * math.pi))
        log_p = -0.5 * self.Kz * L2P - 0.5 * (zs ** 2).sum(-1) + self.log_f(zs, bt)
        lw = log_p - log_q
        lz = torch.logsumexp(lw, dim=1) - math.log(n_draws)
        if not return_ess:
            return lz
        with torch.no_grad():
            ww = torch.softmax(lw, dim=1)
            ess = 1.0 / (ww ** 2).sum(1) / n_draws
        return lz, ess

    # --------------------------------------------------------------------- energy

    def energy(self, house, S):
        """E(S) for observed baskets.  S is a [B, J] binary tensor."""
        b = self.b(house)
        lin = (b * S).sum(-1)
        v = S @ self.phi                                          # [B, Kz]
        pair = 0.5 * ((v * v).sum(-1) - (S * (self.phi ** 2).sum(-1)).sum(-1))
        nc = S.view(-1, self.C, self.P).sum(-1)                   # [B, C]
        pen_c = (self.rho_c.unsqueeze(0) * nc * (nc - 1) / 2.0).sum(-1)
        n = S.sum(-1).long().clamp(max=self.nmax)
        return lin + pair - pen_c - self.rho_0()[n]

    def loglik(self, house, S, n_draws=64, generator=None, return_ess=False):
        """log P(S | non-empty).  Corollary 1 makes the conditioning exact: the empty
        basket has weight 1, so its probability is 1/Z and the conditional normaliser is
        Z - 1."""
        out = self.log_Z(house, n_draws=n_draws, generator=generator,
                         return_ess=return_ess)
        lz, ess = out if return_ess else (out, None)
        # log(Z - 1) = log Z + log(1 - exp(-log Z)), stable for log Z > 0
        lzm1 = lz + torch.log1p(-torch.exp(-lz.clamp_min(1e-6)))
        ll = self.energy(house, S) - lzm1
        return (ll, ess) if return_ess else ll


# ----------------------------------------------------------------------- sampling


@torch.no_grad()
def sample(model, house, n_draws=128, generator=None):
    """Exact given z: walk down the three levels of Eq. 18b.

    1  z from its posterior, by sampling-importance-resampling from the Laplace proposal.
    2  the total size n, from P(n|z) proportional to exp(-rho_0(n)) A_n(z).
    3  the split (r_c) given n, by a backward pass over the suffix products.
    4  the r_c products within each category, by the sequential suffix-ESP algorithm.

    Steps 2-4 are exact.  Step 1 is exact only as n_draws grows; its error is governed by
    the effective sample size, which log_Z reports.
    """
    bt = model.b_tilde(house)
    B, dev, dt = bt.shape[0], bt.device, bt.dtype
    z0 = torch.zeros(B, 1, model.Kz, dtype=dt, device=dev)
    for _ in range(12):
        zz = z0.detach().requires_grad_(True)
        with torch.enable_grad():
            lf = model.log_f(zz, bt).sum()
        z0 = torch.autograd.grad(lf, zz)[0]
    zs = z0 + torch.randn(B, n_draws, model.Kz, dtype=dt, device=dev,
                          generator=generator)
    lp = -0.5 * (zs ** 2).sum(-1) + model.log_f(zs, bt)
    lq = -0.5 * ((zs - z0) ** 2).sum(-1)
    pick = torch.distributions.Categorical(logits=lp - lq).sample()
    z = zs[torch.arange(B, device=dev), pick]                     # [B, Kz]

    logw = bt + z @ model.phi.T
    M = logw.amax(dim=-1, keepdim=True)
    w = torch.exp(logw - M).view(B, model.C, model.P)
    e = esp_dense(w, model.R)
    r = torch.arange(model.R + 1, dtype=dt, device=dev)
    G = torch.exp(-model.rho_c.view(1, model.C, 1) * r * (r - 1) / 2.0) * e

    # suffix products over categories, so the split can be drawn backwards
    suf = [None] * (model.C + 1)
    suf[model.C] = torch.ones(B, 1, dtype=dt, device=dev)
    for c in range(model.C - 1, -1, -1):
        suf[c] = poly_mul_trunc(G[:, c, :], suf[c + 1], model.nmax)

    A = suf[0]
    n_ax = torch.arange(A.shape[-1], dtype=dt, device=dev)
    logpn = torch.log(A.clamp_min(1e-300)) - model.rho_0()[: A.shape[-1]] + n_ax * M
    n = torch.distributions.Categorical(logits=logpn).sample()    # [B]

    S = torch.zeros(B, model.J, dtype=dt, device=dev)
    left = n.clone()
    for c in range(model.C):
        tail = suf[c + 1]
        L = tail.shape[-1]
        idx = (left.unsqueeze(1) - torch.arange(model.R + 1, device=dev).unsqueeze(0))
        ok = (idx >= 0) & (idx < L)
        gathered = tail.gather(1, idx.clamp(0, L - 1)) * ok
        pr = (G[:, c, :] * gathered).clamp_min(0)
        rc = torch.distributions.Categorical(probs=pr + 1e-300).sample()
        for bi in range(B):
            k = int(rc[bi])
            if k:
                chosen = _seq_choose(w[bi, c], k, generator)
                S[bi, c * model.P + torch.tensor(chosen, device=dev)] = 1.0
        left = left - rc
    return S, n


def _seq_choose(w, k, generator=None):
    """Exactly k of the P products with probability proportional to the product of their
    weights, in O(P k) with no rejection, by suffix elementary symmetric polynomials."""
    P = w.shape[0]
    suf = torch.zeros(P + 1, k + 1, dtype=w.dtype, device=w.device)
    suf[P, 0] = 1.0
    for i in range(P - 1, -1, -1):
        suf[i] = suf[i + 1]
        suf[i, 1:] = suf[i + 1, 1:] + w[i] * suf[i + 1, :-1]
    out, need = [], k
    for i in range(P):
        if need == 0:
            break
        if P - i == need:
            out.extend(range(i, P))
            break
        den = suf[i, need]
        if den <= 0:
            out.extend(range(i, i + need))
            break
        p = float(w[i] * suf[i + 1, need - 1] / den)
        if torch.rand(1, generator=generator, device=w.device).item() < p:
            out.append(i)
            need -= 1
    return out
