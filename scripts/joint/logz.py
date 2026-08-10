"""
Exact log-normaliser for P(S | K) proportional to exp E(S), batched and differentiable.

The derivation is in paper/joint_likelihood.html.  In one line: n is fixed given the
composition K, so lambda = 1/(n-1) is a constant, the pair term becomes half a squared
norm, and the Gaussian identity exp(||v||^2/2) = E_z[exp(z'v)] turns the sum over up to
10^124 baskets into

    Z = E_{z ~ N(0,I)} [ prod_c  e_{k_c}( { exp(c_j + sqrt(lambda) z'phi_j) }_{j in C(c)} ) ]

where e_k is the elementary symmetric polynomial, computed by an O(N k) recursion.

Everything here is batched over baskets AND categories with padding, because a per-basket
Python loop is what made the first prototype cost 105 hours per fit.

Three things this module gets right that a first attempt would not:

  * PADDING IS FREE.  A padded slot carries log-weight -inf, hence weight 0, and a zero
    weight contributes nothing to any elementary symmetric polynomial.  So no masking
    logic is needed inside the recursion -- the padding is absorbed by the algebra.

  * THE PROPOSAL IS DETACHED.  Importance sampling is unbiased for ANY proposal, so the
    mode and the proposal covariance are computed under no_grad and treated as constants.
    Autograd through the weights then gives an unbiased gradient of log Z with respect to
    the model parameters, with no need to differentiate through the optimiser.

  * THE GRADIENT OF THE MODE OBJECTIVE IS ANALYTIC.  d log e_k / d log w_j is exactly the
    probability that item j is among the k chosen, computable for all j at once by a
    prefix/suffix pass.  Handing that to the optimiser instead of letting it estimate a
    derivative by finite differences is a measured 63x.
"""
import numpy as np
import torch

NEG = -1e30


def log_esp_k1(logw):
    """log e_1 = log sum w.  81.2% of (basket, category) cells have k_c = 1, and for those
    the recursion below is pure waste: e_1 is a plain log-sum-exp, and the inclusion
    probability is a plain softmax.  Splitting them out is the single largest saving in
    this module."""
    return torch.logsumexp(logw, dim=-1)


def log_esp(logw, kmax):
    """log e_0..e_kmax for each row.  logw is [..., N]; returns [..., kmax+1].

    Runs the recursion e_k <- e_k + w e_{k-1} in the log domain, factoring out the row
    maximum so that exp() never overflows.  Padded entries carry logw = -inf, so their
    weight is 0 and they are invisible to the recursion without any explicit mask.
    """
    M = logw.max(dim=-1, keepdim=True).values
    M = torch.where(torch.isfinite(M), M, torch.zeros_like(M))
    w = torch.exp(logw - M)                       # [..., N], padding -> 0
    shape = logw.shape[:-1]
    e = [torch.ones(shape, device=logw.device, dtype=logw.dtype)]
    e += [torch.zeros(shape, device=logw.device, dtype=logw.dtype) for _ in range(kmax)]
    for i in range(logw.shape[-1]):
        wi = w[..., i]
        for k in range(kmax, 0, -1):
            e[k] = e[k] + wi * e[k - 1]
    out = torch.stack(e, dim=-1).clamp_min(1e-300).log()
    # e_k is homogeneous of degree k, so undoing the shift costs k * M
    ks = torch.arange(kmax + 1, device=logw.device, dtype=logw.dtype)
    return out + ks * M


def inclusion_probs(logw, k_each, kmax):
    """P(item j is among the k chosen), for every row.  [..., N].

    d log e_k / d log w_j = w_j e_{k-1}(w without j) / e_k(w).  Computed by prefix and
    suffix elementary symmetric polynomials so that w_j is never divided out -- dividing
    out is what makes the naive version unstable when one weight dominates.
    """
    M = logw.max(dim=-1, keepdim=True).values
    M = torch.where(torch.isfinite(M), M, torch.zeros_like(M))
    if kmax == 1:
        return torch.softmax(logw, dim=-1)
    w = torch.exp(logw - M)
    N = logw.shape[-1]
    shape = logw.shape[:-1]
    z = lambda: torch.zeros(shape + (kmax + 1,), device=logw.device, dtype=logw.dtype)
    pre = [z() for _ in range(N + 1)]
    pre[0][..., 0] = 1.0
    for i in range(N):
        cur = pre[i].clone()
        cur[..., 1:] = cur[..., 1:] + w[..., i:i + 1] * pre[i][..., :-1]
        pre[i + 1] = cur
    suf = [z() for _ in range(N + 1)]
    suf[N][..., 0] = 1.0
    for i in range(N - 1, -1, -1):
        cur = suf[i + 1].clone()
        cur[..., 1:] = cur[..., 1:] + w[..., i:i + 1] * suf[i + 1][..., :-1]
        suf[i] = cur
    kk = k_each.unsqueeze(-1)                                     # [..., 1]
    ek = torch.gather(pre[N], -1, kk).squeeze(-1)                 # e_k over everything
    out = torch.zeros_like(w)
    for j in range(N):
        # e_{k-1} excluding j = sum_a pre_a(<j) * suf_{k-1-a}(>j)
        acc = torch.zeros(shape, device=logw.device, dtype=logw.dtype)
        for a in range(kmax):
            b = k_each - 1 - a
            ok = (b >= 0) & (b <= kmax)
            bb = b.clamp(0, kmax).unsqueeze(-1)
            acc = acc + torch.where(
                ok, pre[j][..., a] * torch.gather(suf[j + 1], -1, bb).squeeze(-1),
                torch.zeros_like(acc))
        out[..., j] = w[..., j] * acc / ek.clamp_min(1e-300)
    return out


class JointNormaliser:
    """log Z for a batch of compositions, by importance sampling at the mode."""

    def __init__(self, n_draws=128, mode_steps=40, mode_lr=None, kmax=8):
        self.n_draws, self.mode_steps, self.mode_lr, self.kmax = \
            n_draws, mode_steps, mode_lr, kmax

    @staticmethod
    def _esp_split(logw, k_each, kmax):
        """log e_{k} per row, taking the cheap path wherever k == 1."""
        flatw = logw.reshape(-1, logw.shape[-1])
        flatk = k_each.reshape(-1)
        out = torch.zeros(flatw.shape[0], device=logw.device, dtype=logw.dtype)
        one = flatk == 1
        if bool(one.any()):
            out[one] = log_esp_k1(flatw[one])
        many = flatk > 1
        if bool(many.any()):
            le = log_esp(flatw[many], kmax)
            out[many] = torch.gather(le, -1, flatk[many].unsqueeze(-1)).squeeze(-1)
        return out.reshape(logw.shape[:-1])

    def _logint(self, z, cj, phi, k_each, lam):
        """log of the integrand: sum_c log e_{k_c}(exp(c_j + sqrt(lam) z'phi_j)).

        z    [B, D, K]   D proposal draws per basket
        cj   [B, C, N]   per-item offset, padded with NEG
        phi  [B, C, N, K]
        """
        proj = torch.einsum("bdk,bcnk->bdcn", z, phi)
        logw = cj.unsqueeze(1) + lam.sqrt().view(-1, 1, 1, 1) * proj
        logw = torch.where(cj.unsqueeze(1) > NEG / 2, logw,
                           torch.full_like(logw, float("-inf")))
        ke = k_each.unsqueeze(1).expand(-1, z.shape[1], -1)         # [B, D, C]
        per_cat = self._esp_split(logw, ke, self.kmax)              # [B, D, C]
        per_cat = torch.where(ke > 0, per_cat, torch.zeros_like(per_cat))
        return per_cat.sum(-1)                                     # [B, D]

    @torch.no_grad()
    def find_mode(self, cj, phi, k_each, lam, z0=None):
        """Maximise log N(z;0,I) + log integrand(z), batched.

        Setting the exact gradient  -z + sqrt(lam) Phi' pi(z)  to zero gives the fixed
        point  z = sqrt(lam) Phi' pi(z), so simple iteration of that map is the natural
        solver -- it is the self-consistent-field iteration for this saddle point, needs
        no step size, and reaches machine precision in 15-40 passes at every dimension
        tested including K = 64.  A generic optimiser is much worse here: Adam needed
        about 400 steps for the same accuracy, and stopping it early collapsed the
        importance-sampling effective sample size to 0.18.
        """
        B, C, N, K = phi.shape
        z = torch.zeros(B, K, dtype=phi.dtype) if z0 is None else z0.clone()
        rl = lam.sqrt().view(-1, 1)
        for _ in range(self.mode_steps):
            proj = torch.einsum("bk,bcnk->bcn", z, phi)
            logw = cj + lam.sqrt().view(-1, 1, 1) * proj
            logw = torch.where(cj > NEG / 2, logw, torch.full_like(logw, float("-inf")))
            pi = inclusion_probs(logw, k_each, self.kmax)          # [B, C, N]
            pi = torch.where(k_each.unsqueeze(-1) > 0, pi, torch.zeros_like(pi))
            z = rl * torch.einsum("bcn,bcnk->bk", pi, phi)
        return z

    def log_z(self, cj, phi, k_each, lam, z0=None, generator=None):
        """Unbiased-in-Z importance-sampling estimate of log Z.  Differentiable in cj, phi.

        The proposal is built under no_grad and treated as a constant, which is legitimate
        because importance sampling is unbiased for any proposal; autograd through the
        weights therefore gives the gradient of log Z without differentiating the optimiser.
        """
        B, C, N, K = phi.shape
        with torch.no_grad():
            zh = self.find_mode(cj.detach(), phi.detach(), k_each, lam, z0)
            # diagonal curvature at the mode, by central differences on the exact gradient
            eps = 0.15
            def grad_at(zz):
                proj = torch.einsum("bk,bcnk->bcn", zz, phi.detach())
                lw = cj.detach() + lam.sqrt().view(-1, 1, 1) * proj
                lw = torch.where(cj.detach() > NEG / 2, lw,
                                 torch.full_like(lw, float("-inf")))
                pi = inclusion_probs(lw, k_each, self.kmax)
                pi = torch.where(k_each.unsqueeze(-1) > 0, pi, torch.zeros_like(pi))
                return -zz + lam.sqrt().view(-1, 1) * torch.einsum("bcn,bcnk->bk", pi,
                                                                   phi.detach())
            curv = torch.zeros(B, K, dtype=phi.dtype)
            for k in range(K):
                e = torch.zeros(B, K, dtype=phi.dtype)
                e[:, k] = eps
                curv[:, k] = -(grad_at(zh + e)[:, k] - grad_at(zh - e)[:, k]) / (2 * eps)
            sd = (1.0 / curv.clamp_min(0.05)).sqrt().clamp(0.05, 5.0)
            noise = torch.randn(B, self.n_draws, K, dtype=phi.dtype, generator=generator)
            z = zh.unsqueeze(1) + noise * sd.unsqueeze(1)
            LOG2PI = float(np.log(2 * np.pi))
            log_q = (-0.5 * (noise ** 2).sum(-1) - sd.log().sum(-1, keepdim=True)
                     - 0.5 * K * LOG2PI)
        LOG2PI = float(np.log(2 * np.pi))
        log_p = -0.5 * K * LOG2PI - 0.5 * (z ** 2).sum(-1) + \
            self._logint(z, cj, phi, k_each, lam)
        lw = log_p - log_q
        lz = torch.logsumexp(lw, dim=1) - float(np.log(self.n_draws))
        with torch.no_grad():
            w = torch.softmax(lw, dim=1)
            ess = 1.0 / (w ** 2).sum(1) / self.n_draws
        return lz, zh, ess
