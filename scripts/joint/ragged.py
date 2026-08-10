"""
Ragged log-normaliser: same mathematics as logz.py, no padding.

logz.py pads every category to the batch maximum, which is 182 items against a median of
30 -- roughly a 6x waste, and the reason a correct fit measured 17.8 hours instead of the
1 hour the feasibility study projected.

This version keeps every item in one flat array with a row index and uses scatter
operations, so a 5-item category costs 5 slots rather than 182.

The elementary symmetric polynomials come from POWER SUMS via Newton's identities:

    p_i = sum_j w_j^i        (a scatter_add, no padding, no recursion)

    e_1 = p1
    e_2 = (p1^2 - p2) / 2
    e_3 = (p1^3 - 3 p1 p2 + 2 p3) / 6
    e_4 = (p1^4 - 6 p1^2 p2 + 3 p2^2 + 8 p1 p3 - 6 p4) / 24

which covers 99.9% of rows -- k_c is 1 for 81.2%, 2 for 13.3%, 3 for 3.5%.

The inclusion probabilities needed for the mode follow from the same power sums by leaving
one item out, since p_i^(-j) = p_i - w_j^i:

    pi_j = w_j * e_{k-1}(w without j) / e_k(w)

WHERE THIS COULD GO WRONG, and the guard.  Newton's identities subtract quantities of
similar size, so if one weight dominates the others the difference cancels and precision is
lost.  Measured: at a top-two log-weight gap of 27 (a weight ratio of 1e12) the k = 3 case
carries 23% relative error.  On the fitted model that gap has median 0.051 and maximum
1.081, so this never fires -- but a differently scaled model could, so `_cancellation` flags
affected rows and the caller can fall back to the padded recursion in logz.py.
"""
import numpy as np
import torch


def _seg_sum(vals, row_id, n_rows):
    """Sum vals within each row.  vals [..., R], row_id [R] -> [..., n_rows]."""
    out = torch.zeros(vals.shape[:-1] + (n_rows,), device=vals.device, dtype=vals.dtype)
    return out.index_add_(-1, row_id, vals)


def _seg_max(vals, row_id, n_rows):
    out = torch.full(vals.shape[:-1] + (n_rows,), -float("inf"),
                     device=vals.device, dtype=vals.dtype)
    return out.index_reduce_(-1, row_id, vals, "amax", include_self=True)


def log_esp_ragged(logw, row_id, row_k, n_rows, kmax=4):
    """log e_{k_r} for every row r.  Returns [..., n_rows].

    Weights are divided by each row's maximum before the power sums are formed, so the
    largest term is exactly 1 and nothing overflows; the k*M is added back at the end
    because e_k is homogeneous of degree k.
    """
    M = _seg_max(logw, row_id, n_rows)                         # [..., n_rows]
    w = torch.exp(logw - M.index_select(-1, row_id))
    p = [_seg_sum(w ** i, row_id, n_rows) for i in range(1, kmax + 1)]
    e = torch.zeros_like(p[0])
    k1, k2, k3 = (row_k == 1), (row_k == 2), (row_k == 3)
    k4 = row_k == 4
    e = torch.where(k1, p[0], e)
    if kmax >= 2:
        e = torch.where(k2, (p[0] ** 2 - p[1]) / 2, e)
    if kmax >= 3:
        e = torch.where(k3, (p[0] ** 3 - 3 * p[0] * p[1] + 2 * p[2]) / 6, e)
    if kmax >= 4:
        e = torch.where(k4, (p[0] ** 4 - 6 * p[0] ** 2 * p[1] + 3 * p[1] ** 2
                             + 8 * p[0] * p[2] - 6 * p[3]) / 24, e)
    return e.clamp_min(1e-300).log() + row_k.to(logw.dtype) * M


def inclusion_ragged(logw, row_id, row_k, n_rows, kmax=4):
    """P(item is among the k chosen), per item.  Returns [..., R].

    e_{k-1}(w without j) is formed from leave-one-out power sums p_i - w_j^i, which is
    exact and needs no per-row loop.
    """
    M = _seg_max(logw, row_id, n_rows)
    w = torch.exp(logw - M.index_select(-1, row_id))
    p = [_seg_sum(w ** i, row_id, n_rows) for i in range(1, kmax + 1)]
    pj = [q.index_select(-1, row_id) for q in p]                # broadcast back to items
    kj = row_k.index_select(-1, row_id)
    # leave-one-out power sums
    q1 = pj[0] - w
    q2 = pj[1] - w ** 2 if kmax >= 2 else None
    q3 = pj[2] - w ** 3 if kmax >= 3 else None
    num = torch.zeros_like(w)
    num = torch.where(kj == 1, torch.ones_like(w), num)                     # e_0 = 1
    if kmax >= 2:
        num = torch.where(kj == 2, q1, num)                                 # e_1(w_-j)
    if kmax >= 3:
        num = torch.where(kj == 3, (q1 ** 2 - q2) / 2, num)                 # e_2(w_-j)
    if kmax >= 4:
        num = torch.where(kj == 4, (q1 ** 3 - 3 * q1 * q2 + 2 * q3) / 6, num)
    e = torch.zeros_like(p[0])
    e = torch.where(row_k == 1, p[0], e)
    if kmax >= 2:
        e = torch.where(row_k == 2, (p[0] ** 2 - p[1]) / 2, e)
    if kmax >= 3:
        e = torch.where(row_k == 3, (p[0] ** 3 - 3 * p[0] * p[1] + 2 * p[2]) / 6, e)
    if kmax >= 4:
        e = torch.where(row_k == 4, (p[0] ** 4 - 6 * p[0] ** 2 * p[1] + 3 * p[1] ** 2
                                     + 8 * p[0] * p[2] - 6 * p[3]) / 24, e)
    return w * num / e.index_select(-1, row_id).clamp_min(1e-300)


def cancellation_risk(logw, row_id, row_k, n_rows):
    """Rows where Newton's identities lose precision, as a fraction of the leading term.

    e_2 = (p1^2 - p2)/2 loses digits when p2 approaches p1^2, i.e. when one weight
    dominates.  Returns the ratio (p1^2 - p2)/p1^2 per row; values below about 1e-8 mean
    roughly half the mantissa has gone and the padded recursion should be used instead.
    """
    M = _seg_max(logw, row_id, n_rows)
    w = torch.exp(logw - M.index_select(-1, row_id))
    p1 = _seg_sum(w, row_id, n_rows)
    p2 = _seg_sum(w ** 2, row_id, n_rows)
    return torch.where(row_k > 1, (p1 ** 2 - p2) / (p1 ** 2).clamp_min(1e-300),
                       torch.ones_like(p1))


class RaggedNormaliser:
    """log Z by importance sampling at the mode, with no padded dimension anywhere."""

    def __init__(self, n_draws=64, mode_steps=25, kmax=4):
        self.n_draws, self.mode_steps, self.kmax = n_draws, mode_steps, kmax

    def _logint(self, z, cj, phi, row_id, row_k, row_b, n_rows, n_bask, rl):
        """log integrand per (basket, draw).  z [B, D, K] -> [B, D]."""
        # phi [R, K], z[row_b[row_id]] picks each item's basket's draw
        zb = z.index_select(0, row_b.index_select(0, row_id))       # [R, D, K]
        proj = (zb * phi.unsqueeze(1)).sum(-1)                      # [R, D]
        logw = (cj.unsqueeze(1) + rl.index_select(0, row_b.index_select(0, row_id))
                .unsqueeze(1) * proj).transpose(0, 1)               # [D, R]
        le = log_esp_ragged(logw, row_id, row_k, n_rows, self.kmax)  # [D, n_rows]
        return _seg_sum(le, row_b, n_bask).transpose(0, 1)          # [B, D]

    @torch.no_grad()
    def find_mode(self, cj, phi, row_id, row_k, row_b, n_rows, n_bask, rl, z0=None):
        K = phi.shape[-1]
        z = torch.zeros(n_bask, K, dtype=phi.dtype) if z0 is None else z0.clone()
        ib = row_b.index_select(0, row_id)                          # basket of each item
        for _ in range(self.mode_steps):
            proj = (z.index_select(0, ib) * phi).sum(-1)            # [R]
            logw = cj + rl.index_select(0, ib) * proj
            pi = inclusion_ragged(logw, row_id, row_k, n_rows, self.kmax)
            acc = torch.zeros(n_bask, K, dtype=phi.dtype)
            acc.index_add_(0, ib, pi.unsqueeze(-1) * phi)
            z = rl.unsqueeze(-1) * acc
        return z

    def log_z(self, cj, phi, row_id, row_k, row_b, n_rows, n_bask, lam,
              z0=None, generator=None):
        K = phi.shape[-1]
        rl = lam.sqrt()
        with torch.no_grad():
            zh = self.find_mode(cj.detach(), phi.detach(), row_id, row_k, row_b,
                                n_rows, n_bask, rl, z0)
            ib = row_b.index_select(0, row_id)
            eps = 0.15

            def gr(zz):
                proj = (zz.index_select(0, ib) * phi.detach()).sum(-1)
                lw = cj.detach() + rl.index_select(0, ib) * proj
                pi = inclusion_ragged(lw, row_id, row_k, n_rows, self.kmax)
                acc = torch.zeros(n_bask, K, dtype=phi.dtype)
                acc.index_add_(0, ib, pi.unsqueeze(-1) * phi.detach())
                return -zz + rl.unsqueeze(-1) * acc

            curv = torch.zeros(n_bask, K, dtype=phi.dtype)
            for k in range(K):
                e = torch.zeros(n_bask, K, dtype=phi.dtype)
                e[:, k] = eps
                curv[:, k] = -(gr(zh + e)[:, k] - gr(zh - e)[:, k]) / (2 * eps)
            # `inflate` exists only so the experiment can be repeated.  Over-dispersing
            # the proposal is the textbook robustness fix for importance sampling and it
            # is WRONG here: measured at K = 64, a factor of 1.4 drops ESS from 0.76 to
            # 0.001 and 2.5 puts log Z 27 nats out.  In high dimension an inflated
            # Gaussian puts its mass on a shell away from the mode, where the integrand
            # is negligible.  Leave it at 1.
            sd = ((1.0 / curv.clamp_min(0.05)).sqrt().clamp(0.05, 5.0)
                  * getattr(self, "inflate", 1.0))
            noise = torch.randn(n_bask, self.n_draws, K, dtype=phi.dtype,
                                generator=generator)
            z = zh.unsqueeze(1) + noise * sd.unsqueeze(1)
            L2P = float(np.log(2 * np.pi))
            log_q = (-0.5 * (noise ** 2).sum(-1) - sd.log().sum(-1, keepdim=True)
                     - 0.5 * K * L2P)
        L2P = float(np.log(2 * np.pi))
        log_p = (-0.5 * K * L2P - 0.5 * (z ** 2).sum(-1)
                 + self._logint(z, cj, phi, row_id, row_k, row_b, n_rows, n_bask, rl))
        lw_ = log_p - log_q
        lz = torch.logsumexp(lw_, 1) - float(np.log(self.n_draws))
        with torch.no_grad():
            ww = torch.softmax(lw_, 1)
            ess = 1.0 / (ww ** 2).sum(1) / self.n_draws
        return lz, zh, ess
