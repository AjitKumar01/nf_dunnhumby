"""What the model can answer that the baselines cannot.

The differentiator is NOT that basket size responds to price.  Checked in the code: the
BEMB multinomial cannot (log_pn is a counted buffer indexed by n, price is not an input),
but SHOPPER can (item utilities carry price and compete against a fixed checkout scalar, so
cutting prices lengthens the basket), and so can the Bernoulli and the DPPs.

The differentiator is the CROSS-item structure, and it is exact here rather than simulated.
Corollary 2 gives P(j in S) = d log(Z-1) / d b_j.  Differentiating once more,

    d P(k in S) / d b_j  =  d^2 log(Z-1) / d b_j d b_k  =  Cov(x_j, x_k)

so one backward pass through the normaliser yields the whole cross-response matrix, signed:
positive entries are complements, negative are substitutes.  Converting through the price
term gives the cross-price elasticity directly, with no simulation and no finite differences.

For the baselines the same quantity is either identically zero or sign-constrained:
  Bernoulli      Cov(x_j, x_k) = 0 for all j != k, by construction
  symmetric DPP  Cov(x_j, x_k) <= 0 always -- a symmetric determinantal kernel is negatively
                 associated, so it can express substitution but never complementarity
  multinomial    coupling only through competition for a fixed n drawn from a frozen law
  SHOPPER        cross effects exist but come from an ordering average; there is no closed
                 form for Cov(x_j, x_k), it must be estimated by sampling permutations
"""
import sys, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel, log_f_ragged

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
m = RaggedModel(J=int(D["n_item"]), N=int(D["n_user"]), C=int(D["n_cat"]), K=32, Kz=12,
                nmax=120, R=23, S=int(D["n_store"]), Kp=8)
m.load_state_dict(torch.load('../../out/v3_run17.pt', map_location='cpu')); m.double().eval()

va = np.flatnonzero(D["trip_split"] == 1)[:8]
ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(va)
m.house, m.ctx = hh, ctx

# --- inclusion probabilities, exact by Corollary 2 -------------------------------------
b0 = m.b_flat(ix).detach().requires_grad_(True)
saved, m.ctx = m.ctx, None
_lam = m._parameters.pop('lam'); m.lam = _lam.data
def logZ_from_b(bvec):
    phi_i = m.phi[ix.item].detach()
    bt = bvec - 0.5 * (phi_i ** 2).sum(-1)
    z = torch.zeros(ix.B, 1, m.Kz)
    for _ in range(2):
        zz = z.detach().requires_grad_(True)
        with torch.enable_grad():
            lf = _lf(bt.detach(), zz)
        z = torch.autograd.grad(lf.sum(), zz)[0]
    return _lf(bt, z.detach())
def _lf(bt, z):
    from ragged import esp_bucketed, poly_tree, seg_max
    phi_i = m.phi[ix.item].detach()
    proj = (z[ix.item_trip] * phi_i.unsqueeze(1)).sum(-1)
    logw = (bt.unsqueeze(1) + proj).transpose(0, 1)
    M = seg_max(logw, ix.item_trip, ix.B)
    w = torch.exp(logw - M.index_select(-1, ix.item_trip))
    e = esp_bucketed(w, ix.row_of, ix.n_rows, m.R, ix.row_size, ix.item_pos)
    r = torch.arange(m.R + 1, dtype=w.dtype)
    a = torch.exp(-m.rho_c[ix.row_cat].detach().unsqueeze(-1) * r * (r - 1) / 2.0)
    G = a.unsqueeze(0) * e
    Gp = torch.zeros(1, ix.B * ix.Cpad, m.R + 1, dtype=w.dtype); Gp[:, :, 0] = 1.0
    Gp = Gp.index_copy(1, ix.flat_slot, G).view(1, ix.B, ix.Cpad, m.R + 1)
    A = poly_tree(Gp, m.nmax)
    n_ax = torch.arange(A.shape[-1], dtype=w.dtype)
    lg = (torch.log(A.clamp_min(1e-300)) - m.rho_0().detach()[: A.shape[-1]]
          + n_ax * M.unsqueeze(-1))
    return torch.logsumexp(lg[..., 1:], dim=-1).sum()

lz = logZ_from_b(b0)
pi = torch.autograd.grad(lz, b0, create_graph=True)[0]
print(f"inclusion probabilities: mean {float(pi.mean()):.5f}  "
      f"max {float(pi.max()):.4f}  sum(=E[n]) {float(pi.sum())/ix.B:.2f}")

# --- cross-response Cov(x_j, x_k) for the top items of trip 0 --------------------------
sel = (ix.item_trip == 0).nonzero().flatten()
top = sel[torch.argsort(-pi[sel].detach())[:8]]
rows = []
for t in top:
    gk = torch.autograd.grad(pi[t], b0, retain_graph=True)[0]
    rows.append(gk[top].detach())
Cov = torch.stack(rows)
np.set_printoptions(precision=5, suppress=True)
print("\nCov(x_j, x_k) for the 8 most likely products of one trip (x1e3):")
print((Cov * 1e3).numpy())
off = Cov[~torch.eye(len(top), dtype=torch.bool)]
print(f"\noff-diagonal: {int((off > 0).sum())} positive (complements), "
      f"{int((off < 0).sum())} negative (substitutes)")
print(f"  max {float(off.max())*1e3:+.4f}e-3   min {float(off.min())*1e3:+.4f}e-3")
m._parameters['lam'] = _lam; m.ctx = saved
