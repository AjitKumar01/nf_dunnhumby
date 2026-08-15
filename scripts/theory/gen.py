"""Does the sampler reproduce the distribution it claims to sample from?

Two independent checks, against quantities computed WITHOUT sampling:
  E[n]        from size_dist, which sums the size law analytically
  P(j in S)   from Corollary 2, d log(Z-1)/d b_j by autograd
If the sampler is right, basket sizes and per-product frequencies over many draws must
converge to these.  Nothing here is fitted -- it is the same checkpoint both ways.
"""
import sys, time, collections, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
m = RaggedModel(J=int(D["n_item"]), N=int(D["n_user"]), C=int(D["n_cat"]), K=32, Kz=12,
                nmax=120, R=23, S=int(D["n_store"]), Kp=8)
m.load_state_dict(torch.load('../../out/v3_run17.pt', map_location='cpu')); m.double().eval()
va = np.flatnonzero(D["trip_split"] == 1)[:4]
ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(va)
m.house, m.ctx = hh, ctx

g = torch.Generator().manual_seed(0)
en, _ = m.size_moments(ix, n_draws=256, generator=g)
print(f"analytic  E[n] per trip: {[round(float(x),2) for x in en]}")

REP = 400
t0 = time.time()
sizes = [[] for _ in range(ix.B)]
counts = [collections.Counter() for _ in range(ix.B)]
g = torch.Generator().manual_seed(1)
for r in range(REP):
    for b, bask in enumerate(m.sample(ix, n_draws=32, generator=g)):
        sizes[b].append(len(bask))
        counts[b].update(bask)
dt = time.time() - t0
print(f"sampled   E[n] per trip: {[round(float(np.mean(s)),2) for s in sizes]}")
print(f"          ({REP} baskets/trip in {dt:.1f}s = {dt/REP/ix.B*1000:.1f} ms per basket)")

# inclusion probabilities, exact, for trip 0's most likely products
b0 = m.b_flat(ix).detach().requires_grad_(True)
sv, m.ctx = m.ctx, None
_lam = m._parameters.pop('lam'); m.lam = _lam.data
from ragged import log_f_ragged
z = torch.zeros(ix.B, 1, m.Kz)
for _ in range(2):
    zz = z.detach().requires_grad_(True)
    with torch.enable_grad():
        lf = log_f_ragged(m, zz, ix, True).sum()
    z = torch.autograd.grad(lf, zz)[0]
m._parameters['lam'] = _lam; m.ctx = sv
g = torch.Generator().manual_seed(2)
lzs = []
for _ in range(8):
    lzs.append(m.log_Z(ix, n_draws=128, generator=g, drop_empty=True))
del lzs
b0 = m.b_flat(ix).detach().requires_grad_(True)
sv2, m.ctx = m.ctx, None
_lam = m._parameters.pop('lam'); m.lam = _lam.data
def lz_of(bv):
    from ragged import esp_bucketed, poly_tree, seg_max
    ph = m.phi[ix.item].detach()
    bt = bv - 0.5 * (ph ** 2).sum(-1)
    pr = (z.detach()[ix.item_trip] * ph.unsqueeze(1)).sum(-1)
    lw = (bt.unsqueeze(1) + pr).transpose(0, 1)
    M = seg_max(lw, ix.item_trip, ix.B)
    w = torch.exp(lw - M.index_select(-1, ix.item_trip))
    e = esp_bucketed(w, ix.row_of, ix.n_rows, m.R, ix.row_size, ix.item_pos)
    rr = torch.arange(m.R + 1, dtype=w.dtype)
    a = torch.exp(-m.rho_c[ix.row_cat].detach().unsqueeze(-1) * rr * (rr - 1) / 2.0)
    Gp = torch.zeros(1, ix.B * ix.Cpad, m.R + 1, dtype=w.dtype); Gp[:, :, 0] = 1.0
    Gp = Gp.index_copy(1, ix.flat_slot, a.unsqueeze(0) * e).view(1, ix.B, ix.Cpad, m.R + 1)
    A = poly_tree(Gp, m.nmax)
    nx = torch.arange(A.shape[-1], dtype=w.dtype)
    lg = torch.log(A.clamp_min(1e-300)) - m.rho_0().detach()[: A.shape[-1]] + nx * M.unsqueeze(-1)
    return torch.logsumexp(lg[..., 1:], dim=-1).sum()
pi = torch.autograd.grad(lz_of(b0), b0)[0]
m._parameters['lam'] = _lam; m.ctx = sv2

sel = (ix.item_trip == 0).nonzero().flatten()
top = sel[torch.argsort(-pi[sel])[:10]]
print(f"\ntrip 0, ten most likely products:")
print(f"{'item':>7}{'exact P(j)':>12}{'sampled':>10}")
for t in top:
    j = int(ix.item[t])
    print(f"{j:7d}{float(pi[t]):12.4f}{counts[0][j]/REP:10.4f}")
