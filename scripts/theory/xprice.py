"""The counterfactual both models must answer, priced in compute.

Question: cut the shelf price of one product by 10%.  What happens to the purchase
probability of every OTHER product in the basket?  That is the quantity a coupon-targeting
or markdown policy acts on, and the two models reach it by different routes.

  ours     Corollary 2 gives P(k in S) = d log(Z-1)/d b_k.  One more derivative gives
           d P(k in S) / d b_j = Cov(x_j, x_k), and the chain rule through the price term
           (d b_j / d log p_j = -(gamma_h . beta_j)) turns it into the cross-price response.
           Exact, differentiable, one backward pass.

  SHOPPER  has no closed form for P(k in S) -- the set probability is n! E_pi[P(pi)] over
           orderings.  The response has to be simulated: generate baskets at the baseline
           price, generate again at the cut price, difference the frequencies.  That is
           unbiased but noisy, and the noise is the point: a cross-effect of 1e-3 needs on
           the order of 1e6 baskets before the signal clears its own standard error.

Reported below: the exact answer and its cost, then SHOPPER's estimate at increasing sample
counts with the standard error attached, so the compute needed to resolve the same number
is visible rather than asserted.
"""
import sys, time, math, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel, log_f_ragged, esp_bucketed, poly_tree, seg_max
import baselines2 as B2
from baselines2 import Shopper, Batches

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712)
J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
Bt = Batcher(D, F, 120)
va = np.flatnonzero(D["trip_split"] == 1)[:1]
ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(va)

m = RaggedModel(J=J, N=N, C=C, K=32, Kz=12, nmax=120, R=23, S=S, Kp=8)
m.load_state_dict(torch.load('../../out/v3_run17.pt', map_location='cpu'))
m.double().eval(); m.house, m.ctx = hh, ctx

# ---- ours: exact, one backward pass ---------------------------------------------------
t0 = time.time()
z = torch.zeros(ix.B, 1, m.Kz)
for _ in range(2):
    zz = z.detach().requires_grad_(True)
    with torch.enable_grad():
        z = torch.autograd.grad(log_f_ragged(m, zz, ix, True).sum(), zz)[0]
z = z.detach()
b0 = m.b_flat(ix).detach().requires_grad_(True)
sv = m.ctx; m.ctx = None
_lam = m._parameters.pop('lam'); m.lam = _lam.data
def lz_of(bv):
    ph = m.phi[ix.item].detach()
    bt = bv - 0.5 * (ph ** 2).sum(-1)
    pr = (z[ix.item_trip] * ph.unsqueeze(1)).sum(-1)
    lw = (bt.unsqueeze(1) + pr).transpose(0, 1)
    M = seg_max(lw, ix.item_trip, ix.B)
    w = torch.exp(lw - M.index_select(-1, ix.item_trip))
    e = esp_bucketed(w, ix.row_of, ix.n_rows, m.R, ix.row_size, ix.item_pos)
    rr = torch.arange(m.R + 1, dtype=w.dtype)
    a = torch.exp(-m.rho_c[ix.row_cat].detach().unsqueeze(-1) * rr * (rr - 1) / 2.0)
    Gp = torch.zeros(1, ix.B * ix.Cpad, m.R + 1, dtype=w.dtype); Gp[:, :, 0] = 1.0
    Gp = Gp.index_copy(1, ix.flat_slot, a.unsqueeze(0) * e).view(1, ix.B, ix.Cpad, m.R+1)
    A = poly_tree(Gp, m.nmax)
    nx = torch.arange(A.shape[-1], dtype=w.dtype)
    lg = torch.log(A.clamp_min(1e-300)) - m.rho_0().detach()[:A.shape[-1]] + nx*M.unsqueeze(-1)
    return torch.logsumexp(lg[..., 1:], dim=-1).sum()
pi = torch.autograd.grad(lz_of(b0), b0, create_graph=True)[0]
sel = (ix.item_trip == 0).nonzero().flatten()
order = sel[torch.argsort(-pi[sel].detach())]
tgt = order[0]                                     # cut the price of the likeliest product
others = order[1:9]
cov = torch.autograd.grad(pi[tgt], b0, retain_graph=True)[0][others].detach()
m._parameters['lam'] = _lam; m.ctx = sv
# d b_j / d log p_j
gam = m.gamma[hh[0]].detach(); bet = m.beta[ix.item[tgt]].detach()
dlogp = math.log(0.90)
db = -float((gam * bet).sum()) * dlogp
dP_exact = cov * db
t_exact = time.time() - t0
print(f"ours: exact cross-response to a 10% cut on product {int(ix.item[tgt])} "
      f"in {t_exact*1000:.0f} ms")
print(f"  db_j = {db:+.5f};  P(target) = {float(pi[tgt]):.4f}")
for k, o in enumerate(others):
    print(f"  product {int(ix.item[o]):5d}  P {float(pi[o]):.4f}   dP {float(dP_exact[k]):+.6f}")

# ---- SHOPPER: the same numbers, by simulation -----------------------------------------
sm = Shopper(J, N, S, K=32, Kp=8)
sm.load_state_dict(torch.load('../../out/v3_shopper.pt', map_location='cpu'))
sm.double().eval()
Bs = Batches(D, F)
d = Bs.make(va)

@torch.no_grad()
def shopper_freq(nsim, shift_item=None, shift=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = dict(d["ctx"])
    if shift_item is not None:
        c = dict(c); c["dlp"] = c["dlp"].clone()
        c["dlp"][d["item"] == shift_item] += shift
    ps = sm.idx(d["item"], d["st"], d["house"], c)
    it = d["item"]
    rho, al = sm.rho[it], sm.alpha_i[it]
    hit = torch.zeros(len(it))
    for _ in range(nsim):
        alive = torch.ones(len(ps), dtype=torch.bool)
        run = torch.zeros(sm.Ki, dtype=ps.dtype)
        i = 0
        while True:
            u = ps + (rho @ run if i else torch.zeros_like(ps))
            u = torch.cat([u.masked_fill(~alive, -1e30), sm.checkout])
            j = int(torch.multinomial(torch.softmax(u, 0), 1, generator=g))
            if j == len(ps) or i > 118:
                break
            hit[j] += 1; run = (run * i + al[j]) / (i + 1); alive[j] = False; i += 1
    return hit / nsim

pos = {int(v): k for k, v in enumerate(d["item"].tolist())}
tgt_g, oth_g = int(ix.item[tgt]), [int(ix.item[o]) for o in others]
print("\nSHOPPER: the same cross-response, by forward simulation")
print(f"{'baskets':>9}{'time':>8}   dP for the first three products (+/- 1 s.e.)")
for nsim in (200, 1000, 5000):
    t0 = time.time()
    f0 = shopper_freq(nsim, seed=1)
    f1 = shopper_freq(nsim, shift_item=tgt_g, shift=dlogp, seed=2)
    dt = time.time() - t0
    row = []
    for o in oth_g[:3]:
        k = pos[o]
        dP = float(f1[k] - f0[k])
        p = float(f0[k])
        se = math.sqrt(2 * max(p * (1 - p), 1e-9) / nsim)
        row.append(f"{dP:+.5f}+/-{se:.5f}")
    print(f"{nsim:9d}{dt:7.1f}s   " + "  ".join(row))
print(f"\nours resolved every one of these exactly in {t_exact*1000:.0f} ms.")
