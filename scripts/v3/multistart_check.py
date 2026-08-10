"""Is the Laplace proposal covering the wrong mode?

The mode of F(z) = -||z||^2/2 + log f(z) is found by iterating z <- grad log f(z) from
z = 0.  If that map has several fixed points, the proposal covers one of them, the
importance weights are even WITHIN that basin, and ESS is high while log Z is biased
downward.  That is consistent with everything observed: ESS 0.97-1.00 and a likelihood
that still crosses zero.

Test: train to the point where the violation appears, then start the same iteration from
dispersed z and see where it lands.  Then compare the log Z the z=0 proposal gives against
a DEFENSIVE proposal -- a wide Gaussian with many draws, which cannot miss a mode the
narrow one finds but can find mass the narrow one misses.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math, numpy as np, torch
torch.set_default_dtype(torch.float64)
from data import build
from features import Features
from ragged import RaggedModel, log_f_ragged
from fit import Batcher

D = build(); J,N,C,S = (int(D[k]) for k in ("n_item","n_user","n_cat","n_store"))
F = Features(J,S,712); B = Batcher(D,F,60)
m = RaggedModel(J=J,N=N,C=C,K=32,Kz=12,nmax=60,R=4,seed=0,S=S,Kp=8); m.project(0.35)
opt = torch.optim.Adam(m.parameters(), lr=0.005)
tr = np.flatnonzero(D["trip_split"]==0); rng = np.random.default_rng(0)
gen = torch.Generator().manual_seed(0)
print("training to the point where the likelihood crosses zero...", flush=True)
for it in range(1, 151):
    sub = tr[rng.choice(len(tr), size=16, replace=False)]
    ix,ctx,hh,li,lt,lc = B.make(sub); m.house,m.ctx = hh,ctx
    ll = m.loglik(ix,li,lt,lc,n_draws=12,generator=gen)
    loss = -ll.mean(); opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),2.0); opt.step(); m.project(0.35)
    if it % 50 == 0:
        print(f"  it {it}  loss {float(loss):.2f}  max ll {float(ll.max()):+.3f}", flush=True)

sub = tr[rng.choice(len(tr), size=4, replace=False)]
ix,ctx,hh,li,lt,lc = B.make(sub); m.house,m.ctx = hh,ctx

def mode_from(z0, steps=40):
    z = z0.clone()
    for _ in range(steps):
        zz = z.detach().requires_grad_(True)
        with torch.enable_grad():
            lf = log_f_ragged(m, zz, ix, True).sum()
        z = torch.autograd.grad(lf, zz)[0]
    return z.detach()

print("\nMULTI-START on 6 real trips, 7 dispersed starts each", flush=True)
rg = torch.Generator().manual_seed(7)
starts = [torch.zeros(ix.B,1,m.Kz)] + [torch.randn(ix.B,1,m.Kz,generator=rg)*s
                                       for s in (0.5,1.0,2.0,3.0,5.0,8.0)]
modes = [mode_from(z) for z in starts]
with torch.no_grad():
    lfs = torch.stack([log_f_ragged(m, z, ix, True)[:,0] for z in modes])   # [7, B]
for b in range(ix.B):
    zs = torch.stack([mo[b,0] for mo in modes])
    d = torch.cdist(zs, zs).max()
    print(f"  trip {b}: log f at each mode {np.round(lfs[:,b].numpy(),3)}"
          f"   spread {float(lfs[:,b].max()-lfs[:,b].min()):.4f}"
          f"   max ||z_i - z_j|| {float(d):.4f}", flush=True)

print("\nlog Z from the z=0 proposal vs a DEFENSIVE wide proposal", flush=True)
g1 = torch.Generator().manual_seed(11)
with torch.no_grad():
    lz_std = m.log_Z(ix, n_draws=192, generator=g1, drop_empty=True)
    zh = mode_from(torch.zeros(ix.B,1,m.Kz))
    L2P = float(math.log(2*math.pi))
    for sd in (1.0, 2.0, 4.0):
        g2 = torch.Generator().manual_seed(11)
        noise = torch.randn(ix.B, 1024, m.Kz, generator=g2)
        zs = zh + noise*sd
        lq = -0.5*(noise**2).sum(-1) - m.Kz*math.log(sd) - 0.5*m.Kz*L2P
        lp = -0.5*m.Kz*L2P - 0.5*(zs**2).sum(-1) + log_f_ragged(m, zs, ix, True)
        lw = lp - lq
        lz = torch.logsumexp(lw,1) - math.log(1024)
        ess = (1.0/torch.softmax(lw,1).pow(2).sum(1)/1024)
        print(f"  proposal sd {sd:.1f}: log(Z-1) {np.round(lz.numpy(),3)}"
              f"  ESS {np.round(ess.numpy(),3)}", flush=True)
    print(f"  standard (n=256): log(Z-1) {np.round(lz_std.numpy(),3)}", flush=True)
    E = m.energy(li,lt,lc,ix.B)
    print(f"  E(S)             {np.round(E.numpy(),3)}", flush=True)
