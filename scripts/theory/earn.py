"""Is the model paying for structure it does not use?

Three costs that are structural rather than incidental:

  rho_c  forces a TWO-LEVEL normaliser -- an ESP within each category, then a convolution
         across 183 categories.  With rho_c = 0 the whole thing collapses to a single ESP
         over all products, which is exactly the multinomial's shape and its cost.

  Kz     sets the width of the projection [slots x draws x Kz], evaluated four times per
         log_Z.  If the fitted interaction is effectively low rank, most of that is wasted.

  draws  multiply every draw-dependent stage.  ESS near 1.0 says the proposal is close to
         exact, and an exact proposal needs few draws.
"""
import sys, time, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
m = RaggedModel(J=int(D["n_item"]), N=int(D["n_user"]), C=int(D["n_cat"]), K=32, Kz=12,
                nmax=120, R=23, S=int(D["n_store"]), Kp=8)
m.load_state_dict(torch.load('../../out/v3_run11.pt', map_location='cpu')); m.double().eval()

rc = m.rho_c.detach()
print(f"1. rho_c over {len(rc)} categories:")
print(f"   mean {float(rc.mean()):+.4f}  sd {float(rc.std()):.4f}  "
      f"min {float(rc.min()):+.4f}  max {float(rc.max()):+.4f}")
print(f"   |rho_c| < 0.01: {int((rc.abs() < 0.01).sum())}/{len(rc)}   "
      f"< 0.05: {int((rc.abs() < 0.05).sum())}/{len(rc)}")

phi = m.phi.detach()
sv = torch.linalg.svdvals(phi)
cum = (sv ** 2).cumsum(0) / (sv ** 2).sum()
print(f"\n2. phi singular values (Kz={m.Kz}), variance explained:")
print("   " + "  ".join(f"k={k+1}:{float(cum[k]):.3f}" for k in range(m.Kz)))

tr = np.flatnonzero(D["trip_split"] == 0)
ix, ctx, lctx, hh, *_ = Bt.make(tr[:24]); m.house, m.ctx = hh, ctx
g = torch.Generator().manual_seed(0)
ref = m.log_Z(ix, n_draws=1024, generator=g, drop_empty=True).detach()
print(f"\n3. log Z error against a 1024-draw reference, and cost:")
for nd in (4, 8, 16, 32):
    errs = []
    for s in range(5):
        g = torch.Generator().manual_seed(200 + s)
        errs.append((m.log_Z(ix, n_draws=nd, generator=g,
                             drop_empty=True).detach() - ref).abs())
    e = torch.stack(errs)
    t0 = time.time()
    for _ in range(3):
        g = torch.Generator().manual_seed(1)
        m.log_Z(ix, n_draws=nd, generator=g, drop_empty=True)
    dt = (time.time() - t0) / 3
    print(f"   draws {nd:3d}   mean |err| {float(e.mean()):.4f}   "
          f"max {float(e.max()):.4f}   {dt*1000:7.1f} ms")
