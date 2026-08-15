"""Does a warm-started mode reach the same accuracy in fewer steps?

log_Z costs 28 ms per draw plus 755 ms of draw-independent work, and that fixed part is the
mode-finding.  If the mode can be carried across iterations, mode_steps drops and so does
the 61%.  Cold start from z=0 is the current behaviour; warm start uses the mode found on a
previous visit, imitated here by perturbing the converged mode.
"""
import sys, time, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
m = RaggedModel(J=int(D["n_item"]), N=int(D["n_user"]), C=int(D["n_cat"]), K=32, Kz=12,
                nmax=120, R=23, S=int(D["n_store"]), Kp=8)
m.load_state_dict(torch.load('../../out/v3_run11.pt', map_location='cpu')); m.double().eval()
tr = np.flatnonzero(D["trip_split"] == 0)
ix, ctx, lctx, hh, *_ = Bt.make(tr[:24]); m.house, m.ctx = hh, ctx

g = torch.Generator().manual_seed(0)
ref = m.log_Z(ix, n_draws=1024, mode_steps=12, generator=g, drop_empty=True).detach()
g = torch.Generator().manual_seed(0)
_, zstar = m.log_Z(ix, n_draws=8, mode_steps=12, generator=g, drop_empty=True,
                   return_mode=True)
zstar = zstar.detach()

def run(ms, z0, tag):
    errs, esss = [], []
    for s in range(4):
        g = torch.Generator().manual_seed(300 + s)
        lz, ess = m.log_Z(ix, n_draws=8, mode_steps=ms, generator=g, drop_empty=True,
                          return_ess=True, z_init=z0)
        errs.append((lz.detach() - ref).abs()); esss.append(ess)
    t0 = time.time()
    for _ in range(3):
        g = torch.Generator().manual_seed(1)
        m.log_Z(ix, n_draws=8, mode_steps=ms, generator=g, drop_empty=True, z_init=z0)
    dt = (time.time() - t0) / 3
    e = torch.stack(errs)
    print(f"  {tag:26s} mode_steps {ms}   |err| {float(e.mean()):.4f}  "
          f"max {float(e.max()):.4f}   ESS {float(torch.stack(esss).mean()):.3f}"
          f"   {dt*1000:7.1f} ms")

print("cold start (z = 0), the current behaviour:")
for ms in (3, 1, 0):
    run(ms, None, "cold")
# a warm start is last visit's mode, not this one's: perturb it
print("\nwarm start (previous visit's mode, perturbed to imitate parameter drift):")
for sd in (0.05, 0.15):
    zw = zstar + sd * torch.randn(zstar.shape, generator=torch.Generator().manual_seed(9))
    for ms in (1, 0):
        run(ms, zw, f"warm, drift {sd}")
