"""Decisive comparison at the REAL operating point.

run155 logs lam_max = 0.177.  Every earlier row sat at 0.39-0.77, which is where the
quadratic expansion is stressed and where CV showed only a marginal edge.  Here: sweep
lambda around the real value, and use MULTIPLE SEEDS so bias is separated from noise --
a single draw cannot tell them apart, and I reported single draws above.
"""
import math, time, sys
import numpy as np, torch
sys.path.insert(0, '.')
torch.set_default_dtype(torch.float64)
import gauss_est as G
from sim_test import exact, qmc
from sim_test2 import case, En

print(f"{'lam':>6}{'E[n]':>6}{'exact':>9}"
      f"{'CV-8 bias':>11}{'sd':>8}{'ms':>6}"
      f"{'QMC-8 bias':>12}{'sd':>8}{'ms':>6}"
      f"{'QMC-128 bias':>14}{'sd':>8}{'ms':>7}")
for rho, bmean, curv in ((0.45, 0.5, 0.015), (0.60, 0.5, 0.015),
                         (0.75, 0.0, 0.02), (0.96, -1.2, 0.03)):
    J, Kz, nmax = 18, 16, 18
    b, phi, cat, rc, r0 = case(J, Kz, rho, nmax, bmean, curv=curv)
    ex = exact(b, phi, cat, rc, r0, nmax); en = En(b, phi, cat, rc, r0, nmax)
    _, lam = G.log_Z_gauss(b, phi, cat, rc, r0, nmax, "mf")
    res = {}
    for name, fn, n in (("cv8", lambda s: G.log_Z_cv(b, phi, cat, rc, r0, nmax, 8, seed=s), 8),
                        ("q8",  lambda s: qmc(b, phi, cat, rc, r0, nmax, 8, seed=s), 8),
                        ("q128",lambda s: qmc(b, phi, cat, rc, r0, nmax, 128, seed=s), 128)):
        t0 = time.time(); vals = [fn(1000 * s) for s in range(5)]; ms = (time.time()-t0)/5*1e3
        res[name] = (float(np.mean(vals) - ex), float(np.std(vals)), ms)
    print(f"{lam:6.3f}{en:6.2f}{ex:9.3f}"
          f"{res['cv8'][0]:+11.4f}{res['cv8'][1]:8.4f}{res['cv8'][2]:6.1f}"
          f"{res['q8'][0]:+12.4f}{res['q8'][1]:8.4f}{res['q8'][2]:6.1f}"
          f"{res['q128'][0]:+14.4f}{res['q128'][1]:8.4f}{res['q128'][2]:7.1f}", flush=True)
