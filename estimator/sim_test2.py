"""The Gaussian step approximates the law of v_S = sum_{j in S} phi_j.  That is a CLT over
the number of items IN THE BASKET, not over the catalogue -- so E[n] is the parameter that
decides accuracy, and it must be checked at realistic basket sizes (grocery ~8).

Also sweeps J up to 20 (1M subsets, still enumerable) to confirm catalogue size is neutral.
"""
import math, time, itertools, sys
import numpy as np, torch
sys.path.insert(0, '.')
torch.set_default_dtype(torch.float64)
from gauss_est import log_f, log_Z_gauss
from sim_test import exact, qmc

def case(J, Kz, rho, nmax, bmean, seed=7, ncat=4, curv=0.03):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(J, Kz, generator=g, dtype=torch.float64)
    phi = phi / phi.norm(dim=1, keepdim=True) * rho
    b = torch.randn(J, generator=g, dtype=torch.float64) * 0.4 + bmean
    cat = torch.arange(J) % ncat
    return (b, phi, cat, torch.full((ncat,), 0.08, dtype=torch.float64),
            curv * torch.arange(nmax + 1, dtype=torch.float64) ** 2)

def En(b, phi, cat, rho_c, rho0, nmax):
    s = torch.zeros(1, requires_grad=True)
    v = log_f(torch.zeros(phi.shape[1]), b + s, phi, cat, rho_c, rho0, nmax)
    return float(torch.autograd.grad(v, s)[0])

print("Accuracy vs BASKET SIZE (the CLT parameter) and catalogue size.\n")
print(f"{'J':>3}{'Kz':>4}{'rho':>6}{'E[n]':>7}{'lam_max':>9}{'exact':>10}"
      f"{'gauss-mf':>11}{'ms':>7}{'QMC-8':>9}{'ms':>7}")
for J, Kz, rho, bmean, curv in ((16, 16, 0.96, -2.5, 0.05),
                                (16, 16, 0.96, -1.2, 0.03),
                                (16, 16, 0.96,  0.0, 0.02),
                                (16, 16, 0.96, +1.0, 0.015),
                                (18, 16, 0.96, +0.5, 0.015),
                                (20, 16, 0.96, +0.5, 0.015),
                                (20, 32, 0.60, +0.5, 0.015)):
    nmax = J
    b, phi, cat, rc, r0 = case(J, Kz, rho, nmax, bmean, curv=curv)
    ex = exact(b, phi, cat, rc, r0, nmax); en = En(b, phi, cat, rc, r0, nmax)
    t0 = time.time(); gm, lam = log_Z_gauss(b, phi, cat, rc, r0, nmax, "mf"); tg = (time.time()-t0)*1e3
    t0 = time.time(); q8 = qmc(b, phi, cat, rc, r0, nmax, 8); tq = (time.time()-t0)*1e3
    print(f"{J:>3}{Kz:>4}{rho:6.2f}{en:7.2f}{lam:9.3f}{ex:10.4f}"
          f"{gm-ex:+11.5f}{tg:7.1f}{q8-ex:+9.4f}{tq:7.1f}", flush=True)
