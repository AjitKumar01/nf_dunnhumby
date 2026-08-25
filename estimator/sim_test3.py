import math, time, sys
import torch
sys.path.insert(0, '.')
torch.set_default_dtype(torch.float64)
from gauss_est import log_Z_gauss, log_Z_cv
from sim_test import exact, qmc
from sim_test2 import case, En

print("Laplace-tilted importance sampling (exact) vs the plain closed form vs QMC.\n")
print(f"{'J':>3}{'Kz':>4}{'rho':>6}{'E[n]':>7}{'lam':>7}{'exact':>10}"
      f"{'closed':>10}{'CV-4':>10}{'ms':>6}{'CV-8':>10}{'ms':>6}{'QMC-8':>9}{'ms':>6}{'QMC-128':>10}{'ms':>6}")
for J, Kz, rho, bmean, curv in ((16,16,0.96,-1.2,0.03),(16,16,0.96,0.0,0.02),
                                (16,16,0.96,1.0,0.015),(18,16,0.96,0.5,0.015),
                                (20,16,0.96,0.5,0.015),(16,16,1.40,-1.2,0.03)):
    nmax = J
    b, phi, cat, rc, r0 = case(J, Kz, rho, nmax, bmean, curv=curv)
    ex = exact(b, phi, cat, rc, r0, nmax); en = En(b, phi, cat, rc, r0, nmax)
    cl, lam = log_Z_gauss(b, phi, cat, rc, r0, nmax, "mf")
    t0=time.time(); c4 = log_Z_cv(b, phi, cat, rc, r0, nmax, 4); t4=(time.time()-t0)*1e3
    t0=time.time(); c8 = log_Z_cv(b, phi, cat, rc, r0, nmax, 8); t8=(time.time()-t0)*1e3
    t0=time.time(); q8 = qmc(b, phi, cat, rc, r0, nmax, 8); tq=(time.time()-t0)*1e3
    t0=time.time(); q128=qmc(b, phi, cat, rc, r0, nmax,128); tq2=(time.time()-t0)*1e3
    print(f"{J:>3}{Kz:>4}{rho:6.2f}{en:7.2f}{lam:7.3f}{ex:10.4f}{cl-ex:+10.4f}"
          f"{c4-ex:+10.5f}{t4:6.1f}{c8-ex:+10.5f}{t8:6.1f}{q8-ex:+9.4f}{tq:6.1f}"
          f"{q128-ex:+10.5f}{tq2:6.1f}", flush=True)
