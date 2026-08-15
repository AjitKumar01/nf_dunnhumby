"""Is the lift ceiling fundamental, or an artifact of holding b fixed?

In s1 the marginals inflated as phi strengthened, so the lift ratio turned over.  A real fit
does not hold b fixed: it moves b to keep the marginals where the data puts them.  Here b is
solved for each phi so that pi_0 stays at its baseline, isolating the interaction's effect on
DEPENDENCE from its effect on prevalence.
"""
import itertools, numpy as np, torch, math
torch.set_default_dtype(torch.float64)
exec(open('s1_represent.py').read().split('J, K, C = ')[0].split('"""',2)[2])

J, K, C = 12, 4, 3
cat = np.array([0,0,0,0, 1,1,1,1, 2,2,2,2])
rho_c = torch.zeros(C); rho0 = torch.zeros(J + 1)
BASE = -2.0
subs0, p0 = enumerate_model(torch.full((J,), BASE), torch.zeros(J, K), rho_c, cat, rho0)
pi0, _ = marginals_and_lift(subs0, p0, J, [(0, 1)])
target_pi = pi0[0]
print(f"baseline pi = {target_pi:.4f}; b solved each time to hold it there\n")
print("phi.phi   b(0,1)    pi_0     pair lift   joint P(0,1)")
for t in (0.0, 0.5, 0.91, 2.0, 3.0, 4.0, 6.0):
    v = math.sqrt(t)
    PHI = torch.zeros(J, K); PHI[0,0]=v; PHI[1,0]=v
    lo, hi = -12.0, 4.0
    for _ in range(60):                                   # bisect b on products 0 and 1
        mid = 0.5*(lo+hi)
        b = torch.full((J,), BASE); b[0]=mid; b[1]=mid
        subs, p = enumerate_model(b, PHI, rho_c, cat, rho0)
        pi, lift = marginals_and_lift(subs, p, J, [(0,1)])
        if pi[0] > target_pi: hi = mid
        else: lo = mid
    joint = sum(w for S,w in zip(subs,p) if 0 in S and 1 in S)
    print(f"{t:6.2f}  {mid:+7.3f}  {pi[0]:.4f}   {lift[(0,1)]:9.3f}   {joint:.5f}")
