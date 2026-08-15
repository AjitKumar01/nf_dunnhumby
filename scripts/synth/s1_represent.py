"""Does the pairwise term actually produce co-occurrence?  Exact, by enumeration.

Nothing here is estimated.  A small catalogue is enumerated over all 2^J subsets, so P(S),
the marginals and every pair lift are exact.  The question is the one the whole architecture
rests on: if phi_j.phi_k = x, does the pair's lift over independence equal exp(x)?

If it does, representation is sound and the failure is downstream (fitting, sampling, or the
regime).  If it does not, the energy cannot express what we have been asking of it, and no
amount of fitting will help.
"""
import itertools, numpy as np, torch, math
torch.set_default_dtype(torch.float64)

def enumerate_model(b, PHI, rho_c, cat, rho0):
    """Exact P(S) over all subsets.  b [J], PHI [J,K], rho_c [C], cat [J], rho0 [J+1]."""
    J = len(b)
    subs, en = [], []
    for bits in itertools.product([0, 1], repeat=J):
        S = [j for j in range(J) if bits[j]]
        E = sum(float(b[j]) for j in S)
        for x in range(len(S)):
            for y in range(x + 1, len(S)):
                E += float(PHI[S[x]] @ PHI[S[y]])
        for c in range(len(rho_c)):
            nc = sum(1 for j in S if cat[j] == c)
            E -= float(rho_c[c]) * nc * (nc - 1) / 2.0
        E -= float(rho0[len(S)])
        subs.append(S); en.append(E)
    en = np.array(en); p = np.exp(en - en.max()); p /= p.sum()
    return subs, p

def marginals_and_lift(subs, p, J, pairs):
    pi = np.zeros(J)
    for S, w in zip(subs, p):
        for j in S: pi[j] += w
    out = {}
    for (a, c) in pairs:
        joint = sum(w for S, w in zip(subs, p) if a in S and c in S)
        out[(a, c)] = joint / max(pi[a] * pi[c], 1e-300)
    return pi, out

J, K, C = 12, 4, 3
cat = np.array([0,0,0,0, 1,1,1,1, 2,2,2,2])
g = torch.Generator().manual_seed(0)
b = torch.full((J,), -2.0)
rho_c = torch.zeros(C)
rho0 = torch.zeros(J + 1)

print("planted phi.phi   ->   exact pair lift        (expected exp(phi.phi))")
for target in (0.0, 0.5, 0.91, 2.0, 3.0):
    PHI = torch.zeros(J, K)
    # products 0 and 1 are the planted complementary pair, everything else neutral
    v = math.sqrt(abs(target))
    PHI[0, 0] = v; PHI[1, 0] = math.copysign(v, target)
    subs, p = enumerate_model(b, PHI, rho_c, cat, rho0)
    pi, lift = marginals_and_lift(subs, p, J, [(0, 1), (0, 2)])
    print(f"   {float(PHI[0]@PHI[1]):+6.3f}      pair(0,1) {lift[(0,1)]:8.3f}   "
          f"control(0,2) {lift[(0,2)]:6.3f}   expected {math.exp(target):8.3f}   "
          f"pi_0 {pi[0]:.4f}")
