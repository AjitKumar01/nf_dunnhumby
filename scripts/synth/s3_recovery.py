"""Given data generated from a KNOWN phi, does fitting recover it?

Everything here is exact: 2^12 subsets enumerated, so P(S) is closed-form, the sampler is a
multinomial over the full support, and the likelihood needs no importance sampling.  Any
failure is therefore the objective or the optimiser, not the estimator.

Planted structure: products (0,1) complementary, (4,5) substitutable, the rest neutral.
"""
import itertools, numpy as np, torch, math
torch.set_default_dtype(torch.float64)
exec(open('s1_represent.py').read().split('J, K, C = ')[0].split('"""',2)[2])

J, K, C = 12, 4, 3
cat = np.array([0,0,0,0, 1,1,1,1, 2,2,2,2])
rho_c_true = torch.zeros(C); rho0_true = torch.zeros(J+1)
b_true = torch.full((J,), -2.0)
PHI_true = torch.zeros(J, K)
PHI_true[0,0] =  1.0; PHI_true[1,0] =  1.0      # phi.phi = +1.0  complements
PHI_true[4,1] =  1.0; PHI_true[5,1] = -1.0      # phi.phi = -1.0  substitutes

subs, p = enumerate_model(b_true, PHI_true, rho_c_true, cat, rho0_true)
pi_t, lift_t = marginals_and_lift(subs, p, J, [(0,1),(4,5),(0,2)])
print(f"TRUTH  lift(0,1)={lift_t[(0,1)]:.3f}  lift(4,5)={lift_t[(4,5)]:.3f}  "
      f"lift(0,2)={lift_t[(0,2)]:.3f}")

# --- generate baskets by exact multinomial over the support -------------------------
rng = np.random.default_rng(0)
NB = 40000
idx = rng.choice(len(subs), size=NB, p=p)
baskets = [subs[i] for i in idx]
nz = [b for b in baskets if b]
print(f"generated {NB:,} baskets, {len(nz):,} non-empty, mean size {np.mean([len(b) for b in nz]):.2f}")

# --- fit b and PHI by exact maximum likelihood (no importance sampling) -------------
bh = torch.full((J,), -1.0, requires_grad=True)
PH = (torch.randn(J, K, generator=torch.Generator().manual_seed(1))*0.1).requires_grad_(True)
opt = torch.optim.Adam([bh, PH], lr=0.05)
mask = torch.zeros(len(subs), J)
for i,S in enumerate(subs):
    for j in S: mask[i,j] = 1.0
cnt = torch.zeros(len(subs))
for i in idx: cnt[i] += 1
nonempty = mask.sum(1) > 0
for step in range(1500):
    lin = mask @ bh
    v = mask @ PH
    sq = mask @ (PH**2).sum(1)
    pair = 0.5*((v*v).sum(1) - sq)
    E = lin + pair
    logZ = torch.logsumexp(E[nonempty], 0)
    ll = ((E - logZ) * cnt)[nonempty].sum() / cnt[nonempty].sum()
    opt.zero_grad(); (-ll).backward(); opt.step()
    if step % 500 == 499:
        print(f"  step {step+1:5d}  mean log P {float(ll):8.4f}")
with torch.no_grad():
    print(f"\nRECOVERED  phi0.phi1 = {float(PH[0]@PH[1]):+.3f} (true +1.000)   "
          f"phi4.phi5 = {float(PH[4]@PH[5]):+.3f} (true -1.000)   "
          f"phi0.phi2 = {float(PH[0]@PH[2]):+.3f} (true 0)")
    s2,p2 = enumerate_model(bh.detach(), PH.detach(), rho_c_true, cat, rho0_true)
    pi_f, lift_f = marginals_and_lift(s2, p2, J, [(0,1),(4,5),(0,2)])
    print(f"FITTED     lift(0,1)={lift_f[(0,1)]:.3f}  lift(4,5)={lift_f[(4,5)]:.3f}  "
          f"lift(0,2)={lift_f[(0,2)]:.3f}")
