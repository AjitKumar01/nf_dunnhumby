"""Pseudo-likelihood: fit the conditionals, never touch Z.

Every proposal tried so far estimates the same integral and fails in the same place, because
d log Z / d phi_j is a second moment under the MODEL, and inside a log its noise becomes bias
in the gradient -- which is what drives phi to its cap.

P(x_j = 1 | x_-j) has no such problem.  For this energy the conditional log-odds is

    b_j + sum_{k in S, k != j} phi_j.phi_k - rho_c(j) * n_c(j)\\j - [rho_0(n+1) - rho_0(n)]

so the conditional normaliser is 1 + exp(.), exact.  No z-integral, no proposal, no draws.
Consistent for the joint, at some cost in statistical efficiency.

Judged on RECOVERY over seeds against exact enumeration -- the bar the proposals failed.
"""
import itertools, numpy as np, torch, math
torch.set_default_dtype(torch.float64)
J, K = 12, 12
mask = torch.zeros(2**J, J)
for i,bits in enumerate(itertools.product([0,1], repeat=J)):
    for j in range(J):
        if bits[j]: mask[i,j]=1.0
nonempty = mask.sum(1) > 0
m0 = mask[nonempty.nonzero().flatten()]

def E_of(b, PH):
    v = mask @ PH; sq = mask @ (PH**2).sum(1)
    return mask @ b + 0.5*((v*v).sum(1) - sq)

def lift_of(b, PH):
    p = torch.softmax(E_of(b,PH)[nonempty],0)
    pi = (m0*p.unsqueeze(1)).sum(0)
    return float((m0[:,0]*m0[:,1]*p).sum())/float(pi[0]*pi[1])

def pseudo_ll(X, b, PH):
    """mean_j log P(x_j | x_-j) over baskets.  X is [N, J] of 0/1."""
    S = X @ PH                                  # [N, K]  sum of phi over the basket
    # for product j: sum over k in S, k != j  ->  S - x_j * phi_j
    inner = (S.unsqueeze(1) - X.unsqueeze(2)*PH.unsqueeze(0))   # [N, J, K]
    odds = b.unsqueeze(0) + (inner * PH.unsqueeze(0)).sum(-1)   # [N, J]
    return -(torch.nn.functional.softplus(odds) - X*odds).sum(1).mean()

print("recovery over 5 seeds (mean +/- sd), 40,000 baskets")
print(f"{'true':>6}{'exact ML':>11}{'pseudo-LL':>16}   {'true lift':>10}{'pseudo lift':>16}")
for t in (2.0, 3.0, 4.0):
    v=math.sqrt(t)
    PH_t=torch.zeros(J,K); PH_t[0,0]=v; PH_t[1,0]=v
    b_t=torch.full((J,),-2.0)
    pt=torch.softmax(E_of(b_t,PH_t)[nonempty],0)
    lift_t=lift_of(b_t,PH_t)
    rng=np.random.default_rng(0)
    idx=nonempty.nonzero().flatten()
    dr=rng.choice(len(idx),size=40000,p=pt.numpy())
    X = mask[idx[dr]]
    # exact ML control
    cnt=torch.zeros(len(idx))
    for d in dr: cnt[d]+=1
    bh=torch.full((J,),-1.0,requires_grad=True)
    PH=(torch.randn(J,K,generator=torch.Generator().manual_seed(0))*0.1).requires_grad_(True)
    opt=torch.optim.Adam([bh,PH],lr=0.05)
    for _ in range(700):
        Ea=E_of(bh,PH); lz=torch.logsumexp(Ea[nonempty],0)
        ll=((Ea[nonempty]-lz)*cnt).sum()/cnt.sum()
        opt.zero_grad(); (-ll).backward(); opt.step()
    ex_phi=float(PH[0]@PH[1])
    pp=[];lf=[]
    for SEED in range(5):
        bh=torch.full((J,),-1.0,requires_grad=True)
        PH=(torch.randn(J,K,generator=torch.Generator().manual_seed(SEED))*0.1).requires_grad_(True)
        opt=torch.optim.Adam([bh,PH],lr=0.05)
        for _ in range(700):
            loss=-pseudo_ll(X,bh,PH)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pp.append(float(PH[0]@PH[1])); lf.append(lift_of(bh.detach(),PH.detach()))
    print(f"{t:6.2f}{ex_phi:11.3f}{np.mean(pp):11.3f}±{np.std(pp,ddof=1):.2f}   "
          f"{lift_t:10.3f}{np.mean(lf):11.3f}±{np.std(lf,ddof=1):.2f}", flush=True)
