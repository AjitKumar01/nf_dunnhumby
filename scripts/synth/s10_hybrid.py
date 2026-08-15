"""Quadrature along v_top, ANISOTROPIC draws in the complement.

Two estimators, two failure modes.  Quadrature has the lower bias (-0.011 vs +0.002 at
phi.phi = 2, and 23x better at 4) but the higher variance (+/-0.06 vs +/-0.017); in a fit
that variance let phi wander to its cap even though the estimate was unbiased.  The
anisotropic proposal has the lower variance and recovers correctly at phi.phi = 2, but its
bias grows past 3.

The two are orthogonal in the literal sense: quadrature handles the sharp direction, the
proposal handles the flat complement.  Combining them should give the bias of the first and
the variance of the second.  Judged on RECOVERY, not on log Z -- the mistake that cost the
last two candidates.
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

def mode_and_vtop(b, PH, steps=6):
    z = torch.zeros(1, K)
    for _ in range(steps):
        zz = z.detach().requires_grad_(True)
        w = torch.exp(b + zz @ PH.T - 0.5*(PH**2).sum(1))
        z = torch.autograd.grad(torch.log1p(w).sum(), zz)[0]
    zh = z.detach()
    w = torch.exp(b + zh @ PH.T - 0.5*(PH**2).sum(1))[0]
    pi = (w/(1+w)).clamp(1e-12,1-1e-12)
    L = (PH.detach()*(pi*(1-pi)).unsqueeze(1)).T @ PH.detach()
    ev,V = torch.linalg.eigh(L)
    return zh, V[:,-1]

def log_f(b, PH, zs):
    w = torch.exp(b.unsqueeze(0) + zs @ PH.T - 0.5*(PH**2).sum(1).unsqueeze(0))
    return torch.expm1(torch.log1p(w).sum(1)).clamp_min(1e-300).log()

def logZ_aniso(b, PH, nd, gen, s=2.0):
    zh, v = mode_and_vtop(b,PH)
    e0 = torch.randn(nd,K,generator=gen)
    eps = e0 + (s-1.0)*(e0@v).unsqueeze(1)*v.unsqueeze(0)
    r=eps@v; perp2=(eps**2).sum(1)-r**2
    lq = -0.5*(perp2+(r/s)**2) - math.log(s)
    zs = zh+eps
    return torch.logsumexp((-0.5*(zs**2).sum(1)+log_f(b,PH,zs))-lq,0) - math.log(nd)

def logZ_hybrid(b, PH, n_r, n_p, gen, s=1.6):
    """Gauss-Hermite on v_top; in the complement, draws widened by s and reweighted."""
    zh, v = mode_and_vtop(b,PH)
    xr,wr = np.polynomial.hermite_e.hermegauss(n_r)
    xr=torch.tensor(xr); wr=torch.tensor(wr)/math.sqrt(2*math.pi)
    e0 = torch.randn(n_p,K,generator=gen)
    e0 = e0 - (e0@v).unsqueeze(1)*v.unsqueeze(0)          # in the complement
    eps = e0 * s                                          # widened there
    lq_perp = -0.5*(eps**2).sum(1)/s**2 - (K-1)*math.log(s)
    lp_perp = -0.5*(eps**2).sum(1)
    corr = lp_perp - lq_perp                              # importance weight, complement
    parts=[torch.log(wr[i].clamp_min(1e-300)) + corr
           + log_f(b,PH,xr[i]*v.unsqueeze(0)+eps) for i in range(n_r)]
    return torch.logsumexp(torch.stack(parts).reshape(-1),0) - math.log(n_p)

print("recovery over 5 seeds (mean +/- sd) -- single fits are not evidence")
for t in (2.0, 3.0, 4.0):
    v=math.sqrt(t)
    PH_t=torch.zeros(J,K); PH_t[0,0]=v; PH_t[1,0]=v
    b_t=torch.full((J,),-2.0)
    pt=torch.softmax(E_of(b_t,PH_t)[nonempty],0)
    pi_t=(m0*pt.unsqueeze(1)).sum(0)
    lift_t=float((m0[:,0]*m0[:,1]*pt).sum())/float(pi_t[0]*pi_t[1])
    rng=np.random.default_rng(0)
    dr=rng.choice(int(nonempty.sum()),size=40000,p=pt.numpy())
    cnt=torch.zeros(int(nonempty.sum()))
    for d in dr: cnt[d]+=1
    cap=1.5*math.sqrt(t); res={}
    for mode in ("exact","aniso","hybrid"):
      pp=[];ll_=[]
      for SEED in range(5):          # recovery is noisy; single fits are not evidence
        bh=torch.full((J,),-1.0,requires_grad=True)
        PH=(torch.randn(J,K,generator=torch.Generator().manual_seed(SEED))*0.1).requires_grad_(True)
        opt=torch.optim.Adam([bh,PH],lr=0.05); gen=torch.Generator().manual_seed(100+SEED)
        for step in range(700):
            Ea=E_of(bh,PH)
            if mode=="exact":    lz=torch.logsumexp(Ea[nonempty],0)
            elif mode=="aniso":  lz=logZ_aniso(bh,PH,352,gen)
            else:                lz=logZ_hybrid(bh,PH,11,32,gen)
            ll=((Ea[nonempty]-lz)*cnt).sum()/cnt.sum()
            opt.zero_grad(); (-ll).backward(); opt.step()
            with torch.no_grad():
                nn_=PH.norm(dim=1,keepdim=True).clamp_min(1e-12)
                PH.mul_((cap/nn_).clamp(max=1.0))
        with torch.no_grad():
            pf=torch.softmax(E_of(bh,PH)[nonempty],0)
            pif=(m0*pf.unsqueeze(1)).sum(0)
            pp.append(float(PH[0]@PH[1]))
            ll_.append(float((m0[:,0]*m0[:,1]*pf).sum())/float(pif[0]*pif[1]))
      res[mode]=(float(np.mean(pp)), float(np.mean(ll_)), float(np.std(pp,ddof=1)),
                 float(np.std(ll_,ddof=1)))
    print(f"{t:6.2f}{cap**2:6.2f}  ex {res['exact'][0]:6.3f}  "
          f"ani {res['aniso'][0]:6.3f}±{res['aniso'][2]:.2f}  "
          f"hyb {res['hybrid'][0]:6.3f}±{res['hybrid'][2]:.2f}   "
          f"lift true {lift_t:5.2f}  ani {res['aniso'][1]:5.2f}±{res['aniso'][3]:.2f}  "
          f"hyb {res['hybrid'][1]:5.2f}±{res['hybrid'][3]:.2f}", flush=True)
