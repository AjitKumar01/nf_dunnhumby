"""Does quadrature-along-v_top RECOVER a planted phi, where IS drives it to the cap?

Estimating log Z well at fixed parameters is not the same test.  The isotropic mixture also
looked good on the bench and still failed in a fit and on real data.  What matters is whether
the gradient the estimator produces leads back to the truth.  Exact enumeration is the
control; the anisotropic proposal is the previous best, which recovered at phi.phi = 2 and
broke at 3.
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

def logZ_is(b, PH, nd, gen):
    zh,_ = mode_and_vtop(b,PH)
    eps = torch.randn(nd,K,generator=gen); zs = zh+eps
    return torch.logsumexp((-0.5*(zs**2).sum(1)+log_f(b,PH,zs))
                           -(-0.5*eps.pow(2).sum(1)),0) - math.log(nd)

def logZ_quad(b, PH, n_r, n_p, gen):
    zh, v = mode_and_vtop(b,PH)
    xr,wr = np.polynomial.hermite_e.hermegauss(n_r)
    xr=torch.tensor(xr); wr=torch.tensor(wr)/math.sqrt(2*math.pi)
    eps = torch.randn(n_p,K,generator=gen)
    eps = eps - (eps@v).unsqueeze(1)*v.unsqueeze(0)
    parts=[torch.log(wr[i].clamp_min(1e-300))+log_f(b,PH,xr[i]*v.unsqueeze(0)+eps)
           for i in range(n_r)]
    return torch.logsumexp(torch.stack(parts).reshape(-1),0) - math.log(n_p)

print(f"{'true':>6}{'cap':>7}   {'exact':>9}{'IS':>9}{'quad':>9}   "
      f"{'true lift':>10}{'IS lift':>9}{'q lift':>9}")
for t in (2.0, 3.0, 4.0):
    v=math.sqrt(t)
    PH_t=torch.zeros(J,K); PH_t[0,0]=v; PH_t[1,0]=v
    b_t=torch.full((J,),-2.0)
    Et=E_of(b_t,PH_t); pt=torch.softmax(Et[nonempty],0)
    pi_t=(m0*pt.unsqueeze(1)).sum(0)
    lift_t=float((m0[:,0]*m0[:,1]*pt).sum())/float(pi_t[0]*pi_t[1])
    rng=np.random.default_rng(0)
    dr=rng.choice(int(nonempty.sum()),size=40000,p=pt.numpy())
    cnt=torch.zeros(int(nonempty.sum()))
    for d in dr: cnt[d]+=1
    cap=1.5*math.sqrt(t); res={}
    for mode in ("exact","is","quad"):
        bh=torch.full((J,),-1.0,requires_grad=True)
        PH=(torch.randn(J,K,generator=torch.Generator().manual_seed(1))*0.1).requires_grad_(True)
        opt=torch.optim.Adam([bh,PH],lr=0.05); gen=torch.Generator().manual_seed(7)
        for step in range(900):
            Ea=E_of(bh,PH)
            if mode=="exact": lz=torch.logsumexp(Ea[nonempty],0)
            elif mode=="is":  lz=logZ_is(bh,PH,512,gen)
            else:             lz=logZ_quad(bh,PH,11,32,gen)
            ll=((Ea[nonempty]-lz)*cnt).sum()/cnt.sum()
            opt.zero_grad(); (-ll).backward(); opt.step()
            with torch.no_grad():
                nn_=PH.norm(dim=1,keepdim=True).clamp_min(1e-12)
                PH.mul_((cap/nn_).clamp(max=1.0))
        with torch.no_grad():
            pf=torch.softmax(E_of(bh,PH)[nonempty],0)
            pif=(m0*pf.unsqueeze(1)).sum(0)
            res[mode]=(float(PH[0]@PH[1]),
                       float((m0[:,0]*m0[:,1]*pf).sum())/float(pif[0]*pif[1]))
    print(f"{t:6.2f}{cap**2:7.2f}   {res['exact'][0]:9.3f}{res['is'][0]:9.3f}"
          f"{res['quad'][0]:9.3f}   {lift_t:10.3f}{res['is'][1]:9.3f}{res['quad'][1]:9.3f}",
          flush=True)
