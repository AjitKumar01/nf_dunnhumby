"""Trace one generated basket, printing the number at every stage."""
import sys, numpy as np, torch, math
sys.path.insert(0,'../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel, log_f_ragged, poly_mul_trunc

D=build(); F=Features(int(D["n_item"]),int(D["n_store"]),712); Bt=Batcher(D,F,120)
m=RaggedModel(J=int(D["n_item"]),N=int(D["n_user"]),C=int(D["n_cat"]),K=32,Kz=12,
              nmax=120,R=23,S=int(D["n_store"]),Kp=8)
m.load_state_dict(torch.load('../../out/v3_run22_cal.pt',map_location='cpu')); m.double().eval()
va=np.flatnonzero(D["trip_split"]==2)[:1]
ix,ctx,lctx,hh,*_=Bt.make(va); m.house,m.ctx=hh,ctx

with torch.no_grad():
    # --- stage 1: z ---------------------------------------------------------------
    z=torch.zeros(1,1,m.Kz)
    for _ in range(1):
        zz=z.detach().requires_grad_(True)
        with torch.enable_grad(): lf=log_f_ragged(m,zz,ix,True).sum()
        z=torch.autograd.grad(lf,zz)[0]
    g=torch.Generator().manual_seed(4)
    zs=z.detach()+torch.randn(1,16,m.Kz,generator=g)
    lw=(-0.5*(zs**2).sum(-1))+log_f_ragged(m,zs,ix,True)+0.5*((zs-z.detach())**2).sum(-1)
    pick=int(torch.multinomial(torch.softmax(lw,1)[0],1,generator=g))
    zsel=zs[0,pick]
    print(f"STAGE 1  z drawn from 16 candidates, picked #{pick}; ||z|| = {float(zsel.norm()):.3f}")

    # --- weights w_j(z) -----------------------------------------------------------
    phi=m.phi[ix.item]
    logw=m.b_flat(ix)-0.5*(phi**2).sum(-1)+(zsel*phi).sum(-1)
    M=float(logw.max()); w=torch.exp(logw-M)
    print(f"         w_j = exp(b_j + z.phi_j - |phi_j|^2/2) for all {len(w):,} products")
    print(f"         largest w (scaled) {float(w.max()):.4f}, median {float(w.median()):.2e}")

    # --- per category: e_r(w) and G_c[r] ------------------------------------------
    rows=(ix.row_trip==0).nonzero().flatten().tolist()
    polys=[]
    for r_ in rows:
        sel=(ix.row_of==r_).nonzero().flatten()
        ww=w[sel]
        E=torch.zeros(len(ww)+1,m.R+1); E[0,0]=1.0
        for k in range(1,len(ww)+1):
            E[k]=E[k-1].clone(); E[k,1:]+=ww[k-1]*E[k-1,:-1]
        deg=torch.arange(m.R+1,dtype=w.dtype)
        a_c=torch.exp(-m.rho_c[ix.row_cat[r_]]*deg*(deg-1)/2.0)
        polys.append(a_c*E[len(ww)])
    print(f"\nSTAGE 2a per category c: e_r(w_c) = sum over r-subsets of prod w_j")
    print(f"         then G_c[r] = exp(-rho_c r(r-1)/2) * e_r(w_c);  {len(polys)} categories")
    print(f"         example G_c[0..4] for one category: "
          + " ".join(f"{float(polys[0][k]):.3e}" for k in range(5)))

    # --- convolve to A_n ----------------------------------------------------------
    pref=[torch.ones(1,dtype=w.dtype)]
    for G in polys: pref.append(poly_mul_trunc(pref[-1].unsqueeze(0),G.unsqueeze(0),m.nmax)[0])
    A=pref[-1]
    nax=torch.arange(A.shape[-1],dtype=A.dtype)
    lg=torch.log(A.clamp_min(1e-300))-m.rho_0()[:A.shape[-1]]+nax*M
    pn=torch.softmax(lg[1:],0)
    print(f"\nSTAGE 2b A_n = convolution of all {len(polys)} G_c  -> coefficient for each size n")
    print(f"         P(n) = softmax over n of [ log A_n - rho_0(n) + n*M ]")
    for k in (1,3,5,8,12,20):
        print(f"           n={k:3d}   log A_n {float(torch.log(A[k])):9.3f}   "
              f"rho_0 {float(m.rho_0()[k]):8.3f}   P(n) {float(pn[k-1]):.4f}")
    n=int(torch.multinomial(pn,1,generator=g))+1
    print(f"         drawn n = {n}")

    # --- allocation ---------------------------------------------------------------
    print(f"\nSTAGE 3  split n={n} across categories, backwards:")
    print(f"         P(category c takes r | n left) proportional to G_c[r] * A^(c-1)_[n-r]")
    left=n; taken=[]
    for c in range(len(polys)-1,-1,-1):
        G,P=polys[c],pref[c]
        hi=min(left,G.shape[0]-1)
        cand=torch.tensor([float(G[r]*P[left-r]) if left-r<P.shape[0] else 0.0
                           for r in range(hi+1)])
        if float(cand.sum())<=0: continue
        r_take=int(torch.multinomial(cand/cand.sum(),1,generator=g))
        if r_take: taken.append((c,r_take,float(cand[r_take]/cand.sum())))
        left-=r_take
        if left==0: break
    for c,r,p in taken[:4]:
        print(f"           category slot {c:3d} takes {r} item(s)   P = {p:.3f}")

    # --- items within category ----------------------------------------------------
    print(f"\nSTAGE 4  within a category needing r items, walk the SAME e_r backwards:")
    print(f"         P(product k in | need r from first k) = w_k e_(r-1)(w_1..w_k-1) / e_r(w_1..w_k)")
    print(f"\nEvery stage divides by exactly what the next stage sums to, so the four")
    print(f"probabilities multiply back to P(S | z) -- the chain is exact, not approximate.")
