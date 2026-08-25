"""WILLINGNESS TO PAY, derived from the model rather than bolted on.

Adding j to a partial basket S changes the energy by

    Delta_j = b_j + sum_{k in S} phi_j'phi_k - rho_c n_c(S) - [rho_0(n+1) - rho_0(n)]

but in a Gibbs SET model the observed basket is a sample, not the mode, so Delta_j < 0 is
normal even for items that were bought -- an indifference point at Delta_j = 0 is therefore
not the right notion of acceptance here.

What IS well defined is that pi_j = P(j in S) falls monotonically with price.  The
reservation price is taken as the multiple at which pi_j drops to HALF its value at the
price actually paid: the point where the household is as likely to walk away as to buy.
b_j carries price as

    b_j = b_j^0 - (gamma_h.beta_j) * [ m + kappa (dlp_j - m) ]

so setting Delta_j = 0 and solving for dlp_j gives the RESERVATION log-price: the price at
which this household, on this trip, becomes indifferent to this product.

    dlp* = m + ( b_j^0 - gb*m + interaction - drho_0 ) / (gb * kappa)

The reservation MULTIPLE exp(dlp* - dlp_observed) is how much dearer the product could get
before it drops out.  A model with a usable WTP has multiples near 1; a model whose price
coefficient is too small has multiples in the hundreds, and a seller facing it can raise
prices without limit -- which is exactly why the markdown MDP optimum sits at the boundary.

Run:  V3_AFFINITY=1 python3 wtp.py --ckpt v3_run90_best.pt
"""
import argparse, os, sys
import numpy as np, torch
from torch.nn.functional import softplus
torch.set_default_dtype(torch.float64)
from data import build
from features import Features
from fit import Batcher
from ragged import RaggedModel

def log(m): print(f"[wtp] {m}", flush=True)

def main(a):
    D=build(); J,N,C,S=(int(D[k]) for k in ("n_item","n_user","n_cat","n_store"))
    F=Features(J,S,712); Bt=Batcher(D,F,a.nmax)
    m=RaggedModel(J=J,N=N,C=C,K=32,Kz=a.Kz,nmax=a.nmax,R=a.R,S=S,Kp=8)
    blob=torch.load(os.path.join("../../out",a.ckpt),map_location="cpu",weights_only=False)
    sd=blob["model"] if isinstance(blob,dict) and blob.get("format")==2 else blob
    miss,_=m.load_state_dict(sd,strict=False)
    assert not [k for k in miss if k not in
                ("cat_of","price_kappa","factored_size_enabled",
                 "factored_size_log_p")], miss
    co=torch.zeros(J,dtype=torch.long)
    co[torch.as_tensor(D["line_item"],dtype=torch.long)]=torch.as_tensor(D["line_cat"],dtype=torch.long)
    with torch.no_grad(): m.cat_of.copy_(co)
    m.double().eval()
    kap=float(softplus(m.price_kappa)) if hasattr(m,"price_kappa") else 1.0
    if a.kappa>0: kap=a.kappa
    log(f"{a.ckpt} iter {blob.get('iter','?')}, kappa = {kap:.2f}")
    lp=D["line_ptr"]
    idx=np.flatnonzero(D["trip_split"]==1)
    idx=np.array([t for t in idx if 2<=int(lp[t+1])-int(lp[t])<=a.nmax])
    idx=np.sort(np.random.default_rng(0).choice(idx,size=min(a.n_trips,len(idx)),replace=False))
    from ragged import smolyak_grid
    m.quad=smolyak_grid(a.Kz,a.quad_q)
    if a.kappa>0 and hasattr(m,"price_kappa"):
        with torch.no_grad(): m.price_kappa.fill_(float(np.log(np.expm1(a.kappa))))
    grid=np.array([0.0,0.10,0.25,0.50,1.0,2.0,4.0,9.0])      # price multiples - 1
    keep=np.zeros((len(grid),),dtype=np.float64); tot=0.0
    half=[]
    for k in range(0,len(idx),a.chunk):
        ix,ctx,lctx,hh,LI,LT,LC,LU=Bt.make(idx[k:k+a.chunk]); m.house,m.ctx=hh,ctx
        onbasket=torch.zeros(ix.item.shape[0],dtype=torch.bool)
        for b in range(ix.B):
            bs=set(int(x) for x in LI[LT==b])
            sel=(ix.item_trip==b).nonzero().flatten()
            for i in sel:
                if int(ix.item[i]) in bs: onbasket[int(i)]=True
        if not bool(onbasket.any()): continue
        pis=[]
        for g in grid:
            c2=dict(ctx); d=np.log1p(g)
            if a.uniform:
                # EVERY price rises: a common shift.  Governed by the aggregate elasticity
                # (-0.12), and correctly weak -- groceries are necessities, so a general
                # rise shrinks baskets only slightly.  kappa does not act here by design.
                c2["dlp"]=ctx["dlp"]+d
                if "dlp_bar" in ctx: c2["dlp_bar"]=ctx["dlp_bar"]+d
            else:
                # ONE product rises, the rest hold: an idiosyncratic shift.  This is the
                # WTP a retailer actually faces when pricing a single line, and it is where
                # the customer can switch away.  kappa scales exactly this.
                dd=ctx["dlp"].clone(); dd[onbasket]=dd[onbasket]+d
                c2["dlp"]=dd
            m.ctx=c2; pis.append(m.pi_quad(ix)[onbasket].numpy())
        m.ctx=ctx
        P=np.stack(pis)                                   # [len(grid), n_purchased]
        base=P[0]
        # cells whose base probability is numerically zero cannot express a RATIO, and
        # including them silently normalised the base row to 0.801 instead of 1.000.
        good=base>1e-6
        if not good.any(): continue
        P=P[:,good]; base=base[good]
        keep+=(P/base).sum(1); tot+=P.shape[1]
        # first multiple where pi falls below half
        below=P<0.5*base
        h=np.where(below.any(0),grid[np.argmax(below,0)],np.inf)
        half.append(h)
    half=np.concatenate(half)
    log(f"{int(tot):,} purchased (household, product, trip) cells\n")
    log(f"  pi_j retained as price rises (1.00 = unchanged):")
    for g,v in zip(grid,keep/max(tot,1)):
        log(f"    price x{1+g:5.2f}   pi retained {v:6.3f}")
    log(f"\n  reservation price (pi halves) reached by x{grid[-1]+1:.0f}: "
        f"{100*np.isfinite(half).mean():.1f}% of cells")
    if np.isfinite(half).any():
        log(f"  median reservation multiple among those: x{1+np.median(half[np.isfinite(half)]):.2f}")
    log(f"\n  a usable WTP needs pi to fall substantially by x1.25-x1.5.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ckpt",default="v3_run90_best.pt"); p.add_argument("--kappa",type=float,default=0.0)
    p.add_argument("--n-trips",type=int,default=240); p.add_argument("--chunk",type=int,default=24)
    p.add_argument("--Kz",type=int,default=4); p.add_argument("--nmax",type=int,default=120)
    p.add_argument("--R",type=int,default=23)
    p.add_argument("--quad-q",type=int,default=8)
    p.add_argument("--uniform",type=int,default=0)
    main(p.parse_args())
