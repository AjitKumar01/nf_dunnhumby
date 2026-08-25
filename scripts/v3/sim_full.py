"""COMPLETE markdown simulation: every validation trip, every household, every store,
every week, every product in each assortment, with units.

Reward is computed exactly, not sampled:
    E[margin] = sum over assortment slots of  pi_j * E[units_j] * (price_j - cost_j)
    pi_j      = d log(Z-1) / d b_j            exact, one backward pass through the grid
    E[units_j]= 1 + exp(a_q - softplus(gamma_q).softplus(beta_q) * dlp)

Run:  V3_AFFINITY=1 python3 sim_full.py --ckpt v3_run90_best.pt
"""
import argparse, os, sys, time
import numpy as np, torch
from torch.nn.functional import softplus
torch.set_default_dtype(torch.float64)
from data import build
from features import Features
from fit import Batcher
from ragged import RaggedModel, set_quad

def log(m): print(f"[sim] {m}", flush=True)

def main(a):
    D=build(); J,N,C,S=(int(D[k]) for k in ("n_item","n_user","n_cat","n_store"))
    F=Features(J,S,712); Bt=Batcher(D,F,a.nmax)
    m=RaggedModel(J=J,N=N,C=C,K=32,Kz=a.Kz,nmax=a.nmax,R=a.R,S=S,Kp=8)
    blob=torch.load(os.path.join("../../out",a.ckpt),map_location="cpu",weights_only=False)
    sd=blob["model"] if isinstance(blob,dict) and blob.get("format")==2 else blob
    miss,_=m.load_state_dict(sd,strict=False)
    assert not [k for k in miss if k not in ("cat_of","price_kappa")], miss
    co=torch.zeros(J,dtype=torch.long)
    co[torch.as_tensor(D["line_item"],dtype=torch.long)]=torch.as_tensor(D["line_cat"],dtype=torch.long)
    with torch.no_grad(): m.cat_of.copy_(co)
    _q=blob.get("quad") or {} if isinstance(blob,dict) else {}
    _qd=set_quad(m, _q.get("quad_q",a.quad_q), _q.get("qmc_n",0),
                 _q.get("qmc_seed",0), Kz=a.Kz, probe=_q.get("probe", 8), steps=_q.get("steps", 4), chunk=_q.get("chunk", 0))
    m.double().eval()
    logp=torch.as_tensor(np.load("../../basket_input/log_price.npy"))
    idx=np.flatnonzero(D["trip_split"]==(1 if a.split=="valid" else 2))
    lp=D["line_ptr"]
    idx=np.array([t for t in idx if int(lp[t+1])-int(lp[t])<=a.nmax])
    if a.n_trips: idx=np.sort(np.random.default_rng(0).choice(idx,size=min(a.n_trips,len(idx)),replace=False))
    log(f"{a.ckpt} iter {blob.get('iter','?')}; log Z: {_qd}")
    log(f"COVERAGE  {len(idx):,} trips, {len(np.unique(D['trip_store'][idx]))} stores, "
        f"{len(np.unique(D['trip_user'][idx])):,} households, "
        f"{len(np.unique(D['trip_week'][idx]))} weeks, all {J:,} products in each assortment")
    obs_lines=sum(int(lp[t+1])-int(lp[t]) for t in idx)
    obs_units=float(sum(D["line_units"][int(lp[t]):int(lp[t+1])].sum() for t in idx))
    log(f"OBSERVED  {obs_lines:,} lines, {obs_units:,.0f} units "
        f"({obs_lines/len(idx):.2f} lines and {obs_units/len(idx):.2f} units per trip)\n")
    acts=[float(x) for x in a.actions.split(",")]
    print(f"{'action':>9}{'price':>8}{'E[lines]':>11}{'E[units]':>11}{'revenue':>12}"
          f"{'cost':>11}{'MARGIN':>11}{'u/trip':>9}")
    out={}
    for act in acts:
        t0=time.time(); EL=EU=REV=CST=0.0
        for k in range(0,len(idx),a.chunk):
            ix,ctx,lctx,hh,LI,LT,LC,LU=Bt.make(idx[k:k+a.chunk]); m.house,m.ctx=hh,ctx
            c2=dict(ctx); c2["dlp"]=ctx["dlp"]+act
            if "dlp_bar" in ctx: c2["dlp_bar"]=ctx["dlp_bar"]+act
            m.ctx=c2
            pi=m.pi_quad(ix)
            with torch.no_grad():
                day=torch.as_tensor(D["trip_day"][idx[k:k+a.chunk]],dtype=torch.long)
                base=torch.exp(logp[ix.item,day[ix.item_trip]])
                z=m.a_q[ix.item]-(softplus(m.gamma_q[m.house[ix.item_trip]])
                                  *softplus(m.beta_q[ix.item])).sum(-1)*(c2["dlp"])
                eu=1.0+torch.exp(z.clamp(-6.,4.))
            price=base*float(np.exp(act)); cost=base*(1.0-a.margin0)
            # DECLARED outside option: a price rise costs trips.  eta = 0 reproduces the
            # replay environment.  It is an assumption, not an estimate -- see
            # RetailEnv.participation for the three designs that failed to identify it.
            f=1.0
            if a.eta>0:
                p0=0.574
                f=(1/(1+np.exp(-(np.log(p0/(1-p0))-a.eta*act))))/p0
            EL+=f*float(pi.sum()); EU+=f*float((pi*eu).sum())
            REV+=f*float((pi*eu*price).sum()); CST+=f*float((pi*eu*cost).sum())
        out[act]=(EL,EU,REV,CST,REV-CST)
        print(f"{act:+9.3f}{100*(np.exp(act)-1):+7.0f}%{EL:11.0f}{EU:11.0f}{REV:12.0f}"
              f"{CST:11.0f}{REV-CST:11.0f}{EU/len(idx):9.2f}   {time.time()-t0:.0f}s")
    b=max(out,key=lambda k:out[k][4])
    print(f"\nmargin-maximising action {b:+.3f} ({100*(np.exp(b)-1):+.0f}% price)")
    print(f"at action 0: model {out.get(0.0,(0,0))[0]/len(idx):.2f} lines and "
          f"{out.get(0.0,(0,0,0))[1]/len(idx):.2f} units per trip "
          f"vs observed {obs_lines/len(idx):.2f} / {obs_units/len(idx):.2f}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ckpt",default="v3_run90_best.pt")
    p.add_argument("--split",default="valid"); p.add_argument("--n-trips",type=int,default=0)
    p.add_argument("--actions",default="-0.223,-0.105,0.0,0.105,0.223")
    p.add_argument("--margin0",type=float,default=0.30)
    p.add_argument("--eta",type=float,default=0.0,help="DECLARED outside-option elasticity")
    p.add_argument("--Kz",type=int,default=4); p.add_argument("--quad-q",type=int,default=8)
    p.add_argument("--nmax",type=int,default=120); p.add_argument("--R",type=int,default=23)
    p.add_argument("--chunk",type=int,default=24)
    main(p.parse_args())
