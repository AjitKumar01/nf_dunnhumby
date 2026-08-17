import sys, os; sys.path.insert(0, "/Users/ajit/Projects/Causal/nf_dunnhumby/scripts/v3")
import torch, numpy as np
torch.set_default_dtype(torch.float64)
from data import build; from features import Features
from ragged import RaggedModel
from fit import Batcher
D=build(); J,N,C,S=int(D["n_item"]),int(D["n_user"]),int(D["n_cat"]),int(D["n_store"])
NMAX,R=60,4
lp=D["line_ptr"]; lcA=D["line_cat"]; nl=D["trip_nlines"]
tr=np.flatnonzero(D["trip_split"]==0)
maxk=np.empty(len(tr),np.int32)
for i,t in enumerate(tr):
    a,b=int(lp[t]),int(lp[t+1]); maxk[i]=np.bincount(lcA[a:b]).max() if b>a else 0
ok=(nl[tr]<=NMAX)&(maxk<=R)
F=Features(J,S,712); B=Batcher(D,F,NMAX); gen=torch.Generator().manual_seed(0)
for name,pool in (("ALL",tr),("IN-SUPPORT",tr[ok])):
    m=RaggedModel(J=J,N=N,C=C,K=32,Kz=12,nmax=NMAX,R=R,seed=0,S=S,Kp=8); m.project(0.35)
    opt=torch.optim.Adam(m.parameters(),lr=0.005); rng=np.random.default_rng(0)
    print(f"--- {name} ({len(pool):,} trips) ---",flush=True)
    print(f"{'it':>4} {'loss':>9} {'max ll':>9} {'max ll in':>10} {'max ll out':>11} {'ESS':>6}",flush=True)
    for it in range(1,201):
        sub=pool[rng.choice(len(pool),size=20,replace=False)]
        ix,ctx,hh,li,lt,lc2=B.make(sub); m.house,m.ctx=hh,ctx
        ll,ess=m.loglik(ix,li,lt,lc2,n_draws=12,generator=gen,return_ess=True)
        loss=-ll.mean(); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),2.0); opt.step(); m.project(0.35)
        if it%50==0 or it==1:
            with torch.no_grad():
                nn_=np.bincount(lt.numpy(),minlength=len(sub))
                mk=np.array([np.bincount(lc2.numpy()[lt.numpy()==i]).max() if nn_[i] else 0 for i in range(len(sub))])
                good=torch.as_tensor((nn_<=NMAX)&(mk<=R))
                a_=float(ll[good].max()) if good.any() else float('nan')
                b_=float(ll[~good].max()) if (~good).any() else float('nan')
                print(f"{it:4d} {float(loss):9.2f} {float(ll.max()):9.3f} {a_:10.3f} {b_:11.3f} {float(ess.mean()):6.3f}",flush=True)
    print(flush=True)
