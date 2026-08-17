"""Score every model on the SAME trips: evalall's seeded sample, in-support filtered.

Run:  V3_AFFINITY=1 python3 bench_same_trips.py

The baselines already sampled rather than sliced, but from their own draw -- so the
published comparison was never on identical trips.  This uses evalall.sample_split for
all of them.
"""
import sys, os; sys.path.insert(0,'/Users/ajit/Projects/Causal/nf_dunnhumby/scripts/v3')
import numpy as np, torch
torch.set_default_dtype(torch.float64)
from data import build
from features import Features
import evalall as EA
import baselines as BL
import baselines2 as B2
OUT="/Users/ajit/Projects/Causal/nf_dunnhumby/out"
D=build(); J,N,C,S=(int(D[k]) for k in ("n_item","n_user","n_cat","n_store"))
F=Features(J,S,712)
NT=3000; NMAX,R=120,23
picks={s:EA.sample_split(D,s,NT,NMAX,R) for s in ("valid","test")}
for s,v in picks.items():
    wk=D["trip_week"][v]
    print(f"  {s}: {len(v)} trips, weeks {wk.min()}-{wk.max()} ({len(np.unique(wk))} distinct)")
Bt=BL.Batches(D,F)
specs=[("shopper","v3_bl_shopper.pt",lambda: B2.Shopper(J,N,S,K=32,Kp=8),dict()),
       ("multinomial","v3_bl_multinom.pt",lambda: B2.Multinomial(J,N,S,B2.size_law(D),K=32,Kp=8),dict()),
       ("ndpp","v3_bl_ndpp.pt",lambda: B2.NDPP(J,N,S,rank=16,srank=8,K=32,Kp=8),dict())]
print(f"\n{'model':>14}{'valid/basket':>15}{'test/basket':>14}{'valid/line':>13}{'test/line':>12}")
res={}
for nm,ck,ctor,kw in specs:
    p=os.path.join(OUT,ck)
    if not os.path.exists(p):
        print(f"{nm:>14}   checkpoint {ck} not found"); continue
    try:
        m=ctor(); sd=torch.load(p,map_location="cpu",weights_only=False)
        miss,unexp=m.load_state_dict(sd if not isinstance(sd,dict) or "model" not in sd else sd["model"],strict=False)
        if miss: print(f"{nm:>14}   MISSING KEYS {miss[:3]} -- shape mismatch, skipping"); continue
        m.double().eval()
        vb,vl=B2.ev(m,Bt,picks["valid"],**kw); tb,tl=B2.ev(m,Bt,picks["test"],**kw)
        res[nm]=(vb,tb)
        print(f"{nm:>14}{vb:15.3f}{tb:14.3f}{vl:13.4f}{tl:12.4f}")
    except Exception as e:
        print(f"{nm:>14}   {type(e).__name__}: {str(e)[:70]}")
print(f"\n  ours (run90_best, evalall on the same sample): valid -41.412   test -43.025")
