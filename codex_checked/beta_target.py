"""Empirical per-product own-price response, as a calibration target for beta.

Run:  V3_AFFINITY=1 python3 beta_target.py

For each product: regress log(1 + weekly purchases) on that week's mean log-price deviation.
Confounded by promotions and seasonality, so it is a TARGET to pull toward, not a truth --
which is why the penalty is a weak pull on the RANKING of products, not on their levels.
"""
import sys; sys.path.insert(0,'/Users/ajit/Projects/Causal/nf_dunnhumby/scripts/v3')
import numpy as np
from data import build
D=build(); J=int(D["n_item"]); lp=D["line_ptr"]; li=D["line_item"]
dev=np.load("../../basket_input/log_price_dev.npy")
wk=D["trip_week"]; nw=int(wk.max())+1
tr=np.flatnonzero(D["trip_split"]==0)
cnt=np.zeros((J,nw))
for t in tr:
    w=int(wk[t])
    for x in li[int(lp[t]):int(lp[t+1])]: cnt[int(x),w]+=1
pw=np.zeros((J,nw))
for w in range(nw):
    s=slice(w*7,(w+1)*7)
    if s.start<dev.shape[1]: pw[:,w]=dev[:,s].mean(1)
freq=cnt.sum(1)
slope=np.full(J,np.nan); wgt=np.zeros(J)
for j in range(J):
    if freq[j]<200 or (cnt[j]>0).sum()<30: continue
    y=np.log1p(cnt[j]); x=pw[j]; m=np.isfinite(x)
    if m.sum()<20 or x[m].std()<1e-6: continue
    slope[j]=np.polyfit(x[m],y[m],1)[0]; wgt[j]=min(freq[j],5000.0)
ok=np.isfinite(slope)
# the model's own-price response is -softplus(gamma).softplus(beta), so the TARGET for
# softplus(beta_j) is -slope_j, clipped at zero (a positive slope carries no information
# about price sensitivity, only noise)
# A positive fitted slope is noise, not evidence of zero price sensitivity.  Clipping it
# to 0 and keeping full weight tells the model "this product is EXACTLY price-insensitive",
# which is a strong and false claim about 27% of the estimated products.  Keep the sign
# information where it is physically sensible and drop the weight where it is not.
tgt=np.where(ok,np.clip(-slope,0.0,None),0.0)
wgt=np.where(ok & (slope<0),wgt,0.0)
np.savez("../../basket_input/v3_beta_target.npz",target=tgt,weight=wgt,slope=np.nan_to_num(slope))
print(f"wrote v3_beta_target.npz: {int(ok.sum()):,} of {J:,} products with an estimate")
print(f"  empirical slope: median {np.nanmedian(slope[ok]):+.3f}  "
      f"IQR [{np.nanquantile(slope[ok],.25):+.3f}, {np.nanquantile(slope[ok],.75):+.3f}]")
print(f"  target softplus(beta): median {np.median(tgt[ok]):.3f}  "
      f"p90 {np.quantile(tgt[ok],.9):.3f}   (zero for {100*np.mean(tgt[ok]==0):.0f}% -- positive slopes)")
