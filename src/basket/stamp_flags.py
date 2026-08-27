"""Stamp model_flags onto checkpoints written before that key existed.

The price parameterisation is not recoverable from the tensors -- gamma = +0.0207 is a
valid softplus pre-image AND a valid coefficient -- so it must be supplied by whoever
knows how the run was launched.  Guessing is what cost run409 its MRR reading.
"""
import argparse, os, torch

def main(a):
    for p in a.ckpt:
        if not os.path.exists(p):
            print(f"  missing: {p}"); continue
        b = torch.load(p, map_location="cpu", weights_only=False)
        if not (isinstance(b, dict) and b.get("format") == 2):
            print(f"  not format 2, skipped: {os.path.basename(p)}"); continue
        old = (b.get("model_flags") or {}).get("price_soft")
        if old is not None and not a.force:
            print(f"  already stamped price_soft={old}: {os.path.basename(p)}"); continue
        b["model_flags"] = dict(price_soft=int(a.price_soft),
                                poly_degree=int(a.poly_degree))
        tmp = p + ".tmp"
        torch.save(b, tmp); os.replace(tmp, p)
        print(f"  stamped price_soft={a.price_soft} poly_degree={a.poly_degree}: "
              f"{os.path.basename(p)} (iter {b.get('iter')})")

if __name__ == "__main__":
    q = argparse.ArgumentParser()
    q.add_argument("--ckpt", nargs="+", required=True)
    q.add_argument("--price-soft", type=int, required=True)
    q.add_argument("--poly-degree", type=int, default=0)
    q.add_argument("--force", action="store_true")
    main(q.parse_args())
