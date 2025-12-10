#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
from common import zscore_weighted, aggregate_row_weights, pca_weighted_svd, principal_angles

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_npz", required=True)
    ap.add_argument("--subset_csv", required=True, help="Metadata CSV containing a column to split subsets.")
    ap.add_argument("--subset_column", required=True, help="Column name to define subsets (e.g., HSB_LSB).")
    ap.add_argument("--n_components", type=int, default=3)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    F = np.load(args.features_npz, allow_pickle=True)
    X = F["X"]; W = F["W"]; names = [str(n) for n in F["names"]]
    meta = pd.read_csv(args.subset_csv)
    # ensure we can map names -> subset
    name_col = None
    for c in meta.columns:
        if c.lower() in ("name","galaxy","id"):
            name_col = c
            break
    if name_col is None:
        raise ValueError("No name/id column found in subset CSV")

    mlookup = dict(zip(meta[name_col].astype(str), meta[args.subset_column]))
    subset = np.array([mlookup.get(n, np.nan) for n in names], dtype=object)
    uniq = [u for u in pd.unique(subset) if isinstance(u, str) or np.isfinite(u)]
    if len(uniq) < 2:
        raise RuntimeError("Need at least two subsets to compare.")
    # Compute PCA per subset
    results = {}
    for u in uniq:
        m = (subset == u)
        Xz, mu, sd = zscore_weighted(X[m], W[m])
        w = aggregate_row_weights(W[m])
        out = pca_weighted_svd(Xz, w, n_components=args.n_components)
        results[u] = out

    # principal angles between first two subsets (or all pairs)
    keys = list(results.keys())
    lines = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            ui, uj = keys[i], keys[j]
            Ci = results[ui]["components"].T      # [D, k]
            Cj = results[uj]["components"].T
            ang = principal_angles(Ci, Cj, k=min(args.n_components, Ci.shape[1], Cj.shape[1]))
            lines.append((ui, uj, ang.tolist()))
    # Save report
    out_path = os.path.join(args.out_dir, "subset_principal_angles.txt")
    with open(out_path, "w") as f:
        for (ui, uj, ang) in lines:
            f.write(f"{ui} vs {uj}: radians={ang}\n")
    print(f"[OK] Subset stability written to {out_path}")

if __name__ == "__main__":
    main()
