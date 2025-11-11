#!/usr/bin/env python3
import argparse, os, json
import numpy as np
from common import zscore_weighted, aggregate_row_weights, pca_weighted_svd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_npz", required=True, help="NPZ with X and W (from 02_build_curve_matrix.py).")
    ap.add_argument("--n_components", type=int, default=10)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="curve_only", help="Tag for output file naming.")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = np.load(args.features_npz, allow_pickle=True)
    X = data["X"]
    W = data["W"]
    names = list(data["names"])
    # Weighted z-score per feature
    Xz, mu, sd = zscore_weighted(X, W)
    # Aggregate to row weights
    w = aggregate_row_weights(W, method="median")
    # Weighted SVD PCA
    out = pca_weighted_svd(Xz, w, n_components=args.n_components)
    components = out["components"]
    scores = out["scores"]
    evr = out["explained_variance_ratio"]

    out_npz = os.path.join(args.out_dir, f"pca_results_{args.tag}.npz")
    np.savez_compressed(out_npz,
                        components=components,
                        scores=scores,
                        evr=evr,
                        mu=mu, sd=sd, w=w,
                        names=np.array(names))
    # Also write simple CSVs for convenience
    np.savetxt(os.path.join(args.out_dir, f"pca_explained_{args.tag}.csv"),
               evr, delimiter=",", fmt="%.8f")
    print(f"[OK] PCA complete. Wrote {out_npz}")

if __name__ == "__main__":
    main()
