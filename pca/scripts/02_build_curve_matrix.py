#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="NPZ from 01_ingest_sparc.py")
    ap.add_argument("--scalars_json", required=True, help="JSON specifying which scalar columns to use (default names assumed).")
    ap.add_argument("--out_prefix", required=True, help="Prefix for output NPZ files.")
    return ap.parse_args()

def main():
    args = parse_args()
    data = np.load(args.npz, allow_pickle=True)
    curve_mat = data["curve_mat"]       # [N,K]
    weight_mat = data["weight_mat"]     # [N,K]
    scalars = data["scalars"]           # [N, S_basic]
    names = data["names"]
    x_grid = data["x_grid"]
    # scalars_json maps to canonical names: log10_Mbar, log10_Sigma0, log10_Rd, log10_Vf
    with open(args.scalars_json, "r") as f:
        cfg = json.load(f)
    use_cols = cfg.get("use_columns", [])
    # Build curve-only feature matrix
    np.savez_compressed(args.out_prefix + "_curve_only.npz",
                        X=curve_mat, W=weight_mat, names=names, x_grid=x_grid)
    # Build curve+scalars (impute nans with median)
    if use_cols:
        # We already stored the scalars in fixed order [log10_Mbar, log10_Sigma0, log10_Rd, log10_Vf]
        # Create a mask vector picking these by name
        name_to_idx = {"log10_Mbar":0, "log10_Sigma0":1, "log10_Rd":2, "log10_Vf":3}
        idx = [name_to_idx[c] for c in use_cols if c in name_to_idx]
        S = scalars[:, idx].copy()
        # impute
        for j in range(S.shape[1]):
            col = S[:, j]
            med = np.nanmedian(col)
            col[~np.isfinite(col)] = med
            S[:, j] = col
        X = np.hstack([curve_mat, S])
        # Build weights (use curve weights for curve part, 1.0 for scalars)
        W_curve = weight_mat
        W_scal = np.ones((X.shape[0], S.shape[1]), dtype=float)
        W = np.hstack([W_curve, W_scal])
        np.savez_compressed(args.out_prefix + "_curve_plus_scalars.npz",
                            X=X, W=W, names=names, x_grid=x_grid, scalar_names=np.array(use_cols))
    print("[OK] Wrote feature matrices.")

if __name__ == "__main__":
    main()
