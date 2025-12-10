#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca_npz", required=True)
    ap.add_argument("--grid_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k_curve", type=int, default=None, help="If components include curve+scalars, set K for curve length.")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    P = np.load(args.pca_npz, allow_pickle=True)
    with open(args.grid_json, "r") as f:
        G = json.load(f)
    evr = P["evr"]
    comps = P["components"]      # [n_comp, D]
    # Scree (cumulative)
    plt.figure()
    plt.plot(np.arange(1, len(evr)+1), np.cumsum(evr))
    plt.xlabel("Number of PCs")
    plt.ylabel("Cumulative explained variance")
    plt.title("Scree (cumulative)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "scree_cumulative.png"), dpi=160)
    plt.close()

    # Radial loadings for PC1/PC2 over the curve segment
    x_grid = np.array(G["x_grid"], dtype=float)
    if args.k_curve is None:
        # assume pure curves
        k = len(x_grid)
    else:
        k = args.k_curve

    for pc in [0,1,2]:
        if pc >= comps.shape[0]: break
        plt.figure()
        plt.plot(x_grid, comps[pc, :k])
        plt.xlabel("R / R_d")
        plt.ylabel(f"PC{pc+1} loading (curve part)")
        plt.title(f"PC{pc+1} radial loading")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"pc{pc+1}_radial_loading.png"), dpi=160)
        plt.close()

    print("[OK] Wrote scree and radial loading plots.")

if __name__ == "__main__":
    main()
