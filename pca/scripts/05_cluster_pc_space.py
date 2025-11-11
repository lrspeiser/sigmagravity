#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca_npz", required=True)
    ap.add_argument("--features_npz", required=True, help="To access names and optional scalar columns for correlations.")
    ap.add_argument("--max_k", type=int, default=6)
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()

def best_gmm(X, max_k=6, random_state=0):
    best = None
    best_k = None
    best_bic = np.inf
    for k in range(2, max_k+1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best = gmm
            best_k = k
    return best, best_k, best_bic

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    P = np.load(args.pca_npz, allow_pickle=True)
    F = np.load(args.features_npz, allow_pickle=True)
    Z = P["scores"]
    names = F["names"]
    # choose first two/three PCs
    X = Z[:, :3]
    gmm, k, bic = best_gmm(X, max_k=args.max_k, random_state=42)
    labels = gmm.predict(X)
    # Save table
    df = pd.DataFrame({"name": names, "cluster": labels})
    df.to_csv(os.path.join(args.out_dir, "clusters.csv"), index=False)
    # Simple PC1-PC2 scatter with clusters
    plt.figure()
    for c in np.unique(labels):
        m = labels==c
        plt.scatter(Z[m,0], Z[m,1], s=12, label=f"Cluster {c}", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"GMM clusters (k={k}) in PC space")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "pc_scatter_clusters.png"), dpi=160)
    plt.close()
    print(f"[OK] Clustering complete with k={k}; BIC={bic:.1f}")

if __name__ == "__main__":
    main()
