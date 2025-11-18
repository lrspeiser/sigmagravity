"""
Analyze time-coherence scaling across MW, SPARC, and clusters.

Assumptions about input files:
- mw_coherence_test.json: Results from test_mw_coherence.py
- sparc_coherence_test.csv: Results from test_sparc_coherence.py
- cluster_coherence_test.json: Optional results from test_cluster_coherence.py

Usage:
  python time-coherence/analyze_full_coherence_scaling.py \
      --mw-json time-coherence/mw_coherence_test.json \
      --sparc-csv time-coherence/sparc_coherence_test.csv \
      --out-summary time-coherence/coherence_scaling_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute correlation, handling edge cases."""
    if len(x) < 3:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    try:
        return float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return float("nan")


def analyze_sparc(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze SPARC coherence scaling."""
    out: Dict[str, Any] = {}
    
    # Basic stats
    out["N_galaxies"] = int(len(df))
    out["delta_rms_mean"] = float(df["delta_rms"].mean())
    out["delta_rms_median"] = float(df["delta_rms"].median())
    out["delta_rms_std"] = float(df["delta_rms"].std())
    out["improved_frac"] = float((df["delta_rms"] < 0.0).mean())

    # Correlations
    if "sigma_v_kms" in df.columns or "sigma_v_true" in df.columns:
        sig_col = "sigma_v_kms" if "sigma_v_kms" in df.columns else "sigma_v_true"
        sig = df[sig_col].to_numpy(dtype=float)
        
        out["corr_delta_rms_sigma_v"] = safe_corr(sig, df["delta_rms"].to_numpy())
        
        if "ell_coh_mean_kpc" in df.columns:
            out["corr_ellcoh_sigma_v"] = safe_corr(sig, df["ell_coh_mean_kpc"].to_numpy())
            out["ell_coh_mean_kpc"] = float(df["ell_coh_mean_kpc"].mean())
            out["ell_coh_median_kpc"] = float(df["ell_coh_mean_kpc"].median())
        
        if "tau_coh_mean_yr" in df.columns:
            out["corr_taucoh_sigma_v"] = safe_corr(sig, df["tau_coh_mean_yr"].to_numpy())
            out["tau_coh_mean_yr"] = float(df["tau_coh_mean_yr"].mean())
            out["tau_coh_median_yr"] = float(df["tau_coh_mean_yr"].median())

    # σ-bin stats
    if sig_col in df.columns:
        bins = [0, 15, 20, 25, 30, 40, 1000]
        labels = ["<15", "15-20", "20-25", "25-30", "30-40", ">=40"]
        df["sigma_bin"] = pd.cut(df[sig_col], bins=bins, labels=labels, include_lowest=True)
        bin_stats = []
        for label in labels:
            sub = df[df["sigma_bin"] == label]
            if sub.empty:
                continue
            bin_stat = {
                "sigma_bin": str(label),
                "N": int(len(sub)),
                "delta_rms_mean": float(sub["delta_rms"].mean()),
                "delta_rms_median": float(sub["delta_rms"].median()),
            }
            if "ell_coh_mean_kpc" in sub.columns:
                bin_stat["ell_coh_mean_kpc"] = float(sub["ell_coh_mean_kpc"].mean())
            if "tau_coh_mean_yr" in sub.columns:
                bin_stat["tau_coh_mean_yr"] = float(sub["tau_coh_mean_yr"].mean())
            bin_stats.append(bin_stat)
        out["sigma_bin_stats"] = bin_stats

    return out


def analyze_clusters(cluster_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze cluster coherence scaling."""
    out: Dict[str, Any] = {}
    
    # Extract results from cluster test JSON structure
    if "clusters" in cluster_data:
        all_results = []
        for cluster_name, results in cluster_data["clusters"].items():
            if isinstance(results, list):
                all_results.extend(results)
        
        if all_results:
            df_clusters = pd.DataFrame(all_results)
            out["N_clusters"] = len(cluster_data["clusters"])
            
            for col in ["ell_coh_kpc", "tau_coh_yr", "mass_boost", "K_Einstein"]:
                if col in df_clusters.columns:
                    out[f"{col}_mean"] = float(df_clusters[col].mean())
                    out[f"{col}_median"] = float(df_clusters[col].median())
            
            # Count sufficient solutions
            if "sufficient" in df_clusters.columns:
                out["sufficient_frac"] = float(df_clusters["sufficient"].mean())
    
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Analyze time-coherence scaling across MW, SPARC, and clusters."
    )
    parser.add_argument(
        "--mw-json",
        default="time-coherence/mw_coherence_test.json",
        help="Path to MW coherence test JSON.",
    )
    parser.add_argument(
        "--sparc-csv",
        default="time-coherence/sparc_coherence_test.csv",
        help="Path to SPARC coherence test CSV.",
    )
    parser.add_argument(
        "--cluster-json",
        default=None,
        help="Optional path to cluster coherence test JSON.",
    )
    parser.add_argument(
        "--out-summary",
        default="time-coherence/coherence_scaling_summary.json",
    )
    args = parser.parse_args()

    summary: Dict[str, Any] = {}

    print("=" * 80)
    print("TIME-COHERENCE SCALING ANALYSIS")
    print("=" * 80)

    # --- MW ---
    mw_path = Path(args.mw_json)
    if mw_path.is_file():
        mw_data = json.loads(mw_path.read_text())
        # Extract best result (closest to ell0 ~ 5 kpc)
        if "results" in mw_data:
            best_mw = min(mw_data["results"], key=lambda x: abs(x.get("ell_coh_mean_kpc", 0) - 5.0))
            summary["MW"] = {
                "ell_coh_mean_kpc": float(best_mw.get("ell_coh_mean_kpc", float("nan"))),
                "tau_coh_mean_yr": float(best_mw.get("tau_coh_mean_yr", float("nan"))),
                "A_global": float(best_mw.get("A_global", float("nan"))),
                "p": float(best_mw.get("p", float("nan"))),
                "tau_geom_method": str(best_mw.get("tau_geom_method", "unknown")),
            }
        else:
            summary["MW"] = {"error": "No 'results' key in MW JSON"}
    else:
        summary["MW"] = {"error": f"Missing {args.mw_json}"}

    # --- SPARC ---
    sparc_path = Path(args.sparc_csv)
    if sparc_path.is_file():
        df_sparc = pd.read_csv(sparc_path)
        summary["SPARC"] = analyze_sparc(df_sparc)
        print("\n--- SPARC Scaling ---")
        print(f"  N galaxies: {summary['SPARC']['N_galaxies']}")
        print(f"  Mean delta_rms: {summary['SPARC']['delta_rms_mean']:.3f} km/s")
        print(f"  Improved fraction: {summary['SPARC']['improved_frac']:.1%}")
        if "ell_coh_mean_kpc" in summary["SPARC"]:
            print(f"  Mean ell_coh: {summary['SPARC']['ell_coh_mean_kpc']:.2f} kpc")
        if "corr_ellcoh_sigma_v" in summary["SPARC"]:
            print(f"  corr(ell_coh, sigma_v): {summary['SPARC']['corr_ellcoh_sigma_v']:.3f}")
    else:
        summary["SPARC"] = {"error": f"Missing {args.sparc_csv}"}

    # --- Clusters ---
    if args.cluster_json is not None:
        c_path = Path(args.cluster_json)
        if c_path.is_file():
            cluster_data = json.loads(c_path.read_text())
            summary["Clusters"] = analyze_clusters(cluster_data)
            print("\n--- Clusters ---")
            if "N_clusters" in summary["Clusters"]:
                print(f"  N clusters: {summary['Clusters']['N_clusters']}")
            if "ell_coh_kpc_mean" in summary["Clusters"]:
                print(f"  Mean ell_coh: {summary['Clusters']['ell_coh_kpc_mean']:.2f} kpc")
        else:
            summary["Clusters"] = {"error": f"Missing {args.cluster_json}"}

    # --- Cross-system comparison ---
    print("\n--- Cross-System Comparison ---")
    if "MW" in summary and "ell_coh_mean_kpc" in summary["MW"]:
        mw_ell = summary["MW"]["ell_coh_mean_kpc"]
        print(f"  MW ell_coh: {mw_ell:.2f} kpc")
        if "SPARC" in summary and "ell_coh_mean_kpc" in summary["SPARC"]:
            sparc_ell = summary["SPARC"]["ell_coh_mean_kpc"]
            print(f"  SPARC mean ell_coh: {sparc_ell:.2f} kpc")
            print(f"  Ratio (SPARC/MW): {sparc_ell/mw_ell:.2f}")
        if "Clusters" in summary and "ell_coh_kpc_mean" in summary["Clusters"]:
            cluster_ell = summary["Clusters"]["ell_coh_kpc_mean"]
            print(f"  Clusters mean ell_coh: {cluster_ell:.2f} kpc")
            print(f"  Ratio (Clusters/MW): {cluster_ell/mw_ell:.2f}")

    Path(args.out_summary).write_text(json.dumps(summary, indent=2))
    print(f"\nWrote coherence scaling summary to {args.out_summary}")
    print("=" * 80)


if __name__ == "__main__":
    main()


