"""
Analyze cluster coherence results and create summary table.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    cluster_json = Path("time-coherence/cluster_coherence_test.json")
    out_csv = Path("time-coherence/cluster_coherence_summary.csv")
    
    if not cluster_json.exists():
        print(f"Error: {cluster_json} not found. Run test_cluster_coherence.py first.")
        return
    
    with open(cluster_json, "r") as f:
        data = json.load(f)
    
    # Extract results
    rows = []
    
    if "clusters" in data:
        clusters_data = data["clusters"]
    else:
        # Old format: list of results
        clusters_data = {"unknown": data}
    
    for cluster_name, results in clusters_data.items():
        if not isinstance(results, list):
            continue
        
        # Find best result (highest mass boost that's sufficient)
        best = None
        best_boost = 0.0
        
        for r in results:
            if r.get("sufficient", False):
                boost = r.get("mass_boost", 0.0)
                if boost > best_boost:
                    best_boost = boost
                    best = r
        
        if best:
            rows.append({
                "cluster": cluster_name,
                "K_Einstein": best.get("K_Einstein", 0.0),
                "mass_boost": best.get("mass_boost", 1.0),
                "M_total_Msun": best.get("M_total_Msun", 0.0),
                "M_required_Msun": best.get("M_required_Msun", 0.0),
                "v_turb_kms": best.get("v_turb_kms", 0.0),
                "L_turb_kpc": best.get("L_turb_kpc", 0.0),
                "A_global": best.get("A_global", 1.0),
            })
    
    if not rows:
        print("No valid cluster results found")
        return
    
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    
    print("=" * 80)
    print("CLUSTER COHERENCE SUMMARY")
    print("=" * 80)
    print(f"\n{len(df)} clusters analyzed:")
    print()
    print(df.to_string(index=False))
    print(f"\nStatistics:")
    print(f"  Mean K_Einstein: {df['K_Einstein'].mean():.3f}")
    print(f"  Mean mass boost: {df['mass_boost'].mean():.2f}x")
    print(f"  Range: {df['mass_boost'].min():.2f}x - {df['mass_boost'].max():.2f}x")
    print(f"\nSaved to {out_csv}")

if __name__ == "__main__":
    main()

