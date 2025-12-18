#!/usr/bin/env python3
"""Compare axisymmetric vs 3D gradient flow invariants."""
import pandas as pd
import numpy as np

df_3d = pd.read_parquet('data/gaia/6d_brava_galcen_3d.parquet')
df_ax = pd.read_parquet('data/gaia/6d_brava_galcen.parquet')

print("="*70)
print("COMPARISON: Axisymmetric vs 3D Gradients")
print("="*70)

print("\n3D Gradients (KNN):")
print(f"  omega^2: mean={df_3d['omega2'].mean():.2f}, std={df_3d['omega2'].std():.2f}")
print(f"           min={df_3d['omega2'].min():.2f}, max={df_3d['omega2'].max():.2f}")
print(f"  theta^2: mean={df_3d['theta2'].mean():.6f}, std={df_3d['theta2'].std():.6f}")
print(f"           min={df_3d['theta2'].min():.6f}, max={df_3d['theta2'].max():.6f}")
print(f"  C_cov:   mean={df_3d['C_cov'].mean():.3f}, std={df_3d['C_cov'].std():.3f}")
print(f"  Sigma:   mean={df_3d['Sigma'].mean():.6f}, std={df_3d['Sigma'].std():.6f}")

print("\nAxisymmetric Approximation:")
print(f"  omega^2: mean={df_ax['omega2'].mean():.2f}, std={df_ax['omega2'].std():.2f}")
print(f"           min={df_ax['omega2'].min():.2f}, max={df_ax['omega2'].max():.2f}")
print(f"  theta^2: mean={df_ax['theta2'].mean():.6f}, std={df_ax['theta2'].std():.6f}")
print(f"           min={df_ax['theta2'].min():.6f}, max={df_ax['theta2'].max():.6f}")
print(f"  C_cov:   mean={df_ax['C_cov'].mean():.3f}, std={df_ax['C_cov'].std():.3f}")
print(f"  Sigma:   mean={df_ax['Sigma'].mean():.6f}, std={df_ax['Sigma'].std():.6f}")

print("\n" + "="*70)
print("Differences:")
print("="*70)
print(f"  omega^2 ratio (3D/ax): {(df_3d['omega2'].mean() / df_ax['omega2'].mean()):.3f}")
if df_ax['theta2'].mean() > 0:
    theta_ratio = df_3d['theta2'].mean() / df_ax['theta2'].mean()
    print(f"  theta^2 ratio (3D/ax): {theta_ratio:.3f}")
else:
    print(f"  theta^2 ratio (3D/ax): inf (axisymmetric is zero)")
print(f"  C_cov difference: {df_3d['C_cov'].mean() - df_ax['C_cov'].mean():.4f}")
print(f"  Sigma difference: {df_3d['Sigma'].mean() - df_ax['Sigma'].mean():.6f}")
print("="*70)

