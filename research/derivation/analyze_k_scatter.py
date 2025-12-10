#!/usr/bin/env python3
"""Analyze K(R) scatter by radial bin."""
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/henry/dev/sigmagravity/data/gaia/outputs/mw_rar_starlevel_full.csv')
df['K'] = 10**(df['log10_g_obs'] - df['log10_g_bar']) - 1
df = df[(df['K'] > -0.5) & (df['K'] < 50)]

print(f"Total stars: {len(df):,}")
print(f"R range: {df.R_kpc.min():.2f} - {df.R_kpc.max():.2f} kpc")
print()
print("K statistics by R bins:")
print("-" * 70)
bins = [0, 3, 5, 6, 7, 8, 9, 10, 12, 15, 25]
for i in range(len(bins)-1):
    mask = (df.R_kpc >= bins[i]) & (df.R_kpc < bins[i+1])
    sub = df[mask]
    cv = sub.K.std() / sub.K.mean() * 100 if sub.K.mean() > 0 else 0
    print(f"  R = {bins[i]:2d}-{bins[i+1]:2d} kpc: N={len(sub):6,}, K = {sub.K.mean():.3f} +/- {sub.K.std():.3f}  (CV={cv:.0f}%)")

print()
print("Tanh fit prediction at bin centers:")
A, R1, w, c = 0.95, 6.75, 1.78, 1.02
for i in range(len(bins)-1):
    R = (bins[i] + bins[i+1]) / 2
    K_pred = A * np.tanh((R - R1) / w) + c
    print(f"  R = {R:4.1f} kpc: K_pred = {K_pred:.3f}")
