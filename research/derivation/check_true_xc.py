#!/usr/bin/env python3
"""Check R_c vs TRUE R_disk correlation."""
import pandas as pd
import json
from scipy.stats import pearsonr
import numpy as np

# Load data
sparc = pd.read_csv('C:/Users/henry/dev/sigmagravity/data/sparc/sparc_true_rdisk.csv')
sparc['Name'] = sparc['Name'].str.strip()

with open('C:/Users/henry/dev/sigmagravity/derivation/results/sparc_tanh_results.json') as f:
    tanh = json.load(f)

tanh_df = pd.DataFrame(tanh['per_galaxy'])
merged = tanh_df.merge(sparc[['Name', 'Rdisk', 'Vflat']], left_on='name', right_on='Name', how='inner')

# Filter good fits
valid = merged[(merged['r2_tanh'] > 0.5) & (merged['R_c'] > 0) & (merged['R_c'] < 50)]
print(f"N = {len(valid)} galaxies with good fits")
print()

# Correlation
r, p = pearsonr(valid['Rdisk'], valid['R_c'])
print(f"R_c vs R_disk (TRUE photometric scale length):")
print(f"  Pearson r = {r:.3f} (p = {p:.2e})")
print()

# Linear fit
slope, intercept = np.polyfit(valid['Rdisk'], valid['R_c'], 1)
print(f"  Linear fit: R_c = {slope:.2f} × R_disk + {intercept:.2f}")
print(f"  At R_disk = 2.6 kpc (MW): R_c_predicted = {slope*2.6 + intercept:.1f} kpc")
print(f"  Actual MW R_c = 6.75 kpc")
print()

# Compute true x_c
valid['x_c_true'] = valid['R_c'] / valid['Rdisk']
print(f"TRUE x_c = R_c/R_disk statistics:")
print(f"  Mean: {valid['x_c_true'].mean():.2f}")
print(f"  Median: {valid['x_c_true'].median():.2f}")
print(f"  Std: {valid['x_c_true'].std():.2f}")
print()

# MW comparison
mw_x = 6.75 / 2.6
print(f"MW x_c = 6.75/2.6 = {mw_x:.2f}")
print(f"MW percentile in SPARC distribution: {(valid['x_c_true'] < mw_x).mean()*100:.0f}%")
print()

if r > 0.5:
    print("CONCLUSION: Strong correlation - R_c scales with R_disk")
else:
    print("CONCLUSION: Weak/no correlation - R_c does NOT scale simply with R_disk")
