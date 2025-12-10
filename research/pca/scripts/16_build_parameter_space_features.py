#!/usr/bin/env python3
"""
Build parameter-space PCA features for SPARC galaxies.

This creates per-galaxy summary statistics for a different kind of PCA:
- Instead of rotation curve shapes: Summary parameters
- Features: K at key radii, RAR metrics, baryonic properties, etc.

This answers: "What model features explain variance in outcomes?"
vs curve PCA which answers: "What shape modes exist in population?"
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def coherence_function(R, l0=5.0, p=2.0, n_coh=1.5):
    """Burr-XII coherence function"""
    x = (R / l0)**p
    return 1.0 - (1.0 + x)**(-n_coh)

def compute_K_at_radii(R_eval, V_bar, A=0.6, l0=5.0, p=2.0, n_coh=1.5):
    """Compute K values at specific radii"""
    # For simplicity, use global K = A * C(R)
    # (Real implementation would use local density suppression)
    C = coherence_function(R_eval, l0, p, n_coh)
    K = A * C
    return K

def compute_gbar_at_radius(R_target, R_data, V_bar_data):
    """Interpolate to get g_bar at specific radius"""
    # g_bar = V_bar^2 / R
    g_bar_data = V_bar_data**2 / np.maximum(R_data, 0.1) / 3.086e16
    
    # Interpolate to target radius
    if R_target < R_data.min() or R_target > R_data.max():
        return np.nan
    
    g_bar_target = np.interp(R_target, R_data, g_bar_data)
    return g_bar_target

def compute_rar_metrics(V_obs, V_bar, eV_obs):
    """Compute RAR bias and scatter"""
    # For each point: log(g_obs / g_bar)
    g_obs = V_obs**2
    g_bar = V_bar**2
    
    # Filter valid points
    mask = (g_bar > 0) & (g_obs > 0) & np.isfinite(g_obs) & np.isfinite(g_bar)
    
    if mask.sum() < 3:
        return np.nan, np.nan
    
    log_ratio = np.log10(g_obs[mask] / g_bar[mask])
    
    bias = np.mean(log_ratio)  # dex
    scatter = np.std(log_ratio, ddof=1)  # dex
    
    return bias, scatter

def build_galaxy_features(curve_data, meta_row, name):
    """Build feature vector for one galaxy"""
    R = curve_data['R_kpc'].values
    V_obs = curve_data['V_obs'].values
    eV_obs = curve_data['eV_obs'].values
    
    V_disk = curve_data.get('V_disk', np.zeros_like(R)).values
    V_gas = curve_data.get('V_gas', np.zeros_like(R)).values
    V_bul = curve_data.get('V_bul', np.zeros_like(R)).values if 'V_bul' in curve_data else np.zeros_like(R)
    V_bar = np.sqrt(V_disk**2 + V_gas**2 + V_bul**2)
    
    # Baryonic properties
    Rd = meta_row.get('Rd', np.nan)
    Mbar = meta_row.get('Mbar', np.nan)
    Sigma0 = meta_row.get('Sigma0', np.nan)
    Vf = meta_row.get('Vf', np.nan)
    
    # Compute g_bar at key radii (in kpc)
    gbar_2kpc = compute_gbar_at_radius(2.0, R, V_bar)
    gbar_5kpc = compute_gbar_at_radius(5.0, R, V_bar)
    gbar_10kpc = compute_gbar_at_radius(10.0, R, V_bar)
    
    # Compute K at key radii (using fixed parameters for now)
    K_2kpc = compute_K_at_radii(2.0, V_bar, A=0.6, l0=5.0)[0] if not np.isnan(gbar_2kpc) else np.nan
    K_5kpc = compute_K_at_radii(5.0, V_bar, A=0.6, l0=5.0)[0] if not np.isnan(gbar_5kpc) else np.nan
    K_10kpc = compute_K_at_radii(10.0, V_bar, A=0.6, l0=5.0)[0] if not np.isnan(gbar_10kpc) else np.nan
    
    # RAR metrics
    rar_bias, rar_scatter = compute_rar_metrics(V_obs, V_bar, eV_obs)
    
    # BTFR: Baryonic Tully-Fisher
    # log(Mbar) vs log(Vf) - residual from expected relation
    if np.isfinite(Mbar) and np.isfinite(Vf) and Mbar > 0 and Vf > 0:
        log_Mbar = np.log10(Mbar * 1e9)  # Convert to M_sun
        log_Vf = np.log10(Vf)
        # Canonical BTFR: log(Mbar) = 3.5*log(Vf) + const
        # Residual = observed - expected
        btfr_expected = 3.5 * log_Vf - 2.5  # Rough calibration
        btfr_residual = log_Mbar - btfr_expected
    else:
        btfr_residual = np.nan
    
    # Build feature dict (matching template columns)
    features = {
        'galaxy_id': name,
        't_type': meta_row.get('T', np.nan),
        'is_barred': 0,  # Not in SPARC metadata, default to 0
        'incl_deg': meta_row.get('Inc', np.nan),
        'r_d_kpc': Rd,
        'm_star_1e10Msun': Mbar * 0.5 / 10 if np.isfinite(Mbar) else np.nan,  # Rough M_*/Mbar ~ 0.5
        'm_gas_1e10Msun': Mbar * 0.5 / 10 if np.isfinite(Mbar) else np.nan,  # Rough M_gas/Mbar ~ 0.5
        'gbar_2kpc_mps2': gbar_2kpc,
        'gbar_5kpc_mps2': gbar_5kpc,
        'gbar_10kpc_mps2': gbar_10kpc,
        'K_2kpc': K_2kpc,
        'K_5kpc': K_5kpc,
        'K_10kpc': K_10kpc,
        'Gbulge_mean_inner': 1.0,  # Placeholder - would need gate calculation
        'Gbar_mean_inner': 1.0,  # Placeholder
        'Gshear_mean_inner': 1.0,  # Placeholder
        'rar_bias_dex': rar_bias,
        'rar_scatter_dex': rar_scatter,
        'btfr_residual_dex': btfr_residual,
        'v_flat_kms': Vf
    }
    
    return features

def main():
    print("=" * 70)
    print("BUILDING PARAMETER-SPACE PCA FEATURES")
    print("=" * 70)
    
    # Paths
    repo_root = Path(__file__).parent.parent.parent
    curves_dir = repo_root / 'pca' / 'data' / 'raw' / 'sparc_curves'
    meta_file = repo_root / 'pca' / 'data' / 'raw' / 'metadata' / 'sparc_meta.csv'
    output_dir = repo_root / 'pca' / 'outputs' / 'parameter_space'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    meta = pd.read_csv(meta_file)
    
    print(f"\nProcessing {len(meta)} galaxies...")
    
    # Build features for each galaxy
    galaxy_features = []
    
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Building features"):
        name = row['name']
        curve_file = curves_dir / f"{name}.csv"
        
        if not curve_file.exists():
            continue
        
        try:
            curve = pd.read_csv(curve_file)
            features = build_galaxy_features(curve, row, name)
            galaxy_features.append(features)
        except Exception as e:
            continue
    
    # Save
    features_df = pd.DataFrame(galaxy_features)
    output_file = output_dir / 'sparc_parameter_features.csv'
    features_df.to_csv(output_file, index=False)
    
    print(f"\nBuilt features for {len(features_df)} galaxies")
    print(f"Saved to: {output_file}")
    
    print(f"\nFeature columns ({len(features_df.columns)}):")
    for col in features_df.columns:
        n_valid = features_df[col].notna().sum()
        print(f"  {col:25s}: {n_valid}/{len(features_df)} valid")
    
    print("\n" + "=" * 70)
    print("READY FOR PARAMETER-SPACE PCA")
    print("=" * 70)
    print("\nThis PCA will reveal:")
    print("  - Which kernel features (K at different R) cluster together")
    print("  - What drives RAR/BTFR performance")
    print("  - Whether outcomes correlate with specific PC axes")
    print("\nNext: Run parameter-space PCA on this data")
    print(f"      python pca/scripts/17_run_parameter_pca.py")

if __name__ == '__main__':
    main()

