from batch_gpm_test import test_galaxy_gpm

gpm_params = {
    'alpha0': 0.3,
    'ell0_kpc': 2.0,
    'Qstar': 2.0,
    'sigmastar': 25.0,
    'nQ': 2.0,
    'nsig': 2.0,
    'p': 0.5,
    'Mstar_Msun': 2e8,
    'nM': 1.5
}

result = test_galaxy_gpm('DDO154', gpm_params)

print("DDO154 batch test results:")
print(f"  chi2_baryon: {result['chi2_baryon']:.1f}")
print(f"  chi2_gpm: {result['chi2_gpm']:.1f}")
print(f"  improvement: {result['improvement']:.1f}%")
print(f"  alpha: {result['alpha_eff']:.3f}")
print(f"  ell: {result['ell']:.2f}")
print(f"  M_total: {result['M_total']:.2e}")
print(f"  sigma_v: {result['sigma_v']:.2f}")
