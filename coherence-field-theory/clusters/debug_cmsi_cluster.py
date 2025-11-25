"""Debug CMSI cluster calculation."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from galaxies.cmsi_kernel import CMSIParams, cmsi_enhancement, G_NEWTON

# Coma cluster
R = 300.0
M_bary = 1.4e14
v_circ = np.sqrt(G_NEWTON * M_bary / R)
sigma_v = 1000.0
Sigma = 200.0

params = CMSIParams(chi_0=500, alpha_Ncoh=0.45, ell_0_kpc=3.0, include_K_rough=True)

R_arr = np.array([R])
v_arr = np.array([v_circ])
sigma_arr = np.array([sigma_v])
Sigma_arr = np.array([Sigma])

F, diag = cmsi_enhancement(R_arr, v_arr, sigma_arr, params, Sigma_arr)

print("Coma debug:")
print(f"  v_circ = {v_circ:.1f} km/s")
print(f"  F_CMSI total = {F[0]:.4f}")
print(f"  F_cmsi_core = {diag['F_cmsi_core'][0]:.4f}")
print(f"  K_rough = {diag['K_rough'][0]:.4f}")
print(f"  coherent_amplitude = {diag['coherent_amplitude'][0]:.6f}")
print(f"  source_factor = {diag['source_factor'][0]:.4f}")
print(f"  N_coh = {diag['N_coh'][0]:.4f}")
print(f"  f_profile = {diag['f_profile'][0]:.6f}")
print(f"  (v/c)^2 = {diag['v_over_c_squared'][0]:.2e}")

# Breakdown
chi_0 = params.chi_0
v_c2 = diag['v_over_c_squared'][0]
sf = diag['source_factor'][0]
N_coh = diag['N_coh'][0]
f_prof = diag['f_profile'][0]
alpha = params.alpha_Ncoh

expected_amp = chi_0 * v_c2 * sf * (N_coh ** alpha) * f_prof
print(f"\nManual amplitude calculation:")
print(f"  chi_0 * (v/c)^2 = {chi_0 * v_c2:.4f}")
print(f"  * source_factor = {chi_0 * v_c2 * sf:.4f}")
print(f"  * N_coh^alpha = {chi_0 * v_c2 * sf * (N_coh**alpha):.4f}")
print(f"  * f_profile = {expected_amp:.4f}")
print(f"  Expected F_core = 1 + {expected_amp:.4f} = {1 + expected_amp:.4f}")

print(f"\nF_total = F_core * (1 + K_rough)")
print(f"        = {diag['F_cmsi_core'][0]:.4f} * (1 + {diag['K_rough'][0]:.4f})")
print(f"        = {diag['F_cmsi_core'][0] * (1 + diag['K_rough'][0]):.4f}")
