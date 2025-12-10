"""
Cooperative Channeling: Local Density Enhancement
=================================================

Physical idea: Channels form through COLLECTIVE field-line organization.
More nearby stars → more field lines → stronger channel formation.

New formula:
    F = 1 + χ₀ × (Σ_tot/Σ_ref)^ε × (ρ_local/ρ_ref)^ζ × D(R) / (1 + D/D_max)

Key prediction: Declining ρ_local in massive spiral outskirts
naturally suppresses enhancement there — MORPHOLOGY-DEPENDENT without
explicit classification!
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import os
import glob


@dataclass
class CooperativeParams:
    """Parameters for cooperative channeling model."""
    
    # Channel growth exponents
    alpha: float = 1.0       # Channel scale grows with R
    beta: float = 0.5        # Cold systems carve deeper
    gamma: float = 0.3       # Sublinear time accumulation
    
    # Coupling constants
    chi_0: float = 0.3       # Base coupling strength
    epsilon: float = 0.3     # Surface density exponent (Solar System safety)
    zeta: float = 0.3        # NEW: Local density cooperation exponent
    
    # Saturation
    D_max: float = 3.0       # Maximum channel depth
    
    # Time scales
    t_age: float = 10.0      # System age [Gyr]
    tau_0: float = 1.0       # Reference channel formation time [Gyr]
    
    # Reference scales
    Sigma_ref: float = 100.0   # Reference surface density [M_sun/pc^2]
    rho_ref: float = 0.05      # Reference local 3D density [M_sun/pc^3]
    sigma_ref: float = 30.0    # Reference velocity dispersion [km/s]
    R_0: float = 8.0           # Reference radius [kpc]
    scale_height: float = 0.3  # Disk scale height [kpc]


def estimate_local_density(Sigma: np.ndarray, scale_height: float = 0.3) -> np.ndarray:
    """
    Estimate local 3D stellar density from surface density.
    
    ρ_local ≈ Σ / (2 × h_z)
    
    where h_z is the disk scale height (~0.3 kpc for thin disk).
    """
    # Convert Sigma [M_sun/pc^2] to rho [M_sun/pc^3]
    # h_z in pc = scale_height * 1000
    h_z_pc = scale_height * 1000.0
    rho = Sigma / (2.0 * h_z_pc)
    return np.maximum(rho, 1e-15)


def cooperative_enhancement(R: np.ndarray, v_c: np.ndarray, Sigma: np.ndarray,
                            sigma_v: np.ndarray, params: CooperativeParams,
                            diagnostics: Optional[Dict] = None) -> np.ndarray:
    """
    Compute cooperative channeling enhancement.
    
    F = 1 + χ₀ × (Σ/Σ_ref)^ε × (ρ_local/ρ_ref)^ζ × D(R) / (1 + D/D_max)
    """
    # Ensure arrays
    R = np.atleast_1d(np.asarray(R, dtype=float))
    v_c = np.atleast_1d(np.asarray(v_c, dtype=float))
    Sigma = np.atleast_1d(np.asarray(Sigma, dtype=float))
    sigma_v = np.atleast_1d(np.asarray(sigma_v, dtype=float))
    
    # Protect against edge cases
    R_safe = np.maximum(R, 1e-10)
    sigma_safe = np.maximum(sigma_v, 0.01)
    Sigma_safe = np.maximum(Sigma, 1e-15)
    
    # 1. Channel formation time
    tau_ch = params.tau_0 * (sigma_safe / params.sigma_ref) * (params.R_0 / R_safe)
    tau_ch = np.maximum(tau_ch, 0.01)
    
    # 2. Time accumulation term
    time_term = (params.t_age / tau_ch) ** params.gamma
    
    # 3. Velocity coherence term
    coherence_term = (v_c / sigma_safe) ** params.beta
    
    # 4. Radial growth term
    radial_term = (R_safe / params.R_0) ** params.alpha
    
    # 5. Raw channel depth
    D_raw = time_term * coherence_term * radial_term
    
    # 6. Saturated channel depth
    D = D_raw / (1.0 + D_raw / params.D_max)
    
    # 7. Total surface density factor (Solar System safety)
    total_density_factor = (Sigma_safe / params.Sigma_ref) ** params.epsilon
    total_density_factor = np.minimum(total_density_factor, 5.0)  # Cap
    
    # 8. LOCAL density factor (NEW - cooperative effect)
    rho_local = estimate_local_density(Sigma_safe, params.scale_height)
    local_density_factor = (rho_local / params.rho_ref) ** params.zeta
    local_density_factor = np.minimum(local_density_factor, 5.0)  # Cap
    
    # 9. Full enhancement
    enhancement = params.chi_0 * total_density_factor * local_density_factor * D
    F = 1.0 + enhancement
    
    if diagnostics is not None:
        diagnostics['D'] = D
        diagnostics['D_raw'] = D_raw
        diagnostics['Sigma'] = Sigma_safe
        diagnostics['rho_local'] = rho_local
        diagnostics['total_density_factor'] = total_density_factor
        diagnostics['local_density_factor'] = local_density_factor
        diagnostics['enhancement'] = enhancement
        diagnostics['tau_ch'] = tau_ch
    
    return F


def test_solar_system(params: CooperativeParams) -> Tuple[float, bool]:
    """Test Cassini constraint."""
    R_saturn_kpc = 9.5 * 4.85e-9
    v_saturn = 9.7
    Sigma_ss = 1e-20  # Point mass → zero surface density
    sigma_ss = 0.1
    
    F = cooperative_enhancement(
        R=np.array([R_saturn_kpc]),
        v_c=np.array([v_saturn]),
        Sigma=np.array([Sigma_ss]),
        sigma_v=np.array([sigma_ss]),
        params=params
    )[0]
    
    delta_g = F - 1.0
    cassini_limit = 2.3e-5
    
    return delta_g, abs(delta_g) < cassini_limit


# =============================================================================
# SPARC Integration
# =============================================================================

def load_sparc_galaxy(filepath: str) -> Dict:
    """Load a single SPARC galaxy rotation curve."""
    data = {
        'R': [], 'v_obs': [], 'v_err': [],
        'v_gas': [], 'v_disk': [], 'v_bul': [],
        'SBdisk': [], 'SBbul': []
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 6:
                data['R'].append(float(parts[0]))
                data['v_obs'].append(float(parts[1]))
                data['v_err'].append(float(parts[2]))
                data['v_gas'].append(float(parts[3]))
                data['v_disk'].append(float(parts[4]))
                data['v_bul'].append(float(parts[5]))
                if len(parts) >= 8:
                    data['SBdisk'].append(float(parts[6]))
                    data['SBbul'].append(float(parts[7]))
    
    for key in data:
        data[key] = np.array(data[key])
    
    # Compute total baryonic velocity
    v_gas = data['v_gas']
    v_disk = data['v_disk']
    v_bul = data['v_bul']
    
    v_gas_sq = np.sign(v_gas) * v_gas**2
    v_disk_sq = np.sign(v_disk) * v_disk**2
    v_bul_sq = v_bul**2
    
    v_bary_sq = v_gas_sq + v_disk_sq + v_bul_sq
    data['v_bary'] = np.sign(v_bary_sq) * np.sqrt(np.abs(v_bary_sq))
    
    # Surface density from surface brightness
    if len(data['SBdisk']) > 0 and np.any(data['SBdisk'] > 0):
        ML_disk = 0.5
        ML_bul = 0.7
        data['Sigma'] = ML_disk * data['SBdisk'] + ML_bul * data['SBbul']
    else:
        # Estimate from velocity
        G = 4.302e-6
        R_safe = np.maximum(data['R'], 0.1)
        v_bary = np.abs(data['v_bary'])
        M_enc = v_bary**2 * R_safe / G
        data['Sigma'] = M_enc / (np.pi * (R_safe * 1e3)**2)
    
    data['Sigma'] = np.maximum(data['Sigma'], 0.1)
    data['name'] = os.path.basename(filepath).replace('_rotmod.dat', '')
    
    return data


def estimate_sigma_v(R: np.ndarray, v_c: np.ndarray, is_gas_dominated: bool = False) -> np.ndarray:
    """Estimate velocity dispersion."""
    if is_gas_dominated:
        return np.full_like(R, 10.0)
    
    R_inner = 3.0
    R_outer = 10.0
    f_inner = 0.3
    f_outer = 0.1
    
    R_safe = np.maximum(R, 0.1)
    x = np.clip((R_safe - R_inner) / (R_outer - R_inner), 0, 1)
    factor = f_inner + (f_outer - f_inner) * x
    
    sigma_v = v_c * factor
    return np.maximum(sigma_v, 8.0)


def fit_galaxy(data: Dict, params: CooperativeParams) -> Dict:
    """Fit cooperative channeling to a single galaxy."""
    R = data['R']
    v_obs = data['v_obs']
    v_err = data['v_err']
    v_bary = np.abs(data['v_bary'])
    Sigma = data['Sigma']
    
    # Estimate velocity dispersion
    is_gas_dom = np.mean(np.abs(data['v_gas'][-3:])) > np.mean(np.abs(data['v_disk'][-3:]))
    sigma_v = estimate_sigma_v(R, v_bary, is_gas_dominated=is_gas_dom)
    
    # Compute enhancement
    diagnostics = {}
    F = cooperative_enhancement(R, v_bary, Sigma, sigma_v, params, diagnostics)
    
    # Predicted velocity
    v_pred = v_bary * np.sqrt(F)
    
    # Metrics
    rms_pred = np.sqrt(np.mean((v_pred - v_obs)**2))
    rms_bary = np.sqrt(np.mean((v_bary - v_obs)**2))
    
    return {
        'name': data.get('name', 'unknown'),
        'R': R,
        'v_obs': v_obs,
        'v_bary': v_bary,
        'v_pred': v_pred,
        'F': F,
        'Sigma': Sigma,
        'rho_local': diagnostics['rho_local'],
        'rms_pred': rms_pred,
        'rms_bary': rms_bary,
        'delta_rms': rms_pred - rms_bary,
        'improved': rms_pred < rms_bary,
        'diagnostics': diagnostics,
    }


if __name__ == "__main__":
    print("Cooperative Channeling - Sanity Test")
    print("=" * 50)
    
    params = CooperativeParams(zeta=0.3)
    
    # Solar System
    delta_g, passes = test_solar_system(params)
    print(f"\nSolar System (ζ={params.zeta}):")
    print(f"  δg/g = {delta_g:.2e}")
    print(f"  PASSES: {passes}")
    
    # Test with declining Sigma (like massive spiral outer disk)
    R = np.array([2, 5, 10, 15, 20])
    v_c = np.array([200, 210, 220, 215, 210])
    Sigma = np.array([300, 100, 30, 10, 5])  # Exponential decline
    sigma_v = np.array([50, 30, 20, 18, 15])
    
    diag = {}
    F = cooperative_enhancement(R, v_c, Sigma, sigma_v, params, diag)
    
    print(f"\nMassive spiral test (declining Σ):")
    print(f"{'R':>5} {'Σ':>8} {'ρ_loc':>10} {'loc_fac':>8} {'F':>8}")
    print("-" * 45)
    for i in range(len(R)):
        print(f"{R[i]:5.0f} {Sigma[i]:8.1f} {diag['rho_local'][i]:10.6f} "
              f"{diag['local_density_factor'][i]:8.4f} {F[i]:8.4f}")
    
    print(f"\nF(inner) / F(outer) = {F[0]:.3f} / {F[-1]:.3f} = {F[0]/F[-1]:.2f}×")
    print("↑ Local density factor naturally suppresses outer disk enhancement!")
