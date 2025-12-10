"""
Tune cluster parameters to get F~5 (not F~25!)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gravitational_channeling import ChannelingParams, gravitational_channeling


def test_coma(chi_0, zeta, D_max):
    """Test enhancement for Coma cluster."""
    R_E = 300
    sigma_v = 1000
    M_bary = 1.4e14
    M_lens = 7e14
    R_half = 500
    F_needed = M_lens / M_bary  # 5.0
    
    Sigma_0 = M_bary / (4 * np.pi * (R_half * 1e3)**2)
    Sigma = Sigma_0 * (R_half / R_E)
    
    params = ChannelingParams(
        chi_0=chi_0, alpha=1.0, beta=0.5, gamma=0.3, epsilon=0.3,
        zeta=zeta, D_max=D_max, N_crit=1000.0, use_winding=False, t_age=13.0
    )
    
    F, _ = gravitational_channeling(
        R=np.array([R_E]),
        v_bary=np.array([sigma_v]),
        sigma_v=np.array([sigma_v]),
        Sigma=np.array([Sigma]),
        params=params
    )
    
    return F[0], F_needed


print("Tuning cluster parameters to get F~5.0 for Coma")
print("=" * 60)
print(f"{'χ₀':<8} {'ζ':<8} {'D_max':<8} {'F':<10} {'F_need':<10} {'%':<10}")
print("-" * 60)

# Grid search
for chi_0 in [0.5, 0.8, 1.0, 1.2, 1.5]:
    for zeta in [0.3, 0.4, 0.5]:
        for D_max in [3.0, 5.0, 8.0, 10.0]:
            F, F_need = test_coma(chi_0, zeta, D_max)
            ratio = F / F_need * 100
            if 90 < ratio < 120:  # Close to 100%
                print(f"{chi_0:<8.1f} {zeta:<8.1f} {D_max:<8.1f} {F:<10.2f} {F_need:<10.1f} {ratio:<10.1f}% ← GOOD")
            elif 70 < ratio < 130:
                print(f"{chi_0:<8.1f} {zeta:<8.1f} {D_max:<8.1f} {F:<10.2f} {F_need:<10.1f} {ratio:<10.1f}%")

print("\n" + "=" * 60)
print("Testing best candidates on all clusters")
print("=" * 60)

# Best candidate from above (will adjust)
best_params = [
    (0.8, 0.3, 5.0),
    (0.5, 0.4, 8.0),
    (0.5, 0.3, 10.0),
]

CLUSTERS = {
    'Coma': (300, 1000, 1.4e14, 7e14, 500),
    'A2029': (200, 850, 1.0e14, 5e14, 300),
    'A1689': (250, 900, 1.2e14, 6e14, 400),
    'Bullet': (300, 1100, 1.5e14, 9e14, 500),
}

for chi_0, zeta, D_max in best_params:
    print(f"\nχ₀={chi_0}, ζ={zeta}, D_max={D_max}")
    print("-" * 50)
    
    params = ChannelingParams(
        chi_0=chi_0, alpha=1.0, beta=0.5, gamma=0.3, epsilon=0.3,
        zeta=zeta, D_max=D_max, N_crit=1000.0, use_winding=False, t_age=13.0
    )
    
    ratios = []
    for name, (R_E, sigma_v, M_bary, M_lens, R_half) in CLUSTERS.items():
        F_needed = M_lens / M_bary
        Sigma = M_bary / (4 * np.pi * (R_half * 1e3)**2) * (R_half / R_E)
        
        F, _ = gravitational_channeling(
            R=np.array([R_E]),
            v_bary=np.array([sigma_v]),
            sigma_v=np.array([sigma_v]),
            Sigma=np.array([Sigma]),
            params=params
        )
        ratio = F[0] / F_needed * 100
        ratios.append(ratio)
        print(f"  {name}: F={F[0]:.2f}, need={F_needed:.1f}, {ratio:.1f}%")
    
    print(f"  Average: {np.mean(ratios):.1f}%")
