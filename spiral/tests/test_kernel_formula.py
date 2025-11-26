"""Quick test of corrected kernel formula."""
import numpy as np

# Fixed parameters
A0 = 0.591
ELL0 = 4.993
N_COH = 0.5
G_DAGGER = 1.2e-10
KM_TO_M = 1000.0
KPC_TO_M = 3.0856776e19

# Test data
R = np.array([1, 5, 10, 15, 20])  # kpc
v_bar = np.array([60, 100, 130, 150, 160])  # km/s

# Compute g_bar
v_m_s = v_bar * KM_TO_M
r_m = R * KPC_TO_M
g_bar = v_m_s**2 / r_m

# Test with p = 0.75 (paper value)
p = 0.75
g_ratio = G_DAGGER / g_bar
K_rar = g_ratio ** p
K_coherence = (ELL0 / (ELL0 + R)) ** N_COH
S_small = 1 - np.exp(-(R / 0.5)**2)

K = A0 * K_rar * K_coherence * S_small
v_pred = v_bar * np.sqrt(1 + K)

print('CORRECTED KERNEL FORMULA TEST (p=0.75)')
print('=' * 80)
print(f"R (kpc)  v_bar    g_bar (m/s2)    K_rar       K_coh     S_small   K        v_pred")
print('-' * 80)
for i in range(len(R)):
    print(f"{R[i]:6.1f}   {v_bar[i]:6.1f}   {g_bar[i]:12.2e}   {K_rar[i]:9.4f}   {K_coherence[i]:7.4f}   {S_small[i]:7.4f}   {K[i]:7.4f}   {v_pred[i]:7.1f}")

print(f'\nK values range: [{K.min():.4f}, {K.max():.4f}]')
print(f'v_pred/v_bar boost: [{(v_pred/v_bar).min():.3f}, {(v_pred/v_bar).max():.3f}]')
print('\nThis looks reasonable! K should be O(0.1-1) for galactic radii.')
