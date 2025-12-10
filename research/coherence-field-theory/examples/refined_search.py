import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from examples.grid_search_gpm import grid_search
import time

# Refined search: stronger mass suppression
fixed_params = {
    'ell0_kpc': 1.0,      # Smaller coherence length
    'Qstar': 2.0,
    'sigmastar': 25.0,
    'nQ': 2.0,
    'nsig': 2.0,
    'p': 0.5,
    'nM': 2.5             # Stronger mass suppression (was 2.0)
}

test_galaxies = ['DDO154', 'DDO170', 'IC2574', 'NGC2403', 'NGC6503', 'NGC3198', 'UGC02259']

# Focus on higher Mstar
alpha0_range = [0.10, 0.15, 0.20, 0.25]
Mstar_range = [1e9, 2e9, 5e9, 1e10]

print('Refined search: nM=2.5 (stronger mass gate)')
print(f'alpha0: {alpha0_range}')
print(f'Mstar: {[f"{m:.1e}" for m in Mstar_range]}')
print()

start = time.time()
results_df = grid_search(test_galaxies, alpha0_range, Mstar_range, fixed_params)
elapsed = time.time() - start

print(f'\nCompleted in {elapsed:.1f}s')

# Find best
best = results_df.sort_values(['success_rate', 'mean_improvement'], ascending=[False, False]).iloc[0]
print(f'\nBEST: alpha0={best["alpha0"]:.2f}, M*={best["Mstar"]:.1e} Msun')
print(f'Success: {best["n_improved"]:.0f}/{best["n_galaxies"]:.0f} ({best["success_rate"]:.0f}%)')
print(f'Mean improvement: {best["mean_improvement"]:+.1f}%')
print(f'Median improvement: {best["median_improvement"]:+.1f}%')

# Show which galaxies improved
print('\nPer-galaxy:')
for gal in best['galaxy_results']:
    status = '✓' if gal['improvement'] > 0 else '✗'
    print(f"  {status} {gal['name']}: {gal['improvement']:+.1f}%")
