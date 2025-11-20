import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
import numpy as np

loader = RealDataLoader()
gal = loader.load_rotmod_galaxy('DDO154')

v_bar = np.sqrt(gal['v_disk']**2 + gal['v_gas']**2)
v_obs = gal['v_obs']

print(f"DDO154:")
print(f"  v_obs: {v_obs.min():.1f}-{v_obs.max():.1f} km/s")
print(f"  v_bar (SPARC): {v_bar.min():.1f}-{v_bar.max():.1f} km/s")
print(f"  Ratio v_obs/v_bar (outer): {v_obs[-1]/v_bar[-1]:.2f}")
print(f"  Missing mass factor: {(v_obs[-1]/v_bar[-1])**2:.2f}x")
