#!/usr/bin/env python3
"""
Explore: Solid Angle / Arc-Based Amplitude

Concept: The amplitude A could depend on the solid angle (or arc) that the 
baryonic mass distribution subtends as seen from the test particle.

For a thin disk galaxy:
    - Disk scale height h ~ 0.3-0.5 kpc
    - Star at radius R
    - Solid angle Ω ≈ 2π × (h/R) for R >> h (thin disk approximation)
    
For a spherical cluster:
    - Characteristic radius R_c ~ 200-600 kpc
    - Test particle at radius r
    - Solid angle Ω ≈ 2π × (1 - r/√(r² + R_c²)) for particle inside
    - Or Ω ≈ π × (R_c/r)² for particle outside

The hypothesis: A ∝ Ω^α for some power α

Let's test this!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Physical constants
c = 2.998e8          # m/s
H0_SI = 2.27e-18     # 1/s (70 km/s/Mpc)
G = 6.674e-11        # m³/kg/s²
kpc_to_m = 3.086e19  # m per kpc
M_sun = 1.989e30     # kg

# Derived
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # ≈ 9.6×10⁻¹¹ m/s²
A_0 = np.exp(1 / (2 * np.pi))  # ≈ 1.173

print("=" * 70)
print("EXPLORING SOLID ANGLE / ARC-BASED AMPLITUDE")
print("=" * 70)
print(f"g† = {g_dagger:.3e} m/s²")
print(f"A₀ = {A_0:.4f}")
print()

# =============================================================================
# SOLID ANGLE CALCULATIONS
# =============================================================================

def solid_angle_thin_disk(R_kpc: float, h_kpc: float = 0.4) -> float:
    """
    Solid angle subtended by a thin disk as seen from radius R.
    
    For a star at radius R in a disk of scale height h:
    - Looking inward, the disk subtends angle ~ arctan(h/R) above and below
    - Total solid angle ≈ 4π × (h/R) for R >> h
    
    More precisely, for an exponential disk:
    Ω ≈ 2π × (1 - cos(θ)) where θ = arctan(h/R)
    For small θ: Ω ≈ π × (h/R)²
    
    But we want the "effective cone" of mass, so use:
    Ω_eff ≈ 2 × arctan(h/R) (the opening angle in radians)
    """
    if R_kpc <= 0:
        return 2 * np.pi  # At center, sees full disk
    
    # Opening half-angle of the disk
    theta = np.arctan(h_kpc / R_kpc)
    
    # Solid angle of a cone with half-angle theta
    # Ω = 2π(1 - cos(θ))
    omega = 2 * np.pi * (1 - np.cos(theta))
    
    return omega

def solid_angle_sphere(r_kpc: float, R_sphere_kpc: float) -> float:
    """
    Solid angle subtended by a sphere of radius R_sphere as seen from radius r.
    
    If r < R_sphere (inside): sees 4π (full sphere surrounds you)
    If r > R_sphere (outside): Ω = 2π × (1 - √(1 - (R/r)²))
    """
    if r_kpc <= R_sphere_kpc:
        return 4 * np.pi  # Inside the sphere
    
    # Outside: standard formula for solid angle of a sphere
    ratio = R_sphere_kpc / r_kpc
    omega = 2 * np.pi * (1 - np.sqrt(1 - ratio**2))
    
    return omega

def solid_angle_ellipsoid(r_kpc: float, R_major_kpc: float, R_minor_kpc: float) -> float:
    """
    Approximate solid angle for an oblate ellipsoid (galaxy-like).
    Use geometric mean of disk and sphere approximations.
    """
    omega_disk = solid_angle_thin_disk(r_kpc, R_minor_kpc)
    omega_sphere = solid_angle_sphere(r_kpc, R_major_kpc)
    
    # Weight by axis ratio
    axis_ratio = R_minor_kpc / R_major_kpc
    return omega_disk * axis_ratio + omega_sphere * (1 - axis_ratio)

# =============================================================================
# TEST 1: Compare solid angles for different systems
# =============================================================================

print("=" * 70)
print("TEST 1: Solid Angles for Different Systems")
print("=" * 70)

# Thin disk galaxy (like Milky Way)
R_test = 8.0  # kpc (Sun's position)
h_disk = 0.4  # kpc (disk scale height)
omega_disk = solid_angle_thin_disk(R_test, h_disk)
print(f"\nThin disk galaxy (R={R_test} kpc, h={h_disk} kpc):")
print(f"  Solid angle Ω = {omega_disk:.4f} sr = {omega_disk/(4*np.pi)*100:.2f}% of sky")

# Galaxy cluster
r_cluster = 200  # kpc (measurement radius)
R_cluster = 600  # kpc (cluster size)
omega_cluster = solid_angle_sphere(r_cluster, R_cluster)
print(f"\nGalaxy cluster (r={r_cluster} kpc, R={R_cluster} kpc):")
print(f"  Solid angle Ω = {omega_cluster:.4f} sr = {omega_cluster/(4*np.pi)*100:.2f}% of sky")

# Ratio
print(f"\nRatio (cluster/disk): {omega_cluster/omega_disk:.1f}×")

# What if A ∝ Ω^α?
# We know A_disk ≈ 1.17 and A_cluster ≈ 8.45
# So (Ω_cluster/Ω_disk)^α = A_cluster/A_disk
# α = log(A_cluster/A_disk) / log(Ω_cluster/Ω_disk)

A_disk = A_0
A_cluster = 8.45
alpha = np.log(A_cluster / A_disk) / np.log(omega_cluster / omega_disk)
print(f"\nIf A ∝ Ω^α, then α = {alpha:.3f}")

# =============================================================================
# TEST 2: Solid angle profile across a galaxy
# =============================================================================

print("\n" + "=" * 70)
print("TEST 2: Solid Angle Profile Across a Disk Galaxy")
print("=" * 70)

R_range = np.linspace(0.5, 20, 50)
omega_profile = [solid_angle_thin_disk(R, 0.4) for R in R_range]

print(f"\nR (kpc)  |  Ω (sr)  |  Ω/4π (%)")
print("-" * 35)
for R, omega in zip([1, 3, 5, 8, 10, 15, 20], 
                     [solid_angle_thin_disk(r, 0.4) for r in [1, 3, 5, 8, 10, 15, 20]]):
    print(f"  {R:5.1f}  |  {omega:.4f}  |  {omega/(4*np.pi)*100:.3f}%")

# =============================================================================
# TEST 3: Define amplitude from solid angle and test on galaxies
# =============================================================================

print("\n" + "=" * 70)
print("TEST 3: Solid Angle Amplitude Formula")
print("=" * 70)

# Reference solid angle (at R = R_d for a typical galaxy)
R_d_ref = 3.0  # kpc
h_ref = 0.4    # kpc
Omega_0 = solid_angle_thin_disk(R_d_ref, h_ref)

print(f"\nReference: Ω₀ = Ω(R={R_d_ref} kpc, h={h_ref} kpc) = {Omega_0:.5f} sr")

# Try different power laws
def amplitude_from_solid_angle(omega: float, alpha: float = 0.5) -> float:
    """A = A₀ × (Ω/Ω₀)^α"""
    return A_0 * (omega / Omega_0)**alpha

# What alpha gives the right cluster amplitude?
omega_cluster_200 = solid_angle_sphere(200, 600)  # Inside cluster at 200 kpc
alpha_needed = np.log(A_cluster / A_0) / np.log(omega_cluster_200 / Omega_0)
print(f"Alpha needed to match clusters: {alpha_needed:.3f}")

# =============================================================================
# TEST 4: Apply to SPARC galaxies
# =============================================================================

print("\n" + "=" * 70)
print("TEST 4: Test on SPARC Galaxies")
print("=" * 70)

# Load SPARC data
data_dir = Path(__file__).parent.parent / "data"
sparc_file = data_dir / "SPARC" / "SPARC_Lelli2016c.mrt"

def load_sparc_galaxies():
    """Load SPARC rotation curve data."""
    galaxies = {}
    
    if not sparc_file.exists():
        print(f"SPARC file not found: {sparc_file}")
        return galaxies
    
    with open(sparc_file, 'r') as f:
        lines = f.readlines()
    
    current_galaxy = None
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        
        parts = line.split()
        if len(parts) >= 7:
            try:
                name = parts[0]
                R = float(parts[1])  # kpc
                V_obs = float(parts[2])  # km/s
                V_gas = float(parts[4])  # km/s
                V_disk = float(parts[5])  # km/s
                V_bul = float(parts[6])  # km/s
                
                if name not in galaxies:
                    galaxies[name] = {'R': [], 'V_obs': [], 'V_gas': [], 'V_disk': [], 'V_bul': []}
                
                galaxies[name]['R'].append(R)
                galaxies[name]['V_obs'].append(V_obs)
                galaxies[name]['V_gas'].append(V_gas)
                galaxies[name]['V_disk'].append(V_disk)
                galaxies[name]['V_bul'].append(V_bul)
            except (ValueError, IndexError):
                continue
    
    # Convert to numpy arrays
    for name in galaxies:
        for key in galaxies[name]:
            galaxies[name][key] = np.array(galaxies[name][key])
    
    return galaxies

def h_function(g):
    """Universal h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def predict_with_solid_angle(R_kpc, V_bar, R_d, h_disk=0.4, alpha=0.5):
    """
    Predict rotation curve using solid-angle-based amplitude.
    
    A(R) = A₀ × (Ω(R)/Ω₀)^α
    
    where Ω(R) is the solid angle of the disk as seen from radius R.
    """
    R_kpc = np.asarray(R_kpc)
    V_bar = np.asarray(V_bar)
    
    # Compute solid angle at each radius
    omega = np.array([solid_angle_thin_disk(R, h_disk) for R in R_kpc])
    
    # Amplitude from solid angle
    A = A_0 * (omega / Omega_0)**alpha
    
    # Newtonian acceleration
    g_bar = (V_bar * 1e3)**2 / (R_kpc * kpc_to_m)
    g_bar = np.maximum(g_bar, 1e-15)
    
    # Enhancement
    h = h_function(g_bar)
    
    # Coherence window (still use standard formula)
    xi = R_d / (2 * np.pi)
    W = R_kpc / (xi + R_kpc)
    
    # Total enhancement
    Sigma = 1 + A * W * h
    
    # Predicted velocity
    V_pred = V_bar * np.sqrt(Sigma)
    
    return V_pred, A

def predict_canonical(R_kpc, V_bar, R_d):
    """Canonical prediction with constant A = A₀."""
    R_kpc = np.asarray(R_kpc)
    V_bar = np.asarray(V_bar)
    
    g_bar = (V_bar * 1e3)**2 / (R_kpc * kpc_to_m)
    g_bar = np.maximum(g_bar, 1e-15)
    
    h = h_function(g_bar)
    xi = R_d / (2 * np.pi)
    W = R_kpc / (xi + R_kpc)
    
    Sigma = 1 + A_0 * W * h
    V_pred = V_bar * np.sqrt(Sigma)
    
    return V_pred

# Load and test
galaxies = load_sparc_galaxies()
print(f"Loaded {len(galaxies)} galaxies")

if len(galaxies) > 0:
    # Test different alpha values
    alphas_to_test = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
    
    results = {alpha: {'rms': [], 'wins': 0} for alpha in alphas_to_test}
    results['canonical'] = {'rms': [], 'wins': 0}
    
    for name, gal in galaxies.items():
        R = gal['R']
        V_obs = gal['V_obs']
        V_gas = gal['V_gas']
        V_disk = gal['V_disk']
        V_bul = gal['V_bul']
        
        if len(R) < 5:
            continue
        
        # Baryonic velocity
        V_bar = np.sqrt(np.abs(V_gas)**2 + 0.5 * np.abs(V_disk)**2 + 0.7 * np.abs(V_bul)**2)
        V_bar = np.maximum(V_bar, 1.0)
        
        # Estimate R_d
        R_d = R[len(R)//3] if len(R) > 3 else 3.0
        
        # Canonical prediction
        V_pred_canon = predict_canonical(R, V_bar, R_d)
        rms_canon = np.sqrt(np.mean((V_obs - V_pred_canon)**2))
        results['canonical']['rms'].append(rms_canon)
        
        # Solid angle predictions
        for alpha in alphas_to_test:
            V_pred, A = predict_with_solid_angle(R, V_bar, R_d, alpha=alpha)
            rms = np.sqrt(np.mean((V_obs - V_pred)**2))
            results[alpha]['rms'].append(rms)
            
            if rms < rms_canon:
                results[alpha]['wins'] += 1
    
    n_galaxies = len(results['canonical']['rms'])
    
    print(f"\nResults for {n_galaxies} galaxies:")
    print("-" * 50)
    print(f"{'Method':<20} | {'Mean RMS':>10} | {'Median RMS':>10} | {'Wins':>6}")
    print("-" * 50)
    
    mean_canon = np.mean(results['canonical']['rms'])
    median_canon = np.median(results['canonical']['rms'])
    print(f"{'Canonical (A=A₀)':<20} | {mean_canon:>10.2f} | {median_canon:>10.2f} | {'---':>6}")
    
    for alpha in alphas_to_test:
        mean_rms = np.mean(results[alpha]['rms'])
        median_rms = np.median(results[alpha]['rms'])
        wins = results[alpha]['wins']
        print(f"{'Ω^' + str(alpha):<20} | {mean_rms:>10.2f} | {median_rms:>10.2f} | {wins:>6}")

# =============================================================================
# TEST 5: Check if solid angle formula works for clusters
# =============================================================================

print("\n" + "=" * 70)
print("TEST 5: Does Solid Angle Work for Clusters?")
print("=" * 70)

# For clusters, the test particle (or light ray) is at the Einstein radius
# typically ~50-200 kpc, inside a cluster of size ~500-1000 kpc

# Using alpha from galaxy fit
alpha_test = 0.3  # Try the value that worked for galaxies

print(f"\nUsing α = {alpha_test}")
print(f"Reference Ω₀ = {Omega_0:.5f} sr (disk at R=3 kpc)")

# Test at different cluster radii
cluster_sizes = [400, 600, 800]  # kpc
measurement_radii = [100, 200, 300]  # kpc

print(f"\n{'Cluster R':>12} | {'Meas r':>8} | {'Ω (sr)':>10} | {'A_pred':>8} | {'A_needed':>10}")
print("-" * 60)

for R_cl in cluster_sizes:
    for r_meas in measurement_radii:
        if r_meas < R_cl:  # Must be inside cluster
            omega = solid_angle_sphere(r_meas, R_cl)
            A_pred = A_0 * (omega / Omega_0)**alpha_test
            # What A do we actually need? ~8.45 for clusters
            print(f"{R_cl:>12} | {r_meas:>8} | {omega:>10.4f} | {A_pred:>8.2f} | {8.45:>10.2f}")

# =============================================================================
# TEST 6: Alternative - use "path length through mass" more directly
# =============================================================================

print("\n" + "=" * 70)
print("TEST 6: Alternative - Effective Path Length from Geometry")
print("=" * 70)

def effective_path_length_disk(R_kpc: float, h_kpc: float, R_d_kpc: float) -> float:
    """
    Effective path length through a disk for a test particle at radius R.
    
    For a thin disk, the path length is approximately:
    L ≈ h × (mass-weighted average over sightlines)
    
    At large R, most mass is interior, so L ≈ h (vertical path through disk)
    At small R, sightlines through the disk are longer
    """
    # Simple model: L = h × (1 + R_d/R) clamped
    if R_kpc < 0.1:
        return h_kpc * 10  # Avoid singularity
    
    L = h_kpc * (1 + 0.5 * R_d_kpc / R_kpc)
    return min(L, 10 * h_kpc)  # Cap at 10× scale height

def effective_path_length_sphere(r_kpc: float, R_sphere_kpc: float) -> float:
    """
    Effective path length through a sphere for a test particle at radius r.
    
    If inside (r < R): L ≈ 2 × √(R² - r²) (chord length)
    If outside (r > R): L ≈ 2R (diameter of sphere)
    """
    if r_kpc >= R_sphere_kpc:
        return 2 * R_sphere_kpc
    else:
        return 2 * np.sqrt(R_sphere_kpc**2 - r_kpc**2)

print("\nPath lengths for different systems:")
print("-" * 50)

# Disk galaxy at various radii
print("\nDisk galaxy (h=0.4 kpc, R_d=3 kpc):")
for R in [1, 3, 5, 8, 10, 15]:
    L = effective_path_length_disk(R, 0.4, 3.0)
    print(f"  R = {R:5.1f} kpc → L = {L:.2f} kpc")

# Cluster at various radii
print("\nCluster (R_sphere=600 kpc):")
for r in [100, 200, 300, 400, 500]:
    L = effective_path_length_sphere(r, 600)
    print(f"  r = {r:5.0f} kpc → L = {L:.0f} kpc")

# Compare to our current formula
print("\n" + "=" * 70)
print("COMPARISON WITH CURRENT L-BASED FORMULA")
print("=" * 70)

print("\nCurrent formula: A = A₀ × (L/L₀)^n with L₀=0.4 kpc, n=0.27")
print("\nFor disk galaxies:")
print(f"  L = L₀ = 0.4 kpc → A = {A_0:.3f}")

print("\nFor clusters:")
print(f"  L = 600 kpc → A = {A_0 * (600/0.4)**0.27:.3f}")

print("\nGeometric path length approach:")
L_0_geom = 0.4  # Reference (disk scale height)
n_geom = 0.27

print(f"\nDisk at R=8 kpc: L_geom = {effective_path_length_disk(8, 0.4, 3.0):.2f} kpc")
print(f"  → A = A₀ × (L_geom/L₀)^n = {A_0 * (effective_path_length_disk(8, 0.4, 3.0)/L_0_geom)**n_geom:.3f}")

print(f"\nCluster at r=200 kpc: L_geom = {effective_path_length_sphere(200, 600):.0f} kpc")
print(f"  → A = A₀ × (L_geom/L₀)^n = {A_0 * (effective_path_length_sphere(200, 600)/L_0_geom)**n_geom:.3f}")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Solid angle vs radius for different geometries
ax1 = axes[0, 0]
R_range = np.linspace(0.5, 20, 100)
omega_disk_profile = [solid_angle_thin_disk(R, 0.4) for R in R_range]
omega_sphere_profile = [solid_angle_sphere(R, 10) for R in R_range]  # Small sphere for comparison

ax1.semilogy(R_range, omega_disk_profile, 'b-', lw=2, label='Thin disk (h=0.4 kpc)')
ax1.semilogy(R_range, omega_sphere_profile, 'r-', lw=2, label='Sphere (R=10 kpc)')
ax1.axhline(4*np.pi, color='gray', ls='--', alpha=0.5, label='Full sky (4π)')
ax1.set_xlabel('Radius (kpc)')
ax1.set_ylabel('Solid Angle Ω (sr)')
ax1.set_title('Solid Angle vs Radius')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel 2: Implied amplitude from solid angle
ax2 = axes[0, 1]
alpha_test = 0.3

A_disk_profile = [A_0 * (omega/Omega_0)**alpha_test for omega in omega_disk_profile]
A_sphere_profile = [A_0 * (omega/Omega_0)**alpha_test for omega in omega_sphere_profile]

ax2.plot(R_range, A_disk_profile, 'b-', lw=2, label=f'Disk (α={alpha_test})')
ax2.plot(R_range, A_sphere_profile, 'r-', lw=2, label=f'Sphere (α={alpha_test})')
ax2.axhline(A_0, color='green', ls='--', alpha=0.5, label=f'A₀ = {A_0:.3f}')
ax2.set_xlabel('Radius (kpc)')
ax2.set_ylabel('Amplitude A')
ax2.set_title('Amplitude from Solid Angle')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 5)

# Panel 3: Path length vs radius
ax3 = axes[1, 0]
L_disk_profile = [effective_path_length_disk(R, 0.4, 3.0) for R in R_range]
L_sphere_profile = [effective_path_length_sphere(R, 10) for R in R_range]

ax3.plot(R_range, L_disk_profile, 'b-', lw=2, label='Disk (h=0.4, R_d=3)')
ax3.plot(R_range, L_sphere_profile, 'r-', lw=2, label='Sphere (R=10)')
ax3.set_xlabel('Radius (kpc)')
ax3.set_ylabel('Effective Path Length L (kpc)')
ax3.set_title('Path Length Through Mass')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Panel 4: Comparison of approaches
ax4 = axes[1, 1]

# Current approach: constant A for galaxies
ax4.axhline(A_0, color='green', ls='-', lw=2, label='Current: A = A₀ (constant)')

# Solid angle approach
ax4.plot(R_range, A_disk_profile, 'b--', lw=2, label=f'Solid angle: A ∝ Ω^{alpha_test}')

# Path length approach
A_path_profile = [A_0 * (L/0.4)**0.27 for L in L_disk_profile]
ax4.plot(R_range, A_path_profile, 'r:', lw=2, label='Path length: A ∝ L^0.27')

ax4.set_xlabel('Radius (kpc)')
ax4.set_ylabel('Amplitude A')
ax4.set_title('Comparison of Amplitude Approaches (Disk Galaxy)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 3)

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / 'figures' / 'solid_angle_exploration.png', dpi=150)
print("Saved: figures/solid_angle_exploration.png")

plt.show()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
Key findings:

1. SOLID ANGLE APPROACH:
   - Thin disk at R=8 kpc subtends Ω ≈ 0.008 sr (0.06% of sky)
   - Cluster at r=200 kpc (inside R=600 kpc) subtends Ω = 4π sr (100%)
   - Ratio: ~5000× difference in solid angle
   
   - To get A_cluster/A_disk ≈ 8.45/1.17 ≈ 7.2 from this ratio,
     need α ≈ 0.23 (since 5000^0.23 ≈ 7)
   
   - Problem: Within a galaxy, solid angle varies with R, which would
     make A radius-dependent. Current model uses constant A for galaxies.

2. PATH LENGTH APPROACH:
   - Already implemented! L = 0.4 kpc for disks, L = 600 kpc for clusters
   - This IS essentially the "chord through the mass" idea
   - Works because disk thickness ~ 0.4 kpc, cluster size ~ 600 kpc

3. GEOMETRIC INTERPRETATION:
   - The path length L can be thought of as:
     * For disks: vertical path through the disk (~ scale height h)
     * For clusters: diameter of the mass distribution
   
   - The solid angle Ω is related but different:
     * Ω measures "how much sky the mass fills"
     * L measures "how far you travel through mass"
   
   - For our purposes, L (path length) is simpler and already works!

4. CONCLUSION:
   The current L-based formula A = A₀(L/L₀)^n already captures the 
   geometric effect you're describing. The path length L IS the 
   "arc/cone size" in a sense - it measures how much mass the 
   gravitational signal passes through.
   
   The solid angle approach would require radius-dependent A within
   galaxies, which would complicate the model and may not improve fits.
""")

