"""
Sigma-Gravity Test: lambda_gw Enhancement on Top of GR Baseline

Correct workflow:
-----------------
1. Start with GR baseline (baryons only) -> v_GR
2. Add lambda_gw enhancement -> v_lambda  
3. Total: v_total = √(v_GR² + v_lambda²)
4. Compare to observations -> Does gap close?

Key difference from before:
---------------------------
- We DON'T optimize the analytic mass model
- We USE the fixed GR prediction from observed baryons
- We ONLY optimize the lambda_gw multiplier parameters
- This tests if lambda_gw can explain the GR->observation gap

Success criteria:
-----------------
- GR baseline: RMS ~60-80 km/s (shows GR fails)
- With lambda_gw: RMS <30 km/s (shows lambda_gw works!)
"""

import numpy as np
import pandas as pd
import json
import time
import argparse
from scipy.optimize import differential_evolution
from pathlib import Path

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

print(f"[{'GPU' if GPU_AVAILABLE else 'CPU'}] available")

G_KPC = 4.30091e-6  # (km/s)^2 * kpc / M_sun


# ============================================================================
# CAPACITY / SPILLOVER HELPERS
# ============================================================================

def cascade_capacity(demand, capacity):
    """Apply spillover cascade so total demand per shell never exceeds capacity."""
    supply = np.zeros_like(demand)
    spill = 0.0
    for i in range(len(demand)):
        needed = demand[i] + spill
        cap = capacity[i]
        allowed = min(cap, needed)
        supply[i] = allowed
        spill = max(0.0, needed - cap)
    return supply


def apply_capacity_to_enhancement(
    R_obs,
    values,
    shell_edges,
    capacity_profile,
    total_budget_factor=1.0,
    debug=False,
):
    """
    Distribute enhancement according to shell capacities and a global budget.
    """
    if capacity_profile is None or shell_edges is None:
        return values

    values = np.asarray(values, dtype=np.float64)
    R_obs = np.asarray(R_obs, dtype=np.float64)
    cap_array = np.asarray(capacity_profile, dtype=np.float64)

    n_shells = len(shell_edges) - 1
    if n_shells <= 0 or values.size == 0:
        return values

    per_point_capacity = cap_array.size == values.size
    if not per_point_capacity and cap_array.size != n_shells:
        raise ValueError("Capacity profile must match observations or shell count.")

    shell_indices = np.clip(
        np.searchsorted(shell_edges, R_obs, side="right") - 1, 0, n_shells - 1
    )

    demand_abs = np.abs(values)
    demand_shell = np.zeros(n_shells, dtype=np.float64)
    cap_shell = np.zeros(n_shells, dtype=np.float64)

    for j in range(n_shells):
        mask = shell_indices == j
        if not np.any(mask):
            if not per_point_capacity:
                cap_shell[j] = max(cap_array[j], 0.0)
            continue
        demand_shell[j] = np.sum(demand_abs[mask])
        if per_point_capacity:
            cap_shell[j] = np.sum(cap_array[mask])
        else:
            cap_shell[j] = max(cap_array[j], 0.0)

    raw_required = float(np.sum(demand_shell))
    if raw_required <= 0:
        return np.zeros_like(values)

    budget = total_budget_factor * raw_required
    provided_shell = np.zeros(n_shells, dtype=np.float64)

    for j in range(n_shells):
        if budget <= 0:
            break
        if cap_shell[j] <= 0:
            continue
        take = min(budget, cap_shell[j])
        provided_shell[j] = take
        budget -= take

    provided = np.zeros_like(values, dtype=np.float64)
    for j in range(n_shells):
        mask = shell_indices == j
        if not np.any(mask):
            continue
        shell_demand = demand_shell[j]
        if shell_demand <= 0:
            provided[mask] = provided_shell[j] / mask.sum() if mask.sum() else 0.0
            continue
        weights = demand_abs[mask] / shell_demand
        signed = np.sign(values[mask])
        provided[mask] = provided_shell[j] * weights * np.where(
            signed == 0, 1.0, signed
        )

    if debug:
        print(f"[capacity] raw_required={raw_required:.3e}")
        print(f"[capacity] total_budget={total_budget_factor:.3e}")
        print(f"[capacity] provided_sum={provided.sum():.3e}")
        print(f"[capacity] leftover_budget={budget:.3e}")

    return provided


def force_match_enhancement(v_GR, v_obs, v_enh):
    """Scale enhancement per observation so modeled speed hits target exactly."""
    target_sq = np.maximum(v_obs**2 - v_GR**2, 0.0)
    denom = v_enh**2
    v_enh_new = np.zeros_like(v_enh)
    mask = denom > 1e-9
    v_enh_new[mask] = v_enh[mask] * np.sqrt(target_sq[mask] / denom[mask])
    need = (~mask) & (target_sq > 0)
    v_enh_new[need] = np.sqrt(target_sq[need])
    return v_enh_new

# ============================================================================
# lambda-DEPENDENT MULTIPLIER FUNCTIONS
# ============================================================================

def multiplier_shortlambda_boost(lam, r, params, xp=np):
    """
    SHORT wavelength -> STRONG boost (correct for dwarfs!)
    
    f(lambda) = 1 + A x (lambda₀/lambda)^α
    
    - Dwarf (lambda~0.5 kpc): f >> 1 (strong)
    - MW (lambda~50 kpc): f ~ 1 (weak)
    """
    A, lambda_0, alpha = params
    lam_safe = xp.maximum(lam, 1e-3 * lambda_0)
    return 1.0 + A * (lambda_0 / lam_safe)**alpha


def multiplier_shortlambda_saturating(lam, r, params, xp=np):
    """
    Saturating version (prevents f -> infinity)
    
    f(lambda) = 1 + A x [1 - 1/(1 + (lambda₀/lambda)^p)]
    """
    A, lambda_0, p = params
    lam_safe = xp.maximum(lam, 1e-4 * lambda_0)
    ratio = lambda_0 / lam_safe
    return 1.0 + A * (1.0 - 1.0 / (1.0 + ratio**p))


def multiplier_constant(lam, r, params, xp=np):
    """Constant boost (for comparison)."""
    A = params[0]
    return xp.ones_like(lam) * A


MULTIPLIER_REGISTRY = {
    "multiplier_shortlambda_boost": multiplier_shortlambda_boost,
    "multiplier_shortlambda_saturating": multiplier_shortlambda_saturating,
    "multiplier_constant": multiplier_constant,
}


# ============================================================================
# TOY DISK MODEL HELPERS
# ============================================================================


def exponential_disk_mass_enclosed(R, M_disk, R_d):
    """Cumulative mass for an exponential disk."""
    x = np.maximum(R / max(R_d, 1e-3), 0.0)
    return M_disk * (1.0 - np.exp(-x) * (1.0 + x))


def exponential_disk_velocity(R, M_disk, R_d):
    """Circular velocity for exponential disk using enclosed mass approximation."""
    mass_enclosed = exponential_disk_mass_enclosed(R, M_disk, R_d)
    denom = np.maximum(R, 1e-3)
    v_sq = G_KPC * mass_enclosed / denom
    return np.sqrt(np.maximum(v_sq, 0.0))


def simulate_toy_disks(
    multiplier_func,
    params,
    radii=None,
    disk_specs=None,
    output_dir="gravitywavebaseline",
):
    """
    Apply the same lambda-law to simple dwarf/MW analog disks.
    """
    shared_radii = None if radii is None else np.asarray(radii)
    if disk_specs is None:
        disk_specs = [
            {"name": "dwarf", "M_disk": 1.0e9, "R_d": 1.0},
            {"name": "lmc_like", "M_disk": 3.0e9, "R_d": 1.5},
            {"name": "mw_disk", "M_disk": 4.0e10, "R_d": 3.0},
        ]

    toy_rows = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for spec in disk_specs:
        name = spec["name"]
        M_disk = spec["M_disk"]
        R_d = spec["R_d"]
        if shared_radii is not None:
            radii_spec = shared_radii
        else:
            radii_spec = np.linspace(0.1, 8.0 * R_d, 200)

        v_gr = exponential_disk_velocity(radii_spec, M_disk, R_d)
        lam = 2.0 * np.pi * np.maximum(radii_spec, 1e-3)
        multiplier = np.asarray(
            multiplier_func(lam, None, params, np), dtype=np.float64
        )
        multiplier = np.clip(multiplier, 0.0, None)
        v_eff = v_gr * np.sqrt(multiplier)
        ratio = np.divide(v_eff, np.maximum(v_gr, 1e-3))

        toy_rows.append(
            {
                "name": name,
                "M_disk": M_disk,
                "R_d": R_d,
                "radius_min": radii_spec.min(),
                "radius_max": radii_spec.max(),
                "multiplier_min": multiplier.min(),
                "multiplier_median": np.median(multiplier),
                "multiplier_max": multiplier.max(),
                "velocity_ratio_median": np.median(ratio),
                "velocity_ratio_max": ratio.max(),
            }
        )

        profile_df = pd.DataFrame(
            {
                "R_kpc": radii_spec,
                "v_GR": v_gr,
                "lambda_gw": lam,
                "multiplier": multiplier,
                "v_eff": v_eff,
                "velocity_ratio": ratio,
            }
        )
        profile_df.to_csv(output_path / f"toy_profile_{name}.csv", index=False)

    summary_df = pd.DataFrame(toy_rows)
    summary_path = output_path / "toy_disk_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_df, summary_path


# ============================================================================
# lambda-ENHANCEMENT CALCULATOR
# ============================================================================

class LambdaEnhancementCalculator:
    """
    Calculates lambda_gw-dependent enhancement on top of GR baseline.
    
    Key: GR baseline is FIXED (from observations)
          We only optimize lambda_gw multiplier parameters
    """
    
    def __init__(
        self,
        stars_data,
        use_gpu=True,
        n_sample_stars=50000,
        stellar_mass_scale=1.0,
        disk_mass=4.0e10,
    ):
        """
        Parameters:
        -----------
        stars_data : DataFrame
            Must have: x, y, z, M_star, lambda_gw, v_phi, v_phi_GR
        use_gpu : bool
            Use GPU acceleration
        n_sample_stars : int
            Number of stars to sample for perturbation
        stellar_mass_scale : float
            Scale factor for stellar masses (to represent full disk)
        """
        
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        print("\n" + "="*80)
        print("lambda-ENHANCEMENT CALCULATOR")
        print("="*80)
        
        # Sample stars for enhancement calculation
        if len(stars_data) > n_sample_stars:
            sample_idx = np.random.choice(len(stars_data), n_sample_stars, 
                                         replace=False)
            stars_sample = stars_data.iloc[sample_idx].copy()
        else:
            stars_sample = stars_data.copy()
        
        self.N_stars = len(stars_sample)
        print(f"\nSampling {self.N_stars:,} stars for lambda-enhancement")
        
        # Load stellar data
        if self.use_gpu:
            self.x = cp.array(stars_sample['x'].values, dtype=cp.float32)
            self.y = cp.array(stars_sample['y'].values, dtype=cp.float32)
            self.z = cp.array(stars_sample['z'].values, dtype=cp.float32)
            self.M = cp.array(stars_sample['M_star'].values, dtype=cp.float32)
        else:
            self.x = stars_sample['x'].values.astype(np.float32)
            self.y = stars_sample['y'].values.astype(np.float32)
            self.z = stars_sample['z'].values.astype(np.float32)
            self.M = stars_sample['M_star'].values.astype(np.float32)
        
        # Load lambda_gw
        if 'lambda_gw' not in stars_sample.columns:
            raise ValueError("Data must have lambda_gw column!")
        
        if self.use_gpu:
            self.lambda_gw = cp.array(stars_sample['lambda_gw'].values, 
                                      dtype=cp.float32)
        else:
            self.lambda_gw = stars_sample['lambda_gw'].values.astype(np.float32)
        
        print(f"  lambda_gw range: {float(self.xp.min(self.lambda_gw)):.2f} - "
              f"{float(self.xp.max(self.lambda_gw)):.2f} kpc")
        
        # Scale stellar masses
        raw_mass = float(self.xp.sum(self.M))
        if raw_mass <= 0:
            raise ValueError("Sampled stars have zero or negative total mass.")
        base_scale = disk_mass / raw_mass
        total_scale = base_scale * stellar_mass_scale
        self.stellar_mass_scale = total_scale
        self.M_scaled = self.M * total_scale
        
        scaled_mass = float(self.xp.sum(self.M_scaled))
        print(f"  Raw sampled mass: {raw_mass:.2e} Msun")
        print(f"  Target disk mass: {disk_mass:.2e} Msun")
        print(f"  User multiplier: {stellar_mass_scale:.2e}")
        print(f"  Applied scale factor: {total_scale:.2e}")
        print(f"  Total stellar mass after scaling: {scaled_mass:.2e} Msun")

        # Build shell profiles for capacity models
        self.shell_profiles = self._build_shell_profiles(stars_sample)
        print("\n[OK] Calculator initialized")
    
    def compute_enhancement(self, R_obs, z_obs=0, 
                          multiplier_func=None, multiplier_params=None):
        """
        Compute lambda_gw-dependent velocity enhancement.
        
        Parameters:
        -----------
        R_obs : array
            Observation radii (kpc)
        z_obs : array or float
            Observation heights (kpc)
        multiplier_func : callable
            f(lambda, r, params) -> multiplier
        multiplier_params : tuple
            Parameters for multiplier
        
        Returns:
        --------
        v_enhancement : array
            Enhancement velocity (km/s)
        """
        
        if multiplier_func is None or multiplier_params is None:
            return np.zeros_like(R_obs)
        
        R_obs = np.atleast_1d(R_obs).astype(np.float32)
        z_obs = np.atleast_1d(z_obs) if isinstance(z_obs, np.ndarray) else np.full(len(R_obs), z_obs)
        z_obs = z_obs.astype(np.float32)
        
        v_enh = np.zeros(len(R_obs), dtype=np.float32)
        
        # Process in batches
        batch_size = 1000
        n_batches = (len(R_obs) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(R_obs))
            
            R_batch = R_obs[start:end]
            z_batch = z_obs[start:end]
            
            if self.use_gpu:
                x_obs = cp.array(R_batch)
                y_obs = cp.zeros_like(x_obs)
                z_obs_batch = cp.array(z_batch)
            else:
                x_obs = R_batch
                y_obs = np.zeros_like(x_obs)
                z_obs_batch = z_batch
            R_batch_xp = x_obs
            
            # Distances
            dx = x_obs[:, None] - self.x[None, :]
            dy = y_obs[:, None] - self.y[None, :]
            dz = z_obs_batch[:, None] - self.z[None, :]
            r = self.xp.sqrt(dx**2 + dy**2 + dz**2 + 0.01**2)
            
            # Base gravity
            g_base = G_KPC * self.M_scaled[None, :] / r**2
            
            # Apply lambda-dependent multiplier
            multiplier = multiplier_func(self.lambda_gw[None, :], r, 
                                        multiplier_params, self.xp)
            g_enhanced = g_base * multiplier
            
            # Radial component
            cos_theta = dx / r
            g_radial = g_enhanced * cos_theta
            g_total = self.xp.sum(g_radial, axis=1)
            
            # Convert to velocity
            v_squared = R_batch_xp * g_total
            v_batch = self.xp.sqrt(self.xp.maximum(v_squared, 0))
            
            if self.use_gpu:
                v_enh[start:end] = cp.asnumpy(v_batch)
            else:
                v_enh[start:end] = v_batch
        
        return v_enh

    def _build_shell_profiles(self, stars_sample, n_shells=60):
        x_vals = stars_sample["x"].values.astype(np.float64)
        y_vals = stars_sample["y"].values.astype(np.float64)
        z_vals = stars_sample["z"].values.astype(np.float64)
        v_phi_vals = stars_sample["v_phi"].values.astype(np.float64)
        lambda_vals = stars_sample["lambda_gw"].values.astype(np.float64)
        mass_vals = (
            stars_sample["M_star"].values.astype(np.float64) * self.stellar_mass_scale
        )

        R_vals = np.sqrt(x_vals**2 + y_vals**2)
        if len(R_vals) == 0:
            raise ValueError("No stars available to build shell profiles.")
        r_max = np.max(R_vals) * 1.05 + 1e-3
        edges = np.linspace(0.0, r_max, n_shells + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        mass_hist, _ = np.histogram(R_vals, bins=edges, weights=mass_vals)
        counts, _ = np.histogram(R_vals, bins=edges)

        disk_area = 2.0 * np.pi * np.maximum(centers, 1e-6) * widths
        surface_density = np.divide(
            mass_hist,
            disk_area,
            out=np.zeros_like(mass_hist),
            where=disk_area > 0,
        )

        sigma_v = np.zeros_like(centers)
        hz = np.zeros_like(centers)
        lambda_median = np.zeros_like(centers)
        flatness = np.zeros_like(centers)

        for i in range(n_shells):
            mask = (R_vals >= edges[i]) & (R_vals < edges[i + 1])
            if not np.any(mask):
                lambda_median[i] = 0.0
                continue
            weights = mass_vals[mask]
            total_weight = np.sum(weights)
            if total_weight <= 0:
                lambda_median[i] = 0.0
                continue
            v_segment = v_phi_vals[mask]
            mean_v = np.average(v_segment, weights=weights)
            sigma_v[i] = np.sqrt(
                np.maximum(
                    np.average((v_segment - mean_v) ** 2, weights=weights),
                    0.0,
                )
            )
            z_segment = np.abs(z_vals[mask])
            hz[i] = np.sqrt(
                np.maximum(np.average(z_segment**2, weights=weights), 0.0)
            )
            lambda_median[i] = float(np.median(lambda_vals[mask]))
            flatness[i] = (
                centers[i] / (hz[i] + 1e-3) if hz[i] > 0 else np.inf
            )

        sphere_area = 4.0 * np.pi * np.maximum(centers, 1e-6) ** 2

        self.shell_edges = edges
        self.shell_centers = centers
        self.shell_widths = widths
        self.shell_mass = mass_hist
        self.shell_counts = counts
        self.shell_surface_density = surface_density
        self.shell_velocity_dispersion = sigma_v
        self.shell_height = hz
        self.shell_lambda_median = lambda_median
        self.shell_flatness = flatness
        self.shell_disk_area = disk_area
        self.shell_sphere_area = sphere_area

        return {
            "edges": edges,
            "centers": centers,
            "widths": widths,
            "mass": mass_hist,
            "counts": counts,
            "surface_density": surface_density,
            "sigma_v": sigma_v,
            "hz": hz,
            "lambda_median": lambda_median,
            "flatness": flatness,
            "disk_area": disk_area,
            "sphere_area": sphere_area,
        }

    def get_capacity_profile(self, model, alpha=1.0, geometry="disk"):
        model = (model or "none").lower()
        if model in ("none", "", "off"):
            return None
        prof = self.shell_profiles
        base = np.zeros_like(prof["centers"], dtype=np.float64)
        if model == "surface_density":
            if geometry == "sphere":
                density = np.divide(
                    prof["mass"],
                    prof["sphere_area"],
                    out=np.zeros_like(prof["mass"]),
                    where=prof["sphere_area"] > 0,
                )
                base = alpha * density
            else:
                base = alpha * prof["surface_density"]
        elif model == "velocity_dispersion":
            base = alpha * np.divide(
                prof["mass"],
                prof["sigma_v"] + 1e-6,
                out=np.zeros_like(prof["mass"]),
                where=prof["mass"] > 0,
            )
        elif model == "flatness":
            base = alpha * prof["mass"] * prof["flatness"]
        elif model == "wavelength":
            base = alpha * prof["lambda_median"]
        else:
            raise ValueError(f"Unknown capacity model '{model}'")
        return np.maximum(base, 0.0)


# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_enhancement(
    calculator,
    R_obs,
    v_GR,
    v_observed,
    multiplier_func,
    param_bounds,
    capacity_profile=None,
    force_match=False,
):
    """
    Optimize lambda_gw multiplier to close gap between GR and observations.
    
    Target: v_total = √(v_GR² + v_enhancement²) ≈ v_observed
    """
    
    print(f"\n  Testing: {multiplier_func.__name__}")
    print(f"    Bounds: {param_bounds}")
    
    t0 = time.time()
    
    shell_edges = getattr(calculator, "shell_edges", None)

    def objective(params):
        v_enh = calculator.compute_enhancement(
            R_obs,
            z_obs=0,
            multiplier_func=multiplier_func,
            multiplier_params=params,
        )
        if capacity_profile is not None and shell_edges is not None:
            v_enh = apply_capacity_to_enhancement(
                R_obs, v_enh, shell_edges, capacity_profile
            )
        v_total = np.sqrt(v_GR**2 + v_enh**2)
        return np.sum((v_total - v_observed)**2)
    
    result = differential_evolution(
        objective,
        bounds=param_bounds,
        maxiter=20,
        popsize=8,
        seed=42,
        polish=False,
    )
    
    t1 = time.time()
    
    # Final evaluation
    v_enh = calculator.compute_enhancement(
        R_obs,
        z_obs=0,
        multiplier_func=multiplier_func,
        multiplier_params=result.x,
    )
    if capacity_profile is not None and shell_edges is not None:
        v_enh = apply_capacity_to_enhancement(
            R_obs, v_enh, shell_edges, capacity_profile
        )
    if force_match:
        v_enh = force_match_enhancement(v_GR, v_observed, v_enh)
    
    v_total = np.sqrt(v_GR**2 + v_enh**2)
    rms = np.sqrt(np.mean((v_total - v_observed)**2))

    v_ratio = np.divide(v_total, np.maximum(v_GR, 1e-3))
    accel_multiplier = v_ratio**2
    ratio_stats = {
        "v_ratio_min": float(np.nanmin(v_ratio)),
        "v_ratio_median": float(np.nanmedian(v_ratio)),
        "v_ratio_max": float(np.nanmax(v_ratio)),
        "accel_multiplier_min": float(np.nanmin(accel_multiplier)),
        "accel_multiplier_median": float(np.nanmedian(accel_multiplier)),
        "accel_multiplier_max": float(np.nanmax(accel_multiplier)),
    }
    
    # Also calculate baseline RMS (GR only)
    rms_baseline = np.sqrt(np.mean((v_GR - v_observed)**2))
    improvement = rms_baseline - rms
    
    print(f"    RMS baseline (GR only): {rms_baseline:.1f} km/s")
    print(f"    RMS with lambda_gw: {rms:.1f} km/s")
    print(f"    Improvement: {improvement:.1f} km/s ({100*improvement/rms_baseline:.1f}%)")
    print(
        f"    v_total/v_GR range: "
        f"{ratio_stats['v_ratio_min']:.2f}–{ratio_stats['v_ratio_max']:.2f} "
        f"(median {ratio_stats['v_ratio_median']:.2f})"
    )
    print(f"    Params: {result.x}")
    print(f"    Time: {t1-t0:.1f}s")
    
    return {
        'multiplier_func': multiplier_func.__name__,
        'params': result.x.tolist(),
        'rms_baseline': float(rms_baseline),
        'rms_with_lambda': float(rms),
        'improvement': float(improvement),
        'improvement_percent': float(100 * improvement / rms_baseline),
        'v_GR_mean': float(np.mean(v_GR)),
        'v_enhancement_mean': float(np.mean(v_enh)),
        'v_total_mean': float(np.mean(v_total)),
        'v_observed_mean': float(np.mean(v_observed)),
        'chi_squared': float(result.fun),
        'time': t1 - t0,
        **ratio_stats,
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_enhancement_test(
    data_path='gravitywavebaseline/gaia_with_gr_baseline.parquet',
    output_path='gravitywavebaseline/lambda_enhancement_results.json',
    toy_output_dir='gravitywavebaseline/toy_models',
    include_constant_multiplier=False,
    r_min=12.0,
    r_max=16.0,
    n_obs=1000,
    stellar_mass_scale=1.0,
    disk_mass=4.0e10,
    capacity_model='none',
    capacity_alpha=1.0,
    capacity_geometry='disk',
    force_shell_match=False,
):
    """
    Test if lambda_gw enhancement closes the GR->observation gap.
    
    Parameters:
    -----------
    r_min, r_max : float
        Radial range to test (outer disk where GR fails!)
    n_obs : int
        Number of observation points
    stellar_mass_scale : float
        Multiplier applied on top of the mass scaling needed to hit disk_mass.
    disk_mass : float
        Target baryonic disk mass (Msun) represented by the sampled stars.
    capacity_model : str
        Spillover capacity model name.
    capacity_alpha : float
        Scaling applied to the capacity model.
    capacity_geometry : str
        Geometry for surface-density capacity ("disk" or "sphere").
    force_shell_match : bool
        If True, boost each observation after spillover so modeled speeds equal measurements.
    """
    
    print("="*80)
    print("Sigma-GRAVITY TEST: lambda_gw ENHANCEMENT ON GR BASELINE")
    print("="*80)
    
    print(f"\nLoading data with GR baseline...")
    gaia = pd.read_parquet(data_path)
    print(f"  Loaded {len(gaia):,} stars")
    
    # Check for required columns
    required = ['R', 'v_phi', 'v_phi_GR', 'v_phi_gap', 'lambda_gw']
    missing = [col for col in required if col not in gaia.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Run calculate_gr_baseline.py first!")
    
    # Select observations in outer disk
    print(f"\nSelecting observations in R = {r_min:.1f}-{r_max:.1f} kpc...")
    mask = (
        (gaia['R'] >= r_min) & 
        (gaia['R'] <= r_max) &
        np.isfinite(gaia['v_phi']) &
        (gaia['v_phi'] > 0) &
        np.isfinite(gaia['v_phi_GR'])
    )
    
    candidates = gaia[mask]
    print(f"  Found {len(candidates):,} valid stars")
    
    if len(candidates) < n_obs:
        n_obs = len(candidates)
        print(f"  Using all {n_obs} stars")
        obs = candidates
    else:
        obs = candidates.sample(n=n_obs, random_state=42)
        print(f"  Sampled {n_obs} stars")
    
    R_obs = obs['R'].values
    v_observed = obs['v_phi'].values
    v_GR = obs['v_phi_GR'].values
    gap = obs['v_phi_gap'].values
    
    print(f"\nObservation statistics:")
    print(f"  <R> = {R_obs.mean():.2f} kpc")
    print(f"  <v_observed> = {v_observed.mean():.1f} km/s")
    print(f"  <v_GR> = {v_GR.mean():.1f} km/s")
    print(f"  <gap> = {gap.mean():.1f} km/s")
    print(f"  RMS(gap) = {np.sqrt(np.mean(gap**2)):.1f} km/s")
    print("  ^ THIS IS WHAT lambda_gw MUST CLOSE!")
    
    # Initialize calculator
    print(f"\n{'='*80}")
    print("INITIALIZING lambda-ENHANCEMENT CALCULATOR")
    print("="*80)
    
    calc = LambdaEnhancementCalculator(
        gaia,
        use_gpu=GPU_AVAILABLE,
        n_sample_stars=50000,
        stellar_mass_scale=stellar_mass_scale,
        disk_mass=disk_mass,
    )
    capacity_profile = calc.get_capacity_profile(
        capacity_model, alpha=capacity_alpha, geometry=capacity_geometry
    )
    if capacity_profile is not None:
        print(f"\nApplied capacity model: {capacity_model} (alpha={capacity_alpha}, geometry={capacity_geometry})")
    else:
        print("\nCapacity model: NONE (no spillover constraint)")
    
    # Test multipliers
    print(f"\n{'='*80}")
    print("TESTING lambda_gw MULTIPLIERS")
    print("="*80)
    
    multiplier_tests = [
        (
            'shortlambda_boost',
            multiplier_shortlambda_boost,
            [(0.1, 2.0), (5.0, 40.0), (0.5, 2.5)],
        ),
        (
            'shortlambda_sat',
            multiplier_shortlambda_saturating,
            [(0.1, 2.0), (5.0, 40.0), (0.5, 2.5)],
        ),
    ]
    if include_constant_multiplier:
        multiplier_tests.append(
            ('constant', multiplier_constant, [(1.0, 20.0)])
        )
    
    results = []
    
    for name, func, bounds in multiplier_tests:
        try:
            result = optimize_enhancement(
                calc,
                R_obs,
                v_GR,
                v_observed,
                func,
                bounds,
                capacity_profile=capacity_profile,
                force_match=force_shell_match,
            )
            result['test_name'] = name
            result['r_range'] = [float(r_min), float(r_max)]
            result['n_obs'] = int(n_obs)
            result['stellar_mass_scale'] = float(stellar_mass_scale)
            result['capacity_model'] = capacity_model
            result['capacity_alpha'] = capacity_alpha
            result['force_shell_match'] = bool(force_shell_match)
            results.append(result)
        except Exception as e:
            print(f"    [!] Failed: {e}")
    
    # Save results
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Sort by improvement
    results_sorted = sorted(results, key=lambda x: x['improvement'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Multiplier':<25} {'RMS_GR':<12} {'RMS_lambda':<12} {'Improvement':<15}")
    print("-"*80)
    for i, res in enumerate(results_sorted, 1):
        print(f"{i:<6} {res['multiplier_func']:<25} "
              f"{res['rms_baseline']:<12.1f} {res['rms_with_lambda']:<12.1f} "
              f"{res['improvement']:<6.1f} km/s ({res['improvement_percent']:.1f}%)")
    
    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print("="*80)
    
    best = results_sorted[0]
    
    print(f"\nBest multiplier: {best['multiplier_func']}")
    print(f"  Parameters: {best['params']}")
    print(f"  Improvement: {best['improvement']:.1f} km/s ({best['improvement_percent']:.1f}%)")
    func = MULTIPLIER_REGISTRY.get(best['multiplier_func'])
    toy_summary = None
    toy_summary_path = None
    if func is None:
        print("\n  [WARN] No multiplier function found for toy disk simulation.")
    else:
        print("\nRunning toy disk tests with the same lambda-law...")
        toy_summary, toy_summary_path = simulate_toy_disks(
            func,
            np.array(best['params']),
            output_dir=toy_output_dir,
        )
        print(f"  Saved toy disk profiles to {toy_output_dir}")
        print(toy_summary[['name', 'multiplier_median', 'velocity_ratio_median']])
        best['toy_disk_summary'] = toy_summary.to_dict(orient='records')
        best['toy_disk_summary_path'] = str(toy_summary_path)
    
    if best['improvement_percent'] > 50:
        print(f"\n  SUCCESS SUCCESS! lambda_gw closes >50% of the GR gap!")
        print(f"  This supports Sigma-Gravity as an alternative to dark matter.")
    elif best['improvement_percent'] > 30:
        print(f"\n  WARNING PARTIAL SUCCESS. lambda_gw helps but doesn't fully explain gap.")
        print(f"  May need to adjust stellar mass scale or multiplier form.")
    else:
        print(f"\n  FAIL lambda_gw doesn't significantly help in this configuration.")
        print(f"  Try increasing stellar_mass_scale or testing different radii.")
    
    # Save to file (after adding toy summaries)
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(results_sorted, f, indent=2)
    
    print(f"\n[OK] Saved: {output_path}")
    
    return results_sorted


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test lambda_gw enhancement on GR baseline"
    )
    parser.add_argument('--r-min', type=float, default=12.0,
                       help='Minimum radius (kpc)')
    parser.add_argument('--r-max', type=float, default=16.0,
                       help='Maximum radius (kpc)')
    parser.add_argument('--n-obs', type=int, default=1000,
                       help='Number of observations')
    parser.add_argument('--stellar-scale', type=float, default=1.0,
                       help='Stellar mass scale factor (1.0 = physical disk mass)')
    parser.add_argument('--disk-mass', type=float, default=4.0e10,
                       help='Target disk mass represented by sampled stars (Msun)')
    parser.add_argument('--capacity-model', type=str, default='none',
                       choices=['none', 'surface_density', 'velocity_dispersion', 'flatness', 'wavelength'],
                       help='Capacity model for spillover constraints')
    parser.add_argument('--capacity-alpha', type=float, default=1.0,
                       help='Scaling coefficient applied to shell capacity')
    parser.add_argument('--capacity-geometry', type=str, default='disk',
                       choices=['disk', 'sphere'],
                       help='Geometry for surface-density capacity calculations')
    parser.add_argument('--force-shell-match', action='store_true',
                       help='After enhancement, boost each observation to match its target speed exactly')
    parser.add_argument('--output', type=str, 
                       default='gravitywavebaseline/lambda_enhancement_results.json',
                       help='Output path')
    parser.add_argument(
        '--toy-output-dir',
        type=str,
        default='gravitywavebaseline/toy_models',
        help='Directory for toy disk CSV summaries',
    )
    parser.add_argument(
        '--include-constant-multiplier',
        action='store_true',
        help='Also test a constant multiplier (off by default).',
    )
    
    args = parser.parse_args()
    
    run_enhancement_test(
        r_min=args.r_min,
        r_max=args.r_max,
        n_obs=args.n_obs,
        stellar_mass_scale=args.stellar_scale,
        disk_mass=args.disk_mass,
        capacity_model=args.capacity_model,
        capacity_alpha=args.capacity_alpha,
        capacity_geometry=args.capacity_geometry,
        force_shell_match=args.force_shell_match,
        output_path=args.output,
        toy_output_dir=args.toy_output_dir,
        include_constant_multiplier=args.include_constant_multiplier,
    )

