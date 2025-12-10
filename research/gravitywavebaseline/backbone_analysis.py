"""
COMPLETE MASS MODEL: Analytic Backbone + Stellar lambda-Perturbations

Problem with previous approach:
- Sampled stars (~6e5 Msun) << MW total mass (~1e11 Msun)
- Trying to use sparse stars to produce ALL gravity
- Multipliers have nothing to amplify

Solution:
1. Analytic components (disk + halo) provide bulk mass -> baseline v~220 km/s
2. Stellar sample with lambda-multipliers adds PERTURBATION on top
3. Multipliers modify local enhancement, not total gravity

This matches the Sigma-Gravity concept: smooth field + coherence enhancement.
"""

import json
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares

try:
    import cupy as cp

    GPU_AVAILABLE = True
    print("[OK] GPU available")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("[!] CPU only")

# Constants
# Gravitational constant in (km/s)^2 kpc / Msun
G_KPC = 4.30091e-6

# ============================================================================
# ANALYTIC MASS MODELS (BACKBONE)
# ============================================================================


def miyamoto_nagai_disk(R, z=0, M_disk=6e10, a_disk=3.5, b_disk=0.25):
    """
    Miyamoto-Nagai disk potential (flattened).

    Realistic MW disk:
    - M_disk = 6e10 Msun
    - a = 3.5 kpc (scale length)
    - b = 0.25 kpc (scale height)

    Returns v_circ at radius R in km/s.
    """

    z = np.abs(z)
    denom = np.sqrt(R**2 + (a_disk + np.sqrt(z**2 + b_disk**2)) ** 2)
    v_squared = G_KPC * M_disk * R**2 / denom**3
    return np.sqrt(v_squared)


def nfw_halo(R, M_200=1.5e12, c=12):
    """
    NFW dark matter halo.
    Returns v_circ at radius R in km/s.
    """

    h = 0.7
    H_0 = 100 * h  # km/s/Mpc
    rho_crit = 3 * H_0**2 / (8 * np.pi * G_KPC * 1e12)  # Msun/kpc^3
    R_200 = (3 * M_200 / (4 * np.pi * 200 * rho_crit)) ** (1 / 3)
    r_s = R_200 / c
    x = R / r_s
    M_enc = (
        M_200
        * (np.log(1 + x) - x / (1 + x))
        / (np.log(1 + c) - c / (1 + c))
    )
    v_squared = G_KPC * M_enc / R
    return np.sqrt(v_squared)


def hernquist_bulge(R, M_bulge=1.5e10, a_bulge=0.7):
    """Hernquist bulge contribution."""

    v_squared = G_KPC * M_bulge * R / (R + a_bulge) ** 2
    return np.sqrt(v_squared)


def total_analytic_velocity(
    R,
    z=0,
    M_disk=6e10,
    M_bulge=1.5e10,
    M_200=1.5e12,
    include_halo=True,
):
    """
    Total rotation curve from analytic components.
    """

    v_disk = miyamoto_nagai_disk(R, z, M_disk=M_disk)
    v_bulge = hernquist_bulge(R, M_bulge=M_bulge)

    if include_halo:
        v_halo = nfw_halo(R, M_200=M_200)
        v_total = np.sqrt(v_disk**2 + v_bulge**2 + v_halo**2)
    else:
        v_halo = np.zeros_like(R)
        v_total = np.sqrt(v_disk**2 + v_bulge**2)

    return v_total, {"disk": v_disk, "bulge": v_bulge, "halo": v_halo}


def calibrate_masses(R_obs, v_obs, use_halo, init_disk, init_bulge, init_halo):
    """
    Fit analytic component masses so the baseline curve matches observations.
    """

    R_arr = np.asarray(R_obs, dtype=float)
    v_arr = np.asarray(v_obs, dtype=float)

    # Bounds ensure physically reasonable masses
    disk_bounds = (1e9, 5e11)
    bulge_bounds = (1e9, 5e10)
    halo_bounds = (1e10, 5e12)

    params = [np.clip(init_disk, *disk_bounds), np.clip(init_bulge, *bulge_bounds)]
    bounds_low = [disk_bounds[0], bulge_bounds[0]]
    bounds_high = [disk_bounds[1], bulge_bounds[1]]

    if use_halo:
        params.append(np.clip(init_halo, *halo_bounds))
        bounds_low.append(halo_bounds[0])
        bounds_high.append(halo_bounds[1])

    log_init = np.log10(params)
    log_low = np.log10(bounds_low)
    log_high = np.log10(bounds_high)

    def residuals(log_masses):
        mdisk = 10 ** log_masses[0]
        mbulge = 10 ** log_masses[1]
        mhalo = 10 ** log_masses[2] if use_halo else 0.0
        v_model, _ = total_analytic_velocity(
            R_arr, 0.0, mdisk, mbulge, mhalo, include_halo=use_halo
        )
        return v_model - v_arr

    result = least_squares(
        residuals,
        log_init,
        bounds=(log_low, log_high),
        max_nfev=200,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )

    mdisk = 10 ** result.x[0]
    mbulge = 10 ** result.x[1]
    mhalo = 10 ** result.x[2] if use_halo else 0.0

    return {
        "M_disk": float(mdisk),
        "M_bulge": float(mbulge),
        "M_200": float(mhalo),
        "success": result.success,
        "message": result.message,
    }


# ============================================================================
# STELLAR lambda-PERTURBATION CALCULATOR
# ============================================================================


class AnalyticBackboneCalculator:
    """
    Proper mass model:
    1. Analytic components provide baseline gravity
    2. Stellar sample adds lambda-dependent perturbation
    """

    def __init__(
        self,
        stars_data,
        use_gpu=True,
        M_disk=6e10,
        M_bulge=1.5e10,
        M_200=1.5e12,
        use_halo=True,
        n_sample_stars=50000,
    ):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        print("\n" + "=" * 80)
        print("ANALYTIC BACKBONE + STELLAR PERTURBATION CALCULATOR")
        print("=" * 80)

        self.M_disk = M_disk
        self.M_bulge = M_bulge
        self.M_200 = M_200
        self.use_halo = use_halo

        print("\nAnalytic components:")
        print(f"  Disk (Miyamoto-Nagai): {M_disk:.2e} Msun")
        print(f"  Bulge (Hernquist): {M_bulge:.2e} Msun")
        if use_halo:
            print(f"  Halo (NFW): {M_200:.2e} Msun")
        else:
            print("  Halo: DISABLED")

        print(f"\nSampling {n_sample_stars:,} stars for lambda perturbation...")
        if len(stars_data) > n_sample_stars:
            sample_idx = np.random.choice(
                len(stars_data), n_sample_stars, replace=False
            )
            stars_sample = stars_data.iloc[sample_idx].copy()
        else:
            stars_sample = stars_data.copy()

        self.N_stars = len(stars_sample)
        print(f"  Using {self.N_stars:,} stars")

        if self.use_gpu:
            self.x = cp.array(stars_sample["x"].values, dtype=cp.float32)
            self.y = cp.array(stars_sample["y"].values, dtype=cp.float32)
            self.z = cp.array(stars_sample["z"].values, dtype=cp.float32)
            self.M = cp.array(stars_sample["M_star"].values, dtype=cp.float32)
        else:
            self.x = stars_sample["x"].values.astype(np.float32)
            self.y = stars_sample["y"].values.astype(np.float32)
            self.z = stars_sample["z"].values.astype(np.float32)
            self.M = stars_sample["M_star"].values.astype(np.float32)

        self.periods = {}
        for col in stars_sample.columns:
            if col.startswith("lambda_"):
                period_name = col.replace("lambda_", "")
                if self.use_gpu:
                    self.periods[period_name] = cp.array(
                        stars_sample[col].values, dtype=cp.float32
                    )
                else:
                    self.periods[period_name] = stars_sample[
                        col
                    ].values.astype(np.float32)

        print(f"  Periods loaded: {list(self.periods.keys())}")

        M_stars_total = float(self.xp.sum(self.M))
        self.stellar_fraction = 0.05  # stars ~5% disk mass
        self.M_stellar_effective = self.M_disk * self.stellar_fraction
        self.M_scale_factor = self.M_stellar_effective / (M_stars_total + 1e-12)

        print("\nStellar mass scaling:")
        print(f"  Raw stellar mass: {M_stars_total:.2e} Msun")
        print(
            f"  Scaled to: {self.M_stellar_effective:.2e} Msun "
            f"({self.stellar_fraction*100:.0f}% of disk)"
        )
        print(f"  Scale factor: {self.M_scale_factor:.2e}")

        self.M_scaled = self.M * self.M_scale_factor
        print("\n[OK] Calculator initialized")

    def compute_velocity(
        self,
        R_obs,
        z_obs=0,
        period_name="gw",
        multiplier_func=None,
        multiplier_params=None,
    ):
        v_analytic, analytic_components = total_analytic_velocity(
            R_obs,
            z_obs,
            M_disk=self.M_disk,
            M_bulge=self.M_bulge,
            M_200=self.M_200,
            include_halo=self.use_halo,
        )

        if multiplier_func is not None and multiplier_params is not None:
            v_stellar_pert = self._compute_stellar_perturbation(
                R_obs, z_obs, period_name, multiplier_func, multiplier_params
            )
        else:
            v_stellar_pert = np.zeros_like(R_obs)

        v_total = np.sqrt(v_analytic**2 + v_stellar_pert**2)
        components = analytic_components.copy()
        components["stellar_perturbation"] = v_stellar_pert
        components["analytic_total"] = v_analytic
        return v_total, components

    def _compute_stellar_perturbation(
        self, R_obs, z_obs, period_name, multiplier_func, params
    ):
        R_obs = np.atleast_1d(R_obs)
        z_obs = (
            np.atleast_1d(z_obs)
            if isinstance(z_obs, np.ndarray)
            else np.full(len(R_obs), z_obs)
        )

        v_pert = np.zeros(len(R_obs), dtype=np.float32)
        lambda_vals = self.periods[period_name]

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
                R_batch_xp = x_obs
            else:
                x_obs = R_batch
                y_obs = np.zeros_like(x_obs)
                z_obs_batch = z_batch
                R_batch_xp = x_obs

            dx = x_obs[:, None] - self.x[None, :]
            dy = y_obs[:, None] - self.y[None, :]
            dz = z_obs_batch[:, None] - self.z[None, :]
            r = self.xp.sqrt(dx**2 + dy**2 + dz**2 + 0.01**2)

            g_base = G_KPC * self.M_scaled[None, :] / r**2
            multiplier = multiplier_func(lambda_vals[None, :], r, params, self.xp)
            g_enhanced = g_base * multiplier
            cos_theta = dx / r
            g_radial = g_enhanced * cos_theta
            g_total = self.xp.sum(g_radial, axis=1)
            v_squared = R_batch_xp * g_total
            v_batch = self.xp.sqrt(self.xp.maximum(v_squared, 0))

            if self.use_gpu:
                v_pert[start:end] = cp.asnumpy(v_batch)
            else:
                v_pert[start:end] = v_batch

        return v_pert


# ============================================================================
# MULTIPLIER FUNCTIONS
# ============================================================================


def multiplier_none(lam, r, params, xp=np):
    return xp.ones_like(lam)


def multiplier_constant(lam, r, params, xp=np):
    (A,) = params
    return xp.ones_like(lam) * A


def multiplier_power_law(lam, r, params, xp=np):
    A, lambda_0, alpha = params
    return 1.0 + A * (lam / lambda_0) ** alpha


def multiplier_wavelength_bins(lam, r, params, xp=np, edges=None):
    if edges is None:
        edges = xp.logspace(-1, 2, len(params) + 1)
    multiplier = xp.ones_like(lam)
    for i, f_i in enumerate(params):
        if i < len(edges) - 1:
            mask = (lam >= edges[i]) & (lam < edges[i + 1])
            multiplier = xp.where(mask, f_i, multiplier)
    return multiplier


def multiplier_richter_scale(lam, r, params, xp=np):
    A, beta = params
    return A * lam**beta


def multiplier_shortlambda_boost(lam, r, params, xp=np):
    A, lambda_0, alpha = params
    lam_safe = xp.maximum(lam, 1e-3 * lambda_0)
    return 1.0 + A * (lambda_0 / lam_safe) ** alpha


def multiplier_shortlambda_saturating(lam, r, params, xp=np):
    A, lambda_0, p = params
    lam_safe = xp.maximum(lam, 1e-4 * lambda_0)
    ratio = lambda_0 / lam_safe
    return 1.0 + A * (1.0 - 1.0 / (1.0 + ratio**p))


# ============================================================================
# OPTIMIZATION
# ============================================================================


def optimize_with_backbone(
    calculator, R_obs, v_observed, period_name, multiplier_func, param_bounds
):
    print(f"\n  Testing: {period_name} + {multiplier_func.__name__}")
    t0 = time.time()

    def objective(params):
        try:
            v_model, _ = calculator.compute_velocity(
                R_obs,
                z_obs=0,
                period_name=period_name,
                multiplier_func=multiplier_func,
                multiplier_params=params,
            )
            return np.sum((v_model - v_observed) ** 2)
        except Exception as exc:
            raise RuntimeError(f"objective failure: {exc}") from exc

    result = differential_evolution(
        objective,
        bounds=param_bounds,
        maxiter=15,
        popsize=5,
        seed=42,
        polish=False,
    )

    t1 = time.time()

    v_model, components = calculator.compute_velocity(
        R_obs,
        z_obs=0,
        period_name=period_name,
        multiplier_func=multiplier_func,
        multiplier_params=result.x,
    )

    rms = np.sqrt(np.mean((v_model - v_observed) ** 2))

    print(f"    RMS: {rms:.1f} km/s (time: {t1 - t0:.1f}s)")
    print(f"    Params: {result.x}")
    print("    Components:")
    for name, v in components.items():
        contrib = np.mean(v**2) / (np.mean(v_model**2) + 1e-10) * 100
        print(f"      {name}: {np.mean(v):.1f} km/s ({contrib:.1f}%)")

    return {
        "period_name": period_name,
        "multiplier_func": multiplier_func.__name__,
        "params": result.x.tolist(),
        "rms": float(rms),
        "chi_squared": float(result.fun),
        "time": t1 - t0,
        "components": {k: float(np.mean(v)) for k, v in components.items()},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Analytic backbone analysis")
    parser.add_argument(
        "--obs-mode",
        choices=["ring", "global"],
        default="ring",
        help="Observation sampling mode.",
    )
    parser.add_argument(
        "--n-obs",
        type=int,
        default=500,
        help="Number of observation stars to sample.",
    )
    parser.add_argument(
        "--rmin",
        type=float,
        default=7.0,
        help="Minimum cylindrical radius (kpc) when obs-mode=ring.",
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=9.0,
        help="Maximum cylindrical radius (kpc) when obs-mode=ring.",
    )
    parser.add_argument(
        "--results-path",
        default="gravitywavebaseline/backbone_analysis_results.json",
        help="Output JSON path for aggregated results.",
    )
    return parser.parse_args()


def run_backbone_analysis(args):
    print("=" * 80)
    print("ANALYTIC BACKBONE + lambda_gw PERTURBATION ANALYSIS")
    print("=" * 80)
    print("\nApproach:")
    print("  1. Analytic disk+bulge+halo -> baseline v~220 km/s")
    print("  2. Stellar lambda_gw multipliers -> perturbation on top")
    print("  3. Optimize f(lambda_gw) to match Gaia v_phi")

    print("\nLoading Gaia data...")
    gaia = pd.read_parquet("gravitywavebaseline/gaia_with_periods.parquet")
    print(f"  Loaded {len(gaia):,} stars")

    if "v_phi" in gaia.columns:
        mask = np.isfinite(gaia["v_phi"]) & (gaia["v_phi"] > 0)
        if args.obs_mode == "ring":
            window_mask = mask & (gaia["R"].values >= args.rmin) & (
                gaia["R"].values <= args.rmax
            )
            mode_label = f"ring ({args.rmin:.1f}-{args.rmax:.1f} kpc)"
        else:
            window_mask = mask
            mode_label = "global"

        candidate_idx = np.where(window_mask)[0]
        if len(candidate_idx) == 0:
            raise ValueError(
                f"No valid stars found for obs-mode={args.obs_mode} "
                f"and window r=[{args.rmin}, {args.rmax}] kpc"
            )

        sample_size = min(args.n_obs, len(candidate_idx))
        obs_idx = np.random.choice(candidate_idx, sample_size, replace=False)
        R_obs = gaia.iloc[obs_idx]["R"].values
        v_observed = gaia.iloc[obs_idx]["v_phi"].values
        print(
            f"\nObservations: mode={mode_label}, pool={len(candidate_idx)}, "
            f"sampled={len(obs_idx)}"
        )
        print(f"  R range: {R_obs.min():.2f} - {R_obs.max():.2f} kpc")
        print(f"  v_phi range: {v_observed.min():.1f} - {v_observed.max():.1f} km/s")
        print(f"  v_phi mean: {v_observed.mean():.1f} km/s")
    else:
        R_obs = np.linspace(4, 16, 50)
        v_observed = np.ones_like(R_obs) * 220.0
        print("\nUsing synthetic flat curve at 220 km/s")

    print("\n" + "=" * 80)
    print("TEST CONFIGURATIONS")
    print("=" * 80)

    configs = [
        {
            "name": "NO HALO (baryons only)",
            "use_halo": False,
            "M_disk": 6e10,
            "M_bulge": 1.5e10,
            "M_200": 0,
        },
        {
            "name": "WITH HALO (standard LCDM)",
            "use_halo": True,
            "M_disk": 6e10,
            "M_bulge": 1.5e10,
            "M_200": 1.5e12,
        },
    ]

    multiplier_tests = [
        ("shortlambda_boost", multiplier_shortlambda_boost, [(0.0, 5.0), (0.5, 60.0), (0.5, 4.0)]),
        ("shortlambda_sat", multiplier_shortlambda_saturating, [(0.0, 5.0), (0.5, 60.0), (0.5, 4.0)]),
        ("constant", multiplier_constant, [(1, 10)]),
        ("power_law", multiplier_power_law, [(0, 5), (0.1, 50), (0.5, 4)]),
        ("richter", multiplier_richter_scale, [(0.01, 100), (0.5, 3)]),
    ]

    all_results = []

    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"CONFIG: {config['name']}")
        print(f"{'=' * 80}")

        calib = calibrate_masses(
            R_obs,
            v_observed,
            config["use_halo"],
            config["M_disk"],
            config["M_bulge"],
            config["M_200"],
        )
        if not calib["success"]:
            print(f"  [!] Mass calibration failed: {calib['message']}")
            masses = {
                "M_disk": config["M_disk"],
                "M_bulge": config["M_bulge"],
                "M_200": config["M_200"],
            }
        else:
            masses = calib
            print(
                "  Calibrated masses:"
                f" disk={masses['M_disk']:.2e} Msun,"
                f" bulge={masses['M_bulge']:.2e} Msun,"
                f" halo={masses['M_200']:.2e} Msun"
            )

        calc = AnalyticBackboneCalculator(
            gaia,
            use_gpu=GPU_AVAILABLE,
            M_disk=masses["M_disk"],
            M_bulge=masses["M_bulge"],
            M_200=masses["M_200"],
            use_halo=config["use_halo"],
            n_sample_stars=50000,
        )

        print("\n  Baseline (analytic only):")
        v_baseline, _ = calc.compute_velocity(R_obs, z_obs=0)
        rms_baseline = np.sqrt(np.mean((v_baseline - v_observed) ** 2))
        print(f"    RMS: {rms_baseline:.1f} km/s")

        for mult_name, mult_func, bounds in multiplier_tests:
            try:
                result = optimize_with_backbone(
                    calc, R_obs, v_observed, "gw", mult_func, bounds
                )
                result["config"] = config["name"]
                result["rms_baseline"] = float(rms_baseline)
                result["improvement"] = float(rms_baseline - result["rms"])
                all_results.append(result)
            except Exception as e:
                print(f"    [!] Failed: {e}")

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    results_sorted = sorted(all_results, key=lambda x: x["rms"])
    print(
        f"\n{'Rank':<5} {'Config':<25} {'Multiplier':<20} "
        f"{'Baseline':<12} {'RMS':<12} {'dRMS':<10}"
    )
    print("-" * 95)
    for i, res in enumerate(results_sorted, 1):
        print(
            f"{i:<5} {res['config']:<25} {res['multiplier_func']:<20} "
            f"{res['rms_baseline']:<12.1f} {res['rms']:<12.1f} {res['improvement']:<10.1f}"
        )

    output = args.results_path
    with open(output, "w") as f:
        json.dump(results_sorted, f, indent=2)
    print(f"\n[OK] Saved: {output}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    no_halo_results = [r for r in results_sorted if "NO HALO" in r["config"]]
    with_halo_results = [r for r in results_sorted if "WITH HALO" in r["config"]]

    if no_halo_results:
        best_no_halo = no_halo_results[0]
        print("\nBest WITHOUT halo (baryons + lambda multipliers):")
        print(f"  RMS: {best_no_halo['rms']:.1f} km/s")
        print(
            f"  Improvement over baryons-only: {best_no_halo['improvement']:.1f} km/s"
        )
        print(f"  Multiplier: {best_no_halo['multiplier_func']}")
        print(f"  Params: {best_no_halo['params']}")

    if with_halo_results:
        best_with_halo = with_halo_results[0]
        print("\nBest WITH halo (LCDM + lambda multipliers):")
        print(f"  RMS: {best_with_halo['rms']:.1f} km/s")
        print(
            f"  Improvement over LCDM: {best_with_halo['improvement']:.1f} km/s"
        )

    return results_sorted


if __name__ == "__main__":
    ARGS = parse_args()
    RESULTS = run_backbone_analysis(ARGS)

