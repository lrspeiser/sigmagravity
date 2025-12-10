def distribute_formula_spillover(
    R_obs,
    values,
    shell_edges,
    shell_weights,
    total_budget_factor=1.0,
):
    values = np.asarray(values, dtype=np.float64)
    R_obs = np.asarray(R_obs, dtype=np.float64)
    if values.size == 0:
        return values

    shell_weights = np.asarray(shell_weights, dtype=np.float64)
    if shell_weights.size == 0 or np.all(shell_weights <= 0):
        return values

    n_shells = len(shell_edges) - 1
    if n_shells <= 0:
        return values

    shell_indices = np.clip(
        np.searchsorted(shell_edges, R_obs, side="right") - 1, 0, n_shells - 1
    )
    demand_abs = np.abs(values)
    raw_required = float(np.sum(demand_abs))
    if raw_required <= 0:
        return np.zeros_like(values)

    weights = np.clip(shell_weights, 0.0, None)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights /= weights.sum()

    provided_shell = total_budget_factor * raw_required * weights
    provided = np.zeros_like(values, dtype=np.float64)
    for j in range(n_shells):
        mask = shell_indices == j
        if not np.any(mask):
            continue
        shell_demand = float(np.sum(demand_abs[mask]))
        if shell_demand <= 0:
            provided[mask] = provided_shell[j] / mask.sum()
            continue
        local_weights = demand_abs[mask] / shell_demand
        signed = np.sign(values[mask])
        provided[mask] = provided_shell[j] * local_weights * np.where(
            signed == 0, 1.0, signed
        )
    return provided

"""
Apply the Milky Way capacity law to a SPARC rotation curve.

Usage:
  python gravitywavebaseline/sparc_capacity_test.py --galaxy NGC2403 \
      --alpha 2.8271 --gamma 0.8579
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from test_lambda_enhancement import (
    apply_capacity_to_enhancement,
    force_match_enhancement,
    multiplier_shortlambda_boost,
    multiplier_shortlambda_saturating,
)

G_KPC = 4.30091e-6  # (km/s)^2 * kpc / Msun
KM_TO_M = 1000.0
KPC_TO_M = 3.085677581491367e19


def estimate_lambda_values(R, V_gr, spectrum_mode="orbital"):
    R = np.asarray(R, dtype=np.float64)
    spectrum_mode = (spectrum_mode or "orbital").lower()
    if spectrum_mode == "orbital":
        return 2.0 * np.pi * np.maximum(R, 1e-3)
    if spectrum_mode == "dynamical":
        return np.maximum(R, 1e-3)
    if spectrum_mode == "velocity":
        V_safe = np.maximum(V_gr, 1.0)
        period = 2.0 * np.pi * R / V_safe
        return np.maximum(period * V_safe, 1e-3)
    raise ValueError(f"Unknown lambda spectrum mode '{spectrum_mode}'")


def evaluate_lambda_multiplier(lambda_vals, mode, params):
    if mode == "off":
        return np.ones_like(lambda_vals)
    mode = mode.lower()
    if mode == "short_boost":
        func = multiplier_shortlambda_boost
    elif mode == "short_sat":
        func = multiplier_shortlambda_saturating
    else:
        raise ValueError(f"Unsupported lambda-mode '{mode}'")
    lam = np.asarray(lambda_vals, dtype=np.float64)
    multiplier = func(lam, None, params, np)
    return np.clip(np.asarray(multiplier, dtype=np.float64), 0.0, None)


def read_sparc_rotmod(rotmod_path):
    df = pd.read_csv(
        rotmod_path,
        comment="#",
        delim_whitespace=True,
        names=["R", "Vobs", "errV", "Vgas", "Vdisk", "Vbulge", "SBdisk", "SBbulge"],
    )
    df = df.dropna()
    df = df[df["R"] > 0]
    return df.reset_index(drop=True)


def compute_shell_edges_data(R):
    R = np.asarray(R, dtype=np.float64)
    if len(R) < 2:
        delta = 0.1 if len(R) == 0 else 0.5 * R[0]
        return np.array([max(R[0] - delta, 0.0), R[0] + delta])
    mids = 0.5 * (R[:-1] + R[1:])
    edges = np.concatenate(([R[0] * 0.8], mids, [R[-1] * 1.2]))
    edges = np.maximum.accumulate(edges)
    edges[0] = max(edges[0], 1e-4)
    return edges


def compute_shell_edges(R):
    """Backward-compatible alias for older callers."""
    return compute_shell_edges_data(R)


def compute_shell_edges_adaptive(
    R,
    props=None,
    r_coherence_factor=4.0,
    n_shell_min=8,
    n_shell_max=40,
):
    """
    Build galaxy-dependent shells with a finite coherence radius.
    """
    R = np.asarray(R, dtype=np.float64)
    R = R[R > 0]
    if R.size == 0:
        raise ValueError("No positive radii in rotation curve.")

    if props and props.get("R_disk", 0) > 0:
        R_disk = float(props["R_disk"])
    else:
        R_disk = float(np.median(R))

    R_coh = r_coherence_factor * R_disk

    R_min = float(np.min(R))
    R_max_data = float(np.max(R))
    R_max = min(R_coh, 1.2 * R_max_data)

    if props and props.get("M_baryon", 0) > 0:
        M_baryon = float(props["M_baryon"])
    else:
        M_baryon = 1.0e10

    logM = np.log10(max(M_baryon, 1.0e8))
    n_shells = int(10 + 2 * (logM - 9.0))
    n_shells = int(np.clip(n_shells, n_shell_min, n_shell_max))

    inner = R_min * 0.8
    outer = max(R_max * 1.05, inner * 1.1)
    log_inner = np.log(max(inner, 1e-3))
    log_outer = np.log(max(outer, 1.1 * inner))

    u = np.linspace(0.0, 1.0, n_shells + 1)
    edges = np.exp(log_inner + u * (log_outer - log_inner))
    edges = np.maximum.accumulate(edges)
    edges[0] = max(edges[0], 1e-3)
    return edges, R_coh


def compute_shell_edges_mass_adaptive(
    R,
    surface_density,
    props=None,
    r_coherence_factor=4.0,
    n_shell_min=8,
    n_shell_max=40,
):
    """
    Build galaxy-dependent shells using the baryonic surface-density profile.
    """
    R = np.asarray(R, dtype=np.float64)
    Sigma = np.asarray(surface_density, dtype=np.float64)

    mask = (R > 0) & np.isfinite(R) & np.isfinite(Sigma) & (Sigma >= 0)
    if not np.any(mask):
        raise ValueError("No usable (R, Sigma) pairs for mass-adaptive shells.")

    R = R[mask]
    Sigma = Sigma[mask]

    if props and props.get("R_disk", 0) > 0:
        R_disk = float(props["R_disk"])
    else:
        R_disk = float(np.median(R))

    if props and props.get("M_baryon", 0) > 0:
        M_baryon = float(props["M_baryon"])
    else:
        M_baryon = 1.0e10

    R_coh = r_coherence_factor * R_disk

    inside = R <= R_coh
    R_in = R[inside]
    Sigma_in = Sigma[inside]
    if R_in.size < 2:
        edges = compute_shell_edges_data(R)
        return edges, R_coh

    edges_data = compute_shell_edges_data(R_in)
    widths = np.diff(edges_data)
    centers = 0.5 * (edges_data[:-1] + edges_data[1:])

    Sigma_centers = np.interp(centers, R_in, Sigma_in, left=0.0, right=0.0)
    ring_area = 2.0 * np.pi * np.maximum(centers, 1e-3) * widths
    dM = Sigma_centers * ring_area
    total_mass = dM.sum()
    if not np.isfinite(total_mass) or total_mass <= 0:
        edges = compute_shell_edges_data(R)
        return edges, R_coh

    logM = np.log10(max(M_baryon, 1.0e8))
    n_shells = int(10 + 2 * (logM - 9.0))
    n_shells = int(np.clip(n_shells, n_shell_min, n_shell_max))
    n_shells = max(2, min(n_shells, dM.size))

    mass_per_shell = total_mass / n_shells
    cum_mass = np.cumsum(dM)

    new_edges = [edges_data[0]]
    targets = np.linspace(mass_per_shell, total_mass - mass_per_shell, n_shells - 1)
    for m_target in targets:
        R_edge = np.interp(m_target, cum_mass, centers)
        new_edges.append(R_edge)
    new_edges.append(edges_data[-1])

    edges = np.array(new_edges, dtype=np.float64)
    edges = np.maximum.accumulate(edges)
    edges[0] = max(edges[0], 1e-3)
    return edges, R_coh


def compute_shell_edges_gravity_adaptive(
    R,
    surface_density,
    props=None,
    shell_r_factor=4.0,
    shell_n_min=8,
    shell_n_max=40,
    g_floor=1e-5,
    coh_radius_factor=3.0,
):
    """
    Build shells that follow baryonic gravity (1/R^2) with a finite cutoff.
    """
    R = np.asarray(R, dtype=float)
    if R.size == 0:
        raise ValueError("No radii provided for gravity-adaptive shells.")
    R_sorted = np.sort(R[R > 0])
    if R_sorted.size == 0:
        raise ValueError("No positive radii for gravity-adaptive shells.")
    R_min = max(R_sorted[0], 1e-3)
    R_max_obs = R_sorted[-1]

    props = props or {}
    M_baryon = float(props.get("M_baryon", 5.0e10))
    R_disk = float(props.get("R_disk", np.median(R_sorted)))
    if not np.isfinite(R_disk) or R_disk <= 0:
        R_disk = np.median(R_sorted)

    # Outer cutoff from Newtonian gravity floor
    g_floor = max(float(g_floor), 1e-10)
    R_stop_phys = math.sqrt(G_KPC * max(M_baryon, 1.0e8) / g_floor)
    R_stop_max = shell_r_factor * R_max_obs
    R_stop = float(np.clip(R_stop_phys, R_max_obs, R_stop_max))

    # Coherence radius where shells switch from linear to geometric spacing
    R_coh_nom = coh_radius_factor * R_disk
    R_coh_min = 2.0 * R_disk
    R_coh_max = 0.8 * R_stop
    R_coh = float(np.clip(R_coh_nom, R_coh_min, R_coh_max))
    R_coh = max(R_coh, R_min * 1.5)

    if R_stop <= R_min:
        edges = np.array([R_min, R_min * 1.1])
        return edges, R_min, R_stop

    span = R_stop - R_min
    frac_inner = np.clip((R_coh - R_min) / span, 0.3, 0.8)
    n_total = int(np.clip(shell_n_max, shell_n_min, 80))
    n_inner = max(shell_n_min, int(frac_inner * n_total))
    n_outer = max(shell_n_min // 2, n_total - n_inner)
    if n_outer < 0:
        n_outer = 0

    inner_edges = np.linspace(R_min, R_coh, n_inner + 1)
    if n_outer > 0 and R_stop > R_coh:
        outer_edges = np.geomspace(R_coh, R_stop, n_outer + 1)[1:]
        edges = np.concatenate([inner_edges, outer_edges])
    else:
        edges = inner_edges

    edges = np.maximum.accumulate(edges)
    edges[0] = max(edges[0], 1e-3)
    return edges, R_coh, R_stop


def compute_shell_mass_profile(edges, radii, surface_density):
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    if len(centers) == 0:
        return centers, widths, np.array([]), np.array([])

    sigma_interp = np.interp(
        centers,
        radii,
        surface_density,
        left=surface_density[0] if len(surface_density) else 0.0,
        right=surface_density[-1] if len(surface_density) else 0.0,
    )
    area = 2.0 * np.pi * np.maximum(centers, 1e-6) * widths
    shell_mass = sigma_interp * area
    shell_mass = np.clip(shell_mass, 0.0, None)
    mass_cumulative = np.cumsum(shell_mass)
    return centers, widths, shell_mass, mass_cumulative


def compute_formula_weights(
    mode,
    shell_centers,
    shell_mass,
    shell_mass_cum,
    R_disk,
    R_stop,
    params,
    galaxy_props,
):
    if len(shell_centers) == 0:
        return np.array([])

    alpha, beta, gamma, delta, g_cut = params
    R_disk = max(R_disk, 1e-3)
    R_stop = max(R_stop, shell_centers[-1] if len(shell_centers) else R_disk)
    total_mass = shell_mass_cum[-1] if len(shell_mass_cum) else 0.0
    if total_mass <= 0:
        frac_mass = np.ones_like(shell_centers) / len(shell_centers)
        total_mass = 1.0
    else:
        frac_mass = shell_mass / total_mass

    M_enc = shell_mass_cum
    g_bar = G_KPC * np.maximum(M_enc, 0.0) / np.maximum(shell_centers**2, 1e-6)
    g_cut = max(g_cut, 1e-10)

    weights = np.ones_like(shell_centers)
    radial = np.maximum(shell_centers / R_disk, 1e-3)
    taper_grav = (g_bar / (g_bar + g_cut)) ** gamma
    taper_outer = np.exp(-np.power(shell_centers / R_stop, max(delta, 0.0)))
    radius_term = radial**beta

    if mode == "mass_formula":
        weights = (frac_mass**alpha) * radius_term * taper_grav
    elif mode == "radius_formula":
        denom = (1.0 + (shell_centers / R_stop)) ** max(delta, 0.0)
        weights = radius_term / denom
    elif mode == "hybrid":
        weights = (frac_mass**alpha) * radius_term * taper_grav * taper_outer
    else:
        weights = np.ones_like(shell_centers)

    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0:
        weights = np.ones_like(shell_centers)
        total = weights.sum()
    return weights / total


def load_galaxy_properties(summary_path, galaxy_name):
    """Return a dict of global SPARC properties for the requested galaxy."""
    summary = pd.read_csv(summary_path)
    summary["galaxy_name"] = summary["galaxy_name"].str.strip()
    match = summary[summary["galaxy_name"].str.upper() == galaxy_name.upper()]
    if match.empty:
        raise ValueError(
            f"Galaxy '{galaxy_name}' not found in summary {summary_path}"
        )
    return match.iloc[0].to_dict()


def scale_alpha(base_alpha, mode, props, *, mass_ref, sigma_ref, beta, lo, hi):
    alpha = float(base_alpha)
    mode = (mode or "constant").lower()
    if mode == "mass_power":
        mass = float(props.get("M_baryon", 0.0))
        if mass > 0 and mass_ref > 0:
            alpha *= (mass / mass_ref) ** beta
    elif mode == "sigma_power":
        sigma = float(props.get("sigma_velocity", 0.0))
        if sigma > 0 and sigma_ref > 0:
            alpha *= (sigma / sigma_ref) ** beta
    elif mode not in ("constant", ""):
        raise ValueError(f"Unknown alpha scaling '{mode}'")
    return float(np.clip(alpha, lo, hi))


def scale_gamma(base_gamma, mode, props, *, mass_ref, sigma_ref, beta, lo, hi):
    gamma = float(base_gamma)
    mode = (mode or "constant").lower()
    if mode == "mass_log":
        mass = float(props.get("M_baryon", 0.0))
        if mass > 0 and mass_ref > 0:
            gamma += beta * np.log10(mass / mass_ref)
    elif mode == "sigma_log":
        sigma = float(props.get("sigma_velocity", 0.0))
        if sigma > 0 and sigma_ref > 0:
            gamma += beta * np.log10(sigma / sigma_ref)
    elif mode not in ("constant", ""):
        raise ValueError(f"Unknown gamma scaling '{mode}'")
    return float(np.clip(gamma, lo, hi))


def build_capacity(surface_density, radii, alpha, gamma, R_coh=None, p_cut=2.0):
    radii = np.asarray(radii, dtype=np.float64)
    base = np.asarray(surface_density, dtype=np.float64)
    r_norm = np.maximum(radii, 1e-3)
    r_ref = np.median(r_norm[base > 0]) if np.any(base > 0) else np.median(r_norm)
    growth = np.power(r_norm / r_ref, gamma)
    capacity = alpha * base * growth
    if R_coh is not None and R_coh > 0.0:
        taper = np.exp(-np.power(r_norm / R_coh, p_cut))
        capacity = capacity * taper
    return np.maximum(capacity, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Apply MW capacity law to SPARC galaxy.")
    parser.add_argument("--galaxy", required=True, help="Galaxy name, e.g., NGC2403")
    parser.add_argument("--rotmod-dir", default="data/Rotmod_LTG", help="Directory with *_rotmod.dat files")
    parser.add_argument("--alpha", type=float, default=2.8271, help="Capacity scaling coefficient")
    parser.add_argument("--gamma", type=float, default=0.8579, help="Capacity radial exponent")
    parser.add_argument("--force-match", action="store_true", help="After applying capacity, force v_model = v_obs")
    parser.add_argument("--output-dir", default="gravitywavebaseline/sparc_results", help="Directory to save JSON output")
    parser.add_argument("--sparc-summary", default="data/sparc/sparc_combined.csv",
                        help="CSV with global SPARC properties.")
    parser.add_argument("--shell-mode", choices=["data", "adaptive", "mass_adaptive", "gravity_adaptive"], default="data",
                        help="How to define capacity shells.")
    parser.add_argument("--shell-r-factor", type=float, default=4.0,
                        help="Coherence radius factor (R_coh = factor * R_disk).")
    parser.add_argument("--shell-n-min", type=int, default=8,
                        help="Minimum number of shells in adaptive mode.")
    parser.add_argument("--shell-n-max", type=int, default=40,
                        help="Maximum number of shells in adaptive mode.")
    parser.add_argument("--alpha-scaling", choices=["constant", "mass_power", "sigma_power"],
                        default="constant", help="Scaling rule for alpha based on galaxy properties.")
    parser.add_argument("--gamma-scaling", choices=["constant", "mass_log", "sigma_log"],
                        default="constant", help="Scaling rule for gamma based on galaxy properties.")
    parser.add_argument("--mass-ref", type=float, default=6.0e10,
                        help="Reference baryonic mass for scaling (Msun).")
    parser.add_argument("--sigma-ref", type=float, default=30.0,
                        help="Reference velocity dispersion (km/s).")
    parser.add_argument("--alpha-beta", type=float, default=0.0,
                        help="Exponent for alpha scaling (positive increases with property).")
    parser.add_argument("--gamma-beta", type=float, default=0.0,
                        help="Slope for gamma scaling in log-space.")
    parser.add_argument("--alpha-min", type=float, default=0.01,
                        help="Lower bound for scaled alpha.")
    parser.add_argument("--alpha-max", type=float, default=100.0,
                        help="Upper bound for scaled alpha.")
    parser.add_argument("--gamma-min", type=float, default=-5.0,
                        help="Lower bound for scaled gamma.")
    parser.add_argument("--gamma-max", type=float, default=5.0,
                        help="Upper bound for scaled gamma.")
    parser.add_argument("--budget-factor", type=float, default=1.0,
                        help="Total enhancement budget relative to sum(V_req).")
    parser.add_argument("--g-floor", type=float, default=1e-5,
                        help="Gravity floor (km^2/s^2/kpc) for gravity-adaptive shells.")
    parser.add_argument("--coh-radius-factor", type=float, default=3.0,
                        help="Coherence radius factor for gravity-adaptive shells.")
    parser.add_argument("--spillover-mode", choices=["capped", "mass_formula", "radius_formula", "hybrid"],
                        default="capped", help="Select spillover weighting strategy.")
    parser.add_argument("--spill-alpha", type=float, default=0.5,
                        help="Mass exponent for formula spillover.")
    parser.add_argument("--spill-beta", type=float, default=1.0,
                        help="Radial exponent for formula spillover.")
    parser.add_argument("--spill-gamma", type=float, default=1.0,
                        help="Gravity taper exponent for formula spillover.")
    parser.add_argument("--spill-delta", type=float, default=2.0,
                        help="Outer taper exponent for hybrid/radius spillover.")
    parser.add_argument("--spill-gcut", type=float, default=1e-5,
                        help="Gravity cut scale for formula spillover.")
    parser.add_argument("--lambda-mode", choices=["off", "short_boost", "short_sat"],
                        default="off", help="Apply lambda_gw multiplier before capacity.")
    parser.add_argument("--lambda-A", type=float, default=5.0,
                        help="Amplitude for lambda multiplier.")
    parser.add_argument("--lambda-l0", type=float, default=30.0,
                        help="Reference lambda scale (kpc).")
    parser.add_argument("--lambda-alpha", type=float, default=1.5,
                        help="Power/slope for lambda multiplier.")
    parser.add_argument("--lambda-spectrum", choices=["orbital", "dynamical", "velocity"],
                        default="orbital", help="How to estimate lambda per radius.")
    parser.add_argument("--lambda-max-mult", type=float, default=50.0,
                        help="Clip lambda multiplier to avoid runaway.")
    parser.add_argument("--lambda-gate-sigma", action="store_true",
                        help="Scale lambda amplitude by galaxy sigma_velocity.")
    parser.add_argument("--lambda-sigma-beta", type=float, default=0.4,
                        help="Exponent for sigma gating (A_eff = A * clamp(sigma_ref/sigma)^beta).")
    parser.add_argument("--lambda-gate-accel", action="store_true",
                        help="Gate lambda multiplier by baryonic acceleration.")
    parser.add_argument("--lambda-accel-gdag", type=float, default=1e-10,
                        help="Acceleration scale g_dag (m/s^2) for S(g_bar).")
    parser.add_argument("--lambda-accel-q", type=float, default=2.0,
                        help="Power q in S(g_bar) = 1 / (1 + (g_bar/g_dag)^q).")
    parser.add_argument("--pure-lambda", action="store_true",
                        help="Use lambda multiplier as direct GR modifier, no shells/budgets.")
    parser.add_argument("--disable-capacity", action="store_true",
                        help="Skip capacity/spillover and use raw enhancement.")
    args = parser.parse_args()

    rotmod_path = Path(args.rotmod_dir) / f"{args.galaxy}_rotmod.dat"
    if not rotmod_path.exists():
        raise FileNotFoundError(f"SPARC rotmod file not found: {rotmod_path}")

    df = read_sparc_rotmod(rotmod_path)
    if len(df) == 0:
        raise ValueError(f"No usable data points in {rotmod_path}")

    R = df["R"].values
    Vobs = df["Vobs"].values
    Vgas = df["Vgas"].values
    Vdisk = df["Vdisk"].values
    Vbulge = df["Vbulge"].values
    SBdisk = df["SBdisk"].values
    SBbulge = df["SBbulge"].values

    try:
        props = load_galaxy_properties(args.sparc_summary, args.galaxy)
    except Exception as exc:
        print(f"[WARN] {exc}")
        props = {}

    alpha_eff = scale_alpha(
        args.alpha,
        args.alpha_scaling,
        props,
        mass_ref=args.mass_ref,
        sigma_ref=args.sigma_ref,
        beta=args.alpha_beta,
        lo=args.alpha_min,
        hi=args.alpha_max,
    )
    gamma_eff = scale_gamma(
        args.gamma,
        args.gamma_scaling,
        props,
        mass_ref=args.mass_ref,
        sigma_ref=args.sigma_ref,
        beta=args.gamma_beta,
        lo=args.gamma_min,
        hi=args.gamma_max,
    )

    V_gr = np.sqrt(np.maximum(Vgas**2 + Vdisk**2 + Vbulge**2, 0.0))
    V_req = np.sqrt(np.maximum(Vobs**2 - V_gr**2, 0.0))
    sigma_value = float(props.get("sigma_velocity", args.sigma_ref)) if props else args.sigma_ref
    lambda_A_eff = args.lambda_A
    lambda_vals = None
    lambda_multiplier = None
    sigma_gate_value = 1.0
    accel_gate = None
    V_raw = V_req.copy()
    if args.lambda_mode != "off":
        if args.lambda_gate_sigma:
            sigma_safe = max(sigma_value, 1e-6)
            ratio = args.sigma_ref / sigma_safe
            sigma_gate_value = min(1.0, ratio ** args.lambda_sigma_beta)
        lambda_A_eff = args.lambda_A * sigma_gate_value
        lambda_vals = estimate_lambda_values(R, V_gr, args.lambda_spectrum)
        lambda_params = (lambda_A_eff, args.lambda_l0, args.lambda_alpha)
        lambda_multiplier = evaluate_lambda_multiplier(lambda_vals, args.lambda_mode, lambda_params)
        lambda_multiplier = np.maximum(lambda_multiplier, 1.0)
        if args.lambda_max_mult > 0:
            lambda_multiplier = np.minimum(lambda_multiplier, args.lambda_max_mult)
        if args.lambda_gate_accel:
            g_dag = max(args.lambda_accel_gdag, 1e-15)
            q_gate = max(args.lambda_accel_q, 0.1)
            g_bar = np.zeros_like(V_gr, dtype=np.float64)
            mask_r = R > 0
            if np.any(mask_r):
                g_bar[mask_r] = ((V_gr[mask_r] * KM_TO_M) ** 2) / (np.maximum(R[mask_r], 1e-6) * KPC_TO_M)
            accel_gate = 1.0 / (1.0 + np.power(np.maximum(g_bar / g_dag, 0.0), q_gate))
        else:
            accel_gate = np.ones_like(lambda_multiplier, dtype=np.float64)
        boost = np.maximum(lambda_multiplier - 1.0, 0.0) * accel_gate
        final_multiplier = 1.0 + boost
        V_raw = np.sqrt(np.maximum((final_multiplier - 1.0) * np.maximum(V_gr**2, 0.0), 0.0))
        lambda_multiplier = final_multiplier

    surface_density = SBdisk + SBbulge

    props_or_none = props if props else None
    if args.shell_mode == "adaptive":
        edges, R_coh = compute_shell_edges_adaptive(
            R,
            props=props_or_none,
            r_coherence_factor=args.shell_r_factor,
            n_shell_min=args.shell_n_min,
            n_shell_max=args.shell_n_max,
        )
        R_stop = edges[-1]
    elif args.shell_mode == "mass_adaptive":
        edges, R_coh = compute_shell_edges_mass_adaptive(
            R,
            surface_density,
            props=props_or_none,
            r_coherence_factor=args.shell_r_factor,
            n_shell_min=args.shell_n_min,
            n_shell_max=args.shell_n_max,
        )
        R_stop = edges[-1]
    elif args.shell_mode == "gravity_adaptive":
        edges, R_coh, R_stop = compute_shell_edges_gravity_adaptive(
            R,
            surface_density,
            props=props_or_none,
            shell_r_factor=args.shell_r_factor,
            shell_n_min=args.shell_n_min,
            shell_n_max=args.shell_n_max,
            g_floor=args.g_floor,
            coh_radius_factor=args.coh_radius_factor,
        )
        print(f"[shells] gravity_adaptive: R_coh={R_coh:.2f} kpc, R_stop={R_stop:.2f} kpc, shells={len(edges)-1}")
    else:
        edges = compute_shell_edges_data(R)
        R_coh = None
        R_stop = edges[-1]

    shell_centers, shell_widths, shell_mass, shell_mass_cum = compute_shell_mass_profile(
        edges, R, surface_density
    )
    R_disk = float(props.get("R_disk", np.median(R))) if props else float(np.median(R))
    if not np.isfinite(R_disk) or R_disk <= 0:
        R_disk = float(np.median(R))

    capacity = None
    if not args.disable_capacity and not args.pure_lambda:
        capacity = build_capacity(surface_density, R, alpha_eff, gamma_eff, R_coh=R_coh)

    if args.pure_lambda:
        provided = V_raw.copy()
        spill_weights = None
    elif args.disable_capacity:
        provided = V_raw.copy()
        spill_weights = None
    elif args.spillover_mode == "capped":
        provided = apply_capacity_to_enhancement(
            R,
            V_raw,
            edges,
            capacity,
            total_budget_factor=args.budget_factor,
        )
        spill_weights = None
    else:
        spill_params = (
            args.spill_alpha,
            args.spill_beta,
            args.spill_gamma,
            args.spill_delta,
            args.spill_gcut,
        )
        weights = compute_formula_weights(
            args.spillover_mode,
            shell_centers,
            shell_mass,
            shell_mass_cum,
            R_disk,
            R_stop,
            spill_params,
            props,
        )
        provided = distribute_formula_spillover(
            R,
            V_raw,
            edges,
            weights,
            total_budget_factor=args.budget_factor,
        )
        spill_weights = weights.tolist() if len(weights) else None

    if args.force_match:
        provided = force_match_enhancement(V_gr, Vobs, provided)

    V_model = np.sqrt(np.maximum(V_gr**2 + provided**2, 0.0))
    residuals = V_model - Vobs
    rms_vel = float(np.sqrt(np.mean(residuals**2)))

    print(f"Galaxy: {args.galaxy}")
    print(f"Data points: {len(R)}")
    print(f"Alpha: {args.alpha:.4f} -> scaled {alpha_eff:.4f} ({args.alpha_scaling})")
    print(f"Gamma: {args.gamma:.4f} -> scaled {gamma_eff:.4f} ({args.gamma_scaling})")
    if R_coh is not None:
        print(f"Shell mode: {args.shell_mode} (R_coh={R_coh:.2f} kpc)")
    else:
        print(f"Shell mode: {args.shell_mode}")
    if args.lambda_mode != "off" and lambda_multiplier is not None:
        print(
            f"Lambda mode: {args.lambda_mode} ({args.lambda_spectrum}), "
            f"multiplier range {np.min(lambda_multiplier):.2f}-{np.max(lambda_multiplier):.2f} "
            f"(median {np.median(lambda_multiplier):.2f})"
        )
        print(f"  sigma gating: sigma={sigma_value:.2f} km/s -> A_eff={lambda_A_eff:.3f}")
        if args.lambda_gate_accel and accel_gate is not None:
            print(
                f"  accel gate S(g): min {np.min(accel_gate):.3f}, "
                f"median {np.median(accel_gate):.3f}, max {np.max(accel_gate):.3f}"
            )
    if args.pure_lambda:
        print("Capacity: PURE LAMBDA MODE (no shells/budgets)")
    elif args.disable_capacity:
        print("Capacity: DISABLED (using raw enhancement)")
    else:
        print("Capacity: ENABLED (shell spillover active)")
    print(f"Budget factor: {args.budget_factor:.2f}")
    print(f"RMS velocity error: {rms_vel:.2f} km/s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.galaxy}_capacity_test.json"

    report = {
        "galaxy": args.galaxy,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "alpha_eff": alpha_eff,
        "gamma_eff": gamma_eff,
        "alpha_scaling": args.alpha_scaling,
        "gamma_scaling": args.gamma_scaling,
        "shell_mode": args.shell_mode,
        "shell_parameters": {
            "R_coh": R_coh,
            "R_stop": R_stop,
            "r_factor": args.shell_r_factor,
            "n_shell_min": args.shell_n_min,
            "n_shell_max": args.shell_n_max,
            "g_floor": args.g_floor,
            "coh_radius_factor": args.coh_radius_factor,
            "spillover_mode": args.spillover_mode,
            "spill_weights": spill_weights,
        },
        "disable_capacity": bool(args.disable_capacity),
        "pure_lambda": bool(args.pure_lambda),
        "budget_factor": args.budget_factor,
        "spillover_mode": args.spillover_mode,
        "spillover_params": {
            "alpha": args.spill_alpha,
            "beta": args.spill_beta,
            "gamma": args.spill_gamma,
            "delta": args.spill_delta,
            "gcut": args.spill_gcut,
        },
        "lambda_settings": {
            "mode": args.lambda_mode,
            "spectrum": args.lambda_spectrum,
            "max_multiplier": args.lambda_max_mult,
            "A": args.lambda_A,
            "A_effective": float(lambda_A_eff) if args.lambda_mode != "off" else None,
            "lambda0": args.lambda_l0,
            "alpha": args.lambda_alpha,
            "sigma_gate": {
                "enabled": bool(args.lambda_gate_sigma),
                "beta": args.lambda_sigma_beta,
                "sigma_value": float(sigma_value),
                "sigma_ref": args.sigma_ref,
                "gate": float(sigma_gate_value) if args.lambda_mode != "off" else None,
            },
            "accel_gate": {
                "enabled": bool(args.lambda_gate_accel),
                "g_dag": args.lambda_accel_gdag,
                "q": args.lambda_accel_q,
                "min": float(np.min(accel_gate)) if accel_gate is not None else None,
                "median": float(np.median(accel_gate)) if accel_gate is not None else None,
                "max": float(np.max(accel_gate)) if accel_gate is not None else None,
            },
            "multiplier_stats": {
                "min": float(np.min(lambda_multiplier)) if lambda_multiplier is not None else None,
                "median": float(np.median(lambda_multiplier)) if lambda_multiplier is not None else None,
                "max": float(np.max(lambda_multiplier)) if lambda_multiplier is not None else None,
            }
            if lambda_multiplier is not None
            else None,
        },
        "shell_profile": {
            "centers": shell_centers.tolist() if len(shell_centers) else [],
            "mass": shell_mass.tolist() if len(shell_mass) else [],
        },
        "scaling_params": {
            "mass_ref": args.mass_ref,
            "sigma_ref": args.sigma_ref,
            "alpha_beta": args.alpha_beta,
            "gamma_beta": args.gamma_beta,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "gamma_min": args.gamma_min,
            "gamma_max": args.gamma_max,
        },
        "force_match": bool(args.force_match),
        "rms_velocity": rms_vel,
        "n_points": int(len(R)),
        "data": [
            {
                "R": float(R[i]),
                "V_obs": float(Vobs[i]),
                "V_GR": float(V_gr[i]),
                "V_required": float(V_req[i]),
                "raw_enhancement": float(V_raw[i]),
                "lambda_value": float(lambda_vals[i]) if lambda_vals is not None else None,
                "lambda_multiplier": float(lambda_multiplier[i]) if lambda_multiplier is not None else None,
                "capacity": float(capacity[i]) if capacity is not None else None,
                "provided": float(provided[i]),
                "V_model": float(V_model[i]),
            }
            for i in range(len(R))
        ],
    }

    output_path.write_text(json.dumps(report, indent=2))
    print(f"[OK] Saved: {output_path}")


if __name__ == "__main__":
    main()

