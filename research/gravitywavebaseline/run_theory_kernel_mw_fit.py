"""
Fit the first-principles theory kernel so that it matches the empirical MW kernel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from metric_resonance_multiplier import metric_resonance_multiplier
from theory_metric_resonance import compute_theory_kernel, CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def load_mw_slice(parquet_path: str, r_min: float, r_max: float) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    required = {"R", "v_phi", "v_phi_GR"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {parquet_path}: {missing}")
    mask = (
        df["R"].between(r_min, r_max)
        & np.isfinite(df["v_phi"])
        & np.isfinite(df["v_phi_GR"])
    )
    subset = df.loc[mask, ["R", "v_phi", "v_phi_GR"]].copy()
    if subset.empty:
        raise RuntimeError(f"No MW data in [{r_min}, {r_max}] kpc for {parquet_path}")
    return subset


def empirical_kernel(R_kpc: np.ndarray, mw_fit: dict) -> np.ndarray:
    lam_orb = 2.0 * np.pi * R_kpc
    f_emp = metric_resonance_multiplier(
        R_kpc=R_kpc,
        lambda_orb_kpc=lam_orb,
        A=mw_fit["A"],
        ell0_kpc=mw_fit["ell0_kpc"],
        p=mw_fit["p"],
        n_coh=mw_fit["n_coh"],
        lambda_peak_kpc=mw_fit["lambda_peak_kpc"],
        sigma_ln_lambda=mw_fit["sigma_ln_lambda"],
    )
    return f_emp - 1.0


# Global variables for objective function (needed for multiprocessing)
_OBJ_R_KPC = None
_OBJ_K_EMP = None
_OBJ_SIGMA_V = None
_OBJ_BURR_P = None
_OBJ_BURR_N = None
_OBJ_REQUIRE_POSITIVE_CORR = True
_OBJ_INCLUDE_Q_REF = True
_OBJ_USE_GPU = None


def _objective_function(theta: np.ndarray) -> float:
    """Objective function for optimization (module-level for multiprocessing)."""
    global _OBJ_R_KPC, _OBJ_K_EMP, _OBJ_SIGMA_V, _OBJ_BURR_P, _OBJ_BURR_N
    global _OBJ_REQUIRE_POSITIVE_CORR, _OBJ_INCLUDE_Q_REF, _OBJ_USE_GPU
    
    if _OBJ_INCLUDE_Q_REF:
        A_global, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0, Q_ref = theta
    else:
        A_global, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0 = theta
        Q_ref = 1.0  # default
    
    try:
        # Ensure R is a 1D numpy array (may have been corrupted during pickling)
        R_kpc = np.asarray(_OBJ_R_KPC, dtype=np.float64)
        if R_kpc.ndim == 0:
            R_kpc = np.array([R_kpc])
        elif R_kpc.ndim > 1:
            R_kpc = R_kpc.flatten()
        
        # Ensure sigma_v is a valid float
        sigma_v = float(_OBJ_SIGMA_V) if _OBJ_SIGMA_V is not None else 30.0
        burr_p = float(_OBJ_BURR_P) if _OBJ_BURR_P is not None else 0.757
        burr_n = float(_OBJ_BURR_N) if _OBJ_BURR_N is not None else 0.5
        use_gpu = _OBJ_USE_GPU if _OBJ_USE_GPU is not None else False
        
        K_th = compute_theory_kernel(
            R_kpc=R_kpc,
            sigma_v_kms=sigma_v,
            alpha=alpha,
            lam_coh_kpc=lam_coh_kpc,
            lam_cut_kpc=lam_cut_kpc,
            A_global=A_global,
            burr_ell0_kpc=burr_ell0,
            burr_p=burr_p,
            burr_n=burr_n,
            Q_ref=Q_ref,
            use_gpu=use_gpu,
        )
        
        # Ensure K_emp is also a proper numpy array
        K_emp = np.asarray(_OBJ_K_EMP, dtype=np.float64)
        if K_emp.ndim == 0:
            K_emp = np.array([K_emp])
        elif K_emp.ndim > 1:
            K_emp = K_emp.flatten()
        
        # Ensure K_th matches shape
        K_th = np.asarray(K_th, dtype=np.float64)
        if K_th.ndim == 0:
            K_th = np.array([K_th])
        elif K_th.ndim > 1:
            K_th = K_th.flatten()
        
        # Ensure same length
        min_len = min(len(K_th), len(K_emp))
        K_th = K_th[:min_len]
        K_emp = K_emp[:min_len]
        
        # Compute correlation
        corr = float(np.corrcoef(K_th, K_emp)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
        
        # Base chi-squared
        chi2 = rms(K_th - K_emp) ** 2
        
        # Penalty for negative correlation if required
        if _OBJ_REQUIRE_POSITIVE_CORR:
            if corr < 0:
                # Extremely strong penalty: make it impossible to accept negative correlation
                penalty = 1e10 * (1.0 - corr) ** 4
                return chi2 + penalty
            elif corr < 0.7:
                # Strong penalty for weak positive correlation
                penalty = 1000.0 * (0.7 - corr) ** 2
                return chi2 + penalty
            elif corr < 0.9:
                # Moderate penalty to push toward high correlation
                penalty = 10.0 * (0.9 - corr) ** 2
                return chi2 + penalty
        
        return chi2
    except (ValueError, RuntimeError, ZeroDivisionError):
        return 1e10  # Penalty for invalid parameters


def fit_theory_params(
    R_kpc: np.ndarray,
    K_emp: np.ndarray,
    sigma_v_kms: float,
    burr_p: float,
    burr_n: float,
    require_positive_corr: bool = True,
    include_Q_ref: bool = True,
    use_gpu: bool | None = None,
    n_workers: int = 1,
) -> dict:
    """
    Fit theory kernel parameters to match empirical kernel.
    
    If require_positive_corr=True, uses penalty function to strongly enforce
    positive correlation. Also optionally fits Q_ref as an additional parameter.
    """
    global _OBJ_R_KPC, _OBJ_K_EMP, _OBJ_SIGMA_V, _OBJ_BURR_P, _OBJ_BURR_N
    global _OBJ_REQUIRE_POSITIVE_CORR, _OBJ_INCLUDE_Q_REF, _OBJ_USE_GPU
    
    # Set global variables for objective function
    # Ensure numpy arrays (not CuPy) for multiprocessing compatibility
    # Convert from CuPy to NumPy if needed
    if CUPY_AVAILABLE and hasattr(R_kpc, 'get'):  # CuPy array
        _OBJ_R_KPC = cp.asnumpy(R_kpc)
    else:
        _OBJ_R_KPC = np.asarray(R_kpc, dtype=np.float64)
    
    if CUPY_AVAILABLE and hasattr(K_emp, 'get'):  # CuPy array
        _OBJ_K_EMP = cp.asnumpy(K_emp)
    else:
        _OBJ_K_EMP = np.asarray(K_emp, dtype=np.float64)
    
    _OBJ_SIGMA_V = float(sigma_v_kms)
    _OBJ_BURR_P = float(burr_p)
    _OBJ_BURR_N = float(burr_n)
    _OBJ_REQUIRE_POSITIVE_CORR = require_positive_corr
    _OBJ_INCLUDE_Q_REF = include_Q_ref
    # GPU can be used in each worker process independently
    _OBJ_USE_GPU = use_gpu
    
    if include_Q_ref:
        # Extended parameter set: [A_global, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0_kpc, Q_ref]
        bounds = [
            (-50.0, 50.0),   # A_global (wider range)
            (-2.0, 8.0),     # alpha (allow negative for broader spectral model)
            (0.5, 150.0),    # lam_coh_kpc (wider range)
            (20.0, 3000.0),  # lam_cut_kpc (wider range)
            (2.0, 80.0),     # burr_ell0_kpc (wider range)
            (0.1, 10.0),     # Q_ref (new parameter)
        ]
    else:
        bounds = [
            (-50.0, 50.0),   # A_global
            (-2.0, 8.0),     # alpha
            (0.5, 150.0),    # lam_coh_kpc
            (20.0, 3000.0),  # lam_cut_kpc
            (2.0, 80.0),     # burr_ell0_kpc
        ]
    # Try multiple optimization runs with different seeds
    # Use parallel workers for faster evaluation
    best_result = None
    best_obj = np.inf
    
    for seed in [123, 456, 789, 42, 999]:
        try:
            result = differential_evolution(
                _objective_function,
                bounds=bounds,
                maxiter=200,
                polish=True,
                seed=seed,
                atol=1e-6,
                tol=1e-4,
                workers=n_workers if n_workers > 1 else 1,
            )
            if result.success and result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: single run
        best_result = differential_evolution(
            _objective_function,
            bounds=bounds,
            maxiter=200,
            polish=True,
            workers=n_workers if n_workers > 1 else 1,
        )
    
    if include_Q_ref:
        A_global_raw, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0, Q_ref = best_result.x
    else:
        A_global_raw, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0 = best_result.x
        Q_ref = 1.0

    common_kwargs = dict(
        R_kpc=R_kpc,
        sigma_v_kms=sigma_v_kms,
        alpha=alpha,
        lam_coh_kpc=lam_coh_kpc,
        lam_cut_kpc=lam_cut_kpc,
        burr_ell0_kpc=burr_ell0,
        burr_p=burr_p,
        burr_n=burr_n,
        Q_ref=Q_ref,
    )

    common_kwargs['use_gpu'] = use_gpu
    K_raw = compute_theory_kernel(A_global=A_global_raw, **common_kwargs)
    corr_raw = float(np.corrcoef(K_raw, K_emp)[0, 1])
    if not np.isfinite(corr_raw):
        corr_raw = 0.0
    phase_sign = 1.0 if corr_raw >= 0.0 else -1.0
    corr_abs = abs(corr_raw)

    result_dict = {
        "A_global": float(A_global_raw),
        "alpha": float(alpha),
        "lam_coh_kpc": float(lam_coh_kpc),
        "lam_cut_kpc": float(lam_cut_kpc),
        "burr_ell0_kpc": float(burr_ell0),
        "burr_p": float(burr_p),
        "burr_n": float(burr_n),
        "Q_ref": float(Q_ref),
        "phase_sign": float(phase_sign),
        "rms_diff": float(rms(K_raw - K_emp)),
        "chi2_K": float(rms(K_raw - K_emp) ** 2),
        "corr_K_emp_theory": float(corr_abs),
        "optimization_success": bool(best_result.success),
        "optimization_message": str(best_result.message),
        "optimization_nfev": int(best_result.nfev),
    }
    
    return result_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit compute_theory_kernel parameters to MW empirical kernel."
    )
    parser.add_argument(
        "--baseline-parquet",
        default="gravitywavebaseline/gaia_with_gr_baseline.parquet",
    )
    parser.add_argument(
        "--mw-fit-json",
        default="gravitywavebaseline/metric_resonance_mw_fit.json",
    )
    parser.add_argument("--r-min", type=float, default=12.0)
    parser.add_argument("--r-max", type=float, default=16.0)
    parser.add_argument("--sigma-v", type=float, default=30.0, help="MW dispersion (km/s)")
    parser.add_argument(
        "--out-json",
        default="gravitywavebaseline/theory_metric_resonance_mw_fit.json",
    )
    parser.add_argument(
        "--require-positive-corr",
        action="store_true",
        default=True,
        help="Enforce positive correlation with empirical kernel",
    )
    parser.add_argument(
        "--no-Q-ref",
        action="store_true",
        help="Don't fit Q_ref as a free parameter",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=None,
        help="Force GPU usage (CuPy). If not set, auto-detect.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU usage (disable GPU even if available)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers for optimization (default: number of CPUs)",
    )
    args = parser.parse_args()
    
    # Determine GPU usage
    if args.no_gpu:
        use_gpu = False
    elif args.use_gpu:
        use_gpu = True
    else:
        use_gpu = None  # Auto-detect
    
    # Determine number of workers
    if args.n_workers is None:
        n_workers = min(mp.cpu_count(), 10)  # Use up to 10 CPUs
    else:
        n_workers = args.n_workers
    
    if use_gpu and CUPY_AVAILABLE:
        print(f"[GPU] CuPy available - GPU acceleration enabled")
    elif use_gpu and not CUPY_AVAILABLE:
        print(f"[WARN] GPU requested but CuPy not available, using CPU")
        use_gpu = False
    else:
        print(f"[CPU] Using {n_workers} parallel workers")

    df = load_mw_slice(args.baseline_parquet, args.r_min, args.r_max)
    R = df["R"].to_numpy(float)
    v_obs = df["v_phi"].to_numpy(float)
    v_gr = df["v_phi_GR"].to_numpy(float)

    rms_gr = rms(v_obs - v_gr)
    print(f"[info] GR-only RMS in MW slice: {rms_gr:.2f} km/s (N={len(df)})")

    mw_fit = json.loads(Path(args.mw_fit_json).read_text())
    K_emp = empirical_kernel(R, mw_fit)
    print(
        f"[info] Empirical kernel range: K_min={K_emp.min():.4f}, "
        f"K_max={K_emp.max():.4f}"
    )

    burr_p = float(mw_fit.get("p", 1.0))
    burr_n = float(mw_fit.get("n_coh", 0.5))
    
    print(f"[info] Fitting theory kernel (require_positive_corr={args.require_positive_corr}, include_Q_ref={not args.no_Q_ref})")
    theory_fit = fit_theory_params(
        R,
        K_emp,
        args.sigma_v,
        burr_p=burr_p,
        burr_n=burr_n,
        require_positive_corr=args.require_positive_corr,
        include_Q_ref=not args.no_Q_ref,
        use_gpu=use_gpu,
        n_workers=n_workers,
    )
    
    print("[info] Best-fit theory parameters:")
    for key, value in theory_fit.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6g}")
        else:
            print(f"  {key}: {value}")
    
    # Verify the fit quality
    K_th_final = compute_theory_kernel(
        R_kpc=R,
        sigma_v_kms=args.sigma_v,
        alpha=theory_fit["alpha"],
        lam_coh_kpc=theory_fit["lam_coh_kpc"],
        lam_cut_kpc=theory_fit["lam_cut_kpc"],
        A_global=theory_fit["A_global"],
        burr_ell0_kpc=theory_fit["burr_ell0_kpc"],
        burr_p=burr_p,
        burr_n=burr_n,
        Q_ref=theory_fit.get("Q_ref", 1.0),
        use_gpu=use_gpu,
    )
    corr_final = float(np.corrcoef(K_th_final, K_emp)[0, 1])
    print(f"[info] Final correlation: {corr_final:.6f}")
    print(f"[info] Final RMS difference: {theory_fit['rms_diff']:.6e}")

    out = {
        "r_min": args.r_min,
        "r_max": args.r_max,
        "sigma_v": args.sigma_v,
        "n_points": len(df),
        "rms_gr": rms_gr,
        "mw_fit_params": mw_fit,
        "theory_fit_params": theory_fit,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[info] Saved theory MW fit to {args.out_json}")


if __name__ == "__main__":
    main()


