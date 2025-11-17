"""
Apply theory kernel parameters (fit on MW) to a batch of SPARC galaxies.
"""

from __future__ import annotations

import argparse
import glob
import os
import json
from pathlib import Path
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd

from theory_metric_resonance import compute_theory_kernel, CUPY_AVAILABLE


def rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr * arr)))


def load_rotmod(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        usecols=[0, 1, 3, 4, 5],
        names=["R_kpc", "V_obs", "V_gas", "V_disk", "V_bul"],
        engine="python",
    )
    if df.empty:
        raise RuntimeError(f"No data parsed from {path}")
    v_gr = np.sqrt(
        np.clip(
            df["V_gas"].to_numpy() ** 2
            + df["V_disk"].to_numpy() ** 2
            + df["V_bul"].to_numpy() ** 2,
            0.0,
            None,
        )
    )
    df["V_gr"] = v_gr
    return df[["R_kpc", "V_obs", "V_gr"]]


def main():
    parser = argparse.ArgumentParser(
        description="Test theory kernel across SPARC galaxies."
    )
    parser.add_argument("--rotmod-dir", default="data/Rotmod_LTG")
    parser.add_argument("--sparc-summary", default="data/sparc/sparc_combined.csv")
    parser.add_argument("--summary-galaxy-col", default="galaxy_name")
    parser.add_argument("--summary-sigma-col", default="sigma_velocity")
    parser.add_argument("--sigma-ref", type=float, default=25.0)
    parser.add_argument("--beta-sigma", type=float, default=1.0)
    parser.add_argument(
        "--theory-fit-json",
        default="gravitywavebaseline/theory_metric_resonance_mw_fit.json",
    )
    parser.add_argument(
        "--out-csv",
        default="gravitywavebaseline/theory_kernel_sparc_batch.csv",
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
        help="Number of parallel workers for galaxy processing (default: number of CPUs)",
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
        print(f"[CPU] Using {n_workers} parallel workers for galaxy processing")

    theory_fit = json.loads(Path(args.theory_fit_json).read_text())
    th = theory_fit["theory_fit_params"]
    phase_sign = float(th.get("phase_sign", 1.0))

    summary = pd.read_csv(args.sparc_summary)
    sigma_map = dict(
        zip(
            summary[args.summary_galaxy_col].astype(str),
            summary[args.summary_sigma_col].astype(float),
        )
    )

    # Process galaxies in parallel
    def process_galaxy(path: str) -> dict | None:
        """Process a single galaxy."""
        galaxy = Path(path).name.replace("_rotmod.dat", "")
        try:
            df = load_rotmod(path)
        except Exception as exc:
            return None  # Don't print in parallel mode

        R = df["R_kpc"].to_numpy(float)
        V_obs = df["V_obs"].to_numpy(float)
        V_gr = df["V_gr"].to_numpy(float)
        if len(R) < 4:
            return None

        sigma_true = sigma_map.get(galaxy, args.sigma_ref)
        v_flat = np.nanmedian(V_gr[-min(len(V_gr), 5):]) or 200.0
        Q = v_flat / max(sigma_true, 1e-3)
        G_sigma = (Q**args.beta_sigma) / (1.0 + Q**args.beta_sigma)

        K_th = compute_theory_kernel(
            R_kpc=R,
            sigma_v_kms=sigma_true,
            alpha=th["alpha"],
            lam_coh_kpc=th["lam_coh_kpc"],
            lam_cut_kpc=th["lam_cut_kpc"],
            A_global=th["A_global"] * G_sigma,
            burr_ell0_kpc=th.get("burr_ell0_kpc"),
            burr_p=th.get("burr_p", 1.0),
            burr_n=th.get("burr_n", 0.5),
            use_gpu=use_gpu,
        )
        K_aligned = phase_sign * K_th
        f_th = 1.0 + K_aligned
        V_model = V_gr * np.sqrt(np.clip(f_th, 0.0, None))

        rms_gr = rms(V_obs - V_gr)
        rms_th = rms(V_obs - V_model)

        return dict(
            galaxy=galaxy,
            n_points=len(R),
            sigma_v_true=sigma_true,
            Q_gal=Q,
            G_sigma=G_sigma,
            K_mean=float(np.mean(K_aligned)),
            rms_gr=rms_gr,
            rms_theory=rms_th,
            delta_rms=rms_th - rms_gr,
        )
    
    rotmod_paths = sorted(glob.glob(os.path.join(args.rotmod_dir, "*_rotmod.dat")))
    print(f"[info] processing {len(rotmod_paths)} SPARC rotmod files")

    # Process in parallel if multiple workers
    if n_workers > 1:
        with mp.Pool(n_workers) as pool:
            results = pool.map(process_galaxy, rotmod_paths)
        rows = [r for r in results if r is not None]
    else:
        rows = []
        for path in rotmod_paths:
            result = process_galaxy(path)
            if result is not None:
                rows.append(result)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[info] wrote theory kernel batch results to {args.out_csv}")


if __name__ == "__main__":
    main()


