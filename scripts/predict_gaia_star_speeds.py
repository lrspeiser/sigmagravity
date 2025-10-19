#!/usr/bin/env python3
"""
GPU star-level predictions for Gaia MW dataset using saturated-well tail.

See data/gaia/README.md for data sources, parameters, and operational notes.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

try:
    import cupy as cp
except Exception as e:
    raise SystemExit(f"CuPy is required for GPU predictions: {e}")


def gate_c1_cu(R: "cp.ndarray", Rb: float, dR: float) -> "cp.ndarray":
    dR = max(float(dR), 1e-6)
    s = (R - float(Rb)) / dR
    s = cp.clip(s, 0.0, 1.0)
    return 3.0 * s*s - 2.0 * s*s*s


def v2_saturated_extra_cu(R: "cp.ndarray", v_flat: float, R_s: float, m: float) -> "cp.ndarray":
    R_s = max(float(R_s), 1e-6)
    return (float(v_flat)**2) * (1.0 - cp.exp(-cp.power(R / R_s, float(m))))


def main():
    ap = argparse.ArgumentParser(description="Predict Gaia star speeds on GPU using saturated-well tail fit.")
    ap.add_argument("--npz", required=True, help="Path to mw_gaia_144k.npz")
    ap.add_argument("--fit", required=True, help="Path to fit_params.json from maxdepth_gaia pipeline")
    ap.add_argument("--out", required=True, help="Output CSV path for predictions")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index")
    args = ap.parse_args()

    # Select device
    try:
        dev = int(args.device)
        _ = cp.cuda.Device(dev).compute_capability  # probe
    except Exception as e:
        raise SystemExit(f"Failed to initialize CUDA device {args.device}: {e}")

    with cp.cuda.Device(dev):
        # Load fit parameters
        fit_path = Path(args.fit)
        if not fit_path.exists():
            raise SystemExit(f"fit_params.json not found: {fit_path}")
        fit = json.loads(fit_path.read_text())

        # Extract saturated-well parameters
        try:
            Rb = float(fit["boundary"]["R_boundary"]) if "boundary" in fit else float(fit["saturated_well"]["R_boundary"])  # fallback
        except Exception:
            Rb = float(fit.get("R_boundary", np.nan))
        try:
            sw = fit["saturated_well"]["params"]
            v_flat = float(sw.get("v_flat", fit["saturated_well"].get("v_flat")))
            R_s = float(sw["R_s"]) ; m = float(sw["m"])
            dR = float(sw.get("gate_width_kpc", 0.8))
        except Exception as e:
            raise SystemExit(f"Missing saturated-well parameters in fit_params.json: {e}")
        if not np.isfinite(Rb):
            # fallback to bins.rmin as a conservative boundary if absent
            Rb = float(fit.get("bins", {}).get("rmin", 6.0))

        # Load NPZ
        npz_path = Path(args.npz)
        if not npz_path.exists():
            raise SystemExit(f"NPZ not found: {npz_path}")
        d = np.load(npz_path)
        # Required arrays
        required = ["R_kpc","z_kpc","v_obs_kms","v_err_kms","gN_kms2_per_kpc"]
        missing = [k for k in required if k not in d]
        if missing:
            raise SystemExit(f"NPZ missing required keys: {missing}")

        R = cp.asarray(d["R_kpc"], dtype=cp.float64)
        z = cp.asarray(d["z_kpc"], dtype=cp.float64)  # currently unused, reserved for future vertical analysis
        v_obs = cp.asarray(d["v_obs_kms"], dtype=cp.float64)
        v_err = cp.asarray(d["v_err_kms"], dtype=cp.float64)
        gN = cp.asarray(d["gN_kms2_per_kpc"], dtype=cp.float64)
        Sigma_loc = cp.asarray(d["Sigma_loc_Msun_pc2"], dtype=cp.float64) if "Sigma_loc_Msun_pc2" in d else None

        # Baryonic circular speed from provided Newtonian acceleration: vbar^2 = g_N * R
        vbar2 = cp.maximum(0.0, gN * cp.maximum(R, 1e-12))
        vbar = cp.sqrt(vbar2)

        # Saturated-well extra term with C1 gate
        v2_extra = v2_saturated_extra_cu(R, v_flat=v_flat, R_s=R_s, m=m)
        gate = gate_c1_cu(R, Rb=Rb, dR=dR)
        v2_extra_gated = v2_extra * gate

        v_model = cp.sqrt(cp.maximum(0.0, vbar2 + v2_extra_gated))
        resid = v_obs - v_model

        # Move to CPU and write CSV
        import pandas as pd
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = {
            "R_kpc": cp.asnumpy(R),
            "z_kpc": cp.asnumpy(z),
            "v_obs_kms": cp.asnumpy(v_obs),
            "v_err_kms": cp.asnumpy(v_err),
            "gN_kms2_per_kpc": cp.asnumpy(gN),
            "v_baryon_kms": cp.asnumpy(vbar),
            "v_model_kms": cp.asnumpy(v_model),
            "residual_kms": cp.asnumpy(resid),
        }
        if Sigma_loc is not None:
            df["Sigma_loc_Msun_pc2"] = cp.asnumpy(Sigma_loc)
        pd.DataFrame(df).to_csv(out, index=False)

        # Log summary
        try:
            import numpy as _np
            r_cpu = _np.asarray(df["residual_kms"]) ; r_cpu = r_cpu[_np.isfinite(r_cpu)]
            print(f"Wrote {out} (rows={len(df['R_kpc'])})")
            if r_cpu.size:
                print(f"Residuals: mean={r_cpu.mean():.3f} km/s, std={r_cpu.std(ddof=1):.3f} km/s, median={_np.median(r_cpu):.3f} km/s")
        except Exception:
            pass


if __name__ == "__main__":
    main()
