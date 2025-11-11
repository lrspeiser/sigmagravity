#!/usr/bin/env python3
"""
Ingest SPARC-like rotation curves and metadata, normalize to a common R/Rd grid,
optionally normalize velocity by V_f, and write NPZ:
  - curve_mat [N,K]
  - weight_mat [N,K]  (1/σ^2 for each gridpoint)
  - x_grid [K]        (dimensionless R/Rd)
  - scalars table     (per-galaxy scalar features if present)
  - names [N]         (galaxy identifiers)
Also writes a JSON with grid and metadata column mapping for reproducibility.
"""
import argparse, os, json, glob, re
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
from tqdm import tqdm

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves_dir", required=True, help="Dir with per-galaxy CSVs (R_kpc,V_obs,eV_obs[,V_bar|V_star,V_gas]).")
    ap.add_argument("--meta_csv", required=True, help="Metadata CSV with at least: name, Rd[, Vf, Sigma0, Mbar].")
    ap.add_argument("--id_column", default="name", help="Column in meta CSV with galaxy ID (case-insensitive).")
    # Column mappings for curve files (case-insensitive match)
    ap.add_argument("--map_R", default="R_kpc", help="Radius column in curve CSVs (kpc).")
    ap.add_argument("--map_V", default="V_obs", help="Observed circular velocity (km/s).")
    ap.add_argument("--map_eV", default="eV_obs", help="Velocity uncertainty (km/s).")
    ap.add_argument("--map_Vbar", default="", help="Baryon velocity column (km/s). Optional.")
    ap.add_argument("--map_Vstar", default="", help="Stellar velocity column (km/s). Optional, used if V_bar missing.")
    ap.add_argument("--map_Vgas", default="", help="Gas velocity column (km/s). Optional, used if V_bar missing.")
    # Metadata mappings
    ap.add_argument("--map_Rd", default="Rd", help="Disk scale length column (kpc).")
    ap.add_argument("--map_Re", default="Reff", help="Effective radius column (kpc).")
    ap.add_argument("--map_Vf", default="Vf", help="Flat velocity column (km/s), optional for velocity normalization.")
    ap.add_argument("--map_Sigma0", default="Sigma0", help="Central surface density (optional).")
    ap.add_argument("--map_Mbar", default="Mbar", help="Total baryonic mass (optional).")
    # Grid and normalization
    ap.add_argument("--grid_min", type=float, default=0.2)
    ap.add_argument("--grid_max", type=float, default=6.0)
    ap.add_argument("--grid_k", type=int, default=50)
    ap.add_argument("--norm_radius", choices=["Rd","Re","none"], default="Rd")
    ap.add_argument("--norm_velocity", choices=["Vf","none"], default="Vf")
    ap.add_argument("--spline_s", type=float, default=0.0, help="Smoothing factor s for splrep (0 for interpolating).")
    ap.add_argument("--out_npz", required=True, help="Output NPZ path under data/processed/.")
    ap.add_argument("--out_grid_json", default="", help="Optional JSON with grid and mapping info.")
    return ap.parse_args()

def find_curve_file(curves_dir, name):
    # match by filename stem (case-insensitive, spaces/underscore/dash tolerated)
    pat = re.sub(r'\W+', '.*', name.strip(), flags=re.UNICODE)
    regex = re.compile(pat, re.IGNORECASE)
    candidates = []
    for fp in glob.glob(os.path.join(curves_dir, "*.csv")):
        stem = os.path.splitext(os.path.basename(fp))[0]
        if regex.search(stem):
            candidates.append(fp)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # fallback to exact lower case match
        for fp in candidates:
            if os.path.splitext(os.path.basename(fp))[0].lower() == name.strip().lower():
                return fp
        return candidates[0]
    return None

def main():
    args = parse_args()
    meta = pd.read_csv(args.meta_csv)
    # normalize column names
    meta_cols = {c.lower(): c for c in meta.columns}
    id_col = args.id_column.lower()
    if id_col not in meta_cols:
        raise ValueError(f"id_column '{args.id_column}' not in metadata columns: {list(meta.columns)}")
    # Build grid
    x_grid = np.linspace(args.grid_min, args.grid_max, args.grid_k)

    curve_rows = []
    weight_rows = []
    names = []
    scalars = []
    scalar_cols = {
        "log10_Mbar": args.map_Mbar.lower(),
        "log10_Sigma0": args.map_Sigma0.lower(),
        "log10_Rd": args.map_Rd.lower(),
        "log10_Vf": args.map_Vf.lower()
    }

    miss = 0
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Ingest galaxies"):
        name = row[ meta_cols[id_col] ]
        names.append(str(name))
        # required metadata: Rd (always needed for scalars)
        Rd = float(row.get(args.map_Rd, np.nan))
        if not np.isfinite(Rd) or Rd <= 0:
            # skip this galaxy
            miss += 1
            names.pop()
            continue
        # optional Re
        Re = float(row.get(args.map_Re, np.nan))
        Vf = row.get(args.map_Vf, np.nan)
        Vf = float(Vf) if np.isfinite(Vf) else np.nan

        # locate curve CSV
        fp = find_curve_file(args.curves_dir, str(name))
        if fp is None:
            miss += 1
            names.pop()
            continue
        df = pd.read_csv(fp)

        # column matching (case-insensitive)
        cols = {c.lower(): c for c in df.columns}
        def get_col(key):
            k = key.lower()
            if k in cols: return cols[k]
            # try loose match
            for c in cols:
                if c.lower() == k: return cols[c]
            raise KeyError(f"Column '{key}' not found in {fp} columns {list(df.columns)}")

        try:
            Rcol = get_col(args.map_R)
            Vcol = get_col(args.map_V)
            eVcol = get_col(args.map_eV)
        except KeyError:
            miss += 1
            names.pop()
            continue

        R = df[Rcol].to_numpy(dtype=float)
        V = df[Vcol].to_numpy(dtype=float)
        eV = df[eVcol].to_numpy(dtype=float)

        mask = np.isfinite(R) & np.isfinite(V) & np.isfinite(eV) & (eV > 0)
        R, V, eV = R[mask], V[mask], eV[mask]
        if R.size < 5:
            miss += 1
            names.pop()
            continue

        # Build x = R / Rnorm
        nr = args.norm_radius.lower()
        if nr == "rd":
            x = R / Rd
        elif nr == "re" and np.isfinite(Re) and Re > 0:
            x = R / Re
        elif nr == "none":
            x = R  # no radius normalization
        else:
            # fallback to Rd if Re requested but missing
            x = R / Rd

        # spline fit on V(x)
        try:
            tck = splrep(x, V, w=1.0/np.maximum(eV, 1e-6), s=args.spline_s, k=3)
            V_resamp = splev(x_grid, tck)
            eV_resamp = np.interp(x_grid, x, eV)
        except Exception:
            miss += 1
            names.pop()
            continue

        # optional velocity normalization
        if args.norm_velocity.lower() == "vf" and np.isfinite(Vf) and Vf > 0:
            V_resamp = V_resamp / Vf
            eV_resamp = eV_resamp / Vf

        # weights = 1/sigma^2
        w_row = 1.0 / np.maximum(eV_resamp**2, 1e-8)
        curve_rows.append(V_resamp.astype(float))
        weight_rows.append(w_row.astype(float))

        # gather scalars
        sc = []
        # Mbar, Sigma0, Rd, Vf
        val_Mbar = row.get(args.map_Mbar, np.nan)
        val_Sigma0 = row.get(args.map_Sigma0, np.nan)
        val_Rd = Rd
        val_Vf = Vf if np.isfinite(Vf) and Vf>0 else np.nan
        sc.append(np.log10(val_Mbar) if np.isfinite(val_Mbar) and val_Mbar>0 else np.nan)
        sc.append(np.log10(val_Sigma0) if np.isfinite(val_Sigma0) and val_Sigma0>0 else np.nan)
        sc.append(np.log10(val_Rd) if np.isfinite(val_Rd) and val_Rd>0 else np.nan)
        sc.append(np.log10(val_Vf) if np.isfinite(val_Vf) and val_Vf>0 else np.nan)
        scalars.append(sc)

    if not curve_rows:
        raise RuntimeError("No galaxies ingested; check your file mappings.")

    curve_mat = np.vstack(curve_rows)
    weight_mat = np.vstack(weight_rows)
    names_arr = np.array(names, dtype=object)
    scalars_arr = np.array(scalars, dtype=float)

    # Save NPZ
    out = args.out_npz
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out,
                        curve_mat=curve_mat, weight_mat=weight_mat,
                        x_grid=np.array(x_grid, dtype=float),
                        names=names_arr, scalars=scalars_arr)
    # grid json (for plotting)
    grid_info = {
        "x_grid": x_grid.tolist(),
        "norm_radius": args.norm_radius,
        "norm_velocity": args.norm_velocity,
        "grid_min": args.grid_min,
        "grid_max": args.grid_max,
        "grid_k": args.grid_k,
        "meta_columns": list(meta.columns)
    }
    out_grid = args.out_grid_json or os.path.splitext(out)[0] + "_grid.json"
    with open(out_grid, "w") as f:
        json.dump(grid_info, f, indent=2)

    print(f"[OK] Wrote {out} and {out_grid}. Skipped {miss} galaxies.")

if __name__ == "__main__":
    main()
