"""
Apply the first-principles theory kernel to real cluster baryon profiles and
compare the predicted enclosed mass against the lensing requirement at the
observed Einstein radius.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd

from theory_metric_resonance import compute_theory_kernel
from many_path_model.cluster_data_loader import ClusterDataLoader


G_KPC_KM2_S2_MSUN = 4.30091e-6  # gravitational constant (kpc km^2 / s^2 / Msun)
C_LIGHT = 299792.458  # km/s
H0 = 70.0  # km/s/Mpc
OMEGA_M = 0.3
OMEGA_L = 0.7


def comoving_distance(z: float, n_steps: int = 2048) -> float:
    if z <= 0:
        return 0.0
    zz = np.linspace(0.0, z, n_steps)
    Ez = np.sqrt(OMEGA_M * (1 + zz) ** 3 + OMEGA_L)
    integral = np.trapz(1.0 / Ez, zz)
    return (C_LIGHT / H0) * integral  # Mpc


def angular_diameter_distance(z: float) -> float:
    return comoving_distance(z) / (1.0 + z)


def sigma_crit(z_l: float, z_s: float) -> float:
    D_l = angular_diameter_distance(z_l)
    D_s = angular_diameter_distance(z_s)
    if z_s <= z_l or D_s <= 0:
        return np.inf
    D_c_l = comoving_distance(z_l)
    D_c_s = comoving_distance(z_s)
    D_ls = (D_c_s - D_c_l) / (1.0 + z_s)
    if D_ls <= 0:
        return np.inf
    # convert Mpc to kpc
    D_l *= 1e3
    D_s *= 1e3
    D_ls *= 1e3
    return (C_LIGHT**2 / (4 * np.pi * G_KPC_KM2_S2_MSUN)) * (D_s / (D_l * D_ls))


def parse_cluster_list(cluster_arg: str | None, list_path: str | None) -> Iterable[str]:
    names: list[str] = []
    if cluster_arg:
        names.extend([c.strip() for c in cluster_arg.split(",") if c.strip()])
    if list_path:
        for line in Path(list_path).read_text().splitlines():
            entry = line.strip()
            if entry:
                names.append(entry)
    if not names:
        raise ValueError("No cluster names provided.")
    return names


def load_master_catalog() -> pd.DataFrame:
    catalog_path = Path("data/clusters/master_catalog.csv")
    if not catalog_path.exists():
        raise FileNotFoundError("master_catalog.csv not found in data/clusters/")
    df = pd.read_csv(catalog_path)
    df["cluster_name"] = df["cluster_name"].str.upper()
    return df.set_index("cluster_name")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate theory kernel against observed cluster Einstein radii."
    )
    parser.add_argument(
        "--clusters",
        default="MACSJ0416,MACSJ0717,ABELL_1689",
        help="Comma-separated list of cluster directories under data/clusters.",
    )
    parser.add_argument(
        "--cluster-list",
        default=None,
        help="Optional text file listing additional cluster names (one per line).",
    )
    parser.add_argument(
        "--theory-fit-json",
        default="gravitywavebaseline/theory_metric_resonance_mw_fit.json",
    )
    parser.add_argument(
        "--sigma-v-default",
        type=float,
        default=1000.0,
        help="Fallback velocity dispersion in km/s when not specified.",
    )
    parser.add_argument(
        "--out-csv",
        default="gravitywavebaseline/theory_kernel_cluster_summary.csv",
    )
    args = parser.parse_args()

    cluster_names = list(parse_cluster_list(args.clusters, args.cluster_list))
    theory_params = json.loads(Path(args.theory_fit_json).read_text())
    phase_sign = float(theory_params["theory_fit_params"].get("phase_sign", 1.0))
    master_catalog = load_master_catalog()

    loader = ClusterDataLoader(data_dir="data/clusters")
    rows: list[dict] = []

    for raw_name in cluster_names:
        cluster_dir = raw_name.strip().upper()
        catalog_candidates = [cluster_dir]
        if cluster_dir.startswith("ABELL_"):
            catalog_candidates.append("A" + cluster_dir.split("_", 1)[1])
        if "MACSJ" in cluster_dir:
            catalog_candidates.append(cluster_dir.replace("MACSJ", "MACS"))
        elif "MACS" in cluster_dir:
            catalog_candidates.append(cluster_dir.replace("MACS", "MACSJ"))

        catalog_key = next((c for c in catalog_candidates if c in master_catalog.index), None)
        if catalog_key is None:
            raise KeyError(f"{raw_name} not found in master_catalog.csv")

        meta = master_catalog.loc[catalog_key]
        z_l = float(meta["z_lens"])
        z_s = float(meta.get("z_source", 2.0))
        theta_e_obs = float(meta["theta_E_obs_arcsec"])
        theta_e_err = float(meta["theta_E_err_arcsec"])

        try:
            data = loader.load_cluster(cluster_dir, validate=False)
        except FileNotFoundError:
            alt_name = cluster_dir.replace("MACSJ", "MACS") if "MACSJ" in cluster_dir else cluster_dir.replace("MACS", "MACSJ")
            data = loader.load_cluster(alt_name, validate=False)

        R = data.r_kpc
        M_enc, g_bar = loader.compute_baryonic_mass(data)

        th = theory_params["theory_fit_params"]
        K_th = compute_theory_kernel(
            R_kpc=R,
            sigma_v_kms=args.sigma_v_default,
            alpha=th["alpha"],
            lam_coh_kpc=th["lam_coh_kpc"],
            lam_cut_kpc=th["lam_cut_kpc"],
            A_global=th["A_global"],
            burr_ell0_kpc=th.get("burr_ell0_kpc"),
            burr_p=th.get("burr_p", 1.0),
            burr_n=th.get("burr_n", 0.5),
        )

        g_eff = g_bar * (1.0 + phase_sign * K_th)
        M_eff = g_eff * R * R / G_KPC_KM2_S2_MSUN

        sigma_c = sigma_crit(z_l, z_s)
        D_l = angular_diameter_distance(z_l) * 1e3  # kpc
        theta_rad = theta_e_obs * np.pi / (180.0 * 3600.0)
        R_e_kpc = theta_rad * D_l
        M_required = np.pi * R_e_kpc**2 * sigma_c

        M_pred = float(np.interp(R_e_kpc, R, M_eff, left=np.nan, right=np.nan))
        baryon_mass = float(np.interp(R_e_kpc, R, M_enc, left=np.nan, right=np.nan))

        rows.append(
            dict(
                cluster=raw_name,
                z_lens=z_l,
                z_source=z_s,
                theta_E_obs_arcsec=theta_e_obs,
                theta_E_err_arcsec=theta_e_err,
                R_E_kpc=R_e_kpc,
                sigma_crit_Msun_per_kpc2=sigma_c,
                M_required_Msun=M_required,
                M_baryon_at_RE_Msun=baryon_mass,
                M_theory_at_RE_Msun=M_pred,
                mass_ratio_theory=M_pred / M_required if np.isfinite(M_pred) else np.nan,
                mass_ratio_baryon=baryon_mass / M_required if np.isfinite(baryon_mass) else np.nan,
                mass_boost_at_RE=M_pred / baryon_mass if baryon_mass > 0 else np.nan,
            )
        )

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).write_text(out_df.to_csv(index=False))
    print(f"[cluster] wrote summary for {len(out_df)} clusters to {args.out_csv}")


if __name__ == "__main__":
    main()


