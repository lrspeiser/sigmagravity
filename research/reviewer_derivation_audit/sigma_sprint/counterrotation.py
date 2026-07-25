"""Matched-control readiness audit for the proposed direct MaNGA experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.spatial.distance import cdist

MATCH_FEATURES = [
    "log_stellar_mass",
    "log_Re_kpc",
    "sersic_n",
    "axis_ratio",
    "inclination_deg",
    "redshift",
    "jam_chi2_dof",
]


def load_counterrotator_catalog(path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="|", comment="#", skipinitialspace=True)
    frame.columns = [column.strip() for column in frame.columns]
    frame["mangaid"] = frame["MaNGAId"].astype(str).str.strip()
    return frame


def load_manga_jam(path) -> pd.DataFrame:
    with fits.open(path, memmap=True) as hdul:
        basic = hdul[1].data
        nfw = hdul[4].data
        frame = pd.DataFrame(
            {
                "mangaid": np.char.strip(basic["mangaid"].astype(str)),
                "plateifu": np.char.strip(basic["plateifu"].astype(str)),
                # FITS numeric columns are big-endian; materialize native
                # arrays before pandas arithmetic on little-endian hosts.
                "log_stellar_mass": np.asarray(basic["nsa_sersic_mass"]).astype(float),
                "Re_arcsec": np.asarray(basic["Re_arcsec_MGE"]).astype(float),
                "distance_Mpc": np.asarray(basic["DA"]).astype(float),
                "sersic_n": np.asarray(basic["nsa_sersic_n"]).astype(float),
                "axis_ratio": np.asarray(basic["nsa_sersic_ba"]).astype(float),
                "redshift": np.asarray(basic["z"]).astype(float),
                "quality_flag": np.asarray(basic["Qual"]).astype(int),
                "inclination_deg": np.asarray(nfw["inc_deg"]).astype(float),
                "jam_chi2_dof": np.asarray(nfw["chi2_dof"]).astype(float),
                "fdm_Re_secondary": np.asarray(nfw["fdm_Re"]).astype(float),
            }
        )
    frame["Re_kpc"] = frame["Re_arcsec"] * frame["distance_Mpc"] * 1000.0 / 206265.0
    frame["log_Re_kpc"] = np.log10(frame["Re_kpc"])
    return frame


def standardized_mean_differences(cases, controls, features=MATCH_FEATURES):
    rows = []
    for feature in features:
        case = pd.to_numeric(cases[feature], errors="coerce").dropna().to_numpy()
        control = pd.to_numeric(controls[feature], errors="coerce").dropna().to_numpy()
        pooled = np.sqrt((np.var(case, ddof=1) + np.var(control, ddof=1)) / 2.0)
        smd = np.nan if pooled == 0 else (np.mean(case) - np.mean(control)) / pooled
        rows.append(
            {
                "feature": feature,
                "case_mean": float(np.mean(case)),
                "control_mean": float(np.mean(control)),
                "standardized_mean_difference": float(smd),
                "absolute_smd": float(abs(smd)),
            }
        )
    return pd.DataFrame(rows)


def greedy_match_controls(
    cases: pd.DataFrame,
    control_pool: pd.DataFrame,
    *,
    controls_per_case: int = 5,
    features=MATCH_FEATURES,
) -> pd.DataFrame:
    """Greedy unique nearest-neighbor matching without outcome leakage."""
    forbidden = {"fdm_Re_secondary", "Lambda_Re", "predicted_velocity"}
    if forbidden.intersection(features):
        raise ValueError("matching features include an outcome or coherence proxy")
    combined = pd.concat([cases[list(features)], control_pool[list(features)]])
    mean = combined.mean()
    scale = combined.std().replace(0, np.nan)
    case_matrix = ((cases[list(features)] - mean) / scale).to_numpy(dtype=float)
    control_matrix = ((control_pool[list(features)] - mean) / scale).to_numpy(dtype=float)
    distances = cdist(case_matrix, control_matrix)
    available = set(range(len(control_pool)))
    rows = []
    case_order = np.argsort(np.min(distances, axis=1))[::-1]
    for case_index in case_order:
        ranked = np.argsort(distances[case_index])
        selected = [index for index in ranked if index in available][:controls_per_case]
        for rank, control_index in enumerate(selected, start=1):
            available.remove(control_index)
            rows.append(
                {
                    "case_mangaid": cases.iloc[case_index]["mangaid"],
                    "case_plateifu": cases.iloc[case_index]["plateifu"],
                    "control_mangaid": control_pool.iloc[control_index]["mangaid"],
                    "control_plateifu": control_pool.iloc[control_index]["plateifu"],
                    "match_rank": rank,
                    "standardized_distance": float(distances[case_index, control_index]),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_secondary_effect(case_values, matched_control_values, seed=20260718):
    rng = np.random.default_rng(seed)
    differences = np.asarray(case_values) - np.asarray(matched_control_values)
    bootstrap = np.array(
        [np.mean(rng.choice(differences, len(differences), replace=True)) for _ in range(5000)]
    )
    return {
        "mean_matched_difference_case_minus_control": float(np.mean(differences)),
        "bootstrap_95_percent_interval": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def counterrotation_readiness(counterrotator_path, jam_path):
    counter = load_counterrotator_catalog(counterrotator_path)
    jam = load_manga_jam(jam_path)
    jam["is_counterrotator"] = jam["mangaid"].isin(set(counter["mangaid"]))
    required = MATCH_FEATURES + ["fdm_Re_secondary", "mangaid", "plateifu"]
    eligible = jam.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    eligible = eligible[(eligible["quality_flag"] >= 0) & (eligible["Re_kpc"] > 0)]
    cases = eligible[eligible["is_counterrotator"]].reset_index(drop=True)
    controls = eligible[~eligible["is_counterrotator"]].reset_index(drop=True)
    matches = greedy_match_controls(cases, controls)
    matched_controls = controls.set_index("mangaid").loc[matches["control_mangaid"]].reset_index()
    before = standardized_mean_differences(cases, controls)
    after = standardized_mean_differences(cases, matched_controls)
    case_fdm = cases.set_index("mangaid")["fdm_Re_secondary"]
    control_fdm = controls.set_index("mangaid")["fdm_Re_secondary"]
    paired_case = []
    paired_control = []
    for case_id, group in matches.groupby("case_mangaid"):
        paired_case.append(case_fdm.loc[case_id])
        paired_control.append(control_fdm.loc[group["control_mangaid"]].mean())
    secondary = _bootstrap_secondary_effect(paired_case, paired_control)
    direct_maps_present = False
    manifest_rows = []
    ids = pd.concat(
        [
            matches[["case_mangaid", "case_plateifu"]].rename(
                columns={"case_mangaid": "mangaid", "case_plateifu": "plateifu"}
            ),
            matches[["control_mangaid", "control_plateifu"]].rename(
                columns={"control_mangaid": "mangaid", "control_plateifu": "plateifu"}
            ),
        ]
    ).drop_duplicates()
    for _, row in ids.iterrows():
        manifest_rows.append(
            {
                "mangaid": row["mangaid"],
                "plateifu": row["plateifu"],
                "required_product": "MaNGA DR17 DAP MAPS velocity, velocity dispersion, masks, inverse variance",
                "present_in_repository": False,
            }
        )
    map_manifest = pd.DataFrame(manifest_rows)
    max_after_smd = float(after["absolute_smd"].max())
    summary = {
        "catalogued_counterrotators": int(counter["mangaid"].nunique()),
        "counterrotators_with_complete_JAM_matching_fields": int(len(cases)),
        "matched_controls": int(len(matches)),
        "unique_matched_controls": int(matches["control_mangaid"].nunique()),
        "maximum_absolute_SMD_after_matching": max_after_smd,
        "balanced_at_abs_SMD_below_0_1": bool(max_after_smd < 0.1),
        "direct_DAP_maps_present": direct_maps_present,
        "environment_measure_present": False,
        "merger_indicator_present": False,
        "primary_direct_test_gate_passed": False,
        "gate_failures": [
            "MaNGA velocity/dispersion MAPS are absent from the repository.",
            "Environment and merger-history covariates are absent.",
            *( [] if max_after_smd < 0.1 else ["Matched controls do not meet |SMD| < 0.1 for every available covariate."] ),
        ],
        "secondary_JAM_NFW_fdm_comparison": secondary,
        "secondary_warning": (
            "fdm_Re is inferred under a JAM/NFW model and is not a direct observable or "
            "a validation test of modified gravity."
        ),
    }
    return summary, matches, before, after, map_manifest
