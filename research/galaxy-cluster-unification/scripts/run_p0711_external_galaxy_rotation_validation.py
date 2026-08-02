#!/usr/bin/env python3
"""Score frozen P0708 galaxy curves against untouched Iorio et al. tables."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNLOCK = ROOT / "results/p0633_external_validation/unlock_manifest.json"
SUPPLEMENT = ROOT / "data/raw/p0633_little_things_kinematics/stw3285_Supplementary_Data.zip"
CURVES = ROOT / "results/p0708_external_prediction_lock/galaxy_prediction_curves.csv"
AUDIT = ROOT / "results/p0639_registered_baryonic_maps/map_audit.csv"
OUTPUT = ROOT / "results/p0711_external_galaxy_rotation_validation"

MODEL_CANDIDATE = "P0707_time_potential"
MODEL_NEWTONIAN = "Newtonian_3D"
MODEL_AQUAL = "AQUAL_simple_mu_3D"
MODEL_QUMOND = "QUMOND_simple_nu_3D"
MODELS = [MODEL_CANDIDATE, MODEL_NEWTONIAN, MODEL_AQUAL, MODEL_QUMOND]
TABLE_NAMES = {
    "CVnIdwA": "cvidwa",
    "DDO47": "ddo47",
    "DDO50": "ddo50",
    "DDO52": "ddo52",
    "DDO53": "ddo53",
    "DDO87": "ddo87",
    "DDO101": "ddo101",
    "DDO126": "ddo126",
    "DDO133": "ddo133",
    "DDO210": "ddo210",
    "DDO216": "ddo216",
    "NGC1569": "ngc1569",
    "UGC8508": "ugc8508",
}
COLUMNS = [
    "radius_arcsec",
    "radius_kpc_published",
    "rotation_speed_km_s",
    "rotation_speed_error_km_s",
    "asymmetric_drift_km_s",
    "asymmetric_drift_error_km_s",
    "circular_speed_km_s",
    "circular_speed_error_km_s",
    "dispersion_km_s",
    "dispersion_error_km_s",
    "hi_surface_density_msun_pc2",
    "hi_surface_density_error_msun_pc2",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_tables(path: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(path) as outer:
        nested = outer.read("results.zip")
    with zipfile.ZipFile(io.BytesIO(nested)) as archive:
        for galaxy, stem in TABLE_NAMES.items():
            member = f"finalrot/{stem}_onlinetab.txt"
            text = archive.read(member).decode("utf-8", errors="replace")
            values = []
            for line in text.splitlines():
                tokens = line.strip().split()
                if len(tokens) != 12:
                    continue
                try:
                    numeric = [float(value) for value in tokens]
                except ValueError:
                    continue
                values.append(numeric)
            if not values:
                raise RuntimeError(f"no kinematic rows parsed for {galaxy}")
            frame = pd.DataFrame(values, columns=COLUMNS)
            frame.insert(0, "galaxy", galaxy)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def interpolate_model(
    curves: pd.DataFrame, galaxy: str, model: str, radius: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = curves[(curves["system"] == galaxy) & (curves["model"] == model)].sort_values(
        "radius_kpc"
    )
    if source.empty:
        raise RuntimeError(f"missing frozen prediction for {galaxy} {model}")
    r = source["radius_kpc"].to_numpy(dtype=float)
    v = source["circular_speed_km_s"].to_numpy(dtype=float)
    valid = (radius >= r.min()) & (radius <= r.max())
    prediction = np.full_like(radius, np.nan, dtype=float)
    prediction[valid] = np.interp(radius[valid], r, v)
    return prediction, valid


def rmse(residual: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residual))))


def sample_rmse(per_galaxy: pd.DataFrame, column: str) -> float:
    return float(np.sqrt(np.mean(np.square(per_galaxy[column].to_numpy(dtype=float)))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlock", type=Path, default=DEFAULT_UNLOCK)
    args = parser.parse_args()
    unlock = json.loads(args.unlock.resolve().read_text(encoding="utf-8"))
    if unlock["status"] != "authorized_for_exactly_one_external_parse":
        raise RuntimeError("P0709 unlock is missing")
    if unlock["universal_parameter_sha256"] != (
        "bf3f12d6b32ee3f1b0e3bf48a9603c4aafbcd34b2cbdd3de021d689514099a15"
    ):
        raise RuntimeError("frozen parameter hash changed")

    observed = parse_tables(SUPPLEMENT)
    curves = pd.read_csv(CURVES)
    audit = pd.read_csv(AUDIT).set_index("galaxy")
    predictions = []
    scores = []
    for galaxy, group in observed.groupby("galaxy", sort=True):
        primary_radius = group["radius_kpc_published"].to_numpy(dtype=float)
        locked_distance = float(audit.loc[galaxy, "distance_mpc"])
        distance_radius = (
            group["radius_arcsec"].to_numpy(dtype=float) * locked_distance * 1000.0 / 206265.0
        )
        actual = group["circular_speed_km_s"].to_numpy(dtype=float)
        uncertainty = group["circular_speed_error_km_s"].to_numpy(dtype=float)
        row = {
            "galaxy": galaxy,
            "published_distance_mpc": float(
                np.median(
                    group["radius_kpc_published"].to_numpy(dtype=float)
                    * 206265.0
                    / (group["radius_arcsec"].to_numpy(dtype=float) * 1000.0)
                )
            ),
            "locked_map_distance_mpc": locked_distance,
            "published_points": len(group),
        }
        common_primary = np.ones(len(group), dtype=bool)
        common_distance = np.ones(len(group), dtype=bool)
        model_values = {}
        model_distance_values = {}
        for model in MODELS:
            value, valid = interpolate_model(curves, galaxy, model, primary_radius)
            value_distance, valid_distance = interpolate_model(
                curves, galaxy, model, distance_radius
            )
            model_values[model] = value
            model_distance_values[model] = value_distance
            common_primary &= valid & np.isfinite(value)
            common_distance &= valid_distance & np.isfinite(value_distance)
        if common_primary.sum() < 3:
            continue
        row["valid_points"] = int(common_primary.sum())
        row["distance_consistent_valid_points"] = int(common_distance.sum())
        for model in MODELS:
            residual = model_values[model][common_primary] - actual[common_primary]
            row[f"RMSE_{model}"] = rmse(residual)
            weights = 1.0 / np.maximum(np.square(uncertainty[common_primary]), 1e-12)
            row[f"inverse_variance_RMSE_{model}"] = float(
                np.sqrt(np.sum(weights * np.square(residual)) / np.sum(weights))
            )
            if common_distance.sum() >= 3:
                row[f"distance_consistent_RMSE_{model}"] = rmse(
                    model_distance_values[model][common_distance] - actual[common_distance]
                )
            for index in np.flatnonzero(common_primary):
                predictions.append(
                    {
                        "galaxy": galaxy,
                        "model": model,
                        "radius_arcsec": float(group.iloc[index]["radius_arcsec"]),
                        "radius_kpc_published": float(primary_radius[index]),
                        "observed_circular_speed_km_s": float(actual[index]),
                        "observed_error_km_s": float(uncertainty[index]),
                        "predicted_circular_speed_km_s": float(model_values[model][index]),
                        "residual_km_s": float(residual[np.flatnonzero(common_primary).tolist().index(index)]),
                    }
                )
        scores.append(row)

    per_galaxy = pd.DataFrame(scores).sort_values("galaxy").reset_index(drop=True)
    if len(per_galaxy) < 11:
        raise RuntimeError("fewer than 11 galaxies have usable circular-speed support")
    sample = {model: sample_rmse(per_galaxy, f"RMSE_{model}") for model in MODELS}
    best_mond = min([MODEL_AQUAL, MODEL_QUMOND], key=sample.get)
    ratio = sample[MODEL_CANDIDATE] / sample[best_mond]

    morphology_rows = []
    medians = unlock["morphology_median_splits"]
    audit_columns = {
        "concentration_5log_r80_r20": "concentration_5log_r80_r20",
        "lopsidedness_180": "lopsidedness_180",
        "clumpiness_positive_highpass": "clumpiness_positive_highpass",
        "inclination_deg": "inclination_deg",
    }
    joined = per_galaxy.set_index("galaxy").join(audit[list(audit_columns.values())])
    for coordinate, threshold in medians.items():
        column = audit_columns[coordinate]
        for label, selected in {
            "low": joined[column] < float(threshold),
            "high": joined[column] >= float(threshold),
        }.items():
            subset = joined[selected]
            candidate_rmse = float(
                np.sqrt(np.mean(np.square(subset[f"RMSE_{MODEL_CANDIDATE}"])))
            )
            mond_rmse = float(np.sqrt(np.mean(np.square(subset[f"RMSE_{best_mond}"]))))
            morphology_rows.append(
                {
                    "coordinate": coordinate,
                    "bin": label,
                    "threshold": float(threshold),
                    "galaxies": len(subset),
                    "candidate_RMSE_km_s": candidate_rmse,
                    "best_MOND_model": best_mond,
                    "best_MOND_RMSE_km_s": mond_rmse,
                    "candidate_to_MOND_ratio": candidate_rmse / mond_rmse,
                }
            )
    morphology = pd.DataFrame(morphology_rows)
    max_morphology_ratio = float(morphology["candidate_to_MOND_ratio"].max())
    gates = unlock["rejection_thresholds"]["galaxy"]
    gate_results = {
        "minimum_valid_galaxies": len(per_galaxy) >= gates["minimum_valid_galaxies"],
        "equal_galaxy_RMSE": ratio <= gates["equal_galaxy_RMSE_ratio_to_best_frozen_MOND_max"],
        "morphology_bins": max_morphology_ratio
        <= gates["maximum_morphology_bin_RMSE_ratio_to_best_frozen_MOND"],
        "no_target_refit": bool(gates["no_target_refit"]),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    observed.to_csv(OUTPUT / "published_rotation_tables.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUTPUT / "point_predictions.csv", index=False)
    per_galaxy.to_csv(OUTPUT / "per_galaxy_scores.csv", index=False)
    morphology.to_csv(OUTPUT / "morphology_bin_scores.csv", index=False)

    figure, axes = plt.subplots(4, 4, figsize=(16, 14), sharex=False, sharey=False)
    axes = axes.ravel()
    colors = {
        MODEL_CANDIDATE: "#c0392b",
        MODEL_NEWTONIAN: "#7f8c8d",
        MODEL_AQUAL: "#2980b9",
        MODEL_QUMOND: "#27ae60",
    }
    labels = {
        MODEL_CANDIDATE: "candidate",
        MODEL_NEWTONIAN: "Newtonian",
        MODEL_AQUAL: "AQUAL",
        MODEL_QUMOND: "QUMOND",
    }
    for axis, galaxy in zip(axes, sorted(TABLE_NAMES), strict=False):
        data = observed[observed["galaxy"] == galaxy]
        axis.errorbar(
            data["radius_kpc_published"],
            data["circular_speed_km_s"],
            yerr=data["circular_speed_error_km_s"],
            fmt="o",
            ms=3,
            color="black",
            label="published $V_c$",
        )
        for model in MODELS:
            source = curves[(curves["system"] == galaxy) & (curves["model"] == model)]
            axis.plot(
                source["radius_kpc"],
                source["circular_speed_km_s"],
                color=colors[model],
                lw=1.4,
                label=labels[model],
            )
        axis.set_title(galaxy)
        axis.set_xlabel("R (kpc)")
        axis.set_ylabel("speed (km/s)")
        axis.grid(alpha=0.2)
    for axis in axes[len(TABLE_NAMES) :]:
        axis.set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=5)
    figure.suptitle("P0711 untouched LITTLE THINGS circular-speed validation")
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(OUTPUT / "rotation_curve_atlas.png", dpi=180)
    plt.close(figure)

    report = {
        "report_version": "P0711-EXTERNAL-GALAXY-ROTATION-VALIDATION-1.0.0",
        "status": "pass" if all(gate_results.values()) else "fail",
        "P0633_sample_spent": True,
        "universal_parameter_sha256": unlock["universal_parameter_sha256"],
        "supplement_sha256": sha256(SUPPLEMENT),
        "frozen_curve_sha256": sha256(CURVES),
        "valid_galaxies": len(per_galaxy),
        "sample_RMSE_km_s": sample,
        "best_frozen_MOND_model": best_mond,
        "candidate_to_best_MOND_RMSE_ratio": ratio,
        "maximum_morphology_bin_ratio": max_morphology_ratio,
        "gate_results": gate_results,
        "failed_gates": [name for name, passed in gate_results.items() if not passed],
        "primary_radius_rule": "published kpc radius exactly as frozen in P0709",
        "distance_consistency_is_diagnostic_only": True,
        "per_object_gravity_parameters": 0,
        "target_refits": 0,
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = f"""# P0711 untouched galaxy circular-speed validation

- Status: **{report['status'].upper()}**.
- Valid galaxies: **{len(per_galaxy)} / 13**.
- Candidate equal-galaxy RMSE: **{sample[MODEL_CANDIDATE]:.3f} km/s**.
- Best frozen full-field MOND: **{best_mond}**, **{sample[best_mond]:.3f} km/s**.
- Candidate / best-MOND ratio: **{ratio:.4f}** (gate: <= {gates['equal_galaxy_RMSE_ratio_to_best_frozen_MOND_max']:.2f}).
- Newtonian RMSE: **{sample[MODEL_NEWTONIAN]:.3f} km/s**.
- Worst predeclared morphology-bin ratio: **{max_morphology_ratio:.4f}** (gate: <= {gates['maximum_morphology_bin_RMSE_ratio_to_best_frozen_MOND']:.2f}).
- Per-object gravity parameters / target refits: **0 / 0**.
"""
    (OUTPUT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
