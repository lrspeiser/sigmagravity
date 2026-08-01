from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from voidscreen.basin_permittivity import (
    confined_slab_flat_velocity_km_s,
    required_confinement_half_height_kpc,
)
from voidscreen.unified import load_clash_acceleration_frame


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sparc_flat_catalog(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(
                    {
                        "galaxy": line[0:11].strip(),
                        "hubble_type": float(line[12:14]),
                        "inclination_deg": float(line[30:34]),
                        "luminosity_3p6_billion_solar": float(line[40:47]),
                        "effective_radius_kpc": float(line[56:61]),
                        "effective_surface_brightness": float(line[62:70]),
                        "disk_scale_kpc": float(line[71:76]),
                        "HI_mass_billion_solar": float(line[86:93]),
                        "flat_velocity_km_s": float(line[100:105]),
                        "flat_velocity_error_km_s": float(line[106:111]),
                        "quality": int(line[112:115]),
                    }
                )
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Malformed SPARC row at {path}:{line_number}") from exc
    frame = pd.DataFrame(rows)
    if len(frame) != 175 or frame["galaxy"].duplicated().any():
        raise ValueError("SPARC flat catalog must contain 175 unique galaxies")
    return frame


def assign_folds(names: pd.Series, folds: int) -> np.ndarray:
    order = sorted(
        range(len(names)),
        key=lambda index: hashlib.sha256(str(names.iloc[index]).encode()).hexdigest(),
    )
    assignment = np.empty(len(names), dtype=int)
    for rank, index in enumerate(order):
        assignment[index] = rank % folds
    return assignment


def fit_linear(train: pd.DataFrame, features: list[str], target: str) -> dict[str, object]:
    values = train[features].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0.0] = 1.0
    design = np.column_stack([np.ones(len(values)), (values - mean) / scale])
    coefficients, _, _, _ = np.linalg.lstsq(
        design, train[target].to_numpy(dtype=float), rcond=None
    )
    return {"features": features, "mean": mean, "scale": scale, "coefficients": coefficients}


def predict_linear(model: dict[str, object], frame: pd.DataFrame) -> np.ndarray:
    values = frame[model["features"]].to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(values)), (values - model["mean"]) / model["scale"]]
    )
    return design @ model["coefficients"]


def cross_validated_predictions(
    frame: pd.DataFrame,
    *,
    folds: int,
    void_values: np.ndarray | None = None,
) -> pd.DataFrame:
    work = frame.copy()
    if void_values is not None:
        work["void_score"] = void_values
    outputs = []
    structure_features = [
        "log_disk_scale_kpc",
        "log_effective_surface_brightness",
        "gas_fraction",
        "hubble_type",
    ]
    for fold in range(folds):
        train = work.loc[work["fold"] != fold]
        test = work.loc[work["fold"] == fold].copy()
        median_ratio = float(np.median(train["log_height_over_disk_scale"]))
        test["pred_log_height_constant_ratio"] = (
            test["log_disk_scale_kpc"] + median_ratio
        )
        mass_only = fit_linear(train, ["log_baryonic_mass_solar"], "log_required_height_kpc")
        test["pred_log_height_mass_only"] = predict_linear(mass_only, test)
        structure = fit_linear(train, structure_features, "log_required_height_kpc")
        test["pred_log_height_structure"] = predict_linear(structure, test)
        with_void = fit_linear(
            train, structure_features + ["void_score"], "log_required_height_kpc"
        )
        test["pred_log_height_structure_void"] = predict_linear(with_void, test)
        outputs.append(test)
    return pd.concat(outputs).sort_index()


def prediction_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    log_height_residual = (
        frame[prediction_column].to_numpy(dtype=float)
        - frame["log_required_height_kpc"].to_numpy(dtype=float)
    )
    predicted_height = np.power(10.0, frame[prediction_column].to_numpy(dtype=float))
    predicted_velocity = confined_slab_flat_velocity_km_s(
        frame["baryonic_mass_solar"].to_numpy(dtype=float), predicted_height
    )
    observed_velocity = frame["flat_velocity_km_s"].to_numpy(dtype=float)
    log_velocity_residual = np.log10(predicted_velocity / observed_velocity)
    return {
        "log10_height_rmse": float(np.sqrt(np.mean(np.square(log_height_residual)))),
        "log10_velocity_rmse": float(np.sqrt(np.mean(np.square(log_velocity_residual)))),
        "velocity_rmse_km_s": float(
            np.sqrt(np.mean(np.square(predicted_velocity - observed_velocity)))
        ),
        "velocity_median_absolute_error_km_s": float(
            np.median(np.abs(predicted_velocity - observed_velocity))
        ),
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    labels = ["minimum", "p05", "median", "p95", "maximum"]
    return dict(zip(labels, np.quantile(values, [0.0, 0.05, 0.5, 0.95, 1.0]), strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "nbm0_constitutive_basin_protocol.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "nbm0_constitutive_basin",
    )
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    settings = protocol["SPARC_boundary_inversion"]

    catalog_path = ROOT / settings["catalog"]
    environment_path = ROOT / settings["void_environment"]
    frame = parse_sparc_flat_catalog(catalog_path)
    frame = frame.loc[
        (frame["quality"] <= settings["quality_max"])
        & (frame["inclination_deg"] >= settings["inclination_min_deg"])
        & (frame["flat_velocity_km_s"] > 0.0)
        & (frame["flat_velocity_error_km_s"] > 0.0)
        & (frame["disk_scale_kpc"] > 0.0)
        & (frame["effective_surface_brightness"] > 0.0)
    ].copy()
    environment = pd.read_csv(environment_path)[["galaxy", "void_score"]]
    frame = frame.merge(environment, on="galaxy", how="inner", validate="one_to_one")
    stellar_mass = (
        settings["stellar_mass_to_light_3p6"]
        * frame["luminosity_3p6_billion_solar"]
    )
    gas_mass = settings["helium_factor"] * frame["HI_mass_billion_solar"]
    frame["baryonic_mass_solar"] = (stellar_mass + gas_mass) * 1.0e9
    frame["gas_fraction"] = gas_mass / (stellar_mass + gas_mass)
    frame["required_height_kpc"] = required_confinement_half_height_kpc(
        frame["baryonic_mass_solar"], frame["flat_velocity_km_s"]
    )
    frame["height_over_disk_scale"] = (
        frame["required_height_kpc"] / frame["disk_scale_kpc"]
    )
    frame["log_required_height_kpc"] = np.log10(frame["required_height_kpc"])
    frame["log_baryonic_mass_solar"] = np.log10(frame["baryonic_mass_solar"])
    frame["log_height_over_disk_scale"] = np.log10(frame["height_over_disk_scale"])
    frame["log_disk_scale_kpc"] = np.log10(frame["disk_scale_kpc"])
    frame["log_effective_surface_brightness"] = np.log10(
        frame["effective_surface_brightness"]
    )
    frame["fold"] = assign_folds(frame["galaxy"], settings["folds"])

    predictions = cross_validated_predictions(frame, folds=settings["folds"])
    metrics = {
        "constant_h_over_Rdisk": prediction_metrics(
            predictions, "pred_log_height_constant_ratio"
        ),
        "mass_only_BTFR_control": prediction_metrics(
            predictions, "pred_log_height_mass_only"
        ),
        "structure_only": prediction_metrics(predictions, "pred_log_height_structure"),
        "structure_plus_void": prediction_metrics(
            predictions, "pred_log_height_structure_void"
        ),
    }
    structural_rmse = metrics["structure_only"]["log10_height_rmse"]
    void_rmse = metrics["structure_plus_void"]["log10_height_rmse"]
    real_improvement = (structural_rmse - void_rmse) / structural_rmse

    rng = np.random.default_rng(settings["seed"])
    permutation_improvements = []
    original_void = frame["void_score"].to_numpy(dtype=float)
    for _ in range(settings["permutations"]):
        permuted = rng.permutation(original_void)
        permuted_predictions = cross_validated_predictions(
            frame, folds=settings["folds"], void_values=permuted
        )
        permuted_rmse = prediction_metrics(
            permuted_predictions, "pred_log_height_structure_void"
        )["log10_height_rmse"]
        permutation_improvements.append((structural_rmse - permuted_rmse) / structural_rmse)
    permutation_improvements = np.asarray(permutation_improvements)
    permutation_p = float(
        (1 + np.sum(permutation_improvements >= real_improvement))
        / (len(permutation_improvements) + 1)
    )

    geometry_gate = protocol["SPARC_boundary_inversion"]["geometry_continue_gate"]
    void_gate = protocol["SPARC_boundary_inversion"]["void_specific_gate"]
    geometry_pass = bool(
        metrics["structure_only"]["log10_velocity_rmse"]
        <= geometry_gate["heldout_log10_Vflat_rmse_max"]
    )
    confounding = settings["confounding_control_amendment"]
    mass_only_velocity_rmse = metrics["mass_only_BTFR_control"]["log10_velocity_rmse"]
    structure_velocity_rmse = metrics["structure_only"]["log10_velocity_rmse"]
    structure_over_mass_improvement = (
        mass_only_velocity_rmse - structure_velocity_rmse
    ) / mass_only_velocity_rmse
    boundary_identified = bool(
        structure_over_mass_improvement
        >= confounding["minimum_structure_relative_log_velocity_rmse_improvement"]
    )
    void_pass = bool(
        real_improvement >= void_gate["relative_rmse_improvement_min"]
        and permutation_p <= void_gate["permutation_p_max"]
    )

    clash = load_clash_acceleration_frame(ROOT / protocol["cluster_control"]["source"])
    outer_index = clash.groupby("system", sort=True)["radius_kpc"].idxmax()
    outer = clash.loc[outer_index].copy().sort_values("system")
    outer["epsilon_required"] = outer["gbar_m_s2"] / outer["observed_g_m_s2"]
    log_outer_epsilon = np.log10(outer["epsilon_required"].to_numpy(dtype=float))
    outer_constant_scatter = float(np.sqrt(np.mean(np.square(log_outer_epsilon - log_outer_epsilon.mean()))))

    if not geometry_pass:
        decision = "reject perfect flux-confinement closure"
    elif not boundary_identified:
        decision = "reject boundary interpretation as unseparated from the mass-only BTFR control"
    elif not void_pass:
        decision = "retain constitutive geometry only as prior-art benchmark; reject a CF4-void-specific claim"
    else:
        decision = "advance NBP0 to reciprocal covariant action derivation"

    report = {
        "report_version": "NBM0-A3-constitutive-basin-0.1",
        "status": "completed analytic and pre-action data diagnostics",
        "protocol": str(args.protocol.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(args.protocol),
        "input_hashes": {
            str(catalog_path.relative_to(ROOT)).replace("\\", "/"): sha256(catalog_path),
            str(environment_path.relative_to(ROOT)).replace("\\", "/"): sha256(environment_path),
        },
        "astronomical_gravity_parameter_fit_performed": False,
        "analytic_limits": {
            "spherical_constant_permittivity_speed_slope": -0.5,
            "perfect_slab_confined_speed_slope": 0.0,
            "flat_BTFR_requires_height_mass_exponent": 0.5,
        },
        "SPARC_boundary_inversion": {
            "galaxies": int(len(frame)),
            "required_height_kpc_quantiles": quantiles(
                frame["required_height_kpc"].to_numpy(dtype=float)
            ),
            "height_over_disk_scale_quantiles": quantiles(
                frame["height_over_disk_scale"].to_numpy(dtype=float)
            ),
            "cross_validated_metrics": metrics,
            "geometry_continue_gate_pass": geometry_pass,
            "structure_relative_velocity_rmse_improvement_over_mass_only": float(
                structure_over_mass_improvement
            ),
            "boundary_identified_beyond_BTFR_control": boundary_identified,
            "void_relative_height_rmse_improvement": float(real_improvement),
            "void_permutation_p": permutation_p,
            "void_specific_gate_pass": void_pass,
            "permutation_improvement_quantiles": quantiles(permutation_improvements),
        },
        "CLASH_spherical_outer_control": {
            "systems": int(len(outer)),
            "epsilon_required_quantiles": quantiles(
                outer["epsilon_required"].to_numpy(dtype=float)
            ),
            "constant_log10_epsilon_scatter_dex": outer_constant_scatter,
            "theory_neutral_likelihood": False,
        },
        "reciprocal_action_gate_pass": False,
        "reciprocal_action_disposition": "not yet derived; epsilon'(X)|grad Phi|^2 backreaction remains mandatory",
        "same_metric_lensing_gate_pass": False,
        "same_metric_lensing_disposition": "Newtonian constitutive equation alone does not predict Phi+Psi",
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "sparc_boundary_predictions.csv", index=False)
    outer.to_csv(args.output_dir / "clash_outer_epsilon_control.csv", index=False)
    pd.DataFrame(
        {"permutation": np.arange(len(permutation_improvements)), "relative_improvement": permutation_improvements}
    ).to_csv(args.output_dir / "void_permutations.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
