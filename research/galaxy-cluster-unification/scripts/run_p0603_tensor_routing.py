#!/usr/bin/env python3
"""Cross-test a conservative baryonic tensor-routing Green kernel."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_fresh_sample import (  # noqa: E402
    build_source_context,
    prediction_protocol,
    target_from_path,
)
from run_gravity_arc_tomography import (  # noqa: E402
    normalized_in_aperture,
    prediction_for_spec,
    shape_metrics,
)
from run_p0580_conservative_return_sparc import galaxy_force_profile, score  # noqa: E402
from run_p0593_diffusion_cross_domain import acceleration_velocity, characteristic_acceleration  # noqa: E402
from voidscreen.conservative_diffusion import (  # noqa: E402
    low_acceleration_activation,
    radial_shape_activation,
)
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.tensor_routing import (  # noqa: E402
    anisotropic_gaussian_deposit,
    baryonic_field_frames,
    curl_free_deflection_diagnostic,
    redistributed_cumulative_mass_tensor,
    weighted_radii,
)
from voidscreen.unified import G_SI, M_SUN_KG, rar_acceleration  # noqa: E402


SOLAR_RADIUS_M = 6.957e8


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def balanced_folds(names, folds: int) -> dict[str, int]:
    ordered = sorted(
        set(map(str, names)),
        key=lambda name: hashlib.sha256(name.encode("utf-8")).hexdigest(),
    )
    return {name: index % int(folds) for index, name in enumerate(ordered)}


def candidate_specs(protocol: dict) -> list[dict]:
    base = dict(protocol["base_candidate"])
    base.update(
        {
            "candidate_id": "T000_BASE",
            "changed_parameter": "baseline",
            "changed_level": "base",
        }
    )
    specs = [base]
    for parameter, levels in protocol["one_at_a_time"].items():
        if parameter == "unique_candidate_count":
            continue
        for level in levels:
            if level == protocol["base_candidate"][parameter]:
                continue
            spec = dict(protocol["base_candidate"])
            spec[parameter] = level
            spec.update(
                {
                    "candidate_id": f"T{len(specs):03d}_{parameter}_{level}",
                    "changed_parameter": parameter,
                    "changed_level": str(level),
                }
            )
            specs.append(spec)
    if len(specs) != int(protocol["one_at_a_time"]["unique_candidate_count"]):
        raise RuntimeError("P0603 tensor candidate count changed")
    return specs


def shape_factor(mode: str, concentration: float, protocol: dict) -> float:
    constants = protocol["fixed_constants"]
    activation = radial_shape_activation(
        concentration,
        midpoint=constants["shape_midpoint"],
        width=constants["shape_width"],
    )
    if mode == "none":
        return 1.0
    if mode == "sqrt_H":
        return float(np.sqrt(activation))
    if mode == "H":
        return activation
    if mode == "H2":
        return activation**2
    raise ValueError(mode)


def tensor_prediction(context, spec: dict, protocol: dict):
    constants = protocol["fixed_constants"]
    center, r50, r80, concentration = weighted_radii(
        context.positions, context.soft_weights
    )
    frames = baryonic_field_frames(
        context.positions,
        context.soft_weights,
        softening=constants["tidal_softening_kpc"],
    )
    mix = float(spec["external_direction_mix"])
    drift = (1.0 - mix) * frames["inward"] + mix * frames["external"]
    drift_norm = np.linalg.norm(drift, axis=1)
    drift /= np.maximum(drift_norm[:, None], np.finfo(float).tiny)
    drift[drift_norm == 0.0] = frames["inward"][drift_norm == 0.0]
    source_ratio = np.maximum(frames["radius"] / r80, 1.0e-6)
    travel = float(spec["length_over_R80"]) * r80 * np.clip(
        source_ratio ** float(spec["radius_exponent"]), 0.2, 3.0
    )
    endpoints = context.positions + travel[:, None] * drift
    orientation = str(spec["tensor_orientation"])
    if orientation == "tidal":
        axes = frames["tidal"]
    elif orientation == "radial":
        axes = frames["inward"]
    elif orientation == "external":
        axes = frames["external"]
    else:
        raise ValueError(orientation)
    local = anisotropic_gaussian_deposit(
        context.axis_kpc,
        context.positions,
        context.soft_weights,
        frames["tidal"],
        geometric_sigma=float(constants["local_width_over_R80"]) * r80,
        axis_ratio=1.0,
        axis_samples=int(constants["kernel_axis_samples"]),
    )
    routed = anisotropic_gaussian_deposit(
        context.axis_kpc,
        endpoints,
        context.soft_weights,
        axes,
        geometric_sigma=float(spec["width_over_R80"]) * r80,
        axis_ratio=float(spec["tensor_axis_ratio"]),
        axis_samples=int(constants["kernel_axis_samples"]),
    )
    effective_fraction = float(spec["fraction_max"]) * shape_factor(
        str(spec["shape_gate"]), concentration, protocol
    )
    prediction_full = (1.0 - effective_fraction) * local + effective_fraction * routed
    prediction = normalized_in_aperture(prediction_full, context.aperture)
    return prediction, {
        "center_x_kpc": center[0],
        "center_y_kpc": center[1],
        "R50_kpc": r50,
        "R80_kpc": r80,
        "concentration_R50_over_R80": concentration,
        "effective_route_fraction": effective_fraction,
        "full_grid_integral": float(np.sum(prediction_full)),
        **curl_free_deflection_diagnostic(
            prediction_full,
            float(context.axis_kpc[1] - context.axis_kpc[0]),
        ),
    }


def load_cluster_contexts(protocol: dict):
    acquisition = json.loads(
        (ROOT / protocol["cluster_data"]["acquisition_protocol"]).read_text()
    )
    sources = pd.read_csv(ROOT / protocol["cluster_data"]["sources"])
    systems = pd.read_csv(ROOT / protocol["cluster_data"]["systems"]).set_index("system")
    raw = ROOT / acquisition["acquisition"]["output_directory"]
    settings = acquisition["spatial_preprocessing"]
    contexts = {}
    targets = {}
    for system in acquisition["systems"]:
        label = system["label"]
        context, world = build_source_context(system, systems.loc[label], sources, settings)
        models = {model["method"]: model for model in system["models"]}
        lenstool = models["lenstool"]
        glafic = models["glafic"]
        lenstool_path = (
            raw / "models" / system["slug"] / "lenstool" / lenstool["best_filename"]
        )
        glafic_path = (
            raw / "models" / system["slug"] / "glafic" / glafic["best_filename"]
        )
        contexts[label] = context
        targets[(label, "lenstool_best")] = target_from_path(
            lenstool_path, world, context, settings
        )
        targets[(label, "glafic_best")] = target_from_path(
            glafic_path, world, context, settings
        )
    return acquisition, contexts, targets


def control_specs(acquisition: dict) -> list[dict]:
    wanted = set(["LOCAL75", "CENTRAL100", "C0351", "W060"])
    return [item for item in acquisition["locked_candidates"] if item["candidate_id"] in wanted]


def score_clusters(protocol: dict, specs: list[dict]):
    acquisition, contexts, targets = load_cluster_contexts(protocol)
    controls = control_specs(acquisition)
    forward_protocol = prediction_protocol(acquisition)
    records = []
    prediction_cache = {}
    diagnostic_rows = []
    for label, context in contexts.items():
        for spec in specs:
            prediction, diagnostic = tensor_prediction(context, spec, protocol)
            prediction_cache[(label, spec["candidate_id"])] = prediction
            diagnostic_rows.append({"system": label, "candidate_id": spec["candidate_id"], **diagnostic})
        for control in controls:
            prediction_cache[(label, control["candidate_id"])] = prediction_for_spec(
                context, control, forward_protocol
            )
        for target_kind in ("lenstool_best", "glafic_best"):
            target = targets[(label, target_kind)]
            for spec in specs:
                records.append(
                    {
                        "system": label,
                        "target_kind": target_kind,
                        **spec,
                        **shape_metrics(
                            prediction_cache[(label, spec["candidate_id"])],
                            target,
                            context.aperture,
                        ),
                    }
                )
            for control in controls:
                records.append(
                    {
                        "system": label,
                        "target_kind": target_kind,
                        "candidate_id": control["candidate_id"],
                        "changed_parameter": "control",
                        "changed_level": control["candidate_id"],
                        **shape_metrics(
                            prediction_cache[(label, control["candidate_id"])],
                            target,
                            context.aperture,
                        ),
                    }
                )
    return (
        pd.DataFrame(records),
        pd.DataFrame(diagnostic_rows),
        contexts,
        targets,
        prediction_cache,
    )


def cross_validate_clusters(scores: pd.DataFrame, specs: list[dict], folds: int):
    tensor = scores[
        scores.candidate_id.isin([spec["candidate_id"] for spec in specs])
        & scores.target_kind.eq("lenstool_best")
    ].copy()
    assignment = balanced_folds(tensor.system.unique(), folds)
    fold_rows = []
    oof_rows = []
    for fold in range(folds):
        test_systems = {name for name, value in assignment.items() if value == fold}
        training = tensor[~tensor.system.isin(test_systems)]
        means = training.groupby("candidate_id").jensen_shannon.mean().sort_values()
        selected_id = str(means.index[0])
        selected_spec = next(spec for spec in specs if spec["candidate_id"] == selected_id)
        fold_rows.append(
            {
                "fold": fold,
                "selected_candidate_id": selected_id,
                "training_equal_JS": float(means.iloc[0]),
                "training_systems": training.system.nunique(),
                "heldout_systems": len(test_systems),
                **selected_spec,
            }
        )
        held = scores[
            scores.system.isin(test_systems)
            & scores.target_kind.isin(["lenstool_best", "glafic_best"])
            & scores.candidate_id.isin([selected_id, "LOCAL75", "CENTRAL100", "C0351", "W060"])
        ].copy()
        held["fold"] = fold
        held["selected_tensor_candidate_id"] = selected_id
        oof_rows.append(held)
    return pd.DataFrame(fold_rows), pd.concat(oof_rows, ignore_index=True)


def cluster_summary(oof: pd.DataFrame) -> dict:
    output = {}
    for target_kind in ("lenstool_best", "glafic_best"):
        block = oof[oof.target_kind.eq(target_kind)]
        selected = block[~block.candidate_id.isin(["LOCAL75", "CENTRAL100", "C0351", "W060"])]
        refs = {}
        for control in ("LOCAL75", "CENTRAL100", "C0351", "W060"):
            ref = block[block.candidate_id.eq(control)].set_index("system")
            aligned = selected.set_index("system")
            refs[control] = {
                "equal_JS": float(ref.jensen_shannon.mean()),
                "tensor_improvement_fraction": float(
                    1.0 - aligned.jensen_shannon.mean() / ref.jensen_shannon.mean()
                ),
                "systems_tensor_better": int(
                    np.sum(aligned.jensen_shannon < ref.jensen_shannon)
                ),
            }
        output[target_kind] = {
            "tensor_equal_JS": float(selected.jensen_shannon.mean()),
            "tensor_equal_Pearson": float(selected.pearson.mean()),
            "controls": refs,
        }
    return output


def parameter_impacts(scores: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    primary = scores[
        scores.target_kind.eq("lenstool_best")
        & scores.candidate_id.isin([spec["candidate_id"] for spec in specs])
    ]
    means = primary.groupby("candidate_id").jensen_shannon.mean()
    base = next(spec for spec in specs if spec["candidate_id"] == "T000_BASE")
    rows = []
    for parameter in (
        "fraction_max",
        "length_over_R80",
        "radius_exponent",
        "width_over_R80",
        "external_direction_mix",
        "tensor_axis_ratio",
        "tensor_orientation",
        "shape_gate",
    ):
        family = [base] + [spec for spec in specs if spec["changed_parameter"] == parameter]
        values = [(str(spec[parameter]), float(means.loc[spec["candidate_id"]])) for spec in family]
        ordered = sorted(values, key=lambda item: item[1])
        rows.append(
            {
                "parameter": parameter,
                "best_level": ordered[0][0],
                "worst_level": ordered[-1][0],
                "equal_JS_span": ordered[-1][1] - ordered[0][1],
                "best_equal_JS": ordered[0][1],
                "worst_equal_JS": ordered[-1][1],
            }
        )
    return pd.DataFrame(rows).sort_values("equal_JS_span", ascending=False)


def galaxy_and_solar_scores(protocol: dict, specs: list[dict]):
    galaxy_protocol = json.loads(
        (ROOT / protocol["galaxy_data"]["protocol"]).read_text()
    )
    cfg = galaxy_protocol["galaxy_test"]
    raw = pd.read_csv(ROOT / cfg["points"])
    points = raw[(raw.model == cfg["model"]) & (raw.scenario == cfg["scenario"])].copy()
    points["source_point_index"] = points.index
    outer = points[points.split.eq(cfg["split"])].copy().reset_index(drop=True)
    if points.galaxy.nunique() != protocol["galaxy_data"]["galaxies"] or len(outer) != protocol["galaxy_data"]["outer_points"]:
        raise RuntimeError("P0603 galaxy coverage changed")
    profiles = {
        galaxy: galaxy_force_profile(block)
        for galaxy, block in points.groupby("galaxy", sort=False)
    }
    a0 = float(protocol["fixed_constants"]["a0_m_s2"])
    rows = []
    prediction_by_id = {}
    maximum_conservation_error = 0.0
    for spec in specs:
        prediction = np.empty(len(outer), dtype=float)
        for galaxy, raw_indices in outer.groupby("galaxy", sort=False).indices.items():
            indices = np.asarray(raw_indices, dtype=int)
            profile = profiles[galaxy]
            concentration = float(profile["concentration_R50_over_R80"])
            shape = shape_factor(str(spec["shape_gate"]), concentration, protocol)
            screen = low_acceleration_activation(
                characteristic_acceleration(profile),
                a0_m_s2=a0,
                power=float(protocol["fixed_constants"]["source_acceleration_gate_power"]),
            )
            fraction = float(spec["fraction_max"]) * shape * screen
            routed, error = redistributed_cumulative_mass_tensor(
                profile["radius_kpc"],
                profile["mass_solar"],
                r80=profile["R80_kpc"],
                length_over_r80=float(spec["length_over_R80"]),
                radius_exponent=float(spec["radius_exponent"]),
                width_over_r80=float(spec["width_over_R80"]),
                axis_ratio=float(spec["tensor_axis_ratio"]),
                bins=int(protocol["fixed_constants"]["radial_bins"]),
            )
            maximum_conservation_error = max(maximum_conservation_error, error)
            effective_mass = (1.0 - fraction) * profile["mass_solar"] + fraction * routed
            g_eff = G_SI * M_SUN_KG * effective_mass / np.square(
                profile["radius_kpc"] * KPC_M
            )
            velocity = acceleration_velocity(
                profile["radius_kpc"], rar_acceleration(g_eff, a0)
            )
            frame = profile["frame"]
            mask = frame.split.to_numpy(str) == cfg["split"]
            by_source = dict(
                zip(frame.loc[mask, "source_point_index"].to_numpy(int), velocity[mask])
            )
            prediction[indices] = [
                by_source[value]
                for value in outer.loc[indices, "source_point_index"].to_numpy(int)
            ]
        prediction_by_id[spec["candidate_id"]] = prediction
        metrics = score(outer, prediction)
        solar = solar_diagnostic(protocol, spec)
        rows.append({**spec, **metrics, **solar})
    reference = score(outer, outer.velocity_RAR_same_nuisance_km_s.to_numpy(float))
    return pd.DataFrame(rows), prediction_by_id, outer, reference, maximum_conservation_error


def solar_diagnostic(protocol: dict, spec: dict) -> dict:
    samples = 512
    radius = np.linspace(SOLAR_RADIUS_M / samples, SOLAR_RADIUS_M, samples)
    mass = (radius / SOLAR_RADIUS_M) ** 3
    r80 = SOLAR_RADIUS_M * 0.8 ** (1.0 / 3.0)
    r50 = SOLAR_RADIUS_M * 0.5 ** (1.0 / 3.0)
    concentration = r50 / r80
    routed, _ = redistributed_cumulative_mass_tensor(
        radius,
        mass,
        r80=r80,
        length_over_r80=float(spec["length_over_R80"]),
        radius_exponent=float(spec["radius_exponent"]),
        width_over_r80=float(spec["width_over_R80"]),
        axis_ratio=float(spec["tensor_axis_ratio"]),
        bins=int(protocol["fixed_constants"]["radial_bins"]),
    )
    g_r80 = G_SI * M_SUN_KG * 0.8 / r80**2
    activation = low_acceleration_activation(
        g_r80,
        a0_m_s2=float(protocol["fixed_constants"]["a0_m_s2"]),
        power=float(protocol["fixed_constants"]["source_acceleration_gate_power"]),
    )
    fraction = float(spec["fraction_max"]) * shape_factor(
        str(spec["shape_gate"]), concentration, protocol
    ) * activation
    effective = (1.0 - fraction) * mass + fraction * routed
    return {
        "solar_activation": activation,
        "solar_effective_route_fraction": fraction,
        "solar_maximum_absolute_interior_force_change": float(
            np.max(np.abs(effective / mass - 1.0))
        ),
    }


def make_figure(cluster_summary_data, impacts, galaxy_scores, galaxy_reference, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    primary = cluster_summary_data["lenstool_best"]
    labels = ["tensor OOF", "LOCAL75", "CENTRAL100", "C0351", "W060"]
    values = [primary["tensor_equal_JS"]] + [
        primary["controls"][name]["equal_JS"] for name in labels[1:]
    ]
    axes[0].bar(labels, values, color=["#1261A0", "#888888", "#777777", "#AA7744", "#55A868"])
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(ylabel="equal-cluster JS divergence", title="Whole-cluster OOF morphology")
    display = impacts.sort_values("equal_JS_span")
    axes[1].barh(display.parameter, display.equal_JS_span, color="#1261A0")
    axes[1].set(xlabel="equal-JS response span", title="Tensor parameter impact")
    axes[2].scatter(
        galaxy_scores.outer_equal_galaxy_RMSE_km_s,
        galaxy_scores.groupby("candidate_id").size().reindex(galaxy_scores.candidate_id).to_numpy() * 0,
        alpha=0.65,
    )
    axes[2].axvline(
        galaxy_reference["outer_equal_galaxy_RMSE_km_s"],
        color="black",
        linestyle="--",
        label="fixed RAR",
    )
    axes[2].set(xlabel="galaxy equal RMSE (km/s)", yticks=[], title="Axisymmetric galaxy transfer")
    axes[2].legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0603_tensor_routing_protocol.json").read_text()
    )
    specs = candidate_specs(protocol)
    cluster_scores, field_diagnostics, contexts, targets, prediction_cache = score_clusters(
        protocol, specs
    )
    fold_selections, oof = cross_validate_clusters(
        cluster_scores, specs, int(protocol["cluster_validation"]["folds"])
    )
    cluster_metrics = cluster_summary(oof)
    impacts = parameter_impacts(cluster_scores, specs)
    galaxy_scores, galaxy_predictions, galaxy_outer, galaxy_reference, conservation_error = (
        galaxy_and_solar_scores(protocol, specs)
    )

    selection_counts = fold_selections.selected_candidate_id.value_counts()
    consensus_id = str(selection_counts.index[0])
    consensus_spec = next(spec for spec in specs if spec["candidate_id"] == consensus_id)
    selected_prediction = galaxy_outer.copy()
    selected_prediction["prediction_km_s"] = galaxy_predictions[consensus_id]
    selected_prediction["candidate_id"] = consensus_id
    solar_gate = float(protocol["solar_proxy"]["maximum_fractional_interior_force_change"])
    consensus_galaxy = galaxy_scores.set_index("candidate_id").loc[consensus_id]
    report = {
        "report_version": "P0603-TENSOR-ROUTING-RESULTS-0.1.0",
        "status": "complete_spent_map_cross_validation_and_cross_domain_transfer",
        "coverage": {
            "tensor_candidates": len(specs),
            "clusters": len(contexts),
            "cluster_targets": 2 * len(contexts),
            "cluster_folds": len(fold_selections),
            "galaxies": galaxy_outer.galaxy.nunique(),
            "galaxy_outer_points": len(galaxy_outer),
        },
        "cluster_oof": cluster_metrics,
        "fold_selections": fold_selections.to_dict("records"),
        "unique_selected_tensor_candidates": int(fold_selections.selected_candidate_id.nunique()),
        "consensus_candidate": consensus_spec,
        "consensus_selected_folds": int(selection_counts.loc[consensus_id]),
        "parameter_impacts": impacts.to_dict("records"),
        "galaxy_reference_fixed_RAR": galaxy_reference,
        "consensus_galaxy": consensus_galaxy.to_dict(),
        "galaxy_equal_RMSE_ratio_to_fixed_RAR": float(
            consensus_galaxy.outer_equal_galaxy_RMSE_km_s
            / galaxy_reference["outer_equal_galaxy_RMSE_km_s"]
        ),
        "solar": {
            "activation": consensus_galaxy.solar_activation,
            "effective_route_fraction": consensus_galaxy.solar_effective_route_fraction,
            "maximum_absolute_interior_force_change": consensus_galaxy.solar_maximum_absolute_interior_force_change,
            "proxy_gate": solar_gate,
            "passes_proxy_gate": bool(
                consensus_galaxy.solar_maximum_absolute_interior_force_change <= solar_gate
            ),
            "PPN_or_Cassini_metric_derived": False,
        },
        "field_equation_diagnostics": {
            "maximum_full_grid_mass_conservation_error": float(
                np.max(np.abs(field_diagnostics.full_grid_integral - 1.0))
            ),
            "maximum_relative_curl_norm": float(field_diagnostics.relative_curl_norm.max()),
            "maximum_relative_Poisson_residual": float(
                field_diagnostics.relative_poisson_residual.max()
            ),
            "maximum_radial_projection_conservation_error": conservation_error,
        },
        "strict_interpretation": {
            "cluster_targets_are_fresh": False,
            "whole_system_cross_validation_used": True,
            "GLAFIC_used_for_selection": False,
            "dark_component_inserted": False,
            "covariant_action_derived": False,
            "raw_cluster_lensing_tested": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    cluster_scores.to_csv(output / protocol["outputs"]["cluster_candidate_scores"], index=False)
    fold_selections.to_csv(output / protocol["outputs"]["cluster_fold_selections"], index=False)
    oof.to_csv(output / protocol["outputs"]["cluster_oof_scores"], index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    galaxy_scores.to_csv(output / protocol["outputs"]["galaxy_candidate_scores"], index=False)
    selected_prediction.to_csv(output / protocol["outputs"]["selected_galaxy_predictions"], index=False)
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n"
    )
    make_figure(
        cluster_metrics,
        impacts,
        galaxy_scores,
        galaxy_reference,
        output / protocol["outputs"]["figure"],
    )
    primary = cluster_metrics["lenstool_best"]
    summary = (
        "# P0603 tensor-routing result\n\n"
        f"OOF Lenstool equal-cluster JS: **{primary['tensor_equal_JS']:.5f}**; "
        f"improvement over LOCAL75: **{100 * primary['controls']['LOCAL75']['tensor_improvement_fraction']:.2f}%**; "
        f"over CENTRAL100: **{100 * primary['controls']['CENTRAL100']['tensor_improvement_fraction']:.2f}%**.\n\n"
        f"Consensus candidate: **{consensus_id}**, selected in {selection_counts.loc[consensus_id]}/5 folds. "
        f"Galaxy equal RMSE: **{consensus_galaxy.outer_equal_galaxy_RMSE_km_s:.3f} km/s** "
        f"versus fixed RAR **{galaxy_reference['outer_equal_galaxy_RMSE_km_s']:.3f} km/s**.\n\n"
        "The cluster maps are spent; this is exploratory cross-validation, not fresh confirmation.\n"
    )
    (output / protocol["outputs"]["summary"]).write_text(summary)
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
