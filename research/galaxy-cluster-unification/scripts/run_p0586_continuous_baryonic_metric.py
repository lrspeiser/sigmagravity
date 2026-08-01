#!/usr/bin/env python3
"""Test a continuous positive baryonic field metric on raw lenses and SPARC."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_member_tidal_metric import (  # noqa: E402
    MemberTidalLens,
    build_contexts,
    fit_context,
)
from run_p0559_accept_projected_gas_tidal import (  # noqa: E402
    physical_catalogs,
    prepare_registered_maps,
)
from run_p0570_physical_baryon_residual_lensing import source_plane_rms  # noqa: E402
from voidscreen.baryonic_metric import (  # noqa: E402
    build_baryonic_metric_correction_field,
    prepare_baryonic_metric_state,
    prepare_baryonic_metric_workspace,
    spherical_metric_acceleration,
)


G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30
AU_M = 149_597_870_700.0
KPC_M = 3.085677581491367e19
JULIAN_YEAR_DAYS = 365.25
RAD_TO_MAS = 206_264_806.24709636


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_id(epsilon, a0, power, tau, width):
    return (
        f"e{epsilon:.2f}_a{a0 / 1.0e-10:.1f}_n{power:.0f}_"
        f"t{tau:+.1f}_w{width:.2f}"
    ).replace(".", "d").replace("+", "p").replace("-", "m")


def is_identity(epsilon, tau) -> bool:
    return math.isclose(float(epsilon), 1.0) and math.isclose(float(tau), 0.0)


def equal_system_velocity_rmse(frame: pd.DataFrame, prediction) -> tuple[float, float]:
    observed = frame.velocity_observed_adjusted_km_s.to_numpy(float)
    residual = np.asarray(prediction, dtype=float) - observed
    point = float(np.sqrt(np.mean(np.square(residual))))
    temporary = pd.DataFrame(
        {"galaxy": frame.galaxy.to_numpy(str), "square": np.square(residual)}
    )
    equal = float(np.sqrt(temporary.groupby("galaxy").square.mean().mean()))
    return point, equal


def galaxy_prediction(frame, epsilon, a0, power):
    gbar = frame.g_bar_m_s2.to_numpy(float)
    predicted = spherical_metric_acceleration(
        gbar,
        minimum_permittivity=float(epsilon),
        a0_m_s2=float(a0),
        gate_power=float(power),
    )
    radius_m = frame.radius_adjusted_kpc.to_numpy(float) * KPC_M
    return np.sqrt(predicted * radius_m) / 1000.0


def solar_fraction(radius_m, epsilon, a0, power):
    gbar = G_SI * M_SUN_KG / np.square(np.asarray(radius_m, dtype=float))
    predicted = spherical_metric_acceleration(
        gbar,
        minimum_permittivity=float(epsilon),
        a0_m_s2=float(a0),
        gate_power=float(power),
    )
    return predicted / gbar - 1.0


def mercury_precession(epsilon, a0, power, points=32768):
    semimajor = 0.38709893 * AU_M
    eccentricity = 0.205630
    period_days = 87.9691
    anomaly = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    cosine = np.cos(anomaly)
    one_minus_e2 = 1.0 - eccentricity**2
    radius = semimajor * one_minus_e2 / (1.0 + eccentricity * cosine)
    fraction = solar_fraction(radius, epsilon, a0, power)
    perturbation = -(G_SI * M_SUN_KG / radius**2) * fraction
    time_weight = one_minus_e2**1.5 / np.square(1.0 + eccentricity * cosine)
    mean_r_cosine = float(np.mean(perturbation * cosine * time_weight))
    period_seconds = period_days * 86400.0
    mean_motion = 2.0 * np.pi / period_seconds
    mean_rate = (
        -math.sqrt(one_minus_e2)
        / (mean_motion * semimajor * eccentricity)
        * mean_r_cosine
    )
    radians_per_orbit = mean_rate * period_seconds
    orbits_per_century = 100.0 * JULIAN_YEAR_DAYS / period_days
    return radians_per_orbit * orbits_per_century * RAD_TO_MAS


def aggregate_exact(exact: pd.DataFrame, labels: set[str], role: str) -> dict:
    block = exact[
        exact.row_type.eq("system")
        & exact.system_label.isin(labels)
        & exact.role.eq(role)
    ]
    heldout = block.heldout_exact_RMS_arcsec.to_numpy(float)
    finite = heldout[np.isfinite(heldout)]
    return {
        "training_exact_RMS_arcsec": float(
            np.sqrt(np.mean(np.square(block.training_exact_RMS_arcsec.to_numpy(float))))
        ),
        "heldout_exact_RMS_arcsec": (
            float(np.sqrt(np.mean(np.square(finite)))) if finite.size else math.nan
        ),
        "finite_systems": int(finite.size),
        "all_training_roots": bool(block.all_training_roots.all()),
        "all_heldout_roots": bool(block.all_heldout_roots.all()),
    }


def main():
    protocol_path = ROOT / "configs/p0586_continuous_baryonic_metric_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_continuous_baryonic_metric_score":
        raise RuntimeError("P0586 protocol is not frozen")
    p0559 = json.loads(
        (ROOT / protocol["inputs"]["p0559_protocol"]).read_text(encoding="utf-8")
    )
    p0557 = json.loads(
        (ROOT / p0559["inputs"]["p0557_protocol"]).read_text(encoding="utf-8")
    )
    member = json.loads(
        (ROOT / p0559["inputs"]["member_tidal_protocol"]).read_text(encoding="utf-8")
    )
    member["optimization"]["maximum_function_evaluations"] = int(
        p0559["optimization"]["maximum_function_evaluations"]
    )
    contexts, _, _ = build_contexts(
        member, softening_kpc=float(p0559["locked_field"]["softening_kpc"])
    )
    registered = prepare_registered_maps(p0559, contexts)
    physical, physical_audits = physical_catalogs(p0559, contexts, registered)
    physical_audits = physical_audits.set_index("system_label")
    numerical = protocol["numerics"]
    factorial = protocol["factorial"]
    selection_labels = set(protocol["validation"]["selection_systems"])
    validation_labels = set(protocol["validation"]["validation_systems"])

    catalogs = {}
    masses = {}
    workspaces = {}
    states = {}
    for context in contexts:
        label = context.system["label"]
        catalog = physical[label][("accept_absolute", 0.5, True)]
        total_mass = float(
            physical_audits.loc[label, "stellar_mass_assigned_to_map_msun"]
            + physical_audits.loc[label, "projected_ACCEPT_gas_mass_on_map_msun"]
        )
        scale = float(
            context.local_protocol["cosmology_and_coordinates"][
                "angular_scale_kpc_per_arcsec"
            ]
        )
        catalogs[label] = catalog
        masses[label] = total_mass
        print(f"P0586 workspace {label}", flush=True)
        workspace = prepare_baryonic_metric_workspace(
            catalog.x_arcsec.to_numpy(float),
            catalog.y_arcsec.to_numpy(float),
            catalog.normalized_light_weight.to_numpy(float),
            total_mass_msun=total_mass,
            scale_kpc_per_arcsec=scale,
            half_width_arcsec=float(numerical["field_half_width_arcsec"]),
            pixels_per_axis=int(numerical["field_pixels_per_axis"]),
            point_softening_arcsec=float(numerical["point_softening_arcsec"]),
        )
        workspaces[label] = workspace
        for width in map(float, factorial["smoothing_r80_fraction"]):
            states[(label, width)] = prepare_baryonic_metric_state(workspace, width)

    baseline_fits = {}
    for index, context in enumerate(contexts):
        label = context.system["label"]
        print(f"P0586 zero exact fit {label}", flush=True)
        baseline_fits[label] = fit_context(
            context,
            0.0,
            starts=8,
            seed=20261000 + index,
        )

    candidate_grid = list(
        itertools.product(
            map(float, factorial["minimum_permittivity"]),
            map(float, factorial["a0_m_s2"]),
            map(float, factorial["gate_power"]),
            map(float, factorial["anisotropy_tau"]),
            map(float, factorial["smoothing_r80_fraction"]),
        )
    )
    if len(candidate_grid) != int(factorial["candidate_count"]):
        raise RuntimeError("P0586 candidate count differs from the frozen protocol")
    screen_rows = []
    audit_rows = []
    for candidate_index, (epsilon, a0, power, tau, width) in enumerate(candidate_grid):
        cid = candidate_id(epsilon, a0, power, tau, width)
        system_scores = []
        for context in contexts:
            label = context.system["label"]
            if label not in selection_labels:
                continue
            catalog = catalogs[label]
            field = build_baryonic_metric_correction_field(
                catalog.x_arcsec.to_numpy(float),
                catalog.y_arcsec.to_numpy(float),
                catalog.normalized_light_weight.to_numpy(float),
                total_mass_msun=masses[label],
                scale_kpc_per_arcsec=workspaces[label].scale_kpc_per_arcsec,
                minimum_permittivity=epsilon,
                a0_m_s2=a0,
                gate_power=power,
                anisotropy=tau,
                smoothing_r80_fraction=width,
                asymmetry_threshold=float(numerical["asymmetry_threshold"]),
                asymmetry_power=float(numerical["asymmetry_power"]),
                workspace=workspaces[label],
                state=states[(label, width)],
            )
            lens = MemberTidalLens(context.local_protocol, context.fields, field, 1.0)
            baseline = baseline_fits[label]
            score = source_plane_rms(
                lens,
                1.0,
                baseline["fit"]["result"].x,
                baseline["fit"]["sources"],
                context.heldout,
            )
            system_scores.append(score)
            screen_rows.append(
                {
                    "row_type": "system",
                    "candidate_id": cid,
                    "system_label": label,
                    "minimum_permittivity": epsilon,
                    "a0_m_s2": a0,
                    "gate_power": power,
                    "anisotropy_tau": tau,
                    "smoothing_r80_fraction": width,
                    "identity": is_identity(epsilon, tau),
                    "source_plane_RMS_arcsec": score,
                }
            )
            audit_rows.append(
                {
                    "candidate_id": cid,
                    "system_label": label,
                    "minimum_permittivity": epsilon,
                    "a0_m_s2": a0,
                    "gate_power": power,
                    "anisotropy_tau": tau,
                    "smoothing_r80_fraction": width,
                    **field.audit,
                }
            )
        screen_rows.append(
            {
                "row_type": "aggregate",
                "candidate_id": cid,
                "system_label": "selection",
                "minimum_permittivity": epsilon,
                "a0_m_s2": a0,
                "gate_power": power,
                "anisotropy_tau": tau,
                "smoothing_r80_fraction": width,
                "identity": is_identity(epsilon, tau),
                "source_plane_RMS_arcsec": float(
                    np.sqrt(np.mean(np.square(system_scores)))
                ),
            }
        )
        if (candidate_index + 1) % 27 == 0:
            print(
                f"P0586 screen {candidate_index + 1}/{len(candidate_grid)}",
                flush=True,
            )

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    screen = pd.DataFrame(screen_rows)
    audit = pd.DataFrame(audit_rows)
    screen.to_csv(output / protocol["outputs"]["screen_scores"], index=False)
    audit.to_csv(output / protocol["outputs"]["field_audits"], index=False)
    aggregate_screen = screen[screen.row_type.eq("aggregate")].copy()
    identity_score = float(
        aggregate_screen[aggregate_screen.identity].source_plane_RMS_arcsec.iloc[0]
    )
    selected = aggregate_screen[~aggregate_screen.identity].sort_values(
        "source_plane_RMS_arcsec"
    ).iloc[0]
    selected_parameters = {
        "candidate_id": str(selected.candidate_id),
        "minimum_permittivity": float(selected.minimum_permittivity),
        "a0_m_s2": float(selected.a0_m_s2),
        "gate_power": float(selected.gate_power),
        "anisotropy_tau": float(selected.anisotropy_tau),
        "smoothing_r80_fraction": float(selected.smoothing_r80_fraction),
        "selection_source_plane_RMS_arcsec": float(selected.source_plane_RMS_arcsec),
        "selection_improvement_vs_identity_fraction": 1.0
        - float(selected.source_plane_RMS_arcsec) / identity_score,
    }

    impact_rows = []
    coordinate_names = [
        "minimum_permittivity",
        "a0_m_s2",
        "gate_power",
        "anisotropy_tau",
        "smoothing_r80_fraction",
    ]
    for coordinate in coordinate_names:
        means = aggregate_screen.groupby(coordinate).source_plane_RMS_arcsec.mean()
        local = aggregate_screen.copy()
        for other in coordinate_names:
            if other == coordinate:
                continue
            local = local[local[other] == selected[other]]
        impact_rows.append(
            {
                "coordinate": coordinate,
                "best_main_effect_level": float(means.idxmin()),
                "main_effect_span_arcsec": float(means.max() - means.min()),
                "main_effect_relative_span": float(
                    (means.max() - means.min()) / means.mean()
                ),
                "selected_local_span_arcsec": float(
                    local.source_plane_RMS_arcsec.max()
                    - local.source_plane_RMS_arcsec.min()
                ),
                "selected_local_levels": int(len(local)),
            }
        )
    impacts = pd.DataFrame(impact_rows).sort_values(
        "selected_local_span_arcsec", ascending=False
    )
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)

    sparc = pd.read_csv(ROOT / protocol["inputs"]["SPARC_points"])
    sparc = sparc[
        sparc.model.eq("fixed_RAR")
        & sparc.scenario.eq("invariant")
        & sparc.split.eq("outer_holdout")
    ].copy()
    cross_rows = []
    gates = protocol["advance_gates"]
    solar_radius = np.geomspace(1.6 * 6.957e8, 8.43 * AU_M, 1000)
    for epsilon, a0, power in itertools.product(
        map(float, factorial["minimum_permittivity"]),
        map(float, factorial["a0_m_s2"]),
        map(float, factorial["gate_power"]),
    ):
        predicted = galaxy_prediction(sparc, epsilon, a0, power)
        point_rmse, equal_rmse = equal_system_velocity_rmse(sparc, predicted)
        fraction = solar_fraction(solar_radius, epsilon, a0, power)
        maximum = float(np.max(np.abs(fraction)))
        earth = float(np.interp(AU_M, solar_radius, fraction))
        mercury = float(mercury_precession(epsilon, a0, power))
        cross_rows.append(
            {
                "minimum_permittivity": epsilon,
                "a0_m_s2": a0,
                "gate_power": power,
                "SPARC_outer_RMSE_km_s": point_rmse,
                "SPARC_outer_equal_system_RMSE_km_s": equal_rmse,
                "SPARC_pass": point_rmse
                <= float(gates["SPARC_outer_RMSE_km_s_max"]),
                "solar_maximum_fractional_change": maximum,
                "Earth_fractional_change": earth,
                "Mercury_precession_mas_per_century": mercury,
                "Cassini_pass": maximum
                <= float(gates["Cassini_fractional_limit"]),
                "Earth_pass": abs(earth)
                <= float(gates["Earth_fractional_limit"]),
                "Mercury_pass": abs(mercury)
                <= float(gates["Mercury_mas_per_century_limit"]),
            }
        )
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(output / protocol["outputs"]["cross_domain"], index=False)
    selected_cross = cross[
        cross.minimum_permittivity.eq(selected.minimum_permittivity)
        & cross.a0_m_s2.eq(selected.a0_m_s2)
        & cross.gate_power.eq(selected.gate_power)
    ].iloc[0]
    best_galaxy = cross.sort_values("SPARC_outer_RMSE_km_s").iloc[0]
    newton_prediction = np.sqrt(
        sparc.g_bar_m_s2.to_numpy(float)
        * sparc.radius_adjusted_kpc.to_numpy(float)
        * KPC_M
    ) / 1000.0
    newton_rmse, newton_equal = equal_system_velocity_rmse(sparc, newton_prediction)

    exact_rows = []
    prediction_tables = []
    selected_fields = {}
    for context in contexts:
        label = context.system["label"]
        catalog = catalogs[label]
        field = build_baryonic_metric_correction_field(
            catalog.x_arcsec.to_numpy(float),
            catalog.y_arcsec.to_numpy(float),
            catalog.normalized_light_weight.to_numpy(float),
            total_mass_msun=masses[label],
            scale_kpc_per_arcsec=workspaces[label].scale_kpc_per_arcsec,
            minimum_permittivity=selected_parameters["minimum_permittivity"],
            a0_m_s2=selected_parameters["a0_m_s2"],
            gate_power=selected_parameters["gate_power"],
            anisotropy=selected_parameters["anisotropy_tau"],
            smoothing_r80_fraction=selected_parameters["smoothing_r80_fraction"],
            asymmetry_threshold=float(numerical["asymmetry_threshold"]),
            asymmetry_power=float(numerical["asymmetry_power"]),
            workspace=workspaces[label],
            state=states[(label, selected_parameters["smoothing_r80_fraction"])],
        )
        selected_fields[label] = field

    for context_index, context in enumerate(contexts):
        label = context.system["label"]
        for role in ("zero", "selected"):
            if role == "zero":
                fitted = baseline_fits[label]
            else:
                candidate_context = replace(context, correction=selected_fields[label])
                print(f"P0586 selected exact fit {label}", flush=True)
                fitted = fit_context(
                    candidate_context,
                    1.0,
                    starts=8,
                    seed=20261100 + context_index,
                )
            for table in (fitted["training_predictions"], fitted["heldout_predictions"]):
                copy = table.copy()
                copy.insert(3, "role", role)
                copy.insert(4, "candidate_id", selected_parameters["candidate_id"] if role == "selected" else "identity")
                prediction_tables.append(copy)
            exact_rows.append(
                {
                    "row_type": "system",
                    "role": role,
                    "system_label": label,
                    "subset": "selection" if label in selection_labels else "validation",
                    "candidate_id": selected_parameters["candidate_id"] if role == "selected" else "identity",
                    "training_exact_RMS_arcsec": fitted["training"]["exact_radial_RMS_arcsec"],
                    "heldout_exact_RMS_arcsec": fitted["heldout"]["exact_radial_RMS_arcsec"],
                    "all_training_roots": fitted["training"]["all_roots_converged"],
                    "all_heldout_roots": fitted["heldout"]["all_roots_converged"],
                }
            )
    exact = pd.DataFrame(exact_rows)
    for subset, labels in (
        ("selection", selection_labels),
        ("validation", validation_labels),
        ("all_four", selection_labels | validation_labels),
    ):
        for role in ("zero", "selected"):
            aggregate = aggregate_exact(exact, labels, role)
            exact = pd.concat(
                [
                    exact,
                    pd.DataFrame(
                        [
                            {
                                "row_type": "aggregate",
                                "role": role,
                                "system_label": subset,
                                "subset": subset,
                                "candidate_id": selected_parameters["candidate_id"] if role == "selected" else "identity",
                                **aggregate,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    exact.to_csv(output / protocol["outputs"]["exact_scores"], index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )
    validation_zero = exact[
        exact.row_type.eq("aggregate")
        & exact.system_label.eq("validation")
        & exact.role.eq("zero")
    ].iloc[0]
    validation_selected = exact[
        exact.row_type.eq("aggregate")
        & exact.system_label.eq("validation")
        & exact.role.eq("selected")
    ].iloc[0]
    improvement = (
        1.0
        - float(validation_selected.heldout_exact_RMS_arcsec)
        / float(validation_zero.heldout_exact_RMS_arcsec)
    )
    metric_report = json.loads(
        (ROOT / protocol["inputs"]["metric_slip_report"]).read_text(encoding="utf-8")
    )
    compact = float(
        metric_report["comparators"]["compact_halo_validation"][
            "equal_system_radial_RMS_arcsec"
        ]
    )
    compact_ratio = float(validation_selected.heldout_exact_RMS_arcsec) / compact
    max_curl = float(audit.normalized_curl_RMS.max())
    min_eigenvalue = float(audit.metric_minimum_eigenvalue.min())
    exact_roots = bool(
        validation_selected.all_training_roots
        and validation_selected.all_heldout_roots
    )
    report = {
        "report_version": "P0586-CONTINUOUS-BARYONIC-METRIC-RESULTS-0.1.0",
        "status": "complete_continuous_baryonic_metric_test",
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": sha256(protocol_path),
        },
        "coverage": {
            "clusters": len(contexts),
            "selection_clusters": len(selection_labels),
            "validation_clusters": len(validation_labels),
            "screen_candidates": len(aggregate_screen),
            "screen_system_fields": len(audit),
            "SPARC_galaxies": int(sparc.galaxy.nunique()),
            "SPARC_points": len(sparc),
            "exact_fits": int(len(exact[exact.row_type.eq("system")])),
        },
        "selected": selected_parameters,
        "screen": {
            "identity_source_plane_RMS_arcsec": identity_score,
            "selected_source_plane_RMS_arcsec": float(selected.source_plane_RMS_arcsec),
            "improvement_fraction": selected_parameters[
                "selection_improvement_vs_identity_fraction"
            ],
        },
        "exact_validation": {
            "zero_heldout_RMS_arcsec": float(validation_zero.heldout_exact_RMS_arcsec),
            "selected_heldout_RMS_arcsec": float(
                validation_selected.heldout_exact_RMS_arcsec
            ),
            "improvement_fraction": improvement,
            "selected_all_roots": exact_roots,
            "compact_halo_RMS_arcsec": compact,
            "selected_to_compact_ratio": compact_ratio,
        },
        "parameter_impacts": json_safe(impacts.to_dict(orient="records")),
        "cross_domain": {
            "selected": json_safe(selected_cross.to_dict()),
            "best_galaxy_grid_point": json_safe(best_galaxy.to_dict()),
            "Newtonian_outer_RMSE_km_s": newton_rmse,
            "Newtonian_outer_equal_system_RMSE_km_s": newton_equal,
            "fixed_RAR_outer_RMSE_km_s": protocol["cross_domain"]["comparators"][
                "fixed_RAR_outer_RMSE_km_s"
            ],
        },
        "numerical": {
            "maximum_normalized_curl_RMS": max_curl,
            "minimum_metric_eigenvalue": min_eigenvalue,
            "selected_cluster_field_audits": json_safe(
                {
                    label: selected_fields[label].audit for label in selected_fields
                }
            ),
        },
        "gates": {
            "validation_all_roots": exact_roots,
            "validation_improvement_pass": bool(
                exact_roots
                and improvement
                >= float(gates["validation_improvement_vs_zero_fraction_min"])
            ),
            "compact_halo_ratio_pass": bool(
                exact_roots
                and compact_ratio
                <= float(gates["validation_to_compact_halo_RMS_ratio_max"])
            ),
            "SPARC_pass": bool(selected_cross.SPARC_pass),
            "Cassini_pass": bool(selected_cross.Cassini_pass),
            "Earth_pass": bool(selected_cross.Earth_pass),
            "Mercury_pass": bool(selected_cross.Mercury_pass),
            "curl_pass": bool(max_curl <= float(gates["maximum_normalized_curl_RMS"])),
            "positive_metric_pass": bool(
                min_eigenvalue >= float(gates["minimum_metric_eigenvalue"])
            ),
            "per_cluster_gravity_parameters": 0,
            "formula_promoted": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0586 continuous baryonic metric",
        "",
        f"Selected `{selected_parameters['candidate_id']}` on the two selection clusters.",
        f"Fixed-geometry selection RMS: **{selected.source_plane_RMS_arcsec:.4f} arcsec** versus **{identity_score:.4f}** for the identity metric.",
        f"Validation exact RMS: **{validation_selected.heldout_exact_RMS_arcsec:.4f} arcsec** versus **{validation_zero.heldout_exact_RMS_arcsec:.4f}** at zero; change **{100*improvement:.3f}%**.",
        f"Selected spherical SPARC score: **{selected_cross.SPARC_outer_RMSE_km_s:.3f} km/s**; fixed RAR: **{protocol['cross_domain']['comparators']['fixed_RAR_outer_RMSE_km_s']:.3f} km/s**.",
        f"Largest selected-neighborhood parameter: **{impacts.iloc[0].coordinate}**.",
        f"All validation roots: **{exact_roots}**; compact-halo ratio: **{compact_ratio:.3f}**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    top = aggregate_screen[~aggregate_screen.identity].nsmallest(
        12, "source_plane_RMS_arcsec"
    )
    axes[0].barh(top.candidate_id, top.source_plane_RMS_arcsec)
    axes[0].axvline(identity_score, color="black", ls="--", label="identity")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("selection source-plane RMS (arcsec)")
    axes[0].tick_params(axis="y", labelsize=6)
    axes[0].legend()
    system_exact = exact[exact.row_type.eq("system")].pivot(
        index="system_label", columns="role", values="heldout_exact_RMS_arcsec"
    )
    position = np.arange(len(system_exact))
    axes[1].bar(position - 0.18, system_exact.zero, 0.36, label="zero")
    axes[1].bar(position + 0.18, system_exact.selected, 0.36, label="metric")
    axes[1].set_xticks(position, system_exact.index, rotation=30, ha="right")
    axes[1].set_ylabel("heldout exact RMS (arcsec)")
    axes[1].legend()
    axes[2].barh(impacts.coordinate, impacts.selected_local_span_arcsec)
    axes[2].set_xlabel("selected-neighborhood span (arcsec)")
    fig.suptitle("P0586 continuous baryonic field metric")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["selected"], indent=2), flush=True)
    print(json.dumps(report["exact_validation"], indent=2), flush=True)
    print(json.dumps(report["cross_domain"], indent=2), flush=True)
    print(json.dumps(report["gates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
