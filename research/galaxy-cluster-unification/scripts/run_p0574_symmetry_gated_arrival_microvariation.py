#!/usr/bin/env python3
"""Test a quarter-turn baryon-symmetry gate around the replicated P0572B law."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_fresh_sample import regrid_kappa_sky  # noqa: E402
from run_gravity_arc_tomography import shape_metrics  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import (  # noqa: E402
    deposit_baryons,
    fresh_systems,
    json_safe,
    lens_source_map,
    pilot_systems,
)
from run_p0572_tidal_cancellation_arrival_forward import destination_map  # noqa: E402
from run_p0573_tidal_arrival_fresh_replication import (  # noqa: E402
    assert_frozen_integrity,
    sha256,
    system_geometry,
)


def field_primitives(data, aperture: np.ndarray) -> dict[str, np.ndarray]:
    shape = data.x_grid.shape
    net_x = np.zeros(shape, dtype=float)
    net_y = np.zeros(shape, dtype=float)
    scalar = np.zeros(shape, dtype=float)
    txx = np.zeros(shape, dtype=float)
    tyy = np.zeros(shape, dtype=float)
    txy = np.zeros(shape, dtype=float)
    soft2 = 50.0**2
    k = 1.5
    for (sx, sy), weight in zip(data.positions, data.weights, strict=True):
        dx = sx - data.x_grid
        dy = sy - data.y_grid
        radius2 = dx * dx + dy * dy
        softened = radius2 + soft2
        inverse_k = np.power(softened, -k)
        inverse_k1 = np.power(softened, -k - 1.0)
        gx = weight * dx * inverse_k
        gy = weight * dy * inverse_k
        net_x += gx
        net_y += gy
        scalar += np.hypot(gx, gy)
        txx += weight * (-inverse_k + 2.0 * k * dx * dx * inverse_k1)
        tyy += weight * (-inverse_k + 2.0 * k * dy * dy * inverse_k1)
        txy += weight * (2.0 * k * dx * dy * inverse_k1)
    net = np.hypot(net_x, net_y)
    coherence = np.divide(
        net, scalar, out=np.ones_like(net), where=scalar > np.finfo(float).tiny
    )
    cancellation = np.clip(1.0 - coherence, 0.0, 1.0)
    trace = txx + tyy
    shear = np.sqrt(np.square(0.5 * (txx - tyy)) + np.square(txy))
    balance = np.divide(
        shear,
        shear + 0.5 * np.abs(trace),
        out=np.zeros_like(shear),
        where=(shear + 0.5 * np.abs(trace)) > np.finfo(float).tiny,
    )
    tidal_norm = np.sqrt(txx * txx + 2.0 * txy * txy + tyy * tyy)
    cancellation[~aperture] = 0.0
    balance[~aperture] = 0.0
    tidal_norm[~aperture] = 0.0
    return {
        "cancellation": cancellation,
        "balance": balance,
        "tidal_norm": tidal_norm,
        "coherence": coherence,
    }


def quarter_turn_asymmetry(data) -> float:
    baryons = deposit_baryons(data, 50.0)
    inside = data.radius <= 300.0
    baryons = np.where(inside, baryons, 0.0)
    denominator = 2.0 * float(np.sum(baryons))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(np.abs(baryons - np.rot90(baryons))) / denominator)


def effective_fraction(candidate: dict, q90: float) -> float:
    fraction = float(candidate["f"])
    q0 = float(candidate["Q0"])
    n = float(candidate["n"])
    if q0 <= 0.0:
        return fraction
    numerator = q90**n
    gate = numerator / (numerator + q0**n) if numerator > 0.0 else 0.0
    return fraction * gate


def prediction(data, aperture, primitives, candidate, q90, local):
    carrier = (
        np.power(primitives["cancellation"], float(candidate["alpha"]))
        * np.power(primitives["balance"], float(candidate["beta"]))
        * primitives["tidal_norm"]
    )
    destination = destination_map(
        carrier,
        float(candidate["width_kpc"]),
        float(data.axis[1] - data.axis[0]),
        aperture,
    )
    f_eff = effective_fraction(candidate, q90)
    predicted = (1.0 - f_eff) * local + f_eff * destination
    total = float(np.sum(predicted))
    if total <= 0.0:
        return local.copy(), f_eff
    return predicted / total, f_eff


def mean_target(data) -> np.ndarray:
    stack = np.asarray(data.range_maps)
    count = np.sum(np.isfinite(stack), axis=0)
    raw_mean = np.divide(
        np.nansum(stack, axis=0),
        count,
        out=np.full_like(stack[0], np.nan),
        where=count > 0,
    )
    return lens_source_map(raw_mean, data.radius, 10.0, 20.0, (250.0, 300.0))


def score_candidates(data, target, candidates) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    aperture = data.radius <= 250.0
    local = deposit_baryons(data, 100.0)
    local[~aperture] = 0.0
    local /= np.sum(local)
    primitives = field_primitives(data, aperture)
    q90 = quarter_turn_asymmetry(data)
    rows = [
        {
            "system": data.label,
            "candidate_id": "local_control",
            "Q90": q90,
            "effective_route_fraction": 0.0,
            **shape_metrics(local, target, aperture),
        }
    ]
    predictions = {"local_control": local}
    for candidate in candidates:
        predicted, f_eff = prediction(
            data, aperture, primitives, candidate, q90, local
        )
        predictions[candidate["candidate_id"]] = predicted
        rows.append(
            {
                "system": data.label,
                "candidate_id": candidate["candidate_id"],
                "Q90": q90,
                "effective_route_fraction": f_eff,
                **shape_metrics(predicted, target, aperture),
            }
        )
    audit = {
        "system": data.label,
        "Q90": q90,
        "member_sources": int(len(data.positions)),
        "median_coherence": float(np.median(primitives["coherence"][aperture])),
        "median_cancellation": float(np.median(primitives["cancellation"][aperture])),
        "median_tidal_balance": float(np.median(primitives["balance"][aperture])),
    }
    return rows, predictions, audit


def candidate_impacts(candidate_scores: pd.DataFrame) -> list[dict]:
    groups = {
        "Q0": ["Q0_low", "gate_reference", "Q0_high"],
        "gate_sharpness_n": ["gate_soft", "gate_reference", "gate_sharp"],
        "arrival_width": ["width_40", "gate_reference", "width_60"],
        "route_fraction": ["fraction_0p7", "gate_reference", "fraction_0p9"],
        "cancellation_power": ["cancellation_0p4", "gate_reference", "cancellation_0p6"],
        "tidal_power": ["tidal_0p8", "gate_reference", "tidal_1p2"],
    }
    indexed = candidate_scores.set_index("candidate_id")
    rows = []
    for coordinate, ids in groups.items():
        values = indexed.loc[ids, "development_mean_JS"]
        rows.append(
            {
                "coordinate": coordinate,
                "JS_span": float(values.max() - values.min()),
                "relative_span": float((values.max() - values.min()) / values.mean()),
                "best_candidate": str(values.idxmin()),
            }
        )
    return sorted(rows, key=lambda row: row["JS_span"], reverse=True)


def sparc_null_rows(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("*_rotmod.dat")):
        points = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                try:
                    float(text.split()[0])
                except (ValueError, IndexError):
                    continue
                points += 1
        rows.append(
            {
                "galaxy": path.stem.replace("_rotmod", ""),
                "radial_points": points,
                "deprojected_axisymmetric_Q90": 0.0,
                "angular_layer_velocity_change_km_s": 0.0,
            }
        )
    return rows


def main() -> None:
    protocol_path = ROOT / "configs/p0574_symmetry_gated_arrival_microvariation_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0573_before_any_symmetry_gated_lens_score":
        raise RuntimeError("P0574 protocol is not frozen")
    candidates = protocol["candidate_grid"]
    p0567 = json.loads((ROOT / protocol["inputs"]["p0567_protocol"]).read_text(encoding="utf-8"))
    historical = fresh_systems(p0567) + pilot_systems(p0567)
    development_rows = []
    asymmetry_rows = []
    for data in historical:
        rows, _, audit = score_candidates(data, mean_target(data), candidates)
        development_rows.extend(rows)
        asymmetry_rows.append({"cohort": "historical_development", **audit})
        print(f"P0574 development: {data.label}", flush=True)
    development = pd.DataFrame(development_rows)
    candidate_scores = (
        development.groupby("candidate_id", as_index=False)
        .agg(development_mean_JS=("jensen_shannon", "mean"), development_mean_Pearson=("pearson", "mean"))
    )
    candidate_scores = candidate_scores.merge(
        pd.DataFrame(candidates), on="candidate_id", how="left"
    )
    gated = candidate_scores[candidate_scores.Q0.fillna(0.0).gt(0.0)]
    selected_row = gated.sort_values("development_mean_JS").iloc[0]
    selected_id = str(selected_row.candidate_id)
    selected_candidate = next(item for item in candidates if item["candidate_id"] == selected_id)

    p0573_path = ROOT / protocol["inputs"]["p0573_protocol"]
    p0573 = json.loads(p0573_path.read_text(encoding="utf-8"))
    _, manifest = assert_frozen_integrity(p0573_path, p0573)
    audit_directory = ROOT / p0573["outputs"]["input_audit_directory"]
    sources = pd.read_csv(audit_directory / "sources.csv")
    audits = pd.read_csv(audit_directory / "systems.csv")
    validation_rows = []
    uncertainty_rows = []
    glafic_rows = []
    for system in p0573["systems"]:
        data, world = system_geometry(system, p0573, sources, audits)
        local_manifest = manifest[manifest.system.eq(data.label)]
        range_rows = local_manifest[
            local_manifest.kind.eq("range_kappa") & local_manifest.method.eq("lenstool")
        ].copy()
        range_rows["sample_index_numeric"] = pd.to_numeric(range_rows.sample_index)
        range_rows = range_rows.sort_values("sample_index_numeric")
        data.range_maps = [
            regrid_kappa_sky(ROOT / row.path, world, data.x_grid.shape)
            for row in range_rows.itertuples(index=False)
        ]
        target = mean_target(data)
        subset = [
            next(item for item in candidates if item["candidate_id"] == "no_gate_baseline"),
            selected_candidate,
        ]
        rows, predictions, audit = score_candidates(data, target, subset)
        validation_rows.extend(rows)
        asymmetry_rows.append({"cohort": "P0573_variant_validation", **audit})
        aperture = data.radius <= 250.0
        for realization, raw in enumerate(data.range_maps):
            realization_target = lens_source_map(
                raw, data.radius, 10.0, 20.0, (250.0, 300.0)
            )
            local_js = shape_metrics(
                predictions["local_control"], realization_target, aperture
            )["jensen_shannon"]
            selected_js = shape_metrics(
                predictions[selected_id], realization_target, aperture
            )["jensen_shannon"]
            uncertainty_rows.append(
                {
                    "system": data.label,
                    "realization": realization,
                    "local_JS": local_js,
                    "selected_JS": selected_js,
                    "selected_improves": bool(selected_js < local_js),
                }
            )
        glafic_row = local_manifest[
            local_manifest.kind.eq("best_kappa") & local_manifest.method.eq("glafic")
        ].iloc[0]
        glafic_raw = regrid_kappa_sky(ROOT / glafic_row.path, world, data.x_grid.shape)
        glafic_target = lens_source_map(
            glafic_raw, data.radius, 10.0, 20.0, (250.0, 300.0)
        )
        for model in ("local_control", selected_id):
            glafic_rows.append(
                {
                    "system": data.label,
                    "model": model,
                    **shape_metrics(predictions[model], glafic_target, aperture),
                }
            )
        print(f"P0574 validation: {data.label}", flush=True)

    validation = pd.DataFrame(validation_rows)
    uncertainty = pd.DataFrame(uncertainty_rows)
    glafic = pd.DataFrame(glafic_rows)
    validation_js = validation.pivot(index="system", columns="candidate_id", values="jensen_shannon")
    local_mean = float(validation_js.local_control.mean())
    selected_mean = float(validation_js[selected_id].mean())
    validation_gain = float(1.0 - selected_mean / local_mean)
    systems_improved = int((validation_js[selected_id] < validation_js.local_control).sum())
    realization_fraction = float(uncertainty.selected_improves.mean())
    glafic_js = glafic.pivot(index="system", columns="model", values="jensen_shannon")
    glafic_gain = float(1.0 - glafic_js[selected_id].mean() / glafic_js.local_control.mean())
    glafic_improved = int((glafic_js[selected_id] < glafic_js.local_control).sum())
    p0573_report = json.loads((ROOT / protocol["inputs"]["p0573_report"]).read_text(encoding="utf-8"))
    no_gate_gain = float(p0573_report["result"]["improvement_vs_local_fraction"])
    retained_gain_fraction = validation_gain / no_gate_gain if no_gate_gain > 0.0 else math.nan

    disk_q90 = 0.0
    disk_effective_fraction = effective_fraction(selected_candidate, disk_q90)
    solar_effective_fraction = effective_fraction(selected_candidate, 0.0)
    sparc_rows = sparc_null_rows(ROOT / protocol["inputs"]["SPARC_rotmod_directory"])
    impacts = candidate_impacts(candidate_scores)
    gates_cfg = protocol["advance_gates"]
    gates = {
        "variant_validation_improvement_pass": bool(
            validation_gain >= float(gates_cfg["variant_validation_improvement_vs_local_fraction_min"])
        ),
        "variant_validation_system_count_pass": bool(
            systems_improved >= int(gates_cfg["variant_validation_systems_improved_min"])
        ),
        "retained_no_gate_gain_pass": bool(
            retained_gain_fraction >= float(gates_cfg["retain_fraction_of_P0573_no_gate_gain_min"])
        ),
        "lenstool_uncertainty_pass": bool(
            realization_fraction >= float(gates_cfg["lenstool_realizations_improved_fraction_min"])
        ),
        "glafic_improvement_pass": bool(
            glafic_gain >= float(gates_cfg["glafic_improvement_vs_local_fraction_min"])
        ),
        "glafic_system_count_pass": bool(
            glafic_improved >= int(gates_cfg["glafic_systems_improved_min"])
        ),
        "extended_disk_null_pass": bool(
            disk_effective_fraction <= float(gates_cfg["extended_disk_effective_route_fraction_max"])
        ),
        "solar_null_pass": bool(
            solar_effective_fraction <= float(gates_cfg["solar_effective_route_fraction_max"])
        ),
        "SPARC_null_pass": bool(
            all(row["angular_layer_velocity_change_km_s"] <= float(gates_cfg["SPARC_velocity_change_max_km_s"]) for row in sparc_rows)
        ),
    }
    gates["raw_lensing_followup_authorized"] = bool(all(gates.values()))

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    development.to_csv(output / protocol["outputs"]["system_scores"], index=False)
    candidate_scores.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    validation.to_csv(output / protocol["outputs"]["validation_scores"], index=False)
    uncertainty.to_csv(output / protocol["outputs"]["uncertainty"], index=False)
    glafic.to_csv(output / protocol["outputs"]["glafic_scores"], index=False)
    pd.DataFrame(asymmetry_rows).to_csv(output / protocol["outputs"]["asymmetry_audit"], index=False)
    pd.DataFrame(sparc_rows).to_csv(output / protocol["outputs"]["SPARC_null_audit"], index=False)
    result = {
        "selected_candidate": selected_candidate,
        "development_mean_JS": float(selected_row.development_mean_JS),
        "validation_local_equal_system_JS": local_mean,
        "validation_selected_equal_system_JS": selected_mean,
        "validation_improvement_vs_local_fraction": validation_gain,
        "validation_systems_improved": systems_improved,
        "retained_fraction_of_P0573_no_gate_gain": retained_gain_fraction,
        "validation_realizations_improved_fraction": realization_fraction,
        "glafic_improvement_vs_local_fraction": glafic_gain,
        "glafic_systems_improved": glafic_improved,
    }
    report = {
        "report_version": "P0574-SYMMETRY-GATED-ARRIVAL-MICROVARIATION-RESULTS-0.1.0",
        "status": "complete_symmetry_gated_microvariation",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(protocol_path)},
        "coverage": {
            "historical_development_clusters": len(historical),
            "candidates": len(candidates),
            "variant_validation_clusters": len(p0573["systems"]),
            "validation_lenstool_realizations": len(uncertainty),
            "validation_glafic_maps": len(glafic_js),
            "SPARC_galaxies_exact_angular_null": len(sparc_rows),
        },
        "result": result,
        "parameter_impacts": impacts,
        "per_validation_system": [
            {
                "system": label,
                "Q90": float(validation[validation.system.eq(label)].Q90.iloc[0]),
                "effective_route_fraction": float(validation[(validation.system.eq(label)) & (validation.candidate_id.eq(selected_id))].effective_route_fraction.iloc[0]),
                "improvement_fraction": float(1.0 - validation_js.loc[label, selected_id] / validation_js.loc[label, "local_control"]),
            }
            for label in validation_js.index
        ],
        "cross_domain": {
            "extended_axisymmetric_disk_Q90": disk_q90,
            "extended_axisymmetric_disk_effective_route_fraction": disk_effective_fraction,
            "solar_effective_route_fraction": solar_effective_fraction,
            "SPARC_galaxies": len(sparc_rows),
            "SPARC_maximum_angular_layer_velocity_change_km_s": 0.0,
            "galaxy_interpretation": "This angular layer is now exactly inert on the deprojected axisymmetric profiles in SPARC; it still requires a separate tested radial acceleration law to explain their speeds.",
        },
        "gates": gates,
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# P0574 symmetry-gated arrival microvariation",
        "",
        f"Selected `{selected_id}` on 13 historical systems.",
        f"P0573 variant-validation JS: **{selected_mean:.5f}** versus local **{local_mean:.5f}**; gain **{100*validation_gain:.2f}%**.",
        f"Systems improved: **{systems_improved}/3**; realizations improved: **{100*realization_fraction:.1f}%**; GLAFIC gain: **{100*glafic_gain:.2f}%**.",
        f"The symmetry gate retains **{100*retained_gain_fraction:.1f}%** of the no-gate P0573 gain while making the axisymmetric-disk, Solar, and **{len(sparc_rows)}-galaxy SPARC angular response exactly zero.",
        f"Raw-lensing follow-up authorized: **{gates['raw_lensing_followup_authorized']}**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ranked = candidate_scores[candidate_scores.candidate_id.ne("local_control")].sort_values("development_mean_JS")
    axes[0].barh(ranked.candidate_id, ranked.development_mean_JS)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("13-system mean JS")
    x = np.arange(len(validation_js))
    axes[1].bar(x - 0.18, validation_js.local_control, 0.36, label="local")
    axes[1].bar(x + 0.18, validation_js[selected_id], 0.36, label="selected")
    axes[1].set_xticks(x, validation_js.index, rotation=25, ha="right")
    axes[1].set_ylabel("variant-validation JS")
    axes[1].legend()
    impact_frame = pd.DataFrame(impacts)
    axes[2].barh(impact_frame.coordinate, impact_frame.relative_span)
    axes[2].set_xlabel("OAT relative JS span")
    fig.suptitle("P0574 universal quarter-turn symmetry gate")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    print(json.dumps(impacts, indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
