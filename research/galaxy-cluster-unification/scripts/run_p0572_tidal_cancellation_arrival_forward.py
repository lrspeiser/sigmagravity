#!/usr/bin/env python3
"""Forward-test the P0571B tidal-cancellation activation on RELICS maps."""

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
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_tomography import shape_metrics  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import deposit_baryons, json_safe, lens_source_map  # noqa: E402
from run_p0568_baryon_only_tensor_forward import build_contexts  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def activation_and_carriers(data, score_aperture):
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
    coherence = np.divide(net, scalar, out=np.ones_like(net), where=scalar > np.finfo(float).tiny)
    cancellation = np.clip(1.0 - coherence, 0.0, 1.0)
    trace = txx + tyy
    shear = np.sqrt(np.square(0.5 * (txx - tyy)) + np.square(txy))
    tidal_norm = np.sqrt(txx * txx + 2.0 * txy * txy + tyy * tyy)
    balance = np.divide(
        shear,
        shear + 0.5 * np.abs(trace),
        out=np.zeros_like(shear),
        where=(shear + 0.5 * np.abs(trace)) > np.finfo(float).tiny,
    )
    activation = np.sqrt(cancellation) * balance
    activation[~score_aperture] = 0.0
    return activation, {
        "activation_only": activation,
        "field_weighted": activation * scalar,
        "tidal_weighted": activation * tidal_norm,
    }, {
        "activation_RMS": float(np.sqrt(np.mean(activation[score_aperture] ** 2))),
        "activation_maximum": float(np.max(activation[score_aperture])),
        "median_coherence": float(np.median(coherence[score_aperture])),
        "median_tidal_balance": float(np.median(balance[score_aperture])),
    }


def destination_map(carrier, width_kpc, spacing_kpc, aperture):
    smoothed = gaussian_filter(carrier, float(width_kpc) / float(spacing_kpc), mode="constant")
    smoothed = np.maximum(smoothed, 0.0)
    smoothed[~aperture] = 0.0
    total = float(np.sum(smoothed))
    if total <= 0.0:
        return np.zeros_like(smoothed)
    return smoothed / total


def axisymmetric_null(protocol):
    size = 256
    spacing = 10.0
    axis = (np.arange(size) - (size - 1) / 2.0) * spacing
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    dummy = type("Dummy", (), {})()
    dummy.x_grid = xx
    dummy.y_grid = yy
    dummy.positions = np.asarray([[0.0, 0.0]])
    dummy.weights = np.asarray([1.0])
    aperture = np.hypot(xx, yy) <= 250.0
    activation, _, _ = activation_and_carriers(dummy, aperture)
    return float(np.sqrt(np.mean(activation[aperture] ** 2))), float(np.max(np.abs(activation[aperture])))


def main() -> None:
    protocol_path = ROOT / "configs/p0572_tidal_cancellation_arrival_forward_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_tidal_cancellation_forward_map_score":
        raise RuntimeError("P0572 protocol is not frozen")
    p0568 = json.loads((ROOT / protocol["inputs"]["p0568_protocol"]).read_text(encoding="utf-8"))
    p0567 = json.loads((ROOT / protocol["inputs"]["p0567_protocol"]).read_text(encoding="utf-8"))
    contexts = build_contexts(p0568, p0567)
    development = set(protocol["validation"]["development_systems"])
    holdout = set(protocol["validation"]["holdout_systems"])
    spacing = 10.0
    prediction_cache = {}
    rows = []
    audit_rows = []
    for context in contexts:
        label = context.data.label
        cohort = "development" if label in development else "holdout"
        local = deposit_baryons(context.data, float(protocol["factorial"]["locked_local_width_kpc"]))
        local[~context.aperture] = 0.0
        local /= np.sum(local)
        prediction_cache[(label, "local_control")] = local
        rows.append({"system": label, "cohort": cohort, "candidate_id": "local_control", "carrier": "local_control", "arrival_smoothing_kpc": 0.0, "route_fraction_f": 0.0, **shape_metrics(local, context.target, context.aperture)})
        activation, carriers, audit = activation_and_carriers(context.data, context.aperture)
        audit_rows.append({"system": label, **audit})
        for carrier_name in protocol["factorial"]["carrier"]:
            for width in map(float, protocol["factorial"]["arrival_smoothing_kpc"]):
                destination = destination_map(carriers[carrier_name], width, spacing, context.aperture)
                for fraction in map(float, protocol["factorial"]["route_fraction_f"]):
                    candidate_id = f"{carrier_name}__w{width:g}__f{fraction:g}"
                    predicted = (1.0 - fraction) * local + fraction * destination
                    predicted /= np.sum(predicted)
                    prediction_cache[(label, candidate_id)] = predicted
                    rows.append({"system": label, "cohort": cohort, "candidate_id": candidate_id, "carrier": carrier_name, "arrival_smoothing_kpc": width, "route_fraction_f": fraction, **shape_metrics(predicted, context.target, context.aperture)})
        print(f"P0572 candidates: {label}", flush=True)
    scores = pd.DataFrame(rows)
    candidate_scores = (
        scores.groupby(["candidate_id", "carrier", "arrival_smoothing_kpc", "route_fraction_f"], dropna=False)
        .apply(
            lambda block: pd.Series(
                {
                    "development_mean_JS": block.loc[block.cohort.eq("development"), "jensen_shannon"].mean(),
                    "holdout_mean_JS": block.loc[block.cohort.eq("holdout"), "jensen_shannon"].mean(),
                    "development_mean_Pearson": block.loc[block.cohort.eq("development"), "pearson"].mean(),
                    "holdout_mean_Pearson": block.loc[block.cohort.eq("holdout"), "pearson"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    selected = candidate_scores[candidate_scores.carrier.ne("local_control")].sort_values("development_mean_JS").iloc[0]
    local_score = candidate_scores[candidate_scores.carrier.eq("local_control")].iloc[0]
    selected_systems = scores[scores.candidate_id.eq(selected.candidate_id)].copy()
    local_systems = scores[scores.candidate_id.eq("local_control")].copy()
    paired = selected_systems.merge(local_systems[["system", "jensen_shannon"]], on="system", suffixes=("_selected", "_local"))
    holdout_paired = paired[paired.cohort.eq("holdout")]
    holdout_gain = 1.0 - float(selected.holdout_mean_JS) / float(local_score.holdout_mean_JS)
    holdout_improved = int((holdout_paired.jensen_shannon_selected < holdout_paired.jensen_shannon_local).sum())

    uncertainty_rows = []
    glafic_rows = []
    for context in contexts:
        label = context.data.label
        cohort = "development" if label in development else "holdout"
        chosen = prediction_cache[(label, selected.candidate_id)]
        local = prediction_cache[(label, "local_control")]
        for realization, raw in enumerate(context.data.range_maps):
            target = lens_source_map(raw, context.data.radius, spacing, 20.0, (250.0, 300.0))
            chosen_metric = shape_metrics(chosen, target, context.aperture)
            local_metric = shape_metrics(local, target, context.aperture)
            uncertainty_rows.append({"system": label, "cohort": cohort, "realization": realization, "selected_JS": chosen_metric["jensen_shannon"], "local_JS": local_metric["jensen_shannon"], "selected_improves": chosen_metric["jensen_shannon"] < local_metric["jensen_shannon"]})
        for model, prediction in (("selected", chosen), ("local", local)):
            glafic_rows.append({"system": label, "cohort": cohort, "model": model, **shape_metrics(prediction, context.glafic_target, context.aperture)})
    uncertainty = pd.DataFrame(uncertainty_rows)
    glafic = pd.DataFrame(glafic_rows)
    uncertainty_holdout_fraction = float(uncertainty[uncertainty.cohort.eq("holdout")].selected_improves.mean())
    glafic_holdout = glafic[glafic.cohort.eq("holdout")].groupby("model").jensen_shannon.mean()
    glafic_gain = 1.0 - float(glafic_holdout["selected"]) / float(glafic_holdout["local"])
    impact_rows = []
    nonlocal_candidates = candidate_scores[candidate_scores.carrier.ne("local_control")]
    for coordinate in ("carrier", "arrival_smoothing_kpc", "route_fraction_f"):
        means = nonlocal_candidates.groupby(coordinate).development_mean_JS.mean()
        impact_rows.append({"coordinate": coordinate, "minimum_level": str(means.idxmin()), "main_effect_span_JS": float(means.max() - means.min()), "relative_span": float((means.max() - means.min()) / means.mean())})
    impacts = pd.DataFrame(impact_rows).sort_values("main_effect_span_JS", ascending=False)
    axis_rms, axis_max = axisymmetric_null(protocol)
    gates_cfg = protocol["advance_gates"]
    gates = {
        "holdout_improvement_pass": bool(holdout_gain >= float(gates_cfg["holdout_improvement_vs_local_fraction_min"])),
        "holdout_system_count_pass": bool(holdout_improved >= int(gates_cfg["holdout_systems_improved_min"])),
        "glafic_holdout_pass": bool(glafic_gain >= float(gates_cfg["glafic_holdout_improvement_vs_local_fraction_min"])),
        "uncertainty_pass": bool(uncertainty_holdout_fraction >= float(gates_cfg["holdout_uncertainty_realizations_improved_fraction_min"])),
        "axisymmetric_null_pass": bool(axis_rms <= float(gates_cfg["axisymmetric_activation_RMS_max"])),
        "no_per_cluster_parameters": True,
    }
    gates["raw_lensing_followup_authorized"] = bool(all(gates.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / protocol["outputs"]["system_scores"], index=False)
    candidate_scores.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    uncertainty.to_csv(output / protocol["outputs"]["uncertainty"], index=False)
    glafic.to_csv(output / protocol["outputs"]["glafic_scores"], index=False)
    pd.DataFrame(audit_rows).to_csv(output / protocol["outputs"]["activation_audit"], index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    selected_dict = json_safe(selected.to_dict())
    selected_dict.update({"holdout_improvement_vs_local_fraction": holdout_gain, "holdout_systems_improved": holdout_improved, "glafic_holdout_improvement_vs_local_fraction": glafic_gain, "holdout_uncertainty_improved_fraction": uncertainty_holdout_fraction})
    report = {
        "report_version": "P0572-TIDAL-CANCELLATION-ARRIVAL-FORWARD-RESULTS-0.1.0",
        "status": "complete_tidal_cancellation_forward_map_test",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": sha256(protocol_path)},
        "coverage": {"clusters": len(contexts), "development_clusters": len(development), "holdout_clusters": len(holdout), "candidates": int(len(nonlocal_candidates)), "system_candidate_scores": len(scores), "lenstool_realization_comparisons": len(uncertainty)},
        "selected": selected_dict,
        "local_control": json_safe(local_score.to_dict()),
        "parameter_impacts": json_safe(impacts.to_dict(orient="records")),
        "numerical": {"axisymmetric_activation_RMS": axis_rms, "axisymmetric_activation_maximum": axis_max, "maximum_activation": float(pd.DataFrame(audit_rows).activation_maximum.max())},
        "cross_domain": {"SPARC_rotation_change_km_s": 0.0, "solar_fractional_change": 0.0, "Mercury_precession_change_mas_per_century": 0.0, "interpretation": "exact angular null; not an independent galaxy force law"},
        "gates": gates,
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8")
    lines = [
        "# P0572 tidal-cancellation arrival forward test",
        "",
        f"Selected `{selected.candidate_id}` on development maps.",
        f"Holdout JS: **{selected.holdout_mean_JS:.5f}** versus local **{local_score.holdout_mean_JS:.5f}**; gain **{100*holdout_gain:.2f}%**.",
        f"Holdout systems improved: **{holdout_improved}/3**; uncertainty realizations improved: **{100*uncertainty_holdout_fraction:.1f}%**.",
        f"GLAFIC holdout gain: **{100*glafic_gain:.2f}%**.",
        f"Raw-lensing follow-up authorized: **{gates['raw_lensing_followup_authorized']}**.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    holdout_plot = holdout_paired.sort_values("system")
    x = np.arange(len(holdout_plot))
    axes[0].bar(x - 0.18, holdout_plot.jensen_shannon_local, 0.36, label="local")
    axes[0].bar(x + 0.18, holdout_plot.jensen_shannon_selected, 0.36, label="activation")
    axes[0].set_xticks(x, holdout_plot.system, rotation=25, ha="right")
    axes[0].set_ylabel("Jensen-Shannon")
    axes[0].legend()
    frac_curve = nonlocal_candidates.groupby("route_fraction_f").development_mean_JS.mean()
    axes[1].plot(frac_curve.index, frac_curve.values, marker="o")
    axes[1].axvline(float(selected.route_fraction_f), color="black", ls="--")
    axes[1].set_xlabel("routed fraction f")
    axes[1].set_ylabel("mean development JS")
    axes[2].barh(impacts.coordinate, impacts.main_effect_span_JS)
    axes[2].set_xlabel("development main-effect JS span")
    fig.suptitle("P0572 forward arrival map from baryonic tidal cancellation")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["selected"], indent=2))
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
