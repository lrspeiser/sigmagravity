#!/usr/bin/env python3
"""Cross-validate an extent-adaptive conservative gravity-route kernel."""

from __future__ import annotations

import hashlib
import itertools
import json
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

from run_gravity_arc_fresh_sample import (  # noqa: E402
    build_source_context,
    prediction_protocol,
    target_from_path,
)
from run_gravity_arc_tomography import prediction_for_spec, shape_metrics  # noqa: E402
from voidscreen.adaptive_route_kernel import (  # noqa: E402
    adaptive_route_parameters,
    transformed_source_weights,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def candidate_specs(protocol: dict) -> list[dict]:
    base = {**protocol["baseline"], "landing_mode": "endpoint"}
    specs = []
    seen = set()

    def add(spec: dict, role: str, changed: str = ""):
        local = {**base, **spec, "role": role, "changed_parameter": changed}
        key = tuple(
            local[name]
            for name in (
                "feature",
                "base_fraction",
                "extent_slope",
                "base_length_kpc",
                "length_power",
                "base_width_kpc",
                "width_power",
                "gate_power",
                "source_weight_power",
                "landing_mode",
            )
        )
        if key not in seen:
            seen.add(key)
            specs.append(local)

    add({}, "baseline")
    oat = protocol["one_at_a_time"]
    for value in oat["base_fraction"]:
        add({"base_fraction": float(value)}, "one_at_a_time", "base_fraction")
    for feature, value in itertools.product(oat["extent_features"], oat["extent_slope"]):
        add(
            {"feature": feature, "extent_slope": float(value)},
            "one_at_a_time",
            f"extent_slope:{feature}",
        )
    for name in (
        "base_length_kpc",
        "length_power",
        "base_width_kpc",
        "width_power",
        "gate_power",
        "source_weight_power",
    ):
        for value in oat[name]:
            add({name: float(value)}, "one_at_a_time", name)
    for value in oat["landing_mode"]:
        add({"landing_mode": str(value)}, "one_at_a_time", "landing_mode")

    interaction = protocol["interaction_grid"]
    fixed = interaction["fixed"]
    for values in itertools.product(
        interaction["feature"],
        interaction["base_fraction"],
        interaction["extent_slope"],
        interaction["length_power"],
        interaction["base_width_kpc"],
        interaction["gate_power"],
    ):
        feature, fraction, slope, length_power, width, gate = values
        add(
            {
                **fixed,
                "feature": str(feature),
                "base_fraction": float(fraction),
                "extent_slope": float(slope),
                "length_power": float(length_power),
                "base_width_kpc": float(width),
                "gate_power": float(gate),
            },
            "interaction",
            "extent_fraction_length_width_gate",
        )
    for index, spec in enumerate(specs):
        spec["candidate_id"] = f"A{index:04d}"
    return specs


def adaptive_prediction(context, morphology: pd.Series, spec: dict, forward_protocol: dict):
    weights = transformed_source_weights(
        context.soft_weights, float(spec["source_weight_power"])
    )
    parameters = adaptive_route_parameters(
        r50_kpc=float(morphology.r50_kpc),
        concentration=float(morphology.radial_concentration_r50_over_r80),
        source_weights=weights,
        feature=str(spec["feature"]),
        base_fraction=float(spec["base_fraction"]),
        extent_slope=float(spec["extent_slope"]),
        base_length_kpc=float(spec["base_length_kpc"]),
        length_power=float(spec["length_power"]),
        base_width_kpc=float(spec["base_width_kpc"]),
        width_power=float(spec["width_power"]),
        gate_power=float(spec["gate_power"]),
    )
    local_context = replace(context, soft_weights=weights, hard_weights=weights)
    realized = {
        "family": "center_return",
        "fraction": parameters["routing_fraction"],
        "return_scale_kpc": parameters["return_scale_kpc"],
        "exponent": -0.5,
        "width_kpc": parameters["width_kpc"],
        "landing_mode": str(spec["landing_mode"]),
    }
    prediction = prediction_for_spec(local_context, realized, forward_protocol)
    return prediction, parameters


def load_contexts_and_targets(protocol: dict):
    acquisition_path = ROOT / protocol["inputs"]["acquisition_protocol"]
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    analysis_path = ROOT / protocol["inputs"]["analysis_protocol"]
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    sources = pd.read_csv(ROOT / protocol["inputs"]["sources"])
    system_audit = pd.read_csv(ROOT / protocol["inputs"]["systems"]).set_index("system")
    raw = ROOT / acquisition["acquisition"]["output_directory"]
    settings = acquisition["spatial_preprocessing"]
    contexts, targets = {}, {}
    for index, system in enumerate(acquisition["systems"]):
        label = system["label"]
        context, world = build_source_context(system, system_audit.loc[label], sources, settings)
        models = {item["method"]: item for item in system["models"]}
        lenstool_dir = raw / "models" / system["slug"] / "lenstool"
        paths = sorted((lenstool_dir / "range").glob("*_kappa.fits"))
        target_sum = np.zeros_like(context.x_grid)
        for path in paths:
            target_sum += target_from_path(path, world, context, settings)
        lenstool = target_sum / len(paths)
        glafic = models["glafic"]
        glafic_path = raw / "models" / system["slug"] / "glafic" / glafic["best_filename"]
        contexts[label] = context
        targets[(label, "lenstool_ensemble_mean")] = lenstool
        targets[(label, "glafic_best")] = target_from_path(
            glafic_path, world, context, settings
        )
        print(f"Loaded target maps {index + 1}/{len(acquisition['systems'])}: {label}", flush=True)
    return acquisition, analysis, contexts, targets


def one_at_a_time_impacts(scores: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    baseline_id = str(specs[specs.role.eq("baseline")].iloc[0].candidate_id)
    baseline = scores[scores.candidate_id.eq(baseline_id)][
        ["system", "target_kind", "jensen_shannon"]
    ].rename(columns={"jensen_shannon": "baseline_JS"})
    rows = []
    for spec in specs[specs.role.eq("one_at_a_time")].itertuples(index=False):
        block = scores[scores.candidate_id.eq(spec.candidate_id)].merge(
            baseline, on=["system", "target_kind"], validate="one_to_one"
        )
        block["delta_JS"] = block.jensen_shannon - block.baseline_JS
        row = {
            "candidate_id": spec.candidate_id,
            "changed_parameter": spec.changed_parameter,
        }
        for name in (
            "feature",
            "base_fraction",
            "extent_slope",
            "base_length_kpc",
            "length_power",
            "base_width_kpc",
            "width_power",
            "gate_power",
            "source_weight_power",
            "landing_mode",
        ):
            row[name] = getattr(spec, name)
        for target, prefix in (
            ("lenstool_ensemble_mean", "lenstool"),
            ("glafic_best", "glafic"),
        ):
            values = block[block.target_kind.eq(target)].delta_JS.to_numpy(float)
            row[f"{prefix}_median_delta_JS"] = float(np.median(values))
            row[f"{prefix}_win_fraction"] = float(np.mean(values < 0.0))
        row["cross_method_same_direction"] = bool(
            np.sign(row["lenstool_median_delta_JS"])
            == np.sign(row["glafic_median_delta_JS"])
        )
        row["absolute_lenstool_impact"] = abs(row["lenstool_median_delta_JS"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("absolute_lenstool_impact", ascending=False)


def interaction_impacts(scores: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    interaction = specs[specs.role.eq("interaction")]
    merged = scores.merge(interaction, on="candidate_id", validate="many_to_one")
    rows = []
    for parameter in (
        "feature",
        "base_fraction",
        "extent_slope",
        "length_power",
        "base_width_kpc",
        "gate_power",
    ):
        for target, prefix in (
            ("lenstool_ensemble_mean", "lenstool"),
            ("glafic_best", "glafic"),
        ):
            levels = merged[merged.target_kind.eq(target)].groupby(parameter).jensen_shannon.median()
            rows.append(
                {
                    "parameter": parameter,
                    "method": prefix,
                    "marginal_median_JS_span": float(levels.max() - levels.min()),
                    "best_marginal_level": str(levels.idxmin()),
                }
            )
    return pd.DataFrame(rows).sort_values("marginal_median_JS_span", ascending=False)


def loocv_predictions(scores: pd.DataFrame, comparators: pd.DataFrame, systems: list[str]):
    primary = scores[scores.target_kind.eq("lenstool_ensemble_mean")]
    rows = []
    for holdout in systems:
        training = primary[~primary.system.eq(holdout)]
        ranking = training.groupby("candidate_id").jensen_shannon.mean().sort_values()
        selected = str(ranking.index[0])
        for target in ("lenstool_ensemble_mean", "glafic_best"):
            observed = scores[
                scores.system.eq(holdout)
                & scores.target_kind.eq(target)
                & scores.candidate_id.eq(selected)
            ].iloc[0]
            row = {
                "holdout_system": holdout,
                "target_kind": target,
                "selected_candidate_id": selected,
                "training_equal_cluster_mean_JS": float(ranking.iloc[0]),
                "heldout_JS": float(observed.jensen_shannon),
                "heldout_Pearson": float(observed.pearson),
            }
            for comparator in ("C0351", "W060", "LOCAL75", "CENTRAL100"):
                value = float(
                    comparators[
                        comparators.system.eq(holdout)
                        & comparators.target_kind.eq(target)
                        & comparators.comparator.eq(comparator)
                    ].jensen_shannon.iloc[0]
                )
                row[f"{comparator}_JS"] = value
                row[f"improvement_over_{comparator}"] = 1.0 - row["heldout_JS"] / value
            rows.append(row)
    return pd.DataFrame(rows)


def make_figure(loocv, impacts, interaction, realized, selected_id, output):
    systems = loocv[loocv.target_kind.eq("lenstool_ensemble_mean")].holdout_system.tolist()
    x = np.arange(len(systems))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax = axes[0, 0]
    for offset, (target, label) in enumerate(
        (("lenstool_ensemble_mean", "Lenstool"), ("glafic_best", "GLAFIC"))
    ):
        block = loocv[loocv.target_kind.eq(target)].set_index("holdout_system").loc[systems]
        ax.bar(x + (offset - 0.5) * 0.36, 100 * block.improvement_over_C0351, width=0.36, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="LOOCV adaptive improvement over fixed C0351", ylabel="improvement (%)", xticks=x)
    ax.set_xticklabels(systems, rotation=45, ha="right", fontsize=8)
    ax.legend()

    ax = axes[0, 1]
    top = impacts.head(14).sort_values("lenstool_median_delta_JS")
    ax.barh(top.candidate_id, top.lenstool_median_delta_JS, color=np.where(top.lenstool_median_delta_JS < 0, "tab:green", "tab:red"))
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="Largest one-change impacts", xlabel="median delta JS (negative is better)")

    ax = axes[1, 0]
    block = interaction[interaction.method.eq("lenstool")].sort_values("marginal_median_JS_span")
    ax.barh(block.parameter, block.marginal_median_JS_span, color="tab:purple")
    ax.set(title="Adaptive interaction impact", xlabel="marginal median-JS span")

    ax = axes[1, 1]
    block = realized[realized.candidate_id.eq(selected_id)].sort_values("extent_coordinate")
    scatter = ax.scatter(block.extent_coordinate, block.routing_fraction, c=block.r50_kpc, s=70, cmap="viridis")
    ax.plot(block.extent_coordinate, block.routing_fraction, color="tab:blue", alpha=0.5)
    ax.set(title=f"All-data selected law {selected_id}", xlabel="baryonic extent coordinate", ylabel="realized routed fraction")
    fig.colorbar(scatter, ax=ax, label="R50 (kpc)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs/adaptive_route_kernel_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    specs = candidate_specs(protocol)
    spec_frame = pd.DataFrame(specs)
    spec_frame.to_csv(output / protocol["outputs"]["candidate_specs"], index=False)
    acquisition, analysis, contexts, targets = load_contexts_and_targets(protocol)
    morphology = pd.read_csv(ROOT / protocol["inputs"]["morphology"]).set_index("system")
    forward_protocol = prediction_protocol(acquisition)
    comparator_map = {
        item["candidate_id"]: item
        for item in acquisition["locked_candidates"]
        if item["candidate_id"] in protocol["validation"]["comparators"]
    }

    score_rows, realized_rows, comparator_rows = [], [], []
    systems = [item["label"] for item in acquisition["systems"]]
    for system_index, system in enumerate(systems):
        context = contexts[system]
        morph = morphology.loc[system]
        for spec_index, spec in enumerate(specs):
            prediction, realized = adaptive_prediction(context, morph, spec, forward_protocol)
            realized_rows.append(
                {
                    "system": system,
                    "candidate_id": spec["candidate_id"],
                    "r50_kpc": float(morph.r50_kpc),
                    "concentration": float(morph.radial_concentration_r50_over_r80),
                    **realized,
                }
            )
            for target_kind in ("lenstool_ensemble_mean", "glafic_best"):
                metrics = shape_metrics(prediction, targets[(system, target_kind)], context.aperture)
                score_rows.append(
                    {"system": system, "target_kind": target_kind, "candidate_id": spec["candidate_id"], **metrics}
                )
            if (spec_index + 1) % 100 == 0:
                print(f"{system}: candidates {spec_index + 1}/{len(specs)}", flush=True)
        for name, comparator_spec in comparator_map.items():
            prediction = prediction_for_spec(context, comparator_spec, forward_protocol)
            for target_kind in ("lenstool_ensemble_mean", "glafic_best"):
                metrics = shape_metrics(prediction, targets[(system, target_kind)], context.aperture)
                comparator_rows.append(
                    {"system": system, "target_kind": target_kind, "comparator": name, **metrics}
                )
        print(f"Scored system {system_index + 1}/{len(systems)}: {system}", flush=True)

    scores = pd.DataFrame(score_rows)
    realized = pd.DataFrame(realized_rows)
    comparators = pd.DataFrame(comparator_rows)
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    realized.to_csv(output / protocol["outputs"]["realized_parameters"], index=False)
    comparators.to_csv(output / protocol["outputs"]["comparators"], index=False)

    loocv = loocv_predictions(scores, comparators, systems)
    loocv = loocv.merge(
        spec_frame.add_prefix("selected_").rename(columns={"selected_candidate_id": "selected_candidate_id"}),
        on="selected_candidate_id",
        validate="many_to_one",
    )
    loocv.to_csv(output / protocol["outputs"]["loocv"], index=False)
    oat = one_at_a_time_impacts(scores, spec_frame)
    oat.to_csv(output / protocol["outputs"]["one_at_a_time_impacts"], index=False)
    interaction = interaction_impacts(scores, spec_frame)
    interaction.to_csv(output / protocol["outputs"]["interaction_impacts"], index=False)

    primary = loocv[loocv.target_kind.eq("lenstool_ensemble_mean")]
    independent = loocv[loocv.target_kind.eq("glafic_best")]
    same_sign = int(
        np.sum(
            np.sign(primary.set_index("holdout_system").improvement_over_C0351.loc[systems].to_numpy(float))
            == np.sign(independent.set_index("holdout_system").improvement_over_C0351.loc[systems].to_numpy(float))
        )
    )
    values = {
        "median_LOOCV_improvement_over_C0351": float(primary.improvement_over_C0351.median()),
        "clusters_better_than_C0351": int(np.sum(primary.improvement_over_C0351 > 0.0)),
        "median_LOOCV_improvement_over_W060": float(primary.improvement_over_W060.median()),
        "clusters_better_than_W060": int(np.sum(primary.improvement_over_W060 > 0.0)),
        "clusters_better_than_LOCAL75": int(np.sum(primary.improvement_over_LOCAL75 > 0.0)),
        "clusters_better_than_CENTRAL100": int(np.sum(primary.improvement_over_CENTRAL100 > 0.0)),
        "same_sign_improvement_between_methods": same_sign,
    }
    gate = protocol["validation"]["primary_gate"]
    gate_pass = bool(
        values["median_LOOCV_improvement_over_C0351"] > float(gate["median_LOOCV_improvement_over_C0351"])
        and values["clusters_better_than_C0351"] >= int(gate["clusters_better_than_C0351"])
        and values["same_sign_improvement_between_methods"] >= int(gate["same_sign_improvement_between_methods"])
    )
    all_training = scores[scores.target_kind.eq("lenstool_ensemble_mean")].groupby("candidate_id").jensen_shannon.mean().sort_values()
    selected_id = str(all_training.index[0])
    selected_spec = spec_frame[spec_frame.candidate_id.eq(selected_id)].iloc[0].to_dict()
    selection_counts = loocv[loocv.target_kind.eq("lenstool_ensemble_mean")].selected_candidate_id.value_counts().to_dict()

    scalar_path = ROOT / protocol["inputs"]["scalar_parent_scores"]
    scalar_scores = pd.read_csv(scalar_path)
    scalar = scalar_scores[scalar_scores.candidate_id.eq(protocol["cross_domain_controls"]["scalar_parent"])].iloc[0]
    report = {
        "protocol_version": protocol["protocol_version"],
        "coverage": {
            "clusters": len(systems),
            "lensing_methods": 2,
            "candidate_laws": len(specs),
            "map_scores": len(scores),
            "LOOCV_holdouts": len(systems),
        },
        "LOOCV_values": values,
        "LOOCV_gate": gate,
        "LOOCV_gate_passed": gate_pass,
        "selection_frequency": selection_counts,
        "all_cluster_selected_candidate": selected_spec,
        "all_cluster_selected_mean_Lenstool_JS": float(all_training.iloc[0]),
        "largest_one_at_a_time_impacts": oat.head(12).to_dict("records"),
        "interaction_parameter_impacts": interaction.to_dict("records"),
        "cross_domain_controls": {
            "scalar_parent": scalar.candidate_id,
            "galaxy_outer_RMSE_km_s": float(scalar.cross_galaxy_outer_RMSE_km_s),
            "CLASH_absolute_RMSE_dex": float(scalar.cluster_RMSE_dex),
            "Solar_all_proxies_pass": bool(scalar.all_solar_proxies_pass),
            "directional_change_for_single_centered_source": 0.0,
            "raw_RXJ2129_advance_gate_passed": gate_pass,
        },
        "claim_status": protocol["validation"]["claim_status"],
        "claim_limits": protocol["claim_limits"],
        "hashes": {
            "protocol": sha256(config_path),
            "acquisition_protocol": sha256(ROOT / protocol["inputs"]["acquisition_protocol"]),
            "analysis_protocol": sha256(ROOT / protocol["inputs"]["analysis_protocol"]),
            "sources": sha256(ROOT / protocol["inputs"]["sources"]),
            "systems": sha256(ROOT / protocol["inputs"]["systems"]),
            "morphology": sha256(ROOT / protocol["inputs"]["morphology"]),
            "inverse_driver_report": sha256(ROOT / protocol["inputs"]["inverse_driver_report"]),
            "scalar_parent_scores": sha256(scalar_path),
        },
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(loocv, oat, interaction, realized, selected_id, output / protocol["outputs"]["figure"])
    lines = [
        "# Extent-adaptive route-kernel result",
        "",
        f"The frozen family contained **{len(specs)}** conservative route laws and was evaluated by ten leave-one-cluster-out folds against Lenstool, with GLAFIC as an unselected method control.",
        "",
        f"Median held-out improvement over fixed C0351: **{100*values['median_LOOCV_improvement_over_C0351']:.2f}%**; wins: **{values['clusters_better_than_C0351']}/10**; method-sign agreement: **{same_sign}/10**.",
        f"The predeclared transfer gate passed: **{gate_pass}**.",
        "",
        f"The all-cluster exploratory selection is `{selected_id}`. It is a candidate for a fresh sample, not a confirmed law.",
    ]
    (output / protocol["outputs"]["summary"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(json_safe({"LOOCV": values, "gate_passed": gate_pass, "all_cluster_selected": selected_id}), indent=2))


if __name__ == "__main__":
    main()
