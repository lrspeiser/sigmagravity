#!/usr/bin/env python3
"""Training-selected, mass-conserving spatial-vector lens diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_rxj2129_raw_theory_lensing import RawLens, near_bound, score, spec_for
from run_unbounded_running_multicluster_raw import (
    acceleration,
    aggregate_system_scores,
    build_field,
    load_anchors,
    load_system_images,
    predictive_split,
    system_protocol,
)

from voidscreen.spatial_lensing import (
    G_SI,
    MSUN_KG,
    MemberRedistributionField,
    RadialEnhancementField,
)

KPC_M = 3.085677581491367e19


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deep_update(target: dict, changes: dict) -> dict:
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_protocol_config(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    parent = raw.get("parent_protocol")
    if parent is None:
        return raw
    protocol = json.loads((ROOT / parent).read_text(encoding="utf-8"))
    protocol["parent_protocol"] = parent
    for key in ("protocol_version", "frozen_utc", "status", "purpose"):
        protocol[key] = raw[key]
    protocol["pre_score_disclosure"] = [
        *protocol["pre_score_disclosure"],
        *raw.get("additional_pre_score_disclosure", []),
    ]
    return deep_update(protocol, raw.get("overrides", {}))


def variant_name(model: str, dressing: str) -> str:
    return f"{model}__members_{dressing}"


@dataclass
class SystemContext:
    system: dict
    local_protocol: dict
    training: pd.DataFrame
    heldout: pd.DataFrame
    anchors: pd.DataFrame
    fields: dict
    member_table: pd.DataFrame
    member_fields: dict[float, MemberRedistributionField]
    enhancements: dict[str, RadialEnhancementField]
    baseline_parameters: dict[str, np.ndarray]


class SpatialVectorLens(RawLens):
    """Raw lens with a fixed member field minus its circular mean."""

    def __init__(
        self,
        protocol: dict,
        fields: dict,
        *,
        base_model: str,
        member_field: MemberRedistributionField,
        mass_fraction: float,
        dressing: str,
        enhancement: RadialEnhancementField,
    ):
        super().__init__(protocol, fields)
        self.base_model = base_model
        self.member_field = member_field
        self.mass_fraction = float(mass_fraction)
        self.dressing = dressing
        self.enhancement = enhancement

    def alpha(
        self,
        model: str,
        parameters: np.ndarray,
        x_arcsec,
        y_arcsec,
        source_redshift: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        base_x, base_y = super().alpha(
            self.base_model, parameters, x_arcsec, y_arcsec, source_redshift
        )
        if self.mass_fraction == 0.0:
            return base_x, base_y
        x = np.asarray(x_arcsec, dtype=float)
        y = np.asarray(y_arcsec, dtype=float)
        ratio = self.distance_ratio(source_redshift)
        contrast_x, contrast_y = self.member_field.contrast_alpha_arcsec(
            x, y, distance_ratio=ratio
        )
        if self.dressing == "GR_linear":
            multiplier = 1.0
        elif self.dressing == "locked_running":
            multiplier = self.enhancement(np.hypot(x, y))
        else:
            raise ValueError(self.dressing)
        amplitude = self.mass_fraction * multiplier
        return base_x + amplitude * contrast_x, base_y + amplitude * contrast_y


def load_member_table(path: Path, system: dict) -> pd.DataFrame:
    records = []
    cosine = math.cos(math.radians(float(system["center_dec_deg"])))
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 7:
            continue
        try:
            ra = float(fields[1])
            dec = float(fields[2])
            magnitude = float(fields[6])
        except ValueError:
            continue
        records.append(
            {
                "member_id": str(fields[0]),
                "ra_deg": ra,
                "dec_deg": dec,
                "magnitude": magnitude,
                "x_arcsec": (ra - float(system["center_ra_deg"])) * 3600.0 * cosine,
                "y_arcsec": (dec - float(system["center_dec_deg"])) * 3600.0,
            }
        )
    table = pd.DataFrame(records)
    if len(table) < 10:
        raise RuntimeError(f"too few members for {system['label']}")
    relative_light = np.power(
        10.0, -0.4 * (table.magnitude.to_numpy(float) - table.magnitude.min())
    )
    table["normalized_light_weight"] = relative_light / relative_light.sum()
    table["radius_arcsec"] = np.hypot(table.x_arcsec, table.y_arcsec)
    return table.sort_values("member_id").reset_index(drop=True)


def baryonic_normalization_mass_msun(
    anchors: pd.DataFrame, baryonic_settings: dict
) -> tuple[float, float]:
    aperture = baryonic_settings.get("normalization_aperture_kpc")
    if aperture is None:
        row = anchors.sort_values("radius_kpc").iloc[-1]
    else:
        selected = anchors[np.isclose(anchors.radius_kpc.astype(float), float(aperture))]
        if len(selected) != 1:
            raise RuntimeError(f"missing exact common baryonic aperture {aperture} kpc")
        row = selected.iloc[0]
    radius_m = float(row.radius_kpc) * KPC_M
    acceleration_m_s2 = 10.0 ** float(row.log_gbar)
    mass = acceleration_m_s2 * radius_m**2 / (G_SI * MSUN_KG)
    return float(mass), float(row.radius_kpc)


def baseline_parameter_vector(
    geometry: pd.DataFrame, system_name: str, model: str, cutoff_kpc: float
) -> np.ndarray:
    selected = geometry[
        geometry.system.eq(system_name)
        & geometry.model.eq(model)
        & np.isclose(geometry.cutoff_kpc.astype(float), cutoff_kpc)
    ]
    if len(selected) != 1:
        raise RuntimeError(f"missing baseline geometry for {system_name}/{model}")
    row = selected.iloc[0]
    return np.asarray([float(row[label]) for label in spec_for(model).labels])


def optimization_rms(lens: RawLens, model: str, parameters, rows: pd.DataFrame) -> float:
    residual, _ = lens.profiled_residuals(model, np.asarray(parameters), rows)
    image = residual.reshape(-1, 2) * lens.sigma
    return float(np.sqrt(np.mean(np.sum(image**2, axis=1))))


def build_contexts(protocol: dict) -> tuple[list[SystemContext], list[dict], dict[str, str]]:
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    tian = pd.read_csv(
        ROOT / protocol["baryonic_profile"]["input"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    geometry = pd.read_csv(ROOT / protocol["inputs"]["baseline_geometry"])
    grid = protocol["spatial_vector_grid"]
    impact = np.geomspace(
        float(grid["radial_range_arcsec"][0]),
        float(grid["radial_range_arcsec"][1]),
        int(grid["radial_samples"]),
    )
    contexts = []
    audit_rows = []
    input_hashes = {
        "image_catalog": sha256(ROOT / protocol["inputs"]["image_catalog"]),
        "baryonic_profile": sha256(ROOT / protocol["baryonic_profile"]["input"]),
        "baseline_report": sha256(ROOT / protocol["inputs"]["baseline_report"]),
        "baseline_geometry": sha256(ROOT / protocol["inputs"]["baseline_geometry"]),
    }
    for system in protocol["systems"]:
        print(f"prepare system={system['label']}", flush=True)
        local = system_protocol(protocol, system)
        images = load_system_images(catalog, system)
        training, heldout = predictive_split(images)
        anchors = load_anchors(tian, system["label"])
        normalization_mass, normalization_radius = baryonic_normalization_mass_msun(
            anchors, protocol["baryonic_profile"]
        )
        member_path = ROOT / system["member_catalog"]
        members = load_member_table(member_path, system)
        input_hashes[f"members_{system['label']}"] = sha256(member_path)
        distance_mpc = float(
            RawLens(local, {}).cosmo.angular_diameter_distance(float(system["lens_redshift"])).value
        )
        member_fields = {}
        for softening in grid["softening_arcsec"]:
            member_fields[float(softening)] = MemberRedistributionField.build(
                members.x_arcsec.to_numpy(float),
                members.y_arcsec.to_numpy(float),
                members.normalized_light_weight.to_numpy(float),
                total_mass_msun=normalization_mass,
                lens_angular_diameter_distance_mpc=distance_mpc,
                softening_arcsec=float(softening),
                impact_arcsec=impact,
                azimuth_samples=int(grid["azimuth_samples"]),
            )
        fields = {}
        enhancements = {}
        scale = float(local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
        _, gbar_on_impact = acceleration("baryons_GR", impact * scale, anchors, protocol)
        for model in protocol["models"]:
            fields[model], _ = build_field(
                model,
                anchors,
                protocol,
                local,
                float(protocol["baryonic_profile"]["primary_cutoff_kpc"]),
            )
            _, predicted = acceleration(model, impact * scale, anchors, protocol)
            enhancements[model] = RadialEnhancementField(impact, predicted / gbar_on_impact)
        baseline = {
            model: baseline_parameter_vector(
                geometry,
                system["system"],
                model,
                float(protocol["baryonic_profile"]["primary_cutoff_kpc"]),
            )
            for model in protocol["models"]
        }
        independent_phi = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        check_radius = np.geomspace(0.1, 300.0, 40)
        for softening, member_field in member_fields.items():
            x = check_radius[:, None] * np.cos(independent_phi)[None, :]
            y = check_radius[:, None] * np.sin(independent_phi)[None, :]
            ax, ay = member_field.contrast_alpha_arcsec(x, y, distance_ratio=1.0)
            radial_closure = np.mean(
                ax * np.cos(independent_phi)[None, :]
                + ay * np.sin(independent_phi)[None, :],
                axis=1,
            )
            audit_rows.append(
                {
                    "system": system["system"],
                    "label": system["label"],
                    "members": len(members),
                    "softening_arcsec": softening,
                    "normalization_radius_kpc": normalization_radius,
                    "profile_maximum_radius_kpc": float(anchors.radius_kpc.max()),
                    "normalization_mass_msun": normalization_mass,
                    "brightest_light_weight": float(members.normalized_light_weight.max()),
                    "median_member_radius_arcsec": float(members.radius_arcsec.median()),
                    "maximum_independent_circular_mean_residual_arcsec": float(
                        np.max(np.abs(radial_closure))
                    ),
                }
            )
        contexts.append(
            SystemContext(
                system,
                local,
                training,
                heldout,
                anchors,
                fields,
                members,
                member_fields,
                enhancements,
                baseline,
            )
        )
    return contexts, audit_rows, input_hashes


def make_lens(
    context: SystemContext,
    model: str,
    dressing: str,
    fraction: float,
    softening: float,
) -> SpatialVectorLens:
    return SpatialVectorLens(
        context.local_protocol,
        context.fields,
        base_model=model,
        member_field=context.member_fields[float(softening)],
        mass_fraction=float(fraction),
        dressing=dressing,
        enhancement=context.enhancements[model],
    )


def screen_grid(protocol: dict, contexts: list[SystemContext]) -> pd.DataFrame:
    rows = []
    grid = protocol["spatial_vector_grid"]
    first_softening = float(grid["softening_arcsec"][0])
    for model in protocol["models"]:
        for dressing in grid["dressings"]:
            for fraction in grid["mass_fractions"]:
                for softening in grid["softening_arcsec"]:
                    if float(fraction) == 0.0 and float(softening) != first_softening:
                        continue
                    system_rms = []
                    for context in contexts:
                        lens = make_lens(context, model, dressing, fraction, softening)
                        rms = optimization_rms(
                            lens,
                            variant_name(model, dressing),
                            context.baseline_parameters[model],
                            context.training,
                        )
                        system_rms.append(rms)
                        rows.append(
                            {
                                "row_type": "system",
                                "model": model,
                                "dressing": dressing,
                                "mass_fraction": float(fraction),
                                "softening_arcsec": float(softening),
                                "system": context.system["system"],
                                "training_optimization_RMS_arcsec": rms,
                            }
                        )
                    rows.append(
                        {
                            "row_type": "aggregate",
                            "model": model,
                            "dressing": dressing,
                            "mass_fraction": float(fraction),
                            "softening_arcsec": float(softening),
                            "system": "equal_system",
                            "training_optimization_RMS_arcsec": float(
                                np.sqrt(np.mean(np.square(system_rms)))
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def shortlist_grid(protocol: dict, screen: pd.DataFrame) -> list[dict]:
    aggregate = screen[screen.row_type.eq("aggregate")]
    count = int(protocol["spatial_vector_grid"]["shortlist_per_model_and_dressing"])
    selected = []
    for (model, dressing), block in aggregate.groupby(["model", "dressing"], sort=False):
        for row in block.nsmallest(count, "training_optimization_RMS_arcsec").itertuples():
            selected.append(
                {
                    "model": model,
                    "dressing": dressing,
                    "mass_fraction": float(row.mass_fraction),
                    "softening_arcsec": float(row.softening_arcsec),
                    "screen_RMS_arcsec": float(row.training_optimization_RMS_arcsec),
                }
            )
    return selected


def refit_shortlist(
    protocol: dict, contexts: list[SystemContext], shortlist: list[dict]
) -> pd.DataFrame:
    rows = []
    starts = int(protocol["spatial_vector_grid"]["shortlist_refit_multistarts"])
    base_seed = int(protocol["optimization"]["random_seed"])
    for setting_index, setting in enumerate(shortlist):
        model = setting["model"]
        dressing = setting["dressing"]
        name = variant_name(model, dressing)
        print(
            f"shortlist {name} f={setting['mass_fraction']} s={setting['softening_arcsec']}",
            flush=True,
        )
        for system_index, context in enumerate(contexts):
            lens = make_lens(
                context,
                model,
                dressing,
                setting["mass_fraction"],
                setting["softening_arcsec"],
            )
            fitted = lens.fit(
                name,
                context.training,
                starts=starts,
                seed=base_seed + setting_index * 100 + system_index,
                initial_override=context.baseline_parameters[model],
            )
            rows.append(
                {
                    **setting,
                    "system": context.system["system"],
                    "training_optimization_RMS_arcsec": fitted[
                        "optimization_radial_RMS_arcsec"
                    ],
                    "cost": float(fitted["result"].cost),
                    "geometry_at_boundary": any(near_bound(name, fitted["result"].x).values()),
                }
            )
    return pd.DataFrame(rows)


def choose_settings(refits: pd.DataFrame) -> list[dict]:
    choices = []
    keys = ["model", "dressing", "mass_fraction", "softening_arcsec"]
    aggregate = (
        refits.groupby(keys, as_index=False)
        .training_optimization_RMS_arcsec.apply(
            lambda values: float(np.sqrt(np.mean(np.square(values))))
        )
        .rename(columns={"training_optimization_RMS_arcsec": "selection_RMS_arcsec"})
    )
    for (model, dressing), block in aggregate.groupby(["model", "dressing"], sort=False):
        best = block.nsmallest(1, "selection_RMS_arcsec").iloc[0]
        choices.append(
            {
                "model": model,
                "dressing": dressing,
                "mass_fraction": float(best.mass_fraction),
                "softening_arcsec": float(best.softening_arcsec),
                "selection_RMS_arcsec": float(best.selection_RMS_arcsec),
            }
        )
    return choices


def final_score(
    protocol: dict, contexts: list[SystemContext], choices: list[dict]
) -> tuple[dict, list[pd.DataFrame], list[dict]]:
    result = {}
    predictions = []
    parameters = []
    starts = int(protocol["spatial_vector_grid"]["final_refit_multistarts"])
    base_seed = int(protocol["optimization"]["random_seed"]) + 10000
    for choice_index, choice in enumerate(choices):
        model = choice["model"]
        dressing = choice["dressing"]
        name = variant_name(model, dressing)
        result[name] = {"selection": choice, "systems": {}}
        print(
            f"final {name} f={choice['mass_fraction']} s={choice['softening_arcsec']}",
            flush=True,
        )
        for system_index, context in enumerate(contexts):
            lens = make_lens(
                context,
                model,
                dressing,
                choice["mass_fraction"],
                choice["softening_arcsec"],
            )
            fitted = lens.fit(
                name,
                context.training,
                starts=starts,
                seed=base_seed + choice_index * 100 + system_index,
                initial_override=context.baseline_parameters[model],
            )
            train = lens.exact_predictions(
                name,
                fitted["result"].x,
                fitted["sources"],
                context.training,
                stage="training",
            )
            heldout = lens.exact_predictions(
                name,
                fitted["result"].x,
                fitted["sources"],
                context.heldout,
                stage="heldout",
            )
            for table in (train, heldout):
                table.insert(0, "system", context.system["system"])
                table.insert(1, "base_model", model)
                table.insert(2, "dressing", dressing)
                table.insert(3, "mass_fraction", choice["mass_fraction"])
                table.insert(4, "softening_arcsec", choice["softening_arcsec"])
                predictions.append(table)
            system_score = {
                "training": score(train, lens.sigma, free_parameters=len(fitted["result"].x)),
                "heldout": score(heldout, lens.sigma),
                "geometry_at_boundary": near_bound(name, fitted["result"].x),
            }
            result[name]["systems"][context.system["system"]] = system_score
            spec = spec_for(name)
            parameters.append(
                {
                    "system": context.system["system"],
                    "variant": name,
                    "base_model": model,
                    "dressing": dressing,
                    "mass_fraction": choice["mass_fraction"],
                    "softening_arcsec": choice["softening_arcsec"],
                    **dict(zip(spec.labels, fitted["result"].x, strict=True)),
                }
            )
        result[name]["aggregate_heldout"] = aggregate_system_scores(
            [
                result[name]["systems"][context.system["system"]]["heldout"]
                for context in contexts
            ]
        )
    return result, predictions, parameters


def make_figure(report: dict, output: Path) -> None:
    variants = list(report["spatial_variants"])
    systems = list(report["systems"])
    labels = report["system_labels"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x = np.arange(len(systems))
    width = 0.8 / (len(variants) + 2)
    baseline_models = ["GR_plus_cluster_halo"]
    entries = variants + baseline_models
    for index, name in enumerate(entries):
        if name in report["spatial_variants"]:
            values = [
                (
                    value
                    if (
                        value := report["spatial_variants"][name]["systems"][
                            system
                        ]["heldout"]["exact_radial_RMS_arcsec"]
                    )
                    is not None
                    else np.nan
                )
                for system in systems
            ]
        else:
            values = [
                report["baseline"]["system_scores"][system][name]["heldout"][
                    "exact_radial_RMS_arcsec"
                ]
                for system in systems
            ]
        axes[0].bar(x + (index - len(entries) / 2) * width, values, width, label=name)
    axes[0].set_xticks(x, [labels[system] for system in systems], rotation=25)
    axes[0].set_ylabel("held-out radial RMS (arcsec)")
    axes[0].set_title("Member-vector variants by cluster")
    axes[0].legend(fontsize=6)

    names = []
    values = []
    aggregate_valid = []
    for model in report["model_names"]:
        names.append(f"{model}\nspherical")
        values.append(report["baseline"]["primary_aggregate"][model]["equal_system_radial_RMS_arcsec"])
        aggregate_valid.append(True)
    for variant in variants:
        names.append(variant.replace("__members_", "\n"))
        aggregate = report["spatial_variants"][variant]["aggregate_heldout"]
        valid = bool(aggregate["all_roots_converged"])
        values.append(aggregate["equal_system_radial_RMS_arcsec"] if valid else np.nan)
        aggregate_valid.append(valid)
    names.append("compact\nhalo")
    values.append(
        report["baseline"]["primary_aggregate"]["GR_plus_cluster_halo"][
            "equal_system_radial_RMS_arcsec"
        ]
    )
    aggregate_valid.append(True)
    axes[1].bar(np.arange(len(names)), values, color=["#888888"] * 2 + ["#2774AE"] * len(variants) + ["#7B3294"])
    for index, valid in enumerate(aggregate_valid):
        if not valid:
            axes[1].text(
                index,
                0.5,
                "FAILED\nROOT",
                ha="center",
                va="bottom",
                rotation=90,
                color="crimson",
                fontweight="bold",
                fontsize=7,
            )
    axes[1].set_xticks(np.arange(len(names)), names, rotation=25, ha="right", fontsize=7)
    axes[1].set_ylabel("equal-system held-out RMS (arcsec)")
    axes[1].set_title("Mass-conserving spatial test")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/unbounded_running_spatial_vector_protocol.json"
    )
    arguments = parser.parse_args()
    config_path = ROOT / arguments.config
    protocol = load_protocol_config(config_path)
    if protocol["status"] not in {
        "frozen_before_spatial_vector_scores",
        "frozen_after_primary_scores_before_common_aperture_scores",
    }:
        raise RuntimeError("spatial-vector protocol was not frozen")
    output = (ROOT / protocol["outputs"]["report"]).parent
    output.mkdir(parents=True, exist_ok=True)
    contexts, audit_rows, input_hashes = build_contexts(protocol)
    pd.DataFrame(audit_rows).to_csv(ROOT / protocol["outputs"]["member_audit"], index=False)

    print("screen frozen grid at prior training geometry", flush=True)
    screen = screen_grid(protocol, contexts)
    screen.to_csv(ROOT / protocol["outputs"]["grid_screen"], index=False)
    shortlist = shortlist_grid(protocol, screen)
    refits = refit_shortlist(protocol, contexts, shortlist)
    refits.to_csv(ROOT / protocol["outputs"]["selection_refits"], index=False)
    choices = choose_settings(refits)
    spatial, predictions, parameters = final_score(protocol, contexts, choices)

    baseline = json.loads((ROOT / protocol["inputs"]["baseline_report"]).read_text(encoding="utf-8"))
    gates = protocol["advance_gates"]
    gate_audit = {}
    for name, block in spatial.items():
        base_model = block["selection"]["model"]
        aggregate = block["aggregate_heldout"]
        rms = aggregate["equal_system_radial_RMS_arcsec"]
        base_rms = baseline["primary_aggregate"][base_model]["equal_system_radial_RMS_arcsec"]
        halo_rms = baseline["primary_aggregate"]["GR_plus_cluster_halo"]["equal_system_radial_RMS_arcsec"]
        improvement = 1.0 - rms / base_rms
        no_boundary = all(
            not any(block["systems"][context.system["system"]]["geometry_at_boundary"].values())
            for context in contexts
        )
        audit = {
            "all_heldout_roots_converged": aggregate["all_roots_converged"],
            "absolute_RMS_pass": rms <= float(gates["equal_system_heldout_radial_RMS_arcsec_max"]),
            "improvement_over_same_locked_spherical_model_fraction": improvement,
            "improvement_gate_pass": improvement
            >= float(gates["improvement_over_same_locked_spherical_model_fraction_min"]),
            "compact_halo_RMS_ratio": rms / halo_rms,
            "compact_halo_ratio_pass": rms / halo_rms
            <= float(gates["candidate_to_compact_halo_equal_system_RMS_ratio_max"]),
            "no_geometry_parameter_at_boundary": no_boundary,
        }
        audit["all_gates_pass"] = bool(
            audit["all_heldout_roots_converged"]
            and audit["absolute_RMS_pass"]
            and audit["improvement_gate_pass"]
            and audit["compact_halo_ratio_pass"]
            and audit["no_geometry_parameter_at_boundary"]
        )
        gate_audit[name] = audit

    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed mass-conserving spatial-vector diagnostic",
        "protocol": {
            "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(config_path),
        },
        "input_hashes": input_hashes,
        "claim_boundary": protocol["pre_score_disclosure"],
        "systems": [context.system["system"] for context in contexts],
        "system_labels": {context.system["system"]: context.system["label"] for context in contexts},
        "model_names": list(protocol["models"]),
        "selection": choices,
        "spatial_variants": spatial,
        "baseline": {
            "primary_aggregate": baseline["primary_aggregate"],
            "system_scores": baseline["system_scores"],
        },
        "gate_audit": gate_audit,
        "verdict": {
            "survivors": [name for name, audit in gate_audit.items() if audit["all_gates_pass"]],
            "best_variant": min(
                spatial,
                key=lambda name: spatial[name]["aggregate_heldout"]["equal_system_radial_RMS_arcsec"],
            ),
        },
    }
    (ROOT / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        ROOT / protocol["outputs"]["predictions"], index=False
    )
    pd.DataFrame(parameters).to_csv(ROOT / protocol["outputs"]["parameters"], index=False)
    make_figure(report, ROOT / protocol["outputs"]["figure"])

    ranking = sorted(
        spatial,
        key=lambda name: spatial[name]["aggregate_heldout"]["equal_system_radial_RMS_arcsec"],
    )
    lines = [
        "# Unbounded running: mass-conserving spatial-vector diagnostic",
        "",
        "The member-light perturbation subtracts its circular average, so it changes angular structure without adding a radial deflection budget or object-specific gravity amplitude.",
        "",
        "| variant | selected f | softening (arcsec) | held-out RMS (arcsec) | improvement vs spherical | halo ratio | roots | survivor |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for name in ranking:
        block = spatial[name]
        audit = gate_audit[name]
        choice = block["selection"]
        aggregate = block["aggregate_heldout"]
        lines.append(
            f"| {name} | {choice['mass_fraction']:.3f} | {choice['softening_arcsec']:.2f} | "
            f"{aggregate['equal_system_radial_RMS_arcsec']:.3f} | "
            f"{100.0 * audit['improvement_over_same_locked_spherical_model_fraction']:+.1f}% | "
            f"{audit['compact_halo_RMS_ratio']:.2f} | "
            f"{'all' if aggregate['all_roots_converged'] else 'failed'} | "
            f"{'yes' if audit['all_gates_pass'] else 'no'} |"
        )
    lines += [
        "",
        f"Compact one-halo comparator: {baseline['primary_aggregate']['GR_plus_cluster_halo']['equal_system_radial_RMS_arcsec']:.3f} arcsec.",
        f"Survivors: **{', '.join(report['verdict']['survivors']) or 'none'}**.",
    ]
    (ROOT / protocol["outputs"]["summary"]).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print((ROOT / protocol["outputs"]["summary"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
