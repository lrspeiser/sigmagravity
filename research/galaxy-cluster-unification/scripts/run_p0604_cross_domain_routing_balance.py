#!/usr/bin/env python3
"""Map the galaxy-cluster tradeoff of the simplified inward route kernel."""

from __future__ import annotations

import itertools
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

from run_gravity_arc_fresh_sample import prediction_protocol  # noqa: E402
from run_gravity_arc_tomography import (  # noqa: E402
    normalized_in_aperture,
    prediction_for_spec,
    shape_metrics,
)
from run_p0580_conservative_return_sparc import galaxy_force_profile, score  # noqa: E402
from run_p0593_diffusion_cross_domain import acceleration_velocity, characteristic_acceleration  # noqa: E402
from run_p0603_tensor_routing import (  # noqa: E402
    balanced_folds,
    json_safe,
    load_cluster_contexts,
    shape_factor,
)
from voidscreen.conservative_diffusion import low_acceleration_activation  # noqa: E402
from voidscreen.data import KPC_M  # noqa: E402
from voidscreen.tensor_routing import (  # noqa: E402
    anisotropic_gaussian_deposit,
    baryonic_field_frames,
    redistributed_cumulative_mass_tensor,
    weighted_radii,
)
from voidscreen.unified import G_SI, M_SUN_KG, rar_acceleration  # noqa: E402


SOLAR_RADIUS_M = 6.957e8


def candidate_specs(protocol: dict) -> list[dict]:
    specs = []
    for fraction, length, width, gate in itertools.product(
        protocol["grid"]["fraction_max"],
        protocol["grid"]["length_over_R80"],
        protocol["grid"]["width_over_R80"],
        protocol["grid"]["shape_gate"],
    ):
        specs.append(
            {
                "candidate_id": f"F{fraction:g}_L{length:g}_W{width:g}_{gate}",
                "fraction_max": float(fraction),
                "length_over_R80": float(length),
                "width_over_R80": float(width),
                "shape_gate": str(gate),
            }
        )
    if len(specs) != int(protocol["grid"]["candidate_count"]):
        raise RuntimeError("P0604 candidate count changed")
    return specs


def cluster_scores(protocol: dict, specs: list[dict]):
    acquisition, contexts, targets = load_cluster_contexts(
        json.loads((ROOT / protocol["inputs"]["P0603_protocol"]).read_text())
    )
    forward_protocol = prediction_protocol(acquisition)
    controls = [
        item
        for item in acquisition["locked_candidates"]
        if item["candidate_id"] in {"LOCAL75", "CENTRAL100", "C0351", "W060"}
    ]
    records = []
    geometry_cache = {}
    for label, context in contexts.items():
        _, _, r80, concentration = weighted_radii(context.positions, context.soft_weights)
        frames = baryonic_field_frames(context.positions, context.soft_weights, softening=20.0)
        local = anisotropic_gaussian_deposit(
            context.axis_kpc,
            context.positions,
            context.soft_weights,
            frames["tidal"],
            geometric_sigma=float(protocol["fixed_constants"]["local_width_over_R80"]) * r80,
            axis_ratio=1.0,
        )
        for length, width in itertools.product(
            protocol["grid"]["length_over_R80"],
            protocol["grid"]["width_over_R80"],
        ):
            ratio = np.maximum(frames["radius"] / r80, 1.0e-6)
            travel = float(length) * r80 * np.clip(
                ratio ** float(protocol["fixed_constants"]["radius_exponent"]), 0.2, 3.0
            )
            endpoint = context.positions + travel[:, None] * frames["inward"]
            geometry_cache[(label, float(length), float(width))] = anisotropic_gaussian_deposit(
                context.axis_kpc,
                endpoint,
                context.soft_weights,
                frames["inward"],
                geometric_sigma=float(width) * r80,
                axis_ratio=1.0,
            )
        for spec in specs:
            routed = geometry_cache[
                (label, spec["length_over_R80"], spec["width_over_R80"])
            ]
            fraction = spec["fraction_max"] * shape_factor(
                spec["shape_gate"], concentration, protocol
            )
            prediction = normalized_in_aperture(
                (1.0 - fraction) * local + fraction * routed,
                context.aperture,
            )
            for target_kind in ("lenstool_best", "glafic_best"):
                records.append(
                    {
                        "system": label,
                        "target_kind": target_kind,
                        **spec,
                        **shape_metrics(prediction, targets[(label, target_kind)], context.aperture),
                    }
                )
        for control in controls:
            prediction = prediction_for_spec(context, control, forward_protocol)
            for target_kind in ("lenstool_best", "glafic_best"):
                records.append(
                    {
                        "system": label,
                        "target_kind": target_kind,
                        "candidate_id": control["candidate_id"],
                        **shape_metrics(prediction, targets[(label, target_kind)], context.aperture),
                    }
                )
    return pd.DataFrame(records)


def galaxy_scores(protocol: dict, specs: list[dict]):
    galaxy_protocol = json.loads(
        (ROOT / protocol["inputs"]["galaxy_protocol"]).read_text()
    )
    cfg = galaxy_protocol["galaxy_test"]
    raw = pd.read_csv(ROOT / cfg["points"])
    points = raw[(raw.model == cfg["model"]) & (raw.scenario == cfg["scenario"])].copy()
    points["source_point_index"] = points.index
    outer = points[points.split.eq(cfg["split"])].copy().reset_index(drop=True)
    profiles = {
        galaxy: galaxy_force_profile(block)
        for galaxy, block in points.groupby("galaxy", sort=False)
    }
    a0 = float(protocol["fixed_constants"]["a0_m_s2"])
    route_cache = {}
    for galaxy, profile in profiles.items():
        for length, width in itertools.product(
            protocol["grid"]["length_over_R80"],
            protocol["grid"]["width_over_R80"],
        ):
            routed, _ = redistributed_cumulative_mass_tensor(
                profile["radius_kpc"],
                profile["mass_solar"],
                r80=profile["R80_kpc"],
                length_over_r80=float(length),
                radius_exponent=float(protocol["fixed_constants"]["radius_exponent"]),
                width_over_r80=float(width),
                axis_ratio=1.0,
                bins=int(protocol["fixed_constants"]["radial_bins"]),
            )
            route_cache[(galaxy, float(length), float(width))] = routed
    records = []
    for spec in specs:
        prediction = np.empty(len(outer), dtype=float)
        for galaxy, raw_indices in outer.groupby("galaxy", sort=False).indices.items():
            indices = np.asarray(raw_indices, dtype=int)
            profile = profiles[galaxy]
            shape = shape_factor(
                spec["shape_gate"],
                float(profile["concentration_R50_over_R80"]),
                protocol,
            )
            activation = low_acceleration_activation(
                characteristic_acceleration(profile),
                a0_m_s2=a0,
                power=float(protocol["fixed_constants"]["source_acceleration_gate_power"]),
            )
            fraction = spec["fraction_max"] * shape * activation
            routed = route_cache[(galaxy, spec["length_over_R80"], spec["width_over_R80"])]
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
        metrics = score(outer, prediction)
        records.append({**spec, **metrics, **solar_diagnostic(protocol, spec)})
    reference = score(outer, outer.velocity_RAR_same_nuisance_km_s.to_numpy(float))
    result = pd.DataFrame(records)
    result["galaxy_equal_RMSE_ratio_to_fixed_RAR"] = (
        result.outer_equal_galaxy_RMSE_km_s
        / reference["outer_equal_galaxy_RMSE_km_s"]
    )
    return result, reference


def solar_diagnostic(protocol: dict, spec: dict) -> dict:
    radius = np.linspace(SOLAR_RADIUS_M / 512.0, SOLAR_RADIUS_M, 512)
    mass = (radius / SOLAR_RADIUS_M) ** 3
    r80 = SOLAR_RADIUS_M * 0.8 ** (1.0 / 3.0)
    concentration = (0.5 / 0.8) ** (1.0 / 3.0)
    routed, _ = redistributed_cumulative_mass_tensor(
        radius,
        mass,
        r80=r80,
        length_over_r80=spec["length_over_R80"],
        radius_exponent=float(protocol["fixed_constants"]["radius_exponent"]),
        width_over_r80=spec["width_over_R80"],
        axis_ratio=1.0,
        bins=int(protocol["fixed_constants"]["radial_bins"]),
    )
    g_r80 = G_SI * M_SUN_KG * 0.8 / r80**2
    activation = low_acceleration_activation(
        g_r80,
        a0_m_s2=float(protocol["fixed_constants"]["a0_m_s2"]),
        power=float(protocol["fixed_constants"]["source_acceleration_gate_power"]),
    )
    fraction = spec["fraction_max"] * shape_factor(
        spec["shape_gate"], concentration, protocol
    ) * activation
    effective = (1.0 - fraction) * mass + fraction * routed
    return {
        "solar_activation": activation,
        "solar_effective_route_fraction": fraction,
        "solar_maximum_absolute_interior_force_change": float(
            np.max(np.abs(effective / mass - 1.0))
        ),
    }


def cross_validate(protocol, cluster, galaxy, specs):
    tensor_ids = {spec["candidate_id"] for spec in specs}
    primary = cluster[
        cluster.target_kind.eq("lenstool_best") & cluster.candidate_id.isin(tensor_ids)
    ]
    assignment = balanced_folds(primary.system.unique(), protocol["validation"]["cluster_folds"])
    galaxy_ratio = galaxy.set_index("candidate_id").galaxy_equal_RMSE_ratio_to_fixed_RAR
    fold_rows = []
    oof_rows = []
    for budget in protocol["validation"]["galaxy_RMSE_ratio_budgets"]:
        eligible = set(galaxy_ratio[galaxy_ratio <= float(budget)].index)
        if not eligible:
            raise RuntimeError(f"no P0604 candidates satisfy galaxy budget {budget}")
        for fold in range(protocol["validation"]["cluster_folds"]):
            heldout = {name for name, value in assignment.items() if value == fold}
            training = primary[
                ~primary.system.isin(heldout) & primary.candidate_id.isin(eligible)
            ]
            mean = training.groupby("candidate_id").jensen_shannon.mean().sort_values()
            selected_id = str(mean.index[0])
            spec = next(item for item in specs if item["candidate_id"] == selected_id)
            fold_rows.append(
                {
                    "galaxy_RMSE_ratio_budget": float(budget),
                    "fold": fold,
                    "selected_candidate_id": selected_id,
                    "training_equal_JS": float(mean.iloc[0]),
                    "eligible_candidates": len(eligible),
                    "selected_galaxy_RMSE_ratio": float(galaxy_ratio.loc[selected_id]),
                    **spec,
                }
            )
            block = cluster[
                cluster.system.isin(heldout)
                & cluster.target_kind.isin(["lenstool_best", "glafic_best"])
                & cluster.candidate_id.isin(
                    [selected_id, "LOCAL75", "CENTRAL100", "C0351", "W060"]
                )
            ].copy()
            block["galaxy_RMSE_ratio_budget"] = float(budget)
            block["fold"] = fold
            block["selected_candidate_id"] = selected_id
            oof_rows.append(block)
    return pd.DataFrame(fold_rows), pd.concat(oof_rows, ignore_index=True)


def summarize_oof(oof: pd.DataFrame) -> list[dict]:
    rows = []
    controls = ["LOCAL75", "CENTRAL100", "C0351", "W060"]
    for (budget, target_kind), block in oof.groupby(
        ["galaxy_RMSE_ratio_budget", "target_kind"], sort=True
    ):
        selected = block[~block.candidate_id.isin(controls)].set_index("system")
        row = {
            "galaxy_RMSE_ratio_budget": float(budget),
            "target_kind": target_kind,
            "tensor_equal_JS": float(selected.jensen_shannon.mean()),
            "tensor_equal_Pearson": float(selected.pearson.mean()),
        }
        for control in controls:
            reference = block[block.candidate_id.eq(control)].set_index("system")
            row[f"{control}_equal_JS"] = float(reference.jensen_shannon.mean())
            row[f"improvement_vs_{control}"] = float(
                1.0 - selected.jensen_shannon.mean() / reference.jensen_shannon.mean()
            )
            row[f"systems_better_than_{control}"] = int(
                np.sum(selected.jensen_shannon < reference.jensen_shannon)
            )
        rows.append(row)
    return rows


def parameter_impacts(candidate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter in ("fraction_max", "length_over_R80", "width_over_R80", "shape_gate"):
        grouped = candidate.groupby(parameter).cluster_equal_JS.median().sort_values()
        galaxy_grouped = candidate.groupby(parameter).outer_equal_galaxy_RMSE_km_s.median()
        rows.append(
            {
                "parameter": parameter,
                "best_cluster_level": str(grouped.index[0]),
                "worst_cluster_level": str(grouped.index[-1]),
                "cluster_median_JS_span": float(grouped.iloc[-1] - grouped.iloc[0]),
                "galaxy_median_RMSE_span_km_s": float(
                    galaxy_grouped.max() - galaxy_grouped.min()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_median_JS_span", ascending=False)


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/p0604_cross_domain_routing_balance_protocol.json").read_text()
    )
    specs = candidate_specs(protocol)
    clusters = cluster_scores(protocol, specs)
    galaxies, galaxy_reference = galaxy_scores(protocol, specs)
    tensor_cluster = clusters[
        clusters.target_kind.eq("lenstool_best")
        & clusters.candidate_id.isin(galaxies.candidate_id)
    ]
    candidate = galaxies.merge(
        tensor_cluster.groupby("candidate_id", as_index=False).jensen_shannon.mean().rename(
            columns={"jensen_shannon": "cluster_equal_JS"}
        ),
        on="candidate_id",
        validate="one_to_one",
    )
    folds, oof = cross_validate(protocol, clusters, galaxies, specs)
    oof_summary = summarize_oof(oof)
    impacts = parameter_impacts(candidate)
    primary_budget = float(protocol["validation"]["primary_galaxy_RMSE_ratio_budget"])
    primary = next(
        row
        for row in oof_summary
        if row["galaxy_RMSE_ratio_budget"] == primary_budget
        and row["target_kind"] == "lenstool_best"
    )
    primary_folds = folds[folds.galaxy_RMSE_ratio_budget.eq(primary_budget)]
    report = {
        "report_version": "P0604-CROSS-DOMAIN-ROUTING-BALANCE-RESULTS-0.1.0",
        "status": "complete_cross_domain_budgeted_spent_data_mapping",
        "coverage": {
            "candidates": len(candidate),
            "clusters": clusters.system.nunique(),
            "cluster_targets": clusters.target_kind.nunique(),
            "galaxies": 131,
            "galaxy_outer_points": 968,
            "galaxy_budgets": len(protocol["validation"]["galaxy_RMSE_ratio_budgets"]),
            "folds_per_budget": protocol["validation"]["cluster_folds"],
        },
        "galaxy_reference_fixed_RAR": galaxy_reference,
        "oof_summary": oof_summary,
        "primary_budget": primary_budget,
        "primary_oof": primary,
        "primary_unique_selected_candidates": int(primary_folds.selected_candidate_id.nunique()),
        "primary_fold_selections": primary_folds.to_dict("records"),
        "parameter_impacts": impacts.to_dict("records"),
        "solar_all_candidates_pass_proxy": bool(
            np.all(
                candidate.solar_maximum_absolute_interior_force_change
                <= float(protocol["validation"].get("solar_proxy_gate", 1.0e-8))
            )
        ),
        "solar_maximum_change_across_grid": float(
            candidate.solar_maximum_absolute_interior_force_change.max()
        ),
        "strict_interpretation": {
            "fresh_data": False,
            "whole_cluster_holdouts": True,
            "galaxy_constraint_applied_before_cluster_selection": True,
            "raw_lensing_tested": False,
            "PPN_metric_derived": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    folds.to_csv(output / protocol["outputs"]["fold_selections"], index=False)
    oof.to_csv(output / protocol["outputs"]["oof_scores"], index=False)
    impacts.to_csv(output / protocol["outputs"]["parameter_impacts"], index=False)
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    scatter = axes[0].scatter(
        candidate.galaxy_equal_RMSE_ratio_to_fixed_RAR,
        candidate.cluster_equal_JS,
        c=candidate.length_over_R80,
        cmap="viridis",
        s=20,
        alpha=0.7,
    )
    axes[0].axvline(primary_budget, color="black", linestyle="--")
    axes[0].set(xlabel="galaxy RMSE / fixed RAR", ylabel="cluster equal JS", title="Cross-domain frontier")
    fig.colorbar(scatter, ax=axes[0], label="travel / R80")
    lenstool_rows = [row for row in oof_summary if row["target_kind"] == "lenstool_best"]
    axes[1].plot(
        [row["galaxy_RMSE_ratio_budget"] for row in lenstool_rows],
        [row["tensor_equal_JS"] for row in lenstool_rows],
        marker="o",
        label="selected route",
    )
    axes[1].axhline(primary["LOCAL75_equal_JS"], color="gray", linestyle="--", label="LOCAL75")
    axes[1].axhline(primary["CENTRAL100_equal_JS"], color="black", linestyle="--", label="CENTRAL100")
    axes[1].axhline(primary["W060_equal_JS"], color="green", linestyle="--", label="W060")
    axes[1].set(xlabel="allowed galaxy RMSE ratio", ylabel="OOF cluster JS", title="Cost of galaxy compatibility")
    axes[1].legend(fontsize=7)
    display = impacts.sort_values("cluster_median_JS_span")
    axes[2].barh(display.parameter, display.cluster_median_JS_span, color="#1261A0")
    axes[2].set(xlabel="median cluster-JS span", title="Simplified parameter impact")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0604 cross-domain routing balance\n\n"
        f"At the primary galaxy budget of {primary_budget:.2f}x fixed RAR, OOF Lenstool "
        f"cluster JS is **{primary['tensor_equal_JS']:.5f}**. Improvements are "
        f"**{100 * primary['improvement_vs_LOCAL75']:.2f}%** versus LOCAL75 and "
        f"**{100 * primary['improvement_vs_CENTRAL100']:.2f}%** versus CENTRAL100, "
        f"and **{100 * primary['improvement_vs_W060']:.2f}%** versus W060.\n\n"
        "This maps the tradeoff on spent data and is not fresh confirmation.\n"
    )
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
