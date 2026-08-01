#!/usr/bin/env python3
"""Run the frozen P0559 physical ACCEPT gas-surface-density tensor test."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_cpr0_accept_clash_bcg_stellar import load_clash_bcg_properties  # noqa: E402
from run_member_tidal_metric import (  # noqa: E402
    MemberTidalLens,
    build_contexts,
    fit_context,
    fixed_source_local_rms,
    model_name,
)
from run_p0554_all_baryon_route_screen import prepare_hst_map, prepare_xray_maps  # noqa: E402
from run_p0557_baryon_proxy_tidal import (  # noqa: E402
    build_candidate_context,
    compressed_map_catalog,
    json_safe,
    sha256,
)
from run_unbounded_running_multicluster_raw import aggregate_system_scores  # noqa: E402
from voidscreen.accept_profiles import load_accept_profiles  # noqa: E402
from voidscreen.gas_surface_density import (  # noqa: E402
    annular_morphology_factor,
    enclosed_gas_mass_msun,
    projected_gas_surface_density_msun_kpc2,
)


def rms(values):
    values = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def prepare_registered_maps(protocol, contexts):
    p0557 = json.loads((ROOT / protocol["inputs"]["p0557_protocol"]).read_text())
    all_baryon = json.loads(
        (ROOT / p0557["inputs"]["all_baryon_protocol"]).read_text()
    )
    acquisition = json.loads(
        (ROOT / all_baryon["inputs"]["acquisition_protocol"]).read_text()
    )
    reused = json.loads(
        (ROOT / all_baryon["inputs"]["reused_hst_protocol"]).read_text()
    )
    proxy = p0557["proxy_maps"]
    axis = np.arange(
        float(proxy["axis_min_arcsec"]),
        float(proxy["axis_max_arcsec"]) + 0.5 * float(proxy["grid_spacing_arcsec"]),
        float(proxy["grid_spacing_arcsec"]),
    )
    maps = {}
    for context in contexts:
        label = context.system["label"]
        print(f"P0559 registered maps: {label}", flush=True)
        adapter = SimpleNamespace(label=label, local=context.local_protocol)
        images = pd.concat([context.training, context.heldout], ignore_index=True)
        star, _ = prepare_hst_map(
            all_baryon, acquisition, reused, adapter, images, axis
        )
        _, gas, _ = prepare_xray_maps(all_baryon, acquisition, adapter, axis)
        maps[label] = {"axis": axis, "star": star, "gas": gas}
    return maps


def physical_catalogs(protocol, contexts, registered):
    accept = load_accept_profiles(ROOT / protocol["inputs"]["accept_profiles"])
    name_map = json.loads(
        (ROOT / protocol["inputs"]["accept_name_map_protocol"]).read_text()
    )["cluster_name_map"]
    tian = load_clash_bcg_properties(
        ROOT / protocol["inputs"]["tian_table"]
    ).set_index("cluster")
    gas_rule = protocol["gas_map"]
    block = int(
        json.loads((ROOT / protocol["inputs"]["p0557_protocol"]).read_text())[
            "proxy_maps"
        ]["compression_block_pixels"]
    )
    result = {}
    audits = []
    for context in contexts:
        label = context.system["label"]
        scale = float(
            context.local_protocol["cosmology_and_coordinates"][
                "angular_scale_kpc_per_arcsec"
            ]
        )
        axis = registered[label]["axis"]
        xx, yy = np.meshgrid(axis, axis, indexing="xy")
        radius_kpc = np.hypot(xx, yy) * scale
        profile = accept[accept.name.eq(name_map[label])].sort_values("rin_mpc")
        inner = profile.rin_mpc.to_numpy(float) * 1000.0
        outer = profile.rout_mpc.to_numpy(float) * 1000.0
        ne = profile.nelec_cm3.to_numpy(float)
        sigma = projected_gas_surface_density_msun_kpc2(
            radius_kpc, inner, outer, ne, mu_e=float(gas_rule["mu_e"])
        )
        pixel_area = scale**2
        spherical_gas = sigma * pixel_area
        star_image = np.maximum(registered[label]["star"], 0.0)
        stellar_mass = float(tian.loc[label, "stellar_mass_1e11_msun"]) * 1.0e11
        star_mass = star_image / float(star_image.sum()) * stellar_mass
        anchor_radius = float(tian.loc[label, "central_radius_kpc"])
        accept_anchor = enclosed_gas_mass_msun(
            anchor_radius, inner, outer, ne, mu_e=float(gas_rule["mu_e"])
        )
        tian_anchor = float(tian.loc[label, "gas_mass_1e11_msun"]) * 1.0e11
        anchor_scale = tian_anchor / accept_anchor
        factors = {
            power: annular_morphology_factor(
                axis,
                registered[label]["gas"],
                power=float(power),
                smoothing_sigma_arcsec=float(gas_rule["smoothing_sigma_arcsec"]),
                contrast_min=float(gas_rule["contrast_min"]),
                contrast_max=float(gas_rule["contrast_max"]),
            )
            for power in {0.0, 0.5, 1.0}
        }
        maps = {}
        for normalization, multiplier in (
            ("accept_absolute", 1.0),
            ("renormalize_accept_to_tian_spherical_anchor", anchor_scale),
        ):
            for power, factor in factors.items():
                gas_mass = spherical_gas * multiplier * factor
                maps[(normalization, power, True)] = star_mass + gas_mass
                maps[(normalization, power, False)] = gas_mass
        result[label] = {
            key: compressed_map_catalog(axis, image, block_pixels=block, transform="linear")
            for key, image in maps.items()
        }
        audits.append(
            {
                "system_label": label,
                "accept_shells": len(profile),
                "accept_min_kpc": float(inner.min()),
                "accept_max_kpc": float(outer.max()),
                "Tian_anchor_radius_kpc": anchor_radius,
                "ACCEPT_spherical_gas_mass_at_anchor_msun": accept_anchor,
                "Tian_gas_mass_at_anchor_msun": tian_anchor,
                "ACCEPT_to_Tian_anchor_mass_ratio": accept_anchor / tian_anchor,
                "Tian_anchor_renormalization_factor": anchor_scale,
                "projected_ACCEPT_gas_mass_on_map_msun": float(spherical_gas.sum()),
                "stellar_mass_assigned_to_map_msun": stellar_mass,
                "absolute_projected_map_gas_fraction": float(
                    spherical_gas.sum() / (spherical_gas.sum() + stellar_mass)
                ),
                "sqrt_factor_min": float(factors[0.5].min()),
                "sqrt_factor_max": float(factors[0.5].max()),
            }
        )
    return result, pd.DataFrame(audits)


def key_for(model):
    return (
        model["gas_normalization"],
        float(model["gas_power"]),
        True,
    )


def main():
    config_path = ROOT / "configs/p0559_accept_projected_gas_tidal_protocol.json"
    protocol = json.loads(config_path.read_text())
    if not protocol["status"].startswith("frozen_before_any_"):
        raise RuntimeError("P0559 is not frozen before scoring")
    p0557 = json.loads((ROOT / protocol["inputs"]["p0557_protocol"]).read_text())
    member = json.loads(
        (ROOT / protocol["inputs"]["member_tidal_protocol"]).read_text()
    )
    member["optimization"]["maximum_function_evaluations"] = int(
        protocol["optimization"]["maximum_function_evaluations"]
    )
    contexts, _, input_hashes = build_contexts(
        member, softening_kpc=float(protocol["locked_field"]["softening_kpc"])
    )
    registered = prepare_registered_maps(protocol, contexts)
    catalogs, physical_audits = physical_catalogs(protocol, contexts, registered)
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    physical_audits.to_csv(
        output / protocol["outputs"]["physical_map_audits"], index=False
    )
    operator = {"operator_id": "contrast", "subtract_circular_mean": True}
    audit_rows = []
    score_rows = []
    predictions = []
    fit_cache = {}
    aggregates = {}
    starts = int(protocol["optimization"]["starts_per_exact_fit"])
    seed = int(protocol["optimization"]["random_seed"])
    for model_index, model in enumerate(protocol["exact_models"]):
        model_id = model["model_id"]
        coupling = float(model["tensor_t"])
        heldout_scores = []
        subset_scores = []
        for system_index, base in enumerate(contexts):
            label = base.system["label"]
            if coupling == 0.0:
                context = base
            else:
                context = build_candidate_context(
                    base,
                    catalogs[label][key_for(model)],
                    p0557,
                    model_id,
                    operator,
                    pixels_per_axis=int(protocol["locked_field"]["pixels_per_axis"]),
                    softening_kpc=float(protocol["locked_field"]["softening_kpc"]),
                    audit_rows=audit_rows,
                    stage="p0559_exact",
                )
            print(f"P0559 exact: {label} {model_id}", flush=True)
            fitted = fit_context(
                context,
                coupling,
                starts=starts,
                seed=seed + model_index * 100 + system_index,
            )
            fit_cache[(model_id, label)] = (context, fitted)
            heldout_scores.append(fitted["heldout"])
            if label in {"MACS1115", "MACS1931"}:
                subset_scores.append(fitted["heldout"])
            predictions.extend(
                [fitted["training_predictions"], fitted["heldout_predictions"]]
            )
            score_rows.append(
                {
                    "row_type": "system",
                    "model_id": model_id,
                    "system_label": label,
                    "heldout_exact_RMS_arcsec": fitted["heldout"][
                        "exact_radial_RMS_arcsec"
                    ],
                    "all_heldout_roots": fitted["heldout"]["all_roots_converged"],
                }
            )
        aggregates[model_id] = {
            "all_four": aggregate_system_scores(heldout_scores),
            "validation_subset": aggregate_system_scores(subset_scores),
        }
        for scope in ("all_four", "validation_subset"):
            item = aggregates[model_id][scope]
            score_rows.append(
                {
                    "row_type": "aggregate",
                    "model_id": model_id,
                    "system_label": scope,
                    "heldout_exact_RMS_arcsec": item[
                        "equal_system_radial_RMS_arcsec"
                    ],
                    "all_heldout_roots": item["all_roots_converged"],
                }
            )
    scores = pd.DataFrame(score_rows)
    zero = scores[scores.model_id.eq("zero")].set_index("system_label")
    scores["improvement_fraction_vs_zero"] = [
        1.0
        - float(row.heldout_exact_RMS_arcsec)
        / float(zero.loc[row.system_label, "heldout_exact_RMS_arcsec"])
        if np.isfinite(float(row.heldout_exact_RMS_arcsec))
        else -np.inf
        for row in scores.itertuples(index=False)
    ]
    scores.to_csv(output / protocol["outputs"]["scores"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )
    pd.DataFrame(audit_rows).to_csv(
        output / protocol["outputs"]["tensor_audits"], index=False
    )
    diagnostics = []
    primary_id = "accept_absolute_sqrt"
    coupling = float(protocol["locked_field"]["tensor_t"])
    for diagnostic in protocol["fixed_fit_diagnostics"]:
        per_system = {}
        for base in contexts:
            label = base.system["label"]
            key = (
                diagnostic["gas_normalization"],
                float(diagnostic["gas_power"]),
                bool(diagnostic["include_stars"]),
            )
            context = build_candidate_context(
                base,
                catalogs[label][key],
                p0557,
                diagnostic["diagnostic_id"],
                operator,
                pixels_per_axis=int(protocol["locked_field"]["pixels_per_axis"]),
                softening_kpc=float(protocol["locked_field"]["softening_kpc"]),
                audit_rows=audit_rows,
                stage="fixed_fit_diagnostic",
            )
            _, fitted = fit_cache[(primary_id, label)]
            lens = MemberTidalLens(
                context.local_protocol, context.fields, context.correction, coupling
            )
            per_system[label] = fixed_source_local_rms(
                lens,
                model_name(coupling),
                fitted["fit"]["result"].x,
                fitted["fit"]["sources"],
                context.heldout,
            )
        diagnostics.append(
            {
                "diagnostic_id": diagnostic["diagnostic_id"],
                "equal_system_fixed_fit_local_RMS_arcsec": rms(per_system.values()),
                **{f"{key}_local_RMS_arcsec": value for key, value in per_system.items()},
            }
        )
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame.to_csv(
        output / protocol["outputs"]["fixed_fit_diagnostics"], index=False
    )
    pd.DataFrame(audit_rows).to_csv(
        output / protocol["outputs"]["tensor_audits"], index=False
    )
    metric = json.loads(
        (ROOT / protocol["inputs"]["metric_slip_report"]).read_text()
    )
    halo = float(
        metric["comparators"]["compact_halo_validation"][
            "equal_system_radial_RMS_arcsec"
        ]
    )
    primary = aggregates[primary_id]
    zero_aggregate = aggregates["zero"]
    primary_improvement = 1.0 - float(
        primary["all_four"]["equal_system_radial_RMS_arcsec"]
    ) / float(zero_aggregate["all_four"]["equal_system_radial_RMS_arcsec"])
    system_scores = scores[scores.row_type.eq("system")]
    primary_system = system_scores[system_scores.model_id.eq(primary_id)].set_index(
        "system_label"
    )
    zero_system = system_scores[system_scores.model_id.eq("zero")].set_index(
        "system_label"
    )
    all_improve = bool(
        (primary_system.heldout_exact_RMS_arcsec < zero_system.heldout_exact_RMS_arcsec).all()
    )
    ratios = physical_audits.ACCEPT_to_Tian_anchor_mass_ratio.to_numpy(float)
    ratio_lo, ratio_hi = map(float, protocol["advance_gates"]["accept_to_tian_anchor_mass_ratio_range"])
    gate_audit = {
        "absolute_primary_all_roots": bool(primary["all_four"]["all_roots_converged"]),
        "absolute_primary_all_systems_improve": all_improve,
        "absolute_primary_equal_system_improvement_fraction": primary_improvement,
        "absolute_primary_improvement_pass": primary_improvement
        >= float(protocol["advance_gates"]["absolute_primary_equal_system_improvement_fraction_min"]),
        "absolute_primary_validation_subset_to_compact_halo_ratio": float(
            primary["validation_subset"]["equal_system_radial_RMS_arcsec"]
        )
        / halo,
        "compact_halo_ratio_pass": float(
            primary["validation_subset"]["equal_system_radial_RMS_arcsec"]
        )
        / halo
        <= float(protocol["advance_gates"]["absolute_primary_validation_subset_to_compact_halo_ratio_max"]),
        "accept_to_tian_anchor_mass_compatibility_pass": bool(
            np.all((ratios >= ratio_lo) & (ratios <= ratio_hi))
        ),
    }
    report = {
        "report_version": "P0559-ACCEPT-PROJECTED-GAS-TIDAL-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "input_hashes": input_hashes,
        "physical_map_audits": physical_audits.to_dict("records"),
        "scores": scores.to_dict("records"),
        "fixed_fit_diagnostics": diagnostic_frame.to_dict("records"),
        "primary": {
            "zero_all_four_RMS_arcsec": zero_aggregate["all_four"]["equal_system_radial_RMS_arcsec"],
            "absolute_ACCEPT_all_four_RMS_arcsec": primary["all_four"]["equal_system_radial_RMS_arcsec"],
            "improvement_fraction": primary_improvement,
            "all_systems_improve": all_improve,
            "all_roots": primary["all_four"]["all_roots_converged"],
        },
        "comparators": {
            "absolute_ACCEPT_validation_subset_RMS_arcsec": primary["validation_subset"]["equal_system_radial_RMS_arcsec"],
            "compact_halo_RMS_arcsec": halo,
            "ratio": float(primary["validation_subset"]["equal_system_radial_RMS_arcsec"]) / halo,
        },
        "gate_audit": gate_audit,
        "verdict": {
            "all_advancement_gates_pass": bool(
                gate_audit["absolute_primary_all_roots"]
                and gate_audit["absolute_primary_all_systems_improve"]
                and gate_audit["absolute_primary_improvement_pass"]
                and gate_audit["compact_halo_ratio_pass"]
                and gate_audit["accept_to_tian_anchor_mass_compatibility_pass"]
            ),
            "formula_promoted": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    aggregate_plot = scores[
        scores.row_type.eq("aggregate") & scores.system_label.eq("all_four")
    ].sort_values("heldout_exact_RMS_arcsec")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].barh(aggregate_plot.model_id, aggregate_plot.heldout_exact_RMS_arcsec)
    axes[0].set(xlabel="four-cluster held-out exact RMS (arcsec)", title="Projected ACCEPT gas tensor")
    axes[1].bar(physical_audits.system_label, physical_audits.ACCEPT_to_Tian_anchor_mass_ratio)
    axes[1].axhspan(ratio_lo, ratio_hi, alpha=0.15, color="green")
    axes[1].set(ylabel="ACCEPT / Tian gas mass at anchor", title="Independent gas-mass cross-check")
    axes[1].tick_params(axis="x", rotation=30)
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    (output / protocol["outputs"]["summary"]).write_text(
        f"""# P0559 projected ACCEPT gas tensor

The absolute ACCEPT primary changes four-cluster held-out RMS from
{float(zero_aggregate['all_four']['equal_system_radial_RMS_arcsec']):.3f} to
{float(primary['all_four']['equal_system_radial_RMS_arcsec']):.3f} arcsec
({100.0 * primary_improvement:+.3f}%). All roots pass:
{bool(primary['all_four']['all_roots_converged'])}. The ACCEPT/Tian central
mass compatibility gate passes: {gate_audit['accept_to_tian_anchor_mass_compatibility_pass']}.
No formula is promoted.
""",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report["primary"]), indent=2), flush=True)
    print(json.dumps(json_safe(report["verdict"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
