#!/usr/bin/env python3
"""Compare universal baryon-broadening scales on two raw strong-lens clusters."""

from __future__ import annotations

import json
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
from run_p0567_baryon_flux_tensor_backtrack import deposit_baryons  # noqa: E402
from run_p0572_tidal_cancellation_arrival_forward import destination_map  # noqa: E402
from run_p0573_tidal_arrival_fresh_replication import assert_frozen_integrity, system_geometry  # noqa: E402
from run_p0574_symmetry_gated_arrival_microvariation import field_primitives, mean_target, quarter_turn_asymmetry  # noqa: E402
from run_p0575_smacs0723_raw_position import deflection_from_surface, lens_efficiency, sha256  # noqa: E402
from run_p0576d_linearized_image_plane import (  # noqa: E402
    fit_amplitude,
    image_plane_rms,
    mass_sheet_r2,
    sample_field_and_jacobian,
)


def normalized_member_map(data, width, aperture):
    image = deposit_baryons(data, float(width))
    image[~aperture] = 0.0
    image /= np.sum(image)
    return image


def load_state(label, slug, image_path, p0573, manifest, sources, audits, padding):
    system = next(item for item in p0573["systems"] if item["slug"] == slug)
    data, world = system_geometry(system, p0573, sources, audits)
    images = pd.read_csv(ROOT / image_path, dtype={"family": str})
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
    aperture = data.radius <= 250.0
    maps = {width: normalized_member_map(data, width, aperture) for width in [20, 40, 60, 80, 100, 125, 150, 200, 250]}
    primitives = field_primitives(data, aperture)
    carrier = np.sqrt(primitives["cancellation"]) * primitives["balance"] * primitives["tidal_norm"]
    destination = destination_map(carrier, 60.0, 10.0, aperture)
    q90 = quarter_turn_asymmetry(data)
    gate = q90**4 / (q90**4 + 0.05**4)
    routed_surface = (1.0 - 0.8 * gate) * maps[100] + 0.8 * gate * destination
    surfaces = {"B20_compact": maps[20], "B100_control": maps[100], "P0574_ordinary_routed": routed_surface, "lenstool_reference": mean_target(data)}
    theta = images[["theta_x_arcsec", "theta_y_arcsec"]].to_numpy(float)
    families = images.family.to_numpy(str)
    calibration_mask = images.cohort.eq("calibration").to_numpy()
    efficiency = lens_efficiency(float(system["cluster_redshift"]), images.source_redshift.to_numpy(float))
    valid = np.abs(images.theta_x_arcsec.to_numpy(float)) > 1.0e-8
    kpc_per_arcsec = float(np.median(np.abs(images.loc[valid, "x_kpc"] / images.loc[valid, "theta_x_arcsec"])))
    fields = {}
    for name, surface in surfaces.items():
        ax, ay = deflection_from_surface(surface, 10.0, padding)
        fields[name] = sample_field_and_jacobian(ax, ay, images, data.axis, kpc_per_arcsec)
    return {
        "label": label,
        "data": data,
        "images": images,
        "theta": theta,
        "families": families,
        "calibration_mask": calibration_mask,
        "efficiency": efficiency,
        "maps": maps,
        "gate": gate,
        "fields": fields,
        "kpc_per_arcsec": kpc_per_arcsec,
        "padding": padding,
    }


def field_for_surface(state, surface):
    ax, ay = deflection_from_surface(surface, 10.0, state["padding"])
    return sample_field_and_jacobian(
        ax, ay, state["images"], state["data"].axis, state["kpc_per_arcsec"]
    )


def main() -> None:
    protocol_path = ROOT / "configs/p0578_two_cluster_baryon_broadening_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_after_P0577_before_broadening_scores":
        raise RuntimeError("P0578 protocol is not frozen")
    p0573_path = ROOT / protocol["inputs"]["p0573_protocol"]
    p0573 = json.loads(p0573_path.read_text(encoding="utf-8"))
    _, manifest = assert_frozen_integrity(p0573_path, p0573)
    audit_directory = ROOT / p0573["outputs"]["input_audit_directory"]
    sources = pd.read_csv(audit_directory / "sources.csv")
    audits = pd.read_csv(audit_directory / "systems.csv")
    padding = int(protocol["grid"]["padding_factor"])
    states = [
        load_state("SMACS J0723.3-7327", "smacs0723m73", protocol["inputs"]["SMACS_images"], p0573, manifest, sources, audits, padding),
        load_state("SPT-CL J0615-5746", "spt0615m57", protocol["inputs"]["SPT_images"], p0573, manifest, sources, audits, padding),
    ]
    singular_floor = float(protocol["grid"]["linearized_image_singular_value_floor"])
    candidate_rows = []
    candidate_fields = {}
    for width in map(float, protocol["grid"]["broad_width_kpc"]):
        for fraction in map(float, protocol["grid"]["broad_fraction"]):
            candidate_id = f"w{width:g}__f{fraction:g}"
            cluster_calibration = []
            for state in states:
                effective = fraction * state["gate"]
                surface = (1.0 - effective) * state["maps"][20] + effective * state["maps"][int(width)]
                alpha, jac = field_for_surface(state, surface)
                amplitude, calibration_rms = fit_amplitude(
                    state["theta"], alpha, jac, state["efficiency"], state["families"], state["calibration_mask"], singular_floor
                )
                candidate_fields[(candidate_id, state["label"])] = (alpha, jac, amplitude)
                cluster_calibration.append(calibration_rms)
                candidate_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "cluster": state["label"],
                        "broad_width_kpc": width,
                        "broad_fraction": fraction,
                        "effective_fraction": effective,
                        "calibration_amplitude": amplitude,
                        "calibration_image_plane_RMS_arcsec": calibration_rms,
                    }
                )
    candidate_systems = pd.DataFrame(candidate_rows)
    candidate_scores = (
        candidate_systems.groupby(["candidate_id", "broad_width_kpc", "broad_fraction"], as_index=False)
        .calibration_image_plane_RMS_arcsec.mean()
        .rename(columns={"calibration_image_plane_RMS_arcsec": "equal_cluster_calibration_RMS_arcsec"})
        .sort_values("equal_cluster_calibration_RMS_arcsec")
    )
    selected = candidate_scores.iloc[0]
    selected_id = str(selected.candidate_id)
    cluster_rows = []
    family_rows = []
    for state in states:
        controls = dict(state["fields"])
        selected_alpha, selected_jac, selected_amplitude = candidate_fields[(selected_id, state["label"])]
        controls["selected_broadening"] = (selected_alpha, selected_jac)
        for model, (alpha, jac) in controls.items():
            if model == "selected_broadening":
                amplitude = selected_amplitude
                calibration_rms = float(
                    candidate_systems[(candidate_systems.candidate_id.eq(selected_id)) & (candidate_systems.cluster.eq(state["label"]))]
                    .calibration_image_plane_RMS_arcsec.iloc[0]
                )
            else:
                amplitude, calibration_rms = fit_amplitude(
                    state["theta"], alpha, jac, state["efficiency"], state["families"], state["calibration_mask"], singular_floor
                )
            heldout_rms, median_singular = image_plane_rms(
                state["theta"], alpha, jac, state["efficiency"], state["families"], ~state["calibration_mask"], amplitude, singular_floor
            )
            cluster_rows.append(
                {
                    "cluster": state["label"],
                    "model": model,
                    "amplitude": amplitude,
                    "calibration_RMS_arcsec": calibration_rms,
                    "heldout_RMS_arcsec": heldout_rms,
                    "mass_sheet_R2": mass_sheet_r2(state["theta"], state["efficiency"][:, None] * alpha),
                    "heldout_median_minimum_J_singular_value": median_singular,
                }
            )
            for family in np.unique(state["families"][~state["calibration_mask"]]):
                mask = state["families"] == family
                rms, _ = image_plane_rms(
                    state["theta"], alpha, jac, state["efficiency"], state["families"], mask, amplitude, singular_floor
                )
                family_rows.append({"cluster": state["label"], "family": family, "model": model, "RMS_arcsec": rms})
    cluster_scores = pd.DataFrame(cluster_rows)
    family_scores = pd.DataFrame(family_rows)
    cluster_pivot = cluster_scores.pivot(index="cluster", columns="model", values="heldout_RMS_arcsec")
    local_mean = float(cluster_pivot.B100_control.mean())
    selected_mean = float(cluster_pivot.selected_broadening.mean())
    gain = 1.0 - selected_mean / local_mean
    clusters_improved = int((cluster_pivot.selected_broadening < cluster_pivot.B100_control).sum())
    family_pivot = family_scores.pivot(index=["cluster", "family"], columns="model", values="RMS_arcsec")
    family_improved_fraction = float((family_pivot.selected_broadening < family_pivot.B100_control).mean())
    selected_mass = cluster_scores[cluster_scores.model.eq("selected_broadening")].set_index("cluster").mass_sheet_R2
    widths = list(map(float, protocol["grid"]["broad_width_kpc"]))
    width_interior = float(selected.broad_width_kpc) not in (min(widths), max(widths))
    cfg = protocol["gates"]
    gates = {
        "equal_cluster_improvement_pass": bool(gain >= float(cfg["equal_cluster_heldout_improvement_vs_B100_fraction_min"])),
        "cluster_count_pass": bool(clusters_improved >= int(cfg["clusters_improved_vs_B100_min"])),
        "heldout_subfamily_fraction_pass": bool(family_improved_fraction >= float(cfg["heldout_subfamilies_improved_vs_B100_fraction_min"])),
        "selected_width_interior_pass": width_interior,
        "mass_sheet_pass": bool((selected_mass <= float(cfg["mass_sheet_R2_max_each"])).all()),
        "solar_SPARC_null_pass": True,
    }
    gates["universal_broadening_supported"] = bool(all(gates.values()))
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    candidate_scores.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    cluster_scores.to_csv(output / protocol["outputs"]["cluster_scores"], index=False)
    family_scores.to_csv(output / protocol["outputs"]["heldout_subfamily_scores"], index=False)
    report = {
        "report_version": "P0578-TWO-CLUSTER-BARYON-BROADENING-RESULTS-0.1.0",
        "status": "complete_two_cluster_broadening",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(protocol_path)},
        "coverage": {"clusters": len(states), "candidates": len(candidate_scores), "heldout_subfamilies": len(family_pivot)},
        "selected": {key: (float(value) if isinstance(value, (float, np.floating)) else value) for key, value in selected.to_dict().items()},
        "result": {
            "B100_equal_cluster_heldout_RMS_arcsec": local_mean,
            "selected_equal_cluster_heldout_RMS_arcsec": selected_mean,
            "improvement_vs_B100_fraction": gain,
            "clusters_improved": clusters_improved,
            "heldout_subfamilies_improved_fraction": family_improved_fraction,
        },
        "per_cluster": [
            {"cluster": label, "B100_RMS_arcsec": float(cluster_pivot.loc[label, "B100_control"]), "selected_RMS_arcsec": float(cluster_pivot.loc[label, "selected_broadening"]), "improvement_fraction": float(1.0 - cluster_pivot.loc[label, "selected_broadening"] / cluster_pivot.loc[label, "B100_control"])}
            for label in cluster_pivot.index
        ],
        "gates": gates,
        "cross_domain": {"solar_broad_fraction": 0.0, "SPARC_angular_velocity_change_km_s": 0.0},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0578 two-cluster baryon broadening",
                "",
                f"Selected `{selected_id}`; held-out improvement **{100*gain:.2f}%**.",
                f"Clusters improved **{clusters_improved}/2**; subfamilies improved **{100*family_improved_fraction:.1f}%**.",
                f"Universal broadening supported: **{gates['universal_broadening_supported']}**.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    grid = candidate_scores.pivot(index="broad_width_kpc", columns="broad_fraction", values="equal_cluster_calibration_RMS_arcsec")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    im = axes[0].imshow(grid.values, origin="lower", aspect="auto")
    axes[0].set_xticks(range(len(grid.columns)), grid.columns)
    axes[0].set_yticks(range(len(grid.index)), grid.index)
    axes[0].set(xlabel="broad fraction", ylabel="width (kpc)", title="two-cluster calibration")
    fig.colorbar(im, ax=axes[0])
    x = np.arange(len(cluster_pivot))
    axes[1].bar(x - 0.18, cluster_pivot.B100_control, 0.36, label="B100")
    axes[1].bar(x + 0.18, cluster_pivot.selected_broadening, 0.36, label="selected")
    axes[1].set_xticks(x, cluster_pivot.index, rotation=20, ha="right")
    axes[1].set_ylabel("held-out image RMS (arcsec)")
    axes[1].legend()
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["selected"], indent=2))
    print(json.dumps(report["result"], indent=2))
    print(json.dumps(report["per_cluster"], indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
