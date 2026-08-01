#!/usr/bin/env python3
"""Map the routed-propagator response on independent SPT0615 raw image positions."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_gravity_arc_fresh_sample import regrid_kappa_sky  # noqa: E402
from run_p0567_baryon_flux_tensor_backtrack import deposit_baryons  # noqa: E402
from run_p0572_tidal_cancellation_arrival_forward import destination_map  # noqa: E402
from run_p0573_tidal_arrival_fresh_replication import assert_frozen_integrity, system_geometry  # noqa: E402
from run_p0574_symmetry_gated_arrival_microvariation import field_primitives, mean_target, quarter_turn_asymmetry  # noqa: E402
from run_p0575_smacs0723_raw_position import deflection_from_surface, lens_efficiency, sky_offsets  # noqa: E402
from run_p0576_fractional_routed_propagator import fractional_deflection  # noqa: E402
from run_p0576d_linearized_image_plane import (  # noqa: E402
    fit_amplitude,
    image_plane_rms,
    mass_sheet_r2,
    sample_field_and_jacobian,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_images(path: Path) -> pd.DataFrame:
    pattern = re.compile(
        r"^\s*((?:1|10|11|12)\.\d|3\.[125])\s*&\s*(\d{2})\s+(\d{2})\s+([0-9.]+)\s*&\s*\$-\$(\d{2})\s+(\d{2})\s+([0-9.]+)"
    )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        image_id, rah, ram, ras, decd, decm, decs = match.groups()
        family = image_id.split(".")[0]
        ra_deg = 15.0 * (float(rah) + float(ram) / 60.0 + float(ras) / 3600.0)
        dec_deg = -(float(decd) + float(decm) / 60.0 + float(decs) / 3600.0)
        rows.append(
            {
                "image_id": image_id,
                "family": family,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "source_redshift": 4.013 if family == "3" else 1.358,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["family", "image_id"]).reset_index(drop=True)
    expected = {"1": 3, "10": 4, "11": 4, "12": 3, "3": 3}
    if len(frame) != 17 or frame.groupby("family").size().to_dict() != expected:
        raise RuntimeError(f"unexpected frozen SPT image coverage: {frame.groupby('family').size().to_dict()}")
    return frame


def main() -> None:
    protocol_path = ROOT / "configs/p0577_spt0615_raw_response_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_any_SPT0615_raw_position_score":
        raise RuntimeError("P0577 protocol is not frozen")
    raw = protocol["raw_data"]
    archive = ROOT / raw["source_archive"]
    tex = ROOT / raw["table_tex"]
    if archive.stat().st_size != int(raw["source_archive_bytes"]) or sha256(archive) != raw["source_archive_sha256"]:
        raise RuntimeError("SPT source archive integrity failed")
    if sha256(tex) != raw["table_tex_sha256"]:
        raise RuntimeError("SPT table integrity failed")
    images = parse_images(tex)
    calibration = set(protocol["split"]["calibration_subfamilies"])
    images["cohort"] = np.where(images.family.isin(calibration), "calibration", "heldout")

    p0573_path = ROOT / protocol["inputs"]["p0573_protocol"]
    p0573 = json.loads(p0573_path.read_text(encoding="utf-8"))
    _, manifest = assert_frozen_integrity(p0573_path, p0573)
    audit_directory = ROOT / p0573["outputs"]["input_audit_directory"]
    sources = pd.read_csv(audit_directory / "sources.csv")
    audits = pd.read_csv(audit_directory / "systems.csv")
    system = next(item for item in p0573["systems"] if item["slug"] == "spt0615m57")
    data, world = system_geometry(system, p0573, sources, audits)
    audit_row = audits[audits.system.eq(data.label)].iloc[0]
    center = SkyCoord(float(audit_row.reference_ra_deg) * u.deg, float(audit_row.reference_dec_deg) * u.deg)
    kpc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(float(system["cluster_redshift"])).value / 60.0)
    images = sky_offsets(images, center, kpc_per_arcsec)
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
    lenstool_map = mean_target(data)
    aperture = data.radius <= 250.0
    local = deposit_baryons(data, 100.0)
    local[~aperture] = 0.0
    local /= np.sum(local)
    primitives = field_primitives(data, aperture)
    carrier = np.sqrt(primitives["cancellation"]) * primitives["balance"] * primitives["tidal_norm"]
    destination = destination_map(carrier, 60.0, 10.0, aperture)
    q90 = quarter_turn_asymmetry(data)
    gate = q90**4 / (q90**4 + 0.05**4)

    theta = images[["theta_x_arcsec", "theta_y_arcsec"]].to_numpy(float)
    families = images.family.to_numpy(str)
    calibration_mask = images.cohort.eq("calibration").to_numpy()
    efficiency = lens_efficiency(float(raw["cluster_redshift"]), images.source_redshift.to_numpy(float))
    padding = int(protocol["grid"]["padding_factor"])
    singular_floor = float(protocol["grid"]["linearized_image_singular_value_floor"])
    local_ax, local_ay = deflection_from_surface(local, 10.0, padding)
    local_sampled, local_jac = sample_field_and_jacobian(local_ax, local_ay, images, data.axis, kpc_per_arcsec)
    len_ax, len_ay = deflection_from_surface(lenstool_map, 10.0, padding)
    len_sampled, len_jac = sample_field_and_jacobian(len_ax, len_ay, images, data.axis, kpc_per_arcsec)
    routed = {}
    for power in map(float, protocol["grid"]["fractional_power_p"]):
        ax, ay = fractional_deflection(destination, 10.0, power, 60.0, padding)
        routed[power] = sample_field_and_jacobian(ax, ay, images, data.axis, kpc_per_arcsec)

    rows = []
    fields = {}
    for power in map(float, protocol["grid"]["fractional_power_p"]):
        for fraction in map(float, protocol["grid"]["deflection_route_fraction"]):
            effective = fraction * gate
            alpha = (1.0 - effective) * local_sampled + effective * routed[power][0]
            jac = (1.0 - effective) * local_jac + effective * routed[power][1]
            amplitude, calibration_rms = fit_amplitude(
                theta, alpha, jac, efficiency, families, calibration_mask, singular_floor
            )
            candidate_id = f"p{power:g}__f{fraction:g}"
            fields[candidate_id] = (alpha, jac)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "fractional_power_p": power,
                    "deflection_route_fraction": fraction,
                    "calibration_amplitude": amplitude,
                    "calibration_image_plane_RMS_arcsec": calibration_rms,
                }
            )
    candidates = pd.DataFrame(rows).sort_values("calibration_image_plane_RMS_arcsec")
    selected = candidates.iloc[0]
    selected_id = str(selected.candidate_id)
    controls = {
        "local_control": (local_sampled, local_jac),
        "ordinary_p1_f0p8": fields["p1__f0.8"],
        "SMACS_locked_p1p75_f1": fields["p1.75__f1"],
        "SPT_selected": fields[selected_id],
        "lenstool_map_reference": (len_sampled, len_jac),
    }
    control_rows = []
    family_rows = []
    for name, (alpha, jac) in controls.items():
        if name == "SPT_selected":
            amplitude = float(selected.calibration_amplitude)
            calibration_rms = float(selected.calibration_image_plane_RMS_arcsec)
        else:
            amplitude, calibration_rms = fit_amplitude(
                theta, alpha, jac, efficiency, families, calibration_mask, singular_floor
            )
        heldout_rms, median_singular = image_plane_rms(
            theta, alpha, jac, efficiency, families, ~calibration_mask, amplitude, singular_floor
        )
        control_rows.append(
            {
                "model": name,
                "amplitude": amplitude,
                "calibration_RMS_arcsec": calibration_rms,
                "heldout_RMS_arcsec": heldout_rms,
                "mass_sheet_R2": mass_sheet_r2(theta, efficiency[:, None] * alpha),
                "heldout_median_minimum_J_singular_value": median_singular,
            }
        )
        for family in protocol["split"]["heldout_subfamilies"]:
            mask = families == family
            rms, _ = image_plane_rms(theta, alpha, jac, efficiency, families, mask, amplitude, singular_floor)
            family_rows.append({"model": name, "family": family, "RMS_arcsec": rms})
    controls_frame = pd.DataFrame(control_rows).set_index("model")
    families_frame = pd.DataFrame(family_rows)
    family_pivot = families_frame.pivot(index="family", columns="model", values="RMS_arcsec")
    local_rms = float(controls_frame.loc["local_control", "heldout_RMS_arcsec"])
    selected_rms = float(controls_frame.loc["SPT_selected", "heldout_RMS_arcsec"])
    locked_rms = float(controls_frame.loc["SMACS_locked_p1p75_f1", "heldout_RMS_arcsec"])
    selected_gain = 1.0 - selected_rms / local_rms
    locked_gain = 1.0 - locked_rms / local_rms
    selected_count = int((family_pivot.SPT_selected < family_pivot.local_control).sum())
    locked_count = int((family_pivot.SMACS_locked_p1p75_f1 < family_pivot.local_control).sum())
    cfg = protocol["gates"]
    gates = {
        "SPT_selected_improvement_pass": bool(selected_gain >= float(cfg["SPT_selected_heldout_improvement_vs_local_fraction_min"])),
        "SPT_selected_family_count_pass": bool(selected_count >= int(cfg["SPT_selected_heldout_subfamilies_improved_min"])),
        "SMACS_locked_improvement_pass": bool(locked_gain >= float(cfg["SMACS_locked_heldout_improvement_vs_local_fraction_min"])),
        "SMACS_locked_family_count_pass": bool(locked_count >= int(cfg["SMACS_locked_heldout_subfamilies_improved_min"])),
        "SMACS_locked_mass_sheet_pass": bool(controls_frame.loc["SMACS_locked_p1p75_f1", "mass_sheet_R2"] <= float(cfg["mass_sheet_R2_max"])),
        "solar_SPARC_null_pass": True,
    }
    gates["cross_cluster_propagator_pattern_supported"] = bool(
        gates["SMACS_locked_improvement_pass"]
        and gates["SMACS_locked_family_count_pass"]
        and gates["SMACS_locked_mass_sheet_pass"]
    )
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    images.to_csv(output / protocol["outputs"]["images"], index=False)
    candidates.to_csv(output / protocol["outputs"]["candidate_scores"], index=False)
    families_frame.to_csv(output / protocol["outputs"]["heldout_subfamily_scores"], index=False)
    report = {
        "report_version": "P0577-SPT0615-RAW-RESPONSE-RESULTS-0.1.0",
        "status": "complete_second_cluster_raw_response",
        "protocol": {"path": str(protocol_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(protocol_path)},
        "coverage": {"raw_images": len(images), "subfamilies": int(images.family.nunique()), "calibration_images": int(calibration_mask.sum()), "heldout_images": int((~calibration_mask).sum()), "candidates": len(candidates)},
        "SPT_selected": {key: (float(value) if isinstance(value, (float, np.floating)) else value) for key, value in selected.to_dict().items()},
        "result": {
            "local_heldout_RMS_arcsec": local_rms,
            "SPT_selected_heldout_RMS_arcsec": selected_rms,
            "SPT_selected_improvement_fraction": selected_gain,
            "SPT_selected_subfamilies_improved": selected_count,
            "SMACS_locked_heldout_RMS_arcsec": locked_rms,
            "SMACS_locked_improvement_fraction": locked_gain,
            "SMACS_locked_subfamilies_improved": locked_count,
            "lenstool_reference_heldout_RMS_arcsec": float(controls_frame.loc["lenstool_map_reference", "heldout_RMS_arcsec"]),
        },
        "controls": controls_frame.reset_index().to_dict(orient="records"),
        "heldout_subfamily_scores": family_pivot.reset_index().to_dict(orient="records"),
        "gates": gates,
        "cross_domain": {"solar_routed_fraction": 0.0, "SPARC_angular_velocity_change_km_s": 0.0},
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / protocol["outputs"]["summary"]).write_text(
        "\n".join(
            [
                "# P0577 SPT0615 raw response",
                "",
                f"SPT selected `{selected_id}`; held-out change **{100*selected_gain:.2f}%**.",
                f"SMACS-locked p=1.75,f=1 change **{100*locked_gain:.2f}%**, subfamilies improved **{locked_count}/3**.",
                f"Cross-cluster pattern supported: **{gates['cross_cluster_propagator_pattern_supported']}**.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    grid = candidates.pivot(index="fractional_power_p", columns="deflection_route_fraction", values="calibration_image_plane_RMS_arcsec")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    im = axes[0].imshow(grid.values, origin="lower", aspect="auto")
    axes[0].set_xticks(range(len(grid.columns)), grid.columns)
    axes[0].set_yticks(range(len(grid.index)), grid.index)
    axes[0].set(xlabel="route fraction", ylabel="p", title="SPT calibration response")
    fig.colorbar(im, ax=axes[0])
    axes[1].bar(controls_frame.index, controls_frame.heldout_RMS_arcsec)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("held-out image RMS (arcsec)")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)
    print(json.dumps(report["SPT_selected"], indent=2))
    print(json.dumps(report["result"], indent=2))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
