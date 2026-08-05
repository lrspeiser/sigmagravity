#!/usr/bin/env python3
"""Build frozen, target-blind v19H Chandra source maps."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sigma_v19f_chandra_common as common

ROOT = common.ROOT
DEFAULT_CONFIG = ROOT / "configs" / "sigma_v19h_causal_observable_protocol.json"
DEFAULT_ASTROMETRY = ROOT / "results" / "sigma_v19g_chandra_astrometry" / "report.json"
DEFAULT_CLEANING = ROOT / "results" / "sigma_v19f_chandra_cleaning" / "report.json"
DEFAULT_OUTPUT = ROOT / "results" / "sigma_v19h_source_maps"
DEFAULT_SCRATCH = Path("/home/henry/sigma-v19f-chandra")


def load_ciao_helpers():
    """Load pycrates-dependent helpers only inside the CIAO runtime."""
    import run_sigma_v17b_temperature_regions as shared

    return shared


def validate(config_path: Path, astrometry_path: Path, cleaning_path: Path):
    config = common.load_json(config_path)
    astrometry = common.load_json(astrometry_path)
    cleaning = common.load_json(cleaning_path)
    common.validate_parent_hashes(config)
    expected = (
        "frozen after the v19G astrometric gate passed and before any registered "
        "science-image inspection, merged image construction, edge search, spectrum "
        "extraction, temperature or density fit, projection draw, causal-source "
        "construction, replacement-cluster lensing access, or gravity fit"
    )
    if config["status"] != expected:
        raise RuntimeError("v19H source-observable protocol is not frozen")
    if astrometry["status"] != (
        "all_frozen_v19g_observations_hierarchically_registered_to_Gaia_DR3"
    ):
        raise RuntimeError("v19G hierarchical astrometry did not pass")
    if common.sha256(astrometry_path) != config["parents"]["astrometry_report_sha256"]:
        raise RuntimeError("v19G astrometry report differs from its frozen v19H parent")
    if common.sha256(cleaning_path) != config["parents"]["cleaning_report_sha256"]:
        raise RuntimeError("v19F cleaning report differs from its frozen v19H parent")
    if astrometry["registered_science_images_inspected"] is not False:
        raise RuntimeError("registered science images were inspected before v19H")
    if astrometry["observation_count"] != 20 or cleaning["observation_count"] != 20:
        raise RuntimeError("v19H requires all 20 observations")
    return config, astrometry, cleaning


def make_lists(shared, work: Path, values: dict[str, list[Path]]) -> dict[str, Path]:
    outputs = {}
    for name, paths in values.items():
        output = work / f"{name}.lis"
        shared.write_exact(output, "\n".join(str(path) for path in paths) + "\n")
        outputs[name] = output
    return outputs


def weighted_backgrounds(
    shared,
    *,
    energy_ev: tuple[int, int],
    label: str,
    blanksky: list[Path],
    astrometry_rows: list[dict],
    cleaning_rows: dict[int, dict],
    xygrid: str,
    images: Path,
    logs: Path,
    env: dict[str, str],
) -> tuple[Path, Path, list[dict]]:
    raw_images = []
    scales = []
    records = []
    xgrid, ygrid = xygrid.split(",")
    for path, row in zip(blanksky, astrometry_rows, strict=True):
        obsid = int(row["obsid"])
        for keyword, value in sorted(cleaning_rows[obsid]["blanksky_scaling"].items()):
            chip = int(keyword.removeprefix("BKGSCAL"))
            image = images / f"acisf{obsid}_ccd{chip}_{label}_blanksky_counts.img"
            shared.run_step(
                [
                    "dmcopy",
                    (
                        f"{path}[ccd_id={chip},energy={energy_ev[0]}:{energy_ev[1]}]"
                        f"[bin x={xgrid},y={ygrid}]"
                    ),
                    str(image),
                    "clobber=no",
                    "mode=h",
                ],
                logs / f"{obsid}_ccd{chip}_{label}_blanksky_image.log",
                [image],
                env,
            )
            raw_images.append(image)
            scales.append(float(value))
            records.append(
                {"obsid": obsid, "ccd_id": chip, "scale": float(value), "path": str(image)}
            )
    if not raw_images:
        raise RuntimeError(f"no {label} blank-sky chip images were produced")
    scaled = images / f"{label}_scaled_blanksky_counts.img"
    variance = images / f"{label}_scaled_blanksky_variance.img"
    shared.weighted_image_sum(
        raw_images,
        scales,
        scaled,
        f"{label}_scaled_background",
        images,
        logs,
        env,
    )
    shared.weighted_image_sum(
        raw_images,
        [value * value for value in scales],
        variance,
        f"{label}_background_variance",
        images,
        logs,
        env,
    )
    return scaled, variance, records


def flux_map(
    shared,
    *,
    label: str,
    band: tuple[float, float, float],
    lists: dict[str, Path],
    xygrid: str,
    images: Path,
    logs: Path,
    env: dict[str, str],
) -> tuple[Path, Path, dict]:
    output = images / f"{label}_flux"
    output.mkdir(parents=True, exist_ok=True)
    token = f"{band[0]:.1f}-{band[1]:.1f}_thresh"
    counts = output / f"{token}.img"
    exposure = output / f"{token}.expmap"
    step = shared.run_step(
        [
            "flux_obs",
            f"infiles=@{lists['science_reprojected']}",
            f"outroot={output}/",
            f"bands={band[0]}:{band[1]}:{band[2]}",
            f"xygrid={xygrid}",
            f"asolfiles=@{lists['corrected_aspects']}",
            f"badpixfiles=@{lists['badpix']}",
            f"maskfiles=@{lists['detector_masks']}",
            "background=none",
            "parallel=no",
            "cleanup=yes",
            "clobber=no",
            "verbose=1",
            "mode=h",
        ],
        logs / f"{label}_flux_obs.log",
        [counts, exposure],
        env,
    )
    return counts, exposure, step


def build_cluster(
    shared,
    cluster: str,
    config: dict,
    astrometry: dict,
    cleaning: dict,
    scratch: Path,
) -> dict:
    work = scratch / "causal_observables_v19h_v100" / cluster
    events = work / "events"
    images = work / "images"
    logs = work / "logs"
    for path in (events, images, logs):
        path.mkdir(parents=True, exist_ok=True)
    env = shared.isolated_environment(
        __import__("os").environ,
        scratch / "pfiles_causal_observables_v19h_v100" / cluster,
        scratch / "tmp_causal_observables_v19h_v100" / cluster,
    )
    astrometry_rows = sorted(
        (row for row in astrometry["observations"] if row["cluster"] == cluster),
        key=lambda row: int(row["obsid"]),
    )
    cleaning_rows = {
        int(row["obsid"]): row
        for row in cleaning["observations"]
        if row["cluster"] == cluster
    }
    if len(astrometry_rows) != 10 or len(cleaning_rows) != 10:
        raise RuntimeError(f"{cluster} does not have exactly ten frozen observations")
    coord = config["coordinates"]["clusters"][cluster]
    reference = next(
        row for row in astrometry_rows if int(row["obsid"]) == coord["reference_obsid"]
    )
    reference_event = Path(reference["application"]["corrected_events"]["science"]["path"])
    initial = shared.dmcoords(
        reference_event,
        "cel",
        {"ra": coord["initial_center_ra_deg"], "dec": coord["initial_center_dec_deg"]},
        env,
    )
    native_pixel = float(config["coordinates"]["native_acis_pixel_arcsec"])
    half_native = math.ceil(
        config["coordinates"]["grid_half_width_kpc"]
        / coord["kpc_per_arcsec_Planck18"]
        / native_pixel
    )
    binsize = int(config["coordinates"]["image_binsize_native_pixels"])
    xlo = math.floor(initial["x"] - half_native) + 0.5
    ylo = math.floor(initial["y"] - half_native) + 0.5
    pixels = math.ceil((2 * half_native) / binsize)
    xhi = xlo + pixels * binsize
    yhi = ylo + pixels * binsize
    requested_xygrid = f"{xlo}:{xhi}:{binsize},{ylo}:{yhi}:{binsize}"

    science_reprojected = []
    blanksky_reprojected = []
    aspects = []
    badpix = []
    masks = []
    observation_records = []
    for row in astrometry_rows:
        obsid = int(row["obsid"])
        cleaning_row = cleaning_rows[obsid]
        source_region = (
            Path(cleaning_row["clean_event"]).parent
            / "source_detect_b2"
            / "point_sources_expanded.reg"
        )
        science = Path(row["application"]["corrected_events"]["science"]["path"])
        blanksky = Path(row["application"]["corrected_events"]["blanksky"]["path"])
        science_nosrc = events / f"acisf{obsid}_science_nosrc_evt.fits"
        blanksky_nosrc = events / f"acisf{obsid}_blanksky_nosrc_evt.fits"
        shared.run_step(
            [
                "dmcopy",
                f"{science}[exclude sky=region({source_region})]",
                str(science_nosrc),
                "clobber=no",
                "mode=h",
            ],
            logs / f"{obsid}_science_source_exclusion.log",
            [science_nosrc],
            env,
        )
        shared.run_step(
            [
                "dmcopy",
                f"{blanksky}[exclude sky=region({source_region})]",
                str(blanksky_nosrc),
                "clobber=no",
                "mode=h",
            ],
            logs / f"{obsid}_blanksky_source_exclusion.log",
            [blanksky_nosrc],
            env,
        )
        science_reproj = events / f"acisf{obsid}_science_nosrc_reproj_evt.fits"
        blanksky_reproj = events / f"acisf{obsid}_blanksky_nosrc_reproj_evt.fits"
        shared.run_step(
            [
                "reproject_events",
                f"infile={science_nosrc}",
                f"outfile={science_reproj}",
                f"match={reference_event}",
                "aspect=none",
                "random=-1",
                "clobber=no",
                "verbose=1",
                "mode=h",
            ],
            logs / f"{obsid}_science_reproject.log",
            [science_reproj],
            env,
        )
        shared.run_step(
            [
                "reproject_events",
                f"infile={blanksky_nosrc}",
                f"outfile={blanksky_reproj}",
                f"match={reference_event}",
                "aspect=none",
                f"random={obsid}",
                "clobber=no",
                "verbose=1",
                "mode=h",
            ],
            logs / f"{obsid}_blanksky_reproject.log",
            [blanksky_reproj],
            env,
        )
        science_reprojected.append(science_reproj)
        blanksky_reprojected.append(blanksky_reproj)
        corrected_aspects = [
            Path(item["path"]) for item in row["application"]["corrected_aspects"]
        ]
        if len(corrected_aspects) != 1:
            raise RuntimeError(f"expected one aspect file for {cluster} ObsID {obsid}")
        aspects.append(corrected_aspects[0])
        repro = scratch / "repro" / cluster / str(obsid)
        badpix.append(shared.support_file(repro, "*repro_bpix1.fits"))
        masks.append(shared.support_file(repro, "*_msk1.fits"))
        observation_records.append(
            {
                "obsid": obsid,
                "science_reprojected": str(science_reproj),
                "blanksky_reprojected": str(blanksky_reproj),
                "point_source_region": str(source_region),
            }
        )
    lists = make_lists(
        shared,
        work,
        {
            "science_reprojected": science_reprojected,
            "corrected_aspects": aspects,
            "badpix": badpix,
            "detector_masks": masks,
        },
    )
    soft_counts, soft_exposure, soft_step = flux_map(
        shared,
        label="soft",
        band=(0.5, 2.0, 1.5),
        lists=lists,
        xygrid=requested_xygrid,
        images=images,
        logs=logs,
        env=env,
    )
    actual_xygrid, actual_grid = shared.exact_sky_grid(soft_counts, env)
    broad_counts, broad_exposure, broad_step = flux_map(
        shared,
        label="broad",
        band=(0.5, 7.0, 2.3),
        lists=lists,
        xygrid=actual_xygrid,
        images=images,
        logs=logs,
        env=env,
    )
    soft_background, soft_variance, soft_background_records = weighted_backgrounds(
        shared,
        energy_ev=(500, 2000),
        label="soft",
        blanksky=blanksky_reprojected,
        astrometry_rows=astrometry_rows,
        cleaning_rows=cleaning_rows,
        xygrid=actual_xygrid,
        images=images,
        logs=logs,
        env=env,
    )
    broad_background, broad_variance, broad_background_records = weighted_backgrounds(
        shared,
        energy_ev=(500, 7000),
        label="broad",
        blanksky=blanksky_reprojected,
        astrometry_rows=astrometry_rows,
        cleaning_rows=cleaning_rows,
        xygrid=actual_xygrid,
        images=images,
        logs=logs,
        env=env,
    )

    initial_image = shared.dmcoords(
        soft_counts,
        "cel",
        {"ra": coord["initial_center_ra_deg"], "dec": coord["initial_center_dec_deg"]},
        env,
    )
    centroid_radius = (
        config["coordinates"]["centroid_aperture_kpc"]
        / coord["kpc_per_arcsec_Planck18"]
        / config["coordinates"]["output_pixel_arcsec"]
    )
    centroid_config = config["image_products"]["centroid"]
    center_x, center_y, centroid_history = shared.centroid(
        shared.image_values(soft_counts),
        shared.image_values(soft_background),
        shared.image_values(soft_exposure),
        initial_image["logicalx"],
        initial_image["logicaly"],
        centroid_radius,
        0.5,
        centroid_config["convergence_output_pixels"],
        centroid_config["maximum_iterations"],
    )
    final_center = shared.dmcoords(
        soft_counts,
        "logical",
        {"logicalx": center_x, "logicaly": center_y},
        env,
    )
    source_values = shared.image_values(soft_counts)
    background_values = shared.image_values(soft_background)
    exposure_values = shared.image_values(soft_exposure)
    yy, xx = np.indices(source_values.shape, dtype=float)
    radius_centroid = np.hypot(xx + 1.0 - center_x, yy + 1.0 - center_y)
    exposure_threshold = 0.5 * float(np.nanmax(exposure_values))
    aperture = radius_centroid <= centroid_radius
    finite_exposure = np.isfinite(exposure_values) & (exposure_values >= exposure_threshold)
    coverage_fraction = float(np.count_nonzero(aperture & finite_exposure)) / float(
        np.count_nonzero(aperture)
    )
    net_counts = float(
        np.sum((source_values - background_values)[aperture & finite_exposure])
    )
    coverage = config["image_products"]["coverage_gate"]
    gates = {
        "minimum_valid_area_fraction_inside_1000_kpc": coverage_fraction
        >= coverage["minimum_valid_area_fraction_inside_1000_kpc"],
        "minimum_net_morphology_counts_inside_1000_kpc": net_counts
        >= coverage["minimum_net_morphology_counts_inside_1000_kpc"],
        "centroid_converged": centroid_history[-1]["displacement_output_pixels"]
        < centroid_config["convergence_output_pixels"],
        "all_observations_included": len(observation_records) == 10,
    }

    analysis_radius_native = (
        config["coordinates"]["analysis_radius_kpc"]
        / coord["kpc_per_arcsec_Planck18"]
        / native_pixel
    )
    aperture_exposure = images / "analysis_aperture_exposure.img"
    shared.run_step(
        [
            "dmcopy",
            (
                f"{soft_exposure}[sky=circle({final_center['x']},{final_center['y']},"
                f"{analysis_radius_native})][opt full]"
            ),
            str(aperture_exposure),
            "clobber=no",
            "mode=h",
        ],
        logs / "analysis_aperture_exposure.log",
        [aperture_exposure],
        env,
    )
    threshold_epsilon = max(abs(exposure_threshold) * 1e-12, np.finfo(float).eps)
    delta = f"(img1-{exposure_threshold:.17g}+{threshold_epsilon:.17g})"
    analysis_mask = images / "analysis_mask.img"
    shared.image_expression(
        [aperture_exposure],
        analysis_mask,
        f"(short)((1+({delta}/fabs({delta})))/2)",
        logs / "analysis_mask.log",
        env,
    )
    if not np.array_equal(
        np.unique(np.nan_to_num(shared.image_values(analysis_mask), nan=0.0)),
        [0.0, 1.0],
    ):
        raise RuntimeError(f"{cluster} analysis mask is not binary")
    products = {
        "soft_counts": soft_counts,
        "soft_exposure": soft_exposure,
        "soft_scaled_background": soft_background,
        "soft_background_variance": soft_variance,
        "broad_counts": broad_counts,
        "broad_exposure": broad_exposure,
        "broad_scaled_background": broad_background,
        "broad_background_variance": broad_variance,
        "analysis_mask": analysis_mask,
    }
    return {
        "cluster": cluster,
        "reference_obsid": coord["reference_obsid"],
        "requested_xygrid": requested_xygrid,
        "xygrid": actual_xygrid,
        "grid": {**actual_grid, "binsize": binsize},
        "initial_center": initial_image,
        "final_center": final_center,
        "centroid_history": centroid_history,
        "coverage_fraction_inside_1000_kpc": coverage_fraction,
        "net_morphology_counts_inside_1000_kpc": net_counts,
        "exposure_threshold": exposure_threshold,
        "observations": observation_records,
        "soft_background_components": soft_background_records,
        "broad_background_components": broad_background_records,
        "products": {name: str(path) for name, path in products.items()},
        "steps": {"soft_flux_obs": soft_step, "broad_flux_obs": broad_step},
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--astrometry", type=Path, default=DEFAULT_ASTROMETRY)
    parser.add_argument("--cleaning", type=Path, default=DEFAULT_CLEANING)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config.resolve()
    astrometry_path = args.astrometry.resolve()
    cleaning_path = args.cleaning.resolve()
    config, astrometry, cleaning = validate(
        config_path, astrometry_path, cleaning_path
    )
    shared = load_ciao_helpers()
    clusters = []
    for cluster in config["sample"]["clusters"]:
        result = build_cluster(
            shared,
            cluster,
            config,
            astrometry,
            cleaning,
            args.scratch.resolve(),
        )
        clusters.append(result)
        print(
            f"{cluster}: coverage={result['coverage_fraction_inside_1000_kpc']:.4f}, "
            f"net soft counts={result['net_morphology_counts_inside_1000_kpc']:.1f}, "
            f"gates={result['gates']}",
            flush=True,
        )
    failed = [row["cluster"] for row in clusters if not all(row["gates"].values())]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for row in clusters:
        frozen = []
        for role, source_text in sorted(row["products"].items()):
            source = Path(source_text)
            frozen.append(
                shared.snapshot_file(
                    source,
                    output / "frozen_map_products" / row["cluster"] / f"{role}{source.suffix}",
                    role,
                )
            )
        row["frozen_snapshot"] = {
            "files": len(frozen),
            "bytes": sum(item["bytes"] for item in frozen),
            "products": frozen,
        }
    passed = not failed and len(clusters) == 2
    report = {
        "status": (
            "both_clusters_passed_frozen_v19h_source_map_gate"
            if passed
            else "frozen_v19h_source_map_gate_failed"
        ),
        "protocol_version": config["protocol_version"],
        "generated_utc": datetime.now(UTC).isoformat(),
        "config_sha256": common.sha256(config_path),
        "astrometry_report_sha256": common.sha256(astrometry_path),
        "cleaning_report_sha256": common.sha256(cleaning_path),
        "failed_clusters": failed,
        "clusters": clusters,
        "registered_science_images_visually_inspected": False,
        "edge_search_run": False,
        "spectrum_or_response_constructed": False,
        "temperature_or_density_fitted": False,
        "projection_or_clock_drawn": False,
        "causal_source_constructed": False,
        "lensing_target_opened": False,
        "gravity_parameter_changed": False,
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report_path)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
