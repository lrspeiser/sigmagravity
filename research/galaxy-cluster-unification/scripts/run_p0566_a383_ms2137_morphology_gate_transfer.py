#!/usr/bin/env python3
"""Run the frozen A383 + MS2137 morphology-gate transfer."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_metric_slip_raw_lensing import build_fields, model_name as slip_model_name  # noqa: E402
from run_member_tidal_metric import (  # noqa: E402
    MemberTidalLens,
    SystemContext,
    fit_context,
    model_name,
)
from run_p0554_all_baryon_route_screen import event_wcs  # noqa: E402
from run_p0557_baryon_proxy_tidal import (  # noqa: E402
    build_candidate_context,
    json_safe,
)
from run_p0559_accept_projected_gas_tidal import physical_catalogs  # noqa: E402
from run_p0563_accept_tensor_source_plane_response import (  # noqa: E402
    unweighted_source_closure,
)
from run_p0564_baryon_morphology_sign_audit import (  # noqa: E402
    acute_quadrupole_misalignment,
    component_descriptors,
)
from run_rxj2129_raw_theory_lensing import (  # noqa: E402
    FIXED_INITIAL,
    RawLens,
    score,
)
from run_unbounded_running_multicluster_raw import (  # noqa: E402
    load_anchors,
    system_protocol,
)


CONFIG = ROOT / "configs/p0566_a383_ms2137_morphology_gate_transfer_protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def rms(values):
    values = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def verify_acquisition(protocol, acquisition, provenance):
    records = {row["local_path"]: row for row in provenance["records"]}
    expected = 0
    checks = []
    for system in acquisition["systems"]:
        label = system["label"]
        for product in system["hst"]:
            relative = (
                Path(acquisition["outputs"]["directory"])
                / "hst"
                / label
                / product["filename"]
            ).as_posix()
            path = ROOT / relative
            record = records[relative]
            checks.append(
                path.stat().st_size == int(record["size_bytes"])
                and sha256(path) == record["sha256"]
            )
            expected += 1
        for product in system["chandra"]:
            relative = (
                Path(acquisition["outputs"]["directory"])
                / "chandra"
                / label
                / str(product["obsid"])
                / product["filename"]
            ).as_posix()
            path = ROOT / relative
            record = records[relative]
            checks.append(
                path.stat().st_size == int(record["size_bytes"])
                and sha256(path) == record["sha256"]
            )
            expected += 1
    return {"expected_records": expected, "all_hashes_verified": bool(all(checks))}


def load_images(protocol, system):
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    selected = catalog[
        catalog.system.eq(system["system"])
        & catalog.alternative_metric_likelihood_ready.astype(bool)
    ].copy()
    selected["image_id"] = selected.image_id.astype(str)
    selected["source_family"] = selected.source_family.astype(int)
    selected["source_redshift"] = selected.source_redshift.astype(float)
    selected["x_arcsec"] = -selected.delta_x_west_arcsec.astype(float)
    selected["y_arcsec"] = selected.delta_y_north_arcsec.astype(float)
    selected["radius_arcsec"] = np.hypot(selected.x_arcsec, selected.y_arcsec)
    selected = selected.sort_values(["source_family", "image_id"]).reset_index(drop=True)
    if len(selected) != int(system["expected_images"]):
        raise RuntimeError(f"{system['label']} image count changed")
    if selected.source_family.nunique() != int(system["expected_families"]):
        raise RuntimeError(f"{system['label']} family count changed")
    if not np.allclose(
        selected.image_position_sigma_arcsec.astype(float),
        float(system["position_sigma_arcsec"]),
    ):
        raise RuntimeError(f"{system['label']} position uncertainty changed")
    heldout_indices = []
    for _, group in selected.groupby("source_family", sort=True):
        if len(group) >= 3:
            heldout_indices.append(group.sort_values("image_id").index[-1])
    heldout = selected.loc[heldout_indices].copy().sort_values(
        ["source_family", "image_id"]
    )
    training = selected.drop(index=heldout_indices).copy().sort_values(
        ["source_family", "image_id"]
    )
    if len(training) != int(system["expected_training_images"]):
        raise RuntimeError(f"{system['label']} training split changed")
    if len(heldout) != int(system["expected_heldout_images"]):
        raise RuntimeError(f"{system['label']} heldout split changed")
    return selected, training.reset_index(drop=True), heldout.reset_index(drop=True)


def hst_paths(acquisition, label):
    system = next(row for row in acquisition["systems"] if row["label"] == label)
    root = ROOT / acquisition["outputs"]["directory"] / "hst" / label
    science = next(row for row in system["hst"] if row["kind"] == "hst_f160w_science")
    weight = next(row for row in system["hst"] if row["kind"] == "hst_f160w_weight")
    return root / science["filename"], root / weight["filename"]


def prepare_hst_map(protocol, acquisition, context, images, axis):
    settings = protocol["registered_map_construction"]
    science_path, weight_path = hst_paths(acquisition, context.system["label"])
    with fits.open(science_path, memmap=True) as science_hdul, fits.open(
        weight_path, memmap=True
    ) as weight_hdul:
        header = science_hdul[0].header
        science = science_hdul[0].data
        weight = weight_hdul[0].data
        wcs = WCS(header)
        wcs.sip = None
        geometry = context.local_protocol["cosmology_and_coordinates"]
        center_ra = float(geometry["center_ra_deg"])
        center_dec = float(geometry["center_dec_deg"])
        center_x, center_y = wcs.wcs_world2pix([[center_ra, center_dec]], 0)[0]
        matrix = np.asarray(
            [
                [header["CD1_1"], header["CD1_2"]],
                [header["CD2_1"], header["CD2_2"]],
            ],
            dtype=float,
        )
        native_scale = float(np.sqrt(abs(np.linalg.det(matrix))) * 3600.0)
        cut_radius = float(settings["hst_native_cut_radius_arcsec"])
        cut_pixels = int(np.ceil(cut_radius / native_scale))
        x0, x1 = int(round(center_x)) - cut_pixels, int(round(center_x)) + cut_pixels + 1
        y0, y1 = int(round(center_y)) - cut_pixels, int(round(center_y)) + cut_pixels + 1
        cut = np.asarray(science[y0:y1, x0:x1], dtype=np.float64)
        cut_weight = np.asarray(weight[y0:y1, x0:x1], dtype=np.float64)
        yy, xx = np.indices(cut.shape, dtype=float)
        # x is east-positive, matching the internal lens convention.
        x_arcsec = -(xx + x0 - center_x) * native_scale
        y_arcsec = (yy + y0 - center_y) * native_scale
        radius = np.hypot(x_arcsec, y_arcsec)
        valid = (cut_weight > 0.0) & np.isfinite(cut)
        bg_lo, bg_hi = settings["hst_background_annulus_arcsec"]
        outer = valid & (radius >= float(bg_lo)) & (radius <= float(bg_hi))
        if not np.any(outer):
            raise RuntimeError(f"{context.system['label']} lacks HST background annulus")
        background = float(np.median(cut[outer]))
        image_mask = np.zeros_like(valid)
        mask_radius = float(settings["known_image_mask_radius_arcsec"])
        for row in images.itertuples(index=False):
            image_mask |= (
                np.square(x_arcsec - float(row.x_arcsec))
                + np.square(y_arcsec - float(row.y_arcsec))
                <= mask_radius**2
            )
        usable = valid & ~image_mask
        fill = np.full_like(cut, background)
        bins = np.floor(radius / 0.5).astype(int)
        for index in range(int(np.ceil(cut_radius / 0.5)) + 1):
            target = bins == index
            source = target & usable
            if np.any(source):
                fill[target] = float(np.median(cut[source]))
        filled = cut.copy()
        filled[~usable] = fill[~usable]
        positive = np.maximum(filled - background, 0.0)
        target_x, target_y = np.meshgrid(axis, axis)
        cosine = math.cos(math.radians(center_dec))
        target_ra = center_ra + target_x / (3600.0 * cosine)
        target_dec = center_dec + target_y / 3600.0
        pixel_x, pixel_y = wcs.wcs_world2pix(target_ra, target_dec, 0)
        coordinates = np.vstack(
            [(pixel_y - y0).ravel(), (pixel_x - x0).ravel()]
        )
        sampled = map_coordinates(
            positive,
            coordinates,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).reshape(target_x.shape)
        sampled = np.maximum(sampled, 0.0)
    return sampled, {
        "system_label": context.system["label"],
        "map_kind": "hst_f160w",
        "input_count": 2,
        "native_pixel_scale_arcsec": native_scale,
        "background": background,
        "masked_fraction": float(np.mean(image_mask)),
        "positive_cells": int(np.sum(sampled > 0.0)),
        "map_sum": float(sampled.sum()),
    }


def chandra_paths(acquisition, label):
    system = next(row for row in acquisition["systems"] if row["label"] == label)
    root = ROOT / acquisition["outputs"]["directory"] / "chandra" / label
    return [root / str(row["obsid"]) / row["filename"] for row in system["chandra"]]


def prepare_xray_map(protocol, acquisition, context, axis):
    settings = protocol["registered_map_construction"]
    spacing = float(settings["grid_spacing_arcsec"])
    edges = np.r_[axis - 0.5 * spacing, axis[-1] + 0.5 * spacing]
    rate = np.zeros((len(axis), len(axis)), dtype=float)
    counts = np.zeros_like(rate)
    total_exposure = 0.0
    lo, hi = settings["xray_band_keV"]
    geometry = context.local_protocol["cosmology_and_coordinates"]
    paths = chandra_paths(acquisition, context.system["label"])
    for path in paths:
        with fits.open(path, memmap=False) as hdul:
            events = hdul["EVENTS"]
            header = events.header
            wcs = event_wcs(header)
            center_x, center_y = wcs.all_world2pix(
                [[geometry["center_ra_deg"], geometry["center_dec_deg"]]], 1
            )[0]
            pixel_arcsec = abs(float(header["TCDLT11"])) * 3600.0
            x = -(events.data["x"].astype(float) - center_x) * pixel_arcsec
            y = (events.data["y"].astype(float) - center_y) * pixel_arcsec
            energy = events.data["energy"].astype(float) / 1000.0
            use = (energy >= float(lo)) & (energy <= float(hi))
            image, _, _ = np.histogram2d(y[use], x[use], bins=(edges, edges))
            exposure = float(header["EXPOSURE"])
            rate += image / exposure
            counts += image
            total_exposure += exposure
    point = settings["xray_point_source_detection"]
    small_sigma = float(point["small_gaussian_sigma_arcsec"]) / spacing
    broad_sigma = float(point["broad_gaussian_sigma_arcsec"]) / spacing
    small_counts = gaussian_filter(counts, small_sigma, mode="nearest")
    broad_counts = gaussian_filter(counts, broad_sigma, mode="nearest")
    variance = np.maximum(broad_counts, 0.0) / (4.0 * np.pi * small_sigma**2) + 1.0e-6
    significance = (small_counts - broad_counts) / np.sqrt(variance)
    grid_x, grid_y = np.meshgrid(axis, axis)
    outside_core = np.hypot(grid_x, grid_y) >= float(
        point["protected_cluster_core_radius_arcsec"]
    )
    seeds = (
        significance >= float(point["difference_significance_threshold"])
    ) & outside_core
    dilation = int(np.ceil(float(point["mask_dilation_radius_arcsec"]) / spacing))
    yy, xx = np.indices((2 * dilation + 1, 2 * dilation + 1)) - dilation
    structure = np.square(xx) + np.square(yy) <= dilation**2
    point_mask = binary_dilation(seeds, structure=structure)
    broad_rate = gaussian_filter(rate, broad_sigma, mode="nearest")
    masked = rate.copy()
    masked[point_mask] = broad_rate[point_mask]
    masked = np.maximum(masked, 0.0)
    return masked, {
        "system_label": context.system["label"],
        "map_kind": "chandra_soft_rate",
        "input_count": len(paths),
        "total_exposure_ks": total_exposure / 1000.0,
        "soft_events_on_grid": int(counts.sum()),
        "point_source_seed_cells": int(seeds.sum()),
        "point_source_masked_fraction": float(point_mask.mean()),
        "map_sum": float(masked.sum()),
    }


def build_contexts(protocol, base_protocol, metric_protocol):
    tian = pd.read_csv(
        ROOT / protocol["inputs"]["baryonic_profile"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    slip = float(protocol["field_hypothesis"]["locked_metric_slip_s"])
    cutoff = float(metric_protocol["field_closure"]["primary_maximum_radius_kpc"])
    a_dagger = float(metric_protocol["matter_law"]["a_dagger_m_s2"])
    contexts = []
    field_sets = {}
    all_images = {}
    for system in protocol["systems"]:
        local = system_protocol(base_protocol, system)
        local["optimization"]["maximum_function_evaluations"] = int(
            protocol["response_and_exact_score"]["maximum_function_evaluations"]
        )
        images, training, heldout = load_images(protocol, system)
        anchors = load_anchors(tian, system["label"])
        raw_fields, _ = build_fields(
            anchors,
            local,
            [-2.0, 0.0, slip],
            cutoff_kpc=cutoff,
            a_dagger=a_dagger,
        )
        fields = {
            "baryon": raw_fields[slip_model_name(0)],
            "rar_equal": raw_fields[slip_model_name(1)],
            "scalar_slip": raw_fields[slip_model_name(2)],
        }

        def extra_alpha(radius, local_fields=fields):
            return local_fields["scalar_slip"].reduced_alpha_arcsec(
                radius, 1.0
            ) - local_fields["baryon"].reduced_alpha_arcsec(radius, 1.0)

        contexts.append(
            SystemContext(
                system=system,
                local_protocol=local,
                training=training,
                heldout=heldout,
                members=pd.DataFrame(),
                fields={"baryon": fields["baryon"], "scalar_slip": fields["scalar_slip"]},
                correction=None,
                initial_geometry=np.asarray(
                    protocol["response_and_exact_score"]["initial_geometry"], dtype=float
                ),
                extra_alpha=extra_alpha,
            )
        )
        field_sets[system["label"]] = fields
        all_images[system["label"]] = images
    return contexts, field_sets, all_images


def morphology_for(protocol, registered):
    gate = protocol["frozen_sign_gate"]
    rows = []
    for label, maps in registered.items():
        axis = maps["axis"]
        spacing = float(axis[1] - axis[0])
        star = np.maximum(maps["star"], 0.0)
        gas = np.sqrt(
            np.maximum(
                gaussian_filter(
                    np.maximum(maps["gas"], 0.0),
                    sigma=3.0 / spacing,
                    mode="nearest",
                ),
                0.0,
            )
        )
        inner_star = component_descriptors(
            axis, star, float(gate["inner_correlation_aperture_arcsec"])
        )
        inner_gas = component_descriptors(
            axis, gas, float(gate["inner_correlation_aperture_arcsec"])
        )
        inner_mask = inner_star["_mask"] & inner_gas["_mask"]
        correlation = float(
            np.corrcoef(
                inner_star["_normalized_image"][inner_mask],
                inner_gas["_normalized_image"][inner_mask],
            )[0, 1]
        )
        outer_star = component_descriptors(
            axis, star, float(gate["outer_alignment_aperture_arcsec"])
        )
        outer_gas = component_descriptors(
            axis, gas, float(gate["outer_alignment_aperture_arcsec"])
        )
        misalignment = acute_quadrupole_misalignment(
            outer_star["quadrupole_angle_deg"], outer_gas["quadrupole_angle_deg"]
        )
        cos2 = float(np.cos(np.radians(2.0 * misalignment)))
        inner_trigger = correlation > float(
            gate["inner_star_gas_correlation_threshold"]
        )
        outer_trigger = cos2 < float(gate["outer_quadrupole_cos2_threshold"])
        sign = "negative" if inner_trigger and outer_trigger else "positive"
        coupling = -float(gate["universal_magnitude"]) if sign == "negative" else float(
            gate["universal_magnitude"]
        )
        rows.append(
            {
                "system_label": label,
                "inner_star_gas_correlation": correlation,
                "inner_threshold": float(gate["inner_star_gas_correlation_threshold"]),
                "inner_negative_trigger": inner_trigger,
                "outer_quadrupole_misalignment_deg": misalignment,
                "outer_quadrupole_cos2_alignment": cos2,
                "outer_threshold": float(gate["outer_quadrupole_cos2_threshold"]),
                "outer_negative_trigger": outer_trigger,
                "predicted_sign": sign,
                "predicted_coupling": coupling,
            }
        )
    return pd.DataFrame(rows)


def comparator_fit(context, fields, model_id, starts, seed):
    if model_id == "baryons_GR":
        lens_fields = {model_id: fields["baryon"]}
        raw_model = model_id
    elif model_id == "RAR_equal_light_matter":
        lens_fields = {model_id: fields["rar_equal"]}
        raw_model = model_id
    elif model_id == "compact_halo":
        lens_fields = {"baryons_GR": fields["baryon"]}
        raw_model = "GR_plus_cluster_halo"
    else:
        raise ValueError(model_id)
    lens = RawLens(context.local_protocol, lens_fields)
    fitted = lens.fit(
        raw_model,
        context.training,
        starts=starts,
        seed=seed,
    )
    train = lens.exact_predictions(
        raw_model,
        fitted["result"].x,
        fitted["sources"],
        context.training,
        stage="training",
    )
    held = lens.exact_predictions(
        raw_model,
        fitted["result"].x,
        fitted["sources"],
        context.heldout,
        stage="heldout",
    )
    return {
        "fit": fitted,
        "training": score(train, lens.sigma, free_parameters=len(fitted["result"].x)),
        "heldout": score(held, lens.sigma),
        "training_predictions": train,
        "heldout_predictions": held,
    }


def main():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_after_verified_acquisition"):
        raise RuntimeError("P0566 transfer protocol was not frozen before array access")
    acquisition = json.loads(
        (ROOT / protocol["inputs"]["acquisition_protocol"]).read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (ROOT / protocol["inputs"]["acquisition_provenance"]).read_text(encoding="utf-8")
    )
    acquisition_audit = verify_acquisition(protocol, acquisition, provenance)
    if not acquisition_audit["all_hashes_verified"]:
        raise RuntimeError("P0566 acquisition hashes failed before array access")
    p0559 = json.loads((ROOT / protocol["inputs"]["p0559_protocol"]).read_text())
    p0557 = json.loads((ROOT / protocol["inputs"]["p0557_protocol"]).read_text())
    metric = json.loads(
        (ROOT / protocol["inputs"]["metric_slip_protocol"]).read_text()
    )
    base_protocol = json.loads(
        (ROOT / "configs/unbounded_running_multicluster_raw_protocol.json").read_text()
    )
    contexts, field_sets, all_images = build_contexts(protocol, base_protocol, metric)

    maps = protocol["registered_map_construction"]
    axis = np.arange(
        float(maps["axis_min_arcsec"]),
        float(maps["axis_max_arcsec"]) + 0.5 * float(maps["grid_spacing_arcsec"]),
        float(maps["grid_spacing_arcsec"]),
    )
    registered = {}
    map_audits = []
    for context in contexts:
        label = context.system["label"]
        print(f"P0566 open frozen arrays and register {label}", flush=True)
        star, star_audit = prepare_hst_map(
            protocol, acquisition, context, all_images[label], axis
        )
        gas, gas_audit = prepare_xray_map(protocol, acquisition, context, axis)
        if star.sum() <= 0 or gas.sum() <= 0:
            raise RuntimeError(f"{label} registered map is empty")
        registered[label] = {"axis": axis, "star": star, "gas": gas}
        map_audits.extend([star_audit, gas_audit])

    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(map_audits).to_csv(
        output / protocol["outputs"]["map_audits"], index=False
    )
    morphology = morphology_for(protocol, registered)
    morphology.to_csv(output / protocol["outputs"]["morphology"], index=False)
    print("P0566 frozen morphology predictions", flush=True)
    print(morphology.to_string(index=False), flush=True)

    catalogs, physical_audits = physical_catalogs(p0559, contexts, registered)
    physical_audits.to_csv(
        output / protocol["outputs"]["physical_map_audit"], index=False
    )
    key = (
        protocol["physical_tensor_map"]["gas_normalization"],
        float(protocol["physical_tensor_map"]["gas_power"]),
        bool(protocol["physical_tensor_map"]["include_stars"]),
    )
    tensor_audits = []
    tensor_contexts = {}
    for base in contexts:
        label = base.system["label"]
        tensor_contexts[label] = build_candidate_context(
            base,
            catalogs[label][key],
            p0557,
            "accept_absolute_sqrt",
            {"operator_id": "contrast", "subtract_circular_mean": True},
            pixels_per_axis=int(protocol["physical_tensor_map"]["pixels_per_axis"]),
            softening_kpc=float(protocol["physical_tensor_map"]["softening_kpc"]),
            audit_rows=tensor_audits,
            stage="p0566_two_cluster_transfer",
        )
    tensor_frame = pd.DataFrame(tensor_audits)
    tensor_frame.to_csv(output / protocol["outputs"]["tensor_audit"], index=False)

    morphology_map = morphology.set_index("system_label")
    response_rule = protocol["response_and_exact_score"]
    lo = float(response_rule["exploratory_grid_min"])
    hi = float(response_rule["exploratory_grid_max"])
    step = float(response_rule["exploratory_grid_step"])
    full_grid = lo + step * np.arange(int(round((hi - lo) / step)) + 1)
    seeds = list(map(int, response_rule["optimizer_seed_ensembles"]))
    starts = int(response_rule["starts_per_exact_fit"])
    exact_rows = []
    response_rows = []
    comparator_rows = []
    predictions = []
    zero_fits = {}

    for ensemble_index, seed in enumerate(seeds):
        ensemble = f"seed_{ensemble_index + 1}"
        for system_index, base in enumerate(contexts):
            label = base.system["label"]
            context = tensor_contexts[label]
            coupling = float(morphology_map.loc[label, "predicted_coupling"])
            for model_id, model_coupling in [("zero", 0.0), ("morphology_gated_t", coupling)]:
                print(
                    f"P0566 exact {ensemble} {label} {model_id} t={model_coupling:+.2f}",
                    flush=True,
                )
                fitted = fit_context(
                    context,
                    model_coupling,
                    starts=starts,
                    seed=seed + 100 * system_index,
                )
                if model_id == "zero":
                    zero_fits[(ensemble, label)] = fitted
                exact_rows.append(
                    {
                        "ensemble": ensemble,
                        "system_label": label,
                        "model_id": model_id,
                        "coupling": model_coupling,
                        "fit_cost": float(fitted["fit"]["result"].cost),
                        "training_exact_RMS_arcsec": fitted["training"]["exact_radial_RMS_arcsec"],
                        "heldout_exact_RMS_arcsec": fitted["heldout"]["exact_radial_RMS_arcsec"],
                        "all_training_roots": fitted["training"]["all_roots_converged"],
                        "all_heldout_roots": fitted["heldout"]["all_roots_converged"],
                    }
                )
                for frame in [fitted["training_predictions"], fitted["heldout_predictions"]]:
                    local = frame.copy()
                    local["ensemble"] = ensemble
                    local["model_id"] = model_id
                    predictions.append(local)

            zero = zero_fits[(ensemble, label)]
            parameters = zero["fit"]["result"].x
            maximum_q = float(
                tensor_frame.loc[
                    tensor_frame.system_label.eq(label), "maximum_Q_eigenvalue"
                ].iloc[0]
            )
            safe_grid = full_grid[1.0 - np.abs(full_grid) * maximum_q > 0.05]
            near_zero = set(map(float, response_rule["near_zero_pair"]))
            if not near_zero.issubset(set(map(float, safe_grid))):
                raise RuntimeError(f"{label} near-zero response violates ellipticity")
            for grid_coupling in safe_grid:
                lens = MemberTidalLens(
                    context.local_protocol,
                    context.fields,
                    context.correction,
                    float(grid_coupling),
                )
                values = unweighted_source_closure(
                    lens,
                    model_name(float(grid_coupling)),
                    parameters,
                    context.training,
                    context.heldout,
                )
                response_rows.append(
                    {
                        "ensemble": ensemble,
                        "system_label": label,
                        "coupling": float(grid_coupling),
                        **values,
                        "minimum_permittivity_eigenvalue": 1.0
                        - abs(float(grid_coupling)) * maximum_q,
                    }
                )

            comparator_starts = int(
                protocol["comparators"]["starts_per_comparator_fit"]
            )
            for comparator_index, comparator in enumerate(
                ["baryons_GR", "RAR_equal_light_matter", "compact_halo"]
            ):
                print(f"P0566 comparator {ensemble} {label} {comparator}", flush=True)
                fitted = comparator_fit(
                    base,
                    field_sets[label],
                    comparator,
                    comparator_starts,
                    seed + 100 * system_index + 1000 * (comparator_index + 1),
                )
                comparator_rows.append(
                    {
                        "ensemble": ensemble,
                        "system_label": label,
                        "model_id": comparator,
                        "fit_cost": float(fitted["fit"]["result"].cost),
                        "training_exact_RMS_arcsec": fitted["training"]["exact_radial_RMS_arcsec"],
                        "heldout_exact_RMS_arcsec": fitted["heldout"]["exact_radial_RMS_arcsec"],
                        "all_training_roots": fitted["training"]["all_roots_converged"],
                        "all_heldout_roots": fitted["heldout"]["all_roots_converged"],
                        "fitted_lens_geometry_parameters": len(fitted["fit"]["result"].x),
                    }
                )
                for frame in [fitted["training_predictions"], fitted["heldout_predictions"]]:
                    local = frame.copy()
                    local.insert(0, "system_label", label)
                    local["ensemble"] = ensemble
                    local["model_id"] = comparator
                    predictions.append(local)

    exact = pd.DataFrame(exact_rows)
    response = pd.DataFrame(response_rows)
    comparators = pd.DataFrame(comparator_rows)
    negative, positive = map(float, response_rule["near_zero_pair"])
    sign_rows = []
    for (ensemble, label), group in response.groupby(["ensemble", "system_label"]):
        neg = group[group.coupling.eq(negative)].iloc[0]
        pos = group[group.coupling.eq(positive)].iloc[0]
        slope = (
            float(pos.heldout_unweighted_source_plane_RMS_arcsec)
            - float(neg.heldout_unweighted_source_plane_RMS_arcsec)
        ) / (positive - negative)
        sign_rows.append(
            {
                "ensemble": ensemble,
                "system_label": label,
                "near_zero_dRMS_dt_arcsec": slope,
                "near_zero_preferred_sign": "positive" if slope < 0 else "negative" if slope > 0 else "flat",
            }
        )
    signs = pd.DataFrame(sign_rows)
    response = response.merge(signs, on=["ensemble", "system_label"], how="left")

    zero = exact[exact.model_id.eq("zero")].set_index(["ensemble", "system_label"])
    exact["improvement_fraction_vs_zero"] = [
        0.0
        if row.model_id == "zero"
        else 1.0
        - float(row.heldout_exact_RMS_arcsec)
        / float(zero.loc[(row.ensemble, row.system_label), "heldout_exact_RMS_arcsec"])
        if np.isfinite(float(row.heldout_exact_RMS_arcsec))
        else -np.inf
        for row in exact.itertuples(index=False)
    ]
    for ensemble, group in exact.groupby("ensemble"):
        for model_id, block in group.groupby("model_id"):
            exact_rows.append(
                {
                    "ensemble": ensemble,
                    "system_label": "equal_system_aggregate",
                    "model_id": model_id,
                    "coupling": np.nan,
                    "fit_cost": float(block.fit_cost.sum()),
                    "training_exact_RMS_arcsec": rms(block.training_exact_RMS_arcsec),
                    "heldout_exact_RMS_arcsec": rms(block.heldout_exact_RMS_arcsec),
                    "all_training_roots": bool(block.all_training_roots.all()),
                    "all_heldout_roots": bool(block.all_heldout_roots.all()),
                    "improvement_fraction_vs_zero": np.nan,
                }
            )
    aggregate = pd.DataFrame(exact_rows[-4:])
    for ensemble in aggregate.ensemble.unique():
        baseline = float(
            aggregate[
                aggregate.ensemble.eq(ensemble) & aggregate.model_id.eq("zero")
            ].heldout_exact_RMS_arcsec.iloc[0]
        )
        mask = aggregate.ensemble.eq(ensemble)
        aggregate.loc[mask, "improvement_fraction_vs_zero"] = 1.0 - (
            aggregate.loc[mask, "heldout_exact_RMS_arcsec"] / baseline
        )
    exact = pd.concat([exact, aggregate], ignore_index=True)

    comparator_aggregates = []
    for (ensemble, model_id), block in comparators.groupby(["ensemble", "model_id"]):
        comparator_aggregates.append(
            {
                "ensemble": ensemble,
                "system_label": "equal_system_aggregate",
                "model_id": model_id,
                "fit_cost": float(block.fit_cost.sum()),
                "training_exact_RMS_arcsec": rms(block.training_exact_RMS_arcsec),
                "heldout_exact_RMS_arcsec": rms(block.heldout_exact_RMS_arcsec),
                "all_training_roots": bool(block.all_training_roots.all()),
                "all_heldout_roots": bool(block.all_heldout_roots.all()),
                "fitted_lens_geometry_parameters": int(
                    block.fitted_lens_geometry_parameters.sum()
                ),
            }
        )
    comparators = pd.concat(
        [comparators, pd.DataFrame(comparator_aggregates)], ignore_index=True
    )
    exact.to_csv(output / protocol["outputs"]["exact_scores"], index=False)
    response.to_csv(output / protocol["outputs"]["source_plane_response"], index=False)
    comparators.to_csv(output / protocol["outputs"]["comparator_scores"], index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output / protocol["outputs"]["predictions"], index=False
    )

    predicted = morphology_map.predicted_sign.to_dict()
    sign_pass = bool(
        all(
            row.near_zero_preferred_sign == predicted[row.system_label]
            for row in signs.itertuples(index=False)
        )
    )
    candidate = exact[
        exact.model_id.eq("morphology_gated_t")
        & ~exact.system_label.eq("equal_system_aggregate")
    ]
    zero_system = exact[
        exact.model_id.eq("zero")
        & ~exact.system_label.eq("equal_system_aggregate")
    ].set_index(["ensemble", "system_label"])
    roots_pass = bool(candidate.all_heldout_roots.all())
    system_improve = bool(
        all(
            float(row.heldout_exact_RMS_arcsec)
            < float(zero_system.loc[(row.ensemble, row.system_label), "heldout_exact_RMS_arcsec"])
            for row in candidate.itertuples(index=False)
        )
    )
    aggregate_candidate = exact[
        exact.model_id.eq("morphology_gated_t")
        & exact.system_label.eq("equal_system_aggregate")
    ]
    aggregate_improve = bool((aggregate_candidate.improvement_fraction_vs_zero > 0).all())
    validated = bool(sign_pass and roots_pass and system_improve and aggregate_improve)

    report = {
        "report_version": "P0566-A383-MS2137-MORPHOLOGY-GATE-TRANSFER-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": CONFIG.relative_to(ROOT).as_posix(), "sha256": sha256(CONFIG)},
        "acquisition_audit": acquisition_audit,
        "morphology": morphology.to_dict("records"),
        "response_signs": signs.to_dict("records"),
        "exact_scores": exact.to_dict("records"),
        "comparator_scores": comparators.to_dict("records"),
        "physical_map_audit": physical_audits.to_dict("records"),
        "tensor_audit": tensor_audits,
        "gate_audit": {
            "morphology_sign_matches_near_zero_sign_for_both_systems_and_ensembles": sign_pass,
            "exact_candidate_all_roots_for_both_systems_and_ensembles": roots_pass,
            "exact_candidate_improves_zero_for_both_systems_and_ensembles": system_improve,
            "equal_system_exact_candidate_improves_zero_in_both_ensembles": aggregate_improve,
        },
        "primary": {
            "candidate_gate_validated": validated,
            "formula_promoted": False,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for (ensemble, label), group in response.groupby(["ensemble", "system_label"]):
        group = group.sort_values("coupling")
        zero_value = float(
            group.loc[
                group.coupling.eq(0.0),
                "heldout_unweighted_source_plane_RMS_arcsec",
            ].iloc[0]
        )
        axes[0].plot(
            group.coupling,
            100.0 * (1.0 - group.heldout_unweighted_source_plane_RMS_arcsec / zero_value),
            label=f"{label} {ensemble}",
        )
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set(
        xlabel="tensor coupling t",
        ylabel="held-out source-plane improvement (%)",
        title="Conditioning-resistant response",
    )
    axes[0].legend(fontsize=7)
    exact_aggregate = exact[exact.system_label.eq("equal_system_aggregate")]
    for model_id, block in exact_aggregate.groupby("model_id"):
        axes[1].bar(
            [f"{value}\n{model_id}" for value in block.ensemble],
            block.heldout_exact_RMS_arcsec,
            label=model_id,
        )
    axes[1].set(ylabel="equal-system held-out exact RMS (arcsec)", title="Frozen gate")
    axes[1].tick_params(axis="x", rotation=25)
    combined = pd.concat(
        [
            exact_aggregate[["ensemble", "model_id", "heldout_exact_RMS_arcsec"]],
            comparators[
                comparators.system_label.eq("equal_system_aggregate")
            ][["ensemble", "model_id", "heldout_exact_RMS_arcsec"]],
        ],
        ignore_index=True,
    )
    means = combined.groupby("model_id").heldout_exact_RMS_arcsec.mean().sort_values()
    axes[2].barh(means.index, means.values)
    axes[2].set(xlabel="mean equal-system held-out RMS (arcsec)", title="DM / RAR yardsticks")
    fig.savefig(output / protocol["outputs"]["figure"], dpi=180)
    plt.close(fig)

    map_fig, map_axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for row, label in enumerate(["A383", "MS2137"]):
        for column, key_name in enumerate(["star", "gas"]):
            image = np.log1p(
                registered[label][key_name]
                / max(float(np.nanpercentile(registered[label][key_name], 95)), 1.0e-30)
            )
            map_axes[row, column].imshow(
                image,
                origin="lower",
                extent=[axis.min(), axis.max(), axis.min(), axis.max()],
                cmap="magma" if key_name == "star" else "viridis",
            )
            map_axes[row, column].set(
                title=f"{label} {key_name}", xlabel="east (arcsec)", ylabel="north (arcsec)"
            )
    map_fig.savefig(output / protocol["outputs"]["map_figure"], dpi=180)
    plt.close(map_fig)

    aggregate_lines = exact[
        exact.system_label.eq("equal_system_aggregate")
    ][["ensemble", "model_id", "heldout_exact_RMS_arcsec", "improvement_fraction_vs_zero"]]
    (output / protocol["outputs"]["summary"]).write_text(
        "# P0566 A383 + MS2137 morphology-gate transfer\n\n"
        + morphology.to_string(index=False)
        + "\n\n"
        + aggregate_lines.to_string(index=False)
        + "\n\n"
        + f"Gate validated: **{validated}**. No formula is promoted.\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(report["gate_audit"]), indent=2), flush=True)
    print(exact.to_string(index=False), flush=True)
    print(comparators.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
