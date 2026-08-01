#!/usr/bin/env python3
"""Continuous-light, monopole-conserving cluster morphology experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_rxj2129_raw_theory_lensing import RawLens, near_bound, score, spec_for
from run_unbounded_running_multicluster_raw import (
    acceleration as running_acceleration,
    aggregate_system_scores,
    load_anchors,
    load_system_images,
    predictive_split,
    system_protocol,
)
from voidscreen.phenomenology import fixed_rar_enhancement
from voidscreen.raw_lensing import (
    RadialDeflectionField,
    loglog_interpolate_with_tails,
    spherical_deflection_radians,
)
from voidscreen.stellar_morphology_lensing import (
    StellarMorphologyDeflectionField,
    build_stellar_morphology_deflection_field,
)


PARENTS = (
    "fixed_RAR_zero_slip",
    "fixed_RAR_slip_s5",
    "curvature_additive_alpha10",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


@dataclass
class Context:
    system: dict
    local: dict
    training: pd.DataFrame
    heldout: pd.DataFrame
    anchors: pd.DataFrame
    parent_fields: dict[str, RadialDeflectionField]
    baryon_fields: dict[str, RadialDeflectionField]
    light_maps: dict[float, np.ndarray]
    light_audit: dict


class MorphologyLens(RawLens):
    """A radial parent plus one fixed continuous-light angular correction."""

    def __init__(
        self,
        protocol: dict,
        fields: dict[str, RadialDeflectionField],
        *,
        parent: str,
        morphology: StellarMorphologyDeflectionField | None,
        fraction: float,
    ):
        super().__init__(protocol, fields)
        self.parent = parent
        self.morphology = morphology
        self.fraction = float(fraction)

    def alpha(self, model, parameters, x_arcsec, y_arcsec, source_redshift):
        base_x, base_y = super().alpha(
            self.parent, parameters, x_arcsec, y_arcsec, source_redshift
        )
        if self.morphology is None or self.fraction == 0.0:
            return base_x, base_y
        correction_x, correction_y = self.morphology.alpha_arcsec(
            x_arcsec,
            y_arcsec,
            distance_ratio=self.distance_ratio(float(source_redshift)),
        )
        return (
            base_x + self.fraction * correction_x,
            base_y + self.fraction * correction_y,
        )


def parent_acceleration(
    parent: str,
    radius_kpc: np.ndarray,
    anchors: pd.DataFrame,
    running_protocol: dict,
) -> tuple[np.ndarray, np.ndarray]:
    radius = np.asarray(radius_kpc, dtype=float)
    anchor_radius = anchors.radius_kpc.to_numpy(float)
    anchor_gbar = np.power(10.0, anchors.log_gbar.to_numpy(float))
    gbar = loglog_interpolate_with_tails(
        radius,
        anchor_radius,
        anchor_gbar,
        outer_slope=-2.0,
    )
    if parent in {"fixed_RAR_zero_slip", "fixed_RAR_slip_s5"}:
        g_rar = gbar * fixed_rar_enhancement(gbar, 1.2e-10)
        if parent == "fixed_RAR_zero_slip":
            predicted = g_rar
        else:
            predicted = gbar + 3.5 * (g_rar - gbar)
    elif parent == "curvature_additive_alpha10":
        _, predicted = running_acceleration(
            "curvature_additive_alpha10", radius, anchors, running_protocol
        )
    else:
        raise ValueError(parent)
    return gbar, predicted


def build_radial_field(
    parent: str,
    anchors: pd.DataFrame,
    running_protocol: dict,
    local: dict,
    *,
    baryons_only: bool,
) -> RadialDeflectionField:
    cutoff = 3000.0 if parent.startswith("fixed_RAR") else 1.0e6
    radius_grid = np.geomspace(0.1, cutoff, 4096)
    gbar, predicted = parent_acceleration(parent, radius_grid, anchors, running_protocol)
    curve = gbar if baryons_only else predicted

    def lookup(radius):
        return np.exp(np.interp(np.log(radius), np.log(radius_grid), np.log(curve)))

    impact = np.geomspace(0.05, 500.0, 700)
    scale = float(local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"])
    physical = spherical_deflection_radians(
        impact * scale,
        lookup,
        maximum_radius_kpc=cutoff,
        integration_points=800,
    )
    return RadialDeflectionField(impact, physical)


def low_level_wcs(header) -> WCS:
    wcs = WCS(header)
    wcs.sip = None
    return wcs


def prepare_light_maps(
    protocol: dict,
    acquisition: dict,
    system: dict,
    images: pd.DataFrame,
) -> tuple[dict[float, np.ndarray], dict]:
    settings = protocol["light_template"]
    directory = ROOT / acquisition["outputs"]["directory"]
    acquired = next(item for item in acquisition["systems"] if item["label"] == system["label"])
    science_path = directory / acquired["science"]["filename"]
    weight_path = directory / acquired["weight"]["filename"]
    with fits.open(science_path, memmap=True) as science_hdul, fits.open(
        weight_path, memmap=True
    ) as weight_hdul:
        header = science_hdul[0].header
        science = science_hdul[0].data
        weight = weight_hdul[0].data
        wcs = low_level_wcs(header)
        center_x, center_y = wcs.wcs_world2pix(
            [system["center_ra_deg"]], [system["center_dec_deg"]], 0
        )
        center_x, center_y = float(center_x[0]), float(center_y[0])
        native_scale = float(
            np.sqrt(
                abs(
                    np.linalg.det(
                        np.asarray(
                            [
                                [float(header["CD1_1"]), float(header["CD1_2"])],
                                [float(header["CD2_1"]), float(header["CD2_2"])],
                            ]
                        )
                    )
                )
            )
            * 3600.0
        )
        cut_radius = 96.0
        cut_pixels = int(np.ceil(cut_radius / native_scale))
        x0, x1 = int(round(center_x)) - cut_pixels, int(round(center_x)) + cut_pixels + 1
        y0, y1 = int(round(center_y)) - cut_pixels, int(round(center_y)) + cut_pixels + 1
        cut = np.asarray(science[y0:y1, x0:x1], dtype=np.float64)
        cut_weight = np.asarray(weight[y0:y1, x0:x1], dtype=np.float64)
        yy, xx = np.indices(cut.shape, dtype=float)
        x_arcsec = -(xx + x0 - center_x) * native_scale
        y_arcsec = (yy + y0 - center_y) * native_scale
        radius = np.hypot(x_arcsec, y_arcsec)
        valid = (cut_weight > 0.0) & np.isfinite(cut)
        outer = valid & (radius >= 75.0) & (radius <= 90.0)
        background = float(np.median(cut[outer]))
        image_mask = np.zeros_like(valid)
        mask_radius = float(settings["known_lens_image_mask_radius_arcsec"])
        for row in images.itertuples(index=False):
            image_mask |= (
                (x_arcsec - float(row.x_arcsec)) ** 2
                + (y_arcsec - float(row.y_arcsec)) ** 2
                <= mask_radius**2
            )
        usable = valid & ~image_mask
        fill = np.full_like(cut, background)
        annulus_width = 0.5
        bins = np.floor(radius / annulus_width).astype(int)
        for index in range(int(np.ceil(cut_radius / annulus_width)) + 1):
            target = bins == index
            source = target & usable
            if np.any(source):
                fill[target] = float(np.median(cut[source]))
        filled = cut.copy()
        filled[~usable] = fill[~usable]
        positive = np.maximum(filled - background, 0.0)

        size = int(settings["pixels_per_axis"])
        spacing = float(settings["grid_spacing_arcsec"])
        axis = (np.arange(size) - (size - 1) / 2.0) * spacing
        target_x, target_y = np.meshgrid(axis, axis, indexing="xy")
        cosine = math.cos(math.radians(float(system["center_dec_deg"])))
        target_ra = float(system["center_ra_deg"]) + target_x / (3600.0 * cosine)
        target_dec = float(system["center_dec_deg"]) + target_y / 3600.0
        pixel_x, pixel_y = wcs.wcs_world2pix(target_ra, target_dec, 0)
        coordinates = np.vstack([(pixel_y - y0).ravel(), (pixel_x - x0).ravel()])
        light_maps = {}
        scale_kpc = float(system["angular_scale_kpc_per_arcsec"])
        for smoothing in protocol["factor_grid"]["smoothing_kpc"]:
            sigma_pixel = float(smoothing) / (scale_kpc * native_scale)
            smoothed = gaussian_filter(positive, sigma=sigma_pixel, mode="nearest")
            sampled = map_coordinates(
                smoothed,
                coordinates,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            ).reshape(target_x.shape)
            sampled[np.hypot(target_x, target_y) > float(settings["complete_data_radius_arcsec"])] = 0.0
            light_maps[float(smoothing)] = np.maximum(sampled, 0.0)
        audit = {
            "label": system["label"],
            "native_pixel_scale_arcsec": native_scale,
            "background_electrons_per_s": background,
            "masked_image_pixels": int(np.sum(image_mask)),
            "known_images_masked": int(len(images)),
            "valid_fraction_within_60_arcsec": float(np.mean(valid[radius <= 60.0])),
            "positive_light_fraction_within_60_arcsec": {
                str(key): float(np.mean(value[np.hypot(target_x, target_y) <= 60.0] > 0.0))
                for key, value in light_maps.items()
            },
        }
        return light_maps, audit


def build_contexts(protocol: dict) -> tuple[list[Context], dict, dict[str, str]]:
    acquisition_path = ROOT / protocol["inputs"]["acquisition_protocol"]
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    running_path = ROOT / protocol["inputs"]["curvature_protocol"]
    running_protocol = json.loads(running_path.read_text(encoding="utf-8"))
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    tian = pd.read_csv(
        ROOT / protocol["inputs"]["baryonic_profile"],
        sep=r"\s+",
        names=["system", "radius_kpc", "log_gbar", "log_gobs", "err_log_gbar", "err_log_gobs"],
    )
    systems = {item["label"]: item for item in acquisition["systems"]}
    input_hashes = {
        key: sha256(ROOT / value)
        for key, value in protocol["inputs"].items()
        if isinstance(value, str) and (ROOT / value).is_file()
    }
    provenance = json.loads(
        (ROOT / protocol["inputs"]["acquisition_provenance"]).read_text(encoding="utf-8")
    )
    for row in provenance["files"]:
        path = ROOT / row["path"]
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"F160W input hash changed: {path}")
        input_hashes[Path(row["path"]).name] = row["sha256"]

    contexts = []
    for raw_system in running_protocol["systems"][:4]:
        system = {**raw_system, **systems[raw_system["label"]]}
        local = system_protocol(running_protocol, system)
        local["optimization"]["maximum_function_evaluations"] = int(
            protocol["optimization"]["maximum_function_evaluations"]
        )
        images = load_system_images(catalog, system)
        training, heldout = predictive_split(images)
        anchors = load_anchors(tian, system["label"])
        parent_fields = {}
        baryon_fields = {}
        for parent in PARENTS:
            parent_fields[parent] = build_radial_field(
                parent, anchors, running_protocol, local, baryons_only=False
            )
            baryon_fields[parent] = build_radial_field(
                parent, anchors, running_protocol, local, baryons_only=True
            )
        system["angular_scale_kpc_per_arcsec"] = float(
            local["cosmology_and_coordinates"]["angular_scale_kpc_per_arcsec"]
        )
        light_maps, light_audit = prepare_light_maps(
            protocol, acquisition, system, images
        )
        contexts.append(
            Context(
                system,
                local,
                training,
                heldout,
                anchors,
                parent_fields,
                baryon_fields,
                light_maps,
                light_audit,
            )
        )
    return contexts, running_protocol, input_hashes


def morphology_field(
    protocol: dict,
    context: Context,
    parent: str,
    carrier: str,
    smoothing_kpc: float,
    contrast_cap: float,
) -> StellarMorphologyDeflectionField:
    size = int(protocol["light_template"]["pixels_per_axis"])
    spacing = float(protocol["light_template"]["grid_spacing_arcsec"])
    axis = (np.arange(size) - (size - 1) / 2.0) * spacing
    parent_field = context.parent_fields[parent]
    baryon_field = context.baryon_fields[parent]

    def alpha(radius):
        radius = np.asarray(radius, dtype=float)
        parent_alpha = parent_field.reduced_alpha_arcsec(radius, 1.0)
        baryon_alpha = baryon_field.reduced_alpha_arcsec(radius, 1.0)
        if carrier == "baryonic":
            return baryon_alpha
        if carrier == "extra":
            return parent_alpha - baryon_alpha
        if carrier == "full":
            return parent_alpha
        raise ValueError(carrier)

    settings = protocol["light_template"]
    return build_stellar_morphology_deflection_field(
        axis,
        context.light_maps[float(smoothing_kpc)],
        alpha,
        contrast_cap=float(contrast_cap),
        annulus_width_arcsec=float(settings["annular_normalization_width_arcsec"]),
        taper_inner_arcsec=55.0,
        support_radius_arcsec=float(settings["complete_data_radius_arcsec"]),
        radial_samples=2048,
        circular_radii=512,
        circular_azimuths=720,
    )


def check_field_audit(protocol: dict, field: StellarMorphologyDeflectionField) -> None:
    limits = protocol["numerical_audits"]
    audit = field.audit
    checks = {
        "annular_weight": audit["maximum_carrier_weighted_annular_mean_error"]
        <= float(limits["maximum_annular_weight_mean_error"]),
        "annular_convergence": audit["maximum_annular_convergence_mean_fraction"]
        <= float(limits["maximum_annular_convergence_mean_fraction"]),
        "circular_mean": audit["maximum_independent_circular_mean_deflection_arcsec"]
        <= float(limits["maximum_independent_circular_mean_deflection_arcsec"]),
        "curl": audit["normalized_curl_RMS"] <= float(limits["maximum_normalized_curl_RMS"]),
        "source_edge": audit["maximum_edge_delta_convergence"]
        <= float(limits["maximum_edge_delta_convergence"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"morphology field audit failed: {checks}; {audit}")


def field_audit_row(context, parent, carrier, smoothing, cap, field):
    return {
        "label": context.system["label"],
        "parent": parent,
        "carrier": carrier,
        "smoothing_kpc": float(smoothing),
        "contrast_cap": float(cap),
        **field.audit,
    }


def optimization_rms(lens, parent, parameters, rows):
    residual, _ = lens.profiled_residuals(parent, np.asarray(parameters), rows)
    image = residual.reshape(-1, 2) * lens.sigma
    return float(np.sqrt(np.mean(np.sum(image**2, axis=1))))


def build_lens(context, parent, field, fraction):
    return MorphologyLens(
        context.local,
        {parent: context.parent_fields[parent]},
        parent=parent,
        morphology=field,
        fraction=float(fraction),
    )


def candidate_key(row) -> tuple:
    carrier_order = {"baryonic": 0, "extra": 1, "full": 2, "none": 3}
    exact = row.get("exact_training_RMS_arcsec")
    exact_available = exact is not None and np.isfinite(float(exact))
    ranking_rms = (
        float(exact)
        if exact_available
        else float(row["screen_equal_system_RMS_arcsec"])
    )
    return (
        0 if exact_available else 1,
        ranking_rms,
        float(row["redistribution_fraction"]),
        -float(row["smoothing_kpc"]),
        float(row["contrast_cap"]),
        carrier_order[row["carrier"]],
    )


def factor_effects(screen: pd.DataFrame) -> pd.DataFrame:
    records = []
    factors = ["carrier", "smoothing_kpc", "contrast_cap", "redistribution_fraction"]
    for parent, block in screen[screen.redistribution_fraction.gt(0.0)].groupby("parent"):
        y = block.screen_equal_system_RMS_arcsec.to_numpy(float)
        grand = float(np.mean(y))
        total = float(np.sum(np.square(y - grand)))
        explained = 0.0
        for factor in factors:
            means = block.groupby(factor).screen_equal_system_RMS_arcsec.mean()
            multiplier = len(block) / len(means)
            ss = float(multiplier * np.sum(np.square(means.to_numpy(float) - grand)))
            explained += ss
            records.append({"parent": parent, "effect": factor, "sum_squares": ss})
        for left, right in combinations(factors, 2):
            left_mean = block.groupby(left).screen_equal_system_RMS_arcsec.mean()
            right_mean = block.groupby(right).screen_equal_system_RMS_arcsec.mean()
            cell = block.groupby([left, right]).screen_equal_system_RMS_arcsec.mean()
            ss = 0.0
            multiplier = len(block) / len(cell)
            for (a, b), value in cell.items():
                interaction = float(value - left_mean.loc[a] - right_mean.loc[b] + grand)
                ss += multiplier * interaction**2
            explained += ss
            records.append({"parent": parent, "effect": f"{left}_x_{right}", "sum_squares": ss})
        records.append(
            {
                "parent": parent,
                "effect": "higher_order",
                "sum_squares": max(0.0, total - explained),
            }
        )
        for record in records:
            if record["parent"] == parent:
                record["total_sum_squares"] = total
                record["variance_percent"] = 100.0 * record["sum_squares"] / max(
                    total, np.finfo(float).tiny
                )
    return pd.DataFrame(records)


def save_light_figure(contexts, output: Path):
    figure, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    for axis_plot, context in zip(axes.flat, contexts):
        light = context.light_maps[10.0]
        size = light.shape[0]
        spacing = 1.0
        axis = (np.arange(size) - (size - 1) / 2.0) * spacing
        selected = np.abs(axis) <= 70.0
        cut = light[np.ix_(selected, selected)]
        vmax = float(np.quantile(cut[cut > 0.0], 0.995))
        axis_plot.imshow(
            np.arcsinh(cut / max(vmax / 10.0, np.finfo(float).tiny)),
            origin="lower",
            cmap="magma",
            extent=[-70, 70, -70, 70],
        )
        axis_plot.scatter(
            context.training.x_arcsec,
            context.training.y_arcsec,
            marker="x",
            c="cyan",
            s=22,
            linewidths=0.7,
        )
        axis_plot.scatter(
            context.heldout.x_arcsec,
            context.heldout.y_arcsec,
            marker="x",
            c="white",
            s=30,
            linewidths=0.9,
        )
        axis_plot.set(
            title=f"{context.system['label']} masked F160W, 10 kpc",
            xlabel="RA offset (arcsec)",
            ylabel="Dec offset (arcsec)",
            xlim=(70, -70),
            ylim=(-70, 70),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def run_map_audit(protocol, contexts, output):
    rows = []
    for context, parent, carrier, smoothing, cap in product(
        contexts,
        PARENTS,
        protocol["factor_grid"]["carrier"],
        protocol["factor_grid"]["smoothing_kpc"],
        protocol["factor_grid"]["contrast_cap"],
    ):
        print(
            f"map {context.system['label']} {parent} {carrier} l={smoothing} c={cap}",
            flush=True,
        )
        field = morphology_field(protocol, context, parent, carrier, smoothing, cap)
        check_field_audit(protocol, field)
        rows.append(field_audit_row(context, parent, carrier, smoothing, cap, field))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "map_audit.csv", index=False)
    report = {
        "status": "all frozen map-construction audits passed",
        "fields": len(frame),
        "systems": len(contexts),
        "maximum_audits": {
            key: float(frame[key].max())
            for key in [
                "maximum_carrier_weighted_annular_mean_error",
                "maximum_annular_convergence_mean_fraction",
                "maximum_independent_circular_mean_deflection_arcsec",
                "normalized_curl_RMS",
                "maximum_edge_delta_convergence",
            ]
        },
        "light_preprocessing": [context.light_audit for context in contexts],
    }
    (output / "map_audit_report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    save_light_figure(contexts, output / "masked_light_maps.png")
    return report


def run_full(protocol, contexts, input_hashes, output):
    grid = protocol["factor_grid"]
    seed = int(protocol["optimization"]["random_seed"])
    selection_labels = set(protocol["systems_and_split"]["selection_labels"])
    validation_labels = set(protocol["systems_and_split"]["validation_labels"])
    selection_contexts = [c for c in contexts if c.system["label"] in selection_labels]
    controls = {}
    control_rows = []
    control_predictions = []
    control_geometry = []
    print("fit exact f=0 controls", flush=True)
    for parent_index, parent in enumerate(PARENTS):
        for system_index, context in enumerate(contexts):
            lens = build_lens(context, parent, None, 0.0)
            fit = lens.fit(
                parent,
                context.training,
                starts=int(grid["final_replay_starts"]),
                seed=seed + 1000 * parent_index + 20 * system_index,
            )
            train = lens.exact_predictions(
                parent, fit["result"].x, fit["sources"], context.training, stage="training"
            )
            hold = lens.exact_predictions(
                parent, fit["result"].x, fit["sources"], context.heldout, stage="heldout"
            )
            for frame in (train, hold):
                frame.insert(0, "system", context.system["system"])
                frame.insert(1, "system_label", context.system["label"])
                frame.insert(2, "variant", "control_f0")
                frame["carrier"] = "none"
                frame["smoothing_kpc"] = 0.0
                frame["contrast_cap"] = 0.0
                frame["redistribution_fraction"] = 0.0
                control_predictions.append(frame)
            train_score = score(train, lens.sigma, free_parameters=6)
            hold_score = score(hold, lens.sigma, free_parameters=0)
            controls[(context.system["label"], parent)] = {
                "fit": fit,
                "training_score": train_score,
                "heldout_score": hold_score,
            }
            control_rows.append(
                {
                    "parent": parent,
                    "label": context.system["label"],
                    "stage": "training",
                    **train_score,
                }
            )
            control_rows.append(
                {
                    "parent": parent,
                    "label": context.system["label"],
                    "stage": "heldout",
                    **hold_score,
                }
            )
            control_geometry.append(
                {
                    "parent": parent,
                    "variant": "control_f0",
                    "system": context.system["system"],
                    "system_label": context.system["label"],
                    **dict(zip(spec_for(parent).labels, fit["result"].x)),
                    "optimization_cost": float(fit["result"].cost),
                    "geometry_at_boundary": any(near_bound(parent, fit["result"].x).values()),
                }
            )

    print("factor screen", flush=True)
    screen_rows = []
    map_audit_rows = []
    for parent in PARENTS:
        control_rms = [
            optimization_rms(
                build_lens(context, parent, None, 0.0),
                parent,
                controls[(context.system["label"], parent)]["fit"]["result"].x,
                context.training,
            )
            for context in selection_contexts
        ]
        screen_rows.append(
            {
                "parent": parent,
                "carrier": "none",
                "smoothing_kpc": 0.0,
                "contrast_cap": 0.0,
                "redistribution_fraction": 0.0,
                "screen_equal_system_RMS_arcsec": float(np.sqrt(np.mean(np.square(control_rms)))),
            }
        )
        for carrier, smoothing, cap in product(
            grid["carrier"], grid["smoothing_kpc"], grid["contrast_cap"]
        ):
            fields = {}
            for context in selection_contexts:
                field = morphology_field(protocol, context, parent, carrier, smoothing, cap)
                check_field_audit(protocol, field)
                fields[context.system["label"]] = field
                map_audit_rows.append(
                    field_audit_row(context, parent, carrier, smoothing, cap, field)
                )
            for fraction in grid["redistribution_fraction"]:
                if float(fraction) == 0.0:
                    continue
                values = []
                row = {
                    "parent": parent,
                    "carrier": carrier,
                    "smoothing_kpc": float(smoothing),
                    "contrast_cap": float(cap),
                    "redistribution_fraction": float(fraction),
                }
                for context in selection_contexts:
                    lens = build_lens(
                        context, parent, fields[context.system["label"]], fraction
                    )
                    value = optimization_rms(
                        lens,
                        parent,
                        controls[(context.system["label"], parent)]["fit"]["result"].x,
                        context.training,
                    )
                    row[f"{context.system['label']}_screen_RMS_arcsec"] = value
                    values.append(value)
                row["screen_equal_system_RMS_arcsec"] = float(
                    np.sqrt(np.mean(np.square(values)))
                )
                screen_rows.append(row)
    screen = pd.DataFrame(screen_rows)
    screen.to_csv(output / "factor_screen.csv", index=False)
    pd.DataFrame(map_audit_rows).drop_duplicates(
        ["label", "parent", "carrier", "smoothing_kpc", "contrast_cap"]
    ).to_csv(output / "selection_map_audit.csv", index=False)
    effects = factor_effects(screen)
    effects.to_csv(output / "factor_effects.csv", index=False)

    print("shortlist refits", flush=True)
    refit_rows = []
    selected = {}
    diagnostic_best = {}
    for parent_index, parent in enumerate(PARENTS):
        block = screen[screen.parent.eq(parent)].sort_values("screen_equal_system_RMS_arcsec")
        shortlist = block.head(int(grid["shortlist_per_parent"])).to_dict("records")
        control = screen[(screen.parent.eq(parent)) & screen.redistribution_fraction.eq(0.0)].iloc[0].to_dict()
        if not any(float(row["redistribution_fraction"]) == 0.0 for row in shortlist):
            shortlist.append(control)
        parent_refits = []
        for candidate_index, candidate in enumerate(shortlist):
            scores = []
            complete = True
            for system_index, context in enumerate(selection_contexts):
                fraction = float(candidate["redistribution_fraction"])
                field = None
                if fraction > 0.0:
                    field = morphology_field(
                        protocol,
                        context,
                        parent,
                        candidate["carrier"],
                        candidate["smoothing_kpc"],
                        candidate["contrast_cap"],
                    )
                    check_field_audit(protocol, field)
                lens = build_lens(context, parent, field, fraction)
                fit = lens.fit(
                    parent,
                    context.training,
                    starts=int(grid["selection_refit_starts"]),
                    seed=seed + 100000 + parent_index * 10000 + candidate_index * 100 + system_index,
                    initial_override=controls[(context.system["label"], parent)]["fit"]["result"].x,
                )
                prediction = lens.exact_predictions(
                    parent,
                    fit["result"].x,
                    fit["sources"],
                    context.training,
                    stage="selection_training",
                )
                current = score(prediction, lens.sigma, free_parameters=6)
                scores.append(current)
                complete &= bool(current["all_roots_converged"])
            aggregate = aggregate_system_scores(scores)
            row = {
                **candidate,
                "exact_training_RMS_arcsec": aggregate["equal_system_radial_RMS_arcsec"],
                "all_training_roots_converged": complete,
            }
            refit_rows.append(row)
            parent_refits.append(row)
        stable = [row for row in parent_refits if row["all_training_roots_converged"]]
        diagnostic_best[parent] = min(parent_refits, key=candidate_key)
        if stable:
            selected[parent] = min(stable, key=candidate_key)
        else:
            selected[parent] = None
            print(
                f"{parent}: no complete-root selection candidate; "
                "retaining diagnostics and skipping validation replay",
                flush=True,
            )
    pd.DataFrame(refit_rows).to_csv(output / "selection_refits.csv", index=False)

    print("final eight-start replay", flush=True)
    final_predictions = list(control_predictions)
    final_geometry = list(control_geometry)
    final_scores = {}
    for parent_index, parent in enumerate(PARENTS):
        candidate = selected[parent]
        final_scores[parent] = {"control_f0": {}, "selected_morphology": {}}
        for variant in ("control_f0", "selected_morphology"):
            if variant == "selected_morphology" and candidate is None:
                final_scores[parent][variant] = {
                    "status": "no_complete_root_selection_candidate"
                }
                continue
            score_rows = []
            for system_index, context in enumerate(contexts):
                if variant == "control_f0":
                    current = controls[(context.system["label"], parent)]
                    score_rows.append(
                        {
                            "label": context.system["label"],
                            "stage": "heldout",
                            **current["heldout_score"],
                        }
                    )
                    continue
                fraction = float(candidate["redistribution_fraction"])
                if fraction == 0.0:
                    current = controls[(context.system["label"], parent)]
                    score_rows.append(
                        {
                            "label": context.system["label"],
                            "stage": "heldout",
                            **current["heldout_score"],
                        }
                    )
                    continue
                field = morphology_field(
                    protocol,
                    context,
                    parent,
                    candidate["carrier"],
                    candidate["smoothing_kpc"],
                    candidate["contrast_cap"],
                )
                check_field_audit(protocol, field)
                lens = build_lens(context, parent, field, fraction)
                fit = lens.fit(
                    parent,
                    context.training,
                    starts=int(grid["final_replay_starts"]),
                    seed=seed + 200000 + parent_index * 1000 + system_index * 20,
                    initial_override=controls[(context.system["label"], parent)]["fit"]["result"].x,
                )
                train = lens.exact_predictions(
                    parent, fit["result"].x, fit["sources"], context.training, stage="training"
                )
                hold = lens.exact_predictions(
                    parent, fit["result"].x, fit["sources"], context.heldout, stage="heldout"
                )
                for frame in (train, hold):
                    frame.insert(0, "system", context.system["system"])
                    frame.insert(1, "system_label", context.system["label"])
                    frame.insert(2, "variant", variant)
                    frame["carrier"] = candidate["carrier"]
                    frame["smoothing_kpc"] = candidate["smoothing_kpc"]
                    frame["contrast_cap"] = candidate["contrast_cap"]
                    frame["redistribution_fraction"] = candidate["redistribution_fraction"]
                    final_predictions.append(frame)
                heldout_score = score(hold, lens.sigma, free_parameters=0)
                score_rows.append(
                    {"label": context.system["label"], "stage": "heldout", **heldout_score}
                )
                final_geometry.append(
                    {
                        "parent": parent,
                        "variant": variant,
                        "system": context.system["system"],
                        "system_label": context.system["label"],
                        **dict(zip(spec_for(parent).labels, fit["result"].x)),
                        "optimization_cost": float(fit["result"].cost),
                        "geometry_at_boundary": any(near_bound(parent, fit["result"].x).values()),
                    }
                )
            for split_name, labels in {
                "selection_heldout": selection_labels,
                "validation_heldout": validation_labels,
                "all_four_heldout": selection_labels | validation_labels,
            }.items():
                subset = [row for row in score_rows if row["label"] in labels]
                final_scores[parent][variant][split_name] = aggregate_system_scores(subset)

    predictions = pd.concat(final_predictions, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(final_geometry).to_csv(output / "geometry.csv", index=False)
    pd.DataFrame(control_rows).to_csv(output / "control_scores.csv", index=False)

    outcomes = {}
    for parent in PARENTS:
        base = final_scores[parent]["control_f0"]["validation_heldout"]
        common = {
            "galaxy_outer_RMSE_km_s": protocol["parent_radial_laws"][parent][
                "locked_galaxy_outer_RMSE_km_s"
            ],
            "galaxy_prediction_changed": False,
            "radial_bridge_prediction_changed": False,
            "Solar_point_mass_prediction_changed": False,
        }
        if selected[parent] is None:
            outcomes[parent] = {
                "status": "no_complete_root_selection_candidate",
                "selected": None,
                "diagnostic_best_incomplete": diagnostic_best[parent],
                "control_validation": base,
                "selected_validation": None,
                "validation_fractional_improvement": None,
                **common,
            }
        else:
            chosen = final_scores[parent]["selected_morphology"]["validation_heldout"]
            outcomes[parent] = {
                "status": "validation_replayed",
                "selected": selected[parent],
                "diagnostic_best_incomplete": diagnostic_best[parent],
                "control_validation": base,
                "selected_validation": chosen,
                "validation_fractional_improvement": (
                    1.0
                    - float(chosen["equal_system_radial_RMS_arcsec"])
                    / float(base["equal_system_radial_RMS_arcsec"])
                    if base["equal_system_radial_RMS_arcsec"]
                    and chosen["equal_system_radial_RMS_arcsec"]
                    else None
                ),
                **common,
            }

    report = {
        "report_version": protocol["protocol_version"],
        "status": "completed continuous-light morphology response test",
        "protocol": {
            "path": "configs/clash_stellar_morphology_response_protocol.json",
            "sha256": sha256(ROOT / "configs/clash_stellar_morphology_response_protocol.json"),
        },
        "input_hashes": input_hashes,
        "coverage": {
            "parents": len(PARENTS),
            "factor_cells_per_parent_excluding_control": int(
                len(grid["carrier"])
                * len(grid["smoothing_kpc"])
                * len(grid["contrast_cap"])
                * (len(grid["redistribution_fraction"]) - 1)
            ),
            "screen_rows": len(screen),
            "selection_refits": len(refit_rows),
            "raw_clusters": len(contexts),
            "heldout_images": int(sum(len(context.heldout) for context in contexts)),
        },
        "selected_settings": selected,
        "diagnostic_best_incomplete_settings": diagnostic_best,
        "scores": final_scores,
        "outcomes": outcomes,
        "factor_effects": effects.to_dict("records"),
        "references": protocol["references"],
        "claim_boundary": protocol["pre_score_disclosure"],
    }
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_result_figure(report, effects, output / "stellar_morphology_response.png")
    write_summary(report, output / "SUMMARY.md")
    return report


def make_result_figure(report, effects, output):
    def finite_or_nan(value):
        return float(value) if value is not None and np.isfinite(float(value)) else np.nan

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x = np.arange(len(PARENTS))
    base = [
        finite_or_nan(
            report["scores"][parent]["control_f0"]["validation_heldout"][
                "equal_system_radial_RMS_arcsec"
            ]
        )
        for parent in PARENTS
    ]
    chosen = [
        (
            finite_or_nan(
                report["scores"][parent]["selected_morphology"]["validation_heldout"][
                    "equal_system_radial_RMS_arcsec"
                ]
            )
            if report["outcomes"][parent]["selected_validation"] is not None
            else np.nan
        )
        for parent in PARENTS
    ]
    axes[0].bar(x - 0.18, base, 0.36, label="radial parent")
    axes[0].bar(x + 0.18, chosen, 0.36, label="selected F160W morphology")
    axes[0].axhline(report["references"]["raw_compact_halo_RMS_arcsec"], color="black", ls="--", label="compact halo")
    axes[0].set_xticks(x, [name.replace("fixed_RAR_", "RAR ").replace("curvature_additive_", "curv ") for name in PARENTS], rotation=20)
    axes[0].set(ylabel="validation held-out RMS (arcsec)", title="Cross-cluster transfer")
    axes[0].legend(fontsize=8)
    main = effects[effects.effect.isin(["carrier", "smoothing_kpc", "contrast_cap", "redistribution_fraction"])]
    width = 0.24
    for index, parent in enumerate(PARENTS):
        block = main[main.parent.eq(parent)].set_index("effect")
        labels = ["carrier", "smoothing_kpc", "contrast_cap", "redistribution_fraction"]
        axes[1].bar(
            np.arange(len(labels)) + (index - 1) * width,
            [block.loc[label, "variance_percent"] for label in labels],
            width,
            label=parent.replace("fixed_RAR_", "RAR ").replace("curvature_additive_", "curv "),
        )
    axes[1].set_xticks(np.arange(4), ["carrier", "smoothing", "cap", "fraction"], rotation=15)
    axes[1].set(ylabel="screen variance explained (%)", title="Which formula change moves predictions?")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_summary(report, output):
    def metric(value):
        return f"{float(value):.3f}" if value is not None else "n/a"

    def percent(value):
        return f"{100.0 * float(value):+.1f}%" if value is not None else "n/a"

    lines = [
        "# CLASH continuous stellar-morphology response",
        "",
        "The experiment redistributes, but never adds, the radial convergence budget according to masked CLASH F160W light. Galaxy, radial-bridge, and Solar point-mass predictions are exact parent controls.",
        "",
        "| Parent | Selected carrier | smoothing (kpc) | cap | fraction | control validation RMS | selected validation RMS | change | roots |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for parent in PARENTS:
        outcome = report["outcomes"][parent]
        selected = outcome["selected"]
        base = outcome["control_validation"]["equal_system_radial_RMS_arcsec"]
        if selected is None:
            diagnostic = outcome["diagnostic_best_incomplete"]
            lines.append(
                f"| {parent} | no admissible setting (diagnostic: {diagnostic['carrier']}) | "
                f"{diagnostic['smoothing_kpc']:.1f} | {diagnostic['contrast_cap']:.1f} | "
                f"{diagnostic['redistribution_fraction']:.3f} | {metric(base)} | n/a | n/a | False |"
            )
            continue
        result = outcome["selected_validation"]["equal_system_radial_RMS_arcsec"]
        change = outcome["validation_fractional_improvement"]
        roots = outcome["selected_validation"]["all_roots_converged"]
        lines.append(
            f"| {parent} | {selected['carrier']} | {selected['smoothing_kpc']:.1f} | {selected['contrast_cap']:.1f} | {selected['redistribution_fraction']:.3f} | {metric(base)} | {metric(result)} | {percent(change)} | {roots} |"
        )
    lines.extend(
        [
            "",
            "The percentages above are prediction changes, not probabilities that a theory is true. See `report.json` and `factor_effects.csv` for the complete factorial and claim boundaries.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/clash_stellar_morphology_response_protocol.json"
    )
    parser.add_argument("--stage", choices=("map_audit", "full"), default="full")
    args = parser.parse_args()
    config_path = ROOT / args.config
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_light_template_construction_or_lens_scores":
        raise RuntimeError("response protocol is not frozen")
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    contexts, _, input_hashes = build_contexts(protocol)
    if args.stage == "map_audit":
        report = run_map_audit(protocol, contexts, output)
    else:
        map_report = output / "map_audit_report.json"
        if not map_report.exists():
            raise RuntimeError("run --stage map_audit before computing lens scores")
        report = run_full(protocol, contexts, input_hashes, output)
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
