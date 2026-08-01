#!/usr/bin/env python3
"""Run the frozen P0557 registered-baryon proxy tidal experiment."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_member_tidal_metric import (  # noqa: E402
    MemberTidalLens,
    build_contexts,
    fit_context,
    fixed_source_local_rms,
    model_name,
)
from run_p0554_all_baryon_route_screen import (  # noqa: E402
    prepare_hst_map,
    prepare_xray_maps,
)
from run_unbounded_running_multicluster_raw import aggregate_system_scores  # noqa: E402
from voidscreen.tidal_metric import build_tidal_correction_field  # noqa: E402


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(values) -> float:
    values = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def normalized_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["x_arcsec", "y_arcsec", "normalized_light_weight"]].copy()
    weights = result.normalized_light_weight.to_numpy(float)
    if len(result) < 3 or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("proxy catalog is invalid")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("proxy catalog has no positive weight")
    result["normalized_light_weight"] = weights / total
    return result


def compressed_map_catalog(axis, image, *, block_pixels: int, transform: str) -> pd.DataFrame:
    """Compress a registered positive map into deterministic weighted pseudo-sources."""

    axis = np.asarray(axis, dtype=float)
    values = np.maximum(np.asarray(image, dtype=float), 0.0)
    if values.shape != (len(axis), len(axis)):
        raise ValueError("registered map and axis shape disagree")
    if transform == "sqrt":
        values = np.sqrt(values)
    elif transform != "linear":
        raise ValueError(transform)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    rows = []
    step = int(block_pixels)
    for y0 in range(0, len(axis), step):
        for x0 in range(0, len(axis), step):
            cut = values[y0 : y0 + step, x0 : x0 + step]
            total = float(cut.sum())
            if total <= 0.0:
                continue
            cut_x = xx[y0 : y0 + step, x0 : x0 + step]
            cut_y = yy[y0 : y0 + step, x0 : x0 + step]
            rows.append(
                {
                    "x_arcsec": float(np.sum(cut * cut_x) / total),
                    "y_arcsec": float(np.sum(cut * cut_y) / total),
                    "normalized_light_weight": total,
                }
            )
    return normalized_catalog(pd.DataFrame(rows))


def catalog_audit(system: str, component: str, frame: pd.DataFrame) -> dict:
    weights = frame.normalized_light_weight.to_numpy(float)
    x = frame.x_arcsec.to_numpy(float)
    y = frame.y_arcsec.to_numpy(float)
    cx = float(np.sum(weights * x))
    cy = float(np.sum(weights * y))
    return {
        "system_label": system,
        "component": component,
        "pseudo_sources": int(len(frame)),
        "weight_sum": float(weights.sum()),
        "centroid_x_arcsec": cx,
        "centroid_y_arcsec": cy,
        "weighted_RMS_radius_arcsec": float(
            np.sqrt(np.sum(weights * (np.square(x - cx) + np.square(y - cy))))
        ),
        "maximum_weight": float(weights.max()),
    }


def prepare_component_catalogs(protocol: dict, contexts) -> tuple[dict, list[dict]]:
    all_baryon_path = ROOT / protocol["inputs"]["all_baryon_protocol"]
    all_baryon = json.loads(all_baryon_path.read_text(encoding="utf-8"))
    acquisition = json.loads(
        (ROOT / all_baryon["inputs"]["acquisition_protocol"]).read_text(encoding="utf-8")
    )
    reused = json.loads(
        (ROOT / all_baryon["inputs"]["reused_hst_protocol"]).read_text(encoding="utf-8")
    )
    proxy = protocol["proxy_maps"]
    axis = np.arange(
        float(proxy["axis_min_arcsec"]),
        float(proxy["axis_max_arcsec"]) + 0.5 * float(proxy["grid_spacing_arcsec"]),
        float(proxy["grid_spacing_arcsec"]),
    )
    catalogs = {}
    audits = []
    for context in contexts:
        label = str(context.system["label"])
        print(f"registered proxy maps: {label}", flush=True)
        adapter = SimpleNamespace(label=label, local=context.local_protocol)
        known_images = pd.concat([context.training, context.heldout], ignore_index=True)
        star_map, star_map_audit = prepare_hst_map(
            all_baryon, acquisition, reused, adapter, known_images, axis
        )
        _, gas_map, gas_map_audit = prepare_xray_maps(
            all_baryon, acquisition, adapter, axis
        )
        block = int(proxy["compression_block_pixels"])
        member = normalized_catalog(context.members)
        star = compressed_map_catalog(axis, star_map, block_pixels=block, transform="linear")
        gas_linear = compressed_map_catalog(
            axis, gas_map, block_pixels=block, transform="linear"
        )
        gas_sqrt = compressed_map_catalog(
            axis, gas_map, block_pixels=block, transform="sqrt"
        )
        catalogs[label] = {
            "member": member,
            "star": star,
            "gas_linear": gas_linear,
            "gas_sqrt": gas_sqrt,
        }
        for name, frame in catalogs[label].items():
            row = catalog_audit(label, name, frame)
            if name == "star":
                row.update(
                    {
                        "native_positive_cells": star_map_audit["positive_cells"],
                        "input_count": star_map_audit["input_count"],
                    }
                )
            if name.startswith("gas_"):
                row.update(
                    {
                        "native_positive_cells": int(np.sum(gas_map > 0.0)),
                        "input_count": gas_map_audit["input_count"],
                        "total_exposure_ks": gas_map_audit["total_exposure_ks"],
                    }
                )
            audits.append(row)
    return catalogs, audits


def mixed_catalog(components: dict[str, pd.DataFrame], variant: dict) -> pd.DataFrame:
    pieces = []
    mapping = [
        ("member", float(variant["member_fraction"])),
        ("star", float(variant["star_fraction"])),
        (f"gas_{variant['gas_transform']}", float(variant["gas_fraction"])),
    ]
    for component, fraction in mapping:
        if fraction <= 0.0:
            continue
        part = components[component].copy()
        part["normalized_light_weight"] *= fraction
        pieces.append(part)
    if not pieces:
        raise ValueError(f"empty morphology variant {variant['variant_id']}")
    return normalized_catalog(pd.concat(pieces, ignore_index=True))


def proxy_correction(
    context,
    catalog: pd.DataFrame,
    protocol: dict,
    *,
    operator: dict,
    pixels_per_axis: int,
    softening_kpc: float,
):
    scale = float(
        context.local_protocol["cosmology_and_coordinates"][
            "angular_scale_kpc_per_arcsec"
        ]
    )
    env = protocol["environment_tensor"]
    return build_tidal_correction_field(
        catalog.x_arcsec.to_numpy(float),
        catalog.y_arcsec.to_numpy(float),
        catalog.normalized_light_weight.to_numpy(float),
        softening_arcsec=float(softening_kpc) / scale,
        extra_alpha_arcsec=context.extra_alpha,
        half_width_arcsec=float(env["field_half_width_arcsec"]),
        pixels_per_axis=int(pixels_per_axis),
        polar_mean_radii=int(env["polar_mean_radii"]),
        polar_mean_azimuths=int(env["polar_mean_azimuths"]),
        subtract_circular_mean=bool(operator["subtract_circular_mean"]),
    )


def candidate_key(variant_id: str, operator_id: str) -> str:
    return f"{variant_id}__{operator_id}"


def build_candidate_context(
    base,
    catalog,
    protocol,
    variant_id,
    operator,
    *,
    pixels_per_axis,
    softening_kpc,
    audit_rows,
    stage,
):
    correction = proxy_correction(
        base,
        catalog,
        protocol,
        operator=operator,
        pixels_per_axis=pixels_per_axis,
        softening_kpc=softening_kpc,
    )
    audit_rows.append(
        {
            "stage": stage,
            "system_label": base.system["label"],
            "variant_id": variant_id,
            "operator_id": operator["operator_id"],
            "pixels_per_axis": int(pixels_per_axis),
            "softening_kpc": float(softening_kpc),
            "pseudo_sources": int(len(catalog)),
            **correction.audit,
        }
    )
    return replace(base, members=catalog, correction=correction)


def baseline_fits(protocol, contexts):
    results = {}
    starts = int(protocol["optimization"]["baseline_starts"])
    seed = int(protocol["optimization"]["random_seed"])
    for index, context in enumerate(contexts):
        label = context.system["label"]
        print(f"zero-tensor baseline: {label}", flush=True)
        results[label] = fit_context(context, 0.0, starts=starts, seed=seed + index)
    return results


def screen_candidates(protocol, contexts, catalogs, baselines, tensor_audits):
    selection = set(protocol["cluster_split"]["selection_labels"])
    screen = protocol["stages"]["screen"]
    pixels = int(screen["pixels_per_axis"])
    softening = float(protocol["environment_tensor"]["primary_softening_kpc"])
    nonzero_grid = [float(t) for t in protocol["tensor_coupling"]["grid"] if float(t) != 0.0]
    rows = []
    variants = protocol["morphology_variants"]
    operators = protocol["tensor_operators"]
    for variant in variants:
        variant_id = variant["variant_id"]
        for operator in operators:
            operator_id = operator["operator_id"]
            system_contexts = {}
            for base in contexts:
                label = base.system["label"]
                if label not in selection:
                    continue
                catalog = mixed_catalog(catalogs[label], variant)
                system_contexts[label] = build_candidate_context(
                    base,
                    catalog,
                    protocol,
                    variant_id,
                    operator,
                    pixels_per_axis=pixels,
                    softening_kpc=softening,
                    audit_rows=tensor_audits,
                    stage="fixed_source_screen",
                )
            for coupling in nonzero_grid:
                per_training = {}
                per_heldout = {}
                for label, context in system_contexts.items():
                    baseline = baselines[label]
                    lens = MemberTidalLens(
                        context.local_protocol, context.fields, context.correction, coupling
                    )
                    per_training[label] = fixed_source_local_rms(
                        lens,
                        model_name(coupling),
                        baseline["fit"]["result"].x,
                        baseline["fit"]["sources"],
                        context.training,
                    )
                    per_heldout[label] = fixed_source_local_rms(
                        lens,
                        model_name(coupling),
                        baseline["fit"]["result"].x,
                        baseline["fit"]["sources"],
                        context.heldout,
                    )
                rows.append(
                    {
                        "variant_id": variant_id,
                        "operator_id": operator_id,
                        "tensor_t": coupling,
                        "selection_training_local_RMS_arcsec": rms(per_training.values()),
                        "selection_heldout_local_RMS_arcsec": rms(per_heldout.values()),
                        **{f"{key}_training_local_RMS_arcsec": value for key, value in per_training.items()},
                        **{f"{key}_heldout_local_RMS_arcsec": value for key, value in per_heldout.items()},
                    }
                )
    frame = pd.DataFrame(rows).sort_values("selection_training_local_RMS_arcsec")
    diverse = (
        frame.sort_values("selection_training_local_RMS_arcsec")
        .groupby(["variant_id", "operator_id"], as_index=False, sort=False)
        .first()
        .sort_values("selection_training_local_RMS_arcsec")
        .head(int(screen["nonzero_candidates_for_exact_refit"]))
    )
    diverse["shortlisted_for_exact_selection"] = True
    shortlist_keys = {
        (row.variant_id, row.operator_id, float(row.tensor_t))
        for row in diverse.itertuples(index=False)
    }
    frame["shortlisted_for_exact_selection"] = [
        (row.variant_id, row.operator_id, float(row.tensor_t)) in shortlist_keys
        for row in frame.itertuples(index=False)
    ]
    return frame, diverse


def exact_selection(protocol, contexts, catalogs, shortlist, baselines, tensor_audits):
    selection = set(protocol["cluster_split"]["selection_labels"])
    stage = protocol["stages"]["exact_selection"]
    pixels = int(stage["pixels_per_axis"])
    softening = float(protocol["environment_tensor"]["primary_softening_kpc"])
    starts = int(stage["starts_per_fit"])
    seed = int(protocol["optimization"]["random_seed"]) + 10000
    variant_lookup = {row["variant_id"]: row for row in protocol["morphology_variants"]}
    operator_lookup = {row["operator_id"]: row for row in protocol["tensor_operators"]}
    rows = []
    predictions = []
    fitted_candidates = {}
    zero_scores = []
    zero_complete = True
    for base in contexts:
        label = base.system["label"]
        if label not in selection:
            continue
        fitted = baselines[label]
        zero_scores.append(fitted["training"])
        roots = bool(
            fitted["training"]["all_roots_converged"]
            and fitted["heldout"]["all_roots_converged"]
        )
        zero_complete &= roots
        rows.append(
            {
                "row_type": "system",
                "variant_id": "zero",
                "operator_id": "zero",
                "tensor_t": 0.0,
                "system_label": label,
                "training_exact_RMS_arcsec": fitted["training"]["exact_radial_RMS_arcsec"],
                "heldout_exact_RMS_arcsec": fitted["heldout"]["exact_radial_RMS_arcsec"],
                "all_training_and_heldout_roots": roots,
            }
        )
    zero_aggregate = aggregate_system_scores(zero_scores)
    rows.append(
        {
            "row_type": "aggregate",
            "variant_id": "zero",
            "operator_id": "zero",
            "tensor_t": 0.0,
            "system_label": "selection",
            "training_exact_RMS_arcsec": zero_aggregate[
                "equal_system_radial_RMS_arcsec"
            ],
            "heldout_exact_RMS_arcsec": None,
            "all_training_and_heldout_roots": zero_complete,
        }
    )
    for candidate_index, candidate in enumerate(shortlist.itertuples(index=False)):
        variant = variant_lookup[candidate.variant_id]
        operator = operator_lookup[candidate.operator_id]
        coupling = float(candidate.tensor_t)
        scores = []
        complete = True
        fits = {}
        for system_index, base in enumerate(contexts):
            label = base.system["label"]
            if label not in selection:
                continue
            print(
                f"exact selection: {label} {candidate.variant_id}/{candidate.operator_id} t={coupling:g}",
                flush=True,
            )
            catalog = mixed_catalog(catalogs[label], variant)
            context = build_candidate_context(
                base,
                catalog,
                protocol,
                candidate.variant_id,
                operator,
                pixels_per_axis=pixels,
                softening_kpc=softening,
                audit_rows=tensor_audits,
                stage="exact_selection",
            )
            fitted = fit_context(
                context,
                coupling,
                starts=starts,
                seed=seed + candidate_index * 100 + system_index,
            )
            fits[label] = (context, fitted)
            scores.append(fitted["training"])
            predictions.extend([fitted["training_predictions"], fitted["heldout_predictions"]])
            roots = bool(
                fitted["training"]["all_roots_converged"]
                and fitted["heldout"]["all_roots_converged"]
            )
            complete &= roots
            rows.append(
                {
                    "row_type": "system",
                    "variant_id": candidate.variant_id,
                    "operator_id": candidate.operator_id,
                    "tensor_t": coupling,
                    "system_label": label,
                    "training_exact_RMS_arcsec": fitted["training"]["exact_radial_RMS_arcsec"],
                    "heldout_exact_RMS_arcsec": fitted["heldout"]["exact_radial_RMS_arcsec"],
                    "all_training_and_heldout_roots": roots,
                }
            )
        aggregate = aggregate_system_scores(scores)
        rows.append(
            {
                "row_type": "aggregate",
                "variant_id": candidate.variant_id,
                "operator_id": candidate.operator_id,
                "tensor_t": coupling,
                "system_label": "selection",
                "training_exact_RMS_arcsec": aggregate["equal_system_radial_RMS_arcsec"],
                "heldout_exact_RMS_arcsec": None,
                "all_training_and_heldout_roots": complete,
            }
        )
        fitted_candidates[(candidate.variant_id, candidate.operator_id, coupling)] = fits
    frame = pd.DataFrame(rows)
    aggregate = frame[frame.row_type.eq("aggregate")].copy()
    eligible = aggregate[
        aggregate.all_training_and_heldout_roots.astype(bool)
        & aggregate.tensor_t.astype(float).ne(0.0)
    ]
    if eligible.empty:
        return frame, None, predictions, fitted_candidates
    winner = eligible.sort_values("training_exact_RMS_arcsec").iloc[0].to_dict()
    return frame, winner, predictions, fitted_candidates


def validation(protocol, contexts, catalogs, winner, tensor_audits):
    validation_labels = set(protocol["cluster_split"]["validation_labels"])
    stage = protocol["stages"]["validation"]
    pixels = int(stage["pixels_per_axis"])
    softening = float(protocol["environment_tensor"]["primary_softening_kpc"])
    starts = int(stage["starts_per_fit"])
    seed = int(protocol["optimization"]["random_seed"]) + 30000
    variant_lookup = {row["variant_id"]: row for row in protocol["morphology_variants"]}
    operator_lookup = {row["operator_id"]: row for row in protocol["tensor_operators"]}
    rows = []
    predictions = []
    fit_cache = {}
    candidates = [("zero", "zero", 0.0)]
    if winner is not None:
        candidates.append(
            (winner["variant_id"], winner["operator_id"], float(winner["tensor_t"]))
        )
    for candidate_index, (variant_id, operator_id, coupling) in enumerate(candidates):
        scores = []
        complete = True
        for system_index, base in enumerate(contexts):
            label = base.system["label"]
            if label not in validation_labels:
                continue
            if coupling == 0.0:
                context = base
            else:
                variant = variant_lookup[variant_id]
                operator = operator_lookup[operator_id]
                catalog = mixed_catalog(catalogs[label], variant)
                context = build_candidate_context(
                    base,
                    catalog,
                    protocol,
                    variant_id,
                    operator,
                    pixels_per_axis=pixels,
                    softening_kpc=softening,
                    audit_rows=tensor_audits,
                    stage="validation",
                )
            print(f"validation: {label} {variant_id}/{operator_id} t={coupling:g}", flush=True)
            fitted = fit_context(
                context,
                coupling,
                starts=starts,
                seed=seed + candidate_index * 100 + system_index,
            )
            fit_cache[(variant_id, operator_id, coupling, label)] = (context, fitted)
            scores.append(fitted["heldout"])
            predictions.extend([fitted["training_predictions"], fitted["heldout_predictions"]])
            roots = bool(fitted["heldout"]["all_roots_converged"])
            complete &= roots
            rows.append(
                {
                    "row_type": "system",
                    "variant_id": variant_id,
                    "operator_id": operator_id,
                    "tensor_t": coupling,
                    "system_label": label,
                    "heldout_exact_RMS_arcsec": fitted["heldout"]["exact_radial_RMS_arcsec"],
                    "all_heldout_roots": roots,
                }
            )
        aggregate = aggregate_system_scores(scores)
        rows.append(
            {
                "row_type": "aggregate",
                "variant_id": variant_id,
                "operator_id": operator_id,
                "tensor_t": coupling,
                "system_label": "validation",
                "heldout_exact_RMS_arcsec": aggregate["equal_system_radial_RMS_arcsec"],
                "all_heldout_roots": complete,
            }
        )
    return pd.DataFrame(rows), predictions, fit_cache


def softening_sensitivity(protocol, contexts, catalogs, winner, validation_fits, tensor_audits):
    if winner is None:
        return pd.DataFrame()
    labels = set(protocol["cluster_split"]["validation_labels"])
    variant = next(
        row for row in protocol["morphology_variants"] if row["variant_id"] == winner["variant_id"]
    )
    operator = next(
        row for row in protocol["tensor_operators"] if row["operator_id"] == winner["operator_id"]
    )
    coupling = float(winner["tensor_t"])
    pixels = int(protocol["stages"]["validation"]["pixels_per_axis"])
    rows = []
    primary_softening = float(protocol["environment_tensor"]["primary_softening_kpc"])
    softenings = sorted(
        {
            primary_softening,
            *map(float, protocol["environment_tensor"]["softening_sensitivity_kpc"]),
        }
    )
    for softening in softenings:
        per_system = {}
        for base in contexts:
            label = base.system["label"]
            if label not in labels:
                continue
            fitted_context, fitted = validation_fits[
                (winner["variant_id"], winner["operator_id"], coupling, label)
            ]
            if np.isclose(float(softening), primary_softening):
                context = fitted_context
            else:
                catalog = mixed_catalog(catalogs[label], variant)
                context = build_candidate_context(
                    base,
                    catalog,
                    protocol,
                    winner["variant_id"],
                    operator,
                    pixels_per_axis=pixels,
                    softening_kpc=float(softening),
                    audit_rows=tensor_audits,
                    stage="softening_sensitivity",
                )
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
        rows.append(
            {
                "softening_kpc": float(softening),
                "validation_fixed_fit_local_RMS_arcsec": rms(per_system.values()),
                **{f"{key}_local_RMS_arcsec": value for key, value in per_system.items()},
            }
        )
    return pd.DataFrame(rows)


def make_figure(screen, exact, validation_frame, output):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    best = (
        screen.sort_values("selection_training_local_RMS_arcsec")
        .groupby(["variant_id", "operator_id"], as_index=False, sort=False)
        .first()
        .head(10)
        .sort_values("selection_training_local_RMS_arcsec")
    )
    labels = [f"{v}\n{o}, t={t:g}" for v, o, t in zip(best.variant_id, best.operator_id, best.tensor_t)]
    axes[0].barh(labels[::-1], best.selection_training_local_RMS_arcsec.to_numpy()[::-1])
    axes[0].set(xlabel="selection fixed-source RMS (arcsec)", title="Best coupling per proxy/operator")
    aggregate = exact[exact.row_type.eq("aggregate")].sort_values("training_exact_RMS_arcsec")
    labels = [f"{v}\n{o}, t={t:g}" for v, o, t in zip(aggregate.variant_id, aggregate.operator_id, aggregate.tensor_t)]
    axes[1].barh(labels[::-1], aggregate.training_exact_RMS_arcsec.to_numpy()[::-1])
    axes[1].set(xlabel="selection exact training RMS (arcsec)", title="Exact-root candidate refits")
    valid = validation_frame[validation_frame.row_type.eq("aggregate")]
    labels = [f"{v}\n{o}, t={t:g}" for v, o, t in zip(valid.variant_id, valid.operator_id, valid.tensor_t)]
    colors = ["#777777" if v == "zero" else "#4472c4" for v in valid.variant_id]
    axes[2].bar(labels, valid.heldout_exact_RMS_arcsec, color=colors)
    axes[2].set(ylabel="held-out validation RMS (arcsec)", title="Transfer to different clusters")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    config_path = ROOT / "configs/p0557_baryon_proxy_tidal_protocol.json"
    protocol = json.loads(config_path.read_text(encoding="utf-8"))
    if not protocol["status"].startswith("frozen_before_any_"):
        raise RuntimeError("P0557 protocol is not prospectively frozen")
    adequacy = json.loads(
        (ROOT / protocol["inputs"]["all_baryon_input_audit"]).read_text(encoding="utf-8")
    )
    if not adequacy["input_adequacy_pass"]:
        raise RuntimeError("registered all-baryon inputs did not pass their frozen audit")
    member_protocol = json.loads(
        (ROOT / protocol["inputs"]["member_tidal_protocol"]).read_text(encoding="utf-8")
    )
    member_protocol["optimization"]["maximum_function_evaluations"] = int(
        protocol["optimization"]["maximum_function_evaluations"]
    )
    member_protocol["environment_tensor"]["subtract_circular_mean"] = True
    contexts, _, input_hashes = build_contexts(
        member_protocol,
        softening_kpc=float(protocol["environment_tensor"]["primary_softening_kpc"]),
    )
    output = ROOT / protocol["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    catalogs, component_audits = prepare_component_catalogs(protocol, contexts)
    pd.DataFrame(component_audits).to_csv(
        output / protocol["outputs"]["component_audits"], index=False
    )
    baselines = baseline_fits(protocol, contexts)
    tensor_audits = []
    screen, shortlist = screen_candidates(
        protocol, contexts, catalogs, baselines, tensor_audits
    )
    screen.to_csv(output / protocol["outputs"]["screen_scores"], index=False)
    exact, winner, exact_predictions, _ = exact_selection(
        protocol, contexts, catalogs, shortlist, baselines, tensor_audits
    )
    exact.to_csv(output / protocol["outputs"]["exact_selection_scores"], index=False)
    validation_frame, validation_predictions, validation_fits = validation(
        protocol, contexts, catalogs, winner, tensor_audits
    )
    validation_frame.to_csv(
        output / protocol["outputs"]["validation_scores"], index=False
    )
    sensitivity = softening_sensitivity(
        protocol, contexts, catalogs, winner, validation_fits, tensor_audits
    )
    sensitivity.to_csv(output / "softening_sensitivity.csv", index=False)
    pd.DataFrame(tensor_audits).to_csv(
        output / protocol["outputs"]["tensor_audits"], index=False
    )
    predictions = exact_predictions + validation_predictions
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            output / protocol["outputs"]["predictions"], index=False
        )
    metric = json.loads(
        (ROOT / protocol["inputs"]["metric_slip_report"]).read_text(encoding="utf-8")
    )
    member_report = json.loads(
        (ROOT / "results/member_tidal_metric/report.json").read_text(encoding="utf-8")
    )
    valid_aggregate = validation_frame[validation_frame.row_type.eq("aggregate")].set_index(
        "variant_id"
    )
    exact_aggregate = exact[exact.row_type.eq("aggregate")]
    selection_zero_rms = float(
        exact_aggregate.loc[
            exact_aggregate.variant_id.eq("zero"), "training_exact_RMS_arcsec"
        ].iloc[0]
    )
    selection_selected_rms = (
        selection_zero_rms
        if winner is None
        else float(winner["training_exact_RMS_arcsec"])
    )
    zero_rms = float(valid_aggregate.loc["zero", "heldout_exact_RMS_arcsec"])
    if winner is None:
        selected_rms = zero_rms
        selected_roots = True
        improvement = 0.0
    else:
        selected_rms = float(valid_aggregate.loc[winner["variant_id"], "heldout_exact_RMS_arcsec"])
        selected_roots = bool(valid_aggregate.loc[winner["variant_id"], "all_heldout_roots"])
        improvement = 1.0 - selected_rms / zero_rms
    halo = float(
        metric["comparators"]["compact_halo_validation"][
            "equal_system_radial_RMS_arcsec"
        ]
    )
    audits = pd.DataFrame(tensor_audits)
    gates = protocol["advance_gates"]
    gate_audit = {
        "validation_all_roots_converged": selected_roots,
        "validation_RMS_improvement_over_zero_fraction": improvement,
        "validation_RMS_improvement_pass": improvement
        >= float(gates["validation_RMS_improvement_over_zero_fraction_min"]),
        "validation_to_compact_halo_RMS_ratio": selected_rms / halo,
        "validation_to_compact_halo_pass": selected_rms / halo
        <= float(gates["validation_to_compact_halo_RMS_ratio_max"]),
        "maximum_solver_edge_Q_eigenvalue": float(
            audits.maximum_solver_edge_Q_eigenvalue.max()
        ),
        "edge_gate_pass": float(audits.maximum_solver_edge_Q_eigenvalue.max())
        <= float(gates["maximum_solver_edge_Q_eigenvalue"]),
        "maximum_normalized_curl_RMS": float(audits.normalized_curl_RMS.max()),
        "curl_gate_pass": float(audits.normalized_curl_RMS.max())
        <= float(gates["maximum_normalized_curl_RMS"]),
    }
    all_empirical = bool(
        selected_roots
        and gate_audit["validation_RMS_improvement_pass"]
        and gate_audit["validation_to_compact_halo_pass"]
        and gate_audit["edge_gate_pass"]
        and gate_audit["curl_gate_pass"]
    )
    report = {
        "report_version": "P0557-BARYON-PROXY-TIDAL-RESULTS-0.1.0",
        "status": "complete",
        "protocol": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256(config_path)},
        "input_hashes": input_hashes,
        "coverage": {
            "morphology_variants": len(protocol["morphology_variants"]),
            "tensor_operators": len(protocol["tensor_operators"]),
            "nonzero_couplings": len(protocol["tensor_coupling"]["grid"]) - 1,
            "fixed_source_screen_combinations": int(len(screen)),
            "exact_selection_candidates": int(len(shortlist)),
            "selection_clusters": len(protocol["cluster_split"]["selection_labels"]),
            "validation_clusters": len(protocol["cluster_split"]["validation_labels"]),
        },
        "screen_best": screen.iloc[0].to_dict(),
        "exact_selection": {
            "shortlist": shortlist.to_dict("records"),
            "winner": winner,
            "zero_tensor_training_RMS_arcsec": selection_zero_rms,
            "selected_training_RMS_arcsec": selection_selected_rms,
            "selected_improvement_fraction_vs_zero": 1.0
            - selection_selected_rms / selection_zero_rms,
        },
        "validation": {
            "zero_tensor_RMS_arcsec": zero_rms,
            "selected_proxy_tensor_RMS_arcsec": selected_rms,
            "improvement_fraction": improvement,
            "selected_all_roots_converged": selected_roots,
            "scores": validation_frame.to_dict("records"),
        },
        "comparators": {
            "compact_halo_validation_RMS_arcsec": halo,
            "selected_to_compact_halo_RMS_ratio": selected_rms / halo,
            "fixed_RAR_galaxy_outer_RMSE_km_s": float(
                member_report["comparators"]["locked_fixed_RAR_galaxy_outer_RMSE_km_s"]
            ),
        },
        "galaxy_and_solar_control": {
            "formula_change": 0.0,
            "reason": "the tested Q_proxy is explicitly an external cluster-environment tensor",
            "fixed_RAR_galaxy_outer_RMSE_km_s": float(
                member_report["comparators"]["locked_fixed_RAR_galaxy_outer_RMSE_km_s"]
            ),
            "maximum_abs_eta_minus_one_limb_to_Saturn": float(
                metric["Solar_System"]["maximum_abs_eta_minus_one_limb_to_Saturn"]
            ),
            "interpretation": "preservation by scope, not an independent improvement; a final theory must define the external-environment gate covariantly",
        },
        "softening_sensitivity": sensitivity.to_dict("records"),
        "gate_audit": gate_audit,
        "verdict": {
            "all_empirical_gates_pass": all_empirical,
            "formula_promoted": False,
            "gas_mass_map_still_required": True,
        },
        "claim_limits": protocol["claim_limits"],
    }
    (output / protocol["outputs"]["report"]).write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(screen, exact, validation_frame, output / protocol["outputs"]["figure"])
    winner_text = "none" if winner is None else (
        f"{winner['variant_id']}/{winner['operator_id']} at t={winner['tensor_t']:+.3f}"
    )
    summary = f"""# P0557 baryon-proxy tidal experiment

The exact selection winner is **{winner_text}**.  On the two different
validation clusters its held-out exact image-position RMS is
**{selected_rms:.3f} arcsec**, versus **{zero_rms:.3f} arcsec** for the locked
scalar field ({100.0 * improvement:+.2f}%).  The compact-halo comparator is
**{halo:.3f} arcsec**.  All empirical advancement gates pass: **{all_empirical}**.

X-ray brightness remains a shape proxy rather than a gas-mass map, so no
formula is promoted regardless of this screen result.
"""
    (output / protocol["outputs"]["summary"]).write_text(summary, encoding="utf-8")
    print(json.dumps(json_safe(report["validation"]), indent=2), flush=True)
    print(json.dumps(json_safe(report["verdict"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
