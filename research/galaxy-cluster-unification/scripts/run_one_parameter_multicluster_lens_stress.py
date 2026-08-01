#!/usr/bin/env python3
"""Apply the already locked one-parameter lens law to predeclared stress clusters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_one_parameter_multicluster_lens import (  # noqa: E402
    aggregate_system_scores,
    fit_definitions_for_system,
    json_safe,
    load_json,
    model_key,
)
from run_unbounded_running_multicluster_raw import (  # noqa: E402
    load_system_images,
    predictive_split,
)


def main() -> None:
    config_path = ROOT / "configs/one_parameter_multicluster_lens_protocol.json"
    report_path = ROOT / "results/one_parameter_multicluster_lens/report.json"
    protocol = load_json(config_path)
    report = load_json(report_path)
    base_protocol = load_json(ROOT / protocol["inputs"]["base_protocol"])
    base_report = load_json(ROOT / protocol["inputs"]["base_report"])
    catalog = pd.read_csv(ROOT / protocol["inputs"]["image_catalog"])
    tian = pd.read_csv(
        ROOT / protocol["inputs"]["baryonic_profile"],
        sep=r"\s+",
        names=[
            "system",
            "radius_kpc",
            "log_gbar",
            "log_gobs",
            "err_log_gbar",
            "err_log_gobs",
        ],
    )
    systems = {system["label"]: system for system in base_protocol["systems"]}
    labels = protocol["selection_design"]["stress_only"]
    selected = report["selection"]
    definition = {
        "model": model_key(selected["selected_family"], 0, "stress"),
        "family": selected["selected_family"],
        "parameter": float(selected["selected_parameter"]),
        "grid_index": 0,
        "stage": "stress",
    }

    scores_by_label: dict[str, dict] = {}
    predictions = []
    geometry = []
    profiles = []
    for system_index, label in enumerate(labels):
        scores, local_predictions, local_geometry, local_profiles = (
            fit_definitions_for_system(
                systems[label],
                system_index,
                [definition],
                base_protocol,
                protocol,
                catalog,
                tian,
                starts=int(protocol["optimization"]["starts_final_validation"]),
                seed_offset=300_000,
            )
        )
        scores_by_label[label] = scores
        predictions.extend(local_predictions)
        geometry.extend(local_geometry)
        profiles.append(local_profiles)

    model = definition["model"]
    stress = {}
    heldout_scores = []
    training_scores = []
    for label in labels:
        system = systems[label]
        selected_score = scores_by_label[label][model]
        training_scores.append(selected_score["training"])
        heldout = selected_score["heldout"]
        if heldout.get("exact_radial_RMS_arcsec") is not None:
            heldout_scores.append(heldout)
        base_scores = base_report["system_scores"][system["system"]]
        _, heldout_images = predictive_split(load_system_images(catalog, system))
        stress[label] = {
            **selected_score,
            "heldout_images_available": int(len(heldout_images)),
            "comparators": {
                name: base_scores[name]
                for name in (
                    "baryons_GR",
                    "fixed_simple_MOND",
                    "GR_plus_cluster_halo",
                )
            },
        }

    report["stress_tests"] = {
        "status": "post-lock application to the two clusters predeclared as stress-only",
        "selected_family": definition["family"],
        "selected_parameter": definition["parameter"],
        "training_aggregate": aggregate_system_scores(training_scores),
        "heldout_aggregate": (
            aggregate_system_scores(heldout_scores) if heldout_scores else None
        ),
        "per_system": stress,
        "interpretation": (
            "RXJ1347 has no eligible within-family heldout images; its training score "
            "is descriptive only. RXJ2129 is a stress replay, not fresh validation."
        ),
    }
    report_path.write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    output_dir = report_path.parent
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / "stress_predictions.csv", index=False
    )
    pd.DataFrame(geometry).to_csv(output_dir / "stress_geometry.csv", index=False)
    pd.concat(profiles, ignore_index=True).to_csv(
        output_dir / "stress_radial_profiles.csv", index=False
    )

    validation_rms = report["validation"]["selected_law"][
        "equal_system_radial_RMS_arcsec"
    ]
    halo_rms = report["comparators"]["GR_plus_compact_cluster_halo"][
        "equal_system_radial_RMS_arcsec"
    ]
    rxj = stress["RXJ2129"]
    rxj_rms = rxj["heldout"]["exact_radial_RMS_arcsec"]
    rxj_halo_rms = rxj["comparators"]["GR_plus_cluster_halo"]["heldout"][
        "exact_radial_RMS_arcsec"
    ]
    summary = [
        "# One-parameter multi-cluster lens result",
        "",
        f"Selected family: **{definition['family']}**",
        f"Selected universal parameter: **{definition['parameter']:.8g}**",
        f"Development held-out RMS: **{report['selection']['development_heldout']['equal_system_radial_RMS_arcsec']:.3f} arcsec**",
        f"Validation held-out RMS: **{validation_rms:.3f} arcsec**",
        f"Compact-halo validation RMS: **{halo_rms:.3f} arcsec**",
        f"Survives every frozen gate: **{report['gate_audit']['all_advance_gates_pass']}**",
        "",
        "The law passes transfer, root, halo-proximity, and interior-parameter gates but",
        "fails the predeclared absolute target of 2 arcsec. It improves both validation",
        "clusters over baryons, yet the compact halo is much better on MACS1931 and has",
        "the better pooled chi-square. On the post-lock RXJ2129 stress replay the law",
        f"scores {rxj_rms:.3f} arcsec versus {rxj_halo_rms:.3f} for the compact halo and loses one training",
        "root. See `docs/ONE_PARAMETER_MULTICLUSTER_LENS_RESULTS.md`.",
    ]
    (output_dir / "SUMMARY.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )

    for label in labels:
        item = stress[label]
        heldout = item["heldout"]
        print(
            f"{label}: training RMS="
            f"{item['training'].get('exact_radial_RMS_arcsec')}; "
            f"heldout RMS={heldout.get('exact_radial_RMS_arcsec')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
