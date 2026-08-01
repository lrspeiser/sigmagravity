#!/usr/bin/env python3
"""Post-failure held-out oracle over the frozen spatial-vector grid.

This is intentionally not a predictive score.  It asks whether the training
selection missed a useful universal grid point while keeping the already fitted
spherical geometry fixed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_rxj2129_raw_theory_lensing import score
from run_unbounded_running_multicluster_raw import aggregate_system_scores
from run_unbounded_running_spatial_vector import (
    build_contexts,
    json_safe,
    make_lens,
    variant_name,
)


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs/unbounded_running_spatial_vector_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    contexts, _, _ = build_contexts(protocol)
    grid = protocol["spatial_vector_grid"]
    rows = []
    aggregates = {}
    first_softening = float(grid["softening_arcsec"][0])
    for model in protocol["models"]:
        for dressing in grid["dressings"]:
            name = variant_name(model, dressing)
            aggregates[name] = []
            for fraction in grid["mass_fractions"]:
                for softening in grid["softening_arcsec"]:
                    fraction = float(fraction)
                    softening = float(softening)
                    if fraction == 0.0 and softening != first_softening:
                        continue
                    system_scores = []
                    print(
                        f"oracle {name} f={fraction:g} s={softening:g}", flush=True
                    )
                    for context in contexts:
                        lens = make_lens(
                            context, model, dressing, fraction, softening
                        )
                        parameters = context.baseline_parameters[model]
                        _, sources = lens.profiled_residuals(
                            name, parameters, context.training
                        )
                        prediction = lens.exact_predictions(
                            name,
                            parameters,
                            sources,
                            context.heldout,
                            stage="postfailure_heldout_oracle_fixed_geometry",
                        )
                        system_score = score(prediction, lens.sigma)
                        system_scores.append(system_score)
                        rows.append(
                            {
                                "row_type": "system",
                                "variant": name,
                                "base_model": model,
                                "dressing": dressing,
                                "mass_fraction": fraction,
                                "softening_arcsec": softening,
                                "system": context.system["system"],
                                **system_score,
                            }
                        )
                    aggregate = aggregate_system_scores(system_scores)
                    aggregates[name].append(
                        {
                            "mass_fraction": fraction,
                            "softening_arcsec": softening,
                            **aggregate,
                        }
                    )
                    rows.append(
                        {
                            "row_type": "aggregate",
                            "variant": name,
                            "base_model": model,
                            "dressing": dressing,
                            "mass_fraction": fraction,
                            "softening_arcsec": softening,
                            "system": "equal_system",
                            **aggregate,
                        }
                    )

    baseline = json.loads(
        (ROOT / protocol["inputs"]["baseline_report"]).read_text(encoding="utf-8")
    )
    table = pd.DataFrame(rows)
    best_universal = {}
    hidden_per_system = {}
    for name, settings in aggregates.items():
        finite = [
            item
            for item in settings
            if item["all_roots_converged"]
            and item["equal_system_radial_RMS_arcsec"] is not None
            and np.isfinite(item["equal_system_radial_RMS_arcsec"])
        ]
        best_universal[name] = min(
            finite, key=lambda item: item["equal_system_radial_RMS_arcsec"]
        )
        hidden_per_system[name] = {}
        block = table[(table.row_type == "system") & (table.variant == name)]
        for context in contexts:
            system_block = block[
                block.system == context.system["system"]
            ].replace([np.inf, -np.inf], np.nan).dropna(
                subset=["exact_radial_RMS_arcsec"]
            )
            best = system_block.nsmallest(1, "exact_radial_RMS_arcsec").iloc[0]
            hidden_per_system[name][context.system["system"]] = {
                "mass_fraction": float(best.mass_fraction),
                "softening_arcsec": float(best.softening_arcsec),
                "exact_radial_RMS_arcsec": float(best.exact_radial_RMS_arcsec),
            }

    report = {
        "status": "completed explicitly non-predictive post-failure oracle",
        "claim_boundary": [
            "Held-out images are used to choose the reported oracle settings; these are not predictions and cannot create a survivor.",
            "Cluster geometry is fixed at the prior spherical training fit, so this diagnoses missed grid selection rather than providing a full alternative refit.",
            "The hidden per-system oracle is additionally forbidden by the universal-setting requirement.",
        ],
        "best_universal_oracle": best_universal,
        "hidden_per_system_oracle": hidden_per_system,
        "spherical_baseline": {
            model: baseline["primary_aggregate"][model]
            for model in protocol["models"]
        },
        "verdict": {
            "any_universal_oracle_beats_parent": any(
                block["equal_system_radial_RMS_arcsec"]
                < baseline["primary_aggregate"][name.split("__members_")[0]][
                    "equal_system_radial_RMS_arcsec"
                ]
                for name, block in best_universal.items()
            ),
            "predictive_survivor_created": False,
        },
    }
    output = ROOT / "results/unbounded_running_spatial_vector_oracle"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "oracle_grid.csv", index=False)
    (output / "report.json").write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Spatial-vector post-failure oracle",
        "",
        "This diagnostic uses held-out images for selection and cannot be interpreted as prediction.",
        "",
        "| variant | oracle f | softening (arcsec) | held-out RMS (arcsec) | spherical parent (arcsec) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, best in sorted(
        best_universal.items(),
        key=lambda pair: pair[1]["equal_system_radial_RMS_arcsec"],
    ):
        parent = name.split("__members_")[0]
        lines.append(
            f"| {name} | {best['mass_fraction']:.3f} | {best['softening_arcsec']:.2f} | "
            f"{best['equal_system_radial_RMS_arcsec']:.3f} | "
            f"{baseline['primary_aggregate'][parent]['equal_system_radial_RMS_arcsec']:.3f} |"
        )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output / "SUMMARY.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
