#!/usr/bin/env python3
"""Post-result transfer of nonzero member-tidal settings; never qualifying."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_member_tidal_metric import build_contexts, fit_context, json_safe
from run_unbounded_running_multicluster_raw import aggregate_system_scores


def main():
    protocol = json.loads(
        (ROOT / "configs/member_tidal_metric_protocol.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (ROOT / "results/member_tidal_metric/report.json").read_text(encoding="utf-8")
    )
    contexts, _, _ = build_contexts(
        protocol,
        softening_kpc=float(protocol["environment_tensor"]["primary_softening_kpc"]),
    )
    labels = set(protocol["cluster_split"]["validation_labels"])
    validation = [context for context in contexts if context.system["label"] in labels]
    # -0.6 is the strongest negative primary setting with complete selection
    # roots; +0.9 is the positive primary edge; +1.2 is the best complete
    # extended setting.  Their choice was made after the primary t=0 result,
    # so these scores are mechanistic diagnostics only.
    couplings = [-0.6, 0.9, 1.2]
    rows = []
    aggregates = {}
    for coupling_index, coupling in enumerate(couplings):
        scores = []
        for system_index, context in enumerate(validation):
            print(f"posthoc validation system={context.system['label']} t={coupling:g}", flush=True)
            fitted = fit_context(
                context,
                coupling,
                starts=int(protocol["optimization"]["starts_selected_validation"]),
                seed=int(protocol["optimization"]["random_seed"])
                + 160000
                + coupling_index * 100
                + system_index,
            )
            scores.append(fitted["heldout"])
            rows.append(
                {
                    "tensor_t": coupling,
                    "system": context.system["system"],
                    "system_label": context.system["label"],
                    **{f"heldout_{key}": value for key, value in fitted["heldout"].items()},
                }
            )
        aggregate = aggregate_system_scores(scores)
        aggregates[f"{coupling:g}"] = aggregate
        rows.append(
            {
                "tensor_t": coupling,
                "system": "equal_system",
                "system_label": "validation",
                "heldout_exact_radial_RMS_arcsec": aggregate["equal_system_radial_RMS_arcsec"],
                "heldout_all_roots_converged": aggregate["all_roots_converged"],
            }
        )
    output = ROOT / "results/member_tidal_metric"
    pd.DataFrame(rows).to_csv(output / "nonzero_transfer_diagnostic.csv", index=False)
    result = {
        "report_version": "MEMBER-TIDAL-NONZERO-TRANSFER-DIAGNOSTIC-0.1.0",
        "status": "complete_post_result_nonqualifying",
        "disclosure": "The three nonzero couplings were chosen after the frozen primary selected t=0. These results diagnose behavior and cannot replace or rescue the primary result.",
        "couplings": couplings,
        "validation": aggregates,
        "comparators": {
            "frozen_selected_t": report["selection"]["selected_t"],
            "zero_tensor_RMS_arcsec": report["validation"]["zero_tensor"]["equal_system_radial_RMS_arcsec"],
            "compact_halo_RMS_arcsec": report["comparators"]["compact_halo_validation_RMS_arcsec"],
        },
    }
    (output / "nonzero_transfer_diagnostic.json").write_text(
        json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
