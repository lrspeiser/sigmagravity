from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v7_falsification import audit_positive_spin2_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize the three Sigma v7 failures.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v7_positive_spin2_falsification.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v7_positive_spin2_falsification",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = config["formulations"]
    reports = [
        json.loads((ROOT / row["report"]).read_text(encoding="utf-8")) for row in rows
    ]
    audit = audit_positive_spin2_sequence(
        reports,
        formulation_names=[row["name"] for row in rows],
        failure_gate_names=[row["failure_gate"] for row in rows],
        minimum_distinct_failures=int(config["minimum_distinct_failures"]),
    )
    reset_required = bool(audit["gates"]["mechanism_reset_required"])
    report = {
        "status": "completed Sigma v7 positive-spin2 mechanism falsification",
        "mechanism": config["mechanism"],
        "common_required_outcome": config["common_required_outcome"],
        "formulations": audit["formulations"],
        "distinct_candidate_count": int(audit["distinct_candidate_count"]),
        "failed_gate_count": int(audit["failed_gate_count"]),
        "gates": audit["gates"],
        "mechanism_reset_required": reset_required,
        "decision": "retire_positive_spin2_carrier_as_current_sigma_route"
        if reset_required
        else "insufficient_evidence_for_mechanism_reset",
        "reason": "Three materially distinct positive-spin2 implementations fail the same combined carrier objective for independent reasons: Solar-safe amplitude, spherical screening discrimination/amplitude, and closure of a nonzero physical lensing projection. Adding another response term to v7C would evade rather than pass the frozen gate.",
        "scope": "This falsifies the present positive-spin2 carrier route under the Sigma universality and complexity rules. It does not falsify dRGT or Hassan-Rosen gravity as mathematical theories and does not prohibit a future complete coupled-metric proposal from entering as a new mechanism.",
        "next_mechanism_constraints": [
            "the extra field must contribute to the Weyl potential at the same derived order as it changes massive dynamics",
            "the physical metric must be frozen before any source-map solve",
            "the activation cannot depend only on spherical enclosed density",
            "the Solar limit must arise from the same equations without suppressing the required cluster response",
            "no continuation may add a lens-only multiplier or object label"
        ],
        "observational_data_accessed_by_synthesis": False,
        "raw_holdout_opened": False,
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
