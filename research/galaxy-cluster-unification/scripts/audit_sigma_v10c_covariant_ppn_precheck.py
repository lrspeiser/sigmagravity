from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v10c_covariant_ppn_precheck import (
    audit_v10c_covariant_ppn_precheck,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v10C covariant aether map and PPN applicability."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v10c_covariant_ppn_precheck.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v10c_covariant_ppn_precheck",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v10c_covariant_ppn_precheck(
        k_b=float(fixed["K_B"]),
        u=float(fixed["u"]),
    )
    report = {
        "status": "completed Sigma v10C covariant/PPN applicability precheck",
        "candidate": config["candidate"],
        "fixed_values": fixed,
        **audit,
        "decision": "retain_v10c_for_full_AeST_plus_P_PPN_derivation",
        "reason": "The counterterm has an exact covariant coefficient map and does not change the pure-aether alpha1 proxy relative to the Maxwell base. The pure-aether alpha2 formula is singular at c123=0 and omits both fields that make v10C different, so it cannot honestly pass or retire the theory.",
        "scope": "This is an applicability result, not a Solar-System pass. The full moving-source AeST-plus-P field equations, alpha1, alpha2, gamma, beta, Mercury and compact-source limits remain mandatory.",
        "observational_data_accessed": False,
        "raw_holdout_opened": False,
        "data_policy": config["data_policy"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
