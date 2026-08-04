from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v10d_adm_rank import audit_v10d_adm_rank


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v10D aether-rest ADM Legendre rank."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v10d_adm_rank.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v10d_adm_rank",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v10d_adm_rank(
        k_b=float(fixed["K_B"]),
        beta=float(fixed["beta"]),
        scalar_clock_coefficient=float(fixed["scalar_clock_coefficient_nu"]),
        random_samples=int(fixed["random_samples"]),
    )
    report = {
        "status": "completed Sigma v10D aether-rest ADM rank subgate",
        "candidate": config["candidate"],
        **audit,
        "decision": "advance_v10d_to_full_metric_characteristic_gate",
        "reason": "The carrier/metric velocities are related to the Einstein-Hilbert plus six-square basis by a triangular unit-determinant transformation for every carrier background. The combined rest-frame Legendre inertia is constant, the completed aether and scalar clock blocks are positive, and the generic constraint count is AeST's six physical modes plus six carrier modes.",
        "scope": "This proves generic local rank in the aether-rest foliation. It does not yet prove the arbitrary-foliation constraint algebra or the full anisotropic metric characteristic cones.",
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
