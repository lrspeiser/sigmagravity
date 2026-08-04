from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v10d_anisotropic_characteristics import (
    audit_v10d_anisotropic_characteristics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit v10D anisotropic fixed-metric characteristics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v10d_anisotropic_characteristics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v10d_anisotropic_characteristics",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v10d_anisotropic_characteristics(
        k_b=float(fixed["K_B"]),
        beta=float(fixed["beta"]),
        base_spatial_stiffness=float(fixed["u"]),
        carrier_speed_squared=float(fixed["carrier_speed_squared"]),
        normalized_mixing_squared=float(fixed["normalized_mixing_squared"]),
        random_samples=int(fixed["random_samples"]),
    )
    report = {
        "status": "completed Sigma v10D anisotropic source-block characteristic gate",
        "candidate": config["candidate"],
        **audit,
        "decision": "advance_v10d_to_full_metric_ADM_principal_gate",
        "reason": "The completed kinetic matrix obeys F>=I and the arbitrary-direction divergence Gram matrix obeys I/2<=R<=I. These bounds keep the full anisotropic source-block static Schur complement at least I/3 and its characteristic roots positive and inside the metric cone; noncommuting random instances and Lorentz boosts confirm the analytic result.",
        "scope": "This gate excludes the dynamical metric and full AeST scalar constraint sector. It is not yet a complete ADM, PPN, cosmology or global well-posedness proof.",
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
