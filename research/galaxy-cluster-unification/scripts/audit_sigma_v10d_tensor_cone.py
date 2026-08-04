from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v10d_tensor_cone import audit_v10d_tensor_cone


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the exact Sigma v10D axisymmetric tensor cone."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v10d_tensor_cone.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v10d_tensor_cone",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fixed = config["fixed_values"]
    audit = audit_v10d_tensor_cone(
        carrier_speed_squared=float(fixed["carrier_speed_squared"]),
        speed_tolerance=float(fixed["relative_metric_speed_tolerance"]),
        demonstration_anisotropy=float(fixed["demonstration_anisotropy"]),
        scan_maximum_anisotropy=float(fixed["scan_maximum_anisotropy"]),
        scan_samples=int(fixed["scan_samples"]),
    )
    report = {
        "status": "completed Sigma v10D exact tensor-cone falsification",
        "candidate": config["candidate"],
        **audit,
        "decision": "retire_exact_v10d_and_reset_aether_tidal_carrier_family",
        "reason": "On an axisymmetric carrier background the spin-2 sector is exact and has c_TT^2=1+c_P^2(p_parallel-p_perp)^2. Every nonzero anisotropy is therefore outside the physical metric cone; neither the exponential aether completion nor lapse, scalar, or vector constraints enter this symmetry sector.",
        "no_patch_rule": config["mechanism_reset_rule"],
        "scope": "This necessary local characteristic failure retires the exact action. It does not infer an astronomical carrier amplitude or score observational data.",
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
