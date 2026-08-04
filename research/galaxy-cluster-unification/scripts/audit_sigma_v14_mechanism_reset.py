from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voidscreen.sigma_v14_mechanism_reset import audit_reset_protocol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Sigma v14 mechanism reset and frozen postulates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sigma_v14_mechanism_reset.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "sigma_v14_mechanism_reset",
    )
    args = parser.parse_args()
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    report = audit_reset_protocol(protocol, project_root=ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
