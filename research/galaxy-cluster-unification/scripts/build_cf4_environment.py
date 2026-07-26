from __future__ import annotations

import argparse
from pathlib import Path

from voidscreen.environment import build_cf4_environment_table, write_environment_products

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build independent Cosmicflows-4 density scores for all SPARC galaxies."
    )
    parser.add_argument("--sparc-dir", type=Path, default=ROOT / "data" / "raw" / "sparc")
    parser.add_argument("--cf4-dir", type=Path, default=ROOT / "data" / "raw" / "cosmicflows4")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "data" / "derived" / "void_scores_cf4.csv",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "data" / "derived" / "cf4_environment_report.json",
    )
    args = parser.parse_args()

    table = build_cf4_environment_table(args.sparc_dir, args.cf4_dir)
    write_environment_products(
        table,
        args.output_csv,
        args.report_json,
        cf4_dir=args.cf4_dir,
    )
    print(f"Wrote {len(table)} independent environment rows to {args.output_csv}")
    print(f"Wrote provenance and validation report to {args.report_json}")


if __name__ == "__main__":
    main()

