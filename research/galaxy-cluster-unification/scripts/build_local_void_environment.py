from pathlib import Path

from voidscreen.void_geometry import build_local_void_wall_table, write_local_void_products

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    table = build_local_void_wall_table(
        ROOT / "data" / "raw" / "sparc",
        ROOT / "data" / "raw" / "local_voids",
    )
    write_local_void_products(
        table,
        ROOT / "data" / "derived" / "void_wall_scores_local.csv",
        ROOT / "data" / "derived" / "void_wall_scores_local_report.json",
        catalog_dir=ROOT / "data" / "raw" / "local_voids",
    )
    print((ROOT / "data" / "derived" / "void_wall_scores_local_report.json").read_text())


if __name__ == "__main__":
    main()
