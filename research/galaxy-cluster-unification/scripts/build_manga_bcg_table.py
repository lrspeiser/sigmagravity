from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _numbers(field: str) -> list[float]:
    return [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", field)]


def parse_table(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    start = text.index("\\startdata", text.index("Properties and Results of 50 MaNGA BCGs"))
    end = text.index("\\enddata", start)
    rows = []
    for raw_line in text[start:end].splitlines()[1:]:
        if "&" not in raw_line:
            continue
        fields = raw_line.split("&")
        if len(fields) != 11:
            raise ValueError(f"expected 11 table fields, found {len(fields)}: {raw_line}")
        plateifu = fields[0].replace("$^{*}$", "").strip()
        scalars = [_numbers(field)[0] for field in fields[1:7]]
        paired = [_numbers(field)[:2] for field in fields[7:11]]
        if any(len(values) != 2 for values in paired):
            raise ValueError(f"missing value/error pair: {raw_line}")
        rows.append(
            {
                "plateifu": plateifu,
                "redshift": scalars[0],
                "sersic_n": scalars[1],
                "effective_radius_kpc": scalars[2],
                "last_radius_re": scalars[3],
                "dispersion_slope": scalars[4],
                "dispersion_intercept": scalars[5],
                "log_mbar_msun": paired[0][0],
                "err_log_mbar_msun": paired[0][1],
                "mean_sigma_los_km_s": paired[1][0],
                "err_mean_sigma_los_km_s": paired[1][1],
                "log_gbar": paired[2][0],
                "err_log_gbar": paired[2][1],
                "log_gobs": paired[3][0],
                "err_log_gobs": paired[3][1],
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 50 or frame["plateifu"].nunique() != 50:
        raise ValueError("expected 50 unique MaNGA BCG rows")
    frame["radius_kpc"] = frame["effective_radius_kpc"] * frame["last_radius_re"]
    return frame


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the Tian 2024 MaNGA BCG table.")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data" / "raw" / "manga_bcg_tian2024" / "RAR_BCG.tex",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "derived" / "manga_bcg_tian2024.csv",
    )
    args = parser.parse_args()
    frame = parse_table(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    report = {
        "source": str(args.source.relative_to(ROOT)),
        "source_sha256": _sha256(args.source),
        "output": str(args.output.relative_to(ROOT)),
        "rows": len(frame),
        "unique_plateifu": int(frame["plateifu"].nunique()),
        "radius_definition": "effective_radius_kpc * last_radius_re",
    }
    (args.output.with_suffix(".report.json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
