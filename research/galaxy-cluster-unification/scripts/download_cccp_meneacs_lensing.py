#!/usr/bin/env python3
"""Download and extract the Herbonnet et al. weak-lensing mass table."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://export.arxiv.org/e-print/1912.04414"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numbers(value: str) -> list[float]:
    return [float(item) for item in re.findall(r"[-+]?\d+(?:\.\d+)?", value)]


def parse_mass_table(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split("&")]
        if len(fields) != 8 or not fields[0].isdigit():
            continue
        if "\\pm" not in fields[3] or "\\pm" not in fields[4]:
            continue
        if "_{" not in fields[5] or "_{" not in fields[6]:
            continue
        index = int(fields[0])
        nfw200 = numbers(fields[3])
        nfw500 = numbers(fields[4])
        radius500 = numbers(fields[5])
        aperture500 = numbers(fields[6])
        if not (len(nfw200) == len(nfw500) == 2):
            raise ValueError(f"unexpected symmetric mass field: {line}")
        if len(radius500) != 3 or len(aperture500) != 3:
            raise ValueError(f"unexpected asymmetric aperture field: {line}")
        rows.append(
            {
                "index": index,
                "cluster": fields[1],
                "redshift": float(fields[2]),
                "m_nfw200_1e14_msun": nfw200[0],
                "err_m_nfw200_1e14_msun": abs(nfw200[1]),
                "m_nfw500_1e14_msun": nfw500[0],
                "err_m_nfw500_1e14_msun": abs(nfw500[1]),
                "r_ap500_mpc": radius500[0],
                "err_low_r_ap500_mpc": abs(radius500[1]),
                "err_high_r_ap500_mpc": abs(radius500[2]),
                "m_ap500_1e14_msun": aperture500[0],
                "err_low_m_ap500_1e14_msun": abs(aperture500[1]),
                "err_high_m_ap500_1e14_msun": abs(aperture500[2]),
                "xray_state": fields[7].replace("\\", "").strip(),
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates("index").sort_values("index")
    if len(frame) != 100 or frame["index"].tolist() != list(range(1, 101)):
        raise ValueError(f"expected the published 100-row mass table, found {len(frame)}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "raw" / "cccp_meneacs_herbonnet2020",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with urlopen(SOURCE_URL, timeout=180) as response:
        content = response.read()
    archive_path = args.output / "arxiv_1912.04414_source.tar"
    archive_path.write_bytes(content)
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as archive:
        member = archive.getmember("masses.tex")
        handle = archive.extractfile(member)
        if handle is None:
            raise RuntimeError("masses.tex was not readable in the arXiv source")
        tex_content = handle.read()
    tex_path = args.output / "masses.tex"
    tex_path.write_bytes(tex_content)
    table = parse_mass_table(tex_content.decode("utf-8"))
    csv_path = args.output / "weak_lensing_masses.csv"
    table.to_csv(csv_path, index=False)
    provenance = {
        "dataset": "CCCP and MENeaCS updated weak-lensing masses",
        "source_url": SOURCE_URL,
        "arxiv_id": "1912.04414",
        "citation": "Herbonnet et al. 2020, MNRAS 497, 4684",
        "doi": "10.1093/mnras/staa2303",
        "rows": len(table),
        "mass_choice_for_cpr0": "simulation-corrected deprojected aperture M500 at its weak-lensing R500",
        "files": {
            archive_path.name: sha256(archive_path),
            tex_path.name: sha256(tex_path),
            csv_path.name: sha256(csv_path),
        },
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
