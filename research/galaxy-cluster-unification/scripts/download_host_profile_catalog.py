from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
TABLE_NAME = "J/ApJ/805/3/clusters"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the published Chandra gas profiles.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "raw" / "cluster_gas_profiles" / "elkholy2015",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    query = (
        'SELECT "ID","R500","M500","Mgas","n0","alpha","beta","rc","rs","eps" '
        f'FROM "{TABLE_NAME}"'
    )
    parameters = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "votable",
        "QUERY": query,
    }
    with urlopen(f"{TAP_URL}?{urlencode(parameters)}", timeout=180) as response:
        content = response.read()
    votable_path = args.output / "clusters.vot"
    votable_path.write_bytes(content)
    table = Table.read(votable_path, format="votable")
    if len(table) != 46:
        raise RuntimeError(f"expected 46 published profiles, received {len(table)}")
    table.write(args.output / "clusters.csv", format="ascii.csv", overwrite=True)
    provenance = {
        "source": TAP_URL,
        "table": TABLE_NAME,
        "query": query,
        "rows": len(table),
        "votable_sha256": sha256(votable_path),
        "csv_sha256": sha256(args.output / "clusters.csv"),
        "paper": "Elkholy et al. 2015, ApJ 805, 3",
        "doi": "10.1088/0004-637X/805/1/3",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
