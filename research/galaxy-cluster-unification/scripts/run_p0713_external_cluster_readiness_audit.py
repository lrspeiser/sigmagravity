#!/usr/bin/env python3
"""Audit the frozen P0633 raw strong-lensing catalogs for scoring readiness.

This stage deliberately stops before evaluating any gravity prediction.  It
parses only the already-unsealed image catalogs and applies the readiness gates
that were committed in P0633 and copied into the P0709 unlock manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/p0633_external_validation_preregistration.json"
DEFAULT_UNLOCK = ROOT / "results/p0633_external_validation/unlock_manifest.json"
CIBIRKA = ROOT / "data/sealed/p0633_relics_lensing/cibirka2018_arxiv_1803.09557_source.tar.gz"
PLCK = ROOT / "data/sealed/p0633_relics_lensing/daddona2024_tablec1_multiple_images.dat"
OUTPUT = ROOT / "results/p0713_external_cluster_readiness_audit"

CIBIRKA_TABLES = {
    "AS295": "S295",
    "MACS0025": "0025",
    "MACS0159": "0159",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_tex_cell(value: str) -> str:
    value = re.sub(r"\\tablenotemark\{[^}]+\}", "", value)
    value = value.replace("$", "").replace("\\ ", "")
    return value.strip().strip("\\").strip()


def first_number(value: str) -> float | None:
    value = clean_tex_cell(value)
    if not value or value in {"-", '"'}:
        return None
    match = re.search(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else None


def cibirka_table_block(tex: str, label: str) -> str:
    start_token = rf"\label{{table:{label}}}"
    start = tex.find(start_token)
    if start < 0:
        raise RuntimeError(f"Missing Cibirka table label {label}")
    end = tex.find(r"\end{tabular}", start)
    if end < 0:
        raise RuntimeError(f"Missing end of Cibirka table {label}")
    return tex[start:end]


def parse_cibirka() -> pd.DataFrame:
    with tarfile.open(CIBIRKA, "r:gz") as archive:
        tex = archive.extractfile("RELICS_LTM1.tex").read().decode("utf-8", "replace")

    records: list[dict[str, object]] = []
    for cluster, label in CIBIRKA_TABLES.items():
        block = cibirka_table_block(tex, label)
        family_spec: dict[str, float] = {}
        family_model: dict[str, float] = {}
        for raw_line in block.splitlines():
            if "&" not in raw_line or r"\\" not in raw_line:
                continue
            cells = [clean_tex_cell(cell) for cell in raw_line.split("&")]
            if len(cells) != 7:
                continue
            image_id = cells[0].replace(" ", "")
            if not re.fullmatch(r"[cp]?\d+\.\d+", image_id):
                continue
            prefix = image_id[0] if image_id[0].isalpha() else ""
            numeric_id = image_id[1:] if prefix else image_id
            family_id = numeric_id.split(".", 1)[0]
            secure_image = not prefix
            candidate_reason = "" if secure_image else {"c": "catalog_candidate", "p": "model_predicted_not_observed"}[prefix]

            spec = family_spec.get(family_id) if cells[3] == '"' else first_number(cells[3])
            model = family_model.get(family_id) if cells[5] == '"' else first_number(cells[5])
            if spec is not None:
                family_spec[family_id] = spec
            if model is not None:
                family_model[family_id] = model

            sky = SkyCoord(f"{cells[1]} {cells[2]}", unit=(u.hourangle, u.deg), frame="icrs")
            records.append(
                {
                    "cluster": cluster,
                    "family_id": family_id,
                    "image_id": image_id,
                    "ra_deg": sky.ra.deg,
                    "dec_deg": sky.dec.deg,
                    "secure_image": secure_image,
                    "selection_basis": "published_non_candidate_system" if secure_image else candidate_reason,
                    "spectroscopic_redshift": spec,
                    "spectroscopic_quality": "published" if spec is not None else "",
                    "adopted_catalog_redshift": spec if spec is not None else model,
                    "adopted_redshift_kind": "spectroscopic" if spec is not None else ("lens_model" if model is not None else "missing"),
                    "source_catalog": "Cibirka_et_al_2018_table",
                }
            )
    return pd.DataFrame.from_records(records)


def optional_fixed_float(line: str, start: int, end: int, missing: float = -99.0) -> float | None:
    value = line[start:end].strip()
    if not value:
        return None
    parsed = float(value)
    return None if math.isclose(parsed, missing) else parsed


def parse_plck() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for line in PLCK.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        name = line[0:6].strip()
        if name == "-99":
            continue
        match = re.fullmatch(r"(c?)(\d+)\.(\d+)([a-z])", name)
        if not match:
            raise RuntimeError(f"Unexpected PLCK image name {name!r}")
        candidate_prefix, source, clump, counterimage = match.groups()
        gold = line[47:48] == "y"
        zspec = optional_fixed_float(line, 34, 42)
        qf_raw = line[43:46].strip()
        qf = int(qf_raw) if qf_raw and qf_raw != "-99" else None
        secure_spec = zspec is not None and qf in {3, 9}
        records.append(
            {
                "cluster": "PLCKG287",
                "family_id": source,
                "image_id": name,
                "ra_deg": float(line[12:22]),
                "dec_deg": float(line[23:33]),
                "secure_image": bool(gold and not candidate_prefix),
                "selection_basis": "published_golden_sample" if gold and not candidate_prefix else ("catalog_candidate" if candidate_prefix else "not_in_golden_sample"),
                "spectroscopic_redshift": zspec if secure_spec else None,
                "spectroscopic_quality": str(qf) if qf is not None else "",
                "adopted_catalog_redshift": zspec if secure_spec else None,
                "adopted_redshift_kind": "spectroscopic_QF3_or_QF9" if secure_spec else "missing_or_low_quality",
                "source_catalog": "DAddona_et_al_2024_table_C1",
                "clump_id": clump,
                "counterimage_id": counterimage,
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--unlock", type=Path, default=DEFAULT_UNLOCK)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    unlock = json.loads(args.unlock.read_text(encoding="utf-8"))
    if unlock["status"] != "authorized_for_exactly_one_external_parse":
        raise RuntimeError("P0709 unlock manifest is not authorized")

    expected_hashes = {Path(row["path"]).name: row["sha256"] for row in unlock["sealed_constraint_containers"]}
    for path in (CIBIRKA, PLCK):
        if sha256(path) != expected_hashes[path.name]:
            raise RuntimeError(f"Frozen source hash changed: {path}")

    images = pd.concat([parse_cibirka(), parse_plck()], ignore_index=True, sort=False)
    selected_clusters = [row["id"] for row in protocol["cluster_validation"]["systems"]]
    if set(images.cluster) != set(selected_clusters):
        raise RuntimeError("Parsed clusters do not match the frozen P0633 selection")

    family_rows: list[dict[str, object]] = []
    for (cluster, family_id), group in images.groupby(["cluster", "family_id"], sort=True):
        secure = group[group.secure_image]
        spec = secure.spectroscopic_redshift.dropna()
        family_rows.append(
            {
                "cluster": cluster,
                "family_id": family_id,
                "catalog_images": len(group),
                "secure_images": len(secure),
                "secure_family": len(secure) > 0,
                "spectroscopic_family": len(spec) > 0,
                "spectroscopic_redshift": float(spec.median()) if len(spec) else None,
                "redshift_spread": float(spec.max() - spec.min()) if len(spec) else None,
            }
        )
    families = pd.DataFrame.from_records(family_rows)

    gates = protocol["cluster_validation"]["raw_constraint_readiness"]
    cluster_rows: list[dict[str, object]] = []
    for cluster in selected_clusters:
        family = families[(families.cluster == cluster) & families.secure_family]
        secure_images = int(family.secure_images.sum())
        secure_families = len(family)
        spectroscopic_families = int(family.spectroscopic_family.sum())
        family_gate = secure_families >= gates["minimum_secure_families_per_cluster"]
        spec_gate = spectroscopic_families >= gates["minimum_spectroscopic_families_per_cluster"]
        image_gate = secure_images >= gates["minimum_images_per_cluster"]
        cluster_rows.append(
            {
                "cluster": cluster,
                "secure_families": secure_families,
                "spectroscopic_families": spectroscopic_families,
                "secure_images": secure_images,
                "family_gate_passed": family_gate,
                "spectroscopic_gate_passed": spec_gate,
                "image_gate_passed": image_gate,
                "ready": family_gate and spec_gate and image_gate,
            }
        )
    clusters = pd.DataFrame.from_records(cluster_rows)
    ready_clusters = int(clusters.ready.sum())
    required_ready = int(protocol["rejection_thresholds"]["cluster"]["minimum_ready_clusters"])
    all_ready = ready_clusters >= required_ready

    OUTPUT.mkdir(parents=True, exist_ok=True)
    images.sort_values(["cluster", "family_id", "image_id"]).to_csv(OUTPUT / "parsed_image_catalog.csv", index=False)
    families.sort_values(["cluster", "family_id"]).to_csv(OUTPUT / "family_readiness.csv", index=False)
    clusters.to_csv(OUTPUT / "cluster_readiness.csv", index=False)

    report = {
        "stage": "P0713",
        "status": "pass" if all_ready else "fail_data_readiness",
        "evaluation_kind": "frozen_external_validation_data_readiness",
        "formula_scored": False,
        "sample_is_spent": True,
        "source_hashes": {path.name: sha256(path) for path in (CIBIRKA, PLCK)},
        "readiness_thresholds": gates,
        "required_ready_clusters": required_ready,
        "ready_clusters": ready_clusters,
        "cluster_rows": cluster_rows,
        "gate_results": {"minimum_ready_clusters": all_ready},
        "consequence": (
            "Proceed to the frozen four-cluster raw lens score."
            if all_ready
            else "The preregistered four-cluster raw lens score is not validly evaluable; selected clusters may not be replaced after unsealing."
        ),
        "interpretation_boundary": [
            "This failure is a public-catalog readiness failure, not a failure of the candidate gravity formula.",
            "Any raw score on the ready subset is descriptive/exploratory and cannot satisfy the P0633 cluster validation gate.",
            "Model-derived redshifts are retained in the parsed Cibirka table for provenance but do not create a spectroscopic family.",
            "PLCK secure images are exactly the authors' published GOLD=y sample; spectroscopic readiness requires QF 3 or 9.",
        ],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    table_lines = [
        "| Cluster | Secure families | Spectroscopic families | Secure images | Ready |",
        "|---|---:|---:|---:|:---:|",
    ]
    table_lines.extend(
        f"| {row.cluster} | {row.secure_families} | {row.spectroscopic_families} | {row.secure_images} | {'yes' if row.ready else 'no'} |"
        for row in clusters.itertuples(index=False)
    )
    table = "\n".join(table_lines)
    summary = f"""# P0713 frozen external cluster-readiness audit

- Status: **{report['status']}**.
- Ready clusters: **{ready_clusters} / {required_ready} required**.
- Gravity predictions scored: **no**.

{table}

The preregistered four-cluster lensing validation cannot proceed as a passing
P0633 test because selected targets may not be replaced after unsealing.  This
is a data-readiness result, not evidence for or against the candidate equation.
A score on any ready subset must be labeled descriptive/exploratory.
"""
    (OUTPUT / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
