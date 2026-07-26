"""Grouped public-data loaders with hashes and provenance records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

TIAN_FIG2_URL = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/896/70/fig2.dat"
TIAN_README_URL = (
    "https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJ/896/70?format=html&tex=true"
)
MISTELE_RECORD_URL = "https://zenodo.org/api/records/15476959"


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_checksum: str | None = None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    destination.write_bytes(response.content)
    checksum = sha256_file(destination)
    if expected_checksum:
        algorithm, expected = expected_checksum.split(":", 1)
        if algorithm.lower() == "md5":
            actual = hashlib.md5(response.content).hexdigest()  # noqa: S324 - provenance only
        elif algorithm.lower() == "sha256":
            actual = checksum
        else:
            raise ValueError(f"unsupported checksum algorithm: {algorithm}")
        if actual.lower() != expected.lower():
            destination.unlink(missing_ok=True)
            raise ValueError(f"checksum mismatch for {url}")
    return {"url": url, "path": str(destination), "sha256": checksum, "bytes": len(response.content)}


def download_tian2020(data_root) -> dict:
    root = Path(data_root) / "tian2020"
    destination = root / "fig2.dat"
    record = _download(TIAN_FIG2_URL, destination)
    record["path"] = f"tian2020/{destination.name}"
    manifest = {
        "dataset": "Tian et al. 2020 CLASH radial acceleration catalog",
        "readme": TIAN_README_URL,
        "files": [record],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_tian2020(path) -> pd.DataFrame:
    """Load fig2.dat, preserving cluster as the grouping key."""
    columns = ["cluster", "radius_kpc", "log_gbar", "log_gtot", "err_log_gbar", "err_log_gtot"]
    frame = pd.read_csv(path, sep=r"\s+", names=columns, comment="#")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["gbar"] = np.power(10.0, frame["log_gbar"])
    frame["gtot"] = np.power(10.0, frame["log_gtot"])
    frame["dataset"] = "Tian2020_CLASH"
    frame["group_id"] = frame["cluster"]
    return frame


def download_mistele2025(data_root) -> dict:
    """Download the 20 main profiles and their enclosed-mass correlations."""
    root = Path(data_root) / "mistele2025"
    root.mkdir(parents=True, exist_ok=True)
    metadata = requests.get(MISTELE_RECORD_URL, timeout=90)
    metadata.raise_for_status()
    record = metadata.json()
    selected = []
    for item in record["files"]:
        key = item["key"]
        is_main_profile = key.endswith(".csv") and "-corr" not in key and key != "vflat-Mb-M200c.csv"
        is_mass_correlation = key.endswith("-M-corr.csv")
        if not (is_main_profile or is_mass_correlation or key == "vflat-Mb-M200c.csv"):
            continue
        checksum = item.get("checksum")
        file_record = _download(
            item["links"]["self"], root / key, expected_checksum=checksum
        )
        file_record["path"] = f"mistele2025/{key}"
        selected.append(file_record)
    manifest = {
        "dataset": "Mistele et al. 2025 CLASH acceleration profiles",
        "zenodo_record": "https://zenodo.org/records/15476959",
        "record_id": record["id"],
        "files": selected,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_mistele_profiles(directory) -> dict[str, dict]:
    directory = Path(directory)
    profiles: dict[str, dict] = {}
    for path in sorted(directory.glob("*.csv")):
        if "-corr" in path.name or path.name == "vflat-Mb-M200c.csv":
            continue
        key = path.stem.lower()
        profile = pd.read_csv(path)
        correlation_path = directory / f"{key}-M-corr.csv"
        correlation = None
        if correlation_path.exists():
            correlation = pd.read_csv(correlation_path, header=None).to_numpy(dtype=float)
        profiles[key] = {
            "data": profile,
            "mass_correlation": correlation,
            "profile_sha256": sha256_file(path),
            "correlation_sha256": sha256_file(correlation_path) if correlation_path.exists() else None,
        }
    return profiles


def normalize_cluster_name(name: str) -> str:
    value = "".join(character.lower() for character in str(name) if character.isalnum())
    replacements = {
        "macsj041612403": "macs0416",
        "macsj071753745": "macs0717",
        "macsj114952223": "macs1149",
        "abell": "a",
    }
    if value in replacements:
        return replacements[value]
    if value.startswith("abell"):
        return "a" + value[5:]
    if value.startswith("macsj"):
        return "macs" + value[5:9]
    return value


def fox_overlap_names(fox_csv) -> set[str]:
    frame = pd.read_csv(fox_csv)
    if "M500_1e14Msun" in frame:
        frame = frame[pd.to_numeric(frame["M500_1e14Msun"], errors="coerce") > 2.0]
    column = "cluster" if "cluster" in frame else "cluster_name"
    return {normalize_cluster_name(name) for name in frame[column].dropna()}
