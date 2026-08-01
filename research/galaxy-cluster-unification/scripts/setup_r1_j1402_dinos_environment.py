#!/usr/bin/env python3
"""Build the frozen date-bounded J1402 Dolphin replay environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "r1_j1402_dinos_coordinate_replay_protocol.json"
CORRECTION_REPORT = ROOT / "results" / "r1_j1402_dinos_coordinate_audit_corrected" / "report.json"
SOURCE_DIR = ROOT / "data" / "raw" / "r1_j1402" / "software"
SOURCE_ARCHIVE = SOURCE_DIR / "dolphin-v0.0.1-source.tar.gz"
SOURCE_MANIFEST = SOURCE_DIR / "source_manifest.json"
VENV = ROOT / ".venv-r1-j1402-dolphin-v001"
PIP_FREEZE = ROOT / "data" / "derived" / "r1_j1402_dinos_environment_pip_freeze.txt"
COMMIT = "1593c573541d26ae5791835430c68858988a969b"
EXTRACTED_SOURCE = SOURCE_DIR / f"dolphin-{COMMIT}"
SOURCE_URL = f"https://codeload.github.com/ajshajib/dolphin/tar.gz/{COMMIT}"
EXPECTED_REQUIREMENTS_BLOB = "a289b4e0a8ee4176901fab02983be4fffd6cd23a"
PACKAGES = [
    "numpy==1.26.4",
    "scipy==1.11.4",
    "h5py==3.11.0",
    "PyYAML==6.0.2",
    "emcee==3.1.6",
    "schwimmbad==0.4.2",
    "corner==2.2.3",
    "matplotlib==3.8.4",
    "numba==0.58.1",
    "llvmlite==0.41.1",
    "astropy==5.3.4",
    "lenstronomy==1.11.5",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(data: bytes) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest()


def inspect_source_archive(path: Path) -> dict:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        requirements_members = [
            item for item in members if item.name.endswith("/requirements.txt")
        ]
        setup_members = [item for item in members if item.name.endswith("/setup.py")]
        if len(requirements_members) != 1 or len(setup_members) != 1:
            raise ValueError("Dolphin source archive does not have one requirements.txt and setup.py")
        extracted = archive.extractfile(requirements_members[0])
        if extracted is None:
            raise ValueError("cannot read requirements.txt from Dolphin source archive")
        requirements = extracted.read()
        roots = sorted({item.name.split("/", 1)[0] for item in members})
    observed_blob = git_blob_sha1(requirements)
    if observed_blob != EXPECTED_REQUIREMENTS_BLOB:
        raise ValueError(
            f"requirements blob mismatch {observed_blob} != {EXPECTED_REQUIREMENTS_BLOB}"
        )
    if roots != [f"dolphin-{COMMIT}"]:
        raise ValueError(f"unexpected codeload archive root: {roots}")
    return {
        "member_count": len(members),
        "archive_roots": roots,
        "requirements_member": requirements_members[0].name,
        "requirements_git_blob_sha1": observed_blob,
        "requirements_text": requirements.decode("utf-8"),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def acquire_source() -> dict:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_ARCHIVE.exists():
        if not SOURCE_MANIFEST.exists():
            raise FileExistsError("unmanifested Dolphin source archive already exists")
        manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
        if sha256(SOURCE_ARCHIVE) != manifest["archive_sha256"]:
            raise ValueError("existing Dolphin source archive checksum mismatch")
        inspect_source_archive(SOURCE_ARCHIVE)
        return manifest

    temporary = SOURCE_ARCHIVE.with_suffix(SOURCE_ARCHIVE.suffix + ".part")
    request = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent": "SigmaGravity-J1402-replay/0.1"}
    )
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        while chunk := response.read(4 * 1024 * 1024):
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, SOURCE_ARCHIVE)
    inspection = inspect_source_archive(SOURCE_ARCHIVE)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "https://github.com/ajshajib/dolphin",
        "tag": "v0.0.1",
        "commit": COMMIT,
        "source_url": SOURCE_URL,
        "archive_path": str(SOURCE_ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
        "archive_bytes": SOURCE_ARCHIVE.stat().st_size,
        "archive_sha256": sha256(SOURCE_ARCHIVE),
        **inspection,
    }
    atomic_json(SOURCE_MANIFEST, manifest)
    return manifest


def expose_exact_source_tree(manifest: dict) -> dict:
    if not EXTRACTED_SOURCE.exists():
        with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
            for member in archive.getmembers():
                destination = (SOURCE_DIR / member.name).resolve()
                if destination != SOURCE_DIR and SOURCE_DIR.resolve() not in destination.parents:
                    raise ValueError(f"unsafe source archive member: {member.name}")
            archive.extractall(SOURCE_DIR, filter="data")
    requirements = (EXTRACTED_SOURCE / "requirements.txt").read_bytes()
    if git_blob_sha1(requirements) != EXPECTED_REQUIREMENTS_BLOB:
        raise ValueError("extracted requirements blob does not match the frozen commit")
    if not (EXTRACTED_SOURCE / "dolphin" / "processor" / "core.py").exists():
        raise FileNotFoundError("extracted Dolphin processor source is absent")
    manifest = dict(manifest)
    manifest["extracted_source_path"] = str(EXTRACTED_SOURCE.relative_to(ROOT)).replace(
        "\\", "/"
    )
    manifest["extracted_requirements_git_blob_sha1"] = git_blob_sha1(requirements)
    manifest["source_tree_modified"] = False
    atomic_json(SOURCE_MANIFEST, manifest)
    return manifest


def venv_python() -> Path:
    return VENV / "Scripts" / "python.exe"


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def build_environment() -> None:
    python = venv_python()
    if not python.exists():
        run(["py", "-3.10", "-m", "venv", str(VENV)])
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "pip==24.3.1",
            "setuptools==75.3.2",
            "wheel==0.45.1",
        ]
    )
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            *PACKAGES,
        ]
    )
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--editable",
            str(EXTRACTED_SOURCE),
        ]
    )
    freeze = subprocess.check_output(
        [str(python), "-m", "pip", "freeze", "--all"], cwd=ROOT, text=True
    )
    PIP_FREEZE.parent.mkdir(parents=True, exist_ok=True)
    PIP_FREEZE.write_text(freeze, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    corrected = json.loads(CORRECTION_REPORT.read_text(encoding="utf-8"))
    if not corrected["gate_pass"]:
        raise ValueError("corrected structural coordinate gate has not passed")
    if args.plan:
        print(
            json.dumps(
                {
                    "python": protocol["software_lock"]["python"],
                    "venv": str(VENV),
                    "source_url": SOURCE_URL,
                    "commit": COMMIT,
                    "packages": PACKAGES,
                },
                indent=2,
            )
        )
        return
    manifest = expose_exact_source_tree(acquire_source())
    print(
        f"SOURCE OK {manifest['archive_bytes']} bytes sha256={manifest['archive_sha256']}",
        flush=True,
    )
    build_environment()
    run([str(venv_python()), "scripts/audit_r1_j1402_dinos_environment.py"])


if __name__ == "__main__":
    main()
