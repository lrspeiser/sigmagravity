#!/usr/bin/env python3
"""Create the repo-local WSL2/Python 3.10 fastell replay environment."""

from __future__ import annotations

import bz2
import hashlib
import json
import os
import shlex
import subprocess
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORRECTION_PATH = ROOT / "configs" / "r1_j1402_dinos_fastell_dependency_correction.json"
SOURCE_DIR = ROOT / "data" / "raw" / "r1_j1402" / "software"
FASTELL_ARCHIVE = SOURCE_DIR / "fastell4py-3448d580-source.tar.gz"
FASTELL_SOURCE = SOURCE_DIR / "fastell4py-3448d58033ebbf1c"
DOLPHIN_SOURCE = SOURCE_DIR / "dolphin-1593c573541d26ae5791835430c68858988a969b"
MICROMAMBA_ARCHIVE = SOURCE_DIR / "micromamba-linux-64.tar.bz2"
MICROMAMBA_URL = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
MICROMAMBA = ROOT / ".toolchains" / "micromamba" / "bin" / "micromamba"
TOOLCHAIN = ROOT / ".toolchains" / "r1-j1402-gfortran"
VENV = ROOT / ".venv-r1-j1402-dolphin-v001-linux-fastell"
MANIFEST = SOURCE_DIR / "fastell_environment_source_manifest.json"
PIP_FREEZE = ROOT / "data" / "derived" / "r1_j1402_dinos_fastell_environment_pip_freeze.txt"
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


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != destination.resolve() and destination.resolve() not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(destination)


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if len(drive) != 1:
        raise ValueError(f"cannot map non-drive Windows path into WSL: {resolved}")
    tail = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def wsl_bash(script: str, capture: bool = False) -> str:
    command = ["wsl", "-d", "Ubuntu-24.04", "--", "bash", "-lc", script]
    print("WSL", script, flush=True)
    if capture:
        return subprocess.check_output(command, text=True)
    subprocess.run(command, check=True)
    return ""


def acquire_micromamba() -> dict:
    if not MICROMAMBA_ARCHIVE.exists():
        request = urllib.request.Request(
            MICROMAMBA_URL, headers={"User-Agent": "SigmaGravity-J1402-fastell/0.1"}
        )
        temporary = MICROMAMBA_ARCHIVE.with_suffix(".part")
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, MICROMAMBA_ARCHIVE)
    MICROMAMBA.parent.mkdir(parents=True, exist_ok=True)
    if not MICROMAMBA.exists():
        with tarfile.open(MICROMAMBA_ARCHIVE, "r:bz2") as archive:
            member = archive.getmember("bin/micromamba")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("micromamba archive lacks bin/micromamba")
            MICROMAMBA.write_bytes(source.read())
    version = wsl_bash(f"{shlex.quote(wsl_path(MICROMAMBA))} --version", capture=True).strip()
    return {
        "url": MICROMAMBA_URL,
        "archive": str(MICROMAMBA_ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
        "archive_bytes": MICROMAMBA_ARCHIVE.stat().st_size,
        "archive_sha256": sha256(MICROMAMBA_ARCHIVE),
        "binary_sha256": sha256(MICROMAMBA),
        "version": version,
    }


def build() -> None:
    correction = json.loads(CORRECTION_PATH.read_text(encoding="utf-8"))
    expected = correction["primary_source_evidence"]
    if FASTELL_ARCHIVE.stat().st_size != expected["archive_bytes"] or sha256(FASTELL_ARCHIVE) != expected["archive_sha256"]:
        raise ValueError("frozen fastell source archive checksum mismatch")
    if not (FASTELL_SOURCE / "src" / "fastell.f").exists():
        safe_extract(FASTELL_ARCHIVE, FASTELL_SOURCE)
    if not (FASTELL_SOURCE / "src" / "fastell.f").exists():
        raise FileNotFoundError("exact fastell Fortran source is absent")
    if not (DOLPHIN_SOURCE / "dolphin" / "processor" / "core.py").exists():
        raise FileNotFoundError("exact Dolphin source is absent")

    micromamba = acquire_micromamba()
    mm = shlex.quote(wsl_path(MICROMAMBA))
    toolchain = shlex.quote(wsl_path(TOOLCHAIN))
    venv = shlex.quote(wsl_path(VENV))
    fastell = shlex.quote(wsl_path(FASTELL_SOURCE))
    dolphin = shlex.quote(wsl_path(DOLPHIN_SOURCE))
    root = shlex.quote(wsl_path(ROOT))
    if not (TOOLCHAIN / "bin" / "x86_64-conda-linux-gnu-gfortran").exists():
        wsl_bash(
            f"{mm} create --yes --prefix {toolchain} --channel conda-forge "
            "gcc_linux-64=12.4.0 gfortran_linux-64=12.4.0"
        )
    if not (VENV / "pyvenv.cfg").exists():
        wsl_bash(f"uv venv --python 3.10.11 --seed {venv}")
    python = f"{venv}/bin/python"
    package_args = " ".join(shlex.quote(item) for item in PACKAGES)
    wsl_bash(
        f"{python} -m pip install --disable-pip-version-check "
        "pip==24.3.1 setuptools==59.8.0 wheel==0.45.1"
    )
    wsl_bash(f"{python} -m pip install --disable-pip-version-check {package_args}")
    wsl_bash(f"{python} -m pip install --disable-pip-version-check --no-deps --editable {dolphin}")
    build_env = (
        f"export PATH={toolchain}/bin:\"$PATH\"; "
        "export CC=x86_64-conda-linux-gnu-gcc; "
        "export FC=x86_64-conda-linux-gnu-gfortran; "
        "export F77=x86_64-conda-linux-gnu-gfortran; "
    )
    wsl_bash(
        f"ln -sfn x86_64-conda-linux-gnu-gfortran {toolchain}/bin/gfortran; "
        f"{build_env} cd {fastell}/fastell4py; "
        f"{python} -m numpy.f2py -c ../fastell.pyf ../src/fastell.f "
        "--fcompiler=gnu95"
    )
    python_path = f"{fastell}:{dolphin}"
    wsl_bash(
        f"{build_env} export PYTHONPATH={python_path}; cd {root}; "
        f"{python} scripts/audit_r1_j1402_dinos_fastell_environment.py"
    )
    freeze = wsl_bash(f"{python} -m pip freeze --all", capture=True)
    PIP_FREEZE.parent.mkdir(parents=True, exist_ok=True)
    PIP_FREEZE.write_text(freeze, encoding="utf-8")
    compiler = wsl_bash(
        f"{toolchain}/bin/x86_64-conda-linux-gnu-gfortran --version | head -1",
        capture=True,
    ).strip()
    atomic_json(
        MANIFEST,
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "correction_protocol": correction["protocol_version"],
            "fastell_source_archive": str(FASTELL_ARCHIVE.relative_to(ROOT)).replace("\\", "/"),
            "fastell_source_archive_bytes": FASTELL_ARCHIVE.stat().st_size,
            "fastell_source_archive_sha256": sha256(FASTELL_ARCHIVE),
            "fastell_source_commit": expected["commit"],
            "source_tree_modified": False,
            "micromamba": micromamba,
            "compiler": compiler,
            "toolchain_path": str(TOOLCHAIN.relative_to(ROOT)).replace("\\", "/"),
            "environment_path": str(VENV.relative_to(ROOT)).replace("\\", "/"),
            "pip_freeze": str(PIP_FREEZE.relative_to(ROOT)).replace("\\", "/"),
        },
    )


if __name__ == "__main__":
    build()
