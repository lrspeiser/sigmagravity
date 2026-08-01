#!/usr/bin/env python3
"""Audit the Linux fastell compatibility environment without reading science pixels."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import fastell4py
import h5py
import lenstronomy
import numpy as np
import scipy
from dolphin.processor.config import ModelConfig
from lenstronomy.LensModel.Profiles.pemd import PEMD


ROOT = Path(__file__).resolve().parents[1]
CORRECTION_PATH = ROOT / "configs" / "r1_j1402_dinos_fastell_dependency_correction.json"
UPSTREAM_REPORT_PATH = ROOT / "results" / "r1_j1402_dinos_environment_corrected" / "report.json"
REPORT_PATH = ROOT / "results" / "r1_j1402_dinos_fastell_environment" / "report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dist_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> None:
    correction = json.loads(CORRECTION_PATH.read_text(encoding="utf-8"))
    upstream = json.loads(UPSTREAM_REPORT_PATH.read_text(encoding="utf-8"))
    archive = ROOT / correction["primary_source_evidence"]["archive"]
    settings = ROOT / "data/raw/r1_j1402/dinos_repo/2_dolphin_modelling/settings/SDSSJ1402+6321_config.yml"

    model = PEMD()
    alpha_x, alpha_y = model.derivatives(
        np.asarray([0.31, -0.47]),
        np.asarray([0.22, 0.19]),
        theta_E=1.37,
        gamma=2.05,
        e1=0.04,
        e2=0.09,
    )
    config = ModelConfig(str(settings))
    model_list = config.get_kwargs_model()["lens_model_list"]
    compiler = subprocess.check_output(
        ["x86_64-conda-linux-gnu-gfortran", "--version"], text=True
    ).splitlines()[0]
    extension_candidates = sorted(
        Path(fastell4py.__file__).resolve().parent.glob("_fastell*.so")
    )
    if len(extension_candidates) != 1:
        raise RuntimeError(f"expected one compiled fastell extension, got {extension_candidates}")
    extension = extension_candidates[0]

    versions = {
        "python": platform.python_version(),
        "dolphin_distribution": dist_version("dolphin"),
        "lenstronomy": lenstronomy.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "h5py": dist_version("h5py"),
        "PyYAML": dist_version("PyYAML"),
        "emcee": dist_version("emcee"),
        "schwimmbad": dist_version("schwimmbad"),
        "corner": dist_version("corner"),
        "matplotlib": dist_version("matplotlib"),
        "numba": dist_version("numba"),
        "llvmlite": dist_version("llvmlite"),
        "astropy": dist_version("astropy"),
        "fastell4py_source_commit": correction["primary_source_evidence"]["commit"],
        "fortran_compiler": compiler,
    }
    checks = {
        "upstream_coordinate_and_interface_environment_gate_passed": bool(
            upstream["gate_pass"]
        ),
        "python_3_10_environment": versions["python"].startswith("3.10."),
        "critical_scientific_package_versions_unchanged": all(
            versions[name] == value
            for name, value in upstream["versions"].items()
            if name not in {"python"}
        ),
        "fastell_source_archive_checksum_matches_frozen_correction": bool(
            archive.stat().st_size
            == correction["primary_source_evidence"]["archive_bytes"]
            and sha256(archive)
            == correction["primary_source_evidence"]["archive_sha256"]
        ),
        "compiled_fastell_extension_present": extension.is_file(),
        "fastell_PEMD_deflection_is_finite_and_nonzero": bool(
            np.isfinite(alpha_x).all()
            and np.isfinite(alpha_y).all()
            and np.any(np.abs(alpha_x) > 0)
            and np.any(np.abs(alpha_y) > 0)
        ),
        "released_model_remains_PEMD_not_EPL": model_list[0] == "PEMD",
        "released_model_constructs_without_suppress_fastell": bool(
            model.spemd_smooth._fastell4py_bool
        ),
        "science_pixels_not_read": True,
        "forward_model_not_evaluated": True,
        "nonlinear_fit_not_performed": True,
    }
    gate_pass = all(checks.values())
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": correction["protocol_version"],
        "upstream_environment_report": str(UPSTREAM_REPORT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "versions": versions,
        "fastell": {
            "module": str(Path(fastell4py.__file__).resolve()),
            "compiled_extension": str(extension),
            "compiled_extension_bytes": extension.stat().st_size,
            "compiled_extension_sha256": sha256(extension),
            "source_archive": str(archive.relative_to(ROOT)).replace("\\", "/"),
            "source_archive_sha256": sha256(archive),
            "test_deflection_x": np.asarray(alpha_x).tolist(),
            "test_deflection_y": np.asarray(alpha_y).tolist(),
        },
        "checks": checks,
        "gate_pass": gate_pass,
        "decision": (
            "fastell_environment_gate_pass_authorize_exact_stored_chain_replay"
            if gate_pass
            else "stop_J1402_fastell_environment_failure"
        ),
        "authorization": {
            "evaluate_only_stored_chain_coordinates": gate_pass,
            "optimize_nonlinear_model": False,
            "compute_lens_response": False,
            "reduce_KCWI": False,
            "count_toward_ten_system_target": False,
            "infer_gravity_response": False,
            "fit_new_force_or_action": False,
            "authorize_R2": False,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(report["decision"])


if __name__ == "__main__":
    main()
