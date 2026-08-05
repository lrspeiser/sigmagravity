from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v19w2_exact_binmap_response_commissioning.json"
SCRIPT = ROOT / "scripts" / "run_sigma_v19w2_exact_binmap_response_commissioning.py"
REPORT = ROOT / "results" / "sigma_v19w2_exact_binmap_response_commissioning" / "report.json"
SPEC = importlib.util.spec_from_file_location("sigma_v19w2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules.setdefault("pycrates", types.ModuleType("pycrates"))
SPEC.loader.exec_module(MODULE)


def test_exact_bin_mask_is_binary_disjoint_and_reusable(tmp_path: Path):
    source = tmp_path / "binmap.fits"
    values = np.asarray([[0, 0, 1], [2, 1, -1]], dtype=np.int16)
    fits.PrimaryHDU(values).writeto(source)
    destination = tmp_path / "mask.fits"
    first = MODULE.write_exact_bin_mask(source, 1, destination)
    second = MODULE.write_exact_bin_mask(source, 1, destination)
    with fits.open(destination, memmap=False) as hdus:
        mask = np.asarray(hdus[0].data)
    assert set(np.unique(mask)) == {0, 1}
    assert int(mask.sum()) == 2
    assert first["sha256"] == second["sha256"]
    assert first["reused"] is False
    assert second["reused"] is True


def test_frozen_selection_covers_all_observed_implementation_classes():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    classes = " ".join(item["failure_class"] for item in config["commissioning_cells"])
    assert len(config["commissioning_cells"]) == 5
    assert "source polygon" in classes
    assert "background polygon" in classes
    assert "source and background polygons" in classes
    assert "adjacent CCD" in classes
    assert "AF_UNIX path too long" in classes
    assert config["execution"]["attempts_per_commissioning_cell"] == 1
    assert config["execution"]["base_v19w_archive_is_read_only"]
    assert not config["advance"]["v19x_authorized_here"]


def test_implementation_change_preserves_scientific_settings():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    correction = config["implementation_correction"]
    assert "binmap equals bin_id" in correction["source_selection"]
    assert "same exact bin mask" in correction["background_selection"]
    assert "refcoord unset" in correction["response_position"]
    assert "weight=yes" in correction["response_weighting"]
    assert correction["scientific_values_changed"] is False


def test_frozen_runner_hash_is_exact():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["implementation"]["runner"] == SCRIPT.relative_to(ROOT).as_posix()
    assert MODULE.sha256(SCRIPT) == config["implementation"]["runner_sha256"]


def test_completed_commissioning_passes_without_authorizing_v19x():
    if not REPORT.exists():
        return
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == (
        "exact_binmap_response_commissioning_passed_and_recovery_protocol_authorized"
    )
    assert all(report["gates"].values())
    assert len(report["completed_cells"]) == 5
    assert report["full_missing_cell_recovery_authorized"]
    assert not report["base_v19w_archive_modified"]
    assert not report["spectrum_combined_or_fitted"]
    assert not report["gravity_formula_or_parameter_changed"]
