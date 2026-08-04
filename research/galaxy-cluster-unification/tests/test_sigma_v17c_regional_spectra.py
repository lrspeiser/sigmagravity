from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v17c_regional_spectra.py"
CONFIG = ROOT / "configs" / "sigma_v17c_spectral_temperature.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("sigma_v17c_regional", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_regional_runner_is_hard_gated_by_both_integrated_results(tmp_path: Path) -> None:
    module = _load_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"protocol_version": "P"}), encoding="utf-8")
    spectra_path = tmp_path / "spectra.json"
    spectra = {
        "status": "both_frozen_integrated_spectra_extracted_combined_and_grouped",
        "protocol_version": "P",
        "config_sha256": _sha256(config_path),
    }
    spectra_path.write_text(json.dumps(spectra), encoding="utf-8")
    temperatures_path = tmp_path / "temperatures.json"
    temperatures = {
        "status": "both_integrated_temperature_gates_passed",
        "protocol_version": "P",
        "config_sha256": _sha256(config_path),
        "spectral_extraction_report_sha256": _sha256(spectra_path),
        "regional_fit_authorized": True,
    }
    temperatures_path.write_text(json.dumps(temperatures), encoding="utf-8")

    module.validate_authorization(
        config_path,
        {"protocol_version": "P"},
        spectra_path,
        spectra,
        temperatures_path,
        temperatures,
    )
    temperatures["regional_fit_authorized"] = False
    with pytest.raises(RuntimeError, match="not authorized"):
        module.validate_authorization(
            config_path,
            {"protocol_version": "P"},
            spectra_path,
            spectra,
            temperatures_path,
            temperatures,
        )


def test_every_region_cell_has_private_ciao_state(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    captured: dict[str, Path] = {}

    def fake_environment(_base, pfiles: Path, scratch: Path):
        captured["pfiles"] = pfiles
        captured["scratch"] = scratch
        return {}

    monkeypatch.setattr(module, "isolated_environment", fake_environment)
    monkeypatch.setattr(module, "run_step", lambda *_args: {"reused": False})
    monkeypatch.setattr(module, "verify_blanksky_scaling", lambda *_args: {"ok": True})
    monkeypatch.setattr(module, "sha256", lambda _path: "digest")
    task = {
        "cluster": "AS295",
        "region_id": 7,
        "obsid": 16524,
        "ccd_id": 2,
        "command": ["specextract"],
        "log": tmp_path / "cell.log",
        "source_pha": tmp_path / "source.pi",
        "background_pha": tmp_path / "background.pi",
        "arf": tmp_path / "source.arf",
        "rmf": tmp_path / "source.rmf",
        "bkgscale_value": 0.1,
        "source_band_events": 100,
        "background_band_events": 20,
        "response_reference": {},
        "translated_fov": {},
    }
    result = module.execute_regional_cell(task, tmp_path, "spectral_v17c_v106")
    expected_tail = Path("regional") / "AS295" / "region_007" / "16524_ccd2"
    assert str(captured["pfiles"]).endswith(str(expected_tail))
    assert str(captured["scratch"]).endswith(str(expected_tail))
    assert result["region_id"] == 7


def test_region_planning_preserves_frozen_extraction_conventions(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "event_count", lambda *_args: 12)
    monkeypatch.setattr(
        module,
        "event_reference_coordinate",
        lambda *_args: {
            "events": 12,
            "dmcoords_chip_id": 3,
            "ra_deg": 10.0,
            "dec_deg": -20.0,
        },
    )
    monkeypatch.setattr(module, "celestial_coordinate_chip", lambda *_args: 3)
    region = tmp_path / "xaf_4.reg"
    region.write_text("circle(1,1,1)\n", encoding="utf-8")
    context = {
        "obsid": 12260,
        "science": tmp_path / "science.fits",
        "background": tmp_path / "background.fits",
        "aspect": tmp_path / "aspect.lis",
        "mask": tmp_path / "mask.fits",
        "badpix": tmp_path / "badpix.fits",
        "translated_fov": tmp_path / "fov.fits",
        "translated_fov_record": {},
        "blanksky_scaling": {"BKGSCAL3": "0.04"},
    }
    tasks, skipped = module.plan_region(
        "AS295", region, [context], tmp_path / "work", {}
    )
    assert skipped == []
    assert len(tasks) == 1
    command = tasks[0]["command"]
    for token in (
        "bkgresp=no",
        "weight=yes",
        "weight_rmf=yes",
        "correctpsf=no",
        "combine=no",
        "grouptype=NONE",
        "bkg_grouptype=NONE",
        "energy=0.3:11.0:0.01",
        "energy_wmap=500:7000",
        "binwmap=det=8",
        "binarfwmap=1",
        "parallel=no",
        "nproc=1",
    ):
        assert token in command
    assert tasks[0]["region_id"] == 4


def test_regional_runner_matches_frozen_region_gates_and_claim_boundary() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = SCRIPT.read_text(encoding="utf-8")
    assert config["gates"]["regional"]["minimum_passing_regions_per_cluster"] == 12
    assert config["gates"]["regional"]["maximum_reduced_statistic"] == 1.5
    assert config["gates"]["regional"]["maximum_fractional_68_percent_half_width"] == 0.5
    assert '"thermal_stress_constructed": False' in source
    assert '"lensing_target_opened": False' in source
    assert '"regional_temperature_fit_authorized": True' in source
