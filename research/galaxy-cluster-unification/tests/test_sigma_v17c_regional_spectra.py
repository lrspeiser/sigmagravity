from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v17c_regional_spectra.py"
CONFIG = ROOT / "configs" / "sigma_v17c_spectral_temperature.json"
SUPPORT = ROOT / "configs" / "sigma_v17c_regional_response_support.json"
RUNTIME_SUPPORT = (
    ROOT / "configs" / "sigma_v17c_regional_runtime_response_support.json"
)
RUNTIME_SUPPORT_REPORT = (
    ROOT / "results" / "sigma_v17c_regional_runtime_response_support" / "report.json"
)
SUPPORT_REPORT = (
    ROOT / "results" / "sigma_v17c_regional_response_support" / "report.json"
)
INTEGRATED_SPECTRA = ROOT / "results" / "sigma_v17c_integrated_spectra" / "report.json"
INTEGRATED_TEMPERATURES = (
    ROOT / "results" / "sigma_v17c_integrated_temperatures" / "report.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("sigma_v17c_regional", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _response_support(module) -> dict:
    return {
        "admission_rule": {
            "allowed_calibrated_response_skip_reasons": [
                module.OFF_CCD_RESPONSE_REASON,
                module.MISSING_CCD_SUPPORT_REASON,
            ]
        }
    }


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
    result = module.execute_regional_cell(task, tmp_path, "spectral_v17c_v107")
    expected_pfiles_tail = (
        Path("regional") / "AS295" / "region_007" / "16524_ccd2"
    )
    expected_runtime = (
        module.REGIONAL_RUNTIME_TMP_ROOT
        / "spectral_v17c_v107"
        / "AS295"
        / "r007"
        / "16524c2"
    )
    assert str(captured["pfiles"]).endswith(str(expected_pfiles_tail))
    assert captured["scratch"] == expected_runtime
    assert len(str(expected_runtime).encode()) <= module.MAX_REGIONAL_RUNTIME_TMP_BYTES
    assert result["runtime_tmp"] == str(expected_runtime)
    assert result["region_id"] == 7


def test_exact_empty_response_signature_is_quarantined_and_reusable(
    tmp_path: Path,
) -> None:
    module = _load_module()
    individual = tmp_path / "region_012" / "individual"
    logs = tmp_path / "region_012" / "logs"
    individual.mkdir(parents=True)
    logs.mkdir(parents=True)
    outroot = individual / "acisf16524_ccd3_region012"
    source = outroot.with_suffix(".pi")
    background = outroot.with_name(outroot.name + "_bkg.pi")
    arf = outroot.with_suffix(".arf")
    rmf = outroot.with_suffix(".rmf")
    source.write_bytes(b"source")
    background.write_bytes(b"background")
    log = logs / "16524_ccd3_specextract.log"
    log.write_text(
        "Extracting src spectra\nExtracting bkg spectra\n"
        "# specextract: ERROR max() iterable argument is empty\n",
        encoding="utf-8",
    )
    task = {
        "cluster": "AS295",
        "region_id": 12,
        "obsid": 16524,
        "ccd_id": 3,
        "source_pha": source,
        "background_pha": background,
        "arf": arf,
        "rmf": rmf,
        "source_band_events": 7,
        "background_band_events": 5,
        "response_reference": {"ra_deg": 1.0, "dec_deg": 2.0},
        "allow_empty_calibrated_response_skip": True,
        "log": log,
    }

    record = module.classify_and_quarantine_empty_response_support(task)

    assert record is not None
    assert record["reason"] == module.EMPTY_CALIBRATED_RESPONSE_REASON
    assert not source.exists()
    assert not background.exists()
    assert len(record["quarantined_partial_products"]) == 2
    marker = module.response_support_skip_marker(task)
    assert marker.is_file()
    reused = module.load_response_support_skip(task)
    assert reused is not None and reused["reused"] is True
    assert reused["marker_sha256"] == _sha256(marker)


def test_other_partial_response_failures_remain_fatal(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "source.pi"
    background = tmp_path / "background.pi"
    source.write_bytes(b"source")
    background.write_bytes(b"background")
    log = tmp_path / "cell.log"
    log.write_text("some other CIAO error\n", encoding="utf-8")
    task = {
        "cluster": "AS295",
        "region_id": 12,
        "obsid": 16524,
        "ccd_id": 3,
        "source_pha": source,
        "background_pha": background,
        "arf": tmp_path / "source.arf",
        "rmf": tmp_path / "source.rmf",
        "source_band_events": 7,
        "background_band_events": 5,
        "response_reference": {},
        "allow_empty_calibrated_response_skip": True,
        "log": log,
    }

    assert module.classify_and_quarantine_empty_response_support(task) is None
    assert source.is_file() and background.is_file()


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
        "AS295",
        region,
        [context],
        _response_support(module),
        tmp_path / "work",
        {},
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


def test_region_planning_records_an_off_ccd_response_without_fitting(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "event_count", lambda *_args: 1)
    monkeypatch.setattr(
        module,
        "event_reference_coordinate",
        lambda *_args: {
            "events": 1,
            "dmcoords_chip_id": 3,
            "ra_deg": 10.0,
            "dec_deg": -20.0,
        },
    )
    region = tmp_path / "xaf_2.reg"
    region.write_text("circle(1,1,1)\n", encoding="utf-8")
    context = {
        "obsid": 16524,
        "science": tmp_path / "science.fits",
        "background": tmp_path / "background.fits",
        "aspect": tmp_path / "aspect.lis",
        "mask": tmp_path / "mask.fits",
        "badpix": tmp_path / "badpix.fits",
        "translated_fov": tmp_path / "fov.fits",
        "translated_fov_record": {},
        "blanksky_scaling": {"BKGSCAL2": "0.07"},
    }

    tasks, skipped = module.plan_region(
        "AS295",
        region,
        [context],
        _response_support(module),
        tmp_path / "work",
        {},
    )

    assert tasks == []
    assert skipped == [
        {
            "region_id": 2,
            "obsid": 16524,
            "ccd_id": 2,
            "reason": "event_mean_response_reference_maps_off_selected_ccd",
            "source_band_events": 1,
            "background_band_events": 1,
            "response_reference": {
                "events": 1,
                "dmcoords_chip_id": 3,
                "ra_deg": 10.0,
                "dec_deg": -20.0,
            },
        }
    ]


def test_response_support_freeze_is_hashed_and_target_blind() -> None:
    module = _load_module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    support = json.loads(SUPPORT.read_text(encoding="utf-8"))
    spectra = json.loads(INTEGRATED_SPECTRA.read_text(encoding="utf-8"))
    temperatures = json.loads(INTEGRATED_TEMPERATURES.read_text(encoding="utf-8"))

    module.validate_response_support(
        SUPPORT,
        support,
        CONFIG,
        INTEGRATED_SPECTRA,
        spectra,
        INTEGRATED_TEMPERATURES,
        temperatures,
    )
    assert support["parents"]["spectral_protocol_sha256"] == _sha256(CONFIG)
    assert support["parents"]["integrated_spectra_report_sha256"] == _sha256(
        INTEGRATED_SPECTRA
    )
    assert support["parents"]["integrated_temperatures_report_sha256"] == _sha256(
        INTEGRATED_TEMPERATURES
    )
    assert "no fitted count rate" in support["admission_rule"]["no_outcome_selection"]
    assert support["integrity"]["lensing_target_opened"] is False


def test_runtime_response_support_supplement_is_hashed_and_fail_closed() -> None:
    module = _load_module()
    runtime = json.loads(RUNTIME_SUPPORT.read_text(encoding="utf-8"))
    runtime_report = json.loads(RUNTIME_SUPPORT_REPORT.read_text(encoding="utf-8"))
    spectra = json.loads(INTEGRATED_SPECTRA.read_text(encoding="utf-8"))
    temperatures = json.loads(INTEGRATED_TEMPERATURES.read_text(encoding="utf-8"))

    module.validate_runtime_response_support(
        RUNTIME_SUPPORT,
        runtime,
        RUNTIME_SUPPORT_REPORT,
        runtime_report,
        CONFIG,
        INTEGRATED_SPECTRA,
        spectra,
        INTEGRATED_TEMPERATURES,
        temperatures,
        SUPPORT,
        SUPPORT_REPORT,
    )
    rule = runtime["admission_rule"]
    assert rule["allowed_reason"] == module.EMPTY_CALIBRATED_RESPONSE_REASON
    assert "No other CIAO failure" in rule["failure_mode"]
    assert runtime["integrity"]["lensing_target_opened"] is False
    assert runtime_report["config_sha256"] == _sha256(RUNTIME_SUPPORT)
    diagnostic = ROOT / runtime_report["discovery"]["diagnostic_log"]
    assert runtime_report["discovery"]["diagnostic_log_sha256"] == _sha256(
        diagnostic
    )


def test_response_support_diagnostic_logs_are_immutable() -> None:
    report = json.loads(SUPPORT_REPORT.read_text(encoding="utf-8"))

    assert report["config_sha256"] == _sha256(SUPPORT)
    assert report["regional_extraction_authorized"] is True
    for trial in report["isolated_trials"]:
        log = ROOT / trial["log"]
        assert log.is_file()
        assert trial["log_sha256"] == _sha256(log)
        assert trial["arf_created"] is False
        assert trial["rmf_created"] is False
        assert trial["return_code"] != 0
    assert report["decision"]["temperature_or_fit_outcome_used"] is False
    assert report["decision"]["lensing_target_opened"] is False


def test_regional_runner_matches_frozen_region_gates_and_claim_boundary() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = SCRIPT.read_text(encoding="utf-8")
    assert config["gates"]["regional"]["minimum_passing_regions_per_cluster"] == 12
    assert config["gates"]["regional"]["maximum_reduced_statistic"] == 1.5
    assert config["gates"]["regional"]["maximum_fractional_68_percent_half_width"] == 0.5
    assert '"thermal_stress_constructed": False' in source
    assert '"lensing_target_opened": False' in source
    assert '"regional_temperature_fit_authorized": True' in source
    assert "event_mean_response_reference_maps_off_selected_ccd" in source
    assert "response_support_config_sha256" in source
    assert "runtime_response_support_config_sha256" in source
