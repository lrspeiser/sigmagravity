import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17c_spectral_temperature.json"
RUNNER = ROOT / "scripts" / "run_sigma_v17c_integrated_spectra.py"
FITTER = ROOT / "scripts" / "fit_sigma_v17c_integrated_temperatures.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("sigma_v17c_integrated", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fitter():
    spec = importlib.util.spec_from_file_location("sigma_v17c_fit", FITTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v107_freeze_preserves_science_and_bounds_external_concurrency() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["protocol_version"] == "SIGMA-V17C-SPECTRAL-TEMPERATURE-1.0.7"
    assert config["execution"]["work_namespace"] == "spectral_v17c_v107"
    assert config["execution"]["external_parallel_cells"] == 4
    assert config["extraction"]["parallel_inside_specextract"] is False
    assert config["extraction"]["source_grouping_during_extraction"] == "NONE"
    assert config["extraction"]["background_grouping_during_extraction"] == "NONE"
    assert config["integrity"]["temperature_or_abundance_fit_at_freeze"] is False
    assert config["integrity"]["lensing_target_opened"] is False
    assert config["integrity"]["physics_parameter_changed"] is False


def test_every_frozen_parent_hash_is_current() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    parents = {
        "reduction_config_sha256": ROOT / "configs" / "sigma_v17a_chandra_reduction.json",
        "temperature_region_report_sha256": (
            ROOT / "results" / "sigma_v17b_temperature_regions" / "report.json"
        ),
        "spatial_visual_audit_sha256": (
            ROOT
            / "results"
            / "sigma_v17b_temperature_regions"
            / "audit"
            / "visual_audit.json"
        ),
        "hi4pi_provenance_sha256": (
            ROOT / "results" / "sigma_v17c_hi4pi_acquisition" / "provenance.json"
        ),
        "response_commissioning_restoration_sha256": (
            ROOT / "results" / "sigma_v17c_response_commissioning" / "restoration.json"
        ),
    }
    for key, path in parents.items():
        assert path.is_file()
        assert config["parents"][key] == _sha256(path)


def test_each_parallel_cell_receives_private_ciao_state(tmp_path, monkeypatch) -> None:
    runner = _load_runner()
    captured: list[tuple[Path, Path]] = []

    def fake_isolated_environment(base, pfiles, temporary):
        captured.append((pfiles, temporary))
        return {"PFILES": str(pfiles), "TMPDIR": str(temporary)}

    monkeypatch.setattr(runner, "isolated_environment", fake_isolated_environment)
    monkeypatch.setattr(
        runner,
        "run_step",
        lambda command, log, expected, env: {
            "command": command,
            "reused": False,
            "log": str(log),
        },
    )
    monkeypatch.setattr(
        runner,
        "verify_blanksky_scaling",
        lambda source, background, scale, env: {"BKGSCALn": scale},
    )

    products = {}
    for name in ("source_pha", "background_pha", "arf", "rmf"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        products[name] = path
    task = {
        "cluster": "AS295",
        "obsid": 16127,
        "ccd_id": 2,
        "source_band_events": 100,
        "background_band_events": 200,
        "response_reference": {"ra_deg": 1.0, "dec_deg": 2.0},
        **products,
        "bkgscale_value": 1.25,
        "translated_fov": {"translated": "fov.fits"},
        "command": ["specextract", "parallel=no", "nproc=1"],
        "log": tmp_path / "specextract.log",
    }

    result = runner.execute_extraction_cell(task, tmp_path, "spectral_v17c_v107")

    assert result["obsid"] == 16127
    assert result["ccd_id"] == 2
    assert len(captured) == 1
    pfiles, temporary = captured[0]
    assert pfiles != temporary
    assert "AS295" in pfiles.parts
    assert "16127_ccd2" in pfiles.parts
    assert "AS295" in temporary.parts
    assert "16127_ccd2" in temporary.parts


def test_runner_keeps_internal_specextract_serial_and_results_ordered() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"parallel=no"' in source
    assert '"nproc=1"' in source
    assert "ThreadPoolExecutor" in source
    assert "extracted = [future.result() for future in futures]" in source


def test_blanksky_areascal_makes_effective_scale_equal_particle_count_ratio() -> None:
    runner = _load_runner()
    source_exposure = 44611.534044523
    background_exposure = 600000.0
    source_backscal = 0.0011035720258951
    background_backscal = source_backscal
    source_areascal = 1.0
    bkgscale = 0.071173213

    background_areascal = runner.required_background_areascal(
        source_exposure,
        background_exposure,
        source_backscal,
        background_backscal,
        source_areascal,
        bkgscale,
    )
    effective = (
        source_exposure
        / background_exposure
        * source_backscal
        / background_backscal
        * source_areascal
        / background_areascal
    )
    assert abs(effective / bkgscale - 1.0) < 1.0e-12
    assert abs(background_areascal - 1.0 / bkgscale) > 1.0
    source = RUNNER.read_text(encoding="utf-8")
    assert '"key=AREASCAL"' in source
    assert "effective_scale_relative_error_from_BKGSCALn" in source


def _fits_header(cards: list[str]) -> bytes:
    raw = b"".join(card.ljust(80).encode("ascii") for card in [*cards, "END"])
    return raw + b" " * ((-len(raw)) % 2880)


def test_single_pass_fits_header_reader_finds_spectrum_values(tmp_path: Path) -> None:
    runner = _load_runner()
    primary = _fits_header(
        ["SIMPLE  =                    T", "BITPIX  =                    8", "NAXIS   =                    0"]
    )
    spectrum = _fits_header(
        [
            "XTENSION= 'BINTABLE'",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                    1",
            "NAXIS2  =                    1",
            "PCOUNT  =                    0",
            "GCOUNT  =                    1",
            "EXTNAME = 'SPECTRUM'",
            "EXPOSURE=        44611.534044523",
            "BACKSCAL= 0.0011035720258951",
            "AREASCAL=                  1.0",
        ]
    )
    payload = b"\0" + b"\0" * 2879
    path = tmp_path / "spectrum.pi"
    path.write_bytes(primary + spectrum + payload)

    values = runner.fits_numeric_header_values(
        path, ("EXPOSURE", "BACKSCAL", "AREASCAL")
    )
    assert values == {
        "EXPOSURE": 44611.534044523,
        "BACKSCAL": 0.0011035720258951,
        "AREASCAL": 1.0,
    }


def test_integrated_fitter_implements_the_frozen_model_and_failure_gate() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = FITTER.read_text(encoding="utf-8")
    fitter = _load_fitter()

    assert config["model"]["expression"] == "xstbabs * xsapec"
    assert config["model"]["abundance_table"] == "aspl (Asplund et al. 2009)"
    assert 'ui.create_model_component("xstbabs"' in source
    assert 'ui.create_model_component("xsapec"' in source
    assert "ui.subtract(1)" in source
    assert "ui.thaw(thermal.Abundanc)" in source
    assert 'ui.set_conf_opt("sigma", 1.0)' in source
    assert "regional_fit_authorized" in source
    assert "raise SystemExit(2)" in source
    assert fitter.finite_number(1.0) is True
    assert fitter.finite_number(float("nan")) is False


def test_integrated_fit_exception_is_retained_as_a_failed_cluster() -> None:
    fitter = _load_fitter()
    row = fitter.failed_cluster_result({"cluster": "AS295"}, RuntimeError("boom"))

    assert row["cluster"] == "AS295"
    assert row["fit_completed"] is False
    assert row["parameters"]["temperature_keV"] is None
    assert row["gates"]["all_passed"] is False
    assert "RuntimeError: boom" in row["fit_exception"]
