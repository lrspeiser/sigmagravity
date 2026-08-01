import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "solar_screened_isothermal_protocol.json"
REPORT = ROOT / "results" / "solar_screened_isothermal" / "report.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_protocol_and_single_parameter_accounting() -> None:
    protocol = _load(PROTOCOL)
    report = _load(REPORT)
    assert protocol["status"] == "frozen_before_screened_cluster_and_solar_scores"
    assert report["protocol"]["sha256"] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert report["law"]["universal_fitted_gravity_parameters"] == 1
    assert report["law"]["per_object_gravity_or_lensing_amplitudes"] == 0
    assert report["law"]["selected_parameter"] == 10.5


def test_mercury_is_inside_the_published_margin_without_relaxing_it() -> None:
    report = _load(REPORT)
    gates = report["gate_audit"]
    assert gates["Mercury_allowed_absolute_margin_mas_per_century"] == 3.1
    assert abs(gates["Mercury_prediction_mas_per_century"]) < 3.1
    assert gates["Mercury_within_published_one_sigma_margin"] is True
    assert abs(report["solar_system"]["Mercury_unscreened_control_mas_per_century"]) > 3.1
    assert gates["Cassini_fractional_force_proxy_pass"] is True


def test_screened_cluster_holdout_beats_the_limited_compact_halo_aggregate() -> None:
    report = _load(REPORT)
    validation = report["validation"]["aggregate"]
    halo = report["comparators"]["GR_plus_compact_cluster_halo"]
    assert validation["all_roots_converged"] is True
    assert validation["equal_system_radial_RMS_arcsec"] == 5.260607271713773
    assert validation["equal_system_radial_RMS_arcsec"] < halo["equal_system_radial_RMS_arcsec"]
    assert validation["pooled_coordinate_chi2"] < halo["pooled_coordinate_chi2"]
    assert report["gate_audit"]["all_advance_gates_pass"] is True


def test_per_cluster_and_stress_caveats_are_preserved() -> None:
    report = _load(REPORT)
    validation = report["validation"]["per_system"]
    assert validation["MACS1115"]["heldout"]["exact_radial_RMS_arcsec"] < validation["MACS1115"]["comparators"]["GR_plus_cluster_halo"]["heldout"]["exact_radial_RMS_arcsec"]
    assert validation["MACS1931"]["heldout"]["exact_radial_RMS_arcsec"] > validation["MACS1931"]["comparators"]["GR_plus_cluster_halo"]["heldout"]["exact_radial_RMS_arcsec"]
    diagnostics = report["post_result_diagnostics"]
    assert diagnostics["prior_program_absolute_2_arcsec_gate_pass"] is False
    assert diagnostics["RXJ2129_heldout_RMS_ratio_to_compact_halo"] > 6.0
    assert report["verdict"]["raw_ephemeris_validation_completed"] is False


def test_reproducibility_artifacts_exist() -> None:
    output = REPORT.parent
    for name in (
        "coarse_grid.csv",
        "refined_grid.csv",
        "predictions.csv",
        "geometry.csv",
        "radial_profiles.csv",
        "solar_diagnostics.csv",
        "SUMMARY.md",
    ):
        assert (output / name).stat().st_size > 0
