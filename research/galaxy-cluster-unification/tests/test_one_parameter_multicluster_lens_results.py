import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "one_parameter_multicluster_lens_protocol.json"
REPORT = ROOT / "results" / "one_parameter_multicluster_lens" / "report.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_was_frozen_and_has_one_shared_gravity_parameter() -> None:
    protocol = _load(PROTOCOL)
    report = _load(REPORT)
    assert protocol["status"] == "frozen_before_one_parameter_scores"
    assert report["protocol"]["sha256"] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    accounting = report["parameter_accounting"]
    assert accounting["universal_fitted_gravity_parameters"] == 1
    assert accounting["per_cluster_gravity_or_lensing_amplitudes"] == 0


def test_locked_validation_result_and_frozen_gates_are_recorded() -> None:
    report = _load(REPORT)
    assert report["selection"]["selected_family"] == "mass_isothermal_tail"
    assert report["selection"]["selected_parameter"] == 9.0
    validation = report["validation"]["selected_law"]
    assert validation["systems"] == 2
    assert validation["images"] == 6
    assert validation["all_roots_converged"] is True
    assert abs(validation["equal_system_radial_RMS_arcsec"] - 9.42306727956659) < 1e-9
    gates = report["gate_audit"]
    assert gates["compact_halo_ratio_pass"] is True
    assert gates["both_validation_clusters_improve_over_baryons"] is True
    assert gates["selected_parameter_interior_pass"] is True
    assert gates["validation_absolute_RMS_pass"] is False
    assert report["verdict"]["one_parameter_law_survives"] is False


def test_compact_halo_comparison_is_not_overstated() -> None:
    report = _load(REPORT)
    selected = report["validation"]["per_system"]
    base = _load(ROOT / "results" / "unbounded_running_multicluster_raw" / "report.json")
    m1115 = base["system_scores"]["MACS J1115.9+0129"]["GR_plus_cluster_halo"]
    m1931 = base["system_scores"]["MACS J1931.8-2635"]["GR_plus_cluster_halo"]
    assert selected["MACS1115"]["heldout"]["exact_radial_RMS_arcsec"] < m1115["heldout"]["exact_radial_RMS_arcsec"]
    assert selected["MACS1931"]["heldout"]["exact_radial_RMS_arcsec"] > m1931["heldout"]["exact_radial_RMS_arcsec"]
    assert report["validation"]["selected_law"]["pooled_coordinate_chi2"] > report["comparators"]["GR_plus_compact_cluster_halo"]["pooled_coordinate_chi2"]


def test_predeclared_stress_clusters_were_run_after_lock() -> None:
    report = _load(REPORT)
    stress = report["stress_tests"]
    assert set(stress["per_system"]) == {"RXJ1347", "RXJ2129"}
    assert stress["per_system"]["RXJ1347"]["heldout"]["status"] == "no within-family holdout"
    rxj = stress["per_system"]["RXJ2129"]
    assert rxj["heldout"]["all_roots_converged"] is True
    assert rxj["heldout"]["exact_radial_RMS_arcsec"] > rxj["comparators"]["GR_plus_cluster_halo"]["heldout"]["exact_radial_RMS_arcsec"]
    assert rxj["training"]["all_roots_converged"] is False


def test_expected_artifacts_exist() -> None:
    output = REPORT.parent
    for name in (
        "coarse_grid.csv",
        "refined_grid.csv",
        "predictions.csv",
        "geometry.csv",
        "radial_profiles.csv",
        "one_parameter_multicluster_lens.png",
        "stress_predictions.csv",
        "stress_geometry.csv",
        "stress_radial_profiles.csv",
    ):
        assert (output / name).stat().st_size > 0
