import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "solar_screened_galaxy_morphology_protocol.json"
OUTPUT = ROOT / "results" / "solar_screened_galaxy_morphology"
REPORT = OUTPUT / "report.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_protocol_was_frozen_and_cluster_parameter_was_not_refit() -> None:
    protocol = _load(PROTOCOL)
    report = _load(REPORT)
    assert protocol["status"] == "fixed before solar-screened galaxy scores"
    assert report["inputs"]["protocol_sha256"] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert report["law"]["lambda"] == 10.5
    assert report["law"]["global_gravity_parameters_refit_to_galaxies"] == 0
    assert report["law"]["per_galaxy_gravity_parameters"] == 0


def test_complete_frozen_sample_and_morphology_bins() -> None:
    report = _load(REPORT)
    assert report["sample"]["galaxies"] == 131
    assert report["sample"]["inner_train_points"] == 2066
    assert report["sample"]["outer_holdout_points"] == 968
    assert report["sample"]["bin_counts"]["stellar_structure"] == {
        "bulge_dominated": 13,
        "disk_dominated": 104,
        "mixed_disk_bulge": 14,
    }


def test_locked_cluster_law_fails_the_galaxy_transfer() -> None:
    report = _load(REPORT)
    scores = report["overall_outer_scores"]
    assert 18.60 < scores["solar_screened_isothermal"]["RMSE_km_s"] < 18.61
    assert 10.34 < scores["fixed_RAR"]["RMSE_km_s"] < 10.36
    assert scores["solar_screened_isothermal"]["RMSE_ratio_to_fixed_RAR"] > 1.79
    assert report["gate_audit"]["passes_all"] is False


def test_disk_and_mass_failure_pattern_is_preserved() -> None:
    report = _load(REPORT)
    rows = {
        (row["dimension"], row["bin"]): row
        for row in report["screened_tail_type_scores"]
    }
    assert rows[("stellar_structure", "disk_dominated")]["RMSE_ratio_to_fixed_RAR"] > 2.0
    assert rows[("stellar_structure", "mixed_disk_bulge")]["RMSE_ratio_to_fixed_RAR"] < 1.14
    assert rows[("baryonic_mass_family", "dwarf_mass")]["RMSE_ratio_to_fixed_RAR"] > 2.6
    scaling = report["analytic_mass_scaling"]
    assert scaling["baryonic_mass_matched_by_selected_lambda_solar"] > 3.0e11


def test_reproducibility_artifacts_exist() -> None:
    for name in (
        "point_predictions.csv",
        "screened_tail_galaxy_fits.csv",
        "morphology_assignments.csv",
        "type_scores.csv",
        "per_galaxy_scores.csv",
        "galaxy_type_assessment.png",
    ):
        assert (OUTPUT / name).stat().st_size > 0
