import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_sigma_v19at_decam_forced_photometry_validation.py"
SPEC = importlib.util.spec_from_file_location("sigma_v19at", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_v19at_is_one_frozen_validation_rule():
    config = json.loads(
        (ROOT / "configs" / "sigma_v19at_decam_forced_photometry_validation.json").read_text()
    )
    assert config["frozen_measurement"]["variant"] == "area_scaled"
    assert config["frozen_measurement"]["aperture_diameter_arcsec"] == 4.0
    assert config["gates"]["exact_validation_measurements"] == 362
    assert config["gates"]["exact_validation_image_groups"] == 110
    assert not config["authorization"]["inspect_alternative_validation_variant_or_aperture"]
    assert not config["color_model"]["validation_values_used_to_fit_or_calibrate"]


def test_v19at_split_and_member57_gate_are_explicit():
    config = json.loads(
        (ROOT / "configs" / "sigma_v19at_decam_forced_photometry_validation.json").read_text()
    )
    assert len(config["split"]["development_ids"]) == 10
    assert len(config["split"]["validation_ids"]) == 5
    assert set(config["split"]["development_ids"]).isdisjoint(config["split"]["validation_ids"])
    assert "57" in config["split"]["validation_ids"]
    assert config["validation_gates"]["member57_must_have_complete_griz"]


def test_development_color_model_uses_only_frozen_development_aggregates():
    config = json.loads(
        (ROOT / "configs" / "sigma_v19at_decam_forced_photometry_validation.json").read_text()
    )
    aggregates = MODULE.read_csv(ROOT / config["inputs"]["development_aggregates"])
    sample = MODULE.read_csv(ROOT / config["inputs"]["commissioning_sample"])
    bri = {row["object_id"]: row for row in sample}
    parameters, scales, fit_rows = MODULE.fit_development_color_model(config, aggregates, bri)
    assert set(parameters) == {"g_minus_r", "r_minus_i", "i_minus_z"}
    assert all(scale >= 0.05 for scale in scales.values())
    assert {row["member_id"] for row in fit_rows} == set(config["split"]["development_ids"])
    assert {row["member_id"] for row in fit_rows}.isdisjoint(config["split"]["validation_ids"])
