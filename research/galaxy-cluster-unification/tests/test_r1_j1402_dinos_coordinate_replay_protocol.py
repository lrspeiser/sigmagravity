import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r1_j1402_dinos_coordinate_replay_protocol.json"


def protocol() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_replay_protocol_is_honest_about_the_environment_boundary() -> None:
    item = protocol()
    assert item["status"].endswith("before_software_install_or_forward_model_evaluation")
    assert not item["reproducibility_boundary"]["exact_original_environment_known"]
    assert "not a claim" in item["reproducibility_boundary"]["consequence"]
    assert not item["science_fit_seen_at_freeze"]
    assert not item["lens_response_seen_at_freeze"]


def test_date_bounded_primary_software_is_exactly_locked() -> None:
    software = protocol()["software_lock"]
    assert software["python"] == "3.10.x"
    assert software["Dolphin"]["tag"] == "v0.0.1"
    assert software["Dolphin"]["commit"] == "1593c573541d26ae5791835430c68858988a969b"
    assert software["critical_packages"]["lenstronomy"] == "1.11.5"
    assert software["critical_packages"]["numpy"] == "1.26.4"
    assert software["modern_version_control"]["role"].startswith("secondary")


def test_operational_coordinates_are_not_replaced_by_scalar_pixel_size() -> None:
    coordinates = protocol()["coordinate_contract"]
    assert coordinates["source_of_authority"].endswith("image HDF5 dictionary")
    assert coordinates["scalar_pixel_size_field"] == 0.04
    assert coordinates["scalar_field_treatment"] == "record but never substitute for transform_pix2angle"
    assert coordinates["band_order"] == ["F435W", "F555W", "F814W"]
    assert coordinates["image_sizes_pixels"] == [120, 140, 140]
    assert "round-trips" in " ".join(coordinates["required_checks"])


def test_chain_and_forward_replay_gates_are_concrete() -> None:
    item = protocol()
    chain = item["chain_contract"]
    forward = item["forward_replay_gate"]
    auth = item["authorization"]
    assert chain["samples_shape"] == [1_104_000, 23]
    assert chain["walkers"] * chain["steps"] == chain["samples_shape"][0]
    assert chain["best_sample_index"] == 1_101_277
    assert forward["likelihood_tolerance"]["maximum_absolute_delta_per_used_pixel"] == 0.01
    assert len(forward["coordinate_controls"]) == 4
    assert "each coordinate corruption" in forward["negative_control_requirement"]
    assert not auth["load_external_numpy_pickle"]
    assert not auth["optimize_new_nonlinear_lens_model"]
    assert not auth["compute_lens_response_before_replay_pass"]
    assert not auth["infer_gravity_response"]
