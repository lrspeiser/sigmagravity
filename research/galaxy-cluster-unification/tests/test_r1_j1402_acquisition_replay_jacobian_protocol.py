import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r1_j1402_acquisition_replay_jacobian_protocol.json"


def load_protocol() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_protocol_was_frozen_before_science_inspection() -> None:
    protocol = load_protocol()
    assert protocol["selection_blind"]
    assert protocol["status"] == "frozen_before_any_J1402_science_array_download_or_inspection"
    assert not protocol["lens_pixels_seen_at_freeze"]
    assert not protocol["KCWI_science_pixels_seen_at_freeze"]
    assert not protocol["gravity_residuals_seen_at_freeze"]
    assert protocol["candidate"]["selection_reason"].startswith("first residual-blind")


def test_every_download_identity_is_exactly_locked() -> None:
    acquisition = load_protocol()["acquisition"]
    github = acquisition["Dinos_GitHub"]
    assert len(github["commit"]) == 40
    assert len(github["files"]) == 9
    assert all(item["bytes"] > 0 for item in github["files"])
    assert all(len(item["git_blob_sha1"]) == 40 for item in github["files"])
    assert acquisition["Dinos_full_output"]["file_id"] == "1BuAmGW5adsypaIbBrv5z3_Zdf96CXfm7"

    kcwi = acquisition["KCWI"]
    assert len(kcwi["science_ids"]) == 4
    assert len(kcwi["bias_ids"]) == 7
    assert len(kcwi["continuum_bar_ids"]) == 1
    assert len(kcwi["arc_ids"]) == 2
    assert len(kcwi["flat_ids"]) == 17
    assert len(kcwi["standard_star_ids"]) == 5
    ids = (
        kcwi["science_ids"]
        + kcwi["bias_ids"]
        + kcwi["continuum_bar_ids"]
        + kcwi["arc_ids"]
        + kcwi["flat_ids"]
        + [item["koaid"] for item in kcwi["standard_star_ids"]]
    )
    assert len(ids) == len(set(ids)) == 36
    assert all(item.startswith("KB.20220408.") and item.endswith(".fits") for item in ids)


def test_detector_readout_mismatch_is_explicit_and_fail_closed() -> None:
    compatibility = load_protocol()["acquisition"]["KCWI"]["calibration_compatibility"]
    assert "ALL/1" in compatibility["continuum_bar_and_arc"]
    assert "TUP/0" in compatibility["continuum_bar_and_arc"]
    assert compatibility["fallback"].startswith("if the geometry frames fail, stop")


def test_rank_and_dynamics_gates_cannot_be_weakened_by_published_model() -> None:
    protocol = load_protocol()
    basis = protocol["lens_response_basis"]
    rank = protocol["rank_gate"]
    dynamics = protocol["KCWI_reduction_and_dynamics_gate"]
    auth = protocol["authorization"]

    assert basis["knot_radii_in_Einstein_radii"] == [0.7, 0.9, 1.1, 1.3]
    assert len(basis["knot_radii_arcsec"]) == 4
    assert rank["minimum_whitened_singular_modes"] == 3
    assert rank["minimum_mode_signal_to_noise"] == 3.0
    assert rank["relative_numerical_floor"] == 0.001
    assert rank["maximum_retained_condition_number"] == 1000.0
    assert rank["negative_control_maximum_rank"] == 0
    assert dynamics["minimum_independent_numerical_bins_in_lens_support"] == 3
    assert not auth["use_published_Dinos_mass_slope_as_observable"]
    assert not auth["count_toward_ten_system_target_before_all_joint_gates"]
    assert not auth["infer_gravity_response"]
    assert not auth["authorize_R2"]


def test_all_required_nuisance_and_stability_families_are_present() -> None:
    protocol = load_protocol()
    nuisance = protocol["nuisance_projection"]
    variants = protocol["lens_variant_grid"]
    flat_nuisance = json.dumps(nuisance)
    for token in ["shapelet", "lens centroid", "external shear", "mass-sheet", "astrometry", "PSF"]:
        assert token in flat_nuisance
    assert variants["shapelet_orders"] == [4, 6, 8]
    assert len(variants["band_subsets"]) == 4
    assert len(variants["masks"]) == 3
    assert len(variants["finite_difference_arcsec"]) == 3
    assert variants["synthetic_injections"] == 100
    assert "tangential" in variants["negative_control"]
