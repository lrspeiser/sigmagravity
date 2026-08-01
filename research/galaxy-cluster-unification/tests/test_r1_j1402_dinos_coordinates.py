import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INITIAL_REPORT = ROOT / "results" / "r1_j1402_dinos_coordinate_audit" / "report.json"
REPORT = ROOT / "results" / "r1_j1402_dinos_coordinate_audit_corrected" / "report.json"


def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_no_fit_structural_coordinate_gate_passes() -> None:
    item = report()
    assert not item["fit_performed"]
    assert not item["forward_model_evaluated"]
    assert all(item["checks"].values())
    assert item["gate_pass"]
    assert item["decision"] == "corrected_structural_coordinate_gate_pass_authorize_locked_environment_install"


def test_initial_failed_audit_is_preserved_and_explained() -> None:
    initial = json.loads(INITIAL_REPORT.read_text(encoding="utf-8"))
    assert not initial["gate_pass"]
    assert not initial["checks"]["released_mask_matches_settings_bitwise"]
    assert not initial["checks"]["all_PSFs_finite_nonnegative_normalized_61_square"]
    item = report()
    assert item["initial_failed_report"] == "results/r1_j1402_dinos_coordinate_audit/report.json"
    assert len(item["corrections"]["unchanged_scientific_gates"]) == 6


def test_every_band_preserves_its_exact_coordinate_dictionary() -> None:
    rows = report()["band_coordinates"]
    assert [item["band"] for item in rows] == ["F435W", "F555W", "F814W"]
    assert [item["image_shape"] for item in rows] == [[120, 120], [140, 140], [140, 140]]
    assert all(item["transform_matches_settings_bitwise"] for item in rows)
    assert all(item["ra_origin_matches_settings_bitwise"] for item in rows)
    assert all(item["roundtrip_maximum_pixel_error"] <= 1e-10 for item in rows)
    assert all(0.049 < item["pixel_scale_arcsec"] < 0.051 for item in rows)
    assert all(item["stored_shift_is_negative_settings_offset"] for item in rows)


def test_mask_psfs_and_chain_are_exactly_structurally_usable() -> None:
    item = report()
    assert not item["mask"]["bitwise_equal"]
    assert item["mask"]["bitwise_equal_to_complement"]
    assert item["mask"]["complement_disagreeing_pixels"] == 0
    assert all(
        psf["finite"] and psf["nonnegative"] and psf["sum"] > 0
        for psf in item["PSFs"]
    )
    assert item["chain"]["samples_shape"] == [1_104_000, 23]
    assert item["chain"]["best_sample_index"] == 1_101_277
    assert item["chain"]["maximum_log_likelihood"] == -200012404.1189446
    assert item["checks"]["external_numpy_pickle_was_not_loaded"]


def test_scalar_pixel_size_is_not_silently_used() -> None:
    discrepancy = report()["scalar_pixel_size_discrepancy"]
    assert discrepancy["settings_scalar_arcsec"] == 0.04
    assert all(0.049 < value < 0.051 for value in discrepancy["operative_matrix_scales_arcsec"])
    assert "not substituted" in discrepancy["resolution"]
    assert not report()["authorization"]["evaluate_forward_model"]
    assert not report()["authorization"]["compute_lens_response"]
