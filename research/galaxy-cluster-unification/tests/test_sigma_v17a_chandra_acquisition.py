from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "sigma_v17a_chandra_acquisition.json"
PROVENANCE = ROOT / "results" / "sigma_v17a_chandra_acquisition" / "provenance.json"
REDUCTION_CONFIG = ROOT / "configs" / "sigma_v17a_chandra_reduction.json"
ENVIRONMENT_REPORT = ROOT / "results" / "sigma_v17a_ciao_environment" / "report.json"
REPRO_REPORT = ROOT / "results" / "sigma_v17a_chandra_repro" / "report.json"
CLEANING_REPORT = ROOT / "results" / "sigma_v17a_chandra_cleaning" / "report.json"
GAIA_CONFIG = ROOT / "configs" / "sigma_v17a_gaia_astrometry.json"
GAIA_ACQUISITION = ROOT / "results" / "sigma_v17a_gaia_astrometry_acquisition" / "provenance.json"
DIRECT_ASTROMETRY = ROOT / "results" / "sigma_v17a_chandra_astrometry" / "report.json"
HIERARCHICAL_CONFIG = ROOT / "configs" / "sigma_v17a2_hierarchical_astrometry.json"
HIERARCHICAL_ASTROMETRY = ROOT / "results" / "sigma_v17a2_hierarchical_astrometry" / "report.json"
TEMPERATURE_REGION_CONFIG = ROOT / "configs" / "sigma_v17b_temperature_regions.json"
TEMPERATURE_REGION_REPORT = ROOT / "results" / "sigma_v17b_temperature_regions" / "report.json"

EXPECTED_OBSIDS = {
    "AS295": [12260, 16127, 16282, 16524, 16525, 16526],
    "PLCKG287": [17165, 17166, 17494, 17495, 18807],
}
REQUIRED_ROLES = {"evt1", "evt2", "bpix1", "fov1", "asol1", "msk1", "pbk0", "bias0"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_acquisition_protocol_was_frozen_without_lensing_inputs() -> None:
    config = read_json(CONFIG)

    assert config["protocol_version"] == "SIGMA-V17A-CHANDRA-ACQUISITION-1.0.0"
    assert config["status"] == (
        "frozen before downloading any analysis-grade event or calibration product"
    )
    assert {
        cluster: values["obsids"] for cluster, values in config["clusters"].items()
    } == EXPECTED_OBSIDS
    assert "lensing" not in json.dumps(config["included_products"]).lower()
    assert set(config["required_roles_per_obsid"]) == REQUIRED_ROLES


def test_all_selected_archive_products_are_complete_and_content_addressed() -> None:
    config = read_json(CONFIG)
    provenance = read_json(PROVENANCE)

    assert provenance["status"] == ("analysis_grade_Chandra_archive_products_downloaded_and_hashed")
    assert provenance["protocol_version"] == config["protocol_version"]
    assert provenance["config_sha256"] == sha256(CONFIG)
    assert provenance["files"] == 324
    assert provenance["bytes"] == 912_431_878
    assert len(provenance["records"]) == provenance["files"]
    assert sum(row["bytes"] for row in provenance["records"]) == provenance["bytes"]
    assert provenance["lensing_target_opened"] is False
    assert provenance["temperature_map_constructed"] is False

    actual_obsids: dict[str, list[int]] = {}
    for cluster in EXPECTED_OBSIDS:
        actual_obsids[cluster] = sorted(
            row["obsid"] for row in provenance["per_obsid"] if row["cluster"] == cluster
        )
    assert actual_obsids == EXPECTED_OBSIDS

    records_by_obsid: dict[tuple[str, int], list[dict]] = {}
    for record in provenance["records"]:
        key = (record["cluster"], record["obsid"])
        records_by_obsid.setdefault(key, []).append(record)

        path = ROOT / record["relative_path"]
        assert path.is_file()
        assert path.stat().st_size == record["bytes"]
        assert sha256(path) == record["sha256"]

    for cluster, obsids in EXPECTED_OBSIDS.items():
        for obsid in obsids:
            role_counts = Counter(record["role"] for record in records_by_obsid[(cluster, obsid)])
            assert all(role_counts[role] >= 1 for role in REQUIRED_ROLES)


def test_reduction_protocol_is_common_and_frozen_before_map_construction() -> None:
    reduction = read_json(REDUCTION_CONFIG)

    assert reduction["protocol_version"] == "SIGMA-V17A-CHANDRA-REDUCTION-1.0.2"
    assert reduction["status"] == (
        "frozen before reprocessing, event-image inspection, temperature-region "
        "construction, spectral fitting, or reading a v17 dynamical-feature score"
    )
    assert {
        cluster: values["obsids"] for cluster, values in reduction["clusters"].items()
    } == EXPECTED_OBSIDS
    assert reduction["common_map"]["analysis_radius_kpc"] == 350
    assert reduction["common_map"]["target_signal_to_noise"] == 40
    assert reduction["common_map"]["minimum_net_counts_per_region"] == 1300
    assert reduction["common_map"]["minimum_valid_regions_inside_350_kpc"] == 12
    assert reduction["background"]["normalization_energy_keV"] == [9.0, 12.0]
    assert reduction["spectral_model"]["model"] == "xstbabs * xsapec"
    assert reduction["thermal_source"]["mu"] == 0.61
    assert len(reduction["thermal_source"]["forbidden"]) == 5


def test_ciao_runtime_and_official_smoke_suite_pass_the_frozen_gate() -> None:
    reduction = read_json(REDUCTION_CONFIG)
    report = read_json(ENVIRONMENT_REPORT)
    smoke_log = ROOT / report["smoke"]["log"]

    assert report["protocol_version"] == reduction["protocol_version"]
    assert report["config_sha256"] == sha256(REDUCTION_CONFIG)
    assert report["smoke"]["run"] == 37
    assert report["smoke"]["passed"] == 37
    assert report["smoke"]["failed"] == 0
    assert report["smoke"]["skipped"] == 0
    assert report["smoke"]["log_sha256"] == sha256(smoke_log)
    assert report["background_inventory"]["files"] == 134
    assert report["background_inventory"]["bytes"] == 3_086_141_760
    assert all(report["gates"].values())
    assert report["lensing_target_opened"] is False
    assert report["temperature_map_constructed"] is False


def test_common_flare_and_blanksky_cleaning_completed_for_every_observation() -> None:
    reduction = read_json(REDUCTION_CONFIG)
    repro = read_json(REPRO_REPORT)
    report = read_json(CLEANING_REPORT)

    assert report["status"] == ("all_frozen_chandra_observations_flare_cleaned_with_blanksky")
    assert report["protocol_version"] == reduction["protocol_version"]
    assert report["config_sha256"] == sha256(REDUCTION_CONFIG)
    assert report["repro_report_sha256"] == sha256(REPRO_REPORT)
    assert report["observation_count"] == 11
    assert report["clean_exposure_seconds"] == 400_362.3623303224
    assert report["minimum_retained_exposure_fraction"] == 0.9833594028458849
    assert sum(row["point_sources"]["wavdetect_sources"] for row in report["observations"]) == 672
    assert sum(row["point_sources"]["selected_sources"] for row in report["observations"]) == 602
    assert sum(row["original_event_rows"] for row in report["observations"]) == 1_171_047
    assert sum(row["clean_event_rows"] for row in report["observations"]) == 1_165_873
    assert sum(row["blanksky_event_rows"] for row in report["observations"]) == 12_475_717

    for row in report["observations"]:
        assert (
            row["retained_exposure_fraction"]
            >= reduction["flare_filtering"]["minimum_retained_fraction"]
        )
        assert row["point_sources"]["minimum_significance"] == 3.0
        assert row["point_sources"]["ellipse_expansion"] == 1.5
        assert row["blanksky_scaling"]
        assert all(float(value) > 0 for value in row["blanksky_scaling"].values())
        assert "method=sigma" in row["steps"]["deflare"]["command"]
        assert f"random={row['obsid']}" in row["steps"]["blanksky"]["command"]

    assert repro["temperature_map_constructed"] is False
    assert report["astrometry_completed"] is False
    assert report["event_images_visually_inspected"] is False
    assert report["lensing_target_opened"] is False
    assert report["temperature_map_constructed"] is False


def test_all_frozen_observations_were_reprocessed_with_current_caldb() -> None:
    reduction = read_json(REDUCTION_CONFIG)
    acquisition = read_json(PROVENANCE)
    report = read_json(REPRO_REPORT)

    assert report["status"] == "all_frozen_chandra_observations_reprocessed"
    assert report["protocol_version"] == reduction["protocol_version"]
    assert report["config_sha256"] == sha256(REDUCTION_CONFIG)
    assert report["acquisition_provenance_sha256"] == sha256(PROVENANCE)
    assert report["observation_count"] == 11
    assert report["product_files"] == 132
    assert report["product_bytes"] == 466_059_852
    assert sum(row["staging"]["files"] for row in report["observations"]) == 324
    assert sum(row["event"]["event_rows"] for row in report["observations"]) == 1_171_047

    actual_obsids: dict[str, list[int]] = {}
    for cluster in EXPECTED_OBSIDS:
        actual_obsids[cluster] = sorted(
            row["obsid"] for row in report["observations"] if row["cluster"] == cluster
        )
    assert actual_obsids == EXPECTED_OBSIDS

    for row in report["observations"]:
        assert row["event"]["event_rows"] > 0
        assert row["event"]["header"]["DATAMODE"] == "VFAINT"
        assert row["event"]["header"]["PIX_ADJ"] == "EDSER"
        assert row["event"]["current_caldb_comment_present"] is True
        assert "pix_adj=edser" in row["command"]
        assert "mode=h" in row["command"]
        assert len(row["log_sha256"]) == 64
        assert row["staging"]["root_products_promoted_to_standard_obsid_layout"]

    assert acquisition["temperature_map_constructed"] is False
    assert report["lensing_target_opened"] is False
    assert report["event_images_inspected"] is False
    assert report["temperature_map_constructed"] is False


def test_corrected_gaia_reference_cones_are_frozen_and_content_addressed() -> None:
    config = read_json(GAIA_CONFIG)
    report = read_json(GAIA_ACQUISITION)

    assert config["protocol_version"] == "SIGMA-V17A-GAIA-ASTROMETRY-1.0.1"
    assert config["metadata_correction"]["wrong_center_ra_deg"] == 162.70583333333335
    assert config["metadata_correction"]["wrong_center_dec_deg"] == 28.07897222222222
    assert config["metadata_correction"]["corrected_center_ra_deg"] == 177.70583333333335
    assert config["metadata_correction"]["corrected_center_dec_deg"] == -28.07897222222222
    assert config["metadata_correction"]["matching_rules_changed"] is False
    assert report["status"] == "frozen_Gaia_DR3_reference_cones_downloaded_and_hashed"
    assert report["protocol_version"] == config["protocol_version"]
    assert report["config_sha256"] == sha256(GAIA_CONFIG)
    assert report["files"] == 2
    assert report["rows"] == 6_742
    assert report["bytes"] == 1_322_600
    assert {row["cluster"]: row["rows"] for row in report["records"]} == {
        "AS295": 2_127,
        "PLCKG287": 4_615,
    }
    for row in report["records"]:
        path = ROOT / row["relative_path"]
        assert path.stat().st_size == row["bytes"]
        assert sha256(path) == row["sha256"]
    wrong_field = ROOT / config["metadata_correction"]["wrong_field_catalog_preserved_as"]
    assert wrong_field.stat().st_size == 250_531
    assert sha256(wrong_field) == (
        "37087af91a9ecde94fbe1605dd2d11c99d5ab5c3156730570aeea9d1d5940b27"
    )
    assert report["xray_source_crossmatch_run"] is False
    assert report["temperature_map_constructed"] is False


def test_direct_per_observation_gaia_gate_failed_without_applying_transforms() -> None:
    report = read_json(DIRECT_ASTROMETRY)

    assert report["status"] == "frozen_Gaia_DR3_astrometric_gate_failed"
    assert report["all_absolute_gates_passed"] is False
    assert report["observation_count"] == 11
    assert {row["obsid"] for row in report["failed_observations"]} == {
        16282,
        17166,
        17495,
        18807,
    }
    assert all(
        row["gates"]["minimum_final_source_pairs"] is False for row in report["failed_observations"]
    )
    assert all("application" not in row for row in report["observations"])
    assert report["registered_event_images_inspected"] is False
    assert report["lensing_target_opened"] is False
    assert report["temperature_map_constructed"] is False


def test_hierarchical_astrometry_passes_every_frozen_gate_and_updates_products() -> None:
    config = read_json(HIERARCHICAL_CONFIG)
    parent = read_json(DIRECT_ASTROMETRY)
    report = read_json(HIERARCHICAL_ASTROMETRY)

    assert config["protocol_version"] == "SIGMA-V17A2-HIERARCHICAL-ASTROMETRY-1.0.0"
    assert config["parent_direct_report_sha256"] == sha256(DIRECT_ASTROMETRY)
    assert parent["status"] == config["parent_failure"]["formal_status"]
    assert report["status"] == ("all_frozen_observations_hierarchically_registered_to_Gaia_DR3")
    assert report["config_sha256"] == sha256(HIERARCHICAL_CONFIG)
    assert report["parent_direct_report_sha256"] == sha256(DIRECT_ASTROMETRY)
    assert report["observation_count"] == 11
    assert report["reference_obsids"] == {"AS295": 16524, "PLCKG287": 17494}
    assert report["failed_observations"] == []
    assert report["all_hierarchical_gates_passed"] is True
    assert report["transforms_applied"] is True

    assert min(row["match_statistics"]["included_pairs"] for row in report["observations"]) == 4
    assert (
        max(
            row["match_statistics"]["included_rms_recomputed_arcsec"]
            for row in report["observations"]
        )
        == 0.36493835095807625
    )
    assert sum(row["match_statistics"]["included_pairs"] for row in report["observations"]) == 70
    assert {row["stage"] for row in report["observations"]} == {
        "Gaia_anchor",
        "relative_to_Gaia_anchor",
    }

    for row in report["observations"]:
        assert all(row["gates"].values())
        matrix = row["transform_values"]
        assert matrix["a11"] == 1.0
        assert matrix["a12"] == 0.0
        assert matrix["a21"] == 0.0
        assert matrix["a22"] == 1.0
        assert row["application"]["corrected_aspects"]
        for event in row["application"]["corrected_events"].values():
            assert event["asolfile_header"].endswith("_gaia_asol1.fits")
            assert len(event["sha256"]) == 64

    assert report["registered_event_images_inspected"] is False
    assert report["lensing_target_opened"] is False
    assert report["temperature_map_constructed"] is False


def test_temperature_region_protocol_preserves_the_frozen_scientific_rules() -> None:
    config = read_json(TEMPERATURE_REGION_CONFIG)

    assert config["protocol_version"] == "SIGMA-V17B-TEMPERATURE-REGIONS-1.0.2"
    assert config["coordinates"]["analysis_radius_kpc"] == 350
    assert config["science_image"]["minimum_relative_exposure"] == 0.5
    assert config["contour_binning"]["target_signal_to_noise"] == 40
    assert config["contour_binning"]["smoothing_signal_to_noise"] == 20
    assert config["contour_binning"]["geometric_constraint_factor"] == 1.5
    assert config["contour_binning"]["minimum_net_counts_per_region"] == 1300
    assert config["contour_binning"]["minimum_source_fraction"] == 0.8
    assert config["contour_binning"]["minimum_valid_regions"] == 12
    assert config["integrity"]["merged_images_visually_inspected_at_freeze"] is False
    assert config["integrity"]["contour_region_constructed_at_freeze"] is False
    assert config["integrity"]["temperature_map_constructed_at_freeze"] is False
    assert config["integrity"]["lensing_target_opened"] is False
    assert config["integrity"]["physics_or_temperature_parameter_changed"] is False


def test_both_clusters_pass_the_frozen_temperature_region_gate() -> None:
    config = read_json(TEMPERATURE_REGION_CONFIG)
    report = read_json(TEMPERATURE_REGION_REPORT)

    assert report["status"] == "both_clusters_passed_frozen_temperature_region_gate"
    assert report["protocol_version"] == config["protocol_version"]
    assert report["config_sha256"] == sha256(TEMPERATURE_REGION_CONFIG)
    assert report["reduction_config_sha256"] == sha256(REDUCTION_CONFIG)
    assert report["astrometry_report_sha256"] == sha256(HIERARCHICAL_ASTROMETRY)
    assert report["cleaning_report_sha256"] == sha256(CLEANING_REPORT)
    assert report["failed_clusters"] == []

    expected = {
        "AS295": {
            "regions": 29,
            "snapshot_files": 39,
            "snapshot_bytes": 2_205_539,
            "minimum_net_counts": 1626.039737431,
            "minimum_signal_to_noise": 40.001656069456786,
            "minimum_source_fraction": 0.9316074973501804,
        },
        "PLCKG287": {
            "regions": 21,
            "snapshot_files": 31,
            "snapshot_bytes": 1_856_818,
            "minimum_net_counts": 1645.626808764,
            "minimum_signal_to_noise": 40.00787640102009,
            "minimum_source_fraction": 0.9347620859190523,
        },
    }
    assert {row["cluster"] for row in report["clusters"]} == set(expected)
    for row in report["clusters"]:
        target = expected[row["cluster"]]
        assert row["region_count"] == target["regions"]
        assert row["valid_region_count"] == target["regions"]
        assert row["minimum_net_counts"] == target["minimum_net_counts"]
        assert row["minimum_signal_to_noise"] == target["minimum_signal_to_noise"]
        assert row["minimum_source_fraction"] == target["minimum_source_fraction"]
        assert all(row["gates"].values())
        assert row["frozen_snapshot"]["files"] == target["snapshot_files"]
        assert row["frozen_snapshot"]["bytes"] == target["snapshot_bytes"]
        for product in row["frozen_snapshot"]["products"]:
            path = ROOT / product["relative_path"]
            assert path.is_file()
            assert path.stat().st_size == product["bytes"]
            assert sha256(path) == product["sha256"]

    assert report["event_images_visually_inspected"] is False
    assert report["temperature_map_constructed"] is False
    assert report["lensing_target_opened"] is False
