import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import apply_sigma_v19cy_a2319_calibration_candidates as application


def test_frozen_application_scope_and_seals_are_exact() -> None:
    config, topology, provenance = application.validate_inputs(application.DEFAULT_CONFIG)
    assert [candidate["name"] for candidate in config["candidates"]] == [
        "fe55_branch_only",
        "branch_median_common_mode",
        "branch_linear_common_mode",
    ]
    assert len(topology["branches"]) == 7
    assert config["terminal_gate"]["required_candidate_branch_outputs"] == 21
    assert not provenance["validation_or_holdout_asset_accessed"]


def test_residual_candidates_are_fully_determined_by_frozen_branch_summary() -> None:
    branch = {
        "start": 100.0,
        "stop": 300.0,
        "calibration_pixel_residual": {
            "median": 2.0,
            "linear_slope_per_hour": 3.0,
        },
    }
    assert application.candidate_residual({"name": "fe55_branch_only"}, branch, 200.0) == 0
    assert (
        application.candidate_residual(
            {"name": "branch_median_common_mode"}, branch, 100.0
        )
        == 2.0
    )
    assert (
        application.candidate_residual(
            {"name": "branch_linear_common_mode"}, branch, 3800.0
        )
        == 5.0
    )


def test_runtime_command_uses_only_calibration_pixel_and_frozen_parameters() -> None:
    source = inspect.getsource(application.ftcopy_command)
    assert "PIXEL==" in source
    assert "ITYPE==" in source
    command_source = inspect.getsource(application.rslpha2pi_command)
    assert "rslpha2pi" in command_source
    assert "cluster" not in command_source.lower()


def test_each_drift_build_copies_cached_hdus_and_matches_extension_case_insensitively() -> None:
    source = inspect.getsource(application.build_drift_file)
    assert "output_hdus = [hdu.copy() for hdu in start_hdus]" in source
    assert "hdu.name.casefold() == extension.casefold()" in source
    assert "start_hdus[table_index] =" not in source


def test_output_audit_counts_nulls_without_summarizing_energy_values() -> None:
    source = inspect.getsource(application.audit_output)
    assert 'data["PI"]' in source
    assert 'data["EPI2"]' in source
    assert 'data["TEMP"]' in source
    assert "negative_epi2" in source
    assert "null_pi_not_explained_by_negative_epi2" in source
    assert "negative_epi2_without_null_pi" in source
    assert "mean(" not in source
    assert "median(" not in source
    assert "quantile(" not in source
    assert "histogram(" not in source
