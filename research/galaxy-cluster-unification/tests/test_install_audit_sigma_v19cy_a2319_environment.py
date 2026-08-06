import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import install_audit_sigma_v19cy_a2319_environment as environment


def test_frozen_environment_inputs_match_terminal_acquisition() -> None:
    config, provenance = environment.validate_inputs(environment.DEFAULT_CONFIG)
    archives = environment.validate_archives(config, provenance)
    assert len(archives) == 3
    assert sum(item["bytes"] for item in archives) == 1_780_998_985
    assert config["runtime"]["heasoft_version_token"] == "V6.36"
    assert config["runtime"]["xspec_version_token"] == "12.15.1"
    assert config["closed_failure_history"]["caldb_archive_listed_or_extracted"] is False
    assert config["closed_failure_history"]["final_target_created"] is False


def test_windows_drive_path_maps_deterministically_into_wsl(tmp_path: Path) -> None:
    path = tmp_path / "sigma" / "archive.tar.gz"
    mapped = environment.to_wsl_path(path, "Ubuntu-24.04")
    resolved = path.resolve()
    expected_suffix = "/".join(resolved.relative_to(resolved.anchor).parts)
    assert mapped == f"/mnt/{resolved.drive[0].lower()}/{expected_suffix}"


def test_archive_member_validation_accepts_only_frozen_data_root() -> None:
    environment.validate_member_names(
        ["data/xrism/resolve/", "data/xrism/resolve/caldb.indx"],
        "data/",
    )


def test_archive_member_validation_rejects_path_traversal() -> None:
    try:
        environment.validate_member_names(["data/xrism/../../escape"], "data/")
    except RuntimeError as error:
        assert "unsafe" in str(error)
    else:
        raise AssertionError("path traversal was accepted")


def test_archive_member_validation_rejects_wrong_root() -> None:
    try:
        environment.validate_member_names(["software/tools/caldb.config"], "data/")
    except RuntimeError as error:
        assert "outside data/" in str(error)
    else:
        raise AssertionError("archive member outside the frozen root was accepted")


def test_shell_quote_handles_single_quotes() -> None:
    assert environment.shell_quote("sigma'gravity") == "'sigma'\"'\"'gravity'"
