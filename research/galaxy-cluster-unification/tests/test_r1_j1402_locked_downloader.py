import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_r1_j1402_locked_data.py"
SPEC = importlib.util.spec_from_file_location("j1402_download", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_plan_contains_only_the_frozen_46_files() -> None:
    protocol = MODULE.load_protocol()
    plan = MODULE.build_downloads(protocol)
    assert len(plan) == 46
    assert sum(item.group == "dinos_github" for item in plan) == 9
    assert sum(item.group == "dinos_full_output" for item in plan) == 1
    assert sum(item.group.startswith("kcwi_") for item in plan) == 36
    assert len({(item.group, item.identity) for item in plan}) == 46


def test_every_destination_resolves_inside_the_locked_raw_root() -> None:
    for item in MODULE.build_downloads(MODULE.load_protocol()):
        destination = MODULE.resolve_destination(item)
        assert MODULE.RAW_ROOT in destination.parents


def test_archive_urls_and_signatures_are_locked() -> None:
    plan = MODULE.build_downloads(MODULE.load_protocol())
    for item in plan:
        assert item.url.startswith("https://")
        if item.group == "dinos_github":
            assert "6810ea6d8b7f97e8a7c4699d2b81b5da311c64cb" in item.url
            assert item.expected_bytes is not None
            assert len(item.expected_git_blob_sha1) == 40
        elif item.group == "dinos_full_output":
            assert "1BuAmGW5adsypaIbBrv5z3_Zdf96CXfm7" in item.url
            assert item.expected_prefix_hex == "89484446"
        else:
            assert "nph-getKOA" in item.url
            assert item.identity in item.url
            assert item.expected_prefix_hex == "53494d504c45"
