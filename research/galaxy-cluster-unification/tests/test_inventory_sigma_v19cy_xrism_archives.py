import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import inventory_sigma_v19cy_xrism_archives as inventory

REPORT = ROOT / "results" / "sigma_v19cy_direct_icm_velocity_evidence" / "archive_inventory_report.json"


def test_link_parser_collects_only_links() -> None:
    parser = inventory.LinkParser()
    parser.feed('<a href="event_cl/">events</a><a href="file.evt.gz">file</a>')
    assert parser.links == ["event_cl/", "file.evt.gz"]


def test_v19cy_inventory_config_keeps_outcomes_closed() -> None:
    payload = inventory.load_config(inventory.DEFAULT_CONFIG)
    inventory.validate_config(payload)
    assert not payload["evidence_split"]["validation"]["outcome_known_before_freeze"]
    assert not payload["evidence_split"]["holdout"]["outcome_known_before_freeze"]


def test_inventory_root_rejects_non_directory_url() -> None:
    try:
        inventory.inventory_root("https://example.test/file", 1.0)
    except RuntimeError as error:
        assert "not a directory URL" in str(error)
    else:
        raise AssertionError("non-directory archive root was accepted")


def test_terminal_metadata_inventory_preserves_outcome_seals() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["status"] == "named_public_xrism_archives_inventoried_without_scientific_outcome_access"
    assert report["manifest"]["rows"] == 568
    assert report["remote_totals"]["bytes"] == 30_602_430_184
    assert not report["file_bodies_downloaded"]
    assert not report["scientific_velocity_outcomes_opened"]
    assert report["validation_and_holdout_outcome_seals_preserved"]
