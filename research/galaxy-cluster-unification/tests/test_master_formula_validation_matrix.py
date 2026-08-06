import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "master_formula_validation_matrix.json"
DOCUMENT = ROOT / "docs" / "MASTER_FORMULA_VALIDATION_MATRIX.md"


def _load():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    document = DOCUMENT.read_text(encoding="utf-8")
    ids = re.findall(r"^\| ([A-Z]\d{2}) \|", document, flags=re.MULTILINE)
    return config, document, ids


def test_matrix_has_unique_contiguous_ids_and_declared_counts():
    config, _, ids = _load()
    assert len(ids) == config["test_count"]
    assert len(ids) == len(set(ids))

    for prefix, expected_count in config["category_counts"].items():
        observed = sorted(item for item in ids if item.startswith(prefix))
        assert len(observed) == expected_count
        assert observed == [f"{prefix}{number:02d}" for number in range(1, expected_count + 1)]


def test_matrix_preserves_the_project_non_tuning_contract():
    config, document, _ = _load()
    thresholds = config["frozen_core_thresholds"]
    assert thresholds["maximum_universal_physical_constants"] == 5
    assert thresholds["maximum_per_object_gravity_parameters"] == 0
    assert thresholds["maximum_lensing_only_parameters"] == 0
    assert thresholds["strong_lens_root_recovery_fraction_minimum"] == 1.0
    assert thresholds["baryon_to_halo_gap_closure_fraction_minimum"] == 0.75
    assert not config["changes_formula_or_constants"]
    assert not config["opens_validation_or_holdout"]
    assert "A single failure is retained and diagnosed" in document
    assert "three materially different, physically derived closures" in document


def test_claim_levels_do_not_confuse_phenomenology_with_a_dark_matter_replacement():
    config, document, _ = _load()
    levels = config["claim_levels"]
    assert levels["B_viable_relativistic_gravity"]["requires_level"] == "A_useful_low_redshift_law"
    assert levels["C_credible_dark_matter_alternative"]["requires_level"] == "B_viable_relativistic_gravity"
    assert set(levels["C_credible_dark_matter_alternative"]["additional_required_categories"]) == {"D", "K"}
    assert "cannot earn Levels B or C" in document


def test_all_parent_protocols_exist():
    config, _, _ = _load()
    for relative_path in config["parents"]:
        assert (ROOT / relative_path).is_file(), relative_path
