from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0725_aqual_solver_robustness import physics_core
from run_p0728_complete_fine_grid_aqual import selected_manifest

from voidscreen.field_job import file_sha256


def test_80step_candidate_was_already_qualified_and_preserves_physics() -> None:
    config = json.loads(
        (
            ROOT
            / "configs"
            / "p0729_qualified_80step_fine_grid_aqual.json"
        ).read_text(encoding="utf-8")
    )
    p0727 = json.loads((ROOT / config["p0727Report"]).read_text(encoding="utf-8"))
    base = json.loads((ROOT / config["baseManifest"]).read_text(encoding="utf-8"))
    candidate = config["selectedSolverVariant"]
    selected = next(item for item in p0727["manifests"] if item["variant"] == candidate)
    assert config["p0727VariantRequirement"] == "qualifying"
    assert candidate == "hybrid_picard80_newton"
    assert candidate in p0727["qualifyingUniversalVariants"]
    assert selected["solver"] == config["solver"]
    assert physics_core(selected_manifest(base, config)) == physics_core(base)
    assert len(config["systems"]) == 4
    for item in config["additionalLockedFiles"]:
        assert file_sha256(ROOT / item["path"]) == item["sha256"]
