from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0725_aqual_solver_robustness import physics_core, variant_manifest


def test_frozen_variants_change_only_generic_solver_controls() -> None:
    config = json.loads(
        (ROOT / "configs" / "p0725_aqual_solver_robustness.json").read_text(
            encoding="utf-8"
        )
    )
    base = json.loads((ROOT / config["baseManifest"]).read_text(encoding="utf-8"))
    for variant in config["solverVariants"]:
        model = variant_manifest(base, config["solverConstants"], variant)
        assert physics_core(model) == physics_core(base)
        assert model["parameterPolicy"]["perObjectParameters"] == []
        assert model["solver"]["maxIterations"] == 200
        assert model["solver"]["initialization"] in {
            "zero",
            "linearized_unit_coefficient",
        }
