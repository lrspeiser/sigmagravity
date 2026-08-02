from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p0725_aqual_solver_robustness import physics_core
from run_p0726_independent_nonlinear_crosscheck import (
    run_known_answers,
    variant_manifest,
)


def test_frozen_root_variants_preserve_physics_and_known_answer() -> None:
    config = json.loads(
        (ROOT / "configs" / "p0726_independent_nonlinear_crosscheck.json").read_text(
            encoding="utf-8"
        )
    )
    base = json.loads((ROOT / config["baseManifest"]).read_text(encoding="utf-8"))
    for variant in config["solverVariants"]:
        model = variant_manifest(base, config["solverConstants"], variant)
        assert physics_core(model) == physics_core(base)
        assert model["parameterPolicy"]["perObjectParameters"] == []
        assert model["solver"]["nonlinearMethod"] == "newton_krylov"
    known = run_known_answers(config)
    assert len(known) == 3
    assert all(row["converged"] for row in known)
    assert max(row["relativeFieldError"] for row in known) < 1e-6
