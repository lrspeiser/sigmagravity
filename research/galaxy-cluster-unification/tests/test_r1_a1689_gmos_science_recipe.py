from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_science_recipe_stops_before_sky_and_extraction() -> None:
    path = ROOT / "scripts/r1_a1689_gmos_science_recipe.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    calls = [
        node.value.func.attr
        for node in function.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
    ]
    assert calls == [
        "prepare", "addDQ", "addVAR", "overscanCorrect", "biasCorrect",
        "ADUToElectrons", "addVAR", "attachWavelengthSolution", "flatCorrect",
        "QECorrect", "flagCosmicRays", "distortionCorrect", "writeOutputs",
    ]
    assert "skyCorrectFromSlit" not in calls
    assert "findApertures" not in calls
    assert "extractSpectra" not in calls
    assert "stackFrames" not in calls
