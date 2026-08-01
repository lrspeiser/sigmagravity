import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_p0592b_fixed_candidates_and_conclusion():
    fixed = pd.read_csv(ROOT / "results/p0592b_diffusion_scale_null/fixed_candidates.csv")
    assert len(fixed) == 7
    report = json.loads((ROOT / "results/p0592b_diffusion_scale_null/report.json").read_text())
    gates = report["gates"]
    passed = all(value for key, value in gates.items() if key.endswith("_pass"))
    assert (report["conclusion"] == "adaptive_diffusion_scale_beats_fixed_blur") == passed
