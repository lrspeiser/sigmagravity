#!/usr/bin/env python3
"""
Aggregate ablation and sensitivity results into a single paper_release table.

Reads:
- many_path_model/paper_release/tables/galaxy_param_sensitivity.md
- many_path_model/paper_release/tables/cluster_param_sensitivity.md
- many_path_model/paper_release/tables/cluster_param_sensitivity_n10.md

Writes:
- many_path_model/paper_release/tables/ablation_summary.md
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PR = ROOT / 'many_path_model' / 'paper_release'
TAB = PR / 'tables'

IN_FILES = [
    TAB / 'galaxy_param_sensitivity.md',
    TAB / 'cluster_param_sensitivity.md',
    TAB / 'cluster_param_sensitivity_n10.md',
]
OUT = TAB / 'ablation_summary.md'

parts = []
for p in IN_FILES:
    if p.exists():
        parts.append(f"\n\n### {p.name}\n\n" + p.read_text())
    else:
        parts.append(f"\n\n### {p.name}\n\n(MISSING)\n")

OUT.write_text("# Parameter Ablation & Sensitivity Summary\n" + "".join(parts))
print(f"Wrote {OUT}")
