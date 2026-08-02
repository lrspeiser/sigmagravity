#!/usr/bin/env python3
"""Run the frozen P0729 qualified 80-step fine-grid AQUAL test."""

from __future__ import annotations

from run_p0728_complete_fine_grid_aqual import ROOT, main

if __name__ == "__main__":
    main(
        default_config=ROOT
        / "configs"
        / "p0729_qualified_80step_fine_grid_aqual.json",
        default_output=ROOT / "results" / "p0729_qualified_80step_fine_grid_aqual",
        default_work=ROOT / "tmp" / "p0729-worker-cache",
    )
