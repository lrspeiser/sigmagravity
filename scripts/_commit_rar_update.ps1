#!/usr/bin/env pwsh
# Commit RAR analysis updates and push to GitHub

git add README.md
git add data/gaia/outputs/mw_gaia_144k_predicted.csv
git add data/gaia/outputs/mw_rar_starlevel.csv
git add data/gaia/outputs/mw_rar_starlevel.png
git add data/gaia/outputs/mw_rar_starlevel_metrics.txt
git add data/gaia/new/FETCH_STATUS.md
git add scripts/merge_gaia_wedges_to_npz.py
git add scripts/analyze_mw_rar_starlevel.py

git commit -m "Add star-level RAR analysis for Milky Way (Gaia DR3 144k)

- Computed per-star radial acceleration residuals (global + radial bins)
- Generated hexbin scatter plots (obs vs baryon; obs vs model)
- Updated README.md section 5.4 with authoritative metrics and figures
- Global: Sigma-Gravity reduces bias 5.2x (0.415->0.080 dex) vs GR
- Radial bins (6-8, 8-10, 10-12, 12-16 kpc) show consistent improvements
- Created merge script for Gaia wedges (anticenter PM 5k staged)
- Documented fetch status and integration blockers

Artifacts:
- data/gaia/outputs/mw_rar_starlevel_metrics.txt (authoritative)
- data/gaia/outputs/mw_rar_starlevel.png (hexbin scatter)
- data/gaia/outputs/mw_gaia_144k_predicted.csv (143,995 rows)

Commands to reproduce:
  python scripts/predict_gaia_star_speeds.py --npz data/gaia/mw/mw_gaia_144k.npz --fit data/gaia/outputs/mw_pipeline_run/fit_params.json --out data/gaia/outputs/mw_gaia_144k_predicted.csv --device 0
  python scripts/analyze_mw_rar_starlevel.py --pred_csv data/gaia/outputs/mw_gaia_144k_predicted.csv --out_prefix data/gaia/outputs/mw_rar_starlevel --hexbin
"

git push origin main
