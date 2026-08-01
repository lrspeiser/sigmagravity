#!/usr/bin/env bash
set -euo pipefail

sg_root=/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification
sg_env=/home/henry/.local/share/sigmagravity-dragons/miniforge3/envs/dragons-4.2.2/bin
sg_raw="$sg_root/data/raw/r1_a2261_gemini"
sg_cal="$sg_root/data/derived/r1_a2261_gmos_reconstruction/calibrations"
sg_output="$sg_root/data/derived/r1_a2261_gmos_reconstruction/science_cal2d"
sg_config="$sg_root/configs/a2261_dragonsrc.ini"
sg_recipe="$sg_root/scripts/r1_a2261_gmos_science_recipe.reduceFrozenCalibrated2D"
sg_gate="$sg_root/results/r1_a2261_gmos_calibrations/report.json"
sg_bpm="$sg_raw/bpm_20010801_gmos-n_EEV_22_full_3amp.fits"

"$sg_env/python" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["gates"]["P2a_calibration_products_gate_passed"] else 1)' "$sg_gate"
mkdir -p "$sg_output"
cd "$sg_output"

reduce_one() {
  local science=$1
  local bias=$2
  local flat=$3
  local arc=$4
  local stem=${science%.fits}
  if compgen -G "${stem}*_cal2d.fits" >/dev/null; then
    return
  fi
  "$sg_env/reduce" -c "$sg_config" --logfile "${stem}.log" \
    -r "$sg_recipe" \
    --user_cal "processed_bpm:$sg_bpm" "processed_bias:$sg_cal/$bias" \
    "processed_flat:$sg_cal/$flat" "processed_arc:$sg_cal/$arc" \
    "$sg_raw/$science"
}

reduce_one N20080315S0113.fits N20080315S0159_bias.fits N20080315S0112_flat.fits N20080315S0116_arc.fits
reduce_one N20080315S0114.fits N20080315S0159_bias.fits N20080315S0115_flat.fits N20080315S0116_arc.fits
reduce_one N20080316S0116.fits N20080316S0230_bias.fits N20080316S0115_flat.fits N20080316S0119_arc.fits
reduce_one N20080316S0117.fits N20080316S0230_bias.fits N20080316S0118_flat.fits N20080316S0119_arc.fits
