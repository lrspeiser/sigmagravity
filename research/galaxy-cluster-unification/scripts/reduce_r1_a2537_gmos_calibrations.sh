#!/usr/bin/env bash
set -euo pipefail

sg_root=/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification
sg_env=/home/henry/.local/share/sigmagravity-dragons/miniforge3/envs/dragons-4.2.2/bin
sg_raw="$sg_root/data/raw/r1_a2537_gemini"
sg_output="$sg_root/data/derived/r1_a2537_gmos_control/calibrations"
sg_config="$sg_root/configs/a2537_dragonsrc.ini"
sg_database="$sg_root/data/derived/r1_a2537_gmos_control/calibrations.db"
sg_gate="$sg_root/results/r1_a2537_dragons_environment/report.json"

"$sg_env/python" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["gates"]["C1_environment_and_bpm_gate_passed"] else 1)' "$sg_gate"
mkdir -p "$sg_output"
if [[ ! -f "$sg_database" ]]; then
  "$sg_env/caldb" init -d "$sg_database" -v
fi
if ! "$sg_env/caldb" list -d "$sg_database" | grep -q 'bpm_20030101_gmos-s_EEV_22_full_3amp.fits'; then
  "$sg_env/caldb" add -d "$sg_database" "$sg_raw/bpm_20030101_gmos-s_EEV_22_full_3amp.fits"
fi

cd "$sg_output"
if [[ ! -f S20080921S0103_bias.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile bias_20080921.log \
    -p stackBiases:operation=median stackBiases:reject_method=sigclip \
    stackBiases:mclip=True stackBiases:lsigma=3.0 stackBiases:hsigma=3.0 \
    "$sg_raw/S20080921S0103.fits" "$sg_raw/S20080921S0104.fits" \
    "$sg_raw/S20080921S0105.fits" "$sg_raw/S20080921S0106.fits" \
    "$sg_raw/S20080921S0107.fits"
fi

if [[ ! -f S20080922S0146_bias.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile bias_20080922.log \
    -p stackBiases:operation=median stackBiases:reject_method=sigclip \
    stackBiases:mclip=True stackBiases:lsigma=3.0 stackBiases:hsigma=3.0 \
    "$sg_raw/S20080922S0146.fits" "$sg_raw/S20080922S0147.fits" \
    "$sg_raw/S20080922S0148.fits" "$sg_raw/S20080922S0149.fits" \
    "$sg_raw/S20080922S0150.fits"
fi

if [[ ! -f S20080921S0046_flat.fits || ! -f S20080922S0033_flat.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile flats.log \
    "$sg_raw/S20080921S0046.fits" "$sg_raw/S20080921S0049.fits" \
    "$sg_raw/S20080922S0030.fits" "$sg_raw/S20080922S0033.fits"
fi

if [[ ! -f S20080921S0050_arc.fits || ! -f S20080922S0034_arc.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile arcs.log \
    "$sg_raw/S20080921S0050.fits" "$sg_raw/S20080922S0034.fits"
fi

"$sg_env/caldb" list -d "$sg_database"
