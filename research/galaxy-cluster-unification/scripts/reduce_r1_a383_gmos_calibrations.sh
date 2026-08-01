#!/usr/bin/env bash
set -euo pipefail

sg_root=/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification
sg_env=/home/henry/.local/share/sigmagravity-dragons/miniforge3/envs/dragons-4.2.2/bin
sg_raw="$sg_root/data/raw/r1_a383_gemini"
sg_output="$sg_root/data/derived/r1_a383_gmos_reconstruction/calibrations"
sg_config="$sg_root/configs/a383_dragonsrc.ini"
sg_database="$sg_root/data/derived/r1_a383_gmos_reconstruction/calibrations.db"
sg_gate="$sg_root/results/r1_a383_dragons_environment/report.json"

"$sg_env/python" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["gates"]["P1_environment_and_bpm_gate_passed"] else 1)' "$sg_gate"
mkdir -p "$sg_output"
if [[ ! -f "$sg_database" ]]; then
  "$sg_env/caldb" init -d "$sg_database" -v
fi
if ! "$sg_env/caldb" list -d "$sg_database" | grep -q 'bpm_20030101_gmos-s_EEV_22_full_3amp.fits'; then
  "$sg_env/caldb" add -d "$sg_database" "$sg_raw/bpm_20030101_gmos-s_EEV_22_full_3amp.fits"
fi

cd "$sg_output"
if [[ ! -f S20071010S0138_bias.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile bias_20071010.log \
    -p stackBiases:operation=median stackBiases:reject_method=sigclip \
    stackBiases:mclip=True stackBiases:lsigma=3.0 stackBiases:hsigma=3.0 \
    "$sg_raw/S20071010S0138.fits" "$sg_raw/S20071010S0139.fits" \
    "$sg_raw/S20071010S0140.fits" "$sg_raw/S20071010S0141.fits" \
    "$sg_raw/S20071010S0142.fits"
fi

if [[ ! -f S20071013S0037_flat.fits || ! -f S20071016S0041_flat.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile flats.log \
    "$sg_raw/S20071013S0037.fits" "$sg_raw/S20071013S0040.fits" \
    "$sg_raw/S20071016S0036.fits" "$sg_raw/S20071016S0038.fits" \
    "$sg_raw/S20071016S0041.fits"
fi

if [[ ! -f S20071013S0041_arc.fits || ! -f S20071016S0040_arc.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile arcs.log \
    "$sg_raw/S20071013S0041.fits" "$sg_raw/S20071016S0040.fits"
fi

"$sg_env/caldb" list -d "$sg_database"
