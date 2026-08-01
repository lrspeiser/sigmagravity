#!/usr/bin/env bash
set -euo pipefail

sg_root=/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification
sg_env=/home/henry/.local/share/sigmagravity-dragons/miniforge3/envs/dragons-4.2.2/bin
sg_raw="$sg_root/data/raw/r1_a1689_gemini"
sg_output="$sg_root/data/derived/r1_a1689_gmos_reconstruction/calibrations"
sg_config="$sg_root/configs/a1689_dragonsrc.ini"
sg_database="$sg_root/data/derived/r1_a1689_gmos_reconstruction/calibrations.db"

mkdir -p "$sg_output"
if [[ ! -f "$sg_database" ]]; then
  "$sg_env/caldb" init -d "$sg_database" -v
fi
if ! "$sg_env/caldb" list -d "$sg_database" | grep -q 'bpm_20010801_gmos-n_EEV_22_full_3amp.fits'; then
  "$sg_env/caldb" add -d "$sg_database" "$sg_raw/bpm_20010801_gmos-n_EEV_22_full_3amp.fits"
fi

cd "$sg_output"

if [[ ! -f N20090615S0531_bias.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile bias_20090615.log \
    -p stackBiases:operation=median stackBiases:reject_method=sigclip \
    stackBiases:mclip=True stackBiases:lsigma=3.0 stackBiases:hsigma=3.0 \
    "$sg_raw/N20090615S0531.fits" "$sg_raw/N20090615S0532.fits" \
    "$sg_raw/N20090615S0533.fits" "$sg_raw/N20090615S0534.fits" \
    "$sg_raw/N20090615S0535.fits"
fi

if [[ ! -f N20090621S0193_bias.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile bias_20090621.log \
    -p stackBiases:operation=median stackBiases:reject_method=sigclip \
    stackBiases:mclip=True stackBiases:lsigma=3.0 stackBiases:hsigma=3.0 \
    "$sg_raw/N20090621S0193.fits" "$sg_raw/N20090621S0194.fits" \
    "$sg_raw/N20090621S0195.fits" "$sg_raw/N20090621S0196.fits" \
    "$sg_raw/N20090621S0197.fits"
fi

if [[ ! -f N20090615S0078_flat.fits || ! -f N20090621S0039_flat.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile flats.log \
    "$sg_raw/N20090615S0078.fits" "$sg_raw/N20090621S0033.fits" \
    "$sg_raw/N20090621S0036.fits" "$sg_raw/N20090621S0039.fits"
fi

if [[ ! -f N20090615S0080_arc.fits || ! -f N20090621S0040_arc.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile arcs.log \
    "$sg_raw/N20090615S0080.fits" "$sg_raw/N20090621S0037.fits" \
    "$sg_raw/N20090621S0040.fits"
fi

"$sg_env/caldb" list -d "$sg_database"
