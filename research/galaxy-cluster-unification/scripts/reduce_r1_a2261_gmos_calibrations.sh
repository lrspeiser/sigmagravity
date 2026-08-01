#!/usr/bin/env bash
set -euo pipefail

sg_root=/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification
sg_env=/home/henry/.local/share/sigmagravity-dragons/miniforge3/envs/dragons-4.2.2/bin
sg_raw="$sg_root/data/raw/r1_a2261_gemini"
sg_output="$sg_root/data/derived/r1_a2261_gmos_reconstruction/calibrations"
sg_config="$sg_root/configs/a2261_dragonsrc.ini"
sg_database="$sg_root/data/derived/r1_a2261_gmos_reconstruction/calibrations.db"

mkdir -p "$sg_output"
if [[ ! -f "$sg_database" ]]; then
  "$sg_env/caldb" init -d "$sg_database" -v
fi
if ! "$sg_env/caldb" list -d "$sg_database" | grep -q 'bpm_20010801_gmos-n_EEV_22_full_3amp.fits'; then
  "$sg_env/caldb" add -d "$sg_database" "$sg_raw/bpm_20010801_gmos-n_EEV_22_full_3amp.fits"
fi

cd "$sg_output"

if [[ ! -f N20080315S0159_bias.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile bias_20080315.log \
    -p stackBiases:operation=median stackBiases:reject_method=sigclip \
    stackBiases:mclip=True stackBiases:lsigma=3.0 stackBiases:hsigma=3.0 \
    "$sg_raw/N20080315S0159.fits" "$sg_raw/N20080315S0160.fits" \
    "$sg_raw/N20080315S0161.fits" "$sg_raw/N20080315S0162.fits" \
    "$sg_raw/N20080315S0163.fits"
fi

if [[ ! -f N20080316S0230_bias.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile bias_20080316.log \
    -p stackBiases:operation=median stackBiases:reject_method=sigclip \
    stackBiases:mclip=True stackBiases:lsigma=3.0 stackBiases:hsigma=3.0 \
    "$sg_raw/N20080316S0230.fits" "$sg_raw/N20080316S0231.fits" \
    "$sg_raw/N20080316S0232.fits" "$sg_raw/N20080316S0233.fits" \
    "$sg_raw/N20080316S0234.fits"
fi

if [[ ! -f N20080315S0112_flat.fits || ! -f N20080316S0118_flat.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile flats.log \
    "$sg_raw/N20080315S0112.fits" "$sg_raw/N20080315S0115.fits" \
    "$sg_raw/N20080316S0115.fits" "$sg_raw/N20080316S0118.fits"
fi

if [[ ! -f N20080315S0116_arc.fits || ! -f N20080316S0119_arc.fits ]]; then
  "$sg_env/reduce" -c "$sg_config" --logfile arcs.log \
    "$sg_raw/N20080315S0116.fits" "$sg_raw/N20080316S0119.fits"
fi

"$sg_env/caldb" list -d "$sg_database"
