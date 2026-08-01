#!/usr/bin/env bash
# Frozen CIAO reduction for the RX J2129 two-ObsID gas-likelihood preparation.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: reduce_rxj2129_chandra.sh PROJECT_ROOT" >&2
  exit 2
fi

project_root="$(cd "$1" && pwd -P)"
protocol="$project_root/configs/r1_rxj2129_chandra_reduction_protocol.json"
raw_root="$project_root/data/raw/r1_rxj2129_chandra"
out_root="$project_root/data/derived/r1_rxj2129_chandra_reduction"
region_root="$project_root/configs/regions"

[[ -f "$protocol" ]] || { echo "missing frozen protocol: $protocol" >&2; exit 3; }
[[ "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$protocol")" == \
   "frozen_before_ciao_reprocessing_or_calibrated_product_inspection" ]] || {
  echo "refusing to run an unfrozen reduction protocol" >&2
  exit 4
}

version_output="$(ciaover -v)"
ciao_version="$(awk '$1 == "CIAO" && $2 == ":" {print $4; exit}' <<< "$version_output")"
caldb_version="$(awk '$1 == "CALDB" && $2 == ":" {print $3; exit}' <<< "$version_output")"
[[ "$ciao_version" == 4.18.* ]] || { echo "CIAO 4.18 required; found $ciao_version" >&2; exit 5; }
[[ "$caldb_version" == "4.12.4" ]] || { echo "CALDB 4.12.4 required; found $caldb_version" >&2; exit 6; }

mkdir -p "$out_root"

for obsid in 552 9370; do
  obs_root="$out_root/$obsid"
  input_dir="$obs_root/staged_input"
  repro_dir="$obs_root/repro"
  products_dir="$obs_root/products"
  spectra_dir="$obs_root/spectra"
  mkdir -p "$obs_root" "$input_dir" "$products_dir" "$spectra_dir"

  # The CDA HTTP response was transparently decompressed but retained .gz names.
  # Preserve the immutable raw files and stage byte-identical FITS files with
  # extensions that match their actual format for CIAO.
  while IFS= read -r -d '' source_file; do
    relative_file="${source_file#"$raw_root/$obsid/"}"
    target_file="$input_dir/${relative_file%.gz}"
    mkdir -p "$(dirname "$target_file")"
    if [[ ! -f "$target_file" ]] || ! cmp -s "$source_file" "$target_file"; then
      staging_file="${target_file}.partial"
      cp -p "$source_file" "$staging_file"
      mv -f "$staging_file" "$target_file"
    fi
  done < <(find "$raw_root/$obsid" -type f \( -name '*.fits' -o -name '*.fits.gz' \) -print0)

  if [[ ! -f "$repro_dir/acisf$(printf '%05d' "$obsid")_repro_evt2.fits" ]]; then
    mkdir -p "$repro_dir"
    punlearn chandra_repro
    chandra_repro indir="$input_dir" outdir="$repro_dir" check_vf_pha=yes \
      pix_adj=edser cleanup=yes clobber=no verbose=1 mode=h
  fi

  repro_evt="$(find "$repro_dir" -maxdepth 1 -type f -name '*_repro_evt2.fits' -print -quit)"
  [[ -n "$repro_evt" ]] || { echo "missing reprocessed evt2 for $obsid" >&2; exit 7; }

  lightcurve="$products_dir/${obsid}_hard_lc.fits"
  flare_gti="$products_dir/${obsid}_flare_clean.gti"
  clean_evt="$products_dir/${obsid}_clean_evt2.fits"
  blank_evt="$products_dir/${obsid}_blanksky_evt.fits"

  if [[ ! -f "$lightcurve" ]]; then
    punlearn dmextract
    dmextract infile="$repro_evt[energy=2500:7000][sky=region($region_root/r1_rxj2129_flare_background.reg)][bin time=::259.28]" \
      outfile="$lightcurve" opt=ltc1 clobber=no mode=h
  fi
  if [[ ! -f "$flare_gti" ]]; then
    punlearn deflare
    deflare infile="$lightcurve" outfile="$flare_gti" method=clean nsigma=3 \
      plot=no verbose=1 mode=h
  fi
  if [[ ! -f "$clean_evt" ]]; then
    punlearn dmcopy
    dmcopy infile="$repro_evt[@$flare_gti]" outfile="$clean_evt" clobber=no mode=h
  fi
  if [[ ! -f "$blank_evt" ]]; then
    punlearn blanksky
    blanksky evtfile="$clean_evt" outfile="$blank_evt" weight_method=particle \
      tmpdir="$products_dir" clobber=no verbose=1 mode=h
  fi

  soft_root="$products_dir/${obsid}_soft"
  broad_root="$products_dir/${obsid}_broad"
  if [[ ! -f "${soft_root}_0.7-2.0_thresh.img" ]]; then
    punlearn fluximage
    fluximage infile="$clean_evt" outroot="$soft_root" bands="0.7:2.0:1.5" \
      binsize=1 units=time psfecf=0.9 cleanup=yes clobber=no verbose=1 mode=h
  fi
  if [[ ! -f "${broad_root}_0.7-7.0_thresh.img" ]]; then
    punlearn fluximage
    fluximage infile="$clean_evt" outroot="$broad_root" bands="0.7:7.0:2.3" \
      binsize=1 units=time psfecf=0.9 cleanup=yes clobber=no verbose=1 mode=h
  fi

  soft_img="$(find "$products_dir" -maxdepth 1 -type f -name "${obsid}_soft*thresh.img" -print -quit)"
  soft_expmap="$(find "$products_dir" -maxdepth 1 -type f -name "${obsid}_soft*thresh.expmap" -print -quit)"
  broad_img="$(find "$products_dir" -maxdepth 1 -type f -name "${obsid}_broad*thresh.img" -print -quit)"
  broad_expmap="$(find "$products_dir" -maxdepth 1 -type f -name "${obsid}_broad*thresh.expmap" -print -quit)"
  [[ -n "$soft_img" && -n "$soft_expmap" && -n "$broad_img" && -n "$broad_expmap" ]] || {
    echo "missing fluximage products for $obsid" >&2
    exit 8
  }

  psfmap="$products_dir/${obsid}_soft_r90_psfmap.fits"
  if [[ ! -f "$psfmap" ]]; then
    punlearn mkpsfmap
    mkpsfmap infile="$soft_expmap" outfile="$psfmap" energy=1.5 ecf=0.9 clobber=no mode=h
  fi

  srcfile="$products_dir/${obsid}_wavdetect_src.fits"
  if [[ ! -f "$srcfile" ]]; then
    punlearn wavdetect
    wavdetect infile="$broad_img" outfile="$srcfile" scellfile="$products_dir/${obsid}_wavdetect_scell.fits" \
      imagefile="$products_dir/${obsid}_wavdetect_recon.fits" defnbkgfile="$products_dir/${obsid}_wavdetect_nbkg.fits" \
      regfile="$products_dir/${obsid}_wavdetect_src.reg" expfile="$broad_expmap" \
      psffile="$products_dir/${obsid}_broad_0.7-7.0_thresh.psfmap" scales="1 2 4 8" \
      sigthresh=1e-6 clobber=no verbose=1 mode=h
  fi

  for region_name in global_60arcsec annulus_0_5arcsec annulus_5_15arcsec annulus_15_30arcsec annulus_30_60arcsec; do
    region_file="$region_root/r1_rxj2129_${region_name}.reg"
    spectrum_root="$spectra_dir/${obsid}_${region_name}"
    if [[ ! -f "${spectrum_root}.pi" ]]; then
      punlearn specextract
      specextract infile="$clean_evt[energy=700:7000][sky=region($region_file)]" \
        outroot="$spectrum_root" bkgfile="$blank_evt[energy=700:7000][sky=region($region_file)]" \
        bkgresp=no weight=yes correctpsf=no grouptype=NONE binspec=NONE \
        clobber=no verbose=1 mode=h
    fi
  done
done

echo "Frozen RX J2129 CIAO reduction products completed at $out_root"
