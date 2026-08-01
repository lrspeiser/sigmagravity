#!/usr/bin/env bash
# Build frozen MOS2+pn X3 annular source, OOT, QPB, RMF, and ARF products.
set -euo pipefail

analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
x2b_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b
background_root=${x2b_root}/background
x3_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x3/annular_products
x3_mask_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x3/masks
heasoft_prefix=/home/henry/.local/share/sigmagravity-xmm/heasoft-6.36
sas_prefix=/home/henry/.local/share/sigmagravity-xmm/sas-22.1.0/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=/home/henry/.local/share/sigmagravity-xmm/ccf/2026-07-27

center_ra=322.41651
center_dec=0.08923
annulus_ids=(
  a01_010_050kpc
  a02_050_100kpc
  a03_100_175kpc
  a04_175_275kpc
  a05_275_380kpc
  a06_380_500kpc
)
inner_radii_det=(
  53.55796608107742
  267.7898304053871
  535.5796608107742
  937.2644064188547
  1472.844067229629
  2035.202711080942
)
outer_radii_det=(
  267.7898304053871
  535.5796608107742
  937.2644064188547
  1472.844067229629
  2035.202711080942
  2677.8983040538706
)
pn_quadrants=(
  "F T F F"
  "F T F F"
  "F T F F"
  "F T T F"
  "T T T F"
  "T T T T"
)

export CONDA_PREFIX="${heasoft_prefix}"
source "${heasoft_prefix}/bin/heainit.sh" >/dev/null 2>&1
set +u
source "${sas_prefix}/setsas.sh" >/dev/null 2>&1
set -u
export SAS_CCFPATH="${ccf_snapshot}"
export SAS_CCF="${analysis_root}/ccf.cif"
export SAS_ODF="${analysis_root}/0529_0093030201_SCX00000SUM.SAS"
export SAS_VERBOSITY=4
export SAS_SUPPRESS_WARNING=3

[[ -f "${background_root}/esas_outer_annulus/MOS2/.mosback_outer_complete" ]]
[[ -f "${background_root}/esas_outer_annulus/pn/.pnback_outer_complete" ]]
[[ -f "${background_root}/MOS2_corner_preserving_events.ds" ]]
[[ -f "${background_root}/pn_corner_preserving_events.ds" ]]
[[ -f "${background_root}/pn_oot_corner_preserving_events.ds" ]]
[[ -f "${x3_mask_root}/MOS2_point_source_mask_detector.fits" ]]
[[ -f "${x3_mask_root}/MOS2_point_source_mask_sky.fits" ]]
[[ -f "${x3_mask_root}/pn_point_source_mask_detector.fits" ]]
[[ -f "${x3_mask_root}/pn_point_source_mask_sky.fits" ]]
mkdir -p "${x3_root}"

prepare_annulus() {
  local directory=$1
  local instrument=$2
  local image=$3
  local source_prefix=$4
  local inner_radius=$5
  local outer_radius=$6
  local annulus=$7
  local detxy detx dety detector_rows sky_rows
  mkdir -p "${directory}"
  detxy=$(esky2det datastyle=user ra="${center_ra}" dec="${center_dec}" \
    instrument="${instrument}" checkfov=no outunit=det withheader=no \
    calinfostyle=set calinfoset="${image}" verbosity=0 \
    2>"${directory}/annulus_esky2det.log")
  read -r detx dety <<<"${detxy}"
  [[ "${detx}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]
  [[ "${dety}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]
  printf '%s\n' \
    "&&((DETX,DETY) IN circle(${detx},${dety},${outer_radius}))&&!((DETX,DETY) IN circle(${detx},${dety},${inner_radius}))" \
    >"${directory}/annulus_region.txt"
  [[ $(wc -l <"${directory}/annulus_region.txt") -eq 1 ]]
  [[ $(grep -o 'circle(' "${directory}/annulus_region.txt" | wc -l) -eq 2 ]]
  cp -- "${x3_mask_root}/${annulus}/${source_prefix}_point_source_mask_detector.fits" \
    "${directory}/srcdet.fits"
  cp -- "${x3_mask_root}/${annulus}/${source_prefix}_point_source_mask_sky.fits" \
    "${directory}/srcsky.fits"
  detector_rows=$(ftlist "${directory}/srcdet.fits+1" K include=NAXIS2 | \
    awk '$1 == "NAXIS2" { print $3 }')
  sky_rows=$(ftlist "${directory}/srcsky.fits+1" K include=NAXIS2 | \
    awk '$1 == "NAXIS2" { print $3 }')
  [[ "${detector_rows}" =~ ^[0-9]+$ ]]
  [[ "${sky_rows}" =~ ^[0-9]+$ ]]
  [[ "${detector_rows}" -eq "${sky_rows}" ]]
  [[ "${detector_rows}" -le 86 ]]
  printf '%s\n' "${detector_rows}" >"${directory}/source_mask_rows.txt"
}

run_mos2_annulus() {
  local directory=$1
  local mask_rows
  local -a source_args
  mask_rows=$(<"${directory}/source_mask_rows.txt")
  if (( mask_rows > 0 )); then
    source_args=(withsrcrem=yes maskdet=srcdet.fits masksky=srcsky.fits)
  else
    source_args=(withsrcrem=no)
  fi
  if [[ ! -f "${directory}/.mosspectra_complete" ]]; then
    cd "${directory}"
    mosspectra \
      eventfile="${background_root}/MOS2_corner_preserving_events.ds" \
      cornerfile=mos2S002-corevc.fits \
      imagefile=mos2S002-fovimt.fits expmap=mos2S002-expimt.fits \
      spmask=mos2S002-fovspdet.fits mask=mos2S002-maskimt.fits \
      specfile=mos2S002-fovt.pi rmffile=mos2S002.rmf arffile=mos2S002.arf \
      withregion=yes regionfile=annulus_region.txt "${source_args[@]}" \
      pattern=12 keepinterfiles=yes elow=700 ehigh=7000 \
      ccds="T T T T F T T" 2>&1 | tee mosspectra.log
    if (( mask_rows > 0 )); then
      grep -Fq "DET coord region mask :    srcdet.fits" mosspectra.log
    fi
    grep -Fq "annulus_region.txt" mosspectra.log
    grep -Fq "mosspectra analysis complete" mosspectra.log
    touch .mosspectra_complete
  fi
  if [[ ! -f "${directory}/.mosback_complete" ]]; then
    cd "${directory}"
    mosback inspecfile=mos2S002-fovt.pi outspecfile=mos2S002-bkg.pi \
      rmffile=mos2S002.rmf withplotfiles=yes \
      inimgfile=mos2S002-fovimdet-700-7000.fits \
      outimgfile=mos2S002-bkgimdet-700-7000.fits \
      elow=700 ehigh=7000 ccds="T T T T F T T" 2>&1 | tee mosback.log
    [[ -s mos2S002-bkg.pi && -s mos2S002-bkgimdet-700-7000.fits ]]
    grep -Eq "mosback .* ended:" mosback.log
    touch .mosback_complete
  fi
}

run_pn_annulus() {
  local directory=$1
  local quadrants=$2
  local mask_rows
  local -a source_args
  mask_rows=$(<"${directory}/source_mask_rows.txt")
  if (( mask_rows > 0 )); then
    source_args=(withsrcrem=yes maskdet=srcdet.fits masksky=srcsky.fits)
  else
    source_args=(withsrcrem=no)
  fi
  if [[ ! -f "${directory}/.pnspectra_complete" ]]; then
    cd "${directory}"
    pnspectra \
      eventfile="${background_root}/pn_corner_preserving_events.ds" \
      ootevtfile="${background_root}/pn_oot_corner_preserving_events.ds" \
      cornerfile=pnS003-corevc.fits ootcornfile=pnS003-corevc-oot.fits \
      imagefile=pnS003-fovimt.fits ootimgfile=pnS003-fovimt-oot.fits \
      expmap=pnS003-expimt.fits spmask=pnS003-fovspdet.fits \
      mask=pnS003-maskimt.fits specfile=pnS003-fovt.pi \
      ootspecfile=pnS003-fovt-oot.pi rmffile=pnS003.rmf arffile=pnS003.arf \
      withregion=yes regionfile=annulus_region.txt "${source_args[@]}" \
      pattern=4 keepinterfiles=yes elow=700 ehigh=7000 \
      badpixelresolution=1 quads="${quadrants}" 2>&1 | tee pnspectra.log
    if (( mask_rows > 0 )); then
      grep -Fq "DET region file selected also exists" pnspectra.log
    fi
    grep -Fq "annulus_region.txt" pnspectra.log
    grep -Fq "pnspectra analysis complete" pnspectra.log
    touch .pnspectra_complete
  fi
  if [[ ! -f "${directory}/.pnback_complete" ]]; then
    cd "${directory}"
    pnback inspecfile=pnS003-fovt.pi inspecoot=pnS003-fovt-oot.pi \
      outspecfile=pnS003-bkg.pi outspecoot=pnS003-bkg-oot.pi \
      rmffile=pnS003.rmf withplotfiles=yes \
      inimgfile=pnS003-fovimdet-700-7000.fits \
      inimgoot=pnS003-fovimootdet-700-7000.fits \
      outimgfile=pnS003-bkgimdet-700-7000.fits \
      elow=700 ehigh=7000 quads="${quadrants}" 2>&1 | tee pnback.log
    [[ -s pnS003-bkg.pi && -s pnS003-bkgimdet-700-7000.fits ]]
    grep -Eq "pnback .* ended:" pnback.log
    touch .pnback_complete
  fi
}

for index in "${!annulus_ids[@]}"; do
  annulus=${annulus_ids[${index}]}
  inner=${inner_radii_det[${index}]}
  outer=${outer_radii_det[${index}]}
  mos2_directory=${x3_root}/${annulus}/MOS2
  pn_directory=${x3_root}/${annulus}/pn
  printf '%s\n' "${pn_quadrants[${index}]}" >"${x3_root}/${annulus}_pn_quadrants.pending"
  prepare_annulus "${mos2_directory}" EMOS2 \
    "${x2b_root}/detect_band2_1200_2000eV/mos2S002-fovimt.fits" \
    MOS2 "${inner}" "${outer}" "${annulus}"
  prepare_annulus "${pn_directory}" EPN \
    "${x2b_root}/detect_band2_1200_2000eV/pnS003-fovimt.fits" \
    pn "${inner}" "${outer}" "${annulus}"
  run_mos2_annulus "${mos2_directory}"
  mv -- "${x3_root}/${annulus}_pn_quadrants.pending" "${pn_directory}/pn_quadrants.txt"
  run_pn_annulus "${pn_directory}" "${pn_quadrants[${index}]}"
  touch "${x3_root}/${annulus}/.annulus_complete"
  printf 'Completed frozen X3 annulus %s\n' "${annulus}"
done

touch "${x3_root}/.x3_annular_products_complete"
printf 'Frozen X3 annular ESAS root: %s\n' "${x3_root}"
