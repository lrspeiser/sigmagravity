#!/usr/bin/env bash
# Compare frozen baseline and sqrt(2)-refined detector maps for a04->a04 PSF ARFs.
set -euo pipefail

analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
background_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b/background
x3_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x3/annular_products
convergence_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x4/map_resolution_convergence_v0_1
heasoft_prefix=/home/henry/.local/share/sigmagravity-xmm/heasoft-6.36
sas_prefix=/home/henry/.local/share/sigmagravity-xmm/sas-22.1.0/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=/home/henry/.local/share/sigmagravity-xmm/ccf/2026-07-27

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

[[ -f "${x3_root}/.x3_annular_products_complete" ]]
mkdir -p "${convergence_root}"

reject_fatal_log() {
  local log_file=$1
  if grep -Eiq '\*\* .*: error|detmapXBoundsExceeded|detmapYBoundsExceeded|zeroSumDetmap' "${log_file}"; then
    printf 'Fatal X4 map-convergence record in %s\n' "${log_file}" >&2
    return 1
  fi
}

build_uniform_map() {
  local directory=$1
  local event_file=$2
  local pattern_max=$3
  local dimension=$4
  local label=$5
  local counts=${directory}/${label}_counts.fits
  local zero=${directory}/${label}_zero.fits
  local uniform=${directory}/${label}.fits
  if [[ ! -f "${directory}/.${label}_complete" ]]; then
    evselect table="${event_file}:EVENTS" \
      expression="(PI in [700:7000])&&(PATTERN<=${pattern_max})&&(FLAG==0)" \
      withfilteredset=no keepfilteroutput=no withimageset=yes imageset="${counts}" \
      xcolumn=DETX ycolumn=DETY imagebinning=imageSize squarepixels=yes \
      ximagesize="${dimension}" yimagesize="${dimension}" \
      withxranges=yes ximagemin=-26000 ximagemax=26000 \
      withyranges=yes yimagemin=-26000 yimagemax=26000 ignorelegallimits=yes \
      writedss=yes updateexposure=yes 2>&1 | tee "${directory}/${label}_evselect.log"
    fcarith infile="${counts}" const=0 outfil="${zero}" ops=MUL copyprime=yes clobber=no
    fcarith infile="${zero}" const=1 outfil="${uniform}" ops=ADD copyprime=yes clobber=no
    [[ -s "${uniform}" ]]
    touch "${directory}/.${label}_complete"
  fi
}

run_instrument() {
  local instrument=$1
  local event_file=$2
  local spectrum=$3
  local pattern_max=$4
  local lower=$5
  local directory=${convergence_root}/${instrument}
  local baseline_map=${directory}/${lower}_uniform_650.fits
  local refined_map=${directory}/${lower}_uniform_920.fits
  local rmf=${directory}/${instrument}_a04.rmf
  mkdir -p "${directory}"
  build_uniform_map "${directory}" "${event_file}" "${pattern_max}" 650 "${lower}_uniform_650"
  build_uniform_map "${directory}" "${event_file}" "${pattern_max}" 920 "${lower}_uniform_920"

  if [[ ! -f "${directory}/.rmf_complete" ]]; then
    rmfgen spectrumset="${spectrum}" rmfset="${rmf}" detmaptype=dataset \
      detmaparray="${baseline_map}" filterdss=yes extendedsource=yes modelee=no \
      withbadpixcorr=no applyxcaladjustment=no applyabsfluxcorr=no \
      2>&1 | tee "${directory}/rmfgen.log"
    reject_fatal_log "${directory}/rmfgen.log"
    [[ -s "${rmf}" ]]
    touch "${directory}/.rmf_complete"
  fi

  for resolution in 650 920; do
    local map=${directory}/${lower}_uniform_${resolution}.fits
    local arf=${directory}/${instrument}_a04_from_a04_cross_${resolution}.arf
    local log=${directory}/arfgen_cross_${resolution}.log
    if [[ ! -f "${directory}/.cross_${resolution}_complete" ]]; then
      arfgen spectrumset="${spectrum}" crossreg_spectrumset="${spectrum}" \
        crossregionarf=yes arfset="${arf}" withrmfset=yes rmfset="${rmf}" \
        detmaptype=dataset detmaparray="${map}" filterdss=yes \
        extendedsource=yes modelee=yes psfmodel=ELLBETA modelootcorr=yes \
        withbadpixcorr=yes badpixlocation="${event_file}" withbadpixres=yes \
        badpixelresolution=1 applyxcaladjustment=no applyabsfluxcorr=no \
        2>&1 | tee "${log}"
      reject_fatal_log "${log}"
      [[ -s "${arf}" ]]
      touch "${directory}/.cross_${resolution}_complete"
    fi
  done
  touch "${directory}/.instrument_convergence_complete"
}

run_instrument MOS2 "${background_root}/MOS2_corner_preserving_events.ds" \
  "${x3_root}/a04_175_275kpc/MOS2/mos2S002-fovt.pi" 12 mos2
run_instrument pn "${background_root}/pn_corner_preserving_events.ds" \
  "${x3_root}/a04_175_275kpc/pn/pnS003-fovt.pi" 4 pn

touch "${convergence_root}/.map_resolution_convergence_complete"
printf 'Frozen X4 map-resolution products: %s\n' "${convergence_root}"
