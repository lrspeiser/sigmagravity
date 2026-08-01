#!/usr/bin/env bash
# Compare the promoted 920 map against its predeclared sqrt(2) refinement.
set -euo pipefail

analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
background_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b/background
x3_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x3/annular_products
baseline_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x4/map_resolution_convergence_v0_1
refinement_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x4/map_resolution_convergence_v0_2
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
run_tag=$(date -u +%Y%m%dT%H%M%SZ)

[[ -f "${x3_root}/.x3_annular_products_complete" ]]
mkdir -p "${refinement_root}"

reject_fatal_log() {
  local log_file=$1
  if grep -Eiq '\*\* .*: error|detmapXBoundsExceeded|detmapYBoundsExceeded|zeroSumDetmap' "${log_file}"; then
    printf 'Fatal X4 promoted-map convergence record in %s\n' "${log_file}" >&2
    return 1
  fi
}

preserve_partial() {
  local path=$1
  if [[ -e "${path}" ]]; then
    mv -- "${path}" "${path}.incomplete_${run_tag}"
  fi
}

run_instrument() {
  local instrument=$1
  local event_file=$2
  local spectrum=$3
  local pattern_max=$4
  local lower=$5
  local baseline_directory=${baseline_root}/${instrument}
  local directory=${refinement_root}/${instrument}
  local counts=${directory}/${lower}_uniform_1302_counts.fits
  local zero=${directory}/${lower}_uniform_1302_zero.fits
  local uniform=${directory}/${lower}_uniform_1302.fits
  local baseline_rmf=${baseline_directory}/${instrument}_a04.rmf
  local arf=${directory}/${instrument}_a04_from_a04_cross_1302.arf
  local log=${directory}/arfgen_cross_1302.log
  mkdir -p "${directory}"
  cd "${directory}"

  [[ -f "${baseline_directory}/.${lower}_uniform_920_complete" ]]
  [[ -f "${baseline_directory}/.cross_920_complete" ]]
  [[ -s "${baseline_rmf}" ]]

  if [[ ! -f "${directory}/.${lower}_uniform_1302_complete" ]]; then
    preserve_partial "${counts}"
    preserve_partial "${zero}"
    preserve_partial "${uniform}"
    evselect table="${event_file}:EVENTS" \
      expression="(PI in [700:7000])&&(PATTERN<=${pattern_max})&&(FLAG==0)" \
      withfilteredset=no keepfilteroutput=no withimageset=yes imageset="${counts}" \
      xcolumn=DETX ycolumn=DETY imagebinning=imageSize squarepixels=yes \
      ximagesize=1302 yimagesize=1302 \
      withxranges=yes ximagemin=-26000 ximagemax=26000 \
      withyranges=yes yimagemin=-26000 yimagemax=26000 ignorelegallimits=yes \
      writedss=yes updateexposure=yes 2>&1 | tee "${directory}/${lower}_uniform_1302_evselect.log"
    fcarith infile="${counts}" const=0 outfil="${zero}" ops=MUL copyprime=yes clobber=no
    fcarith infile="${zero}" const=1 outfil="${uniform}" ops=ADD copyprime=yes clobber=no
    [[ -s "${uniform}" ]]
    touch "${directory}/.${lower}_uniform_1302_complete"
  fi

  if [[ ! -f "${directory}/.cross_1302_complete" ]]; then
    preserve_partial "${arf}"
    preserve_partial "${log}"
    arfgen spectrumset="${spectrum}" crossreg_spectrumset="${spectrum}" \
      crossregionarf=yes arfset="${arf}" withrmfset=yes rmfset="${baseline_rmf}" \
      detmaptype=dataset detmaparray="${uniform}" filterdss=yes \
      extendedsource=yes modelee=yes psfmodel=ELLBETA modelootcorr=yes \
      withbadpixcorr=yes badpixlocation="${event_file}" withbadpixres=yes \
      badpixelresolution=1 applyxcaladjustment=no applyabsfluxcorr=no \
      2>&1 | tee "${log}"
    reject_fatal_log "${log}"
    [[ -s "${arf}" ]]
    touch "${directory}/.cross_1302_complete"
  fi
  touch "${directory}/.instrument_convergence_complete"
}

run_instrument MOS2 "${background_root}/MOS2_corner_preserving_events.ds" \
  "${x3_root}/a04_175_275kpc/MOS2/mos2S002-fovt.pi" 12 mos2 &
mos2_pid=$!
run_instrument pn "${background_root}/pn_corner_preserving_events.ds" \
  "${x3_root}/a04_175_275kpc/pn/pnS003-fovt.pi" 4 pn &
pn_pid=$!
parallel_status=0
wait "${mos2_pid}" || parallel_status=1
wait "${pn_pid}" || parallel_status=1
if (( parallel_status != 0 )); then
  printf 'At least one promoted X4 map-resolution branch failed.\n' >&2
  exit 1
fi

touch "${refinement_root}/.map_resolution_convergence_complete"
printf 'Promoted X4 map-resolution products: %s\n' "${refinement_root}"
