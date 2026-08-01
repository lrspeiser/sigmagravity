#!/usr/bin/env bash
# Build the frozen MOS2+pn X4 direct, cross-region, and central-source responses.
set -euo pipefail

repo_root=/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification
analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
background_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b/background
x3_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x3/annular_products
x4_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x4/cross_region_responses
heasoft_prefix=/home/henry/.local/share/sigmagravity-xmm/heasoft-6.36
sas_prefix=/home/henry/.local/share/sigmagravity-xmm/sas-22.1.0/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=/home/henry/.local/share/sigmagravity-xmm/ccf/2026-07-27

annulus_ids=(
  a01_010_050kpc
  a02_050_100kpc
  a03_100_175kpc
  a04_175_275kpc
  a05_275_380kpc
  a06_380_500kpc
)
map_dimensions=(2917 1651 996 920 920 920)
unique_map_dimensions=(2917 1651 996 920)

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
grep -Fq '"X4_map_resolution_convergence_passed": true' \
  "${repo_root}/results/r1_rxj2129_xmm_x4_map_resolution_convergence/report.json"
mkdir -p "${x4_root}/detmaps"

reject_fatal_log() {
  local log_file=$1
  if grep -Eiq '\*\* .*: error|detmapXBoundsExceeded|detmapYBoundsExceeded|zeroSumDetmap' "${log_file}"; then
    printf 'Fatal X4 response record in %s\n' "${log_file}" >&2
    return 1
  fi
}

preserve_partial() {
  local path=$1
  if [[ -e "${path}" ]]; then
    mv -- "${path}" "${path}.incomplete_${run_tag}"
  fi
}

build_uniform_map() {
  local instrument=$1
  local lower=$2
  local event_file=$3
  local pattern_max=$4
  local dimension=$5
  local directory=${x4_root}/detmaps/${instrument}
  local counts=${directory}/${lower}_uniform_${dimension}_counts.fits
  local zero=${directory}/${lower}_uniform_${dimension}_zero.fits
  local uniform=${directory}/${lower}_uniform_${dimension}.fits
  mkdir -p "${directory}"
  if [[ ! -f "${directory}/.${lower}_uniform_${dimension}_complete" ]]; then
    preserve_partial "${counts}"
    preserve_partial "${zero}"
    preserve_partial "${uniform}"
    preserve_partial "${directory}/${lower}_uniform_${dimension}.log"
    evselect table="${event_file}:EVENTS" \
      expression="(PI in [700:7000])&&(PATTERN<=${pattern_max})&&(FLAG==0)" \
      withfilteredset=no keepfilteroutput=no withimageset=yes imageset="${counts}" \
      xcolumn=DETX ycolumn=DETY imagebinning=imageSize squarepixels=yes \
      ximagesize="${dimension}" yimagesize="${dimension}" \
      withxranges=yes ximagemin=-26000 ximagemax=26000 \
      withyranges=yes yimagemin=-26000 yimagemax=26000 ignorelegallimits=yes \
      writedss=yes updateexposure=yes 2>&1 | tee "${directory}/${lower}_uniform_${dimension}.log"
    fcarith infile="${counts}" const=0 outfil="${zero}" ops=MUL copyprime=yes clobber=no
    fcarith infile="${zero}" const=1 outfil="${uniform}" ops=ADD copyprime=yes clobber=no
    [[ -s "${uniform}" ]]
    touch "${directory}/.${lower}_uniform_${dimension}_complete"
  fi
}

spectrum_path() {
  local instrument=$1
  local annulus=$2
  if [[ "${instrument}" == MOS2 ]]; then
    printf '%s\n' "${x3_root}/${annulus}/MOS2/mos2S002-fovt.pi"
  else
    printf '%s\n' "${x3_root}/${annulus}/pn/pnS003-fovt.pi"
  fi
}

run_instrument() {
  local instrument=$1
  local lower=$2
  local event_file=$3
  local pattern_max=$4
  local dimension
  mkdir -p "${x4_root}/work/${instrument}"
  cd "${x4_root}/work/${instrument}"
  for dimension in "${unique_map_dimensions[@]}"; do
    build_uniform_map "${instrument}" "${lower}" "${event_file}" \
      "${pattern_max}" "${dimension}"
  done

  local output_index input_index output_annulus input_annulus output_spectrum input_spectrum
  local output_directory output_dimension input_dimension cross_dimension map rmf direct central
  for output_index in "${!annulus_ids[@]}"; do
    output_annulus=${annulus_ids[${output_index}]}
    output_dimension=${map_dimensions[${output_index}]}
    output_spectrum=$(spectrum_path "${instrument}" "${output_annulus}")
    output_directory=${x4_root}/${instrument}/${output_annulus}
    map=${x4_root}/detmaps/${instrument}/${lower}_uniform_${output_dimension}.fits
    rmf=${output_directory}/${instrument}_${output_annulus}.rmf
    direct=${output_directory}/${instrument}_${output_annulus}_direct.arf
    central=${output_directory}/${instrument}_${output_annulus}_from_central_source50.arf
    mkdir -p "${output_directory}"
    cd "${output_directory}"

    if [[ ! -f .rmf_complete ]]; then
      preserve_partial "${rmf}"
      preserve_partial rmfgen.log
      rmfgen spectrumset="${output_spectrum}" rmfset="${rmf}" detmaptype=dataset \
        detmaparray="${map}" filterdss=yes extendedsource=yes modelee=no \
        withbadpixcorr=no applyxcaladjustment=no applyabsfluxcorr=no \
        2>&1 | tee rmfgen.log
      reject_fatal_log rmfgen.log
      [[ -s "${rmf}" ]]
      touch .rmf_complete
    fi

    if [[ ! -f .direct_complete ]]; then
      preserve_partial "${direct}"
      preserve_partial arfgen_direct.log
      arfgen spectrumset="${output_spectrum}" arfset="${direct}" \
        withrmfset=yes rmfset="${rmf}" detmaptype=dataset detmaparray="${map}" \
        filterdss=yes extendedsource=yes modelee=no modelootcorr=yes \
        withbadpixcorr=yes badpixlocation="${event_file}" withbadpixres=yes \
        badpixelresolution=1 applyxcaladjustment=no applyabsfluxcorr=no \
        2>&1 | tee arfgen_direct.log
      reject_fatal_log arfgen_direct.log
      [[ -s "${direct}" ]]
      touch .direct_complete
    fi

    if [[ ! -f .central_complete ]]; then
      preserve_partial "${central}"
      preserve_partial arfgen_central.log
      arfgen spectrumset="${output_spectrum}" arfset="${central}" \
        withrmfset=yes rmfset="${rmf}" detmaptype=dataset detmaparray="${map}" \
        filterdss=yes extendedsource=no modelee=yes withsourcepos=yes \
        sourcecoords=eqpos sourcex=322.4166857950391 sourcey=0.08898852034328401 \
        psfmodel=ELLBETA modelootcorr=yes withbadpixcorr=yes \
        badpixlocation="${event_file}" withbadpixres=yes badpixelresolution=1 \
        applyxcaladjustment=no applyabsfluxcorr=no 2>&1 | tee arfgen_central.log
      reject_fatal_log arfgen_central.log
      [[ -s "${central}" ]]
      touch .central_complete
    fi

    for input_index in "${!annulus_ids[@]}"; do
      input_annulus=${annulus_ids[${input_index}]}
      input_dimension=${map_dimensions[${input_index}]}
      if (( input_dimension > output_dimension )); then
        cross_dimension=${input_dimension}
      else
        cross_dimension=${output_dimension}
      fi
      input_spectrum=$(spectrum_path "${instrument}" "${input_annulus}")
      map=${x4_root}/detmaps/${instrument}/${lower}_uniform_${cross_dimension}.fits
      if [[ ! -f ".cross_from_${input_annulus}_complete" ]]; then
        preserve_partial "${output_directory}/${instrument}_${output_annulus}_from_${input_annulus}_cross.arf"
        preserve_partial "arfgen_cross_from_${input_annulus}.log"
        arfgen spectrumset="${output_spectrum}" crossreg_spectrumset="${input_spectrum}" \
          crossregionarf=yes \
          arfset="${output_directory}/${instrument}_${output_annulus}_from_${input_annulus}_cross.arf" \
          withrmfset=yes rmfset="${rmf}" detmaptype=dataset detmaparray="${map}" \
          filterdss=yes extendedsource=yes modelee=yes psfmodel=ELLBETA modelootcorr=yes \
          withbadpixcorr=yes badpixlocation="${event_file}" withbadpixres=yes \
          badpixelresolution=1 applyxcaladjustment=no applyabsfluxcorr=no \
          2>&1 | tee "arfgen_cross_from_${input_annulus}.log"
        reject_fatal_log "arfgen_cross_from_${input_annulus}.log"
        [[ -s "${output_directory}/${instrument}_${output_annulus}_from_${input_annulus}_cross.arf" ]]
        touch ".cross_from_${input_annulus}_complete"
      fi
    done
    touch .output_annulus_complete
    printf 'Completed X4 %s output %s\n' "${instrument}" "${output_annulus}"
  done
  touch "${x4_root}/${instrument}/.instrument_complete"
}

run_instrument MOS2 mos2 "${background_root}/MOS2_corner_preserving_events.ds" 12 &
mos2_pid=$!
run_instrument pn pn "${background_root}/pn_corner_preserving_events.ds" 4 &
pn_pid=$!
parallel_status=0
wait "${mos2_pid}" || parallel_status=1
wait "${pn_pid}" || parallel_status=1
if (( parallel_status != 0 )); then
  printf 'At least one full X4 detector branch failed.\n' >&2
  exit 1
fi

touch "${x4_root}/.x4_response_products_complete"
printf 'Frozen X4 response root: %s\n' "${x4_root}"
