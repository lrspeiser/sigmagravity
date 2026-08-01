#!/usr/bin/env bash
# Validate the frozen X4 SAS response interface on pn output a01/input a02.
set -euo pipefail

analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
background_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b/background
x3_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x3/annular_products
interface_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x4/interface_v0_1_pn_a01_from_a02
heasoft_prefix=/home/henry/.local/share/sigmagravity-xmm/heasoft-6.36
sas_prefix=/home/henry/.local/share/sigmagravity-xmm/sas-22.1.0/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=/home/henry/.local/share/sigmagravity-xmm/ccf/2026-07-27

event_file=${background_root}/pn_corner_preserving_events.ds
output_spectrum=${x3_root}/a01_010_050kpc/pn/pnS003-fovt.pi
input_spectrum=${x3_root}/a02_050_100kpc/pn/pnS003-fovt.pi
detmap_counts=${interface_root}/pn_uniform_detmap_counts.fits
detmap_zero=${interface_root}/pn_uniform_detmap_zero.fits
detmap_uniform=${interface_root}/pn_uniform_detmap.fits
output_rmf=${interface_root}/pn_output_a01.rmf
direct_arf=${interface_root}/pn_output_a01_direct.arf
cross_arf=${interface_root}/pn_output_a01_from_input_a02_cross.arf
central_arf=${interface_root}/pn_output_a01_from_central_source50.arf

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
[[ -s "${event_file}" ]]
[[ -s "${output_spectrum}" ]]
[[ -s "${input_spectrum}" ]]
mkdir -p "${interface_root}"

reject_fatal_log() {
  local log_file=$1
  if grep -Eiq '\*\* .*: error|detmapXBoundsExceeded|detmapYBoundsExceeded|zeroSumDetmap' "${log_file}"; then
    printf 'Fatal X4 pn interface record in %s\n' "${log_file}" >&2
    return 1
  fi
}

if [[ ! -f "${interface_root}/.detmap_complete" ]]; then
  evselect table="${event_file}:EVENTS" expression='(PI in [700:7000])&&(PATTERN<=4)&&(FLAG==0)' \
    withfilteredset=no keepfilteroutput=no withimageset=yes imageset="${detmap_counts}" \
    xcolumn=DETX ycolumn=DETY imagebinning=imageSize squarepixels=yes \
    ximagesize=2048 yimagesize=2048 withxranges=yes ximagemin=-26000 ximagemax=26000 \
    withyranges=yes yimagemin=-26000 yimagemax=26000 ignorelegallimits=yes \
    writedss=yes updateexposure=yes 2>&1 | tee "${interface_root}/detmap_evselect.log"
  fcarith infile="${detmap_counts}" const=0 outfil="${detmap_zero}" ops=MUL \
    copyprime=yes clobber=no
  fcarith infile="${detmap_zero}" const=1 outfil="${detmap_uniform}" ops=ADD \
    copyprime=yes clobber=no
  [[ -s "${detmap_uniform}" ]]
  touch "${interface_root}/.detmap_complete"
fi

if [[ ! -f "${interface_root}/.rmf_complete" ]]; then
  rmfgen spectrumset="${output_spectrum}" rmfset="${output_rmf}" \
    detmaptype=dataset detmaparray="${detmap_uniform}" filterdss=yes \
    extendedsource=yes modelee=no withbadpixcorr=no \
    applyxcaladjustment=no applyabsfluxcorr=no \
    2>&1 | tee "${interface_root}/rmfgen.log"
  reject_fatal_log "${interface_root}/rmfgen.log"
  [[ -s "${output_rmf}" ]]
  touch "${interface_root}/.rmf_complete"
fi

if [[ ! -f "${interface_root}/.direct_arf_complete" ]]; then
  arfgen spectrumset="${output_spectrum}" arfset="${direct_arf}" \
    withrmfset=yes rmfset="${output_rmf}" detmaptype=dataset \
    detmaparray="${detmap_uniform}" filterdss=yes extendedsource=yes modelee=no \
    modelootcorr=yes withbadpixcorr=yes badpixlocation="${event_file}" \
    withbadpixres=yes badpixelresolution=1 \
    applyxcaladjustment=no applyabsfluxcorr=no \
    2>&1 | tee "${interface_root}/arfgen_direct.log"
  reject_fatal_log "${interface_root}/arfgen_direct.log"
  [[ -s "${direct_arf}" ]]
  touch "${interface_root}/.direct_arf_complete"
fi

if [[ ! -f "${interface_root}/.cross_arf_complete" ]]; then
  arfgen spectrumset="${output_spectrum}" crossreg_spectrumset="${input_spectrum}" \
    crossregionarf=yes arfset="${cross_arf}" withrmfset=yes rmfset="${output_rmf}" \
    detmaptype=dataset detmaparray="${detmap_uniform}" filterdss=yes \
    extendedsource=yes modelee=yes psfmodel=ELLBETA modelootcorr=yes \
    withbadpixcorr=yes badpixlocation="${event_file}" withbadpixres=yes \
    badpixelresolution=1 applyxcaladjustment=no applyabsfluxcorr=no \
    2>&1 | tee "${interface_root}/arfgen_cross.log"
  reject_fatal_log "${interface_root}/arfgen_cross.log"
  [[ -s "${cross_arf}" ]]
  touch "${interface_root}/.cross_arf_complete"
fi

if [[ ! -f "${interface_root}/.central_arf_complete" ]]; then
  arfgen spectrumset="${output_spectrum}" arfset="${central_arf}" \
    withrmfset=yes rmfset="${output_rmf}" detmaptype=dataset \
    detmaparray="${detmap_uniform}" filterdss=yes extendedsource=no modelee=yes \
    withsourcepos=yes sourcecoords=eqpos sourcex=322.4166857950391 \
    sourcey=0.08898852034328401 psfmodel=ELLBETA modelootcorr=yes \
    withbadpixcorr=yes badpixlocation="${event_file}" withbadpixres=yes \
    badpixelresolution=1 applyxcaladjustment=no applyabsfluxcorr=no \
    2>&1 | tee "${interface_root}/arfgen_central.log"
  reject_fatal_log "${interface_root}/arfgen_central.log"
  [[ -s "${central_arf}" ]]
  touch "${interface_root}/.central_arf_complete"
fi

touch "${interface_root}/.interface_complete"
printf 'Frozen X4 pn interface products: %s\n' "${interface_root}"
