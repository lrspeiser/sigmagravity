#!/usr/bin/env bash
# Extract the frozen full-FOV-minus-sources ESAS spectra and corner/FWC products.
set -euo pipefail

analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
background_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b/background
esas_root=${background_root}/esas_full_fov
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

mkdir -p "${esas_root}/MOS1" "${esas_root}/MOS2" "${esas_root}/pn"

validate_mask() {
  local mask_file=$1
  [[ -s "${mask_file}" ]]
  [[ $(ftlist "${mask_file}+1" K include=NAXIS2 | awk '$1 == "NAXIS2" { print $3 }') -eq 87 ]]
}
validate_mask "${background_root}/MOS1_point_source_mask_detector.fits"
validate_mask "${background_root}/MOS1_point_source_mask_sky.fits"
validate_mask "${background_root}/MOS2_point_source_mask_detector.fits"
validate_mask "${background_root}/MOS2_point_source_mask_sky.fits"
validate_mask "${background_root}/pn_point_source_mask_detector.fits"
validate_mask "${background_root}/pn_point_source_mask_sky.fits"

cp -- "${background_root}/MOS1_point_source_mask_detector.fits" "${esas_root}/MOS1/srcdet.fits"
cp -- "${background_root}/MOS1_point_source_mask_sky.fits" "${esas_root}/MOS1/srcsky.fits"
cp -- "${background_root}/MOS2_point_source_mask_detector.fits" "${esas_root}/MOS2/srcdet.fits"
cp -- "${background_root}/MOS2_point_source_mask_sky.fits" "${esas_root}/MOS2/srcsky.fits"
cp -- "${background_root}/pn_point_source_mask_detector.fits" "${esas_root}/pn/srcdet.fits"
cp -- "${background_root}/pn_point_source_mask_sky.fits" "${esas_root}/pn/srcsky.fits"

if [[ ! -f "${esas_root}/MOS1/.mosspectra_complete" ]]; then
  cd "${esas_root}/MOS1"
  mosspectra \
    eventfile="${background_root}/MOS1_corner_preserving_events.ds" \
    cornerfile=mos1S001-corevc.fits \
    imagefile=mos1S001-fovimt.fits expmap=mos1S001-expimt.fits \
    spmask=mos1S001-fovspdet.fits mask=mos1S001-maskimt.fits \
    specfile=mos1S001-fovt.pi rmffile=mos1S001.rmf arffile=mos1S001.arf \
    withregion=no withsrcrem=yes \
    maskdet=srcdet.fits masksky=srcsky.fits \
    pattern=12 keepinterfiles=yes elow=500 ehigh=7000 \
    ccds="T T T T T T T" 2>&1 | tee mosspectra.log
  grep -Fq "DET coord region mask :    srcdet.fits" mosspectra.log
  touch .mosspectra_complete
fi

if [[ ! -f "${esas_root}/MOS2/.mosspectra_complete" ]]; then
  cd "${esas_root}/MOS2"
  mosspectra \
    eventfile="${background_root}/MOS2_corner_preserving_events.ds" \
    cornerfile=mos2S002-corevc.fits \
    imagefile=mos2S002-fovimt.fits expmap=mos2S002-expimt.fits \
    spmask=mos2S002-fovspdet.fits mask=mos2S002-maskimt.fits \
    specfile=mos2S002-fovt.pi rmffile=mos2S002.rmf arffile=mos2S002.arf \
    withregion=no withsrcrem=yes \
    maskdet=srcdet.fits masksky=srcsky.fits \
    pattern=12 keepinterfiles=yes elow=500 ehigh=7000 \
    ccds="T T T T F T T" 2>&1 | tee mosspectra.log
  grep -Fq "DET coord region mask :    srcdet.fits" mosspectra.log
  touch .mosspectra_complete
fi

if [[ ! -f "${esas_root}/pn/.pnspectra_complete" && \
      -s "${esas_root}/pn/pnS003-fovt.pi" && \
      -s "${esas_root}/pn/pnS003-fovt-oot.pi" && \
      -s "${esas_root}/pn/pnS003.rmf" && \
      -s "${esas_root}/pn/pnS003.arf" ]] && \
   grep -Fq "pnspectra analysis complete" "${esas_root}/pn/pnspectra.log" && \
   grep -Fq "DET region file selected also exists" "${esas_root}/pn/pnspectra.log"; then
  touch "${esas_root}/pn/.pnspectra_complete"
fi

if [[ ! -f "${esas_root}/pn/.pnspectra_complete" ]]; then
  cd "${esas_root}/pn"
  pnspectra \
    eventfile="${background_root}/pn_corner_preserving_events.ds" \
    ootevtfile="${background_root}/pn_oot_corner_preserving_events.ds" \
    cornerfile=pnS003-corevc.fits ootcornfile=pnS003-corevc-oot.fits \
    imagefile=pnS003-fovimt.fits ootimgfile=pnS003-fovimt-oot.fits \
    expmap=pnS003-expimt.fits spmask=pnS003-fovspdet.fits \
    mask=pnS003-maskimt.fits specfile=pnS003-fovt.pi \
    ootspecfile=pnS003-fovt-oot.pi rmffile=pnS003.rmf arffile=pnS003.arf \
    withregion=no withsrcrem=yes \
    maskdet=srcdet.fits masksky=srcsky.fits \
    pattern=4 keepinterfiles=yes elow=500 ehigh=7000 \
    quads="T T T T" 2>&1 | tee pnspectra.log
  grep -Fq "DET region file selected also exists" pnspectra.log
  touch .pnspectra_complete
fi

printf 'ESAS full-FOV extraction root: %s\n' "${esas_root}"
