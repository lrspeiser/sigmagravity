#!/usr/bin/env bash
# Extract the frozen local outer annulus and build its ESAS QPB spectra.
set -euo pipefail

analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
x2b_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b
background_root=${x2b_root}/background
full_fov_root=${background_root}/esas_full_fov
outer_root=${background_root}/esas_outer_annulus
heasoft_prefix=/home/henry/.local/share/sigmagravity-xmm/heasoft-6.36
sas_prefix=/home/henry/.local/share/sigmagravity-xmm/sas-22.1.0/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=/home/henry/.local/share/sigmagravity-xmm/ccf/2026-07-27

center_ra=322.41651
center_dec=0.08923
inner_radius_det=3481.26779527
outer_radius_det=4820.21694730

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

[[ -f "${full_fov_root}/MOS2/.mosback_complete" ]]
[[ -f "${full_fov_root}/pn/.pnback_complete" ]]
mkdir -p "${outer_root}/MOS2" "${outer_root}/pn"

prepare_region() {
  local directory=$1
  local instrument=$2
  local image=$3
  local source_prefix=$4
  local detxy detx dety
  detxy=$(esky2det datastyle=user ra="${center_ra}" dec="${center_dec}" \
    instrument="${instrument}" checkfov=no outunit=det withheader=no \
    calinfostyle=set calinfoset="${image}" verbosity=0 \
    2>"${directory}/outer_annulus_esky2det.log")
  read -r detx dety <<<"${detxy}"
  [[ "${detx}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]
  [[ "${dety}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]
  printf '%s\n' \
    "&&((DETX,DETY) IN circle(${detx},${dety},${outer_radius_det}))&&!((DETX,DETY) IN circle(${detx},${dety},${inner_radius_det}))" \
    >"${directory}/outer_region.txt"
  [[ $(wc -l <"${directory}/outer_region.txt") -eq 1 ]]
  [[ $(grep -o 'circle(' "${directory}/outer_region.txt" | wc -l) -eq 2 ]]
  cp -- "${background_root}/${source_prefix}_point_source_mask_detector.fits" \
    "${directory}/srcdet.fits"
  cp -- "${background_root}/${source_prefix}_point_source_mask_sky.fits" \
    "${directory}/srcsky.fits"
  [[ $(ftlist "${directory}/srcdet.fits+1" K include=NAXIS2 | \
    awk '$1 == "NAXIS2" { print $3 }') -eq 87 ]]
  [[ $(ftlist "${directory}/srcsky.fits+1" K include=NAXIS2 | \
    awk '$1 == "NAXIS2" { print $3 }') -eq 87 ]]
}

prepare_region "${outer_root}/MOS2" EMOS2 \
  "${x2b_root}/detect_band2_1200_2000eV/mos2S002-fovimt.fits" MOS2
prepare_region "${outer_root}/pn" EPN \
  "${x2b_root}/detect_band2_1200_2000eV/pnS003-fovimt.fits" pn

if [[ ! -f "${outer_root}/MOS2/.mosspectra_outer_complete" ]]; then
  cd "${outer_root}/MOS2"
  mosspectra \
    eventfile="${background_root}/MOS2_corner_preserving_events.ds" \
    cornerfile=mos2S002-corevc.fits \
    imagefile=mos2S002-fovimt.fits expmap=mos2S002-expimt.fits \
    spmask=mos2S002-fovspdet.fits mask=mos2S002-maskimt.fits \
    specfile=mos2S002-fovt.pi rmffile=mos2S002.rmf arffile=mos2S002.arf \
    withregion=yes regionfile=outer_region.txt withsrcrem=yes \
    maskdet=srcdet.fits masksky=srcsky.fits \
    pattern=12 keepinterfiles=yes elow=500 ehigh=7000 \
    ccds="T T T T F T T" 2>&1 | tee mosspectra.log
  grep -Fq "DET coord region mask :    srcdet.fits" mosspectra.log
  grep -Fq "outer_region.txt" mosspectra.log
  grep -Fq "mosspectra analysis complete" mosspectra.log
  touch .mosspectra_outer_complete
fi

if [[ ! -f "${outer_root}/MOS2/.mosback_outer_complete" ]]; then
  cd "${outer_root}/MOS2"
  mosback inspecfile=mos2S002-fovt.pi outspecfile=mos2S002-bkg.pi \
    rmffile=mos2S002.rmf withplotfiles=yes \
    inimgfile=mos2S002-fovimdet-500-7000.fits \
    outimgfile=mos2S002-bkgimdet-500-7000.fits \
    elow=500 ehigh=7000 ccds="T T T T F T T" 2>&1 | tee mosback.log
  [[ -s mos2S002-bkg.pi && -s mos2S002-bkgimdet-500-7000.fits ]]
  grep -Eq "mosback .* ended:" mosback.log
  touch .mosback_outer_complete
fi

if [[ ! -f "${outer_root}/pn/.pnspectra_outer_complete" ]]; then
  cd "${outer_root}/pn"
  pnspectra \
    eventfile="${background_root}/pn_corner_preserving_events.ds" \
    ootevtfile="${background_root}/pn_oot_corner_preserving_events.ds" \
    cornerfile=pnS003-corevc.fits ootcornfile=pnS003-corevc-oot.fits \
    imagefile=pnS003-fovimt.fits ootimgfile=pnS003-fovimt-oot.fits \
    expmap=pnS003-expimt.fits spmask=pnS003-fovspdet.fits \
    mask=pnS003-maskimt.fits specfile=pnS003-fovt.pi \
    ootspecfile=pnS003-fovt-oot.pi rmffile=pnS003.rmf arffile=pnS003.arf \
    withregion=yes regionfile=outer_region.txt withsrcrem=yes \
    maskdet=srcdet.fits masksky=srcsky.fits \
    pattern=4 keepinterfiles=yes elow=500 ehigh=7000 \
    quads="T T T T" 2>&1 | tee pnspectra.log
  grep -Fq "DET region file selected also exists" pnspectra.log
  grep -Fq "outer_region.txt" pnspectra.log
  grep -Fq "pnspectra analysis complete" pnspectra.log
  touch .pnspectra_outer_complete
fi

if [[ ! -f "${outer_root}/pn/.pnback_outer_complete" ]]; then
  cd "${outer_root}/pn"
  pnback inspecfile=pnS003-fovt.pi inspecoot=pnS003-fovt-oot.pi \
    outspecfile=pnS003-bkg.pi outspecoot=pnS003-bkg-oot.pi \
    rmffile=pnS003.rmf withplotfiles=yes \
    inimgfile=pnS003-fovimdet-500-7000.fits \
    inimgoot=pnS003-fovimootdet-500-7000.fits \
    outimgfile=pnS003-bkgimdet-500-7000.fits \
    elow=500 ehigh=7000 quads="T T T T" 2>&1 | tee pnback.log
  [[ -s pnS003-bkg.pi && -s pnS003-bkgimdet-500-7000.fits ]]
  grep -Eq "pnback .* ended:" pnback.log
  touch .pnback_outer_complete
fi

printf 'Frozen local outer-annulus ESAS root: %s\n' "${outer_root}"
