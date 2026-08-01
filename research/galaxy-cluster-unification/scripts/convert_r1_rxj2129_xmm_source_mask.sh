#!/usr/bin/env bash
# Convert the immutable source mask into each EPIC detector coordinate frame.
set -euo pipefail

repo_project=/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification
x2b_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b
background_root=${x2b_root}/background
analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
heasoft_prefix=/home/henry/.local/share/sigmagravity-xmm/heasoft-6.36
sas_prefix=/home/henry/.local/share/sigmagravity-xmm/sas-22.1.0/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=/home/henry/.local/share/sigmagravity-xmm/ccf/2026-07-27
sky_mask=${repo_project}/data/derived/r1_rxj2129_xmm_x2/point_source_mask_convregion_sky.txt

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

"${heasoft_prefix}/bin/python" "${repo_project}/scripts/write_r1_rxj2129_xmm_sky_mask_ascii.py"
cd "${background_root}"

# SAS 22.1.0 convregion-0.12.2 invokes esky2det through /bin/sh with the
# non-POSIX redirect `>& detpos.txt`; Ubuntu's /bin/sh is dash, so mode=2
# aborts before writing a coordinate. Reproduce its documented circle
# algorithm directly: esky2det for each centre and 0.05-arcsec DET pixels for
# each radius (one arcminute = 1200 DET units). The immutable sky inputs and
# image/calibration frames are unchanged.
convert_circles() {
  local instrument=$1
  local image=$2
  local output=$3
  local log=$4
  local exclusion ra dec radius_arcmin detxy detx dety radius_det
  : >"${output}"
  : >"${log}"
  while read -r exclusion ra dec radius_arcmin; do
    [[ "${exclusion}" == "!CIRCLE" ]] || {
      printf 'Unexpected mask record: %s %s %s %s\n' \
        "${exclusion}" "${ra}" "${dec}" "${radius_arcmin}" >&2
      return 1
    }
    detxy=$(esky2det datastyle=user ra="${ra}" dec="${dec}" \
      instrument="${instrument}" checkfov=no outunit=det withheader=no \
      calinfostyle=set calinfoset="${image}" verbosity=0 2>>"${log}")
    read -r detx dety <<<"${detxy}"
    [[ "${detx}" =~ ^-?[0-9]+([.][0-9]+)?$ && \
       "${dety}" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || {
      printf 'No finite detector coordinate for %s %s (%s)\n' \
        "${ra}" "${dec}" "${instrument}" >&2
      return 1
    }
    radius_det=$(awk -v radius="${radius_arcmin}" \
      'BEGIN { printf "%.4f", radius * 1200.0 }')
    printf '!CIRCLE %.4f %.4f %s\n' \
      "${detx}" "${dety}" "${radius_det}" >>"${output}"
  done <"${sky_mask}"
  [[ $(wc -l <"${output}") -eq 87 ]] || {
    printf 'Expected 87 detector exclusions in %s\n' "${output}" >&2
    return 1
  }
}

convert_circles EMOS1 \
  "${x2b_root}/detect_band2_1200_2000eV/mos1S001-fovimt.fits" \
  MOS1_point_source_mask_detector.txt MOS1_esky2det.log
convert_circles EMOS2 \
  "${x2b_root}/detect_band2_1200_2000eV/mos2S002-fovimt.fits" \
  MOS2_point_source_mask_detector.txt MOS2_esky2det.log
convert_circles EPN \
  "${x2b_root}/detect_band2_1200_2000eV/pnS003-fovimt.fits" \
  pn_point_source_mask_detector.txt pn_esky2det.log

convert_sky_circles() {
  local image=$1
  local output=$2
  local log=$3
  local exclusion ra dec radius_arcmin conversion xy x y radius_xy
  : >"${output}"
  : >"${log}"
  while read -r exclusion ra dec radius_arcmin; do
    [[ "${exclusion}" == "!CIRCLE" ]] || return 1
    conversion=$(ecoordconv imageset="${image}" withcoords=yes coordtype=EQPOS \
      x="${ra}" y="${dec}" verbosity=0 2>&1)
    printf '%s\n' "${conversion}" >>"${log}"
    xy=$(printf '%s\n' "${conversion}" | \
      awk '/^[[:space:]]*X: Y:/ { print $3, $4; exit }')
    read -r x y <<<"${xy}"
    [[ "${x}" =~ ^-?[0-9]+([.][0-9]+)?$ && \
       "${y}" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || {
      printf 'No finite sky X/Y coordinate for %s %s\n' "${ra}" "${dec}" >&2
      return 1
    }
    radius_xy=$(awk -v radius="${radius_arcmin}" \
      'BEGIN { printf "%.4f", radius * 1200.0 }')
    printf '!CIRCLE %.4f %.4f %s\n' \
      "${x}" "${y}" "${radius_xy}" >>"${output}"
  done <"${sky_mask}"
  [[ $(wc -l <"${output}") -eq 87 ]]
}

convert_sky_circles \
  "${x2b_root}/detect_band2_1200_2000eV/mos1S001-fovimt.fits" \
  MOS1_point_source_mask_sky_xy.txt MOS1_ecoordconv.log
convert_sky_circles \
  "${x2b_root}/detect_band2_1200_2000eV/mos2S002-fovimt.fits" \
  MOS2_point_source_mask_sky_xy.txt MOS2_ecoordconv.log
convert_sky_circles \
  "${x2b_root}/detect_band2_1200_2000eV/pnS003-fovimt.fits" \
  pn_point_source_mask_sky_xy.txt pn_ecoordconv.log

write_region_expression() {
  local detector_mask=$1
  local region_file=$2
  local exclusion detx dety radius_det
  : >"${region_file}"
  while read -r exclusion detx dety radius_det; do
    [[ "${exclusion}" == "!CIRCLE" ]] || return 1
    printf '&&!((DETX,DETY) IN circle(%s,%s,%s))' \
      "${detx}" "${dety}" "${radius_det}" >>"${region_file}"
  done <"${detector_mask}"
  printf '\n' >>"${region_file}"
  [[ $(grep -o 'circle(' "${region_file}" | wc -l) -eq 87 ]]
  [[ $(wc -l <"${region_file}") -eq 1 ]]
}

write_region_expression MOS1_point_source_mask_detector.txt MOS1_point_source_exclusions.txt
write_region_expression MOS2_point_source_mask_detector.txt MOS2_point_source_exclusions.txt
write_region_expression pn_point_source_mask_detector.txt pn_point_source_exclusions.txt

"${heasoft_prefix}/bin/python" \
  "${repo_project}/scripts/build_r1_rxj2129_xmm_fits_source_masks.py" \
  --background-root "${background_root}"

printf 'Detector source masks: %s\n' "${background_root}"
