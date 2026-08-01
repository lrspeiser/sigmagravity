#!/usr/bin/env bash
# Run only the frozen RX J2129 X2b1 point-source detection tasks.
set -euo pipefail

repo_project=/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification
analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
x2b_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b
heasoft_prefix=/home/henry/.local/share/sigmagravity-xmm/heasoft-6.36
sas_prefix=/home/henry/.local/share/sigmagravity-xmm/sas-22.1.0/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=/home/henry/.local/share/sigmagravity-xmm/ccf/2026-07-27
pn_gti=${repo_project}/data/derived/r1_rxj2129_xmm_x2/pn_flare_gti.fits

test -f "${analysis_root}/.x2a_clean_events_complete"
test -f "${repo_project}/configs/r1_rxj2129_xmm_background_mask_protocol.json"
mkdir -p "${x2b_root}"

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

cd "${analysis_root}"
if [[ ! -f .x2b_pn_oot_clean_complete ]]; then
  evselect table=0529_0093030201_EPN_S003_OutOfTimeEvts.ds:EVENTS \
    filteredset=pn_oot_clean_events.ds withfilteredset=yes keepfilteroutput=yes \
    filtertype=expression filterexposure=yes updateexposure=yes \
    expression="(FLAG==0) && (PATTERN<=4) && (PI in [300:10000]) && GTI(${pn_gti},TIME)" \
    2>&1 | tee x2b_pn_oot_clean.log
  touch .x2b_pn_oot_clean_complete
fi

bands=("500 1200" "1200 2000" "2000 7000")
band_index=0
for band in "${bands[@]}"; do
  band_index=$((band_index + 1))
  read -r elow ehigh <<<"${band}"
  band_root=${x2b_root}/detect_band${band_index}_${elow}_${ehigh}eV
  mkdir -p "${band_root}"
  if [[ ! -f "${band_root}/.cheese_complete" ]]; then
    cd "${band_root}"
    cheese \
      mos1file="${analysis_root}/MOS1_clean_events.ds" \
      mos2file="${analysis_root}/MOS2_clean_events.ds" \
      pnfile="${analysis_root}/pn_clean_events.ds" \
      pnootfile="${analysis_root}/pn_oot_clean_events.ds" \
      elowlist="${elow}" ehighlist="${ehigh}" \
      scale=0.8 mlmin=10 ratetotal=1000 dist=0 keepinterfiles=yes \
      2>&1 | tee cheese.log
    test -s emllist.fits
    touch .cheese_complete
  fi
done

printf 'X2b detection root: %s\n' "${x2b_root}"
