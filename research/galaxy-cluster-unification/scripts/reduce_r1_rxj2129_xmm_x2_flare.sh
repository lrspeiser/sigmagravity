#!/usr/bin/env bash
# Execute the frozen RX J2129 XMM X2a flare-filter stage only.
set -euo pipefail

repo_project=/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification
analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
heasoft_prefix=/home/henry/.local/share/sigmagravity-xmm/heasoft-6.36
sas_prefix=/home/henry/.local/share/sigmagravity-xmm/sas-22.1.0/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=/home/henry/.local/share/sigmagravity-xmm/ccf/2026-07-27
derived_root=${repo_project}/data/derived/r1_rxj2129_xmm_x2

test -f "${analysis_root}/.epproc_normal_complete"
test -f "${analysis_root}/.epproc_oot_complete"
mkdir -p "${derived_root}"

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
timemin=152241400
timemax=152300200

if [[ ! -f .x2a_rate_curves_complete ]]; then
  evselect table=0529_0093030201_EMOS1_S001_ImagingEvts.ds:EVENTS \
    withrateset=yes rateset=MOS1_high_energy_rate.ds timecolumn=TIME \
    timebinsize=100 withtimeranges=yes timemin=${timemin} timemax=${timemax} \
    maketimecolumn=yes makeratecolumn=yes \
    expression='(#XMMEA_EM) && (PATTERN==0) && (PI in [10000:12000])' \
    2>&1 | tee x2a_MOS1_rate.log
  evselect table=0529_0093030201_EMOS2_S002_ImagingEvts.ds:EVENTS \
    withrateset=yes rateset=MOS2_high_energy_rate.ds timecolumn=TIME \
    timebinsize=100 withtimeranges=yes timemin=${timemin} timemax=${timemax} \
    maketimecolumn=yes makeratecolumn=yes \
    expression='(#XMMEA_EM) && (PATTERN==0) && (PI in [10000:12000])' \
    2>&1 | tee x2a_MOS2_rate.log
  evselect table=0529_0093030201_EPN_S003_ImagingEvts.ds:EVENTS \
    withrateset=yes rateset=pn_high_energy_rate.ds timecolumn=TIME \
    timebinsize=100 withtimeranges=yes timemin=${timemin} timemax=${timemax} \
    maketimecolumn=yes makeratecolumn=yes \
    expression='(FLAG==0) && (PATTERN==0) && (PI in [12000:14000])' \
    2>&1 | tee x2a_pn_rate.log
  touch .x2a_rate_curves_complete
fi

if [[ ! -f .x2a_gti_complete ]]; then
  "${heasoft_prefix}/bin/python" \
    "${repo_project}/scripts/derive_r1_rxj2129_xmm_flare_gtis.py"
  touch .x2a_gti_complete
fi

if [[ ! -f .x2a_clean_events_complete ]]; then
  evselect table=0529_0093030201_EMOS1_S001_ImagingEvts.ds:EVENTS \
    filteredset=MOS1_clean_events.ds withfilteredset=yes keepfilteroutput=yes \
    filtertype=expression filterexposure=yes updateexposure=yes \
    expression="(#XMMEA_EM) && (PATTERN<=12) && (PI in [300:10000]) && GTI(${derived_root}/MOS1_flare_gti.fits,TIME)" \
    2>&1 | tee x2a_MOS1_clean.log
  evselect table=0529_0093030201_EMOS2_S002_ImagingEvts.ds:EVENTS \
    filteredset=MOS2_clean_events.ds withfilteredset=yes keepfilteroutput=yes \
    filtertype=expression filterexposure=yes updateexposure=yes \
    expression="(#XMMEA_EM) && (PATTERN<=12) && (PI in [300:10000]) && GTI(${derived_root}/MOS2_flare_gti.fits,TIME)" \
    2>&1 | tee x2a_MOS2_clean.log
  evselect table=0529_0093030201_EPN_S003_ImagingEvts.ds:EVENTS \
    filteredset=pn_clean_events.ds withfilteredset=yes keepfilteroutput=yes \
    filtertype=expression filterexposure=yes updateexposure=yes \
    expression="(FLAG==0) && (PATTERN<=4) && (PI in [300:10000]) && GTI(${derived_root}/pn_flare_gti.fits,TIME)" \
    2>&1 | tee x2a_pn_clean.log
  touch .x2a_clean_events_complete
fi

printf 'X2a analysis root: %s\n' "${analysis_root}"
