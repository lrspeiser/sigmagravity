#!/usr/bin/env bash
# Build corner-preserving background lists and run the valid MOS anomaly gate.
set -euo pipefail

repo_project=/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification
analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
background_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b/background
heasoft_prefix=/home/henry/.local/share/sigmagravity-xmm/heasoft-6.36
sas_prefix=/home/henry/.local/share/sigmagravity-xmm/sas-22.1.0/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=/home/henry/.local/share/sigmagravity-xmm/ccf/2026-07-27
gti_root=${repo_project}/data/derived/r1_rxj2129_xmm_x2

mkdir -p "${background_root}"
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

cd "${background_root}"
if [[ ! -f .corner_preserving_events_complete ]]; then
  evselect table="${analysis_root}/0529_0093030201_EMOS1_S001_ImagingEvts.ds:EVENTS" \
    filteredset=MOS1_corner_preserving_events.ds withfilteredset=yes \
    keepfilteroutput=yes filtertype=expression filterexposure=yes updateexposure=yes \
    expression="GTI(${gti_root}/MOS1_flare_gti.fits,TIME)" \
    2>&1 | tee MOS1_corner_preserving_evselect.log
  evselect table="${analysis_root}/0529_0093030201_EMOS2_S002_ImagingEvts.ds:EVENTS" \
    filteredset=MOS2_corner_preserving_events.ds withfilteredset=yes \
    keepfilteroutput=yes filtertype=expression filterexposure=yes updateexposure=yes \
    expression="GTI(${gti_root}/MOS2_flare_gti.fits,TIME)" \
    2>&1 | tee MOS2_corner_preserving_evselect.log
  evselect table="${analysis_root}/0529_0093030201_EPN_S003_ImagingEvts.ds:EVENTS" \
    filteredset=pn_corner_preserving_events.ds withfilteredset=yes \
    keepfilteroutput=yes filtertype=expression filterexposure=yes updateexposure=yes \
    expression="GTI(${gti_root}/pn_flare_gti.fits,TIME)" \
    2>&1 | tee pn_corner_preserving_evselect.log
  evselect table="${analysis_root}/0529_0093030201_EPN_S003_OutOfTimeEvts.ds:EVENTS" \
    filteredset=pn_oot_corner_preserving_events.ds withfilteredset=yes \
    keepfilteroutput=yes filtertype=expression filterexposure=yes updateexposure=yes \
    expression="GTI(${gti_root}/pn_flare_gti.fits,TIME)" \
    2>&1 | tee pn_oot_corner_preserving_evselect.log
  touch .corner_preserving_events_complete
fi

if [[ ! -f .MOS1_corner_emanom_complete ]]; then
  emanom eventfile=MOS1_corner_preserving_events.ds \
    cornerfile=MOS1_valid_corner_events.ds writekeys=yes writelog=yes keepcorner=yes \
    2>&1 | tee MOS1_corner_emanom_task.log
  touch .MOS1_corner_emanom_complete
fi

if [[ ! -f .MOS2_corner_emanom_complete ]]; then
  emanom eventfile=MOS2_corner_preserving_events.ds \
    cornerfile=MOS2_valid_corner_events.ds writekeys=yes writelog=yes keepcorner=yes \
    2>&1 | tee MOS2_corner_emanom_task.log
  touch .MOS2_corner_emanom_complete
fi

printf 'X2b2 corner-preserving root: %s\n' "${background_root}"
