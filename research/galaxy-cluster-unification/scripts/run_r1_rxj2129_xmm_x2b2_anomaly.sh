#!/usr/bin/env bash
# Run only the frozen RX J2129 MOS anomalous-CCD gate on working copies.
set -euo pipefail

analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
background_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b/background
heasoft_prefix=/home/henry/.local/share/sigmagravity-xmm/heasoft-6.36
sas_prefix=/home/henry/.local/share/sigmagravity-xmm/sas-22.1.0/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=/home/henry/.local/share/sigmagravity-xmm/ccf/2026-07-27

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
if [[ ! -f .working_copies_complete ]]; then
  cp --preserve=timestamps --reflink=auto "${analysis_root}/MOS1_clean_events.ds" MOS1_background_work.ds.part
  mv MOS1_background_work.ds.part MOS1_background_work.ds
  cp --preserve=timestamps --reflink=auto "${analysis_root}/MOS2_clean_events.ds" MOS2_background_work.ds.part
  mv MOS2_background_work.ds.part MOS2_background_work.ds
  touch .working_copies_complete
fi

if [[ ! -f .MOS1_emanom_complete ]]; then
  emanom eventfile=MOS1_background_work.ds cornerfile=MOS1_corner_events.ds \
    writekeys=yes writelog=yes keepcorner=yes 2>&1 | tee MOS1_emanom_task.log
  touch .MOS1_emanom_complete
fi

if [[ ! -f .MOS2_emanom_complete ]]; then
  emanom eventfile=MOS2_background_work.ds cornerfile=MOS2_corner_events.ds \
    writekeys=yes writelog=yes keepcorner=yes 2>&1 | tee MOS2_emanom_task.log
  touch .MOS2_emanom_complete
fi

printf 'X2b2 anomaly root: %s\n' "${background_root}"
