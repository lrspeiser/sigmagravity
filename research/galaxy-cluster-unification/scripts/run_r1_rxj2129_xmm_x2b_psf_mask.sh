#!/usr/bin/env bash
# Run only the frozen RX J2129 X2b1 PSF-radius and immutable-mask stage.
set -euo pipefail

repo_project=/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification
analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
heasoft_prefix=/home/henry/.local/share/sigmagravity-xmm/heasoft-6.36
sas_prefix=/home/henry/.local/share/sigmagravity-xmm/sas-22.1.0/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=/home/henry/.local/share/sigmagravity-xmm/ccf/2026-07-27

test -f "${repo_project}/data/derived/r1_rxj2129_xmm_x2/point_source_catalog_pre_psf.csv"
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

"${heasoft_prefix}/bin/python" \
  "${repo_project}/scripts/build_r1_rxj2129_xmm_point_source_mask.py"
