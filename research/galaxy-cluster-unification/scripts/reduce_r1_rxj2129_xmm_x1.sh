#!/usr/bin/env bash
# Execute only the frozen RX J2129 XMM X1 calibration stage.
set -euo pipefail

repo_root=/mnt/c/Users/henry/Documents/Codex/2026-07-18/sigmagravity-frontiers-main/research/galaxy-cluster-unification
source_odf=${repo_root}/data/raw/r1_rxj2129_xmm/0093030201/ODF
work_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201
staged_odf=${work_root}/ODF
analysis_root=${work_root}/analysis
heasoft_prefix=/home/henry/.local/share/sigmagravity-xmm/heasoft-6.36
sas_prefix=/home/henry/.local/share/sigmagravity-xmm/sas-22.1.0/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=/home/henry/.local/share/sigmagravity-xmm/ccf/2026-07-27

mkdir -p "${staged_odf}" "${analysis_root}"

if [[ ! -f "${staged_odf}/.staging_complete" ]]; then
  while IFS= read -r -d '' source_file; do
    source_name=$(basename "${source_file}")
    if [[ "${source_name}" == *.gz ]]; then
      gzip -t "${source_file}"
      target_file=${staged_odf}/${source_name%.gz}
      gzip -cd "${source_file}" > "${target_file}.part"
      mv "${target_file}.part" "${target_file}"
    else
      cp --preserve=timestamps "${source_file}" "${staged_odf}/${source_name}"
    fi
  done < <(find "${source_odf}" -maxdepth 1 -type f -print0 | sort -z)
  source_count=$(find "${source_odf}" -maxdepth 1 -type f | wc -l)
  staged_count=$(find "${staged_odf}" -maxdepth 1 -type f ! -name '*.part' | wc -l)
  [[ "${source_count}" -eq 284 ]]
  [[ "${staged_count}" -eq 284 ]]
  find "${staged_odf}" -maxdepth 1 -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > "${work_root}/staged_ODF.sha256"
  touch "${staged_odf}/.staging_complete"
fi

export CONDA_PREFIX="${heasoft_prefix}"
source "${heasoft_prefix}/bin/heainit.sh" >/dev/null 2>&1
set +u
source "${sas_prefix}/setsas.sh" >/dev/null 2>&1
set -u
export SAS_CCFPATH="${ccf_snapshot}"
export SAS_CCF="${analysis_root}/ccf.cif"
export SAS_ODF="${staged_odf}"
export SAS_VERBOSITY=4
export SAS_SUPPRESS_WARNING=3
export ANALYSIS_ROOT="${analysis_root}"

cd "${analysis_root}"
if [[ ! -f .cifbuild_complete ]]; then
  cifbuild withccfpath=no analysisdate=2026-07-27 category=XMMCCF \
    calindexset="${SAS_CCF}" fullpath=yes 2>&1 | tee cifbuild.log
  test -s "${SAS_CCF}"
  touch .cifbuild_complete
fi

if [[ ! -f .odfingest_complete ]]; then
  odfingest odfdir="${staged_odf}" withodfdir=yes outdir="${analysis_root}" \
    usecanonicalname=yes writepath=yes usehousekeeping=yes \
    findinstrumentmodes=yes oalcheck=yes 2>&1 | tee odfingest.log
  mapfile -t summary_files < <(find "${analysis_root}" -maxdepth 1 -type f -name '*SUM.SAS' -print)
  [[ "${#summary_files[@]}" -eq 1 ]]
  touch .odfingest_complete
fi

mapfile -t summary_files < <(find "${analysis_root}" -maxdepth 1 -type f -name '*SUM.SAS' -print)
[[ "${#summary_files[@]}" -eq 1 ]]
export SAS_ODF="${summary_files[0]}"

if [[ ! -f .emproc_complete ]]; then
  emproc 2>&1 | tee emproc.log
  touch .emproc_complete
fi

if [[ ! -f .epproc_oot_complete ]]; then
  if [[ ! -f .epproc_complete ]]; then
    epproc withoutoftime=yes 2>&1 | tee epproc_oot.log
    touch .epproc_complete
  fi
  mapfile -t pn_oot_source < <(find "${analysis_root}" -maxdepth 1 -type f \
    -name '0529_0093030201_EPN_S003_ImagingEvts.ds' -print)
  [[ "${#pn_oot_source[@]}" -eq 1 ]]
  pn_oot_target="${analysis_root}/0529_0093030201_EPN_S003_OutOfTimeEvts.ds"
  cp --preserve=timestamps --reflink=auto "${pn_oot_source[0]}" "${pn_oot_target}.part"
  mv "${pn_oot_target}.part" "${pn_oot_target}"
  sha256sum "${pn_oot_target}" > "${analysis_root}/pn_oot.sha256"
  touch .epproc_oot_complete
fi

if [[ ! -f .epproc_normal_complete ]]; then
  epproc withoutoftime=no 2>&1 | tee epproc_normal.log
  test -s "${analysis_root}/0529_0093030201_EPN_S003_ImagingEvts.ds"
  test -s "${analysis_root}/0529_0093030201_EPN_S003_OutOfTimeEvts.ds"
  touch .epproc_normal_complete
fi

sasversion
printf 'X1 analysis root: %s\n' "${analysis_root}"
