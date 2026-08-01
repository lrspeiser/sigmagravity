#!/usr/bin/env bash
# Build ESAS quiescent-particle background spectra and detector images.
set -euo pipefail

analysis_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/analysis
esas_root=/home/henry/.local/share/sigmagravity-xmm/work/rxj2129/0093030201/x2b/background/esas_full_fov
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

for instrument in MOS1 MOS2 pn; do
  [[ -f "${esas_root}/${instrument}/.${instrument,,}spectra_complete" ]] || {
    if [[ "${instrument}" == pn ]]; then
      [[ -f "${esas_root}/pn/.pnspectra_complete" ]]
    else
      [[ -f "${esas_root}/${instrument}/.mosspectra_complete" ]]
    fi
  }
done

if [[ ! -f "${esas_root}/MOS1/.mosback_complete" ]]; then
  cd "${esas_root}/MOS1"
  mosback inspecfile=mos1S001-fovt.pi outspecfile=mos1S001-bkg.pi \
    rmffile=mos1S001.rmf withplotfiles=yes \
    inimgfile=mos1S001-fovimdet-500-7000.fits \
    outimgfile=mos1S001-bkgimdet-500-7000.fits \
    elow=500 ehigh=7000 ccds="T T T T T T T" 2>&1 | tee mosback.log
  [[ -s mos1S001-bkg.pi && -s mos1S001-bkgimdet-500-7000.fits ]]
  touch .mosback_complete
fi

if [[ ! -f "${esas_root}/MOS2/.mosback_complete" ]]; then
  cd "${esas_root}/MOS2"
  mosback inspecfile=mos2S002-fovt.pi outspecfile=mos2S002-bkg.pi \
    rmffile=mos2S002.rmf withplotfiles=yes \
    inimgfile=mos2S002-fovimdet-500-7000.fits \
    outimgfile=mos2S002-bkgimdet-500-7000.fits \
    elow=500 ehigh=7000 ccds="T T T T F T T" 2>&1 | tee mosback.log
  [[ -s mos2S002-bkg.pi && -s mos2S002-bkgimdet-500-7000.fits ]]
  touch .mosback_complete
fi

if [[ ! -f "${esas_root}/pn/.pnback_complete" ]]; then
  cd "${esas_root}/pn"
  pnback inspecfile=pnS003-fovt.pi inspecoot=pnS003-fovt-oot.pi \
    outspecfile=pnS003-bkg.pi outspecoot=pnS003-bkg-oot.pi \
    rmffile=pnS003.rmf withplotfiles=yes \
    inimgfile=pnS003-fovimdet-500-7000.fits \
    inimgoot=pnS003-fovimootdet-500-7000.fits \
    outimgfile=pnS003-bkgimdet-500-7000.fits \
    elow=500 ehigh=7000 quads="T T T T" 2>&1 | tee pnback.log
  [[ -s pnS003-bkg.pi && -s pnS003-bkgimdet-500-7000.fits ]]
  touch .pnback_complete
fi

printf 'ESAS QPB root: %s\n' "${esas_root}"
