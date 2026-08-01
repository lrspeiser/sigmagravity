#!/usr/bin/env bash
# Acquire the frozen RX J2129 XMM software environment without running XMM tasks.
set -euo pipefail

stage=${1:-}
env_root=/home/henry/.local/share/sigmagravity-xmm
installer_dir=${env_root}/installers
sas_archive=${installer_dir}/sas_22.1.0-a8f2c2afa-20250304-ubuntu24.04-gcc13.3.0-x86_64.tgz
sas_url=https://heasarc.gsfc.nasa.gov/FTP/xmm/software/sas/latest/Linux/Ubuntu24.04/sas_22.1.0-a8f2c2afa-20250304-ubuntu24.04-gcc13.3.0-x86_64.tgz
heasoft_prefix=${env_root}/heasoft-6.36
sas_install_parent=${env_root}/sas-22.1.0
sas_prefix=${sas_install_parent}/xmmsas_22.1.0-a8f2c2afa-20250304
ccf_snapshot=${env_root}/ccf/2026-07-27
mamba_exe=/home/henry/.local/share/sigmagravity-dragons/miniforge3/bin/mamba

mkdir -p "${installer_dir}" "${env_root}/ccf"

case "${stage}" in
  sas-download)
    if [[ ! -f "${sas_archive}" ]]; then
      curl --fail --location --continue-at - --retry 5 --retry-delay 5 \
        --output "${sas_archive}.part" "${sas_url}"
      mv "${sas_archive}.part" "${sas_archive}"
    fi
    sha256sum "${sas_archive}"
    stat --printf='%s bytes\n' "${sas_archive}"
    ;;
  heasoft)
    if [[ ! -x "${heasoft_prefix}/bin/ftversion" ]]; then
      "${mamba_exe}" install --yes --prefix "${heasoft_prefix}" \
        'heasoft=6.36' xspec-data \
        --channel https://heasarc.gsfc.nasa.gov/FTP/software/conda/ \
        --channel conda-forge \
        --strict-channel-priority
    fi
    "${mamba_exe}" list --prefix "${heasoft_prefix}" '^(heasoft|xspec-data)$'
    ;;
  sas-python)
    "${mamba_exe}" install --yes --prefix "${heasoft_prefix}" \
      astropy numpy matplotlib-base requests pyqt beautifultable scipy \
      notebook 'astroquery>=0.4.3' pytest pypdf2 \
      --channel https://heasarc.gsfc.nasa.gov/FTP/software/conda/ \
      --channel conda-forge \
      --strict-channel-priority
    "${heasoft_prefix}/bin/python" -m pip install --no-deps \
      'https://files.pythonhosted.org/packages/c2/cb/b9a20a6479a44eb9bbc2cb313469feed3da94707f325195dee2d6ca491bd/pyds9-1.8.1.tar.gz#sha256=b4f198f5d29b749f721c491f8384f6293e43ec417bd0492be36bffb5c3904b2a'
    ;;
  sas-install)
    if [[ ! -d "${sas_prefix}" ]]; then
      mkdir -p "${sas_install_parent}"
      tar -xzf "${sas_archive}" -C "${sas_install_parent}"
      cd "${sas_install_parent}"
      bash install.sh
    fi
    if [[ ! -f "${sas_prefix}/.configuration_complete" ]]; then
      export SAS_PERL=/usr/bin/perl
      export PATH="${heasoft_prefix}/bin:${PATH}"
      test -x "${heasoft_prefix}/bin/python"
      cd "${sas_prefix}"
      ./configure_install
      touch .configuration_complete
    fi
    export CONDA_PREFIX="${heasoft_prefix}"
    source "${heasoft_prefix}/bin/heainit.sh" >/dev/null 2>&1
    set +u
    source "${sas_prefix}/setsas.sh" >/dev/null 2>&1
    set -u
    sasversion
    ;;
  verify)
    export CONDA_PREFIX="${heasoft_prefix}"
    source "${heasoft_prefix}/bin/heainit.sh"
    env | grep -E '^(HEADAS|LHEASOFT|PATH)='
    set +u
    source "${sas_prefix}/setsas.sh"
    set -u
    command -v ftversion
    ftversion
    command -v sasversion
    sasversion
    ;;
  versions)
    export CONDA_PREFIX="${heasoft_prefix}"
    source "${heasoft_prefix}/bin/heainit.sh" >/dev/null 2>&1
    set +u
    source "${sas_prefix}/setsas.sh" >/dev/null 2>&1
    set -u
    sasversion
    ftversion
    cifbuild -v
    odfingest -v
    emproc -v
    epproc -v
    timeout 20 xspec --version
    "${heasoft_prefix}/bin/python" -c 'import astropy, numpy, matplotlib, requests, pyds9, PyQt5, beautifultable, scipy, notebook, astroquery, pytest, PyPDF2; print("SAS Python imports passed")'
    ;;
  ccf)
    mkdir -p "${ccf_snapshot}"
    cd "${ccf_snapshot}"
    wget -m -nH --no-remove-listing -N -np -r --cut-dirs=4 \
      -e robots=off -l 1 -R 'index.html*' \
      https://heasarc.gsfc.nasa.gov/FTP/xmm/data/CCF/
    touch .mirror_complete
    find . -type f ! -name CCF_MANIFEST.sha256 -print0 \
      | sort -z \
      | xargs -0 sha256sum > CCF_MANIFEST.sha256
    sha256sum CCF_MANIFEST.sha256
    find . -type f ! -name CCF_MANIFEST.sha256 -printf '%s\n' \
      | awk '{bytes += $1; files += 1} END {printf "%d files %d bytes\n", files, bytes}'
    ;;
  *)
    printf 'usage: %s {sas-download|heasoft|sas-python|sas-install|verify|versions|ccf}\n' "$0" >&2
    exit 2
    ;;
esac
