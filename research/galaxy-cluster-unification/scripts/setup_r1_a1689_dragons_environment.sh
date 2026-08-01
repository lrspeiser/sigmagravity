#!/usr/bin/env bash
set -euo pipefail

sg_runtime=/home/henry/.local/share/sigmagravity-dragons
sg_conda="$sg_runtime/miniforge3/bin/conda"
sg_installer="$sg_runtime/Miniforge3-Linux-x86_64.sh"

if [[ ! -x "$sg_conda" ]]; then
  mkdir -p "$sg_runtime"
  curl -L --fail --retry 3 \
    -o "$sg_installer" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$sg_installer" -b -p "$sg_runtime/miniforge3"
fi

if ! "$sg_conda" env list | grep -q '/dragons-4.2.2$'; then
  "$sg_conda" create -y -n dragons-4.2.2 \
    -c http://astroconda.gemini.edu/public \
    -c conda-forge \
    python=3.12 numpy=1.26 dragons=4.2.2
fi

"$sg_conda" run -n dragons-4.2.2 python -m pip install --disable-pip-version-check ppxf==9.4.8
