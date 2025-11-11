#!/usr/bin/env bash
set -euo pipefail
ENV_NAME=${1:-rc-pca}
conda create -n "${ENV_NAME}" python=3.11 -y
conda activate "${ENV_NAME}"
pip install -r requirements.txt
echo "[OK] Environment '${ENV_NAME}' is ready. For GPU autoencoder, install torch with CUDA per pytorch.org."
