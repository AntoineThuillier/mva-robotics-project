#!/bin/bash
# One-time setup of the Python environment on Narval (Compute Canada / Alliance).
# Run it on a LOGIN node (compute nodes have no internet), from the repo root:
#     bash scripts/narval/setup_env.sh
set -e

VENV=${VENV:-$HOME/venvs/amp}

module load StdEnv/2023 python/3.11
virtualenv --no-download "$VENV"
source "$VENV/bin/activate"
pip install --no-index --upgrade pip

# numpy and torch: from the Alliance wheelhouse (built for the cluster).
pip install --no-index numpy torch

# mujoco: wheelhouse if available, otherwise PyPI.
pip install --no-index mujoco || pip install mujoco

python -c "import numpy, torch, mujoco; print('OK: numpy', numpy.__version__, '| torch', torch.__version__, '| mujoco', mujoco.__version__)"
echo ">>> Environment ready in $VENV"
