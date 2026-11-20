#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$ROOT"
CONDA=${CONDA_EXE:-$HOME/miniconda3/bin/conda}
if [[ ! -x "$NEXARM_PYTHON" ]]; then
    "$CONDA" env create --prefix "$NEXARM_CONDA_PREFIX" --file environment.yml --yes
fi
"$NEXARM_PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"$NEXARM_PYTHON" -m pip install -r requirements/act.txt ruff
"$NEXARM_PYTHON" -m pip install --no-deps lerobot==0.4.3
"$NEXARM_PYTHON" -m pip install --no-deps -e .
"$NEXARM_PYTHON" -c 'import torch; print(torch.__version__, torch.cuda.get_device_name()); print((torch.randn(32,32,device="cuda") @ torch.randn(32,32,device="cuda")).shape)'
