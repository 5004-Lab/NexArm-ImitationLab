# Shared shell environment; sourced by the launch scripts.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ ! -f "$ROOT/.env" ]] || source "$ROOT/.env"
export NEXARM_CONDA_PREFIX=${NEXARM_CONDA_PREFIX:-$HOME/miniconda3/envs/nexarm-il}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
NEXARM_PYTHON="$NEXARM_CONDA_PREFIX/bin/python"
