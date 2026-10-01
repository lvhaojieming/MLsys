#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_PLUGINS=''
export PYTHONPATH=/home/zjh/miniconda3/envs/engine/lib/python3.12/site-packages
cd "$task_root"
exec /home/zjh/miniconda3/envs/engine/bin/python -S "$task_root/scripts/run_router_pipeline.py" "$@"
