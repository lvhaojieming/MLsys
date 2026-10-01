#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH=/home/zjh/miniconda3/envs/engine/lib/python3.12/site-packages
exec /home/zjh/miniconda3/envs/engine/bin/python -S "$task_root/scripts/plot_router_training.py" "$@"
