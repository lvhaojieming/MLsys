#!/usr/bin/env bash
set -euo pipefail
: "${NODE_RANK:?Set NODE_RANK to 0 or 1}"
: "${TRAIN_CONFIG:?Set TRAIN_CONFIG to the training JSON path}"
: "${MASTER_ADDR:=10.107.206.213}"
: "${MASTER_PORT:=29613}"
: "${NNODES:=2}"
: "${NPROC_PER_NODE:=8}"
: "${HCCL_SOCKET_IFNAME:=enp61s0f0}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HCCL_SOCKET_IFNAME OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
exec python3 -m torch.distributed.run --nnodes="$NNODES" --nproc-per-node="$NPROC_PER_NODE" \
  --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  scripts/train_router_distributed.py --device npu --config "$TRAIN_CONFIG" "$@"
