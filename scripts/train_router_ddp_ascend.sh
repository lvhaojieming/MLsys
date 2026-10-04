#!/usr/bin/env bash
# Six local NPU ranks; expert loss scoring is distributed over six remote nodes.
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-1,2,3,4,5,6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
AWQ_URLS=()
GPTQ_URLS=()
for host in ${AWQ_HOSTS:-10.107.206.208 10.107.206.209 10.107.206.213}; do
  for port in {19000..19007}; do AWQ_URLS+=("http://$host:$port"); done
done
for host in ${GPTQ_HOSTS:-10.107.206.210 10.107.206.211 10.107.206.216}; do
  for port in {19000..19007}; do GPTQ_URLS+=("http://$host:$port"); done
done
EXTRA_ARGS=()
if [[ "${STREAM_DATA:-0}" == 1 ]]; then EXTRA_ARGS+=(--stream); fi
if [[ -n "${RAW_SOURCE_HOST:-}" ]]; then EXTRA_ARGS+=(--raw-source-host "$RAW_SOURCE_HOST"); fi
exec torchrun --standalone --nnodes=1 --nproc-per-node="${ROUTER_RANKS:-6}" \
  scripts/train_router_online_npu.py \
  --config "${TRAIN_CONFIG:-configs/qwen3_14b_router_npu_pilot.json}" \
  --requests "${REQUESTS:-/workspace/zhangjinhao/router-data/pilot-gap/requests.jsonl}" \
  --awq-url "${AWQ_URLS[@]}" --gptq-url "${GPTQ_URLS[@]}" \
  --expert-concurrency "${EXPERT_CONCURRENCY:-8}" \
  --score-window "${SCORE_WINDOW:-192}" \
  --loss-cache "${LOSS_CACHE:-/workspace/zhangjinhao/router-training/paired-loss-reuse.jsonl}" "${EXTRA_ARGS[@]}"
