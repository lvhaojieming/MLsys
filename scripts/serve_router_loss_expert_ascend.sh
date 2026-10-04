#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set the converted expert checkpoint directory}"
: "${RUNTIME_PATH:?Set the directory containing adapter and python-deps}"
: "${EXPERT_KIND:?Set awq or gptq}"
if [[ "$EXPERT_KIND" != awq && "$EXPERT_KIND" != gptq ]]; then exit 2; fi
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES="${NPU_CARD:-1}"
export VLLM_USE_V1=0 VLLM_VERSION=0.8.4 VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS="${SCORING_CPU_THREADS:-16}"
export MOQE_ASCEND_INT4_ADAPTER=1 MOQE_SCORE_CPU_SAMPLER=1
export PYTHONPATH="$repo_root/scripts/scoring_bootstrap:$repo_root/scripts:$RUNTIME_PATH/adapter:$RUNTIME_PATH/python-deps:/workspace/vllm:/workspace/vllm-ascend:${PYTHONPATH:-}"
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --quantization moqe_ascend_int4 \
  --served-model-name "moqe-qwen3-$EXPERT_KIND" --host 0.0.0.0 --port "${PORT:-18120}" \
  --tensor-parallel-size 1 --dtype float16 --max-model-len 12288 \
  --max-num-seqs "${MAX_NUM_SEQS:-2}" --gpu-memory-utilization "${HBM_UTILIZATION:-0.5}" --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-12288}" \
  --enforce-eager --disable-frontend-multiprocessing --disable-log-requests
