#!/usr/bin/env bash
# Run on expert host .213; start the one-epoch router pilot after label checks.
set -euo pipefail
while ! docker exec vllm-ascend test -f /tmp/accuracy-router-labels-v2.exit; do sleep 15; done
test "$(docker exec vllm-ascend cat /tmp/accuracy-router-labels-v2.exit)" = 0
docker exec vllm-ascend python3 /workspace/MLsys-gap/scripts/audit_accuracy_router_labels.py \
  --output /workspace/zhangjinhao/router-data/accuracy-gsm-mmlu-6k-v2
test ! -e /tmp/accuracy-gsm-mmlu-6k-v2
docker cp vllm-ascend:/workspace/zhangjinhao/router-data/accuracy-gsm-mmlu-6k-v2 /tmp/accuracy-gsm-mmlu-6k-v2
ssh -n root@10.107.206.212 'test ! -e /root/zhangjinhao/router-single-npu/router-data/accuracy-gsm-mmlu-6k-v2 && test ! -e /root/zhangjinhao/router-single-npu/router-training/accuracy-gsm-mmlu-6k-v2 && mkdir -p /root/zhangjinhao/router-single-npu/router-data'
scp -r /tmp/accuracy-gsm-mmlu-6k-v2 root@10.107.206.212:/root/zhangjinhao/router-single-npu/router-data/
ssh -n root@10.107.206.212 'npu-smi info | awk '\''/Process id /{proc=1} proc && /^\| [1-6] +[0-9] +\| +[0-9]+/{busy=1} END{exit busy}'\'''
ssh -n root@10.107.206.212 'docker exec -d \
  -e ASCEND_RT_VISIBLE_DEVICES=1,2,3,4,5,6 \
  -e TRAIN_CONFIG=configs/qwen3_14b_router_accuracy_pilot.json \
  -e REQUESTS=/workspace/zhangjinhao/router-data/accuracy-gsm-mmlu-6k-v2/labeled.jsonl \
  -e LOSS_CACHE=/workspace/zhangjinhao/router-data/accuracy-gsm-mmlu-6k-v2/expert-loss-cache.jsonl \
  -e BALANCED_VALIDATION=/workspace/zhangjinhao/validation-balanced-c4-wiki/requests.jsonl \
  -e ROUTER_RANKS=6 -e EXPERT_CONCURRENCY=8 -e SCORE_WINDOW=384 \
  moqe-router-single bash -c '\''cd /workspace/zhangjinhao/MLsys; bash scripts/train_router_ddp_ascend.sh > /workspace/zhangjinhao/accuracy-router-train-v2.log 2>&1; echo $? > /workspace/zhangjinhao/accuracy-router-train-v2.exit'\'''
echo '{"stage":"router_training_launched","node":"10.107.206.212","npu_cards":[1,2,3,4,5,6],"epochs":1,"batch_per_rank":8,"global_batch":48}'
