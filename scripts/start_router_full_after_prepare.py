#!/usr/bin/env python3
"""Run on .209: wait for tokenization on .213, transfer and launch .212 DDP."""
from concurrent.futures import ThreadPoolExecutor
import json
import shlex
import subprocess
import time
import urllib.request

SOURCE = '10.107.206.213'
TARGET = '10.107.206.212'
SOURCE_DATA = '/workspace/zhangjinhao/router-data/full-gap'
SOURCE_HOST = '/root/zhangjinhao/router-full-gap'
TARGET_HOST = '/root/zhangjinhao/router-single-npu/router-data/full-gap'


def ssh(host, command, timeout=120):
    return subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', '-o',
        'ConnectTimeout=10', host, command], capture_output=True, text=True,
        timeout=timeout, check=True).stdout.strip()


def log(stage, **values):
    print(json.dumps(dict(stage=stage, time=time.time(), **values)), flush=True)


def check_expert(pair):
    host, port, expected = pair
    with urllib.request.urlopen(f'http://10.107.206.{host}:{port}/v1/models', timeout=10) as response:
        names = [m['id'] for m in json.load(response)['data']]
    if expected not in names:
        raise RuntimeError(f'wrong expert identity: {host}:{port}: {names}')


def main():
    log('waiting_for_tokenization')
    while True:
        command = f'if test -f {SOURCE_DATA}/prepare.exit; then cat {SOURCE_DATA}/prepare.exit; fi'
        try:
            status = ssh(SOURCE, 'docker exec vllm-ascend sh -c ' + shlex.quote(command), timeout=45)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            time.sleep(30)
            continue
        if status:
            if status != '0':
                raise RuntimeError('tokenization failed; inspect full-gap/prepare.log')
            break
        time.sleep(30)
    manifest = json.loads(ssh(SOURCE, f'docker exec vllm-ascend cat {SOURCE_DATA}/requests.manifest.json'))
    log('tokenization_complete', counts=manifest['counts'])
    ssh(SOURCE, f'test ! -e {SOURCE_HOST} && docker cp vllm-ascend:{SOURCE_DATA} {SOURCE_HOST}', timeout=3600)
    ssh(TARGET, f'mkdir -p {TARGET_HOST}')
    for name in ('requests.jsonl', 'requests.manifest.json'):
        log('transferring', file=name)
        subprocess.run(['scp', '-q', '-3', f'{SOURCE}:{SOURCE_HOST}/{name}',
                        f'{TARGET}:{TARGET_HOST}/{name}'], check=True)
    source_hash = ssh(SOURCE, f'sha256sum {SOURCE_HOST}/requests.jsonl', timeout=600).split()[0]
    target_hash = ssh(TARGET, f'sha256sum {TARGET_HOST}/requests.jsonl', timeout=600).split()[0]
    if source_hash != target_hash:
        raise RuntimeError('dataset checksum mismatch')
    experts = [(h, p, 'moqe-qwen3-awq') for h in (208, 209, 213) for p in range(19000, 19008)]
    experts += [(h, p, 'moqe-qwen3-gptq') for h in (210, 211, 216) for p in range(19000, 19008)]
    with ThreadPoolExecutor(max_workers=48) as pool:
        list(pool.map(check_expert, experts))
    script = ('export TRAIN_CONFIG=configs/qwen3_14b_router_npu_full.json '
              'REQUESTS=/workspace/zhangjinhao/router-data/full-gap/requests.jsonl '
              'STREAM_DATA=1 SCORE_WINDOW=192 '
              'LOSS_CACHE=/workspace/zhangjinhao/router-training/gap-pilot-1024-48-scorers-ddp6/expert-loss-cache.jsonl; '
              'bash /workspace/zhangjinhao/MLsys/scripts/train_router_ddp_ascend.sh '
              '> /workspace/zhangjinhao/router-gap-full.log 2>&1; '
              'echo $? > /workspace/zhangjinhao/router-gap-full.exit')
    ssh(TARGET, 'test ! -e /root/zhangjinhao/router-single-npu/router-training/gap-full-48-scorers-ddp6/metrics.jsonl')
    ssh(TARGET, 'docker exec -d moqe-router-single bash -lc ' + shlex.quote(script))
    log('training_launched', node=TARGET, npu_cards=[1, 2, 3, 4, 5, 6], global_batch=48,
        epochs=1, validation_interval=5000, dataset_sha256=source_hash)


if __name__ == '__main__':
    main()
