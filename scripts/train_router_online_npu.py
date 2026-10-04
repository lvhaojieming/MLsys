#!/usr/bin/env python3
"""One-epoch NPU router training with prefetched vLLM reference-loss scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from itertools import islice
from queue import Queue, Empty
from array import array
import os
from datetime import timedelta
from pathlib import Path
import random
import sys
import time
import urllib.request
import urllib.error
import threading
import subprocess
import shlex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from moqe_router.config import RouterArchitecture
from moqe_router.model import EmbeddingRouter
from moqe_router.training.data import TrainingExample, collate_requests
from moqe_router.training.embedding import FrozenEmbeddingProvider
from moqe_router.training.objective import gap_weighted_terms
from moqe_router.training.checkpoint_policy import BestCheckpointPolicy
from moqe_router.training.trainer import TrainingConfig, build_scheduler, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--requests')
    parser.add_argument('--awq-url', nargs='+', required=True)
    parser.add_argument('--gptq-url', nargs='+', required=True)
    parser.add_argument('--loss-cache', help='Optional previously computed paired loss cache')
    parser.add_argument('--score-window', type=int, default=64,
                        help='Prefetch this many samples; router batch size stays in config')
    parser.add_argument('--stream', action='store_true', help='Bounded-memory input using the preparation manifest')
    parser.add_argument('--raw-source-host', help='Host serving a CPU raw-tokenization stream on port 19190')
    parser.add_argument('--expert-concurrency', type=int, default=2, help='Maximum in-flight requests per expert instance')
    args = parser.parse_args()
    if args.expert_concurrency < 1:
        parser.error('--expert-concurrency must be positive')
    if not args.raw_source_host and not args.requests:
        parser.error('--requests or --raw-source-host is required')
    config = TrainingConfig.from_json(args.config)
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.npu.set_device(local_rank)
    device = torch.device('npu', local_rank)
    if world > 1:
        dist.init_process_group('hccl', timeout=timedelta(minutes=30))
    rank = dist.get_rank() if world > 1 else 0
    global_batch = config.batch_size * world
    if args.score_window < global_batch or args.score_window % global_batch:
        raise ValueError('score-window must be a multiple of the global router batch size')
    if args.score_window < config.batch_size:
        raise ValueError('score-window must be at least the router batch size')
    if config.epochs != 1:
        raise ValueError('this validation entry point requires exactly one epoch')
    architecture = RouterArchitecture.from_json(config.architecture_config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.jsonl').exists():
        raise FileExistsError('choose a fresh output directory')
    if world > 1:
        dist.barrier()
    torch.manual_seed(config.seed)
    producer = None
    raw_counts = {}
    if args.raw_source_host:
        valid = []
        header = [None]
        if rank == 0:
            producer = urllib.request.urlopen('http://' + args.raw_source_host + ':19190/stream', timeout=1800)
            for line in producer:
                row = json.loads(line)
                if row.get('event') == 'ready':
                    header[0] = row
                    break
                if row.get('split') != 'valid':
                    raise ValueError('raw stream must begin with fixed validation samples')
                valid.append(row)
            if header[0] is None:
                raise RuntimeError('raw tokenizer exited before readiness; inspect the raw-stream server log')
        if world > 1:
            dist.broadcast_object_list(header, src=0)
        train_count = header[0]['estimated_train_samples']
        valid_count = header[0]['validation_samples']
        def live_rows():
            for line in producer:
                row = json.loads(line)
                if row.get('event') == 'complete':
                    raw_counts.update(row['counts'])
                    continue
                if row.get('split') != 'train':
                    raise ValueError('unexpected raw-stream record')
                yield row
            producer.close()
            if 'train' not in raw_counts:
                raise RuntimeError('raw tokenizer failed before completing one pass')
        train = live_rows() if rank == 0 else []
    elif args.stream:
        manifest = json.loads(Path(args.requests).with_suffix('.manifest.json').read_text())
        train_count, valid_count = manifest['counts']['train'], manifest['counts']['valid']
        if not train_count or not valid_count:
            raise ValueError('nonempty train and validation required')
        valid = []
        offsets = array('Q')
        if rank == 0:
            observed = 0
            with Path(args.requests).open('rb') as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    row = json.loads(line)
                    if row['split'] == 'valid':
                        valid.append(row)
                    elif row['split'] == 'train':
                        observed += 1
                        offsets.append(offset)
                    else:
                        raise ValueError('unexpected dataset split')
            if observed != train_count or len(valid) != valid_count:
                raise ValueError('dataset and preparation manifest counts differ')
        def train_rows():
            valid_groups = {r['group_id'] for r in valid}
            random.Random(config.seed).shuffle(offsets)
            with Path(args.requests).open('rb') as stream:
                for offset in offsets:
                    stream.seek(offset)
                    row = json.loads(stream.readline())
                    if row['group_id'] in valid_groups:
                        raise ValueError('train/validation groups overlap')
                    yield row
        train = train_rows() if rank == 0 else []
    else:
        rows = [json.loads(s) for s in Path(args.requests).read_text().splitlines()]
        train = [r for r in rows if r['split'] == 'train']
        valid = [r for r in rows if r['split'] == 'valid']
        if not train or not valid or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('nonempty disjoint train/validation samples required')
        if {r['group_id'] for r in train} & {r['group_id'] for r in valid}:
            raise ValueError('train/validation groups overlap')
        random.Random(config.seed).shuffle(train)
        train_count, valid_count = len(train), len(valid)
    torch.backends.mha.set_fastpath_enabled(False)
    embedding = FrozenEmbeddingProvider.from_checkpoint(config.base_model_path,
        weight_key=config.embedding_weight_key, embedding_dim=architecture.embedding_dim).to(device).eval()
    router = EmbeddingRouter(architecture).to(device)
    train_model = DistributedDataParallel(router, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else router
    optimizer = torch.optim.AdamW(router.parameters(), lr=config.lr, betas=config.betas,
                                 eps=config.eps, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps=math.ceil(train_count/global_batch),
                               warmup_ratio=config.warmup_ratio, min_lr_ratio=config.min_lr_ratio)
    metrics = output / 'metrics.jsonl'
    def record(**values):
        if rank != 0:
            return
        values['time'] = time.time()
        text = json.dumps(values, allow_nan=False)
        print(text, flush=True)
        with metrics.open('a') as f:
            f.write(text+'\n')
    def digest(model):
        h = hashlib.sha256()
        for tensor in model.state_dict().values():
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before = digest(router)
    pools = [[url.rstrip('/') for url in args.awq_url], [url.rstrip('/') for url in args.gptq_url]]
    expected_names = ['moqe-qwen3-awq', 'moqe-qwen3-gptq']
    for pool, expected in zip(pools, expected_names):
        for url in pool:
            with urllib.request.urlopen(url+'/v1/models', timeout=30) as response:
                available = [m['id'] for m in json.load(response)['data']]
            if expected not in available:
                raise ValueError(f'expert identity mismatch: {url}: {available}')
    cache = {}
    if args.loss_cache:
        for line in Path(args.loss_cache).read_text().splitlines():
            value = json.loads(line)
            if value['expert_ids'] != list(architecture.expert_ids):
                raise ValueError('cached expert order mismatch')
            if len(value['expert_losses']) != 2 or not all(math.isfinite(x) for x in value['expert_losses']):
                raise ValueError('invalid cached expert losses')
            cache[value['id']] = value
    endpoint_slots = {url: threading.BoundedSemaphore(args.expert_concurrency) for pool in pools for url in pool}
    def score(url, name, row):
        ids = row['input_ids'] + row['target_ids']
        payload = dict(model=name, prompt=ids, max_tokens=1, temperature=0, seed=42,
                       echo=True, logprobs=1)
        request = urllib.request.Request(url+'/v1/completions', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with endpoint_slots[url]:
            for attempt in range(6):
                try:
                    with urllib.request.urlopen(request, timeout=1800) as response:
                        result = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                    if isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                        raise
                    if attempt == 5:
                        raise
                    time.sleep(min(30, 2 ** attempt))
        if result['usage']['prompt_tokens'] != len(ids):
            raise ValueError('expert altered prompt token count')
        logp = result['choices'][0]['logprobs']['token_logprobs']
        if len(logp) != len(ids)+1:
            raise ValueError('echo token alignment mismatch')
        values = logp[len(row['input_ids']):len(ids)]
        if len(values) != len(row['target_ids']) or not values or not all(v is not None and math.isfinite(v) for v in values):
            raise ValueError('invalid reference token log probabilities')
        return -sum(values)/len(values)
    def token_hash(row):
        return hashlib.sha256(json.dumps(row['input_ids']+row['target_ids']).encode()).hexdigest()
    def paired(batch):
        scoring_started = time.monotonic()
        pending = {}
        queues = [Queue(), Queue()]
        for row in batch:
            cached = cache.get(row['id'])
            if cached:
                if cached['tokens_sha256'] != token_hash(row) or cached['target_tokens'] != len(row['target_ids']):
                    raise ValueError('cached loss belongs to different tokens')
                continue
            pending[row['id']] = [None, None]
            for queue in queues:
                queue.put(row)
        def consume(expert, url, name):
            while True:
                try:
                    row = queues[expert].get_nowait()
                except Empty:
                    return
                pending[row['id']][expert] = score(url, name, row)
        workers = [score_pool.submit(consume, expert, url, name)
                   for expert, (pool, name) in enumerate(zip(pools, expected_names))
                   for url in pool for _ in range(args.expert_concurrency)]
        for worker in workers:
            worker.result()
        examples = []
        for row in batch:
            losses = cache[row['id']]['expert_losses'] if row['id'] in cache else pending[row['id']]
            if row['id'] not in cache:
                value = dict(id=row['id'], expert_ids=list(architecture.expert_ids),
                    expert_losses=losses, target_tokens=len(row['target_ids']),
                    tokens_sha256=token_hash(row))
                with (output/'expert-loss-cache.jsonl').open('a') as f:
                    f.write(json.dumps(value)+'\n')
                cache[row['id']] = value
            examples.append(TrainingExample(row['id'], tuple(row['input_ids']), row['max_new_tokens'], tuple(losses)))
        elapsed = time.monotonic() - scoring_started
        record(stage='expert_scoring', samples=len(batch), new_scored_samples=len(pending),
               seconds=elapsed, new_samples_per_second=len(pending)/elapsed if pending else None)
        return examples
    def batches(data):
        iterator = iter(data)
        first = list(islice(iterator, args.score_window)) if rank == 0 else []
        pending = prefetch.submit(paired, first) if first else None
        while True:
            payload = [None]
            if rank == 0:
                try:
                    if pending is None:
                        payload[0] = {'done': True}
                    else:
                        payload[0] = {'examples': pending.result()}
                        next_chunk = list(islice(iterator, args.score_window))
                        pending = prefetch.submit(paired, next_chunk) if next_chunk else None
                except Exception as error:
                    payload[0] = {'error': repr(error)}
            if world > 1:
                dist.broadcast_object_list(payload, src=0)
            if payload[0].get('done'):
                break
            if 'error' in payload[0]:
                raise RuntimeError('expert scoring failed: '+payload[0]['error'])
            examples = payload[0]['examples']
            for start in range(0, len(examples), global_batch):
                slice_ = [(example, True) for example in examples[start:start+global_batch]]
                # Keep all DDP ranks active even when the final global batch has
                # fewer than world samples. Padding contributes zero loss/metrics.
                if len(slice_) < world:
                    slice_ += [(examples[i % len(examples)], False) for i in range(world-len(slice_))]
                part = slice_[rank::world]
                batch = collate_requests([example for example, _ in part])
                batch['real_mask'] = torch.tensor([real for _, real in part], dtype=torch.float32)
                yield {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    policy = BestCheckpointPolicy(config.validation_interval_steps)
    def validate_and_save(step):
        router.eval()
        numerator = denominator = regret = correct = count = 0.
        expert_totals = torch.zeros(2, device=device)
        routed_counts = torch.zeros(2, device=device)
        oracle_loss_total = 0.
        with torch.inference_mode():
            for batch in batches(valid):
                with torch.autocast('npu', dtype=torch.bfloat16):
                    logits = router(embedding(batch['input_ids']), batch['attention_mask'], batch['max_new_tokens'])
                losses = batch['expert_losses']
                ce, weights = gap_weighted_terms(logits, losses, config.temperature, config.gap_alpha, config.gap_scale)
                real = batch['real_mask']
                numerator += (ce*weights*real).sum().item(); denominator += (weights*real).sum().item()
                selected = logits.argmax(-1); oracle = losses.min(-1)
                regret += ((losses.gather(1, selected[:, None]).squeeze(1)-oracle.values)*real).sum().item()
                correct += ((selected == oracle.indices)*real).sum().item(); count += real.sum().item()
                expert_totals += (losses*real[:, None]).sum(0)
                routed_counts += torch.bincount(selected, weights=real, minlength=2)
                oracle_loss_total += (oracle.values*real).sum().item()
        summary = torch.tensor([numerator, denominator, regret, correct, count, oracle_loss_total], device=device)
        if world > 1:
            dist.all_reduce(summary)
            dist.all_reduce(expert_totals)
            dist.all_reduce(routed_counts)
        numerator, denominator, regret, correct, count, oracle_loss_total = summary.tolist()
        record(split='valid', epoch=1, step=step, loss=numerator/denominator, mean_routing_regret=regret/count,
               top1_accuracy=correct/count, sample_count=int(count),
               fixed_expert_mean_loss=(expert_totals/count).tolist(),
               fixed_expert_mean_regret=((expert_totals-oracle_loss_total)/count).tolist(),
               routed_expert_counts=routed_counts.tolist())
        assert count == valid_count
        improved = policy.observe(step, regret/count)
        if rank == 0 and improved:
            save_checkpoint(output/'checkpoint_best.pt', router=router, optimizer=optimizer, scheduler=scheduler,
                epoch=1, best_validation_regret=policy.best_regret, architecture=architecture,
                training_config=config, global_step=step)
            record(stage='best_checkpoint_saved', step=step, best_validation_regret=policy.best_regret)
        if world > 1:
            dist.barrier()
        train_model.train()
    record(stage='started', epochs=1, device='npu', train_samples=train_count, validation_samples=valid_count,
           expert_ids=architecture.expert_ids, loss='gap_weighted_sequence_soft_target_cross_entropy',
           tau=config.temperature, alpha=config.gap_alpha, gap_scale=config.gap_scale,
           frozen_embedding=True, full_prompt=True, synthetic=False,
           expert_concurrency=args.expert_concurrency, scoring_pools=pools, cached_samples=len(cache), score_window=args.score_window,
           world_size=world, batch_per_rank=config.batch_size, global_batch_size=global_batch,
           validation_interval_steps=config.validation_interval_steps, checkpoint_policy='best_only_by_mean_routing_regret', streaming_input=args.stream or bool(args.raw_source_host),
           raw_source_host=args.raw_source_host, train_count_is_estimate=bool(args.raw_source_host))
    with ThreadPoolExecutor(max_workers=sum(map(len, pools)) * args.expert_concurrency) as score_pool, ThreadPoolExecutor(max_workers=1) as prefetch:
        train_model.train()
        torch.manual_seed(config.seed + rank)
        seen, step = 0, 0
        for batch in batches(train):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('npu', dtype=torch.bfloat16):
                logits = train_model(embedding(batch['input_ids']), batch['attention_mask'], batch['max_new_tokens'])
            ce, weights = gap_weighted_terms(logits, batch['expert_losses'], config.temperature, config.gap_alpha, config.gap_scale)
            weights = weights * batch['real_mask']
            weighted_sum = (ce*weights).sum()
            denominator_tensor = weights.sum().detach()
            if world > 1:
                dist.all_reduce(denominator_tensor)
            loss = weighted_sum * world / denominator_tensor
            finite = torch.isfinite(loss).to(torch.int32)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError('non-finite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(router.parameters(), config.max_grad_norm, foreach=False)
            if not torch.isfinite(norm).item():
                raise FloatingPointError('non-finite gradient')
            optimizer.step(); scheduler.step()
            totals = torch.stack((weighted_sum.detach(), weights.sum(), batch['real_mask'].sum(),
                (batch['expert_losses'][:,0]*batch['real_mask']).sum(),
                (batch['expert_losses'][:,1]*batch['real_mask']).sum()))
            if world > 1:
                dist.all_reduce(totals)
            step += 1; seen += int(totals[2].item())
            record(split='train', epoch=1, step=step, trained_samples=seen, loss=(totals[0]/totals[1]).item(),
                   weight_mean=(totals[1]/totals[2]).item(), weight_max=weights.max().item(),
                   expert_loss_mean=(totals[3:5]/totals[2]).tolist(), grad_norm=norm.item(),
                   allocated_bytes=torch.npu.memory_allocated(), reserved_bytes=torch.npu.memory_reserved())
            if policy.due(step):
                validate_and_save(step)
        if policy.due(step, final=True):
            validate_and_save(step)
    if args.raw_source_host:
        final_counts = [raw_counts if rank == 0 else None]
        if world > 1:
            dist.broadcast_object_list(final_counts, src=0)
        train_count = final_counts[0]['train']
    assert seen == train_count
    assert not embedding.weight.requires_grad and embedding.weight.grad is None
    final_digest = digest(router)
    assert final_digest != before, 'router parameters did not update'
    if world > 1:
        hashes = [None] * world
        dist.all_gather_object(hashes, final_digest)
        assert len(set(hashes)) == 1, 'router weights differ between ranks'
    if rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return
    record(stage='complete', epochs=1, trained_samples=seen, steps=step, checkpoint=str(output/'checkpoint_best.pt'),
           best_validation_regret=policy.best_regret,
           router_parameters_changed=True, embedding_frozen=True, world_size=world, parameter_consistency=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

