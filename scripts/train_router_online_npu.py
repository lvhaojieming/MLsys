#!/usr/bin/env python3
"""One-epoch NPU router training with prefetched vLLM reference-loss scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import torch_npu  # noqa: F401
from moqe_router.config import RouterArchitecture
from moqe_router.model import EmbeddingRouter
from moqe_router.training.data import TrainingExample, collate_requests
from moqe_router.training.embedding import FrozenEmbeddingProvider
from moqe_router.training.objective import gap_weighted_terms
from moqe_router.training.trainer import TrainingConfig, build_scheduler, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--awq-url', required=True)
    parser.add_argument('--gptq-url', required=True)
    args = parser.parse_args()
    config = TrainingConfig.from_json(args.config)
    if config.epochs != 1:
        raise ValueError('this validation entry point requires exactly one epoch')
    architecture = RouterArchitecture.from_json(config.architecture_config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'metrics.jsonl').exists():
        raise FileExistsError('choose a fresh output directory')
    rows = [json.loads(s) for s in Path(args.requests).read_text().splitlines()]
    train = [r for r in rows if r['split'] == 'train']
    valid = [r for r in rows if r['split'] == 'valid']
    if not train or not valid or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('nonempty disjoint train/validation samples required')
    if {r['group_id'] for r in train} & {r['group_id'] for r in valid}:
        raise ValueError('train/validation groups overlap')
    torch.manual_seed(config.seed)
    random.Random(config.seed).shuffle(train)
    torch.npu.set_device(0)
    device = torch.device('npu:0')
    torch.backends.mha.set_fastpath_enabled(False)
    embedding = FrozenEmbeddingProvider.from_checkpoint(config.base_model_path,
        weight_key=config.embedding_weight_key, embedding_dim=architecture.embedding_dim).to(device).eval()
    router = EmbeddingRouter(architecture).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=config.lr, betas=config.betas,
                                 eps=config.eps, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps=math.ceil(len(train)/config.batch_size),
                               warmup_ratio=config.warmup_ratio, min_lr_ratio=config.min_lr_ratio)
    metrics = output / 'metrics.jsonl'
    def record(**values):
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
    urls = [args.awq_url.rstrip('/'), args.gptq_url.rstrip('/')]
    expected_names = ['moqe-qwen3-awq', 'moqe-qwen3-gptq']
    for url, expected in zip(urls, expected_names):
        with urllib.request.urlopen(url+'/v1/models', timeout=30) as response:
            available = [m['id'] for m in json.load(response)['data']]
        if expected not in available:
            raise ValueError(f'expert identity mismatch: {url}: {available}')
    def score(url, name, row):
        ids = row['input_ids'] + row['target_ids']
        payload = dict(model=name, prompt=ids, max_tokens=1, temperature=0, seed=42,
                       echo=True, logprobs=1)
        request = urllib.request.Request(url+'/v1/completions', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.load(response)
        if result['usage']['prompt_tokens'] != len(ids):
            raise ValueError('expert altered prompt token count')
        logp = result['choices'][0]['logprobs']['token_logprobs']
        if len(logp) != len(ids)+1:
            raise ValueError('echo token alignment mismatch')
        values = logp[len(row['input_ids']):len(ids)]
        if len(values) != len(row['target_ids']) or not values or not all(v is not None and math.isfinite(v) for v in values):
            raise ValueError('invalid reference token log probabilities')
        return -sum(values)/len(values)
    def paired(batch):
        examples = []
        for row in batch:
            futures = [score_pool.submit(score, url, name, row) for url, name in zip(urls, expected_names)]
            losses = [f.result() for f in futures]
            with (output/'expert-loss-cache.jsonl').open('a') as f:
                f.write(json.dumps(dict(id=row['id'], expert_ids=architecture.expert_ids,
                    expert_losses=losses, target_tokens=len(row['target_ids']),
                    tokens_sha256=hashlib.sha256(json.dumps(row['input_ids']+row['target_ids']).encode()).hexdigest()))+'\n')
            examples.append(TrainingExample(row['id'], tuple(row['input_ids']), row['max_new_tokens'], tuple(losses)))
        return collate_requests(examples)
    def batches(data):
        chunks = [data[i:i+config.batch_size] for i in range(0, len(data), config.batch_size)]
        pending = prefetch.submit(paired, chunks[0])
        for i in range(len(chunks)):
            batch = pending.result()
            if i+1 < len(chunks):
                pending = prefetch.submit(paired, chunks[i+1])
            yield {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    record(stage='started', epochs=1, device='npu', train_samples=len(train), validation_samples=len(valid),
           expert_ids=architecture.expert_ids, loss='gap_weighted_sequence_soft_target_cross_entropy',
           tau=config.temperature, alpha=config.gap_alpha, gap_scale=config.gap_scale,
           frozen_embedding=True, full_prompt=True, synthetic=False)
    with ThreadPoolExecutor(max_workers=2) as score_pool, ThreadPoolExecutor(max_workers=1) as prefetch:
        router.train()
        seen, step = 0, 0
        for batch in batches(train):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('npu', dtype=torch.bfloat16):
                logits = router(embedding(batch['input_ids']), batch['attention_mask'], batch['max_new_tokens'])
            ce, weights = gap_weighted_terms(logits, batch['expert_losses'], config.temperature, config.gap_alpha, config.gap_scale)
            loss = (ce*weights).sum()/weights.sum()
            if not torch.isfinite(loss).item():
                raise FloatingPointError('non-finite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(router.parameters(), config.max_grad_norm, foreach=False)
            if not torch.isfinite(norm).item():
                raise FloatingPointError('non-finite gradient')
            optimizer.step(); scheduler.step()
            step += 1; seen += len(batch['ids'])
            record(split='train', epoch=1, step=step, trained_samples=seen, loss=loss.item(),
                   weight_mean=weights.mean().item(), weight_max=weights.max().item(),
                   expert_loss_mean=batch['expert_losses'].mean(0).tolist(), grad_norm=norm.item(),
                   allocated_bytes=torch.npu.memory_allocated(), reserved_bytes=torch.npu.memory_reserved())
        router.eval()
        numerator = denominator = regret = correct = count = 0.
        with torch.inference_mode():
            for batch in batches(valid):
                with torch.autocast('npu', dtype=torch.bfloat16):
                    logits = router(embedding(batch['input_ids']), batch['attention_mask'], batch['max_new_tokens'])
                losses = batch['expert_losses']
                ce, weights = gap_weighted_terms(logits, losses, config.temperature, config.gap_alpha, config.gap_scale)
                numerator += (ce*weights).sum().item(); denominator += weights.sum().item()
                selected = logits.argmax(-1); oracle = losses.min(-1)
                regret += (losses.gather(1, selected[:, None]).squeeze(1)-oracle.values).sum().item()
                correct += (selected == oracle.indices).sum().item(); count += len(selected)
        record(split='valid', epoch=1, loss=numerator/denominator, mean_routing_regret=regret/count,
               top1_accuracy=correct/count, sample_count=int(count))
    assert seen == len(train) and count == len(valid)
    assert not embedding.weight.requires_grad and embedding.weight.grad is None
    assert digest(router) != before, 'router parameters did not update'
    save_checkpoint(output/'checkpoint_last.pt', router=router, optimizer=optimizer, scheduler=scheduler,
                    epoch=1, best_validation_regret=regret/count, architecture=architecture,
                    training_config=config, global_step=step)
    record(stage='complete', epochs=1, trained_samples=seen, steps=step, checkpoint=str(output/'checkpoint_last.pt'),
           router_parameters_changed=True, embedding_frozen=True)


if __name__ == '__main__':
    main()

