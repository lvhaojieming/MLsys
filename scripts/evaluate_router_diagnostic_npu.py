#!/usr/bin/env python3
"""Evaluate an existing router checkpoint on labeled requests using one NPU."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import torch_npu  # noqa: F401
from moqe_router.config import RouterArchitecture
from moqe_router.model import EmbeddingRouter
from moqe_router.training.data import RequestDataset, collate_requests
from moqe_router.training.embedding import FrozenEmbeddingProvider
from moqe_router.training.objective import loss_aware_terms


def summarize(rows, names):
    n = len(rows)
    confusion = [[0 for _ in names] for _ in names]
    for row in rows:
        confusion[row['oracle_index']][row['selected_index']] += 1
    return dict(samples=n, top1_accuracy=sum(r['correct'] for r in rows)/n,
        confusion_matrix_oracle_rows_selected_columns=confusion,
        oracle_expert_counts=[sum(r) for r in confusion],
        routed_expert_counts=[sum(r[j] for r in confusion) for j in range(len(names))],
        per_expert_oracle_recall=[confusion[j][j]/sum(confusion[j]) if sum(confusion[j]) else None for j in range(len(names))],
        mean_routing_regret=sum(r['regret'] for r in rows)/n,
        mean_selected_reference_nll=sum(r['selected_loss'] for r in rows)/n,
        mean_oracle_reference_nll=sum(min(r['expert_losses']) for r in rows)/n,
        fixed_expert_mean_regret=[sum(r['expert_losses'][j]-min(r['expert_losses']) for r in rows)/n for j in range(len(names))],
        mean_router_probabilities=[sum(r['probabilities'][j] for r in rows)/n for j in range(len(names))],
        uniform_soft_target_cross_entropy=sum(r['soft_target_ce'] for r in rows)/n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--requests', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('batch-size must be positive')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint_path = Path(a.checkpoint)
    before = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    arch_data = dict(checkpoint['architecture_config'])
    arch_data['expert_ids'] = tuple(arch_data['expert_ids'])
    architecture = RouterArchitecture(**arch_data)
    config = checkpoint['training_config']
    dataset = RequestDataset(a.requests, architecture.expert_ids, config['max_prompt_tokens'])
    torch.npu.set_device(0)
    device = torch.device('npu:0')
    torch.backends.mha.set_fastpath_enabled(False)
    embedding = FrozenEmbeddingProvider.from_checkpoint(config['base_model_path'],
        weight_key=config['embedding_weight_key'], embedding_dim=architecture.embedding_dim).to(device).eval()
    router = EmbeddingRouter(architecture).to(device)
    router.load_state_dict(checkpoint['router_state_dict'], strict=True)
    router.eval()
    rows = []
    started = time.monotonic()
    with torch.inference_mode(), (out/'predictions.jsonl').open('w') as stream:
        for start in range(0, len(dataset), a.batch_size):
            batch = collate_requests(dataset.examples[start:start+a.batch_size])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            with torch.autocast('npu', dtype=torch.bfloat16):
                logits = router(embedding(batch['input_ids']), batch['attention_mask'], batch['max_new_tokens'])
            probabilities = logits.float().softmax(-1).cpu().tolist()
            ce, _ = loss_aware_terms(logits, batch['expert_losses'], config['temperature'])
            ce = ce.cpu().tolist()
            losses = batch['expert_losses'].cpu().tolist()
            for i, sample_id in enumerate(batch['ids']):
                selected = max(range(len(architecture.expert_ids)), key=lambda j: probabilities[i][j])
                oracle = min(range(len(architecture.expert_ids)), key=lambda j: losses[i][j])
                row = dict(id=sample_id, source=sample_id.split(':')[0], probabilities=probabilities[i],
                    expert_losses=losses[i], selected_index=selected, oracle_index=oracle,
                    correct=selected == oracle, selected_loss=losses[i][selected],
                    regret=losses[i][selected]-losses[i][oracle], soft_target_ce=ce[i])
                rows.append(row)
                stream.write(json.dumps(row)+'\n')
            if len(rows) % 128 == 0:
                print(json.dumps(dict(stage='evaluating', completed=len(rows), total=len(dataset))), flush=True)
    sources = defaultdict(list)
    for row in rows:
        sources[row['source']].append(row)
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == before
    report = dict(checkpoint=str(checkpoint_path), checkpoint_sha256=before,
        checkpoint_step=checkpoint['global_step'], expert_ids=list(architecture.expert_ids),
        diagnostic_only=True, independent_generalization_estimate=False,
        checkpoint_unchanged=True, elapsed_seconds=time.monotonic()-started,
        overall=summarize(rows, architecture.expert_ids),
        sources={s: summarize(items, architecture.expert_ids) for s, items in sources.items()})
    (out/'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
