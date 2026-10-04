#!/usr/bin/env python3
"""Compare router checkpoints using the exact prompts and recorded expert answers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import torch_npu  # noqa: F401
from moqe_router.config import RouterArchitecture
from moqe_router.model import EmbeddingRouter
from moqe_router.training.data import TrainingExample, collate_requests
from moqe_router.training.embedding import FrozenEmbeddingProvider


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', required=True)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--base-model-path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    rows = [json.loads(s) for s in Path(a.requests).read_text().splitlines()]
    if not rows or a.batch_size < 1:
        raise ValueError('nonempty requests and positive batch size required')
    torch.npu.set_device(0)
    device = torch.device('npu:0')
    torch.backends.mha.set_fastpath_enabled(False)
    reports = []
    embedding = None
    embedding_dim = None
    for model_index, path in enumerate(a.checkpoints):
        checkpoint_path = Path(path)
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        data = dict(checkpoint['architecture_config'])
        data['expert_ids'] = tuple(data['expert_ids'])
        arch = RouterArchitecture(**data)
        if arch.expert_ids != ('qwen3-14b/awq-w4a16/v1','qwen3-14b/gptq-w4a16/v1'):
            raise ValueError('unexpected expert order')
        if any(len(r['input_ids']) > checkpoint['training_config']['max_prompt_tokens'] for r in rows):
            raise ValueError('prompt exceeds checkpoint context; truncation forbidden')
        if embedding is None:
            embedding = FrozenEmbeddingProvider.from_checkpoint(a.base_model_path,
                weight_key=checkpoint['training_config']['embedding_weight_key'], embedding_dim=arch.embedding_dim).to(device).eval()
            embedding_dim = arch.embedding_dim
        if arch.embedding_dim != embedding_dim:
            raise ValueError('checkpoint embedding dimensions differ')
        router = EmbeddingRouter(arch).to(device)
        router.load_state_dict(checkpoint['router_state_dict'],strict=True)
        router.eval()
        counts, predictions = Counter(), []
        with torch.inference_mode(), (out/f'router-{model_index}-predictions.jsonl').open('w') as stream:
            for start in range(0,len(rows),a.batch_size):
                chunk = rows[start:start+a.batch_size]
                examples = [TrainingExample(r['id'],tuple(r['input_ids']),r['max_new_tokens'],(0.,0.)) for r in chunk]
                batch = collate_requests(examples)
                with torch.autocast('npu',dtype=torch.bfloat16):
                    logits = router(embedding(batch['input_ids'].to(device)),
                        batch['attention_mask'].to(device),batch['max_new_tokens'].to(device))
                probabilities = logits.float().softmax(-1).cpu().tolist()
                for row, probability in zip(chunk,probabilities):
                    selected = max(range(2),key=lambda j:probability[j])
                    correct = row['expert_correct'][selected]
                    counts['correct'] += correct
                    counts[['awq_selected','gptq_selected'][selected]] += 1
                    counts[row['category']+'_count'] += 1
                    counts[row['category']+'_correct'] += correct
                    prediction = dict(id=row['id'],probabilities=probability,selected_expert=['awq','gptq'][selected],
                        correct=correct,category=row['category'],answer=row['expert_answers'][selected])
                    predictions.append(prediction)
                    stream.write(json.dumps(prediction)+'\n')
                if len(predictions)%256==0:
                    print(json.dumps(dict(router=model_index,completed=len(predictions),total=len(rows))),flush=True)
        assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()==digest
        report = dict(checkpoint=path,checkpoint_step=checkpoint['global_step'],checkpoint_sha256=digest,
            samples=len(rows),correct=counts['correct'],accuracy=counts['correct']/len(rows),
            routed_expert_counts=[counts['awq_selected'],counts['gptq_selected']],
            exclusive_opportunity_recall={name:counts[name+'_correct']/counts[name+'_count'] if counts[name+'_count'] else None
                for name in ('awq_only','gptq_only')}, counts=dict(counts),checkpoint_unchanged=True)
        reports.append(report)
        print(json.dumps(report),flush=True)
        del router
    report = dict(samples=len(rows),mode='cached-expert-answer replay; NPU router forward',
        scoring='original lm-eval exact_match',
        fixed_expert_accuracy=[sum(r['expert_correct'][j] for r in rows)/len(rows) for j in range(2)],
        answer_oracle_accuracy=sum(any(r['expert_correct']) for r in rows)/len(rows),
        routers=reports)
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
