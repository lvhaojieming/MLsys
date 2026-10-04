#!/usr/bin/env python3
"""Diagnose routing sensitivity to full prompts, generation budgets and batching."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer
from moqe_router.config import RouterArchitecture
from moqe_router.model import EmbeddingRouter
from moqe_router.training.data import TrainingExample, collate_requests
from moqe_router.training.embedding import FrozenEmbeddingProvider


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests',required=True)
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--base-model-path',required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    rows = [json.loads(s) for s in Path(a.requests).read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(a.base_model_path,local_files_only=True)
    short = [tok.apply_chat_template([{'role':'user','content':r['question']}],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=False) for r in rows]
    torch.npu.set_device(0)
    device = torch.device('npu:0')
    torch.backends.mha.set_fastpath_enabled(False)
    embedding = None
    reports = []
    for model_index,path in enumerate(a.checkpoints):
        c = torch.load(path,map_location='cpu',weights_only=True)
        arch_data = dict(c['architecture_config'])
        arch_data['expert_ids'] = tuple(arch_data['expert_ids'])
        arch = RouterArchitecture(**arch_data)
        if arch.expert_ids != ('qwen3-14b/awq-w4a16/v1','qwen3-14b/gptq-w4a16/v1'):
            raise ValueError('expert order mismatch')
        if embedding is None:
            embedding = FrozenEmbeddingProvider.from_checkpoint(a.base_model_path,
                weight_key=c['training_config']['embedding_weight_key'],embedding_dim=arch.embedding_dim).to(device).eval()
        router = EmbeddingRouter(arch).to(device).eval()
        loaded = router.load_state_dict(c['router_state_dict'],strict=True)
        def forward(indices,inputs,budget,precision='bf16'):
            batch = collate_requests([TrainingExample(rows[i]['id'],tuple(inputs[i]),budget,(0.,0.)) for i in indices])
            with torch.inference_mode(), torch.autocast('npu',dtype=torch.bfloat16,enabled=precision=='bf16'):
                logits = router(embedding(batch['input_ids'].to(device)),batch['attention_mask'].to(device),batch['max_new_tokens'].to(device))
            return logits.float().cpu()
        full = [r['input_ids'] for r in rows]
        report = dict(checkpoint=path,step=c['global_step'],strict_load_missing=list(loaded.missing_keys),
            strict_load_unexpected=list(loaded.unexpected_keys),expert_order=list(arch.expert_ids),cases={})
        for name,inputs,budget in [('full_1024',full,1024),('full_2048',full,2048),('question_1024',short,1024),('question_2048',short,2048)]:
            logits = torch.cat([forward(range(start,min(start+8,len(rows))),inputs,budget) for start in range(0,len(rows),8)])
            probabilities = logits.softmax(-1)[:,0]
            chosen = logits.argmax(-1)
            case = dict(samples=len(rows),awq_selected=int((chosen==0).sum()),gptq_selected=int((chosen==1).sum()),
                mean_awq_probability=float(probabilities.mean()),min_awq_probability=float(probabilities.min()),max_awq_probability=float(probabilities.max()),
                mean_prompt_tokens=sum(map(len,inputs))/len(inputs))
            report['cases'][name] = case
            with (out/f'router-{model_index}-{name}.jsonl').open('w') as stream:
                for row,z in zip(rows,logits.tolist()):
                    stream.write(json.dumps(dict(id=row['id'],logits=z))+'\n')
            print(json.dumps(dict(router=model_index,case=name,**case)),flush=True)
        indices = range(8)
        batch_logits = forward(indices,full,1024)
        single_logits = torch.cat([forward([i],full,1024) for i in indices])
        fp32_logits = forward(indices,full,1024,'fp32')
        report['batch_precision_check'] = dict(samples=8,batch1_vs8_winner_changes=int((single_logits.argmax(-1)!=batch_logits.argmax(-1)).sum()),
            bf16_vs_fp32_winner_changes=int((fp32_logits.argmax(-1)!=batch_logits.argmax(-1)).sum()),
            single_logits=single_logits.tolist(),batched_logits=batch_logits.tolist(),fp32_logits=fp32_logits.tolist())
        reports.append(report)
        del router
    (out/'metrics.json').write_text(json.dumps(dict(mode='input sensitivity only; no answer-accuracy evaluation',routers=reports),indent=2))
    print(json.dumps(reports),flush=True)


if __name__=='__main__':
    main()
