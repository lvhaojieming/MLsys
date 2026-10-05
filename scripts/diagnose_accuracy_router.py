#!/usr/bin/env python3
"""Inspect saved router probabilities, task targets, and reference NLL cases."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import urllib.request
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
import torch_npu
from transformers import AutoTokenizer
from moqe_router.config import RouterArchitecture
from moqe_router.model import EmbeddingRouter
from moqe_router.training.data import TrainingExample, collate_requests
from moqe_router.training.embedding import FrozenEmbeddingProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint['architecture_config']
    config['expert_ids'] = tuple(config['expert_ids'])
    architecture = RouterArchitecture(**config)
    torch.npu.set_device(0)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device('npu:0')
    embedding = FrozenEmbeddingProvider.from_checkpoint(checkpoint['training_config']['base_model_path'],
        weight_key='model.embed_tokens.weight', embedding_dim=architecture.embedding_dim).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['training_config']['base_model_path'], local_files_only=True)
    router = EmbeddingRouter(architecture).to(device).eval()
    router.load_state_dict(checkpoint['router_state_dict'], strict=True)
    rows = [json.loads(line) for line in Path(args.labels).read_text().splitlines()]
    for offset in range(0, len(rows), 8):
        part = rows[offset:offset+8]
        batch = collate_requests([TrainingExample(r['id'], tuple(r['input_ids']), r['max_new_tokens'],
            tuple(r['expert_losses']), tuple(r['expert_correctness'])) for r in part])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.inference_mode(), torch.autocast('npu', dtype=torch.bfloat16):
            logits = router(embedding(batch['input_ids']), batch['attention_mask'], batch['max_new_tokens'])
        probs = logits.float().softmax(-1).cpu().tolist()
        for row, prob in zip(part, probs):
            row['router_probs'] = prob
        if offset % 512 == 0:
            print(json.dumps(dict(stage='router_probability_scan', samples=min(offset+8,len(rows)), total=len(rows))),flush=True)
    tau = checkpoint['training_config']['temperature']
    summaries = {}
    for split in ('train', 'valid'):
        part = [r for r in rows if r['split'] == split]
        costs = torch.tensor([r['expert_losses'] for r in part])
        q = torch.softmax(-(costs-costs.min(-1,keepdim=True).values)/tau, -1)
        counts = Counter(''.join(map(str,r['expert_correctness'])) for r in part)
        known_mass = sum(v for k,v in counts.items() if k != '00')
        constant_optimum = (counts['10'] + .5*counts['11'] + .1*q[:,0].sum().item()) / (known_mass + .1*len(part))
        groups = {}
        for label in ('10','01','11','00'):
            group = [r for r in part if ''.join(map(str,r['expert_correctness'])) == label]
            if group:
                groups[label] = dict(samples=len(group), mean_awq_probability=sum(r['router_probs'][0] for r in group)/len(group),
                                    awq_selected=sum(r['router_probs'][0]>=r['router_probs'][1] for r in group))
        summaries[split] = dict(samples=len(part), outcome_counts=dict(counts),
            nll_awq_wins=int((costs[:,0]<costs[:,1]).sum()), mean_nll_awq_target=q[:,0].mean().item(),
            nll_target_above_099=int((q[:,0]>.99).sum()), constant_router_optimum_awq=constant_optimum,
            awq_probability_range=[min(r['router_probs'][0] for r in part),max(r['router_probs'][0] for r in part)],
            groups=groups)
    cases = []
    for source in ('gsm8k','mmlu'):
        for outcome in ([0,1],[1,0],[1,1]):
            choices = [r for r in rows if r['split']=='valid' and r['source']==source and r['expert_correctness']==outcome]
            if not choices:
                continue
            row = choices[0]
            case = {k: row[k] for k in ('id','source','split','expert_correctness','expert_losses','router_probs','expert_answers','reference_answer')}
            case['prompt'] = tokenizer.decode(row['input_ids'], skip_special_tokens=False)
            costs = torch.tensor(row['expert_losses'])
            q = torch.softmax(-(costs-costs.min())/tau, -1).tolist()
            c = row['expert_correctness']
            target = [v/sum(c) for v in c] if sum(c) else [0.,0.]
            p = row['router_probs'][0]
            case['nll_target'] = q
            case['task_target'] = target
            case['combined_target_awq'] = (target[0]+.1*q[0])/(1.1 if sum(c) else .1)
            case['gradient_z_awq'] = dict(task=(p-target[0]) if sum(c) else 0., auxiliary=.1*(p-q[0]))
            breakdown = []
            for url,name in [('http://10.107.206.208:19000','moqe-qwen3-awq'),('http://10.107.206.210:19000','moqe-qwen3-gptq')]:
                payload = dict(model=name,prompt=row['input_ids']+row['target_ids'],max_tokens=1,temperature=0,seed=42,echo=True,logprobs=1)
                request = urllib.request.Request(url+'/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=180) as response:
                    result=json.load(response)
                logp=result['choices'][0]['logprobs']['token_logprobs'][len(row['input_ids']):len(row['input_ids'])+len(row['target_ids'])]
                if len(logp)!=len(row['target_ids']) or any(v is None for v in logp):
                    raise RuntimeError('reference token logprob alignment failed')
                breakdown.append(dict(answer_token_mean_nll=-sum(logp[:-1])/len(logp[:-1]), eos_nll=-logp[-1],
                    full_mean_nll=-sum(logp)/len(logp), target_tokens=len(logp)))
            case['rescored_nll_breakdown'] = breakdown
            cases.append(case)
            print(json.dumps(dict(stage='case', **{k:case[k] for k in ('id','expert_correctness','expert_losses','router_probs','task_target','nll_target','gradient_z_awq','rescored_nll_breakdown')})),flush=True)
    artifact = dict(checkpoint=args.checkpoint,summary=summaries,cases=cases)
    Path(args.output).write_text(json.dumps(artifact,ensure_ascii=False,indent=2))
    print(json.dumps(dict(stage='diagnostics_complete',summary=summaries,output=args.output)),flush=True)


if __name__ == '__main__':
    main()
