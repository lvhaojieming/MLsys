#!/usr/bin/env python3
"""Update the Router as paired frozen-expert losses arrive from separate GPUs."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''): h.update(data)
    return h.hexdigest()


def read_rows(source,pilot):
    result={}; ids=set(); groups={}
    for split in ('train','valid','test'):
        rows=[]
        for line in (source/f'{split}.jsonl').open():
            r=json.loads(line)
            assert r['id'] not in ids; ids.add(r['id'])
            assert groups.setdefault(r['group_id'],split)==split
            assert r['split']==split
            rows.append({k:r[k] for k in ('id','source','source_id','group_id','prompt_hash','split','input_ids','target_ids','max_new_tokens')})
        if pilot:
            buckets=defaultdict(list)
            for r in rows: buckets[r['source']].append(r)
            per_source=12 if split=='train' else 2
            rows=[]
            for candidates in buckets.values():
                longest=max(candidates,key=lambda r:len(r['input_ids']))
                ordered=[candidates[0]]
                if longest['id']!=ordered[0]['id']: ordered.append(longest)
                used={r['id'] for r in ordered}
                ordered.extend(r for r in candidates if r['id'] not in used)
                rows.extend(ordered[:per_source])
        result[split]=rows
    return result


def train_online(deployment_path,config_path,resume=None,pilot=False):
    import torch
    from torch.nn import functional as F
    from moqe_router.config import RouterArchitecture
    from moqe_router.expert_scoring import validate_deployment
    from moqe_router.model import EmbeddingRouter
    from moqe_router.training.embedding import FrozenEmbeddingProvider
    from moqe_router.training.metrics import routing_metrics
    from moqe_router.training.objective import build_loss_aware_targets, loss_aware_router_loss
    from moqe_router.training.online_losses import PairedExpertLossCache,prefetched_batches
    from moqe_router.training.trainer import (TrainingConfig,build_scheduler,save_checkpoint,load_checkpoint,
                                            _finite_or_raise,_move_batch,_log_record,_require_bf16_cuda,validate)
    deployment=json.loads(Path(deployment_path).read_text()); validate_deployment(deployment)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(deployment['router_gpu']): raise ValueError('Router must use its dedicated physical GPU')
    config=TrainingConfig.from_json(config_path)
    architecture=RouterArchitecture.from_json(ROOT/config.architecture_config)
    if list(architecture.expert_ids)!=[e['expert_id'] for e in deployment['experts']]: raise ValueError('expert order mismatch')
    source=ROOT/deployment['cleaned_data']; cache_dir=ROOT/deployment['labeled_data']
    output=ROOT/config.output_dir
    if json.loads((source/'validation.json').read_text())['status']!='passed': raise ValueError('source must pass validation')
    identity={'source_hashes':{s:sha256(source/f'{s}.jsonl') for s in ('train','valid','test')},
              'expert_ids':[e['expert_id'] for e in deployment['experts']],
              'models':[e['model_path'] for e in deployment['experts']],
              'quantization':[e['quantization'] for e in deployment['experts']],
              'dtype':'float16','target_contract':'mean target NLL including EOS; identical token IDs; causal next-token logprobs',
              'limit_per_split':0,'pilot_selection':None}
    if pilot: cache_dir=cache_dir.with_name(cache_dir.name+'-online-pilot')
    cache_dir.mkdir(parents=True,exist_ok=True)
    identity_path=cache_dir/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity: raise ValueError('loss cache/model contract changed')
    identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    rows=read_rows(source,pilot)
    cache=PairedExpertLossCache(deployment['experts'],cache_dir)
    device=_require_bf16_cuda(); random.seed(config.seed); torch.manual_seed(config.seed); torch.cuda.manual_seed_all(config.seed)
    embedding=FrozenEmbeddingProvider.from_checkpoint(config.base_model_path,weight_key=config.embedding_weight_key,
                                                    embedding_dim=architecture.embedding_dim).to(device).eval()
    assert not any(p.requires_grad for p in embedding.parameters())
    router=EmbeddingRouter(architecture).to(device)
    optimizer=torch.optim.AdamW(router.parameters(),lr=config.lr,betas=config.betas,eps=config.eps,weight_decay=config.weight_decay)
    scheduler=build_scheduler(optimizer,total_steps=math.ceil(len(rows['train'])/config.batch_size)*config.epochs,
                              warmup_ratio=config.warmup_ratio,min_lr_ratio=config.min_lr_ratio)
    output.mkdir(parents=True,exist_ok=True); metrics=output/'metrics.jsonl'
    best=output/'checkpoint_best.pt'; last=output/'checkpoint_last.pt'
    if resume is None and any(p.exists() for p in (metrics,best,last)): raise FileExistsError('training artifacts exist; use --resume')
    epoch_done=0; best_regret=float('inf'); step=0
    if resume:
        epoch_done,best_regret,step=load_checkpoint(resume,router=router,optimizer=optimizer,scheduler=scheduler,
                                                architecture=architecture,device=device)
    _log_record({'stage':'online_training_started','train_rows':len(rows['train']),'valid_rows':len(rows['valid']),
                 'expert_gpus':[e['gpu'] for e in deployment['experts']],'router_gpu':deployment['router_gpu'],
                 'cached_counts':[len(s) for s in cache.scores],'cache_dir':str(cache_dir),
                 'paired_losses_required':True,'prefetch_batches':6},metrics)
    try:
        for epoch in range(epoch_done+1,config.epochs+1):
            epoch_rows=list(rows['train']); random.Random(config.seed+epoch).shuffle(epoch_rows)
            router.train(); trained=0
            for batch in prefetched_batches(epoch_rows,cache,config.batch_size):
                sample_ids,input_ids,mask,budget,losses=_move_batch(batch,device)
                targets=build_loss_aware_targets(losses,config.temperature)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad(): vectors=embedding(input_ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=router(vectors,mask,budget)
                    loss=loss_aware_router_loss(logits,losses,config.temperature)
                for name,value in (('expert_losses',losses),('router_logits',logits),('router_loss',loss)):
                    _finite_or_raise(name,value,epoch=epoch,sample_ids=sample_ids,loss=loss)
                loss.backward(); grad=torch.nn.utils.clip_grad_norm_(router.parameters(),config.max_grad_norm)
                _finite_or_raise('grad_norm',grad,epoch=epoch,sample_ids=sample_ids,loss=loss)
                optimizer.step(); scheduler.step(); step+=1; trained+=len(sample_ids)
                _log_record({'epoch':epoch,'step':step,'split':'train','loss':float(loss.detach().float().cpu()),
                             'expert_loss_mean':losses.mean(0).tolist(),'batch_samples':len(sample_ids),
                             'cached_counts':[len(s) for s in cache.scores],
                             'paired_samples_trained':trained,'lr':optimizer.param_groups[0]['lr'],
                             'grad_norm':float(grad.detach().float().cpu()),'sample_ids':sample_ids},metrics)
            assert trained==len(rows['train'])
            valid=validate(router,embedding,prefetched_batches(rows['valid'],cache,config.batch_size),
                           device=device,temperature=config.temperature,epoch=epoch,
                           gap_alpha=config.gap_alpha,gap_scale=config.gap_scale)
            _log_record(valid,metrics); regret=float(valid['mean_routing_regret'])
            if regret<best_regret:
                best_regret=regret
                save_checkpoint(best,router=router,optimizer=optimizer,scheduler=scheduler,epoch=epoch,
                                best_validation_regret=best_regret,architecture=architecture,training_config=config,global_step=step)
            save_checkpoint(last,router=router,optimizer=optimizer,scheduler=scheduler,epoch=epoch,
                            best_validation_regret=best_regret,architecture=architecture,training_config=config,global_step=step)
        checkpoint=torch.load(best,map_location=device,weights_only=True)
        router.load_state_dict(checkpoint['router_state_dict']); router.eval()
        selected=0.; oracle=0.; baseline=torch.zeros(2,dtype=torch.float64); choices=torch.zeros(2,dtype=torch.int64); correct=0; n=0
        with torch.inference_mode():
            for batch in prefetched_batches(rows['test'],cache,config.batch_size):
                sample_ids,ids,mask,budget,losses=_move_batch(batch,device)
                with torch.autocast('cuda',dtype=torch.bfloat16): logits=router(embedding(ids),mask,budget)
                l=losses.double().cpu(); pick=logits.argmax(-1).cpu()
                selected+=l.gather(1,pick[:,None]).sum().item(); oracle+=l.min(-1).values.sum().item()
                baseline+=l.sum(0); choices+=torch.bincount(pick,minlength=2)
                correct+=(pick==l.argmin(-1)).sum().item(); n+=len(sample_ids)
        report={'split':'test','rows':n,'expert_ids':list(architecture.expert_ids),'top1_accuracy':correct/n,
                'mean_selected_nll':selected/n,'mean_oracle_nll':oracle/n,'mean_routing_regret':(selected-oracle)/n,
                'constant_expert_mean_nll':(baseline/n).tolist(),'routed_counts':choices.tolist(),
                'gain_over_best_constant_nll':baseline.min().item()/n-selected/n}
        (output/'test_metrics.json').write_text(json.dumps(report,indent=2)+'\n'); _log_record(report,metrics)
        for split,split_rows in rows.items():
            path=cache_dir/f'{split}.jsonl'; temporary=path.with_suffix('.jsonl.tmp')
            with temporary.open('w') as f:
                for r in split_rows:
                    d={k:v for k,v in r.items() if k!='target_ids'}
                    d.update(expert_ids=list(architecture.expert_ids),expert_losses=[s[r['id']]['mean_target_nll'] for s in cache.scores])
                    f.write(json.dumps(d,ensure_ascii=False)+'\n')
            temporary.replace(path)
        _log_record({'stage':'complete','checkpoint':str(best),'test_report':str(output/'test_metrics.json')},metrics)
    finally: cache.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment',required=True); p.add_argument('--config',required=True)
    p.add_argument('--resume'); p.add_argument('--pilot',action='store_true')
    a=p.parse_args(); train_online(a.deployment,a.config,a.resume,a.pilot)
