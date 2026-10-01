#!/usr/bin/env python3
"""Score cleaned requests on separate experts and join losses in fixed order."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from moqe_router.expert_scoring import validate_deployment


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def label(config_path,limit=0):
    import httpx
    config=json.loads(Path(config_path).read_text()); validate_deployment(config)
    source=ROOT/config['cleaned_data']; destination=ROOT/config['labeled_data']
    if limit: destination=destination.with_name(destination.name+'-pilot')
    destination.mkdir(parents=True,exist_ok=True)
    validation=json.loads((source/'validation.json').read_text())
    if validation['status']!='passed': raise ValueError('cleaned data validation must pass')
    source_hashes={s:digest(source/f'{s}.jsonl') for s in ('train','valid','test')}
    identity={'source_hashes':source_hashes,'expert_ids':[e['expert_id'] for e in config['experts']],
              'models':[e['model_path'] for e in config['experts']],
              'quantization':[e['quantization'] for e in config['experts']],
              'dtype':'float16','target_contract':'mean target NLL including EOS; identical token IDs; causal next-token logprobs',
              'limit_per_split':limit,'pilot_selection':'source balanced, first and longest prompts' if limit else None}
    identity_path=destination/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('existing label directory belongs to a different source/model contract')
    identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    all_rows=[]
    for split in ('train','valid','test'):
        split_rows=[]
        with (source/f'{split}.jsonl').open() as f:
            for i,line in enumerate(f):
                r=json.loads(line); r['split']=split; split_rows.append(r)
        if limit:
            buckets=defaultdict(list)
            for r in split_rows: buckets[r['source']].append(r)
            queues=[]
            for rows in buckets.values():
                longest=max(rows,key=lambda r:len(r['input_ids']))
                ordered=[rows[0]]+([longest] if longest['id']!=rows[0]['id'] else [])
                ordered.extend(r for r in rows if r['id'] not in {x['id'] for x in ordered[:2]})
                queues.append(ordered)
            selected=[]; level=0
            while len(selected)<min(limit,len(split_rows)):
                for queue in queues:
                    if level<len(queue) and len(selected)<limit: selected.append(queue[level])
                level+=1
            split_rows=selected
        all_rows.extend(split_rows)
    # Sorting only changes scoring order, not membership or original splits.
    all_rows.sort(key=lambda r:len(r['input_ids'])+len(r['target_ids']))
    def worker(index):
        expert=config['experts'][index]
        client=httpx.Client(base_url=f'http://127.0.0.1:{expert["port"]}',timeout=3600,trust_env=False)
        health=client.get('/health'); health.raise_for_status(); h=health.json()
        if h['expert_id']!=expert['expert_id'] or h['cuda_visible_devices']!=str(expert['gpu']):
            raise ValueError('expert endpoint/GPU identity mismatch')
        path=destination/f'expert-{index}.jsonl'; scores={}
        if path.exists():
            for line in path.open():
                d=json.loads(line)
                if d['id'] in scores: raise ValueError('duplicate expert score')
                scores[d['id']]=d
        pending=[r for r in all_rows if r['id'] not in scores]
        size=config['scoring_batch_size']
        with path.open('a') as f:
            for start in range(0,len(pending),size):
                batch=pending[start:start+size]
                payload={'rows':[{k:r[k] for k in ('id','input_ids','target_ids')} for r in batch]}
                response=client.post('/score',json=payload); response.raise_for_status(); result=response.json()
                if result['expert_id']!=expert['expert_id'] or str(result['gpu'])!=str(expert['gpu']):
                    raise ValueError('scoring returned the wrong expert/GPU')
                if [d['id'] for d in result['rows']]!=[r['id'] for r in batch]:
                    raise ValueError('scoring changed request order or IDs')
                for r,d in zip(batch,result['rows']):
                    if d['target_tokens']!=len(r['target_ids']) or not math.isfinite(d['mean_target_nll']) or d['mean_target_nll']<0:
                        raise ValueError('invalid expert target score')
                    d.update(expert_id=expert['expert_id'],gpu=expert['gpu'],
                             tokens_sha256=hashlib.sha256(json.dumps(r['input_ids']+r['target_ids']).encode()).hexdigest())
                    f.write(json.dumps(d)+'\n'); scores[d['id']]=d
                f.flush()
                print(json.dumps({'stage':'label','expert':index,'gpu':expert['gpu'],'scored':len(scores),'total':len(all_rows)}),flush=True)
        client.close()
        return scores
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,i) for i in range(2)]
        scores=[f.result() for f in futures]
    counts=Counter(); winners=Counter(); groups={}; seen=set()
    streams={s:(destination/f'{s}.jsonl.tmp').open('w') for s in ('train','valid','test')}
    try:
        for r in all_rows:
            assert r['id'] not in seen; seen.add(r['id'])
            assert groups.setdefault(r['group_id'],r['split'])==r['split']
            fingerprint=hashlib.sha256(json.dumps(r['input_ids']+r['target_ids']).encode()).hexdigest()
            if any(s[r['id']]['tokens_sha256']!=fingerprint for s in scores): raise ValueError('scored tokens differ from source')
            losses=[s[r['id']]['mean_target_nll'] for s in scores]
            row={k:r[k] for k in ('id','source','source_id','group_id','prompt_hash','split','input_ids','max_new_tokens')}
            row.update(expert_ids=identity['expert_ids'],expert_losses=losses,target_tokens=len(r['target_ids']),
                       scoring_tokens_sha256=fingerprint)
            streams[r['split']].write(json.dumps(row,ensure_ascii=False)+'\n')
            counts[r['split']]+=1; winners[min(range(2),key=lambda i:losses[i])]+=1
    finally:
        for f in streams.values(): f.close()
    for split in streams: (destination/f'{split}.jsonl.tmp').replace(destination/f'{split}.jsonl')
    summary={**identity,'status':'complete','rows':len(seen),'split_counts':counts,'oracle_winner_counts':winners,
             'sha256':{s:digest(destination/f'{s}.jsonl') for s in streams}}
    (destination/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'stage':'label_complete','rows':len(seen),'counts':counts,'oracle_winners':winners}),flush=True)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/qwen3_14b_router_deployment.json'))
    p.add_argument('--limit-per-split',type=int,default=0)
    a=p.parse_args(); label(a.config,a.limit_per_split)
