#!/usr/bin/env python3
"""Generate both experts' zero-shot answers and score recorded question-only routing."""
import os
os.environ['USE_TORCH'] = '0'
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from queue import Queue, Empty
import re
import threading
import time
import urllib.error
import urllib.request
from transformers import AutoTokenizer

NUMBER = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
HASH_NUMBER = re.compile(r'####\s*(-?\d[\d,]*(?:\.\d+)?)')


def extract(text, require_hash=False):
    matches = HASH_NUMBER.findall(text)
    if not matches and not require_hash:
        matches = NUMBER.findall(text)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(',',''))
    except InvalidOperation:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests',required=True)
    p.add_argument('--router-logits',required=True)
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--concurrency-per-endpoint',type=int,default=8)
    a = p.parse_args()
    if a.concurrency_per_endpoint < 1:
        p.error('positive concurrency required')
    root = Path(a.output)
    root.mkdir(parents=True,exist_ok=True)
    rows = [json.loads(s) for s in Path(a.requests).read_text().splitlines()]
    logits = {r['id']:r['logits'] for r in map(json.loads,Path(a.router_logits).read_text().splitlines())}
    if len(rows)!=1319 or set(logits)!={r['id'] for r in rows}:
        raise ValueError('complete paired 1319-question inputs required')
    tok = AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    for row in rows:
        row['question_ids'] = tok.apply_chat_template([{'role':'user','content':row['question']}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        row['selected'] = max(range(2),key=lambda j:logits[row['id']][j])
        row['prompt_sha256'] = hashlib.sha256(json.dumps(row['question_ids']).encode()).hexdigest()
    protocol = dict(samples=len(rows),prompt='user message containing original question only',
        thinking=False,temperature=0,seed=42,max_tokens=1024,stop=['Question:','</s>','<|im_end|>'],
        numeric_scoring='prefer final #### number; otherwise last numeric value; Decimal exact comparison',
        hash_format_scoring='require #### number; Decimal exact comparison',
        router_selection_counts=[sum(r['selected']==j for r in rows) for j in range(2)])
    manifest = root/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:
        raise ValueError('existing output has a different evaluation protocol')
    manifest.write_text(json.dumps(protocol,indent=2))
    by_id = {r['id']:r for r in rows}
    results = {}
    output = root/'expert-answers.jsonl'
    if output.exists():
        for line in output.open():
            r = json.loads(line)
            if r['prompt_sha256']!=by_id[r['id']]['prompt_sha256']:
                raise ValueError('cached input mismatch')
            results[(r['id'],r['expert'])] = r
    hosts = [['10.107.206.208','10.107.206.209','10.107.206.213'],['10.107.206.210','10.107.206.211','10.107.206.216']]
    pools = [[f'http://{h}:{port}' for h in group for port in range(19000,19008)] for group in hosts]
    names = ['moqe-qwen3-awq','moqe-qwen3-gptq']
    def health(e,url):
        with urllib.request.urlopen(url+'/v1/models',timeout=15) as response:
            models = json.load(response)['data']
        if names[e] not in [m['id'] for m in models]:
            raise ValueError('expert identity mismatch')
    with ThreadPoolExecutor(max_workers=48) as executor:
        for f in [executor.submit(health,e,url) for e in range(2) for url in pools[e]]:
            f.result()
    queues = [Queue(),Queue()]
    for e in range(2):
        for row in rows:
            if (row['id'],e) not in results:
                queues[e].put(row)
    lock = threading.Lock()
    started = time.monotonic()
    def worker(e,url,stream):
        while True:
            try:
                row = queues[e].get_nowait()
            except Empty:
                return
            payload = dict(model=names[e],prompt=row['question_ids'],max_tokens=1024,
                temperature=0,seed=42,stop=protocol['stop'])
            request = urllib.request.Request(url+'/v1/completions',data=json.dumps(payload).encode(),
                headers={'Content-Type':'application/json'})
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(request,timeout=1800) as response:
                        answer = json.load(response)
                    break
                except (urllib.error.URLError,TimeoutError,ConnectionError):
                    if attempt==3:
                        raise
                    time.sleep(2**attempt)
            if answer['usage']['prompt_tokens']!=len(row['question_ids']):
                raise ValueError('served prompt token count mismatch')
            choice = answer['choices'][0]
            r = dict(id=row['id'],expert=e,endpoint=url,prompt_sha256=row['prompt_sha256'],
                answer=choice['text'],finish_reason=choice['finish_reason'],usage=answer['usage'])
            with lock:
                results[(row['id'],e)] = r
                stream.write(json.dumps(r)+'\n');stream.flush()
                if len(results)%100==0:
                    print(json.dumps(dict(stage='generating',completed=len(results),total=2*len(rows),
                        elapsed_seconds=time.monotonic()-started)),flush=True)
    with output.open('a') as stream, ThreadPoolExecutor(max_workers=48*a.concurrency_per_endpoint) as executor:
        jobs = [executor.submit(worker,e,url,stream) for e in range(2) for url in pools[e]
            for _ in range(a.concurrency_per_endpoint)]
        for future in jobs:
            future.result()
    metrics = {}
    with (root/'routed-answers.jsonl').open('w') as stream:
        for row in rows:
            gold = extract(row['reference'],True)
            if gold is None:
                raise ValueError('missing reference final answer')
            correct = []
            hash_correct = []
            for e in range(2):
                text = results[(row['id'],e)]['answer']
                correct.append(extract(text)==gold)
                hash_correct.append(extract(text,True)==gold)
            stream.write(json.dumps(dict(id=row['id'],question=row['question'],reference=row['reference'],
                selected_expert=row['selected'],expert_correct=correct,expert_hash_correct=hash_correct,
                expert_answers=[results[(row['id'],e)]['answer'] for e in range(2)]))+'\n')
            for name,values in [('numeric_final',correct),('hash_format',hash_correct)]:
                counts = metrics.setdefault(name,Counter())
                counts['awq_correct']+=values[0];counts['gptq_correct']+=values[1]
                counts['router_correct']+=values[row['selected']]
                counts['oracle_correct']+=any(values)
                category = 'both_correct' if all(values) else 'awq_only' if values[0] else 'gptq_only' if values[1] else 'both_wrong'
                counts[category]+=1
                counts[category+'_router_correct']+=values[row['selected']]
    report = dict(samples=len(rows),router_checkpoint_step=100000,mode='fresh zero-shot expert generation',
        protocol=protocol,elapsed_seconds=time.monotonic()-started,
        finish_reasons={str(e):dict(Counter(results[(r['id'],e)]['finish_reason'] for r in rows)) for e in range(2)},
        metrics={name:dict(counts=dict(counts),accuracies={key:counts[key]/len(rows) for key in ('awq_correct','gptq_correct','router_correct','oracle_correct')}) for name,counts in metrics.items()})
    (root/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
