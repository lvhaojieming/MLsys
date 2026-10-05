#!/usr/bin/env python3
"""Build correctness and paired-NLL router labels from GSM8K/MMLU train splits."""
import os
os.environ['USE_TORCH'] = '0'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
from queue import Queue, Empty
import re
import threading
import time
import urllib.error
import urllib.request
from transformers import AutoTokenizer
from prepare_router_training_full import requests

EXPERT_IDS = ['qwen3-14b/awq-w4a16/v1','qwen3-14b/gptq-w4a16/v1']
GSM_NUMBER = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
GSM_HASH = re.compile(r'####\s*(-?\d[\d,]*(?:\.\d+)?)')
MMLU_ANSWER = re.compile(r'(?i)(?:answer|correct option)\s*(?:is\s*)?:?\s*\(?\s*([A-D])(?:\b|\))')


def number_answer(text, reference=False):
    found = GSM_HASH.findall(text)
    if not found and not reference:
        boxed = re.findall(r'\\boxed\{\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}', text)
        if boxed:
            found = boxed
        else:
            # Final answers can repeat numbers from the question after the
            # result, e.g. "60 days to write 3 books of 400 pages each".
            plain = text.replace('**', '').replace('__', '')
            markers = list(re.finditer(r'(?i)\b(?:final\s+)?answer\s*(?:is\b|:)', plain))
            if markers:
                conclusion = plain[markers[-1].end():].strip()
                conclusion = conclusion.split('\n\n')[0]
                equal_results = re.findall(r'=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)', conclusion)
                numbers = GSM_NUMBER.findall(conclusion)
                found = equal_results[-1:] if equal_results else numbers[:1]
            else:
                found = GSM_NUMBER.findall(text)
    if not found:
        return None
    try:
        return Decimal(found[-1].replace(',',''))
    except InvalidOperation:
        return None


def mmlu_answer(text):
    text = text.replace('**', '').replace('__', '')
    answers = MMLU_ANSWER.findall(text)
    if answers:
        return answers[-1].upper()
    match = re.match(r'^\s*\(?([A-D])(?:[.)\s:]|$)',text,re.I)
    return match.group(1).upper() if match else None


def post(url, payload):
    request = urllib.request.Request(url+'/v1/completions',data=json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request,timeout=1800) as response:
                return json.load(response)
        except (urllib.error.URLError,TimeoutError,ConnectionError):
            if attempt==5:
                raise
            time.sleep(min(30,2**attempt))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',required=True)
    p.add_argument('--tokenizer',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mmlu-samples',type=int,default=3300)
    p.add_argument('--gsm-samples',type=int,default=3300)
    p.add_argument('--validation-fraction',type=float,default=.10)
    p.add_argument('--seed',type=int,default=20261005)
    p.add_argument('--concurrency-per-endpoint',type=int,default=8)
    p.add_argument('--reuse-gsm-labels', help='Reuse completed GSM8K answers with identical tokenized prompts')
    a=p.parse_args()
    if min(a.mmlu_samples,a.gsm_samples,a.concurrency_per_endpoint)<1 or not 0<a.validation_fraction<.5:
        p.error('positive sample counts and concurrency; validation fraction in (0,.5) required')
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    rng=random.Random(a.seed)

    # MMLU's raw pool is sampled without replacement. GSM uses the complete
    # requested sample from its official train split. Question groups stay intact.
    from prepare_router_training_full import raw_rows, SOURCES
    mmlu_rows=list(raw_rows(Path(a.raw_root),'mmlu'))
    if a.mmlu_samples>len(mmlu_rows):
        raise ValueError('requested MMLU sample exceeds the available auxiliary_train split')
    selected_mmlu=set(rng.sample(range(len(mmlu_rows)),a.mmlu_samples))
    candidates=[]
    selected_subjects={}
    mmlu_iter=iter(mmlu_rows)
    for index,row in enumerate(mmlu_iter):
        if index in selected_mmlu:
            selected_subjects[index]=row.get('subject','unknown')
    del mmlu_rows
    sources=('gsm8k','mmlu')
    for source in sources:
        found=0
        for group,sid,prompt,answer in requests(Path(a.raw_root),source,tokenizer):
            if source=='mmlu' and int(sid) not in selected_mmlu:
                continue
            if source=='gsm8k' and found>=a.gsm_samples:
                break
            if not prompt or not answer:
                continue
            input_ids=tokenizer.apply_chat_template(prompt,tokenize=True,add_generation_prompt=True,
                enable_thinking=False,return_dict=False)
            if isinstance(input_ids,dict):
                input_ids=input_ids['input_ids']
            target_ids=tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id]
            if not input_ids or len(target_ids)>2048 or len(input_ids)+len(target_ids)>12287:
                continue
            content=json.dumps(prompt,ensure_ascii=False,sort_keys=True)
            group_id=source+':'+hashlib.sha256(content.encode()).hexdigest()
            source_id=f'{source}:{sid}'
            category='gsm8k' if source=='gsm8k' else 'mmlu'
            correct_ref=(number_answer(answer,reference=True) if source=='gsm8k'
                         else answer.strip().upper())
            if correct_ref is None:
                continue
            candidates.append(dict(id=source_id,source=source,group_id=group_id,category=category,
                subject=selected_subjects.get(int(sid),'unknown') if source=='mmlu' else None,
                input_ids=input_ids,target_ids=target_ids,max_new_tokens=1024,
                reference_answer=answer,reference_value=str(correct_ref)))
            found+=1
        if source=='gsm8k' and found<a.gsm_samples:
            raise ValueError(f'only {found} eligible GSM8K train samples, requested {a.gsm_samples}')
    source_candidates={s:[r for r in candidates if r['source']==s] for s in sources}
    if min(len(source_candidates[s]) for s in sources)<1:
        raise ValueError('empty task sample after tokenization/context checks')
    # Split by question group. Duplicate text cannot cross the train/valid boundary.
    for source in sources:
        groups=list(dict.fromkeys(r['group_id'] for r in source_candidates[source]))
        rng.shuffle(groups)
        hold_count=max(1,round(len(groups)*a.validation_fraction))
        heldout=set(groups[:hold_count])
        for row in source_candidates[source]:
            row['split']='valid' if row['group_id'] in heldout else 'train'
    with (out/'requests.jsonl').open('w') as f:
        for row in candidates:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest=dict(stage='requests_ready',samples=len(candidates),
        by_source={s:{split:sum(r['source']==s and r['split']==split for r in candidates) for split in ('train','valid')} for s in sources},
        expert_ids=EXPERT_IDS,seed=a.seed,training_source_splits={'gsm8k':'main/train','mmlu':'auxiliary_train'},
        evaluation_test_split_used=False,prompt='one user question, no few-shot exemplars',
        groups_disjoint=True,nll_scoring='mean reference answer token NLL including EOS',
        correctness='GSM8K final number; MMLU answer letter',nll_aux_weight=.1)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)

    hosts=[['10.107.206.208','10.107.206.209','10.107.206.213'],
           ['10.107.206.210','10.107.206.211','10.107.206.216']]
    pools=[[f'http://{host}:{port}' for host in row for port in range(19000,19008)] for row in hosts]
    names=['moqe-qwen3-awq','moqe-qwen3-gptq']
    def check(expert,url):
        with urllib.request.urlopen(url+'/v1/models',timeout=30) as response:
            ids=[model['id'] for model in json.load(response)['data']]
        if names[expert] not in ids:
            raise ValueError(f'expert identity mismatch at {url}: {ids}')
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures=[pool.submit(check,e,url) for e in range(2) for url in pools[e]]
        for future in futures:
            future.result()

    tasks=[Queue(),Queue()]
    reused={}
    if a.reuse_gsm_labels:
        for line in Path(a.reuse_gsm_labels).read_text().splitlines():
            row=json.loads(line)
            if row['source']=='gsm8k':
                reused[row['id']]=row
    for row in candidates:
        if row['id'] in reused:
            prior=reused[row['id']]
            if any(prior[k]!=row[k] for k in ('input_ids','target_ids','split','reference_value','max_new_tokens')):
                raise ValueError('reused GSM8K sample does not match new request')
            continue
        for expert in range(2):
            tasks[expert].put(row)
    lock=threading.Lock()
    out_rows=out/'labeled.jsonl'
    cache_file=out/'expert-loss-cache.jsonl'
    started=time.monotonic()
    completed=set(reused)
    def worker(expert,url,rows_stream,cache_stream):
        while True:
            try:
                row=tasks[expert].get_nowait()
            except Empty:
                return
            generation=post(url,dict(model=names[expert],prompt=row['input_ids'],max_tokens=row['max_new_tokens'],
                temperature=0,seed=42,stop=['</s>','<|im_end|>']))
            if generation['usage']['prompt_tokens']!=len(row['input_ids']):
                raise ValueError('generated answer used an altered prompt')
            answer=generation['choices'][0]['text']
            if row['category']=='gsm8k':
                predicted=number_answer(answer)
                is_correct=predicted is not None and predicted==Decimal(row['reference_value'])
            else:
                predicted=mmlu_answer(answer)
                is_correct=predicted is not None and predicted==row['reference_value']
            score_ids=row['input_ids']+row['target_ids']
            scored=post(url,dict(model=names[expert],prompt=score_ids,max_tokens=1,temperature=0,seed=42,
                echo=True,logprobs=1))
            if scored['usage']['prompt_tokens']!=len(score_ids):
                raise ValueError('expert altered reference-scoring prompt')
            logprobs=scored['choices'][0]['logprobs']['token_logprobs']
            if len(logprobs)!=len(score_ids)+1:
                raise ValueError('reference logprob alignment mismatch')
            target_logprobs=logprobs[len(row['input_ids']):len(score_ids)]
            if len(target_logprobs)!=len(row['target_ids']) or any(x is None for x in target_logprobs):
                raise ValueError('missing reference-token log probability')
            loss=-sum(target_logprobs)/len(target_logprobs)
            output=dict(id=row['id'],expert=expert,answer=answer,correct=int(is_correct),loss=loss,
                finish_reason=generation['choices'][0]['finish_reason'],prompt_tokens=len(row['input_ids']))
            with lock:
                pending.setdefault(row['id'],[None,None])[expert]=output
                pair=pending[row['id']]
                if pair[0] is not None and pair[1] is not None:
                    digest=hashlib.sha256(json.dumps(score_ids).encode()).hexdigest()
                    label=dict(id=row['id'],source=row['source'],category=row['category'],group_id=row['group_id'],
                        split=row['split'],input_ids=row['input_ids'],target_ids=row['target_ids'],
                        max_new_tokens=row['max_new_tokens'],expert_ids=EXPERT_IDS,
                        expert_losses=[pair[0]['loss'],pair[1]['loss']],expert_correctness=[pair[0]['correct'],pair[1]['correct']],
                        expert_answers=[pair[0]['answer'],pair[1]['answer']],reference_answer=row['reference_answer'],
                        expert_finish_reasons=[pair[0]['finish_reason'],pair[1]['finish_reason']],
                        reference_value=row['reference_value'],tokens_sha256=digest)
                    rows_stream.write(json.dumps(label,ensure_ascii=False)+'\n');rows_stream.flush()
                    cache_stream.write(json.dumps(dict(id=row['id'],expert_ids=EXPERT_IDS,
                        expert_losses=label['expert_losses'],target_tokens=len(row['target_ids']),tokens_sha256=digest))+'\n')
                    cache_stream.flush()
                    completed.add(row['id'])
                    if len(completed)%100==0:
                        print(json.dumps(dict(stage='paired_labels',completed=len(completed),total=len(candidates),
                            elapsed_seconds=time.monotonic()-started)),flush=True)
    pending={}
    with out_rows.open('w') as rows_stream,cache_file.open('w') as cache_stream, \
            ThreadPoolExecutor(max_workers=sum(map(len,pools))*a.concurrency_per_endpoint) as workers:
        for row in candidates:
            if row['id'] in reused:
                prior=reused[row['id']]
                prior['expert_correctness']=[int(number_answer(answer)==Decimal(row['reference_value']))
                                             for answer in prior['expert_answers']]
                rows_stream.write(json.dumps(prior,ensure_ascii=False)+'\n')
                cache_stream.write(json.dumps(dict(id=row['id'],expert_ids=EXPERT_IDS,
                    expert_losses=prior['expert_losses'],target_tokens=len(row['target_ids']),
                    tokens_sha256=prior['tokens_sha256']))+'\n')
        rows_stream.flush(); cache_stream.flush()
        print(json.dumps(dict(stage='reused_gsm_labels', samples=len(reused))),flush=True)
        futures=[workers.submit(worker,e,url,rows_stream,cache_stream) for e in range(2)
            for url in pools[e] for _ in range(a.concurrency_per_endpoint)]
        for future in futures:
            future.result()
    if len(completed)!=len(candidates):
        raise RuntimeError(f'incomplete correctness/loss labels: {len(completed)}/{len(candidates)}')
    labels=[json.loads(line) for line in out_rows.read_text().splitlines()]
    for split in ('train','valid'):
        manifest[split+'_metrics']={}
        for source in sources:
            part=[r for r in labels if r['split']==split and r['source']==source]
            manifest[split+'_metrics'][source]=dict(samples=len(part),
                expert_correct=[sum(r['expert_correctness'][i] for r in part)/len(part) for i in range(2)],
                both_correct=sum(all(r['expert_correctness']) for r in part),
                awq_only=sum(r['expert_correctness']==[1,0] for r in part),
                gptq_only=sum(r['expert_correctness']==[0,1] for r in part),
                both_wrong=sum(r['expert_correctness']==[0,0] for r in part))
    manifest['stage']='complete'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    main()
