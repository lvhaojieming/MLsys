#!/usr/bin/env python3
"""Dedicated local GPU expert: exact-token teacher scoring and generation."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from moqe_router.expert_scoring import mean_target_nll, validate_deployment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/qwen3_14b_router_deployment.json'))
    parser.add_argument('--expert-index',type=int,required=True,choices=(0,1))
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text()); validate_deployment(config)
    expert=config['experts'][args.expert_index]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(expert['gpu']):
        raise RuntimeError('CUDA_VISIBLE_DEVICES must exactly match the dedicated expert GPU')
    from fastapi import FastAPI, HTTPException
    import uvicorn
    from vllm import LLM, SamplingParams
    engine=LLM(model=expert['model_path'],tokenizer=config['experts'][0]['model_path'],
               dtype='float16',quantization=expert['quantization'],
               max_model_len=config['max_context_tokens'],gpu_memory_utilization=config['gpu_memory_utilization'],
               max_num_seqs=32,max_num_batched_tokens=16384,enforce_eager=True,seed=42,
               enable_prefix_caching=False)
    # No shared inference objects or CUDA allocations between experts.
    lock=threading.Lock(); app=FastAPI(); vocab=engine.llm_engine.model_config.get_vocab_size()
    def tokens(ids):
        if not isinstance(ids,list) or not ids or any(type(t) is not int or t<0 or t>=vocab for t in ids):
            raise HTTPException(422,'token IDs must be a nonempty valid integer list')
        return ids

    @app.get('/health')
    def health():
        return {**expert,'state':'READY','pid':os.getpid(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                'max_context_tokens':config['max_context_tokens'],'dtype':'float16','scoring':'mean_target_nll_including_eos'}

    @app.post('/score')
    def score(body:dict):
        rows=body.get('rows')
        if not isinstance(rows,list) or not 1<=len(rows)<=64: raise HTTPException(422,'1..64 rows required')
        prompts=[]
        for row in rows:
            ids=tokens(row['input_ids'])+tokens(row['target_ids'])
            if len(ids)+1>config['max_context_tokens']: raise HTTPException(422,'full prompt+target exceeds context; no truncation')
            prompts.append({'prompt_token_ids':ids})
        with lock:
            outputs=engine.generate(prompts,SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=0),use_tqdm=False)
        result=[]
        for row,output in zip(rows,outputs):
            expected=row['input_ids']+row['target_ids']
            if output.prompt_token_ids!=expected: raise RuntimeError('expert changed supplied token IDs')
            loss=mean_target_nll(row['input_ids'],row['target_ids'],output.prompt_logprobs)
            result.append({'id':row['id'],'mean_target_nll':loss,'target_tokens':len(row['target_ids'])})
        return {'expert_id':expert['expert_id'],'gpu':expert['gpu'],'rows':result}

    @app.post('/generate')
    def generate(body:dict):
        ids=tokens(body.get('input_ids')); budget=body.get('max_new_tokens',256)
        if type(budget) is not int or not 1<=budget<=2048: raise HTTPException(422,'max_new_tokens must be 1..2048')
        if len(ids)+budget>config['max_context_tokens']: raise HTTPException(422,'request exceeds context')
        with lock:
            outputs=engine.generate([{'prompt_token_ids':ids}],SamplingParams(temperature=0,max_tokens=budget),use_tqdm=False)
        output=outputs[0].outputs[0]
        return {'expert_id':expert['expert_id'],'gpu':expert['gpu'],'text':output.text,
                'generated_ids':output.token_ids,'finish_reason':output.finish_reason}
    uvicorn.run(app,host='127.0.0.1',port=expert['port'],log_level='warning')


if __name__=='__main__': main()
