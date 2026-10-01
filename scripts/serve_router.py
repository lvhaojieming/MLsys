#!/usr/bin/env python3
"""Route a complete request on its own GPU and invoke the selected GPU expert."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--config',default=str(ROOT/'configs/qwen3_14b_router_deployment.json'))
    a=p.parse_args(); config=json.loads(Path(a.config).read_text())
    from moqe_router.expert_scoring import validate_deployment
    validate_deployment(config)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(config['router_gpu']): raise RuntimeError('router GPU placement mismatch')
    import torch
    import httpx
    from fastapi import FastAPI,HTTPException
    from transformers import AutoTokenizer
    import uvicorn
    from evaluate_router import load_router
    from moqe_router.physical import PoolRegistry,Replica,ReplicaState
    from moqe_router.routing import TwoStageRouter
    device=torch.device('cuda:0'); model,embedding,architecture=load_router(a.checkpoint,device)
    if list(architecture.expert_ids)!=[e['expert_id'] for e in config['experts']]: raise RuntimeError('checkpoint expert order mismatch')
    tokenizer=AutoTokenizer.from_pretrained(config['experts'][0]['model_path'],local_files_only=True)
    registry=PoolRegistry(); replicas=[]
    client=httpx.Client(timeout=3600,trust_env=False)
    for e in config['experts']:
        endpoint=f'http://127.0.0.1:{e["port"]}'
        response=client.get(endpoint+'/health'); response.raise_for_status(); h=response.json()
        if h['expert_id']!=e['expert_id'] or h['cuda_visible_devices']!=str(e['gpu']): raise RuntimeError('expert GPU identity mismatch')
        replicas.append(Replica(model_family=architecture.model_family,expert_id=e['expert_id'],pool_id=e['expert_id'],
                               replica_id=f'gpu-{e["gpu"]}',endpoint=endpoint,max_context_tokens=h['max_context_tokens'],
                               state=ReplicaState.READY,checkpoint_version=e['model_path']))
    registry.publish(1,tuple(replicas)); routing=TwoStageRouter(model,registry)
    lock=threading.Lock(); app=FastAPI()
    @app.get('/health')
    def health():
        return {'state':'READY','router_gpu':config['router_gpu'],'checkpoint':str(a.checkpoint),
                'expert_ids':list(architecture.expert_ids),'experts':config['experts']}

    @app.post('/generate')
    def generate(body:dict):
        messages=body.get('messages')
        if messages is None and isinstance(body.get('prompt'),str): messages=[{'role':'user','content':body['prompt']}]
        if not isinstance(messages,list) or not messages: raise HTTPException(422,'provide messages or prompt')
        for m in messages:
            if not isinstance(m,dict) or m.get('role') not in ('system','user','assistant') or not isinstance(m.get('content'),str) or not m['content'].strip():
                raise HTTPException(422,'invalid message')
        if messages[-1]['role']!='user': raise HTTPException(422,'last message must be user')
        budget=body.get('max_new_tokens',256)
        if type(budget) is not int or not 1<=budget<=2048: raise HTTPException(422,'max_new_tokens must be 1..2048')
        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if len(ids)>8192: raise HTTPException(422,'complete prompt exceeds router limit; no truncation')
        tensor=torch.tensor([ids],device=device); mask=torch.ones_like(tensor,dtype=torch.bool)
        with lock,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            decision=routing.route(embedding(tensor),mask,torch.tensor([budget],device=device))
        try:
            response=client.post(decision.replica.endpoint+'/generate',json={'input_ids':ids,'max_new_tokens':budget})
            response.raise_for_status(); result=response.json()
        except httpx.HTTPError as e: raise HTTPException(503,'selected expert inference failed') from e
        if result['expert_id']!=decision.expert_id: raise HTTPException(502,'expert identity mismatch')
        result.update(router_gpu=config['router_gpu'],expert_ranking=list(decision.expert_ranking),prompt_tokens=len(ids))
        return result
    uvicorn.run(app,host='127.0.0.1',port=config['router_port'],log_level='warning')


if __name__=='__main__': main()
