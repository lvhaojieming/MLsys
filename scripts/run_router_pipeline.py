#!/usr/bin/env python3
"""Start isolated experts and train the Router while paired losses are produced."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from moqe_router.expert_scoring import validate_deployment


def health(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response: return json.load(response)
    except Exception: return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/qwen3_14b_router_deployment.json'))
    p.add_argument('--stage',choices=('experts','label','all'),default='all')
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--training-mode',choices=('online','offline'),default='online')
    a=p.parse_args(); config=json.loads(Path(a.config).read_text()); validate_deployment(config)
    output=ROOT/'outputs/qwen3-14b-router-hq-v2'; output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',VLLM_NO_USAGE_STATS='1',VLLM_PLUGINS='')
    processes=[]; deployment=[]
    for index,expert in enumerate(config['experts']):
        existing=health(expert['port'])
        if existing:
            if existing['expert_id']!=expert['expert_id'] or existing['cuda_visible_devices']!=str(expert['gpu']):
                raise RuntimeError('existing endpoint does not match the required expert placement')
        else:
            child=env.copy(); child['CUDA_VISIBLE_DEVICES']=str(expert['gpu'])
            log=(output/f'expert-{index}.log').open('a')
            proc=subprocess.Popen([sys.executable,'-S',str(ROOT/'scripts/serve_quantized_expert.py'),
                                   '--config',a.config,'--expert-index',str(index)],cwd=ROOT,env=child,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((proc,log,index))
        deployment.append({**expert,'endpoint':f'http://127.0.0.1:{expert["port"]}'})
    deadline=time.monotonic()+600
    while not all(health(e['port']) for e in config['experts']):
        for proc,log,index in processes:
            if proc.poll() is not None: raise RuntimeError(f'expert {index} failed; see expert-{index}.log')
        if time.monotonic()>deadline: raise TimeoutError('expert startup timeout')
        time.sleep(2)
    (output/'deployment.json').write_text(json.dumps(deployment,indent=2)+'\n')
    print(json.dumps({'stage':'experts_ready','experts':deployment}),flush=True)
    if a.stage=='experts': return
    destination=ROOT/config['labeled_data']
    if a.stage=='label' or a.training_mode=='offline':
        from label_router_data import label
        destination=label(a.config,limit=8 if a.pilot else 0)
        if a.stage=='label': return
    train_config=json.loads((ROOT/config['training_config']).read_text())
    data_source=ROOT/config['cleaned_data'] if a.training_mode=='online' else destination
    train_config['train_data']=str(data_source/'train.jsonl'); train_config['valid_data']=str(data_source/'valid.jsonl')
    if a.pilot:
        train_config.update(epochs=1,num_workers=0,output_dir=str(output/('online-pilot' if a.training_mode=='online' else 'pilot')))
    effective=output/(f'training-{a.training_mode}-pilot.json' if a.pilot else 'training-effective.json')
    effective.write_text(json.dumps(train_config,indent=2)+'\n')
    child=env.copy(); child['CUDA_VISIBLE_DEVICES']=str(config['router_gpu'])
    if a.training_mode=='online':
        cmd=[sys.executable,'-S',str(ROOT/'scripts/train_router_online.py'),'--deployment',a.config,'--config',str(effective)]
        if a.pilot: cmd+=['--pilot']
    else:
        cmd=[sys.executable,'-S',str(ROOT/'scripts/train_router.py'),'--config',str(effective)]
    last=ROOT/train_config['output_dir']/'checkpoint_last.pt'
    already_complete=False
    if last.exists():
        import torch
        previous=torch.load(last,map_location='cpu',weights_only=True)
        already_complete=previous['epoch']>=train_config['epochs']
        if not already_complete: cmd+=['--resume',str(last)]
    print(json.dumps({'stage':'training','mode':a.training_mode,'gpu':config['router_gpu'],'config':str(effective)}),flush=True)
    if not already_complete: subprocess.run(cmd,cwd=ROOT,env=child,check=True)
    best=ROOT/train_config['output_dir']/'checkpoint_best.pt'
    if a.training_mode=='offline':
        subprocess.run([sys.executable,'-S',str(ROOT/'scripts/evaluate_router.py'),'--checkpoint',str(best),
                        '--data',str(destination/'test.jsonl'),'--output',str(best.parent/'test_metrics.json')],cwd=ROOT,env=child,check=True)
    if not a.pilot:
        log=(output/'router-service.log').open('a')
        service=subprocess.Popen([sys.executable,'-S',str(ROOT/'scripts/serve_router.py'),'--checkpoint',str(best),'--config',a.config],
                                 cwd=ROOT,env=child,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(120):
            if health(config['router_port']): break
            if service.poll() is not None: raise RuntimeError('router service failed; see router-service.log')
            time.sleep(2)
        else: raise TimeoutError('router service startup timeout')
    print(json.dumps({'stage':'complete','checkpoint':str(best),'pilot':a.pilot}),flush=True)


if __name__=='__main__': main()
