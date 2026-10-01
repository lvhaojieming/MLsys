#!/usr/bin/env python3
"""Evaluate held-out routing regret against both constant expert policies."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))


def load_router(checkpoint_path,device):
    import torch
    from moqe_router.config import RouterArchitecture
    from moqe_router.model import EmbeddingRouter
    from moqe_router.training.embedding import FrozenEmbeddingProvider
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    settings=checkpoint['architecture_config']; settings['expert_ids']=tuple(settings['expert_ids'])
    architecture=RouterArchitecture(**settings)
    router=EmbeddingRouter(architecture).to(device)
    router.load_state_dict(checkpoint['router_state_dict']); router.eval()
    config=checkpoint['training_config']
    embedding=FrozenEmbeddingProvider.from_checkpoint(config['base_model_path'],weight_key=config['embedding_weight_key'],
                                                    embedding_dim=architecture.embedding_dim).to(device).eval()
    return router,embedding,architecture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    a=p.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from moqe_router.training.data import RequestDataset,collate_requests
    device=torch.device('cuda:0'); router,embedding,architecture=load_router(a.checkpoint,device)
    data=RequestDataset(a.data,architecture.expert_ids,8192)
    loader=DataLoader(data,batch_size=4,collate_fn=collate_requests)
    count=0; correct=0; selected_sum=0.; oracle_sum=0.; baseline=torch.zeros(2,dtype=torch.float64); choices=torch.zeros(2,dtype=torch.int64)
    with torch.inference_mode():
        for b in loader:
            ids=b['input_ids'].to(device); mask=b['attention_mask'].to(device); budget=b['max_new_tokens'].to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=router(embedding(ids),mask,budget)
            losses=b['expert_losses'].double(); chosen=logits.argmax(-1).cpu()
            selected_sum+=losses.gather(1,chosen[:,None]).sum().item()
            oracle_sum+=losses.min(-1).values.sum().item()
            baseline+=losses.sum(0); choices+=torch.bincount(chosen,minlength=2)
            correct+=(chosen==losses.argmin(-1)).sum().item(); count+=len(chosen)
    result={'split':'test','rows':count,'expert_ids':list(architecture.expert_ids),'top1_accuracy':correct/count,
            'mean_selected_nll':selected_sum/count,'mean_oracle_nll':oracle_sum/count,
            'mean_routing_regret':(selected_sum-oracle_sum)/count,
            'constant_expert_mean_nll':(baseline/count).tolist(),'routed_counts':choices.tolist(),
            'gain_over_best_constant_nll':baseline.min().item()/count-selected_sum/count}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
