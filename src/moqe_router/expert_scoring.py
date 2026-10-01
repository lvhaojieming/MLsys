"""Teacher-forced target NLL with explicit prompt/continuation boundaries."""
from __future__ import annotations

import math
from typing import Any


def mean_target_nll(prompt_ids: list[int], target_ids: list[int], prompt_logprobs: list[Any]) -> float:
    """vLLM position i contains log P(token_i | token_0 ... token_{i-1}).

    Include the first answer token and the final EOS, exclude every prompt
    position and the unrelated one-token generation used to request logprobs.
    """
    if not prompt_ids or not target_ids:
        raise ValueError('prompt and target must be nonempty')
    if len(prompt_logprobs) != len(prompt_ids) + len(target_ids):
        raise ValueError('prompt logprob positions do not match teacher-forced input')
    losses = []
    for position, token in enumerate(target_ids, len(prompt_ids)):
        probabilities = prompt_logprobs[position]
        if probabilities is None or token not in probabilities:
            raise ValueError(f'missing observed target token at position {position}')
        value = probabilities[token]
        logprob = float(value.logprob if hasattr(value, 'logprob') else value)
        if not math.isfinite(logprob) or logprob > 1e-5:
            raise ValueError('invalid target token log probability')
        losses.append(-logprob)
    return math.fsum(losses) / len(losses)


def validate_deployment(config: dict) -> None:
    experts = config['experts']
    if len(experts) != 2:
        raise ValueError('this deployment requires exactly two experts')
    if len({e['expert_id'] for e in experts}) != 2 or len({e['port'] for e in experts}) != 2:
        raise ValueError('expert IDs and ports must be distinct')
    def gpu_set(value):
        parts=str(value).split(',')
        if any(not part.strip().isdigit() for part in parts):
            raise ValueError('physical GPU indices must be nonnegative integers')
        return {int(part.strip()) for part in parts}
    gpu_sets = [gpu_set(e['gpu']) for e in experts]
    if not all(gpu_sets) or gpu_sets[0] & gpu_sets[1]:
        raise ValueError('AWQ and GPTQ must use disjoint physical GPUs')
    router_gpus=gpu_set(config['router_gpu'])
    if len(router_gpus)!=1: raise ValueError('router uses one dedicated GPU')
    if router_gpus & set.union(*gpu_sets):
        raise ValueError('router GPU must be separate from the expert GPUs')
