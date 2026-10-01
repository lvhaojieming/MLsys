from types import SimpleNamespace
import math

import pytest

from moqe_router.expert_scoring import mean_target_nll,validate_deployment


def test_target_boundary_includes_first_token_and_eos_excludes_prompt():
    # The huge prompt loss must not contribute; both answer tokens must.
    probabilities=[None,{20:SimpleNamespace(logprob=-1000)},
                   {30:SimpleNamespace(logprob=-2)}, {99:SimpleNamespace(logprob=-4)}]
    assert mean_target_nll([10,20],[30,99],probabilities)==3


def test_missing_or_nonfinite_target_logprobs_fail_closed():
    with pytest.raises(ValueError): mean_target_nll([10],[99],[None,None])
    with pytest.raises(ValueError): mean_target_nll([10],[99],[None,{99:-math.inf}])
    with pytest.raises(ValueError): mean_target_nll([10],[99],[None,{99:0.2}])


def test_experts_cannot_share_a_gpu():
    c={'experts':[{'expert_id':'awq','gpu':'0','port':1},{'expert_id':'gptq','gpu':'1','port':2}],'router_gpu':'2'}
    validate_deployment(c)
    c['experts'][1]['gpu']='0'
    with pytest.raises(ValueError,match='disjoint'): validate_deployment(c)
    c['experts'][1]['gpu']='00'
    with pytest.raises(ValueError,match='disjoint'): validate_deployment(c)
