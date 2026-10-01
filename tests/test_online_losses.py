from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from moqe_router.training.data import TrainingExample
from moqe_router.training.online_losses import PairedExpertLossCache,prefetched_batches


def test_no_training_example_until_both_expert_losses_arrive():
    cache=PairedExpertLossCache.__new__(PairedExpertLossCache)
    cache.executor=ThreadPoolExecutor(max_workers=2); cache.scores=[{},{}]
    first_done=threading.Event(); allow_second=threading.Event()
    row={'id':'same-sample','input_ids':[11,12],'target_ids':[13,99],'max_new_tokens':10}
    def score(index,rows):
        if index==1: assert allow_second.wait(3)
        cache.scores[index][rows[0]['id']]={'mean_target_nll':float(index+1)}
        if index==0: first_done.set()
    cache._ensure_one=score
    with ThreadPoolExecutor(max_workers=1) as caller:
        future=caller.submit(cache.ensure,[row])
        assert first_done.wait(3)
        assert not future.done(), 'one expert must never trigger a Router update'
        allow_second.set(); example=future.result(timeout=3)[0]
    cache.executor.shutdown()
    assert example.sample_id=='same-sample' and example.expert_losses==(1.,2.)


def test_prefetch_preserves_pairing_order_and_each_sample_once():
    rows=[{'id':str(i),'input_ids':[i+1],'max_new_tokens':8} for i in range(10)]
    calls=[]
    class Cache:
        def ensure(self,window):
            calls.append([r['id'] for r in window])
            return [TrainingExample(r['id'],tuple(r['input_ids']),8,(float(r['id']),float(r['id'])+.5)) for r in window]
    batches=list(prefetched_batches(rows,Cache(),batch_size=2,window_batches=2))
    assert [sample for b in batches for sample in b['ids']]==[r['id'] for r in rows]
    assert calls==[['0','1','2','3'],['4','5','6','7'],['8','9']]
    for batch in batches:
        assert batch['expert_losses'][:,1].sub(batch['expert_losses'][:,0]).tolist()==[.5]*len(batch['ids'])


def test_duplicate_ids_in_loss_window_are_rejected():
    cache=PairedExpertLossCache.__new__(PairedExpertLossCache)
    with pytest.raises(ValueError,match='duplicate'):
        cache.ensure([{'id':'a'},{'id':'a'}])
