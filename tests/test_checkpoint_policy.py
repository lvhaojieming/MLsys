import pytest
from moqe_router.training.checkpoint_policy import BestCheckpointPolicy


def test_periodic_validation_and_final_without_duplicate():
    policy = BestCheckpointPolicy()
    assert not policy.due(4999)
    assert policy.due(5000)
    assert policy.observe(5000, .2)
    assert not policy.due(5000, final=True)
    assert not policy.due(9999)
    assert policy.due(10000)
    assert not policy.observe(10000, .3)
    assert policy.due(10022, final=True)
    assert policy.observe(10022, .1)
    assert policy.best_regret == .1


def test_short_run_and_tied_metric_keep_best():
    policy = BestCheckpointPolicy()
    assert policy.due(22, final=True)
    assert policy.observe(22, .1)
    assert not policy.observe(5000, .1)
    assert not policy.observe(10000, .2)
    assert policy.best_regret == .1


def test_invalid_metrics_do_not_replace_best():
    policy = BestCheckpointPolicy()
    policy.observe(5000, .1)
    with pytest.raises(ValueError):
        policy.observe(10000, float('nan'))
    assert policy.best_regret == .1


@pytest.mark.parametrize('interval', [0, -1, 1.5, True])
def test_invalid_interval(interval):
    with pytest.raises(ValueError):
        BestCheckpointPolicy(interval)
