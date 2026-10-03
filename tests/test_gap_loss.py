import pytest
import torch
from moqe_router.training.objective import gap_weighted_terms, gap_weighted_router_loss


def test_gap_weights_detach_and_normalize():
    expert = torch.tensor([[2., 2.], [2., 2.1], [2., 3.]], requires_grad=True)
    logits = torch.tensor([[0., 0.], [0., 1.], [0., 1.]], requires_grad=True)
    ce, weights = gap_weighted_terms(logits, expert, .1)
    assert torch.allclose(weights, torch.tensor([1., 2., 1. + 2./1.1]))
    assert not weights.requires_grad
    loss = gap_weighted_router_loss(logits, expert)
    assert torch.allclose(loss, (weights * ce).sum()/weights.sum())
    loss.backward()
    assert expert.grad is None
    assert torch.isfinite(logits.grad).all()


def test_alpha_zero_is_unweighted_and_order_invariant():
    logits = torch.tensor([[1., -1.], [-.5, 1.]])
    expert = torch.tensor([[2., 2.01], [3., 2.]])
    ce, _ = gap_weighted_terms(logits, expert, .1, 0.)
    assert torch.allclose(gap_weighted_router_loss(logits, expert, alpha=0.), ce.mean())
    assert torch.allclose(gap_weighted_router_loss(logits, expert),
                          gap_weighted_router_loss(logits.flip(-1), expert.flip(-1)))


@pytest.mark.parametrize('alpha,scale', [(-1., .1), (2., 0.), (float('nan'), .1)])
def test_invalid_weight_configuration(alpha, scale):
    with pytest.raises(ValueError):
        gap_weighted_router_loss(torch.zeros(2, 2), torch.ones(2, 2), alpha=alpha, gap_scale=scale)
