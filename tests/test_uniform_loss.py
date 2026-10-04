import torch
from moqe_router.training.objective import (
    build_loss_aware_targets, loss_aware_terms, loss_aware_router_loss,
)


def test_uniform_weights_and_detached_expert_targets():
    experts = torch.tensor([[2., 2.], [2., 2.1], [2., 3.]], requires_grad=True)
    logits = torch.tensor([[0., 0.], [0., 1.], [0., 1.]], requires_grad=True)
    ce, weights = loss_aware_terms(logits, experts, .1)
    assert torch.equal(weights, torch.ones(3))
    expected = -(build_loss_aware_targets(experts, .1) * logits.log_softmax(-1)).sum(-1).mean()
    loss = loss_aware_router_loss(logits, experts)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert experts.grad is None
    assert torch.isfinite(logits.grad).all()


def test_multiple_experts_and_order_invariance():
    logits = torch.tensor([[1., -.5, 0.], [-.5, 1., 2.]])
    experts = torch.tensor([[2., 2.01, 3.], [3., 2., 2.5]])
    assert torch.allclose(loss_aware_router_loss(logits, experts),
                          loss_aware_router_loss(logits.flip(-1), experts.flip(-1)))


def test_ddp_masked_gradient_matches_global_unweighted_mean():
    torch.manual_seed(42)
    inputs = torch.randn(18, 4)
    experts = torch.rand(18, 2) * 3
    real = torch.tensor([1.] * 16 + [0., 0.])
    parameter = torch.randn(4, 2, requires_grad=True)
    ce, _ = loss_aware_terms(inputs @ parameter, experts, .1)
    expected, = torch.autograd.grad((ce * real).sum()/real.sum(), parameter)
    gradients = []
    for rank in range(6):
        ce, weights = loss_aware_terms(inputs[rank::6] @ parameter, experts[rank::6], .1)
        weights = weights * real[rank::6]
        local_loss = 6 * (ce * weights).sum()/real.sum()
        gradient, = torch.autograd.grad(local_loss, parameter)
        gradients.append(gradient)
    assert torch.allclose(torch.stack(gradients).mean(0), expected, atol=1e-6)
