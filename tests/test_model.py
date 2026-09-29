import unittest

try:
    import torch
except ImportError:  # Allows pure-control-plane tests without ML dependencies.
    torch = None

if torch is not None:
    from moqe_router.config import RouterArchitecture
    from moqe_router.model import EmbeddingRouter
    from moqe_router.physical import PoolRegistry, Replica, ReplicaState
    from moqe_router.routing import TwoStageRouter


@unittest.skipIf(torch is None, "PyTorch is required for L1 architecture tests")
class EmbeddingRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = RouterArchitecture(
            model_family="qwen3-14b",
            expert_ids=("awq-v1", "gptq-v1", "int8-v1"),
            embedding_dim=8,
            hidden_dim=16,
            num_heads=4,
            chunk_size=3,
            dropout=0,
        )
        self.model = EmbeddingRouter(self.config).eval()

    def test_default_has_two_transformer_layers(self) -> None:
        self.assertEqual(self.config.encoder_layers, 2)
        self.assertEqual(len(self.model.local_encoder.layers), 1)
        self.assertEqual(len(self.model.global_encoder.layers), 1)

    def test_one_logit_vector_per_sequence_and_padding_invariance(self) -> None:
        embeddings = torch.randn(2, 7, 8)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
        budget = torch.tensor([128, 512])
        with torch.inference_mode():
            first = self.model(embeddings, mask, budget)
            changed_padding = embeddings.clone()
            changed_padding[0, 5:] = 1000
            second = self.model(changed_padding, mask, budget)
        self.assertEqual(first.shape, (2, 3))
        torch.testing.assert_close(first[0], second[0])

    def test_invalid_padding_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "right-padded"):
            self.model(torch.randn(1, 3, 8), torch.tensor([[1, 0, 1]]), torch.tensor([1]))

    def test_every_valid_token_contributes_to_the_decision(self) -> None:
        config = RouterArchitecture(
            model_family="qwen3-14b",
            expert_ids=("awq-v1",),
            embedding_dim=4,
            hidden_dim=8,
            num_heads=2,
            chunk_size=3,
            dropout=0,
        )
        model = EmbeddingRouter(config).eval()
        embeddings = torch.randn(1, 17, 4, requires_grad=True)
        logits = model(embeddings, torch.ones(1, 17), torch.tensor([12]))
        logits.sum().backward()
        self.assertEqual(logits.shape, (1, 1))
        self.assertTrue(bool((embeddings.grad.abs().sum(dim=-1) > 0).all()))

    def test_two_stage_route_masks_unavailable_expert(self) -> None:
        # Force a stable rank: int8 > gptq > awq; only gptq is READY.
        with torch.no_grad():
            self.model.head[-1].weight.zero_()
            self.model.head[-1].bias.copy_(torch.tensor([0.0, 1.0, 2.0]))
        registry = PoolRegistry()
        registry.publish(
            1,
            (
                Replica(
                    model_family="qwen3-14b",
                    expert_id="gptq-v1",
                    pool_id="p1",
                    replica_id="r1",
                    endpoint="http://r1:8000",
                    max_context_tokens=100,
                    state=ReplicaState.READY,
                    checkpoint_version="sha-gptq-1",
                ),
            ),
        )
        decision = TwoStageRouter(self.model, registry).route(
            torch.randn(1, 4, 8), torch.ones(1, 4), torch.tensor([10])
        )
        self.assertEqual(decision.expert_id, "gptq-v1")
        self.assertEqual(decision.replica.replica_id, "r1")
        self.assertEqual(decision.expert_ranking[0], "int8-v1")


if __name__ == "__main__":
    unittest.main()
