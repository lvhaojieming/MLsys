import unittest

from moqe_router.physical import (
    PhysicalRouter,
    PoolRegistry,
    Replica,
    ReplicaState,
    RouteUnavailable,
)


def replica(pool: str, name: str, state: ReplicaState = ReplicaState.READY) -> Replica:
    return Replica(
        model_family="qwen3-14b",
        expert_id="awq-v1",
        pool_id=pool,
        replica_id=name,
        endpoint=f"http://{name}:8000",
        max_context_tokens=8192,
        state=state,
        checkpoint_version="sha-awq-1",
    )


class PhysicalRouterTests(unittest.TestCase):
    def test_ready_filter_and_pool_then_replica_round_robin(self) -> None:
        registry = PoolRegistry()
        snapshot = registry.publish(
            1,
            (
                replica("a800", "a1"),
                replica("a800", "a2"),
                replica("a800", "a3", ReplicaState.DRAINING),
                replica("huawei", "h1"),
            ),
        )
        selector = PhysicalRouter()
        choices = [
            selector.choose(
                snapshot,
                model_family="qwen3-14b",
                expert_id="awq-v1",
                required_context_tokens=4096,
            ).replica_id
            for _ in range(4)
        ]
        self.assertEqual(choices, ["a1", "h1", "a2", "h1"])
        self.assertEqual(snapshot.eligible_experts("qwen3-14b", 8193), set())

    def test_new_epoch_replaces_ready_set_atomically(self) -> None:
        registry = PoolRegistry()
        old = registry.publish(1, (replica("a800", "a1"),))
        new = registry.publish(2, (replica("a800", "a1", ReplicaState.DRAINING),))
        self.assertEqual(old.eligible_experts("qwen3-14b", 100), {"awq-v1"})
        self.assertEqual(new.eligible_experts("qwen3-14b", 100), set())
        with self.assertRaises(ValueError):
            registry.publish(2, ())
        with self.assertRaises(RouteUnavailable):
            PhysicalRouter().choose(
                new,
                model_family="qwen3-14b",
                expert_id="awq-v1",
                required_context_tokens=100,
            )

    def test_pool_cannot_mix_expert_versions(self) -> None:
        registry = PoolRegistry()
        other = Replica(
            model_family="qwen3-14b",
            expert_id="gptq-v1",
            pool_id="a800",
            replica_id="g1",
            endpoint="http://g1:8000",
            max_context_tokens=8192,
            state=ReplicaState.READY,
            checkpoint_version="sha-gptq-1",
        )
        with self.assertRaises(ValueError):
            registry.publish(1, (replica("a800", "a1"), other))

    def test_one_expert_id_cannot_hide_different_checkpoint_versions(self) -> None:
        registry = PoolRegistry()
        original = replica("a800", "a1")
        changed = Replica(
            model_family="qwen3-14b",
            expert_id="awq-v1",
            pool_id="huawei",
            replica_id="h1",
            endpoint="http://h1:8000",
            max_context_tokens=8192,
            state=ReplicaState.READY,
            checkpoint_version="sha-different",
        )
        with self.assertRaisesRegex(ValueError, "multiple checkpoint versions"):
            registry.publish(1, (original, changed))


if __name__ == "__main__":
    unittest.main()
