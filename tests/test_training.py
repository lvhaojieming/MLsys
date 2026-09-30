import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from moqe_router.config import RouterArchitecture
from moqe_router.model import EmbeddingRouter
from moqe_router.training.data import RequestDataset, collate_requests
from moqe_router.training.embedding import FrozenEmbeddingProvider
from moqe_router.training.metrics import routing_metrics
from moqe_router.training.objective import (
    build_loss_aware_targets,
    soft_target_cross_entropy,
)
from moqe_router.training.trainer import (
    TrainingConfig,
    build_scheduler,
    load_checkpoint,
    save_checkpoint,
    train,
)


def tiny_architecture() -> RouterArchitecture:
    return RouterArchitecture(
        model_family="test",
        expert_ids=("awq", "gptq", "int8"),
        embedding_dim=4,
        hidden_dim=8,
        num_heads=2,
        chunk_size=2,
        dropout=0,
    )


def tiny_training_config(tmp_path) -> TrainingConfig:
    return TrainingConfig(
        architecture_config=str(tmp_path / "architecture.json"),
        train_data=str(tmp_path / "train.jsonl"),
        valid_data=str(tmp_path / "valid.jsonl"),
        base_model_path=str(tmp_path),
        embedding_weight_key="model.embed_tokens.weight",
        output_dir=str(tmp_path / "output"),
        seed=42,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        warmup_ratio=0.05,
        min_lr_ratio=0.1,
        max_grad_norm=1.0,
        temperature=0.1,
        precision="bf16",
        num_workers=0,
        pin_memory=False,
        max_prompt_tokens=16,
    )


def test_soft_target_is_normalized_and_prefers_lower_loss():
    target = build_loss_aware_targets(torch.tensor([[1.0, 2.0, 3.0]]), 0.1)
    assert target[0, 0] > target[0, 1] > target[0, 2]
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(1))


def test_soft_target_is_invariant_to_per_request_loss_offset():
    first = build_loss_aware_targets(torch.tensor([[1.0, 2.0, 3.0]]), 0.1)
    shifted = build_loss_aware_targets(torch.tensor([[101.0, 102.0, 103.0]]), 0.1)
    torch.testing.assert_close(first, shifted)


def test_dataset_and_dynamic_right_padding(tmp_path):
    rows = [
        {
            "id": "short",
            "input_ids": [3, 4, 5],
            "max_new_tokens": 8,
            "expert_losses": [1.0, 2.0, 3.0],
        },
        {
            "id": "long",
            "input_ids": [6, 7, 8, 9, 10],
            "max_new_tokens": 16,
            "expert_losses": [3.0, 2.0, 1.0],
        },
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    dataset = RequestDataset(path, ("awq", "gptq", "int8"), max_prompt_tokens=8)
    batch = collate_requests([dataset[0], dataset[1]])

    assert batch["ids"] == ["short", "long"]
    assert batch["input_ids"].dtype == torch.int64
    assert batch["input_ids"].tolist() == [[3, 4, 5, 0, 0], [6, 7, 8, 9, 10]]
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["attention_mask"].tolist() == [
        [True, True, True, False, False],
        [True, True, True, True, True],
    ]
    assert batch["max_new_tokens"].dtype == torch.int64
    assert batch["expert_losses"].dtype == torch.float32
    assert batch["expert_losses"].shape == (2, 3)


def test_dataset_rejects_duplicate_ids_excess_length_and_wrong_expert_order(tmp_path):
    duplicate = {
        "id": "same",
        "input_ids": [1],
        "max_new_tokens": 1,
        "expert_losses": [1.0, 2.0, 3.0],
    }
    path = tmp_path / "duplicate.jsonl"
    path.write_text(json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n")
    with pytest.raises(ValueError, match="duplicate id"):
        RequestDataset(path, ("awq", "gptq", "int8"), 4)

    too_long = {**duplicate, "id": "long", "input_ids": [1, 2, 3, 4, 5]}
    path.write_text(json.dumps(too_long) + "\n")
    with pytest.raises(ValueError, match="exceeding max_prompt_tokens"):
        RequestDataset(path, ("awq", "gptq", "int8"), 4)

    wrong_order = {
        **duplicate,
        "id": "order",
        "expert_ids": ["gptq", "awq", "int8"],
    }
    path.write_text(json.dumps(wrong_order) + "\n")
    with pytest.raises(ValueError, match="exactly match"):
        RequestDataset(path, ("awq", "gptq", "int8"), 4)


def test_frozen_embedding_stops_backward_and_router_modules_have_finite_gradients():
    torch.manual_seed(1)
    architecture = tiny_architecture()
    provider = FrozenEmbeddingProvider(torch.randn(16, 4), embedding_dim=4)
    router = EmbeddingRouter(architecture)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]], dtype=torch.int64)
    attention_mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]]
    )
    budgets = torch.tensor([8, 16], dtype=torch.int64)
    expert_losses = torch.tensor([[1.0, 1.5, 2.0], [2.0, 1.0, 3.0]])

    embeddings = provider(input_ids)
    target = build_loss_aware_targets(expert_losses, 0.1)
    loss = soft_target_cross_entropy(router(embeddings, attention_mask, budgets), target)
    loss.backward()

    assert provider.weight.grad is None
    for name in (
        "input_projection",
        "local_encoder",
        "token_pool_score",
        "global_encoder",
        "chunk_pool_score",
        "head",
    ):
        gradients = [
            parameter.grad
            for parameter in getattr(router, name).parameters()
            if parameter.requires_grad
        ]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_optimizer_updates_router_but_not_frozen_embedding():
    torch.manual_seed(2)
    architecture = tiny_architecture()
    provider = FrozenEmbeddingProvider(torch.randn(16, 4), embedding_dim=4)
    router = EmbeddingRouter(architecture)
    optimizer = torch.optim.AdamW(router.parameters(), lr=0.01)
    embedding_before = provider.weight.detach().clone()
    router_before = {name: value.detach().clone() for name, value in router.state_dict().items()}

    logits = router(
        provider(torch.tensor([[1, 2, 3]], dtype=torch.int64)),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([4], dtype=torch.int64),
    )
    target = build_loss_aware_targets(torch.tensor([[1.0, 2.0, 3.0]]), 0.1)
    loss = soft_target_cross_entropy(logits, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert any(
        not torch.equal(value, router_before[name])
        for name, value in router.state_dict().items()
    )
    assert torch.equal(provider.weight, embedding_before)
    assert provider.weight.grad is None
    assert all(
        parameter is not provider.weight
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def test_routing_regret_matches_hand_calculation():
    logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    expert_losses = torch.tensor([[1.0, 5.0], [2.0, 1.0]])
    metrics = routing_metrics(logits, expert_losses)
    assert metrics.top1_accuracy == 0.0
    assert metrics.mean_routing_regret == 2.5


def test_checkpoint_resume_restores_all_training_state(tmp_path):
    torch.manual_seed(3)
    architecture = tiny_architecture()
    training_config = tiny_training_config(tmp_path)
    router = EmbeddingRouter(architecture)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=training_config.lr, betas=training_config.betas
    )
    scheduler = build_scheduler(
        optimizer, total_steps=8, warmup_ratio=0.25, min_lr_ratio=0.1
    )
    logits = router(
        torch.randn(2, 3, 4),
        torch.ones(2, 3, dtype=torch.bool),
        torch.tensor([4, 5], dtype=torch.int64),
    )
    logits.sum().backward()
    optimizer.step()
    scheduler.step()

    path = tmp_path / "checkpoint_last.pt"
    save_checkpoint(
        path,
        router=router,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        best_validation_regret=0.25,
        architecture=architecture,
        training_config=training_config,
        global_step=1,
    )
    restored_router = EmbeddingRouter(architecture)
    restored_optimizer = torch.optim.AdamW(
        restored_router.parameters(), lr=training_config.lr, betas=training_config.betas
    )
    restored_scheduler = build_scheduler(
        restored_optimizer, total_steps=8, warmup_ratio=0.25, min_lr_ratio=0.1
    )
    epoch, best_regret, global_step = load_checkpoint(
        path,
        router=restored_router,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        architecture=architecture,
        device=torch.device("cpu"),
    )

    assert (epoch, best_regret, global_step) == (1, 0.25, 1)
    for expected, actual in zip(router.parameters(), restored_router.parameters()):
        torch.testing.assert_close(actual, expected)
    original_optimizer = optimizer.state_dict()
    loaded_optimizer = restored_optimizer.state_dict()
    assert original_optimizer["param_groups"] == loaded_optimizer["param_groups"]
    assert original_optimizer["state"].keys() == loaded_optimizer["state"].keys()
    for key in original_optimizer["state"]:
        for state_name, expected in original_optimizer["state"][key].items():
            actual = loaded_optimizer["state"][key][state_name]
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected)
            else:
                assert actual == expected
    assert scheduler.state_dict() == restored_scheduler.state_dict()


def test_embedding_loader_reads_single_and_sharded_safetensors(tmp_path):
    expected = torch.randn(7, 4)
    save_file(
        {
            "model.embed_tokens.weight": expected,
            "model.layers.0.weight": torch.randn(2, 2),
        },
        tmp_path / "model.safetensors",
    )
    provider = FrozenEmbeddingProvider.from_checkpoint(
        tmp_path, weight_key="model.embed_tokens.weight", embedding_dim=4
    )
    torch.testing.assert_close(provider.weight, expected)
    assert not provider.weight.requires_grad

    (tmp_path / "model.safetensors").unlink()
    shard = "model-00002-of-00002.safetensors"
    save_file({"model.embed_tokens.weight": expected}, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": shard}}),
        encoding="utf-8",
    )
    sharded = FrozenEmbeddingProvider.from_checkpoint(
        tmp_path, weight_key="model.embed_tokens.weight", embedding_dim=4
    )
    torch.testing.assert_close(sharded.weight, expected)


def test_embedding_router_rejects_floating_max_new_tokens():
    router = EmbeddingRouter(tiny_architecture())
    with pytest.raises(ValueError, match="integer dtype"):
        router(
            torch.randn(1, 2, 4),
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([4.0]),
        )


def test_training_config_rejects_non_bf16_precision(tmp_path):
    with pytest.raises(ValueError, match="precision"):
        replace(tiny_training_config(tmp_path), precision="fp32")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="requires a BF16 CUDA GPU",
)
def test_single_gpu_bf16_training_end_to_end(tmp_path):
    architecture = tiny_architecture()
    architecture.to_json(tmp_path / "architecture.json")
    rows = [
        {
            "id": "sample-1",
            "input_ids": [1, 2, 3],
            "max_new_tokens": 4,
            "expert_losses": [1.0, 2.0, 3.0],
        },
        {
            "id": "sample-2",
            "input_ids": [2, 3],
            "max_new_tokens": 8,
            "expert_losses": [3.0, 1.0, 2.0],
        },
    ]
    encoded = "\n".join(json.dumps(row) for row in rows) + "\n"
    (tmp_path / "train.jsonl").write_text(encoded, encoding="utf-8")
    (tmp_path / "valid.jsonl").write_text(encoded, encoding="utf-8")
    save_file(
        {"model.embed_tokens.weight": torch.randn(16, 4)},
        tmp_path / "model.safetensors",
    )
    config = replace(tiny_training_config(tmp_path), epochs=1)

    train(config)

    output = tmp_path / "output"
    assert (output / "checkpoint_last.pt").is_file()
    assert (output / "checkpoint_best.pt").is_file()
    records = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["split"] for record in records] == ["train", "valid"]
    assert set(records[-1]) == {
        "epoch",
        "split",
        "loss",
        "top1_accuracy",
        "mean_routing_regret",
    }
