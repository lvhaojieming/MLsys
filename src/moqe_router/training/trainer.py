"""Single-process, single-GPU BF16 training for the request-level Router."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..config import RouterArchitecture
from ..model import EmbeddingRouter
from .data import RequestDataset, collate_requests
from .embedding import FrozenEmbeddingProvider
from .metrics import routing_metrics
from .objective import build_loss_aware_targets


@dataclass(frozen=True)
class TrainingConfig:
    architecture_config: str
    train_data: str
    valid_data: str
    base_model_path: str
    embedding_weight_key: str
    output_dir: str
    seed: int
    epochs: int
    batch_size: int
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    warmup_ratio: float
    min_lr_ratio: float
    max_grad_norm: float
    temperature: float
    precision: str
    num_workers: int
    pin_memory: bool
    max_prompt_tokens: int

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.architecture_config,
                self.train_data,
                self.valid_data,
                self.base_model_path,
                self.embedding_weight_key,
                self.output_dir,
            )
        ):
            raise ValueError("all path and embedding key settings must be nonempty strings")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer")
        if type(self.max_prompt_tokens) is not int or self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be a positive integer")
        if type(self.pin_memory) is not bool:
            raise ValueError("pin_memory must be boolean")
        if self.precision != "bf16":
            raise ValueError("precision must be 'bf16'")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("betas must contain two values in [0, 1)")
        for name in ("lr", "eps", "max_grad_norm", "temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        config_path = Path(path)
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            data["betas"] = tuple(data["betas"])
            return cls(**data)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid training config {config_path}: {exc}") from exc


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> LambdaLR:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    warmup_steps = math.ceil(total_steps * warmup_ratio)

    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 1:
            return min_lr_ratio
        progress = min(max((step - warmup_steps) / (decay_steps - 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_multiplier)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    path: str | Path,
    *,
    router: EmbeddingRouter,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    best_validation_regret: float,
    architecture: RouterArchitecture,
    training_config: TrainingConfig,
    global_step: int,
) -> None:
    _atomic_torch_save(
        {
            "router_state_dict": router.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation_regret": best_validation_regret,
            "architecture_config": asdict(architecture),
            "training_config": asdict(training_config),
            "global_step": global_step,
        },
        Path(path),
    )


def load_checkpoint(
    path: str | Path,
    *,
    router: EmbeddingRouter,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    architecture: RouterArchitecture,
    device: torch.device,
) -> tuple[int, float, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    expected_architecture = asdict(architecture)
    if checkpoint.get("architecture_config") != expected_architecture:
        raise ValueError(
            "checkpoint architecture_config does not exactly match the current "
            "architecture, including expert_ids order"
        )
    router.load_state_dict(checkpoint["router_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    best_regret = checkpoint["best_validation_regret"]
    global_step = checkpoint.get("global_step", scheduler.last_epoch)
    if type(epoch) is not int or epoch < 0:
        raise ValueError("checkpoint epoch must be a nonnegative integer")
    if not isinstance(best_regret, (int, float)) or math.isnan(best_regret):
        raise ValueError("checkpoint best_validation_regret is invalid")
    if type(global_step) is not int or global_step < 0:
        raise ValueError("checkpoint global_step must be a nonnegative integer")
    return epoch, float(best_regret), global_step


def _finite_or_raise(
    name: str,
    tensor: Tensor,
    *,
    epoch: int,
    sample_ids: list[str],
    loss: Tensor | float | None,
) -> None:
    if bool(torch.isfinite(tensor).all()):
        return
    if isinstance(loss, Tensor):
        loss_value: float | str = float(loss.detach().float().cpu())
    elif loss is None:
        loss_value = "unavailable"
    else:
        loss_value = loss
    raise FloatingPointError(
        f"non-finite {name}: epoch={epoch}, sample_ids={sample_ids}, loss={loss_value}"
    )


def _move_batch(
    batch: dict[str, Tensor | list[str]], device: torch.device
) -> tuple[list[str], Tensor, Tensor, Tensor, Tensor]:
    sample_ids = batch["ids"]
    if not isinstance(sample_ids, list):
        raise TypeError("batch ids must be a list")
    tensors: list[Tensor] = []
    for key in ("input_ids", "attention_mask", "max_new_tokens", "expert_losses"):
        value = batch[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"batch {key} must be a tensor")
        tensors.append(value.to(device, non_blocking=True))
    return sample_ids, tensors[0], tensors[1], tensors[2], tensors[3]


def validate(
    router: EmbeddingRouter,
    embedding_provider: FrozenEmbeddingProvider,
    loader: DataLoader,
    *,
    device: torch.device,
    temperature: float,
    epoch: int,
) -> dict[str, float | int | str]:
    router.eval()
    loss_sum = 0.0
    correct = 0
    regret_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            sample_ids, input_ids, attention_mask, max_new_tokens, expert_losses = (
                _move_batch(batch, device)
            )
            _finite_or_raise(
                "expert_losses",
                expert_losses,
                epoch=epoch,
                sample_ids=sample_ids,
                loss=None,
            )
            embeddings = embedding_provider(input_ids)
            target_probs = build_loss_aware_targets(expert_losses, temperature)
            _finite_or_raise(
                "target_probs",
                target_probs,
                epoch=epoch,
                sample_ids=sample_ids,
                loss=None,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = router(embeddings, attention_mask, max_new_tokens)
                log_probs = F.log_softmax(logits, dim=-1)
                loss = -(target_probs * log_probs).sum(dim=-1).mean()
            _finite_or_raise(
                "logits", logits, epoch=epoch, sample_ids=sample_ids, loss=loss
            )
            _finite_or_raise("loss", loss, epoch=epoch, sample_ids=sample_ids, loss=loss)
            metrics = routing_metrics(logits, expert_losses)
            batch_size = metrics.sample_count
            loss_sum += float(loss.float().cpu()) * batch_size
            correct += metrics.top1_correct
            regret_sum += metrics.routing_regret_sum
            sample_count += batch_size
    router.train()
    return {
        "epoch": epoch,
        "split": "valid",
        "loss": loss_sum / sample_count,
        "top1_accuracy": correct / sample_count,
        "mean_routing_regret": regret_sum / sample_count,
    }


def _log_record(record: dict[str, Any], metrics_path: Path) -> None:
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
    print(encoded, flush=True)
    with metrics_path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def _require_bf16_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Router training requires a CUDA GPU; no CUDA device is available")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the current CUDA device does not support BF16 training")
    return torch.device("cuda", torch.cuda.current_device())


def train(training_config: TrainingConfig, resume: str | Path | None = None) -> None:
    device = _require_bf16_cuda()
    random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    torch.cuda.manual_seed_all(training_config.seed)

    architecture = RouterArchitecture.from_json(training_config.architecture_config)
    train_dataset = RequestDataset(
        training_config.train_data,
        architecture.expert_ids,
        training_config.max_prompt_tokens,
    )
    valid_dataset = RequestDataset(
        training_config.valid_data,
        architecture.expert_ids,
        training_config.max_prompt_tokens,
    )
    generator = torch.Generator().manual_seed(training_config.seed)
    loader_options = {
        "batch_size": training_config.batch_size,
        "num_workers": training_config.num_workers,
        "pin_memory": training_config.pin_memory,
        "collate_fn": collate_requests,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_options)

    embedding_provider = FrozenEmbeddingProvider.from_checkpoint(
        training_config.base_model_path,
        weight_key=training_config.embedding_weight_key,
        embedding_dim=architecture.embedding_dim,
    ).to(device)
    embedding_provider.eval()
    if any(parameter.requires_grad for parameter in embedding_provider.parameters()):
        raise AssertionError("FrozenEmbeddingProvider contains trainable parameters")

    router = EmbeddingRouter(architecture).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=training_config.lr,
        betas=training_config.betas,
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
    )
    total_steps = len(train_loader) * training_config.epochs
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=training_config.warmup_ratio,
        min_lr_ratio=training_config.min_lr_ratio,
    )

    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    last_path = output_dir / "checkpoint_last.pt"
    best_path = output_dir / "checkpoint_best.pt"
    if resume is None and any(path.exists() for path in (metrics_path, last_path, best_path)):
        raise FileExistsError(
            f"output_dir already contains training artifacts; use --resume or a new directory: "
            f"{output_dir}"
        )

    completed_epoch = 0
    best_validation_regret = float("inf")
    global_step = 0
    if resume is not None:
        completed_epoch, best_validation_regret, global_step = load_checkpoint(
            resume,
            router=router,
            optimizer=optimizer,
            scheduler=scheduler,
            architecture=architecture,
            device=device,
        )
        if completed_epoch >= training_config.epochs:
            raise ValueError(
                f"checkpoint already completed epoch {completed_epoch}, but config epochs="
                f"{training_config.epochs}"
            )

    router.train()
    for epoch in range(completed_epoch + 1, training_config.epochs + 1):
        for batch in train_loader:
            sample_ids, input_ids, attention_mask, max_new_tokens, expert_losses = (
                _move_batch(batch, device)
            )
            _finite_or_raise(
                "expert_losses",
                expert_losses,
                epoch=epoch,
                sample_ids=sample_ids,
                loss=None,
            )
            with torch.no_grad():
                embeddings = embedding_provider(input_ids)
            target_probs = build_loss_aware_targets(
                expert_losses, training_config.temperature
            )
            _finite_or_raise(
                "target_probs",
                target_probs,
                epoch=epoch,
                sample_ids=sample_ids,
                loss=None,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = router(embeddings, attention_mask, max_new_tokens)
                log_probs = F.log_softmax(logits, dim=-1)
                loss = -(target_probs * log_probs).sum(dim=-1).mean()
            _finite_or_raise(
                "logits", logits, epoch=epoch, sample_ids=sample_ids, loss=loss
            )
            _finite_or_raise("loss", loss, epoch=epoch, sample_ids=sample_ids, loss=loss)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                router.parameters(), training_config.max_grad_norm
            )
            _finite_or_raise(
                "grad_norm",
                grad_norm,
                epoch=epoch,
                sample_ids=sample_ids,
                loss=loss,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            _log_record(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "split": "train",
                    "loss": float(loss.detach().float().cpu()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm.detach().float().cpu()),
                },
                metrics_path,
            )

        validation = validate(
            router,
            embedding_provider,
            valid_loader,
            device=device,
            temperature=training_config.temperature,
            epoch=epoch,
        )
        _log_record(validation, metrics_path)
        validation_regret = float(validation["mean_routing_regret"])
        if validation_regret < best_validation_regret:
            best_validation_regret = validation_regret
            save_checkpoint(
                best_path,
                router=router,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_validation_regret=best_validation_regret,
                architecture=architecture,
                training_config=training_config,
                global_step=global_step,
            )
        save_checkpoint(
            last_path,
            router=router,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_validation_regret=best_validation_regret,
            architecture=architecture,
            training_config=training_config,
            global_step=global_step,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training JSON config")
    parser.add_argument("--resume", help="checkpoint_last.pt to resume")
    args = parser.parse_args()
    train(TrainingConfig.from_json(args.config), args.resume)


if __name__ == "__main__":
    main()
