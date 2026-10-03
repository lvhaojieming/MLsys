"""DDP training of the router from fixed sequence-level expert loss labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from ..config import RouterArchitecture
from ..model import EmbeddingRouter
from .data import RequestDataset, collate_requests
from .embedding import FrozenEmbeddingProvider
from .objective import build_loss_aware_targets, gap_weighted_terms
from .trainer import TrainingConfig, build_scheduler, save_checkpoint, load_checkpoint, _move_batch, _log_record


def runtime(kind: str) -> tuple[torch.device, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if kind == "npu":
        import torch_npu  # noqa: F401: registers the HCCL backend
        if not torch.npu.is_available():
            raise RuntimeError("NPU is unavailable; check device mounts, driver and CANN")
        torch.npu.set_device(local_rank)
        device, backend = torch.device("npu", local_rank), "hccl"
    elif kind == "cuda":
        torch.cuda.set_device(local_rank)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device must support BF16")
        device, backend = torch.device("cuda", local_rank), "nccl"
    else:
        device, backend = torch.device("cpu"), "gloo"
    dist.init_process_group(backend, timeout=timedelta(minutes=15))
    # The native fused Transformer evaluation path is not portable to NPU.
    torch.backends.mha.set_fastpath_enabled(False)
    return device, dist.get_rank(), dist.get_world_size()


def seed(value: int) -> None:
    random.seed(value)
    torch.manual_seed(value)


def train(config: TrainingConfig, *, device_kind: str, resume: str | None = None) -> None:
    device, rank, world = runtime(device_kind)
    try:
        seed(config.seed)
        architecture = RouterArchitecture.from_json(config.architecture_config)
        fingerprints = []
        for path in (config.architecture_config, config.train_data, config.valid_data):
            digest = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            fingerprints.append(digest.hexdigest())
        contracts = [None] * world
        dist.all_gather_object(contracts, fingerprints)
        if any(contract != fingerprints for contract in contracts):
            raise ValueError("architecture or labeled dataset differs between ranks")
        datasets = [RequestDataset(path, architecture.expert_ids, config.max_prompt_tokens)
                    for path in (config.train_data, config.valid_data)]
        if len(datasets[0]) < world:
            raise ValueError("training dataset needs at least one sample per rank")
        if {x.sample_id for x in datasets[0].examples} & {x.sample_id for x in datasets[1].examples}:
            raise ValueError("train and validation sample IDs overlap")
        sampler = DistributedSampler(datasets[0], world, rank, seed=config.seed, drop_last=False)
        options = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                       collate_fn=collate_requests, pin_memory=config.pin_memory and device_kind == "cuda")
        train_loader = DataLoader(datasets[0], sampler=sampler, **options)
        # Exact validation partition, without DistributedSampler's padding duplicates.
        valid_loader = DataLoader(Subset(datasets[1], list(range(rank, len(datasets[1]), world))), **options)
        embedding = FrozenEmbeddingProvider.from_checkpoint(
            config.base_model_path, weight_key=config.embedding_weight_key,
            embedding_dim=architecture.embedding_dim).to(device).eval()
        router = EmbeddingRouter(architecture).to(device)
        ddp = DistributedDataParallel(router, device_ids=None if device_kind == "cpu" else [device.index],
                                      broadcast_buffers=False)
        optimizer = torch.optim.AdamW(router.parameters(), lr=config.lr, betas=config.betas,
                                      eps=config.eps, weight_decay=config.weight_decay)
        scheduler = build_scheduler(optimizer, total_steps=len(train_loader) * config.epochs,
                                    warmup_ratio=config.warmup_ratio, min_lr_ratio=config.min_lr_ratio)
        output = Path(config.output_dir)
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        metrics = output / "metrics.jsonl"
        artifacts_exist = not resume and any((output / name).exists() for name in
                                            ("metrics.jsonl", "checkpoint_last.pt", "checkpoint_best.pt"))
        occupied = torch.tensor(int(artifacts_exist), device=device, dtype=torch.int32)
        dist.all_reduce(occupied, op=dist.ReduceOp.MAX)
        if occupied.item():
            raise FileExistsError("training artifacts exist; choose a new output directory or --resume")
        epoch_done, best, step = 0, float("inf"), 0
        if resume:
            epoch_done, best, step = load_checkpoint(resume, router=router, optimizer=optimizer,
                                                    scheduler=scheduler, architecture=architecture, device=device)
            if epoch_done >= config.epochs:
                raise ValueError("checkpoint has already completed the configured epochs")
        if rank == 0:
            _log_record(dict(stage="distributed_training_started", device=device_kind, world_size=world,
                             batch_per_rank=config.batch_size, global_batch_size=config.batch_size * world,
                             train_samples=len(datasets[0]), validation_samples=len(datasets[1]),
                             sampler_padding_samples=len(sampler) * world - len(datasets[0]),
                             frozen_embedding=True, loss="gap_weighted_sequence_soft_target_cross_entropy",
                             gap_alpha=config.gap_alpha, gap_scale=config.gap_scale), metrics)
        for epoch in range(epoch_done + 1, config.epochs + 1):
            sampler.set_epoch(epoch)
            seed(config.seed + epoch * world + rank)
            ddp.train()
            for batch in train_loader:
                _, ids, mask, budget, losses = _move_batch(batch, device)
                targets = build_loss_aware_targets(losses, config.temperature)
                optimizer.zero_grad(set_to_none=True)
                vectors = embedding(ids)
                with torch.autocast(device_type=device_kind, dtype=torch.bfloat16):
                    logits = ddp(vectors, mask, budget)
                ce, weights = gap_weighted_terms(logits, losses, config.temperature, config.gap_alpha, config.gap_scale)
                weight_total = weights.sum().detach()
                dist.all_reduce(weight_total)
                numerator = (ce * weights).sum()
                # DDP averages gradients: compensate to obtain a global weighted mean.
                loss = numerator * world / weight_total
                finite = torch.isfinite(loss).to(torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("non-finite training loss on at least one rank")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(router.parameters(), config.max_grad_norm, foreach=False)
                if not torch.isfinite(norm).item():
                    raise FloatingPointError("non-finite synchronized gradient norm")
                optimizer.step()
                scheduler.step()
                step += 1
                total = torch.stack((numerator.detach(), weights.sum()))
                dist.all_reduce(total)
                if rank == 0:
                    _log_record(dict(epoch=epoch, step=step, split="train", loss=(total[0]/total[1]).item(),
                                     lr=optimizer.param_groups[0]["lr"], grad_norm=norm.item()), metrics)
            router.eval()
            totals = torch.zeros(5, device=device, dtype=torch.float32)
            with torch.inference_mode():
                for batch in valid_loader:
                    _, ids, mask, budget, losses = _move_batch(batch, device)
                    with torch.autocast(device_type=device_kind, dtype=torch.bfloat16):
                        logits = router(embedding(ids), mask, budget)
                    ce, weights = gap_weighted_terms(logits, losses, config.temperature, config.gap_alpha, config.gap_scale)
                    selected = logits.argmax(-1)
                    oracle = losses.min(-1)
                    totals[0] += (ce * weights).sum()
                    totals[4] += weights.sum()
                    totals[1] += (selected == oracle.indices).sum()
                    totals[2] += (losses.gather(1, selected[:, None]).squeeze(1) - oracle.values).sum()
                    totals[3] += ids.shape[0]
            dist.all_reduce(totals)
            if not torch.isfinite(totals).all().item() or int(totals[3].item()) != len(datasets[1]):
                raise RuntimeError("validation produced non-finite metrics or incorrect coverage")
            regret = (totals[2] / totals[3]).item()
            improved = regret < best
            best = min(best, regret)
            if rank == 0:
                _log_record(dict(epoch=epoch, split="valid", loss=(totals[0]/totals[4]).item(),
                                 top1_accuracy=(totals[1]/totals[3]).item(), mean_routing_regret=regret,
                                 sample_count=int(totals[3].item())), metrics)
                args = dict(router=router, optimizer=optimizer, scheduler=scheduler, epoch=epoch,
                            best_validation_regret=best, architecture=architecture,
                            training_config=config, global_step=step)
                save_checkpoint(output / "checkpoint_last.pt", **args)
                if improved:
                    save_checkpoint(output / "checkpoint_best.pt", **args)
            dist.barrier()
        if rank == 0:
            _log_record(dict(stage="complete", checkpoint=str(output / "checkpoint_best.pt")), metrics)
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("npu", "cuda", "cpu"), default="npu")
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(TrainingConfig.from_json(args.config), device_kind=args.device, resume=args.resume)
