# L1 Router 训练

## 范围

本阶段只训练 request-level 质量 Router。每条 request 输出一个固定顺序的 expert logits 向量，不运行量化 expert，不做物理派发、服务、负载均衡或分布式训练。

候选 expert 都是同一基础模型的完整量化版本。当前部署假设是每个量化 expert 独占一张 GPU，但训练代码不包含 expert-to-GPU 管理。

## 数据契约

训练与验证各使用一个 JSONL 文件。每行格式如下：

```json
{
  "id": "sample-000001",
  "input_ids": [101, 203, 405, 992],
  "max_new_tokens": 512,
  "expert_losses": [1.82, 1.75, 1.93]
}
```

`id` 在单个数据集中必须唯一。`input_ids` 只含真实 prompt token，必须非空且不提前 padding；超过 `max_prompt_tokens` 会报错，不会截断。`expert_losses` 是各量化 expert 在相同 target 上的 mean target-token negative log-likelihood，长度必须等于 expert 数且全部 finite。

`expert_losses` 的位置严格对应 architecture config 的 `expert_ids`，训练器不会排序或重映射。数据生产方可以在每行额外写入 `expert_ids`；若写入，训练器会要求它与 architecture config 完全相等，从而显式检查顺序。未写入时，列表按配置顺序解释。

collate 按当前 batch 的最长 prompt 动态右 padding：

| 字段 | shape | dtype |
| --- | --- | --- |
| `input_ids` | `[B,T]` | `torch.int64` |
| `attention_mask` | `[B,T]` | `torch.bool` |
| `max_new_tokens` | `[B]` | `torch.int64` |
| `expert_losses` | `[B,M]` | `torch.float32` |

## Embedding 与目标函数

`FrozenEmbeddingProvider` 只读取 `model.safetensors` 中的指定 embedding tensor；若 checkpoint 分片，则先从 `model.safetensors.index.json` 的 `weight_map` 找到所在 shard。它不会加载其它 Transformer 权重。embedding 维度必须等于 `RouterArchitecture.embedding_dim`，权重永久冻结且不进入 optimizer。

对每条样本先计算：

\[
\Delta_{i,m}=L_{i,m}-\min_j L_{i,j},\qquad
q_{i,m}=\frac{\exp(-\Delta_{i,m}/\tau)}{\sum_j\exp(-\Delta_{i,j}/\tau)}.
\]

训练 loss 为 soft-target cross entropy：

\[
\mathcal L=-\frac1B\sum_i\sum_m q_{i,m}\log\operatorname{softmax}(s_i)_m.
\]

因此同一条 request 的所有 expert loss 加上同一个常数不会改变 target。默认 `temperature=0.1`。

## 运行与输出

训练要求单进程、单张支持 BF16 的 CUDA GPU。所有参数从 training config 读取：

```bash
python scripts/train_router.py \
  --config configs/qwen3_14b_router_train.json
```

恢复训练：

```bash
python scripts/train_router.py \
  --config configs/qwen3_14b_router_train.json \
  --resume outputs/qwen3-14b-router/checkpoint_last.pt
```

每个 batch 的 train loss、learning rate 和 gradient norm，以及每个 epoch 的 validation loss、top-1 routing accuracy 和 mean routing regret，会同时写到 stdout 与 `metrics.jsonl`。`checkpoint_last.pt` 每个 epoch 更新；`checkpoint_best.pt` 按最低 validation mean routing regret 更新。checkpoint 包含 Router、AdamW、scheduler、epoch、历史最佳 regret、architecture config 与 training config。
