# Router 架构定稿（只含结构，不含训练）

## 1. 本阶段边界

本阶段只实现从**冻结的基础模型 token embedding 输出**到路由决策的数据路径。不实现专家 loss 标注、损失函数、训练脚本、Gateway、vLLM 请求代理或容量感知调度。当前网络随机初始化，不能直接用于真实专家选择。

## 2. 两级职责

```mermaid
flowchart LR
  A[完整请求的 token embeddings] --> B[L1: 质量 Router]
  B --> C[每条请求一个专家 logits 向量]
  C --> D[根据 READY 执行位置屏蔽专家]
  D --> E[一个量化专家 expert_id]
  E --> F[L2: 物理 Router]
  F --> G[一个池 pool_id + 一个副本 replica_id]
```

- **L1** 根据 prompt 表征和请求生成预算，在同一基础模型的量化专家之间排序。输出维度固定为专家数 `M`，而不是 token 数。L1 不读 GPU 队列、池地址或实时负载。
- **L2** 只对 L1 已选专家查找可行的物理池和副本。第一版根据模型族、专家版本、上下文上限和 `READY` 状态过滤，再在池与副本间分别轮询。它不训练第二个神经网络，不改变 L1 的质量目标。
- 请求开始执行后固定 `expert_id + pool_id + replica_id`。prefill 和 decode 由同一副本完成。两级路由不表示两个阶段各路由一次。
- 若该专家在决策与派发之间失去就绪副本，Gateway 后续应重新尝试 L1 排序中的下一位；若已开始流式输出，不做跨专家续写。当前包只返回决策，不承担实际 HTTP 派发与最终准入确认。

## 3. L1：embedding 后的网络

**输入契约**

| 张量 | 形状 | 说明 |
|---|---|---|
| `embeddings` | `[B,T,D]` | 同一冻结基础模型的 `embed_tokens` 输出；不是完整 LLM 的隐藏层输出 |
| `attention_mask` | `[B,T]` | 1 为 prompt token，0 为右侧 padding；每行至少一个 token |
| `max_new_tokens` | `[B]` | 请求在路由时可见的输出预算 |
| `original_prompt_lengths` | `[B]`，可选 | 上游压缩过 prompt 时的原长度；压缩仍须保留首／中／尾信息 |

**网络流水线**

1. 对每条有效序列按原 token 次序取首段、中段、尾段，各不超过 `W=tokens_per_region` 个 token；短 prompt 的窗口可重叠。
2. 用 `Linear(D,H)+LayerNorm` 将基础 embedding 投影到小维度 `H`。
3. 加窗口内位置 embedding 和三段类型 embedding。
4. 将三段拼成最多 `3W` 个位置，通过一层小型 Transformer encoder。padding 作为 key mask；此模块属于 Router，不运行任何量化专家。
5. 在每段内做可学习 attention pooling，得到三个 `[H]` 向量。
6. 拼接三个向量、对数归一化的原 prompt 长度和 `max_new_tokens`，由 MLP 输出 `[B,M]` **raw logits**。

当前示例配置为 `D=5120, H=256, W=128, heads=4, encoder_layers=1`。`D=5120` 与 Qwen3-14B 官方配置一致；其他值是架构起点，需在后续准确率和 Router 延迟实验中选择，不能当作已验证最优值。该结构的注意力长度至多 `3W=384`，与完整 8K prompt 长度分离。

**为什么不在网络内部做可用性 mask：**卡池状态变化比训练好的权重快。L1 网络始终给固定顺序的所有专家输出分数；在线层按注册表屏蔽无执行位置的专家。这样增加相同专家的副本不改变网络维度。

## 4. embedding 来源与部署约束

- Qwen3-14B Router 必须使用与其量化专家共同基础模型对应的冻结 embedding 表、相同 tokenizer 与 chat template；Qwen3-8B 另建 Router。不能混用不同基础模型的 token embeddings。
- 本包刻意不加载 `embed_tokens`，以免把模型仓库格式、量化后端和路由网络耦合。后续 Gateway／embedding adapter 负责只加载或引用这一层，并在模型发布时校验来源版本。
- 若为降低 embedding 开销，上游可先按首／中／尾规则选择 token ID，再查询 embedding 表，传入合并后的三个窗口及 `original_prompt_lengths`。这与传完整 embeddings 的语义须通过一致性测试；不可直接截断为仅前缀。
- 路由网络的显存、提取 embeddings 的代价、TTFT 增量都应在后续系统实验实测。不要因为本结构参数较小，就假定完整路由链路没有显存或时延成本。

## 5. L2：注册表与动态池

部署控制器向 `PoolRegistry` 发布完整、单调递增 epoch 的快照。每个 `Replica` 指定模型族、专家 ID、池 ID、副本 ID、endpoint、上下文上限、状态和 checkpoint 版本。同一池不能混不同专家；同一专家 ID 不能在快照中对应两个 checkpoint 版本。

`LOADING/WARMING/DRAINING/FAILED/STOPPED` 不接新请求，仅 `READY` 可进入候选。旧快照对已经绑定的请求仍有日志价值；新请求使用最新快照。Gateway 未来需要派发前准入确认，避免状态刚切到 `DRAINING` 时仍发送新流。

加入**同一专家**的新池：只更新注册表，不需要改 L1 头。加入**新量化专家**：需要新 `expert_id`、新输出维度和后续训练，当前代码不会把它悄悄放进现有 Router。

## 6. 已明确留给下一阶段的内容

1. 对每位专家生成请求级 loss vector，确定 L1 损失与训练协议。
2. 构建训练和开发数据，并评估三段表示与更简单的 mean pooling 等结构。
3. 冻结模型权重和 embedding 来源，发布 Router artifact。
4. 接入 Gateway、vLLM 池、健康探针、排空与故障处理。
5. 最后另行研究队列／容量调度；它可能改变 L2 规则及 L1/L2 的联合决策方式，但不属于当前架构代码。
