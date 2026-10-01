# 用高质量 v2 数据训练双专家 Router

一条命令运行在线专家 loss 计算、Router 更新、独立测试和本地推理服务：

```bash
cd /home/zjh/MLsys
bash scripts/train_router_hq.sh
```

默认在线训练：每个窗口最多 24 条请求，同时计算两个专家的 NLL，按相同样本 ID
配对后，以 batch 4 更新 Router，并预取下一个窗口。无需等待全数据集专家 loss
计算完毕。训练集 20,008 条，验证集 1,088 条，测试集 1,089 条，共 10 个 epoch。
专家冻结，因此后续 epoch 复用已计算的 loss。中断后再次运行会复用 loss 缓存，
从已保存的 epoch checkpoint 继续；当前未完成的 epoch 会重跑。
`--training-mode offline` 可显式使用先计算全部 loss 再训练的方式。

GPU 放置由 `configs/qwen3_14b_router_deployment.json` 明确规定：

| 进程 | 物理 GPU | 本地接口 |
|---|---:|---|
| Qwen3-14B AWQ | 0 | 127.0.0.1:19080 |
| Qwen3-14B GPTQ Int4 | 1 | 127.0.0.1:19081 |
| Router 训练/推理 | 2 | 127.0.0.1:19082（训练完成后） |

两个专家是独立进程，每个进程的 `CUDA_VISIBLE_DEVICES` 只暴露自己的 GPU，
内部 `cuda:0` 分别映射物理 GPU 0 和 1。配置不允许专家共享 GPU，也不允许
Router 使用专家 GPU。训练只更新 Router，冻结的专家服务持续驻留各自 GPU。
接口仅监听本机。

## 标签与数据边界

- 输入为 `dataset/cleaned/v2_hq/{train,valid,test}.jsonl`，保留原分区和同源组。
- 两个专家都使用同一 AWQ tokenizer/template 和现有完整 token IDs，非
  thinking 模式，不截断。部署计算 dtype 统一 float16；AWQ/GPTQ 使用各自
  Marlin 量化执行内核。
- 将 `input_ids + target_ids` 作为 teacher-forced 序列，vLLM 返回位置 i 的
  `log P(token_i | token_<i)`。取 `[len(input_ids):]` 的平均负对数概率，包含
  首个答案 token 和 EOS，不包含 prompt、padding 或额外生成 token。
- 按固定 `[AWQ, GPTQ]` 顺序写入 `expert_losses` 和 `expert_ids`。不存在随机
  标签、人工指定专家胜负，或以单次生成答对/答错替代 NLL。
- 输出 `dataset/router_hq_v2`，逐专家 loss 留存 JSONL，可续跑；记录原数据
  SHA256、模型路径、量化执行方式和精确 token 指纹，拒绝复用不兼容结果。
- 请求预算沿用清洗策略：GSM8K 1024，其余 2048。标签使用参考答案；推理
  路由只读取 prompt 和预算，不读取参考答案或专家标签。

## 训练与独立评估

双专家架构：`configs/qwen3_14b_router_two_experts.json`，输出头维度为 2。
训练参数：`configs/qwen3_14b_router_hq_train.json`，10 epochs、batch 4、BF16。
使用 AWQ checkpoint 中未量化的 `model.embed_tokens.weight` 作为冻结 embedding，
不需要下载本机不存在的未量化基座。模型对全部 prompt token 做分块层次编码。

损失为针对 `softmax(-expert_losses / 0.1)` 的交叉熵。按验证集的平均 routing
regret 选择 best checkpoint；测试集不参与训练或 checkpoint 选择。
`test_metrics.json` 同时报告路由 NLL、oracle NLL、两个固定专家策略的 NLL、
路由比例和相对最佳固定专家的收益。收益可能为零或负值，不预先宣称动态路由
优于固定专家；参考 NLL 也不等于最终生成正确率。

所有产物在 `outputs/qwen3-14b-router-hq-v2`：`metrics.jsonl`、
`checkpoint_best.pt`、`checkpoint_last.pt`、`test_metrics.json`、
`training-effective.json`、`deployment.json` 和各服务日志。

先验证小规模在线完整流程（训练 48 条，验证与测试各 8 条，覆盖四来源及最长请求，训练 1 epoch）：

```bash
bash scripts/train_router_hq.sh --pilot
```

在线 pilot 使用独立的 `dataset/router_hq_v2-online-pilot` 和输出 `online-pilot/`。
仅部署专家或仅计算全量专家 loss 可用 `--stage experts`、`--stage label`。

## 监控与论文绘图

每次更新将原始 Router 交叉熵、两个专家在同一 batch 上的平均 NLL、梯度范数、
学习率、epoch、全局 step、样本 ID 写入 `metrics.jsonl`。每个 epoch 结束记录
验证 loss、专家选择准确率和 routing regret。缓存数量可能不同，但每次更新
必须等待两个专家对同一批样本的 loss 都到齐。

```bash
tail -f /home/zjh/MLsys/outputs/qwen3-14b-router-hq-v2/pipeline-online.log
```

无需停止训练，随时导出当前曲线：

```bash
cd /home/zjh/MLsys
bash scripts/plot_router_training.sh
```

`figures/` 保存 `train_loss.csv`（未平滑）、`validation.csv`、按样本数加权的
`epoch_loss.csv`、原始日志快照、配置副本、快照 SHA256 与绘图元数据，以及
`loss_curves.pdf`、`.svg`、300 dpi `.png`。每次运行覆盖当前快照；保留论文
最终版本时复制整个目录。曲线默认显示最近 50 次更新的移动均值，可用
`--window 100` 调整；平滑仅用于显示。专家参数冻结，专家 NLL 曲线变化来自
batch 样本组成变化。尚未完成的 epoch 均值不能作为完整 epoch 结果，检查 CSV
的 `validation_recorded` 列；验证曲线在第一次完整验证后自动加入图中。

## 训练后的路由推理

```bash
curl http://127.0.0.1:19082/generate \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"解释 Python 可变默认参数的问题，并给出修正代码。"}],"max_new_tokens":256}'
```

请求先在 GPU 2 上产生专家排名，再经现有 TwoStageRouter/PoolRegistry 选择
READY 专家，将相同完整 prompt token IDs 发给 GPU 0 或 GPU 1 的独立服务。
返回文本、expert_id、实际 gpu、专家排名和生成结束原因。专家失败时返回错误，
不会静默换成一个没有被选中的模型。

核心代码：`scripts/serve_quantized_expert.py`、`scripts/label_router_data.py`、
`scripts/run_router_pipeline.py`、`scripts/train_router_online.py`、
`scripts/evaluate_router.py`、`scripts/serve_router.py`；训练器实现位于
`src/moqe_router/training/trainer.py`。
