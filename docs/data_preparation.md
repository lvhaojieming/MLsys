# Router 训练数据准备

`scripts/clean_router_data.py` 全量读取本机已下载的 WildChat、GSM8K
`main/train`、Magicoder 和 LongAlign，输出离线 expert 打分输入。
LMSYS 需要 Hugging Face 授权，当前不包含在清洗输出中。

## 运行

需要 `pyarrow`、`transformers`、`tokenizers`、`pyahocorasick`。
本机已有 Conda `engine` 环境，额外的匹配依赖保存在
`dataset/.cleaning-deps`，不修改模型或原始数据。

```bash
cd /home/zjh/MLsys
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/zjh/miniconda3/envs/engine/lib/python3.12/site-packages \
/home/zjh/miniconda3/envs/engine/bin/python -S scripts/clean_router_data.py \
  --workers 24 --output dataset/cleaned/v1
```

输出目录必须不存在，脚本拒绝覆盖。调试时使用新的输出目录和
`--limit-per-source 512`；正式处理默认读取全部记录。

## 清洗与划分政策

- 每个原始对话只取最后一个 assistant 为参考答案，保留此前完整历史。
  不把已有 assistant 回答误当成当前请求的模型输入。
- 严格检查 system/user/assistant 角色顺序、非空消息、控制字符和文本字段。
  WildChat 带 redacted 或 toxic 标记的原始对话过滤。
- 保留代码空格、缩进及换行；只统一行尾和开头 BOM。assistant 开头显式
  `<think>...</think>` 块移除，未闭合或去掉后为空的记录过滤；自然语言解题
  步骤保留。此版本统一使用非 thinking 模式。
- 按 NFKC、casefold 和空白折叠后的完整 prompt 去重，保留第一个合格答案。
  此规范化只用于去重，不改写实际输入文本；不保证语义近似去重。
- 本地 GSM8K test 和已有 MMLU/GSM8K 比较记录用于污染过滤：在 prompt 历史中
  匹配规范化题目的前 96 字符，题目至少 40 字符。这是保守指纹匹配，不是
  语义污染检测。
- 使用本机 AWQ tokenizer/chat template，`enable_thinking=False`，完整 prompt
  不能超过 8192 tokens，不截断。参考答案加一个 EOS 不能超过生成预算：
  GSM8K 1024，其余 2048；部署上下文检查采用 40960 上限。
- 按共享完整 prompt、WildChat 原始对话和足够长的相同首个请求、Magicoder
  seed、LongAlign 规范化文档前 4096 字符分组。组经过合并后，以 seed=42
  哈希划分 90%/5%/5% train/valid/test。前缀不同的相同文档仍可能漏分组。

## 输出

`dataset/cleaned/v1/{train,valid,test}.jsonl` 保存全部合格记录。
`starter/` 保存按语言/编程语言和 prompt 长度分层抽样的起步集；训练配额
分别为 WildChat 20000、GSM8K 5000、Magicoder 10000、LongAlign 5000，验证和
测试配额各取训练配额的 1/18 向上取整。不足配额时使用该分区全部合格样本。

每条记录包含 provenance、完整 messages/reference、固定生成预算、分组、
prompt/target token IDs 和长度。`target_ids` 是参考文本独立编码后追加一个
EOS；后续 expert 打分时使用 `input_ids + target_ids`，仅评分 target 区域，
正确进行 causal shift，不能重新使用各 expert 目录不同的 chat template。

`manifest.json` 包含完整过滤统计、分区数量、tokenizer/template 哈希及文件
SHA256；原始下载版本见 `dataset/download_manifest.json`。`records.sqlite`
保留清洗中间记录和抽样索引，JSONL 是下游接口。

## 验证与下一阶段

```bash
OMP_NUM_THREADS=1 \
PYTHONPATH=/home/zjh/miniconda3/envs/engine/lib/python3.12/site-packages \
/home/zjh/miniconda3/envs/engine/bin/python -S scripts/validate_cleaned_data.py \
  dataset/cleaned/v1
```

验证逐条检查 ID/prompt 唯一、组不跨分区、角色和 token 合约、配额子集与
完整集的一致性，并对每个来源/分区前 10 条独立重新编码。
结果写入 `validation.json`。

这些文件不是现有训练器能直接使用的带标签数据。下一步需要冻结 AWQ/GPTQ，
按同一参考答案计算 mean target-token NLL，按固定 `[AWQ, GPTQ]` 顺序写入
`expert_losses`。不能用虚构 loss 或把现有答对/答错记录冒充 NLL。
生成预算是此清洗版本制定的策略，并非原始请求的真实观测预算。
参考回答未验证事实正确性或代码通过率，需要独立评估。
