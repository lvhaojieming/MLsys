# 内容质量清洗 v2

本轮只清理数据，不进行专家打标、Router 训练或模型修改。
原有 `dataset/cleaned/v1` 完整保留。输出为 `dataset/cleaned/v2_hq`。

## 范围与筛选

逐条读取 v1 的全部 799,954 条记录，沿用已完成的结构、编码、污染过滤、
去重和同源分区。新增筛选不会重新随机划分，避免已有组跨分区。

- 剔除参考答案开头的道歉/拒答、占位实现、明显未结束答案、未闭合代码块
  和重复段落。规则偏保守，可能丢弃正常讨论中的这些字符串。
- WildChat 限原始语言标签为英文或中文，并检查基本脚本一致性；剔除短而
  低信息的末轮请求、已有拒答历史及角色扮演类请求。短请求即使在多轮历史
  中合理也会被保守剔除。语言标签仍不是独立语言识别结果。
- GSM8K 用只允许数字及算术运算的 AST/Fraction 校验每个 `<<算式=结果>>`，
  检查最后计算结果与 `####` 最终答案相同。接受显示小数精度内的四舍五入，
  不支持的表达式剔除。不能据此证明应用题建模、单位或全部文字推理正确。
- Magicoder 要求明确问题和完整代码块；Python 代码通过 AST 解析，排除明显
  stub，不执行任何参考代码。其他语言未做编译或单元测试。
- LongAlign 保留真正长上下文请求，不截断原文。

## 完整上下文质量复核

规则筛选后，从各分区按已有语言/代码语言及长度层进行比例分配，以固定
哈希种子选样。WildChat 和 Magicoder 各取至多 train 10,000、valid/test
各 556；GSM8K、LongAlign 取全部通过规则的样本。
这是面向精度的小型精选集，未选入复核池的记录不被宣称为低质量或已复核。

本地 `Qwen3-14B-AWQ` 作为内容审查器，读取完整历史及参考答案，不截断，
非 thinking、temperature=0；约束先输出简短核验分析，再给 verdict 的 JSON。审查相关性、完整性、
代码需求、数学推理、文档依据及明显事实错误，有疑问剔除。只有 PASS 导出。
文档和对话作为不可信数据呈交审查，避免其中指令替代审查规则。

开始处理每个分片前，校准已知正确算术、错误算术、错误代码、缺失文档和
提示注入五个案例；校准不通过即停止该分片。校准只能发现明显失效，不能
证明审查器准确率。模型复核结果有误判，不能替代人工标注和代码执行测试。
审查器也属于后续可能使用的专家家族，因此后续训练评估应注意选择偏差。
导出时进一步去除规范化后完全相同、至少 120 字符的参考答案，避免不同
请求复制同一长答案；首条保留，重复记录及保留 ID 写入审计。此检查不等于
语义近似去重，也可能丢弃合理的共享答案。

## 产物及复现

`manifest.json` 记录全部规则筛查计数、复核池配额、审查结果、最终分区计数
和哈希；`screening.sqlite` 留存规则拒绝原因及候选索引；`review/` 留存复核
输入、逐条决定和校准；`train/valid/test.jsonl` 仅有通过的请求及参考答案；
`README.md` 和 `samples.jsonl` 提供完整样例；`validation.json` 为最终检查。
没有 `expert_losses`，仍需后续真实专家标签才能用于现有 Router 训练器。

```bash
python scripts/refine_router_quality.py prepare
# engine 环境有 vLLM；按需设置 CUDA_VISIBLE_DEVICES 指向空闲 GPU。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/home/zjh/miniconda3/envs/engine/lib/python3.12/site-packages \
/home/zjh/miniconda3/envs/engine/bin/python -S scripts/refine_router_quality.py judge --shard 0
# 依次或多 GPU 并发完成全部 shard 0..7，然后：
python scripts/refine_router_quality.py finalize
PYTHONPATH=/home/zjh/miniconda3/envs/engine/lib/python3.12/site-packages \
/home/zjh/miniconda3/envs/engine/bin/python -S scripts/refine_router_quality.py validate
```

prepare 拒绝覆盖已有输出；judge 可从已写入的决定续跑。先用 `--limit 8`
进行小规模复核验证，pilot 结果另存，不冒充全量完成。
