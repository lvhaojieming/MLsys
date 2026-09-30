# MLsys: MoQE routing foundation

This repository implements the routing architecture and an offline supervised
training pipeline. It does not include a trained checkpoint, expert-loss data
generation, a vLLM gateway, or capacity-aware scheduling.

## Routing contract

```text
frozen base-model token embeddings [B, T, D]
             |
             v
L1 quality router: every prompt token
             -> projection -> local Transformer layer over all token chunks
             -> global Transformer layer over all chunk summaries
             -> attention pooling -> M expert logits
             |
             v
mask experts with no READY compatible replica; choose one expert
             |
             v
L2 physical router: choose a READY pool, then a replica of that expert
             |
             v
one request remains on this expert + replica for all prefill and decode
```

L1 produces **one logit vector per entire request**, never per-token decisions.
It uses all prompt embeddings, prompt length and the caller's output budget.
The base embedding layer stays outside the Router network and is frozen during
training. The config's ordered `expert_ids` defines the exact output-head
order. A different base model, such as Qwen3-8B, needs a separate config and
router.

L2 is rule-based in this version: it filters by model family, expert, `READY`
state and context limit, then round-robins over pools and replicas. It never
changes the expert chosen by L1 unless that expert has no feasible execution
position, in which case the next L1 candidate is tried. Adding another pool for
an existing expert changes the registry, not the L1 output shape. A new
quantization expert requires a new head/config and later retraining.

The two levels are **expert selection and physical placement**, not separate
prefill and decode routing. There is no teacher model or queue-price objective.

## Package layout

| File | Responsibility |
| --- | --- |
| `src/moqe_router/config.py` | Validated, versioned architecture and ordered expert IDs |
| `src/moqe_router/model.py` | Post-embedding L1 network returning raw expert logits |
| `src/moqe_router/physical.py` | Atomic pool snapshots and L2 placement |
| `src/moqe_router/routing.py` | Compose L1 and L2 for one request |
| `src/moqe_router/training/` | Data, frozen embedding, objective, metrics, and single-GPU trainer |
| `scripts/train_router.py` | Training CLI |
| `configs/qwen3_14b_router.json` | Illustrative architecture config; check `embedding_dim` against the exact base checkpoint |

## Model input

`EmbeddingRouter.forward()` expects:

- `embeddings`: `[B, T, D]` token embedding outputs from the **same frozen base**
  whose quantizations form the expert set. These are not hidden states from a
  full LLM forward pass.
- `attention_mask`: `[B, T]`, right-padded, with at least one valid token per row.
- `max_new_tokens`: `[B]`, the request's generation budget.

The module consumes **every valid prompt token**. It projects to `hidden_dim`,
runs one local Transformer layer separately over consecutive `chunk_size` token
blocks, attention-pools each block, then runs one global Transformer layer over
all block summaries. A final attention pool and MLP emit `[B, M]` **raw logits**.
For an 8K-token prompt with `chunk_size=128`, all 8K tokens contribute through
64 block summaries. No prompt truncation or head/middle/tail sampling is used.
There is no softmax or availability masking inside the neural network: the
registry changes independently of its weights.

```python
import torch
from moqe_router.config import RouterArchitecture
from moqe_router.model import EmbeddingRouter

config = RouterArchitecture.from_json("configs/qwen3_14b_router.json")
router = EmbeddingRouter(config).eval()

# embed_tokens is the frozen base model's token embedding module, supplied by
# the caller. Do not run the full language model merely to obtain Router input.
with torch.inference_mode():
    embeddings = embed_tokens(input_ids)  # [B, T, config.embedding_dim]
    logits = router(embeddings, attention_mask, max_new_tokens)
```

The example architecture has randomly initialized weights. Train and evaluate
it on your actual expert inventory before using its rankings on live traffic.

## Train the quality router

Install the training dependencies with `python -m pip install -e '.[train,test]'`.
Prepare separate train and validation JSONL files. Each line is one request;
the loss list follows `RouterArchitecture.expert_ids` exactly:

```json
{"id":"sample-1","input_ids":[151644,872,198],"max_new_tokens":128,"expert_losses":[1.42,1.31,1.28]}
```

`input_ids` must be the **complete prompt** tokenized with the same tokenizer
and chat template as the base model. Measure every expert's loss on the same
reference continuation, using mean target-token negative log-likelihood.
Lower loss means better quality. Keep evaluation requests separate
from training requests. The repository does not yet generate these labels.

The base model directory must contain `model.safetensors` or
`model.safetensors.index.json` and the shard containing
`model.embed_tokens.weight`. The trainer loads only that tensor on CPU and
keeps it frozen. It never loads quantized expert models; only router parameters
enter the optimizer. The loss is cross entropy against a soft distribution
built from per-request relative expert losses.

```bash
python scripts/train_router.py \
  --config configs/qwen3_14b_router_train.json
```

The output contains `metrics.jsonl`, `checkpoint_best.pt`, and
`checkpoint_last.pt`. The best checkpoint uses the lowest validation mean
routing regret. See `docs/router_training.md` for the complete contract and
resume command.

## Registry lifecycle boundary

The deployment controller will publish a complete `PoolRegistry` snapshot with
a strictly increasing epoch. Only `READY` replicas can receive new requests.
`DRAINING` replicas are absent from new route decisions but remain responsible
for streams already bound to them. The eventual gateway must perform a final
admission check because the replica may begin draining after the snapshot was
read. Mid-stream migration to another expert is unsupported.

`TwoStageRouter.route()` returns the selected `expert_id`, `pool_id`,
`replica_id`, endpoint and registry epoch. It does **not** send an HTTP request;
that is the future gateway's responsibility.

## Development

Use a Python environment with PyTorch installed:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

`configs/qwen3_14b_router.json` is a starting point. Before creating a real
checkpoint, confirm the precise model's tokenizer/chat template version and
candidate expert IDs. Its `embedding_dim=5120` matches the published
[Qwen3-14B config](https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json).
Keep Qwen3-8B and Qwen3-14B in separate model families.
