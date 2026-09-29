# MLsys: MoQE routing foundation

This repository currently implements the **routing architecture only**. It does
not include a trained checkpoint, a loss function, a training pipeline, a vLLM
gateway, or capacity-aware scheduling. Those pieces will be added after the
architecture and expert inventory are fixed.

## Routing contract

```text
frozen base-model token embeddings [B, T, D]
             |
             v
L1 quality router: first / middle / last prompt windows
             -> projection -> two lightweight Transformer encoder layers
             -> attention pooling per window -> M expert logits
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
It uses prompt embeddings, original prompt length and the caller's output budget.
The base embedding layer stays outside this package and should be frozen. The
config's ordered `expert_ids` defines the exact output-head order. A different
base model, such as Qwen3-8B, needs a separate config and router.

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
| `configs/qwen3_14b_router.json` | Illustrative architecture config; check `embedding_dim` against the exact base checkpoint |

## Model input

`EmbeddingRouter.forward()` expects:

- `embeddings`: `[B, T, D]` token embedding outputs from the **same frozen base**
  whose quantizations form the expert set. These are not hidden states from a
  full LLM forward pass.
- `attention_mask`: `[B, T]`, right-padded, with at least one valid token per row.
- `max_new_tokens`: `[B]`, the request's generation budget.
- `original_prompt_lengths`: optional `[B]` if upstream has already preserved
  first/middle/last windows while shortening a long prompt.

The module selects up to `tokens_per_region` tokens from the beginning, center
and end of each prompt, projects to `hidden_dim`, adds local position and region
embeddings, runs two small Transformer encoder layers, attention-pools each region,
then emits `[B, M]` **raw logits**. There is no softmax or availability masking
inside the neural network: the registry changes independently of its weights.

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

The example architecture has randomly initialized weights. **Its rankings are
not useful until a later training stage is implemented.** Do not put this model
on live traffic yet.

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
