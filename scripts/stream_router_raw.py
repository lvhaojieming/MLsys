#!/usr/bin/env python3
"""CPU tokenization stream: fixed validation prefix followed by raw-source training."""
import os
os.environ['USE_TORCH'] = '0'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from transformers import AutoTokenizer
from prepare_router_training_full import SOURCES, requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--valid-per-source', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.valid_per_source < 1:
        raise ValueError('valid-per-source must be positive')
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    totals = Counter()
    counts = {s: Counter() for s in SOURCES}
    heldout = {s: set() for s in SOURCES}
    def emit(row):
        print(json.dumps(row, ensure_ascii=False), flush=True)
    def eligible(source):
        for group, sid, prompt, answer in requests(Path(a.raw_root), source, tokenizer):
            counts[source]['considered'] += 1
            if isinstance(prompt[0], int):
                ids, targets = prompt, answer + [tokenizer.eos_token_id]
            else:
                ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                targets = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            if not ids or not targets or len(targets) > 2048 or len(ids) + len(targets) > 12287:
                counts[source]['outside_context_or_target_limit'] += 1
                continue
            yield dict(id=f'{source}:{sid}', source=source, group_id=group,
                       input_ids=ids, target_ids=targets, max_new_tokens=2048)
    generators = {s: iter(eligible(s)) for s in SOURCES}
    for source in SOURCES:
        for _ in range(a.valid_per_source):
            row = next(generators[source])
            heldout[source].add(row['group_id'])
            row['split'] = 'valid'
            emit(row)
            totals['valid'] += 1
        print(json.dumps(dict(stage='validation_source_ready', source=source)), file=sys.stderr, flush=True)
    raw_counts = json.loads((Path(a.raw_root) / 'raw_counts.json').read_text())
    weights = {r['source']: r.get('training_questions', r['training_rows']) for r in raw_counts}
    estimate = sum(weights.values())
    emit(dict(event='ready', validation_samples=totals['valid'], estimated_train_samples=estimate))
    rng = random.Random(a.seed)
    active = list(SOURCES)
    while active:
        source = rng.choices(active, weights=[weights[s] for s in active], k=1)[0]
        try:
            row = next(generators[source])
        except StopIteration:
            active.remove(source)
            continue
        if row['group_id'] in heldout[source]:
            counts[source]['additional_heldout_samples'] += 1
            continue
        row['split'] = 'train'
        emit(row)
        totals['train'] += 1
        counts[source]['train'] += 1
        if totals['train'] % 10000 == 0:
            print(json.dumps(dict(stage='tokenized', **totals)), file=sys.stderr, flush=True)
    emit(dict(event='complete', counts=dict(totals), sources={s: dict(c) for s, c in counts.items()}))


if __name__ == '__main__':
    main()
