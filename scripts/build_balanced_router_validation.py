#!/usr/bin/env python3
"""Build a group-isolated diagnostic split from cached reference NLL labels."""
import os
os.environ['USE_TORCH'] = '0'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from prepare_router_training_full import requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--cache', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--margin', type=float, default=.01)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    labels = {}
    for path in a.cache:
        with open(path) as stream:
            for line in stream:
                row = json.loads(line)
                labels[row['id']] = row
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    selected, groups, counts = [], set(), Counter()
    for source, quota in [('wikitext2', 128), ('c4', 384)]:
        natural_groups, eligible = set(), 0
        candidates = {key: value for key, value in labels.items() if key.startswith(source + ':')}
        last_doc = max(int(key.split(':')[1]) for key in candidates)
        for group, sid, prompt, answer in requests(Path(a.raw_root), source, tok):
            if int(sid.split(':')[0]) > last_doc:
                break
            ids, targets = prompt, answer + [tok.eos_token_id]
            if not ids or not targets or len(targets) > 2048 or len(ids) + len(targets) > 12287:
                continue
            eligible += 1
            if eligible <= 128:
                natural_groups.add(group)
                continue
            if group in natural_groups or group in groups:
                continue
            key = f'{source}:{sid}'
            label = candidates.get(key)
            if label is None:
                continue
            losses = label['expert_losses']
            delta = losses[0] - losses[1]
            if abs(delta) < a.margin:
                continue
            winner = 'gptq' if delta > 0 else 'awq'
            if counts[(source, winner)] >= quota:
                continue
            digest = hashlib.sha256(json.dumps(ids + targets).encode()).hexdigest()
            if digest != label['tokens_sha256']:
                raise ValueError(f'token hash mismatch: {key}')
            selected.append(dict(id=key, source=source, group_id=group, input_ids=ids,
                target_ids=targets, max_new_tokens=2048, split='valid_balanced',
                expert_losses=losses, expert_ids=label['expert_ids'], winner=winner))
            groups.add(group)
            counts[(source, winner)] += 1
            if all(counts[(source, w)] == quota for w in ('awq', 'gptq')):
                break
        if any(counts[(source, w)] != quota for w in ('awq', 'gptq')):
            raise ValueError(f'insufficient distinct held-out groups: {source}, {counts}')
        print(json.dumps(dict(source=source, counts={w: counts[(source,w)] for w in ('awq','gptq')})), flush=True)
    with (out / 'requests.jsonl').open('w') as stream:
        for row in selected:
            stream.write(json.dumps(row) + '\n')
    (out / 'excluded_groups.json').write_text(json.dumps(sorted(groups)))
    manifest = dict(samples=len(selected), winners=dict(Counter(r['winner'] for r in selected)),
        sources={s: {w: counts[(s,w)] for w in ('awq','gptq')} for s in ('wikitext2','c4')},
        minimum_reference_nll_margin=a.margin, selection='first eligible distinct cached groups',
        diagnostic_only=True, requires_fresh_training=True,
        labels='cached mean reference continuation NLL including EOS',
        natural_validation_preserved=True, token_hash_verified=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
