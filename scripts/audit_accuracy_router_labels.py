#!/usr/bin/env python3
"""Check generated task supervision before starting router training."""
import argparse
import json
import math
from pathlib import Path
from prepare_accuracy_router_labels import mmlu_answer, number_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    directory = Path(args.output)
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows = [json.loads(line) for line in (directory / 'labeled.jsonl').read_text().splitlines()]
    cache = [json.loads(line) for line in (directory / 'expert-loss-cache.jsonl').read_text().splitlines()]
    if manifest['stage'] != 'complete' or len(rows) != manifest['samples']:
        raise RuntimeError('labels are incomplete')
    ids = {r['id'] for r in rows}
    if len(ids) != len(rows) or ids != {r['id'] for r in cache} or len(cache) != len(rows):
        raise RuntimeError('duplicate samples or unmatched loss cache')
    train_groups = {r['group_id'] for r in rows if r['split'] == 'train'}
    valid_groups = {r['group_id'] for r in rows if r['split'] == 'valid'}
    if train_groups & valid_groups:
        raise RuntimeError('train and validation overlap')
    for row in rows:
        if row['expert_ids'] != manifest['expert_ids']:
            raise RuntimeError('expert ordering mismatch')
        if any(not math.isfinite(v) for v in row['expert_losses']):
            raise RuntimeError('nonfinite expert loss')
        if row['expert_correctness'] not in ([0, 0], [0, 1], [1, 0], [1, 1]):
            raise RuntimeError('invalid correctness target')
    failed = False
    for source in ('gsm8k', 'mmlu'):
        part = [r for r in rows if r['source'] == source]
        extract = number_answer if source == 'gsm8k' else mmlu_answer
        missing = [sum(extract(r['expert_answers'][i]) is None for r in part) for i in range(2)]
        truncated = [sum(r.get('expert_finish_reasons', [None, None])[i] == 'length' for r in part) for i in range(2)]
        report = dict(stage='label_audit', source=source, samples=len(part),
                      parse_missing=missing, truncated=truncated,
                      expert_accuracy=[sum(r['expert_correctness'][i] for r in part)/len(part) for i in range(2)])
        print(json.dumps(report), flush=True)
        if max(missing) / len(part) > .05:
            print(json.dumps(dict(stage='label_quality_error', source=source,
                                  message='Over 5% unparsed answers; training blocked')), flush=True)
            failed = True
    if failed:
        raise RuntimeError('task supervision quality check failed')
    print(json.dumps(dict(stage='label_audit_passed', samples=len(rows))), flush=True)


if __name__ == '__main__':
    main()
