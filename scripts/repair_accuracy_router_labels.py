#!/usr/bin/env python3
"""Re-score saved expert answers after label generation completes."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import shutil
import time
from prepare_accuracy_router_labels import number_answer, mmlu_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--wait-for-exit')
    args = parser.parse_args()
    if args.wait_for_exit:
        exit_path = Path(args.wait_for_exit)
        while not exit_path.exists():
            time.sleep(15)
        if exit_path.read_text().strip() != '0':
            raise RuntimeError('label generation failed; refusing to repair incomplete output')
    output = Path(args.output)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    labels_path = output / 'labeled.jsonl'
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    if manifest['stage'] != 'complete' or len(labels) != manifest['samples']:
        raise RuntimeError('complete manifest and exact label count required')
    changed = 0
    for row in labels:
        correctness = []
        for answer in row['expert_answers']:
            if row['category'] == 'gsm8k':
                predicted = number_answer(answer)
                correct = predicted is not None and predicted == Decimal(row['reference_value'])
            else:
                predicted = mmlu_answer(answer)
                correct = predicted is not None and predicted == row['reference_value']
            correctness.append(int(correct))
        changed += sum(a != b for a, b in zip(row['expert_correctness'], correctness))
        row['expert_correctness'] = correctness
    backup = output / 'labeled.before-answer-parser-fix.jsonl'
    if not backup.exists():
        shutil.copy2(labels_path, backup)
    temporary = output / 'labeled.corrected.tmp'
    with temporary.open('w') as stream:
        for row in labels:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(labels_path)
    for split in ('train', 'valid'):
        for source in ('gsm8k', 'mmlu'):
            part = [r for r in labels if r['split'] == split and r['source'] == source]
            manifest[split + '_metrics'][source] = dict(samples=len(part),
                expert_correct=[sum(r['expert_correctness'][i] for r in part)/len(part) for i in range(2)],
                both_correct=sum(r['expert_correctness'] == [1, 1] for r in part),
                awq_only=sum(r['expert_correctness'] == [1, 0] for r in part),
                gptq_only=sum(r['expert_correctness'] == [0, 1] for r in part),
                both_wrong=sum(r['expert_correctness'] == [0, 0] for r in part))
    manifest['answer_parser_version'] = 2
    manifest['corrected_expert_labels'] = changed
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(stage='answer_labels_repaired', samples=len(labels), changed=changed)), flush=True)


if __name__ == '__main__':
    main()
