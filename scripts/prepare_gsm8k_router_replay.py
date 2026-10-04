#!/usr/bin/env python3
"""Reconstruct the audited prompts of paired GSM8K expert evaluations."""
import os
os.environ['USE_TORCH'] = '0'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluations', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.evaluations)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    experts = []
    audits = []
    for name in ('awq','gptq'):
        all_rows = [json.loads(line) for line in (root/f'{name}-samples-merged.jsonl').open()]
        rows = [r for r in all_rows if r['filter'] == 'strict-match']
        index = {r['doc_id']: r for r in rows}
        if len(index) != len(rows):
            raise ValueError('duplicate doc ids')
        experts.append(index)
        audit = {}
        for file in root.glob(f'{name}-shard*-audit.jsonl'):
            for line in file.open():
                r = json.loads(line)
                audit[(r['messages_sha256'], r['answer'])] = r
        audits.append(audit)
    if set(experts[0]) != set(range(1319)) or experts[0].keys() != experts[1].keys():
        raise ValueError('expected the same complete 1319-question test split')
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    counts = Counter()
    with (out/'requests.jsonl').open('w') as stream:
        for doc_id in sorted(experts[0]):
            awq, gptq = [e[doc_id] for e in experts]
            if awq['arguments'] != gptq['arguments'] or awq['doc'] != gptq['doc']:
                raise ValueError('experts evaluated different inputs')
            if awq['filter'] != gptq['filter']:
                raise ValueError('different evaluation filters')
            arguments = awq['arguments']['gen_args_0']
            messages = json.loads(arguments['arg_0'][0])
            settings = arguments['arg_1']
            if settings['chat_template_kwargs']['enable_thinking'] is not False:
                raise ValueError('unexpected thinking mode')
            ids = tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            message_hash = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            answers = [r['resps'][0][0] for r in (awq,gptq)]
            for e in range(2):
                audit = audits[e][(message_hash, answers[e])]
                if audit['prompt_tokens_preflight'] != len(ids) or audit['usage']['prompt_tokens'] != len(ids):
                    raise ValueError('reconstructed input differs from served prompt')
                if audit['max_tokens'] != settings['max_gen_toks']:
                    raise ValueError('generation budget mismatch')
            correct = [bool(r['exact_match']) for r in (awq,gptq)]
            category = ('both_correct' if all(correct) else 'awq_only' if correct[0]
                else 'gptq_only' if correct[1] else 'both_wrong')
            counts[category] += 1
            row = dict(id=f'gsm8k-test:{doc_id}', doc_id=doc_id, input_ids=ids,
                max_new_tokens=settings['max_gen_toks'], expert_correct=correct,
                expert_answers=answers, expert_filtered_answers=[r['filtered_resps'] for r in (awq,gptq)],
                question=awq['doc']['question'], reference=awq['doc']['answer'],
                category=category, messages_sha256=message_hash, filter=awq['filter'])
            stream.write(json.dumps(row)+'\n')
    manifest = dict(samples=1319, categories=dict(counts),
        mode='replay cached expert responses with newly evaluated router decisions',
        prompt_and_budget_audit_verified=True, thinking=False,
        scoring='recorded lm-eval exact_match; no answer-extraction changes',
        evaluations=str(root), tokenizer=str(a.tokenizer))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
