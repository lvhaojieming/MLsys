#!/usr/bin/env python3
"""Stream all available training-source requests into tokenized router data."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


SOURCES = ('WildChat-1M', 'gsm8k', 'Magicoder-OSS-Instruct-75K',
           'LongAlign-10k', 'mmlu', 'mbpp', 'qasper', 'wikitext2', 'c4')


def raw_rows(root, source):
    for file in sorted((root / source).rglob('*')):
        if '.cache' in file.parts or not file.is_file():
            continue
        # Only training portions are eligible for router train/validation.
        if source in ('gsm8k', 'mbpp', 'wikitext2', 'WildChat-1M', 'mmlu') and not (
            file.name.startswith('train-') or file.name.startswith('auxiliary_train-')
        ):
            continue
        if file.suffix == '.parquet':
            for batch in pq.ParquetFile(file).iter_batches(batch_size=128):
                yield from batch.to_pylist()
        elif file.suffix == '.jsonl' or file.name.endswith('.json.gz'):
            opener = gzip.open if file.name.endswith('.gz') else open
            with opener(file, 'rt', encoding='utf-8') as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)


def requests(root, source, tokenizer):
    for index, row in enumerate(raw_rows(root, source)):
        group = f'{source}:{index}'
        if source in ('WildChat-1M', 'LongAlign-10k'):
            messages = row['conversation'] if source == 'WildChat-1M' else row['messages']
            end = next((i for i in range(len(messages)-1, 0, -1)
                        if messages[i]['role'] == 'assistant'), None)
            if end is None:
                continue
            group = f"{source}:{row.get('conversation_hash', row.get('id', index))}"
            prompt = [{'role': m['role'], 'content': m['content']} for m in messages[:end]]
            yield group, str(index), prompt, messages[end]['content']
        elif source == 'gsm8k':
            yield group, str(index), [{'role': 'user', 'content': row['question']}], row['answer']
        elif source == 'Magicoder-OSS-Instruct-75K':
            yield group, str(index), [{'role': 'user', 'content': row['problem']}], row['solution']
        elif source == 'mmlu':
            row = row.get('train', row)
            prompt = row['question'] + '\n' + '\n'.join(
                f'{chr(65+i)}. {choice}' for i, choice in enumerate(row['choices'])) + '\nAnswer:'
            yield group, str(index), [{'role': 'user', 'content': prompt}], chr(65 + int(row['answer']))
        elif source == 'mbpp':
            yield group, str(index), [{'role': 'user', 'content': row['text']}], row['code']
        elif source == 'qasper':
            full_text = row['full_text']
            paragraphs = full_text['paragraphs'] if isinstance(full_text, dict) else [s['paragraphs'] for s in full_text]
            document = row['title'] + '\n' + row['abstract'] + '\n' + '\n'.join('\n'.join(s) for s in paragraphs)
            qas = row['qas']
            for question_index, question in enumerate(qas['question']):
                answer = qas['answers'][question_index]['answer'][0]
                reference = answer.get('free_form_answer') or ' '.join(answer.get('extractive_spans', []))
                if not reference and answer.get('yes_no') is not None:
                    reference = 'Yes' if answer['yes_no'] else 'No'
                if not reference and answer.get('unanswerable'):
                    reference = 'unanswerable'
                if reference:
                    yield group, f'{index}:{question_index}', [
                        {'role': 'user', 'content': document + '\nQuestion: ' + question}], reference
        else:
            tokens = tokenizer.encode(row['text'], add_special_tokens=False)
            for start in range(0, len(tokens), 2048):
                window = tokens[start:start+2048]
                if len(window) < 4:
                    continue
                midpoint = len(window) // 2
                yield group, f'{index}:{start}', window[:midpoint], window[midpoint:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--valid-per-source', type=int, default=128)
    parser.add_argument('--resume', action='store_true', help='Reuse already tokenized rows and append remaining samples')
    args = parser.parse_args()
    if args.valid_per_source < 1:
        raise ValueError('valid-per-source must be positive')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    summary = {}
    counts = Counter()
    existing_ids = set()
    existing_counts = {source: Counter() for source in SOURCES}
    valid_groups = {source: set() for source in SOURCES}
    if output.exists():
        with output.open('r+b') as previous:
            while True:
                offset = previous.tell()
                line = previous.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if previous.read():
                        raise ValueError('invalid JSON inside the existing dataset')
                    output.with_suffix('.incomplete-tail.bin').write_bytes(line)
                    previous.truncate(offset)
                    break
                if row['id'] in existing_ids:
                    raise ValueError('duplicate sample in existing dataset')
                existing_ids.add(row['id'])
                existing_counts[row['source']][row['split']] += 1
                counts[row['split']] += 1
                if row['split'] == 'valid':
                    valid_groups[row['source']].add(row['group_id'])
            previous.seek(0, 2)
            if previous.tell():
                previous.seek(-1, 2)
                if previous.read(1) != b'\n':
                    previous.seek(0, 2)
                    previous.write(b'\n')
        print(json.dumps(dict(stage='resumed', existing_samples=len(existing_ids))), flush=True)
    with output.open('a' if args.resume else 'w', encoding='utf-8') as stream:
        for source in SOURCES:
            selected_groups = valid_groups[source]
            source_counts = existing_counts[source].copy()
            for group, sample_id, prompt, reference in requests(Path(args.raw_root), source, tokenizer):
                source_counts['considered'] += 1
                if f'{source}:{sample_id}' in existing_ids:
                    continue
                if isinstance(prompt[0], int):
                    input_ids, target_ids = prompt, reference + [tokenizer.eos_token_id]
                else:
                    input_ids = tokenizer.apply_chat_template(
                        prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                    target_ids = tokenizer.encode(reference, add_special_tokens=False) + [tokenizer.eos_token_id]
                if not input_ids or not target_ids or len(target_ids) > 2048 or len(input_ids) + len(target_ids) > 12287:
                    source_counts['outside_context_or_target_limit'] += 1
                    continue
                if group in selected_groups:
                    split = 'valid'
                elif source_counts['valid'] < args.valid_per_source:
                    selected_groups.add(group)
                    split = 'valid'
                else:
                    split = 'train'
                value = dict(id=f'{source}:{sample_id}', source=source, group_id=group,
                             split=split, input_ids=input_ids, target_ids=target_ids,
                             max_new_tokens=2048)
                stream.write(json.dumps(value, ensure_ascii=False) + '\n')
                counts[split] += 1
                source_counts[split] += 1
                if source_counts['considered'] % 10000 == 0:
                    print(json.dumps(dict(source=source, **source_counts)), flush=True)
            if source_counts['valid'] < args.valid_per_source or not source_counts['train']:
                raise RuntimeError(f'insufficient full-data samples from {source}: {source_counts}')
            summary[source] = dict(source_counts)
            print(json.dumps(dict(source=source, completed=True, **source_counts)), flush=True)
    output.with_suffix('.manifest.json').write_text(json.dumps(dict(
        counts=dict(counts), sources=summary, cleaning=False, prompt_truncation=False,
        context_limit=12288, target_limit=2048, valid_per_source=args.valid_per_source,
        split='first eligible source groups for validation; all remaining groups for training',
    ), indent=2) + '\n')
    print(json.dumps(dict(stage='complete', counts=dict(counts))), flush=True)


if __name__ == '__main__':
    main()
