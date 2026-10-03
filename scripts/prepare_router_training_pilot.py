#!/usr/bin/env python3
"""Select 1024 raw training requests without cleaning or prompt truncation."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.raw_root)
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    sources = ['WildChat-1M', 'gsm8k', 'Magicoder-OSS-Instruct-75K', 'LongAlign-10k',
               'mmlu', 'mbpp', 'qasper', 'wikitext2', 'c4']
    output = Path(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    def raw_rows(source):
        for file in sorted((root/source).rglob('*')):
            if '.cache' in file.parts or not file.is_file():
                continue
            if source == 'gsm8k' and not file.name.startswith('train-'):
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
    def requests(source):
        for index, r in enumerate(raw_rows(source)):
            group = source+':'+str(index)
            if source in ('WildChat-1M', 'LongAlign-10k'):
                messages = r['conversation'] if source == 'WildChat-1M' else r['messages']
                end = next((i for i in range(len(messages)-1, 0, -1) if messages[i]['role']=='assistant'), None)
                if end is None:
                    continue
                group = source+':'+str(r.get('conversation_hash', r.get('id', index)))
                yield group, str(index), [{'role':m['role'], 'content':m['content']} for m in messages[:end]], messages[end]['content']
            elif source == 'gsm8k':
                yield group, str(index), [{'role':'user','content':r['question']}], r['answer']
            elif source == 'Magicoder-OSS-Instruct-75K':
                yield group, str(index), [{'role':'user','content':r['problem']}], r['solution']
            elif source == 'mmlu':
                if 'train' in r:
                    r = r['train']
                prompt = r['question']+'\n'+'\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(r['choices']))+'\nAnswer:'
                yield group, str(index), [{'role':'user','content':prompt}], chr(65+int(r['answer']))
            elif source == 'mbpp':
                yield group, str(index), [{'role':'user','content':r['text']}], r['code']
            elif source == 'qasper':
                full_text = r['full_text']
                paragraphs = full_text['paragraphs'] if isinstance(full_text, dict) else [section['paragraphs'] for section in full_text]
                document = r['title']+'\n'+r['abstract']+'\n'+'\n'.join('\n'.join(section) for section in paragraphs)
                qas = r['qas']
                for q, question in enumerate(qas['question']):
                    annotations = qas['answers'][q]['answer']
                    answer = annotations[0]
                    reference = answer.get('free_form_answer') or ' '.join(answer.get('extractive_spans', []))
                    if not reference and answer.get('yes_no') is not None:
                        reference = 'Yes' if answer['yes_no'] else 'No'
                    if not reference and answer.get('unanswerable'):
                        reference = 'unanswerable'
                    if reference:
                        yield group, f'{index}:{q}', [{'role':'user','content':document+'\nQuestion: '+question}], reference
            else:
                text = r['text']
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) < 4:
                    continue
                # Plain-text corpora provide causal continuation windows, not chat answers.
                tokens = tokens[:2048]
                midpoint = max(1, len(tokens)//2)
                yield group, str(index), tokens[:midpoint], tokens[midpoint:]
    counts = Counter(); selection = {}
    with output.open('w', encoding='utf-8') as stream:
        for number, source in enumerate(sources):
            train_quota = 114 if number < 7 else 113
            valid_quota = 7 if number < 8 else 8
            train_groups = set(); filled = Counter(); bounds = 0
            for group, sid, messages, reference in requests(source):
                if isinstance(messages[0], int):
                    ids, target = messages, reference+[tokenizer.eos_token_id]
                else:
                    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                    target = tokenizer.encode(reference, add_special_tokens=False)+[tokenizer.eos_token_id]
                if not ids or not target or len(target)>2048 or len(ids)+len(target)>12287:
                    bounds += 1
                    continue
                split = 'train' if filled['train'] < train_quota else 'valid'
                if split == 'valid' and group in train_groups:
                    continue
                if split == 'train':
                    train_groups.add(group)
                row = dict(id=source+':'+sid, source=source, group_id=group, split=split,
                           input_ids=ids, target_ids=target, max_new_tokens=2048)
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                filled[split] += 1; counts[split] += 1
                if filled['train']==train_quota and filled['valid']==valid_quota:
                    break
            if filled['train'] != train_quota or filled['valid'] != valid_quota:
                raise RuntimeError(f'insufficient fitting pilot samples: {source}: {filled}')
            selection[source] = dict(filled, outside_pilot_context_or_target_bounds=bounds)
            print(json.dumps(dict(source=source, **selection[source])), flush=True)
    assert counts['train']==1024 and counts['valid']==64
    output.with_suffix('.manifest.json').write_text(json.dumps(dict(counts=counts, sources=selection,
        cleaning=False, prompt_truncation=False, context_limit=12288, target_limit=2048,
        selection='first fitting raw training samples per source; disjoint source groups'), indent=2)+'\n')


if __name__ == '__main__':
    main()
