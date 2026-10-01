#!/usr/bin/env python3
"""Clean all local raw datasets into reproducible request/reference splits.

This prepares offline expert scoring inputs, never fabricates expert_losses.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import unicodedata

import pyarrow.parquet as pq

SOURCES = ("WildChat-1M", "gsm8k", "Magicoder-OSS-Instruct-75K", "LongAlign-10k")
BUDGETS = {"WildChat-1M": 2048, "gsm8k": 1024, "Magicoder-OSS-Instruct-75K": 2048, "LongAlign-10k": 2048}
QUOTAS = dict(zip(SOURCES, (20000, 5000, 10000, 5000)))
SEED = 42
PROMPT_LIMIT = 8192
CONTEXT_LIMIT = 40960


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def json_rows(path):
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if line.strip():
                yield index, json.loads(line)


def load_benchmarks(root, comparison):
    questions = []
    for path in (root / "gsm8k/main").glob("test-*.parquet"):
        questions.extend(pq.read_table(path, columns=["question"])["question"].to_pylist())
    if comparison.is_file():
        questions.extend(row["question"] for _, row in json_rows(comparison) if row.get("question"))
    # A normalized first-96-character fingerprint is a conservative contamination
    # filter, including benchmark questions pasted into longer user messages.
    return sorted({canonical(q)[:96] for q in questions if len(canonical(q)) >= 40})


def init_worker(tokenizer_path, benchmarks):
    global TOKENIZER, PROMPT_ENCODER, ANSWER_ENCODER, BENCH_MATCHER
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    PROMPT_ENCODER = Tokenizer.from_str(TOKENIZER.backend_tokenizer.to_str())
    PROMPT_ENCODER.enable_truncation(max_length=PROMPT_LIMIT + 1)
    ANSWER_ENCODER = Tokenizer.from_str(TOKENIZER.backend_tokenizer.to_str())
    ANSWER_ENCODER.enable_truncation(max_length=2049)
    import ahocorasick
    BENCH_MATCHER = ahocorasick.Automaton()
    for index, text in enumerate(benchmarks):
        BENCH_MATCHER.add_word(text, index)
    BENCH_MATCHER.make_automaton()


def normalized_messages(messages):
    if not isinstance(messages, list) or not messages:
        return None, "invalid_messages"
    normalized = []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            return None, "invalid_role_or_nontext"
        content = content.replace("\r\n", "\n").replace("\r", "\n").removeprefix("\ufeff")
        if not content.strip():
            return None, "empty_message"
        if any(ord(c) < 32 and c not in "\n\t" for c in content):
            return None, "control_character"
        # Use the same non-thinking policy for both experts. Keep natural-language
        # reasoning, such as GSM8K solutions, but remove explicit initial think blocks.
        if role == "assistant" and content.lstrip().startswith("<think>"):
            if "</think>" not in content:
                return None, "unclosed_thinking_block"
            content = content.split("</think>", 1)[1].lstrip("\n")
            if not content.strip():
                return None, "empty_after_thinking_block"
        if "<|im_start|>" in content or "<|im_end|>" in content:
            return None, "embedded_chat_control_token"
        normalized.append({"role": role, "content": content})
    dialogue = normalized[1:] if normalized[0]["role"] == "system" else normalized
    if len(dialogue) < 2 or len(dialogue) % 2 or any(m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(dialogue)):
        return None, "invalid_role_sequence"
    return normalized, None


def prepare(source, source_id, raw):
    if source == "WildChat-1M":
        messages = raw["conversation"]
        # In this first training version exclude redacted conversations rather
        # than attempting to reconstruct missing request/reference semantics.
        if raw.get("redacted") or any(m.get("redacted") for m in messages):
            return None, "redacted"
        if raw.get("toxic") or any(m.get("toxic") for m in messages):
            return None, "source_toxic_flag"
        metadata = {k: raw.get(k) for k in ("language", "model", "turn")}
        group = "wild:" + str(raw["conversation_hash"])
        source_id = str(raw["conversation_hash"])
    elif source == "gsm8k":
        messages = [{"role": "user", "content": raw["question"]}, {"role": "assistant", "content": raw["answer"]}]
        group = "gsm:" + sha(canonical(raw["question"]))
        metadata = {"language": "English", "config": "main", "split": "train", "final_answer": raw["answer"].rsplit("####", 1)[-1].strip()}
    elif source == "Magicoder-OSS-Instruct-75K":
        messages = [{"role": "user", "content": raw["problem"]}, {"role": "assistant", "content": raw["solution"]}]
        group = "code:" + sha(canonical(raw.get("seed") or raw["problem"]))
        source_id = str(raw["index"])
        metadata = {"language": "English", "code_language": raw["lang"]}
    else:
        messages = raw["messages"]
        document = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        group = "long:" + sha(canonical(document)[:4096])
        source_id = str(raw["id"])
        metadata = {"original_length": raw.get("length")}
    messages, error = normalized_messages(messages)
    if error:
        return None, error
    prompt_messages, reference = messages[:-1], messages[-1]["content"]
    if any(next(BENCH_MATCHER.iter(canonical(m["content"])), None) is not None for m in prompt_messages):
        return None, "benchmark_question_overlap"
    # Treat same prompt with different references as one request, to avoid
    # conflicting labels and cross-split duplicates. First occurrence wins.
    key = sha(json.dumps([(m["role"], canonical(m["content"])) for m in prompt_messages], ensure_ascii=False))
    if source == "LongAlign-10k":
        metadata["language"] = "Chinese" if sum("\u4e00" <= c <= "\u9fff" for c in prompt_messages[-1]["content"]) > len(prompt_messages[-1]["content"]) * .15 else "English-or-other"
    first_user = next(m["content"] for m in prompt_messages if m["role"] == "user")
    # Group sufficiently long identical initial user requests across WildChat
    # conversations, even if their subsequent turns differ.
    extra_group = "initial:" + sha(canonical(first_user)) if source == "WildChat-1M" and len(canonical(first_user)) >= 80 else None
    return {"id": source + ":" + source_id + ":" + key[:16], "source": source, "source_id": source_id,
            "group_id": group, "extra_group": extra_group, "prompt_hash": key,
            "messages": prompt_messages, "reference_answer": reference,
            "max_new_tokens": BUDGETS[source], "metadata": metadata}, None


def process_batch(items):
    rejects = Counter()
    candidates, prompt_texts, target_texts = [], [], []
    for source, source_id, raw in items:
        try:
            record, error = prepare(source, source_id, raw)
            if error:
                rejects[error] += 1
                continue
            text = TOKENIZER.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            candidates.append(record)
            prompt_texts.append(text)
            target_texts.append(record["reference_answer"])
        except (KeyError, TypeError, ValueError, AttributeError):
            rejects["invalid_record"] += 1
    prompt_encodings = PROMPT_ENCODER.encode_batch(prompt_texts, add_special_tokens=False)
    target_encodings = ANSWER_ENCODER.encode_batch(target_texts, add_special_tokens=False)
    accepted = []
    for record, prompt, target in zip(candidates, prompt_encodings, target_encodings):
        if len(prompt.ids) > PROMPT_LIMIT:
            rejects["prompt_over_8192"] += 1
        elif len(target.ids) + 1 > record["max_new_tokens"]:
            rejects["reference_over_generation_budget"] += 1
        elif not target.ids:
            rejects["empty_reference_tokens"] += 1
        else:
            record.update(input_ids=prompt.ids, target_ids=target.ids + [TOKENIZER.eos_token_id],
                          prompt_tokens=len(prompt.ids), target_tokens=len(target.ids) + 1)
            if record["prompt_tokens"] + record["max_new_tokens"] > CONTEXT_LIMIT:
                rejects["expert_context_limit"] += 1
                continue
            length_bucket = "0-512" if len(prompt.ids) <= 512 else "513-2048" if len(prompt.ids) <= 2048 else "2049-4096" if len(prompt.ids) <= 4096 else "4097-8192"
            record["length_bucket"] = length_bucket
            record["stratum"] = record["metadata"].get("code_language", record["metadata"].get("language", "unknown")) + "|" + length_bucket
            accepted.append(record)
    return items[0][0], len(items), dict(rejects), accepted


def source_batches(root, source, limit, size=256):
    if source == "WildChat-1M":
        def rows():
            for path in sorted((root / source / "data").glob("*.parquet")):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=size, columns=["conversation_hash", "conversation", "language", "model", "turn", "redacted", "toxic"]):
                    yield from enumerate(batch.to_pylist())
        iterator = rows()
    elif source == "gsm8k":
        def rows():
            index = 0
            for path in sorted((root / source / "main").glob("train-*.parquet")):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=size):
                    for row in batch.to_pylist():
                        yield index, row
                        index += 1
        iterator = rows()
    else:
        path = root / source / ("long.jsonl" if source == "LongAlign-10k" else "data-oss_instruct-decontaminated.jsonl")
        iterator = json_rows(path)
    batch = []
    for count, (index, row) in enumerate(iterator):
        if limit and count >= limit:
            break
        batch.append((source, str(index), row))
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


class Groups:
    def __init__(self):
        self.parents = {}

    def find(self, group):
        root = group
        while self.parents.get(root, root) != root:
            root = self.parents[root]
        while group != root:
            parent = self.parents[group]
            self.parents[group] = root
            group = parent
        return root

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parents.setdefault(a, a)
        self.parents.setdefault(b, b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)


def split_for(group):
    bucket = int(sha(str(SEED) + ":" + group)[:16], 16) % 100
    return "train" if bucket < 90 else "valid" if bucket < 95 else "test"


def allocations(counts, budget):
    total = sum(counts.values())
    budget = min(total, budget)
    if not total:
        return {}
    parts = {k: math.floor(v * budget / total) for k, v in counts.items()}
    order = sorted(counts, key=lambda k: (-(counts[k] * budget / total - parts[k]), k))
    for k in order[:budget - sum(parts.values())]:
        parts[k] += 1
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("dataset/cleaned/v1"))
    parser.add_argument("--tokenizer", default="/home/zjh/mlsys/model/Qwen3-14B-AWQ")
    parser.add_argument("--benchmark", type=Path, default=Path("/home/zjh/mlsys/test-acc/gptq_awq_comparison_20260930/per_question.jsonl"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-per-source", type=int, default=0, help="0 means all records")
    args = parser.parse_args()
    sys.path.insert(0, str(args.dataset_root.resolve() / ".cleaning-deps"))
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    benchmarks = load_benchmarks(args.dataset_root, args.benchmark)
    groups = Groups()
    database = sqlite3.connect(args.output / "records.sqlite")
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("CREATE TABLE records (prompt_hash TEXT PRIMARY KEY, source TEXT, group_id TEXT, stratum TEXT, sample_key TEXT, split TEXT, payload TEXT)")
    stats = {s: {"scanned": 0, "rejected": Counter(), "accepted": 0} for s in SOURCES}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args.tokenizer, benchmarks)) as executor:
        for source in SOURCES:
            # Keep bounded in-flight batches; process results in source order so
            # duplicate retention is deterministic regardless of worker timing.
            pending = []
            def consume(result):
                name, scanned, rejected, records = result
                stats[name]["scanned"] += scanned
                stats[name]["rejected"].update(rejected)
                for record in records:
                    group = record["group_id"]
                    groups.join(group, group)
                    extra = record.pop("extra_group")
                    if extra:
                        groups.join(group, extra)
                    existing = database.execute("SELECT group_id FROM records WHERE prompt_hash=?", (record["prompt_hash"],)).fetchone()
                    if existing:
                        groups.join(group, existing[0])
                        stats[name]["rejected"]["duplicate_normalized_prompt"] += 1
                        continue
                    key = sha(str(SEED) + ":" + record["prompt_hash"])
                    database.execute("INSERT INTO records VALUES (?,?,?,?,?,NULL,?)", (record["prompt_hash"], name, group, record["stratum"], key, json.dumps(record, ensure_ascii=False, separators=(",", ":"))))
                    stats[name]["accepted"] += 1
                if stats[name]["scanned"] % 8192 == 0:
                    database.commit()
                    print(f"PROGRESS {name} scanned={stats[name]['scanned']} accepted={stats[name]['accepted']}", flush=True)
            for batch in source_batches(args.dataset_root, source, args.limit_per_source):
                pending.append(executor.submit(process_batch, batch))
                if len(pending) >= args.workers * 2:
                    consume(pending.pop(0).result())
            for future in pending:
                consume(future.result())
            database.commit()
            print(f"SOURCE_DONE {source} {dict(stats[source])}", flush=True)
            (args.output / "progress.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    split_counts = defaultdict(Counter)
    group_splits = {}
    prompt_hashes = set()
    files = {}
    for split in ("train", "valid", "test"):
        files[split] = (args.output / f"{split}.jsonl").open("w", encoding="utf-8")
    updates = []
    print("EXPORT full splits", flush=True)
    for prompt_hash, source, group, payload in database.execute("SELECT prompt_hash,source,group_id,payload FROM records ORDER BY rowid"):
        root = groups.find(group)
        split = split_for(root)
        assert group_splits.setdefault(root, split) == split
        assert prompt_hash not in prompt_hashes
        prompt_hashes.add(prompt_hash)
        record = json.loads(payload)
        record["group_id"] = root
        record["split"] = split
        files[split].write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        split_counts[source][split] += 1
        updates.append((split, root, prompt_hash))
    for stream in files.values():
        stream.close()
    database.executemany("UPDATE records SET split=?,group_id=? WHERE prompt_hash=?", updates)
    database.commit()
    database.execute("CREATE INDEX sampling ON records(source,split,stratum,sample_key)")
    starter = args.output / "starter"
    starter.mkdir()
    selected = defaultdict(Counter)
    for split in ("train", "valid", "test"):
        with (starter / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for source in SOURCES:
                counts = dict(database.execute("SELECT stratum,COUNT(*) FROM records WHERE source=? AND split=? GROUP BY stratum", (source, split)))
                quota = QUOTAS[source] if split == "train" else math.ceil(QUOTAS[source] / 18)
                for stratum, count in sorted(allocations(counts, quota).items()):
                    for group, payload in database.execute("SELECT group_id,payload FROM records WHERE source=? AND split=? AND stratum=? ORDER BY sample_key LIMIT ?", (source, split, stratum, count)):
                        record = json.loads(payload)
                        record.update(group_id=group, split=split)
                        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                        selected[source][split] += 1
    manifest = {
        "status": "complete", "purpose": "clean request/reference inputs for offline expert NLL scoring; not labeled Router training JSONL",
        "seed": SEED, "pilot_limit_per_source": args.limit_per_source,
        "tokenizer": args.tokenizer, "tokenizer_json_sha256": hashlib.sha256((Path(args.tokenizer) / "tokenizer.json").read_bytes()).hexdigest(),
        "chat_template_sha256": sha(json.loads((Path(args.tokenizer) / "tokenizer_config.json").read_text())["chat_template"]),
        "enable_thinking": False, "expert_ids": ["qwen3-14b/awq-w4a16/v1", "qwen3-14b/gptq-w4a16/v1"],
        "max_prompt_tokens": PROMPT_LIMIT, "context_limit": CONTEXT_LIMIT, "generation_budgets": BUDGETS,
        "target_encoding": "reference encoded without added special tokens, followed by one eos token; score all target_ids only",
        "request_policy": "one last-assistant target per raw conversation; preserve full history",
        "deduplication": "NFKC + casefold + whitespace-normalized prompt messages; keep first accepted reference",
        "benchmark_fingerprints": len(benchmarks), "benchmark_filter": "normalized first 96 characters, at least 40 characters, substring matched in prompt history",
        "group_split": "90/5/5 hash split; exact shared prompt, long first WildChat request, code seed and LongAlign document prefix grouped",
        "starter_sampling": "proportional language/code-language and prompt-length strata with hash ordering; source train quotas 20000/5000/10000/5000, holdout quotas ceil(train quota/18)",
        "statistics": stats, "full_split_counts": split_counts, "starter_split_counts": selected,
        "missing_sources": ["lmsys/lmsys-chat-1m (gated; data shards unavailable)"],
        "limitations": ["No semantic near-duplicate guarantee", "References not verified for factual correctness or executable code accuracy", "Benchmark filter is conservative fingerprint matching, not semantic contamination detection", "Same documents with different prefixes may evade source grouping"],
    }
    file_manifest = []
    for path in sorted(args.output.glob("*.jsonl")) + sorted(starter.glob("*.jsonl")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        file_manifest.append({"path": str(path.relative_to(args.output)), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    manifest["files"] = file_manifest
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    database.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    database.close()
    print("COMPLETE " + json.dumps({"full": split_counts, "starter": selected}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
