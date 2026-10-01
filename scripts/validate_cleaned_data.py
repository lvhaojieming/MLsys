#!/usr/bin/env python3
"""Independently audit split isolation, token contracts, and sample encoding."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.path / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(manifest["tokenizer"], local_files_only=True)
    vocabulary_size = len(tokenizer)
    results = {}
    full_membership = {}
    for directory, expected_key in ((args.path, "full_split_counts"), (args.path / "starter", "starter_split_counts")):
        ids, hashes, groups = set(), set(), {}
        counts = defaultdict(Counter)
        encoded = Counter()
        for split in ("train", "valid", "test"):
            with (directory / f"{split}.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    assert row["split"] == split
                    assert row["id"] not in ids, ("duplicate id", row["id"])
                    assert row["prompt_hash"] not in hashes, "duplicate prompt"
                    ids.add(row["id"])
                    hashes.add(row["prompt_hash"])
                    assert groups.setdefault(row["group_id"], split) == split, "group leakage"
                    assert row["messages"][-1]["role"] == "user"
                    assert all(m["content"].strip() for m in row["messages"])
                    assert row["reference_answer"].strip()
                    assert len(row["input_ids"]) == row["prompt_tokens"] <= manifest["max_prompt_tokens"]
                    assert 0 < len(row["target_ids"]) == row["target_tokens"] <= row["max_new_tokens"]
                    assert row["target_ids"][-1] == tokenizer.eos_token_id
                    assert row["prompt_tokens"] + row["max_new_tokens"] <= manifest["context_limit"]
                    assert all(type(t) is int and 0 <= t < vocabulary_size for t in row["input_ids"] + row["target_ids"])
                    assert "expert_losses" not in row, "unscored data cannot have synthetic labels"
                    if directory == args.path:
                        full_membership[row["id"]] = split
                    else:
                        assert full_membership[row["id"]] == split, "starter sample not in corresponding full split"
                    key = (row["source"], split)
                    if encoded[key] < 10:
                        assert tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False) == row["input_ids"], "prompt encoding mismatch"
                        assert tokenizer.encode(row["reference_answer"], add_special_tokens=False) + [tokenizer.eos_token_id] == row["target_ids"], "answer encoding mismatch"
                        encoded[key] += 1
                    counts[row["source"]][split] += 1
        assert {s: dict(c) for s, c in counts.items()} == manifest[expected_key]
        results[directory.name] = {"rows": len(ids), "groups": len(groups), "encoding_checks": sum(encoded.values()), "counts": counts}
        print(f"VALIDATED {directory}: rows={len(ids)} groups={len(groups)}", flush=True)
    for source, stats in manifest["statistics"].items():
        assert stats["scanned"] == stats["accepted"] + sum(stats["rejected"].values()), source
        assert stats["accepted"] == sum(manifest["full_split_counts"][source].values()), source
    (args.path / "validation.json").write_text(json.dumps({"status": "passed", "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("ALL_CHECKS_PASSED", flush=True)


if __name__ == "__main__":
    main()
