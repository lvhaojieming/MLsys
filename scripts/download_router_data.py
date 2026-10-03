#!/usr/bin/env python3
"""Fetch the four raw sources used by the existing cleaning pipeline."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SOURCES = (
    ("WildChat-1M", "allenai/WildChat-1M", ["data/*.parquet"]),
    ("gsm8k", "openai/gsm8k", ["main/*.parquet"]),
    ("Magicoder-OSS-Instruct-75K", "ise-uiuc/Magicoder-OSS-Instruct-75K",
     ["data-oss_instruct-decontaminated.jsonl"]),
    ("LongAlign-10k", "THUDM/LongAlign-10k", ["long.jsonl"]),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", choices=[s[0] for s in SOURCES],
                        default=[s[0] for s in SOURCES])
    args = parser.parse_args()
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dataset_root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    api = HfApi()
    for name, repo, patterns in SOURCES:
        if name not in args.sources:
            continue
        revision = manifest.get(name, {}).get("revision") or api.dataset_info(repo).sha
        directory = args.dataset_root / name
        snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                          allow_patterns=patterns, local_dir=directory, max_workers=2)
        files = sorted(f for pattern in patterns for f in directory.glob(pattern) if f.is_file())
        if not files:
            raise RuntimeError(f"No matching raw data downloaded for {repo}")
        hashes = {}
        for file in files:
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(file.relative_to(directory))] = digest.hexdigest()
        manifest[name] = dict(repo_id=repo, revision=revision, files=hashes)
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
        print(json.dumps(dict(stage="downloaded", source=name, files=len(files))), flush=True)


if __name__ == "__main__":
    main()
