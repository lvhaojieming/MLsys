"""Pair frozen-expert losses by sample ID and prefetch them during training."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .data import TrainingExample, collate_requests


def token_fingerprint(row: dict) -> str:
    return hashlib.sha256(json.dumps(row['input_ids'] + row['target_ids']).encode()).hexdigest()


class PairedExpertLossCache:
    """Every returned row has both losses for identical tokens, in expert order.

    The two endpoint calls may finish at different times. No partially scored
    row can reach the Router optimizer. Frozen-expert results are reusable.
    """

    def __init__(self, experts: list[dict], directory: Path):
        import httpx
        self.experts = experts
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.clients = []
        self.scores: list[dict[str, dict]] = []
        self.files = []
        self.executor = ThreadPoolExecutor(max_workers=2)
        for index, expert in enumerate(experts):
            client = httpx.Client(base_url=f'http://127.0.0.1:{expert["port"]}', timeout=3600, trust_env=False)
            response = client.get('/health'); response.raise_for_status()
            health = response.json()
            if health['expert_id'] != expert['expert_id'] or health['cuda_visible_devices'] != str(expert['gpu']):
                raise ValueError('expert endpoint/GPU identity mismatch')
            self.clients.append(client)
            scores = {}
            path = directory / f'expert-{index}.jsonl'
            if path.exists():
                for line in path.open():
                    row = json.loads(line)
                    if row['id'] in scores: raise ValueError('duplicate cached sample ID')
                    if row['expert_id'] != expert['expert_id']: raise ValueError('cached expert order mismatch')
                    scores[row['id']] = row
            self.scores.append(scores)
            self.files.append(path.open('a'))

    def _ensure_one(self, index: int, rows: list[dict]) -> None:
        expert = self.experts[index]
        scores = self.scores[index]
        for row in rows:
            if row['id'] in scores:
                cached=scores[row['id']]
                if cached['tokens_sha256'] != token_fingerprint(row) or cached['target_tokens'] != len(row['target_ids']):
                    raise ValueError('cached loss belongs to different prompt/target tokens')
                if not math.isfinite(cached['mean_target_nll']) or cached['mean_target_nll']<0:
                    raise ValueError('invalid cached expert loss')
        missing = [r for r in rows if r['id'] not in scores]
        if not missing: return
        payload = {'rows': [{k: r[k] for k in ('id', 'input_ids', 'target_ids')} for r in missing]}
        response = self.clients[index].post('/score', json=payload); response.raise_for_status()
        result = response.json()
        if result['expert_id'] != expert['expert_id'] or str(result['gpu']) != str(expert['gpu']):
            raise ValueError('loss response expert/GPU identity mismatch')
        if [r['id'] for r in result['rows']] != [r['id'] for r in missing]:
            raise ValueError('experts must score exactly the same requested sample IDs')
        for row, score in zip(missing, result['rows']):
            loss = score['mean_target_nll']
            if not math.isfinite(loss) or loss < 0 or score['target_tokens'] != len(row['target_ids']):
                raise ValueError('invalid expert target loss')
            score.update(expert_id=expert['expert_id'], gpu=expert['gpu'], tokens_sha256=token_fingerprint(row))
            self.files[index].write(json.dumps(score) + '\n')
            scores[row['id']] = score
        self.files[index].flush()

    def ensure(self, rows: list[dict]) -> list[TrainingExample]:
        if not rows or len(rows) > 64: raise ValueError('loss window must contain 1..64 requests')
        if len({r['id'] for r in rows}) != len(rows): raise ValueError('duplicate IDs in loss window')
        futures = [self.executor.submit(self._ensure_one, index, rows) for index in range(2)]
        # Wait for BOTH experts; completion speed never changes dataset membership.
        for future in futures: future.result()
        examples = []
        for row in rows:
            losses = tuple(self.scores[index][row['id']]['mean_target_nll'] for index in range(2))
            examples.append(TrainingExample(row['id'], tuple(row['input_ids']), row['max_new_tokens'], losses))
        return examples

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        for client in self.clients: client.close()
        for stream in self.files: stream.close()


def prefetched_batches(rows: list[dict], cache: PairedExpertLossCache, batch_size: int, window_batches: int = 6):
    """Compute the next expert-loss window while training the current one.

    Preserve the caller's shuffled sample order; each row appears exactly once.
    The first window need not wait for losses on the rest of the dataset.
    """
    if not 1<=batch_size<=64 or window_batches<1: raise ValueError('batch size must be 1..64 and prefetch count positive')
    window = batch_size * min(window_batches,64//batch_size)
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        future = prefetch.submit(cache.ensure, rows[:window]) if rows else None
        for start in range(0, len(rows), window):
            examples = future.result()
            next_start = start + window
            if next_start < len(rows):
                future = prefetch.submit(cache.ensure, rows[next_start:next_start + window])
            for offset in range(0, len(examples), batch_size):
                yield collate_requests(examples[offset:offset + batch_size])
