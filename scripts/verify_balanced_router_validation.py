#!/usr/bin/env python3
"""Re-score diagnostic references through native vLLM endpoints without training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import urllib.request


def score(url, model, row):
    ids = row['input_ids'] + row['target_ids']
    request = urllib.request.Request(url + '/v1/completions',
        data=json.dumps(dict(model=model, prompt=ids, max_tokens=1, temperature=0,
            seed=42, echo=True, logprobs=1)).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    logp = result['choices'][0]['logprobs']['token_logprobs']
    if result['usage']['prompt_tokens'] != len(ids) or len(logp) != len(ids) + 1:
        raise ValueError('prompt/logprob alignment mismatch')
    values = logp[len(row['input_ids']):len(ids)]
    if not values or any(v is None or not math.isfinite(v) for v in values):
        raise ValueError('invalid target log probabilities')
    return -sum(values) / len(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', required=True)
    p.add_argument('--awq-hosts', nargs='+', default=['10.107.206.208','10.107.206.209','10.107.206.213'])
    p.add_argument('--gptq-hosts', nargs='+', default=['10.107.206.210','10.107.206.211','10.107.206.216'])
    a = p.parse_args()
    rows = [json.loads(line) for line in Path(a.requests).read_text().splitlines()]
    pools = [[f'http://{host}:{port}' for host in hosts for port in range(19000,19008)]
        for hosts in (a.awq_hosts, a.gptq_hosts)]
    # One sequential worker per endpoint; avoids introducing batch-dependent concurrency.
    def worker(expert, endpoint):
        results = []
        for i in range(endpoint, len(rows), len(pools[expert])):
            value = score(pools[expert][endpoint], 'moqe-qwen3-' + ('awq' if expert == 0 else 'gptq'), rows[i])
            results.append((i, expert, value))
        return results
    fresh = [[None, None] for row in rows]
    with ThreadPoolExecutor(max_workers=sum(map(len,pools))) as executor:
        futures = [executor.submit(worker,e,j) for e in range(2) for j in range(len(pools[e]))]
        for future in futures:
            for i,e,value in future.result():
                fresh[i][e] = value
    changed = [rows[i]['id'] for i, values in enumerate(fresh)
        if (values[0] > values[1]) != (rows[i]['expert_losses'][0] > rows[i]['expert_losses'][1])]
    report = dict(samples=len(rows), winner_changes=len(changed), changed_ids=changed,
        fresh_gptq_wins=sum(v[1] < v[0] for v in fresh),
        fresh_awq_wins=sum(v[0] < v[1] for v in fresh),
        minimum_fresh_margin=min(abs(v[0]-v[1]) for v in fresh),
        max_loss_difference=max(abs(fresh[i][e]-r['expert_losses'][e]) for i,r in enumerate(rows) for e in range(2)))
    path = Path(a.requests).with_name('rescore-audit.json')
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if changed:
        raise ValueError('diagnostic winners changed; rebuild before using the split')


if __name__ == '__main__':
    main()
