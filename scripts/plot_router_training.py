#!/usr/bin/env python3
"""Export a snapshot of raw online training metrics and publication figures."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_snapshot(path):
    data = path.read_bytes()
    # The writer can be midway through its last append. Keep complete lines only.
    complete = data[:data.rfind(b'\n') + 1]
    records = [json.loads(line) for line in complete.splitlines() if line.strip()]
    return complete, records


def write_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def moving_mean(values, window):
    # Trailing window: no future values enter a displayed point.
    result, total = [], 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        result.append(total / min(i + 1, window))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'outputs/qwen3-14b-router-hq-v2')
    parser.add_argument('--window', type=int, default=50)
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    run = args.run_dir.resolve()
    raw, records = read_snapshot(run / 'metrics.jsonl')
    train, valid = [], []
    seen = set()
    for record in records:
        if record.get('split') == 'train':
            key = (record['epoch'], record['step'])
            if key in seen:
                raise ValueError(f'Duplicate update {key}; separate restarted attempts before plotting')
            seen.add(key)
            row = dict(epoch=record['epoch'], step=record['step'], router_loss=record['loss'],
                       awq_nll=record['expert_loss_mean'][0], gptq_nll=record['expert_loss_mean'][1],
                       batch_samples=record['batch_samples'], lr=record['lr'], grad_norm=record['grad_norm'],
                       paired_samples_trained=record['paired_samples_trained'])
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f'Nonfinite training metric at {key}')
            if len(record['sample_ids']) != row['batch_samples']:
                raise ValueError(f'Batch ID count mismatch at {key}')
            train.append(row)
        elif record.get('split') == 'valid':
            row = {key: record[key] for key in ('epoch', 'loss', 'top1_accuracy', 'mean_routing_regret')}
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError('Nonfinite validation metric')
            valid.append(row)
    if not train:
        raise ValueError('No training updates to plot')
    if any(b['step'] <= a['step'] for a, b in zip(train, train[1:])):
        raise ValueError('Steps are not strictly increasing; split restarted attempts first')
    out = run / 'figures'
    out.mkdir(exist_ok=True)
    (out / 'metrics_snapshot.jsonl').write_bytes(raw)
    write_csv(out / 'train_loss.csv', train, list(train[0]))
    write_csv(out / 'validation.csv', valid, ['epoch', 'loss', 'top1_accuracy', 'mean_routing_regret'])
    epochs = []
    for epoch in sorted({row['epoch'] for row in train}):
        rows = [row for row in train if row['epoch'] == epoch]
        samples = sum(row['batch_samples'] for row in rows)
        row = dict(epoch=epoch, updates=len(rows), samples=samples,
                   validation_recorded=any(v['epoch'] == epoch for v in valid))
        for key in ('router_loss', 'awq_nll', 'gptq_nll'):
            row[key] = sum(r[key] * r['batch_samples'] for r in rows) / samples
        epochs.append(row)
    write_csv(out / 'epoch_loss.csv', epochs, list(epochs[0]))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2 if valid else 1, 2, figsize=(10, 6.4 if valid else 3.3), squeeze=False)
    steps = [r['step'] for r in train]
    loss = [r['router_loss'] for r in train]
    ax = axes[0, 0]
    ax.plot(steps, loss, color='#0072B2', alpha=.22, linewidth=.6, label='Raw minibatch CE')
    ax.plot(steps, moving_mean(loss, args.window), color='#0072B2', linewidth=1.4,
            label=f'Trailing mean ({args.window} updates)')
    ax.set(xlabel='Optimizer step', ylabel='Router cross-entropy', title='Router training loss')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for key, label, color in [('awq_nll', 'AWQ', '#009E73'), ('gptq_nll', 'GPTQ Int4', '#D55E00')]:
        values = [r[key] for r in train]
        ax.plot(steps, values, color=color, alpha=.15, linewidth=.5)
        ax.plot(steps, moving_mean(values, args.window), color=color, linewidth=1.2, label=label)
    ax.set(xlabel='Optimizer step', ylabel='Mean target NLL (nats/token)', title='Frozen experts on paired minibatches')
    ax.legend(fontsize=8)
    if valid:
        e = [v['epoch'] for v in valid]
        axes[1, 0].plot(e, [v['loss'] for v in valid], 'o-', color='#0072B2')
        axes[1, 0].set(xlabel='Completed epoch', ylabel='Router cross-entropy', title='Validation loss')
        axes[1, 1].plot(e, [v['mean_routing_regret'] for v in valid], 'o-', color='#CC79A7')
        axes[1, 1].set(xlabel='Completed epoch', ylabel='Mean routing regret (nats/token)', title='Validation routing regret')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(out / f'loss_curves.{suffix}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    metadata = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                    metrics_sha256=hashlib.sha256(raw).hexdigest(), updates=len(train),
                    last_step=train[-1]['step'], last_epoch=train[-1]['epoch'],
                    training_samples=sum(r['batch_samples'] for r in train), validation_epochs=len(valid),
                    smoothing='Trailing arithmetic mean for display only; CSV contains unsmoothed values',
                    smoothing_window=args.window, expert_order=['AWQ', 'GPTQ Int4'],
                    note='Expert curves vary with minibatch composition; expert weights are frozen. '
                         'Epoch CSV can contain an incomplete current epoch; check validation_recorded.')
    for name in ('training-effective.json', 'deployment.json'):
        if (run / name).exists():
            (out / name).write_bytes((run / name).read_bytes())
    (out / 'plot_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), **metadata), indent=2))


if __name__ == '__main__':
    main()
