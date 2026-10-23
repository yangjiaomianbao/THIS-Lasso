"""Full E1: bootstrap seed clusters and export figures and raw paired differences."""
import csv
from pathlib import Path
import sys
import tomllib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r4_experiment import write_rows
from r5_batch_stats import bootstrap_indices, describe, paired_values, validate_full

METRICS = ('precision', 'recall', 'F1', 'MCC', 'AUPRC_AP', 'TP', 'FP', 'FN', 'TN')
COLORS = dict(M0='#60666e', M1='#0072b2', M2='#d55e00', M3='#009e73')


def vector(rows, seeds, ratio, method, key):
    by_seed = {int(r['seed']):r for r in rows if float(r['noise_ratio']) == ratio and r['method'] == method}
    return np.array([float(by_seed[s][key]) if by_seed[s]['status'] == 'OK' and by_seed[s].get(key, '') != '' else np.nan for s in seeds])


def make_summaries(rows, strata, c, output):
    seeds = c['seeds'];indices = bootstrap_indices(len(seeds), c['bootstrap_replicates'], c['bootstrap_seed'])
    np.savetxt(output/'bootstrap_seed_indices.csv', indices, delimiter=',', fmt='%d')
    summaries, paired, raw_paired, strength, sensitivity = [], [], [], [], []
    for ratio in c['noise_ratios']:
        for method in c['method_order']:
            group = [r for r in rows if float(r['noise_ratio']) == ratio and r['method'] == method]
            availability = dict(attempted=30, OK_count=sum(r['status'] == 'OK' for r in group),
                NA_count=sum(r['status'] == 'NA' for r in group), FAILED_count=sum(r['status'] == 'FAILED' for r in group))
            for order in ('pairwise', 'triple'):
                for metric in METRICS:
                    values = vector(rows, seeds, ratio, method, order+'_'+metric)
                    summaries.append(dict(noise_ratio=ratio, method=method, order=order, metric=metric,
                                          **availability, **describe(values, indices)))
                    if metric in ('F1', 'AUPRC_AP'):
                        new_values = values[2:]
                        sensitivity.append(dict(noise_ratio=ratio, method=method, order=order, metric=metric,
                            n_total=28, n_available=int(np.isfinite(new_values).sum()),
                            mean=float(np.nanmean(new_values)) if np.isfinite(new_values).any() else float('nan')))
            for order in ('pairwise', 'triple'):
                for label in ('weak', 'medium', 'strong'):
                    chosen = {int(r['seed']):r for r in strata if float(r['noise_ratio']) == ratio
                              and r['method'] == method and r['order'] == order and r['strength'] == label}
                    values = np.array([float(chosen[s]['recall']) if chosen[s]['status'] == 'OK' and chosen[s]['recall'] != '' else np.nan for s in seeds])
                    strength.append(dict(noise_ratio=ratio, method=method, order=order, strength=label, **describe(values, indices)))
        for a, b in [('M2', 'M1'), ('M2', 'M0'), ('M3', 'M2')]:
            for order in ('pairwise', 'triple'):
                raw = [dict(seed=s, noise_ratio=ratio, contrast=a+'-'+b, order=order) for s in seeds]
                for metric in METRICS:
                    differences = paired_values(vector(rows, seeds, ratio, a, order+'_'+metric),
                                                vector(rows, seeds, ratio, b, order+'_'+metric))
                    paired.append(dict(noise_ratio=ratio, contrast=a+'-'+b, order=order, metric=metric,
                                       **describe(differences, indices)))
                    for record, delta in zip(raw, differences):
                        record[metric+'_difference'] = float(delta)
                raw_paired.extend(raw)
    write_rows(output/'summary_30.csv', summaries);write_rows(output/'paired_summary_30.csv', paired)
    write_rows(output/'paired_seed_differences.csv', raw_paired);write_rows(output/'strength_summary_30.csv', strength)
    write_rows(output/'sensitivity_new_28.csv', sensitivity)
    return summaries, paired, strength


def curves(ax, records, c, filters, ylabel):
    for method in c['method_order']:
        selected = [r for r in records if r.get('method') == method and all(r.get(k) == v for k, v in filters.items())]
        selected.sort(key=lambda r:r['noise_ratio'])
        x = np.array([r['noise_ratio'] for r in selected]);y = np.array([r['mean'] for r in selected])
        lo = np.array([r['ci_low'] for r in selected]);hi = np.array([r['ci_high'] for r in selected])
        ax.plot(x, y, 'o-', label=method, color=COLORS[method], markersize=4)
        ax.fill_between(x, lo, hi, color=COLORS[method], alpha=.13)
    ax.set(xlabel='Response noise SD ratio', ylabel=ylabel, xticks=c['noise_ratios'])
    ax.grid(alpha=.2)


def plot_results(summaries, paired, strength, c, output):
    directory = output/'figures';directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10, 'pdf.fonttype':42})
    def save(fig, name):
        fig.savefig(directory/(name+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(directory/(name+'.pdf'), bbox_inches='tight');plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, label in zip(axes, ('F1', 'AUPRC_AP'), ('Triple F1', 'Triple AP')):
        curves(ax, summaries, c, dict(order='triple', metric=metric), label);ax.set_ylim(-.03, 1.04)
    axes[0].legend();fig.suptitle('E1: 30 paired seeds; pointwise 95% seed-bootstrap intervals')
    fig.tight_layout();save(fig, 'triple_f1_ap')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ('FP', 'FN')):
        curves(ax, summaries, c, dict(order='triple', metric=metric), 'Triple '+metric+' per seed')
    axes[0].legend();fig.suptitle('E1: false positives and false negatives, 30 paired seeds')
    fig.tight_layout();save(fig, 'triple_fp_fn')
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for row, order in enumerate(('pairwise', 'triple')):
        for col, label in enumerate(('weak', 'medium', 'strong')):
            curves(axes[row, col], strength, c, dict(order=order, strength=label), 'True-edge recall')
            axes[row, col].set(title=order+' / '+label, ylim=(-.03, 1.04))
    axes[0, 0].legend();fig.suptitle('Recall by preregistered TRUE physical strength; pointwise 95% intervals')
    fig.tight_layout();save(fig, 'strength_recall')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ('F1', 'AUPRC_AP')):
        for contrast, offset, color in [('M2-M1', -.002, '#0072b2'), ('M2-M0', .002, '#d55e00')]:
            chosen = sorted([r for r in paired if r['order'] == 'triple' and r['metric'] == metric and r['contrast'] == contrast], key=lambda r:r['noise_ratio'])
            x = np.array([r['noise_ratio'] for r in chosen])+offset;y = np.array([r['mean'] for r in chosen])
            lo = np.array([r['ci_low'] for r in chosen]);hi = np.array([r['ci_high'] for r in chosen])
            ax.plot(x, y, 'o-', label=contrast, color=color)
            ax.vlines(x, lo, hi, color=color)
            ax.plot(x, lo, '_', color=color);ax.plot(x, hi, '_', color=color)
        ax.axhline(0., color='black', linewidth=.8);ax.set(xlabel='Response noise SD ratio', ylabel='Paired triple '+metric+' difference', xticks=c['noise_ratios'])
        ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Pointwise paired-seed bootstrap; NA at zero noise for BIC')
    fig.tight_layout();save(fig, 'paired_triple_differences')


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--output', type=Path, required=True);args = p.parse_args()
    c = tomllib.loads((args.output/'config_locked.toml').read_text());validate_full(c)
    rows = list(csv.DictReader((args.output/'trials.csv').open()))
    strata = list(csv.DictReader((args.output/'strength_recall.csv').open()))
    if len(rows) != 480 or len(strata) != 2880:
        raise ValueError('Full E1 trial inventory is incomplete')
    summaries, paired, strength = make_summaries(rows, strata, c, args.output)
    plot_results(summaries, paired, strength, c, args.output)
    print('R5_E1_FULL_STATISTICS_AND_FIGURES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
