"""Render only complete three-seed native-data comparisons on common concepts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

from .metrics import classification_metrics, concept_metrics, paired_concept_metrics
from .report_utils import backbone_label, write_metrics

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
COLORS = {'LF-CBM': '#2878B5', 'LF-Pi': '#E15759', '非 CBM': '#F28E2B'}


def read(path, fields):
    with np.load(path) as data:
        return {key: data[key].copy() for key in fields}


def complete(directory, seed, paired):
    report = directory / 'report.json'
    metadata = directory / 'single_draw/metadata.json'
    if not report.exists() or not metadata.exists():
        return None
    required = [directory / f'lfcbm_test_{view}.npz' for view in ['original', 'changed']]
    required += [directory / 'lfcbm_train_original.npz']
    required += [directory / 'single_draw' / f'lfpi_global_{split}_{view}.npz'
                 for split in ['train', 'val', 'test'] for view in ['original', 'changed']]
    if not all(p.is_file() for p in required):
        return None
    r = json.loads(report.read_text())
    m = json.loads(metadata.read_text())
    assert r['training_seed'] == seed and not r['test_selection']
    assert r['train_views'] == (['original', 'changed'] if paired else ['original'])
    assert m['inference_draws'] == 1 and m['inference_seed'] == 3000
    return r


def summarize(values):
    a = np.asarray(values, dtype=float)
    assert len(a) == 3 and np.isfinite(a).all()
    return {'mean': float(a.mean()), 'std': float(a.std(ddof=1)),
            'values': a.tolist(), 'seeds': SEEDS, 'n': 3}


def evaluate(dataset, bone, folder, paired, reports, directories):
    cache = ROOT / 'artifacts/common' / dataset / folder
    raw = {view: read(cache / f'test_{view}.npz', ['ids', 'labels', 'targets', 'known'])
           for view in ['original', 'changed']}
    reference = raw['original']
    assert np.array_equal(reference['ids'], raw['changed']['ids'])
    assert np.array_equal(reference['labels'], raw['changed']['labels'])
    manifest = json.loads((cache / 'manifest.json').read_text())
    eligible_path = Path(manifest['cache_root']) / 'background_eligible.npy'
    mask = np.load(eligible_path)[reference['ids']].astype(bool) if eligible_path.exists() else np.ones(len(reference['ids']), bool)
    assert mask.any()
    common = set(reports[0]['selected_indices'])
    for r in reports:
        assert r['dataset'] == dataset and r['feature_provenance'] == manifest
        assert r['teacher_provenance'] == reports[0]['teacher_provenance']
        common &= set(r['selected_indices'])
    common = sorted(common)
    if not common:
        raise ValueError('No common concept columns across the three actual trained seeds')
    per_seed = []
    fields = ['ids', 'labels', 'logits', 'concepts', 'targets', 'known', 'concept_names']
    for seed, report, directory in zip(SEEDS, reports, directories):
        selected = report['selected_indices']
        local = {column: i for i, column in enumerate(selected)}
        columns = [local[c] for c in common]
        row = {'seed': seed, 'retained_concepts': len(selected), 'methods': {}}
        for method, source, prefix in [('LF-CBM', directory, 'lfcbm_'),
                                       ('LF-Pi', directory / 'single_draw', 'lfpi_global_')]:
            data = {view: read(source / f'{prefix}test_{view}.npz', fields)
                    for view in ['original', 'changed']}
            train = read(source / f'{prefix}train_original.npz', ['concepts'])['concepts']
            result = {'inference_draws': 1, 'source': str(source)}
            for view in ['original', 'changed']:
                d = data[view]
                assert np.array_equal(d['ids'], reference['ids'])
                assert np.array_equal(d['labels'], reference['labels'])
                assert np.array_equal(d['targets'][:, columns], raw[view]['targets'][:, common])
                assert np.array_equal(d['known'][:, columns], raw[view]['known'][:, common])
                assert list(d['concept_names']) == report['selected_names']
                assert np.isfinite(d['concepts']).all() and np.isfinite(train).all()
                result[view] = {
                    'accuracy': classification_metrics(d['logits'][mask], d['labels'][mask])['accuracy'],
                    'concepts': concept_metrics(d['concepts'][mask][:, columns],
                                               d['targets'][mask][:, columns], d['known'][mask][:, columns])}
            result['standardized_concept_shift'] = paired_concept_metrics(
                data['original']['concepts'][mask][:, columns],
                data['changed']['concepts'][mask][:, columns], train[:, columns])['standardized_mae']
            row['methods'][method] = result
        control = ROOT / 'artifacts/noncbm' / f"{dataset}_{folder}{'_paired' if paired else ''}" / f'seed{seed}'
        controls = [control / 'results.json', control / 'test_predictions.npz', control / 'changed_predictions.npz']
        if all(p.is_file() for p in controls):
            r = json.loads(controls[0].read_text())
            assert r['seed'] == seed and r['nec'] is None
            assert Path(r['input_paths']['test']).resolve() == (cache / 'test_original.npz').resolve()
            result = {'source': str(control), 'nec': None}
            for view, file in [('original', 'test_predictions.npz'), ('changed', 'changed_predictions.npz')]:
                d = read(control / file, ['ids', 'labels', 'logits'])
                assert np.array_equal(d['ids'], reference['ids']) and np.array_equal(d['labels'], reference['labels'])
                result[view] = {'accuracy': classification_metrics(d['logits'][mask], d['labels'][mask])['accuracy']}
            row['methods']['非 CBM'] = result
        per_seed.append(row)
    aggregate = {}
    methods = set.intersection(*(set(r['methods']) for r in per_seed))
    for method in COLORS:
        if method not in methods:
            continue
        result = {view: {'accuracy': summarize([r['methods'][method][view]['accuracy'] for r in per_seed])}
                  for view in ['original', 'changed']}
        if method != '非 CBM':
            for view in ['original', 'changed']:
                for key in ['mAP', 'pearson_macro']:
                    result[view][key] = summarize([r['methods'][method][view]['concepts'][key] for r in per_seed])
            result['standardized_concept_shift'] = summarize([r['methods'][method]['standardized_concept_shift'] for r in per_seed])
        aggregate[method] = result
    gain = {view: summarize([r['methods']['LF-Pi'][view]['accuracy'] - r['methods']['LF-CBM'][view]['accuracy'] for r in per_seed])
            for view in ['original', 'changed']}
    return {'dataset': dataset, 'backbone': bone, 'protocol': 'paired' if paired else 'original',
            'seeds': SEEDS, 'test_subjects': int(mask.sum()), 'shared_concept_indices': common,
            'shared_concepts': len(common), 'inference_draws': 1,
            'error_bars': 'sample standard deviation over three training seeds, ddof=1',
            'concept_evaluation': 'same native concept columns and known masks across all three seeds and both methods',
            'per_seed': per_seed, 'aggregate': aggregate, 'paired_accuracy_gain': gain}


def bars(ax, names, values, scale, title, ylabel):
    means = [scale * v['mean'] for v in values]
    errors = [scale * v['std'] for v in values]
    x = np.arange(len(names))
    patches = ax.bar(x, means, yerr=errors, capsize=5,
                     color=[COLORS[n] for n in names], width=.6)
    ax.bar_label(patches, fmt='%.2f', padding=5)
    for i, v in enumerate(values):
        ax.scatter(i + np.asarray([-.12, 0, .12]), scale * np.asarray(v['values']),
                   color='#303030', s=15, zorder=3)
    ax.set_xticks(x, names); ax.set_title(title); ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    font = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    spec = json.loads((ROOT / 'configs/lf_matched_runs.json').read_text())
    completed, pending = [], []
    for dataset in spec['datasets']:
        for bone in spec['backbones']:
            folder = spec['backbone_cache_names'][bone]
            for paired in [False, True]:
                parent = ROOT / 'artifacts/matched/lf' / dataset / folder
                if paired:
                    parent /= 'paired_augmentation'
                dirs = [parent / f'seed{s}' for s in SEEDS]
                reports = [complete(d, s, paired) for d, s in zip(dirs, SEEDS)]
                if any(r is None for r in reports):
                    pending.append({'dataset': dataset, 'backbone': bone, 'paired': paired})
                    continue
                completed.append(evaluate(dataset, bone, folder, paired, reports, dirs))
    print(json.dumps({'complete_groups': len(completed), 'pending_groups': len(pending)}), flush=True)
    if not completed:
        return
    out = ROOT / 'reports'
    lines = ['# 三个训练种子的汇总结果', '']
    for row in completed:
        tag = f"{row['dataset']}_{row['backbone']}_{row['protocol']}"
        title = f"{row['dataset'].upper()} · {backbone_label(row['backbone'])} · {'背景增强训练' if row['protocol']=='paired' else '原图训练'}"
        agg = row['aggregate']
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
        for col, view, label in [(0, 'original', '原图'), (1, 'changed', '换背景')]:
            names = list(agg)
            bars(axes[0, col], names, [agg[n][view]['accuracy'] for n in names], 100,
                 label + '分类准确率', '准确率（%）')
            names = ['LF-CBM', 'LF-Pi']
            bars(axes[1, col], names, [agg[n][view]['mAP'] for n in names], 100,
                 label + '概念 mAP', 'mAP（%）')
            axes[0, col].set_ylim(0, 105); axes[1, col].set_ylim(0, 105)
        fig.suptitle(title + ' · 3 个种子 · 均值与标准差')
        name = f'seeds_{tag}.png'; fig.savefig(out / name, dpi=170); plt.close(fig)
        lines += [f'## {title}', '', f'![三种子分类与概念结果]({name})', '']
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
        names = ['LF-CBM', 'LF-Pi']
        for ax, view, title_part in zip(axes[:2], ['original', 'changed'], ['原图概念相关系数', '换背景概念相关系数']):
            bars(ax, names, [agg[n][view]['pearson_macro'] for n in names], 1, title_part, 'Pearson r')
            ax.set_ylim(-.1, 1.05)
        bars(axes[2], names, [agg[n]['standardized_concept_shift'] for n in names], 1,
             '换背景后的概念偏移', '训练标准差为单位的平均偏移')
        axes[2].set_ylim(bottom=0)
        fig.suptitle(title + f" · 相同概念 {row['shared_concepts']} 个 · 相同测试图 {row['test_subjects']} 张")
        name = f'seeds_shift_{tag}.png'; fig.savefig(out / name, dpi=170); plt.close(fig)
        lines += [f'![三种子相关性与偏移]({name})', '']
    write_metrics(out / 'seed_summary.json', completed)
    (out / 'SEED_RESULT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
