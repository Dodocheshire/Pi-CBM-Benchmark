"""Aggregate completed three-seed NEC fits at matched requested budgets."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import torch

from .metrics import classification_metrics, nec
from .report_seeds import SEEDS, COLORS, bars, read, summarize
from .report_utils import backbone_cache_folder, backbone_label, write_metrics

ROOT = Path(__file__).resolve().parents[1]
ARMS = {'LF-CBM': 'lfcbm', 'LF-Pi': 'lfpi_global_1draw'}


def collect(group):
    dataset, backbone, protocol = group['dataset'], group['backbone'], group['protocol']
    parent = ROOT / 'artifacts/heads' / dataset / backbone / protocol
    directories = {method: [parent / f'seed{s}' / arm for s in SEEDS]
                   for method, arm in ARMS.items()}
    for dirs in directories.values():
        if not all((d / 'report.json').is_file() and (d / 'heads.pt').is_file() for d in dirs):
            return None
    cache = ROOT / 'artifacts/common' / dataset / backbone_cache_folder(backbone)
    manifest = json.loads((cache / 'manifest.json').read_text())
    reference = read(cache / 'test_original.npz', ['ids', 'labels'])
    eligible = Path(manifest['cache_root']) / 'background_eligible.npy'
    mask = np.load(eligible)[reference['ids']].astype(bool) if eligible.exists() else np.ones(len(reference['ids']), bool)
    assert int(mask.sum()) == group['test_subjects']
    methods = {}
    for method, dirs in directories.items():
        reports = [json.loads((d / 'report.json').read_text()) for d in dirs]
        budgets = set(str(b['target_nec']) for b in reports[0]['budgets'])
        for seed, directory, r, source in zip(SEEDS, dirs, reports, group['per_seed']):
            assert r['seed'] == seed and r['dataset'] == dataset
            assert r['selection_split'] == 'val' and r['test_selection'] is False
            assert r['train_views'] == (['original', 'changed'] if protocol == 'paired' else ['original'])
            assert Path(r['source']).resolve() == Path(source['methods'][method]['source']).resolve()
            budgets &= set(str(b['target_nec']) for b in r['budgets'])
        budgets = sorted(budgets, key=float)
        if not budgets:
            raise ValueError('No shared requested NEC budgets across registered seeds')
        per_seed = []
        for seed, directory, r in zip(SEEDS, dirs, reports):
            state = torch.load(directory / 'heads.pt', map_location='cpu', weights_only=True)
            heads = {'best_val': state['sparse']['best'], 'dense': state['dense']['head'],
                     **state['sparse']['budgets']}
            evaluations = {(p['head'], p['view']): p for p in r['test_evaluations']}
            points = []
            for head in [*budgets, 'best_val', 'dense']:
                actual = nec(heads[head]['weight'].numpy())
                for view in ['original', 'changed']:
                    recorded = evaluations[head, view]
                    assert actual['nec'] == recorded['nec']
                    assert actual['nonzero_connections'] == recorded['nonzero_connections']
                    assert actual['per_class'] == recorded['per_class']
                    data = read(directory / f'{head}_{view}.npz', ['ids', 'labels', 'logits'])
                    assert np.array_equal(data['ids'], reference['ids'])
                    assert np.array_equal(data['labels'], reference['labels'])
                    accuracy = classification_metrics(data['logits'][mask], data['labels'][mask])['accuracy']
                    points.append({'head': head, 'view': view, 'requested_nec': float(head) if head in budgets else None,
                                   'actual_nec': actual['nec'], 'per_class_connections': actual['per_class'],
                                   'accuracy': accuracy})
            per_seed.append({'seed': seed, 'directory': str(directory), 'concept_count': r['concept_count'],
                             'unconverged_path_points': r['unconverged_path_points'], 'evaluations': points})
        points = []
        for head in [*budgets, 'best_val', 'dense']:
            for view in ['original', 'changed']:
                values = [next(p for p in r['evaluations'] if p['head'] == head and p['view'] == view)
                          for r in per_seed]
                points.append({'head': head, 'view': view, 'requested_nec': float(head) if head in budgets else None,
                               'actual_nec': summarize([p['actual_nec'] for p in values]),
                               'accuracy': summarize([p['accuracy'] for p in values])})
        methods[method] = {'per_seed': per_seed, 'aggregate': points}
    return {'dataset': dataset, 'backbone': backbone, 'protocol': protocol, 'seeds': SEEDS,
            'test_subjects': int(mask.sum()), 'methods': methods,
            'noncbm': group['aggregate'].get('非 CBM'),
            'aggregation': 'same requested budgets; x = mean actual NEC; no interpolation; errors = sample std ddof1',
            'head_selection': 'validation only; original frozen head also applied to background-changed test'}


def main():
    source = ROOT / 'reports/seed_summary.json'
    if not source.exists():
        return
    font = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    groups = json.loads(source.read_text())
    completed = [r for group in groups if (r := collect(group)) is not None]
    print(json.dumps({'complete_nec_groups': len(completed), 'waiting_nec_groups': len(groups)-len(completed)}), flush=True)
    if not completed:
        return
    lines = ['# 三个训练种子的 NEC 曲线', '']
    out = ROOT / 'reports'
    for row in completed:
        tag = f"{row['dataset']}_{row['backbone']}_{row['protocol']}"
        title = f"{row['dataset'].upper()} · {backbone_label(row['backbone'])} · {'背景增强训练' if row['protocol']=='paired' else '原图训练'}"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
        for ax, view, label in zip(axes, ['original', 'changed'], ['原图测试', '换背景测试']):
            for method, data in row['methods'].items():
                points = [p for p in data['aggregate'] if p['view'] == view and p['requested_nec'] is not None]
                points.sort(key=lambda p: p['actual_nec']['mean'])
                ax.errorbar([p['actual_nec']['mean'] for p in points], [100*p['accuracy']['mean'] for p in points],
                            xerr=[p['actual_nec']['std'] for p in points], yerr=[100*p['accuracy']['std'] for p in points],
                            fmt='o-', color=COLORS[method], capsize=3, label=method)
            if row['noncbm']:
                reference = row['noncbm'][view]['accuracy']
                mean, std = 100*reference['mean'], 100*reference['std']
                ax.axhline(mean, color=COLORS['非 CBM'], linestyle='--', label='非 CBM（3 个头种子）')
                ax.axhspan(mean-std, mean+std, color=COLORS['非 CBM'], alpha=.12)
            ax.set_xlabel('每类有效概念连接数（实际 NEC 的均值）')
            ax.set_ylabel('分类准确率（%）'); ax.set_title(label); ax.set_ylim(0, 102)
            ax.spines[['top', 'right']].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=3)
        fig.suptitle(title + ' · 3 个种子 · 均值与标准差')
        name = f'nec_seeds_{tag}.png'; fig.savefig(out/name, dpi=170); plt.close(fig)
        lines += [f'## {title}', '', f'![多种子实际 NEC 曲线]({name})', '']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
        for ax, view, label in zip(axes, ['original', 'changed'], ['原图测试', '换背景测试']):
            names, values = [], []
            for method, data in row['methods'].items():
                names.append(method)
                values.append(next(p for p in data['aggregate'] if p['head']=='dense' and p['view']==view)['accuracy'])
            if row['noncbm']:
                names.append('非 CBM'); values.append(row['noncbm'][view]['accuracy'])
            bars(ax, names, values, 100, label + ' · 稠密分类头', '分类准确率（%）')
            ax.set_ylim(0, 105)
        fig.suptitle(title + ' · 充分概念连接 · 3 个种子')
        name = f'nec_seeds_dense_{tag}.png'; fig.savefig(out/name, dpi=170); plt.close(fig)
        lines += [f'![多种子稠密头分类结果]({name})', '']
    write_metrics(out/'nec_seed_summary.json', completed)
    (out/'NEC_SEED_RESULT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
