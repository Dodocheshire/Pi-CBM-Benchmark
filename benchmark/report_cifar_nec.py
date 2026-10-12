"""Plot fixed sparse/dense heads on complete official CIFAR-C exports."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

from .metrics import classification_metrics, nec
from .report_generalization import NAMES, COLORS
from .report_utils import backbone_label, backbone_cache_folder, write_metrics

KINDS = {'gaussian_noise': '高斯噪声', 'gaussian_blur': '高斯模糊',
         'contrast': '降低对比度', 'jpeg_compression': 'JPEG 压缩'}


def controls(dataset, bone, protocol):
    bone = backbone_cache_folder(bone)
    suffix = '_gaussian_noise_augmented' if protocol == 'corruption_augmented' else ''
    base = Path('artifacts/noncbm')/f'{dataset}_{bone}{suffix}'
    records = []
    with np.load(Path('artifacts/common')/dataset/bone/'test_original.npz') as original:
        ids, labels = original['ids'], original['labels']
    expected = {f'{kind}_{severity}' for kind in KINDS for severity in (1, 3, 5)}
    for path in sorted(base.glob('seed*/shifts/report.json')):
        report = json.loads(path.read_text())
        if report['settings'] != 12:
            continue
        assert not report['refitting'] and report['nec'] is None
        assert {r['setting'] for r in report['evaluation']} == expected
        assert hashlib.sha256((path.parents[1]/'best.pt').read_bytes()).hexdigest() == report['head_sha256']
        for row in report['evaluation']:
            with np.load(path.parent/f"{row['setting']}.npz") as saved:
                assert np.array_equal(saved['ids'], ids) and np.array_equal(saved['labels'], labels)
                assert classification_metrics(saved['logits'], labels)['accuracy'] == row['accuracy']
        records.append({'seed': report['seed'], 'source_report': str(path),
                        'head_sha256': report['head_sha256'], 'records': report['evaluation']})
    return records


def collect():
    groups = {}
    expected = {f'{kind}_{severity}' for kind in KINDS for severity in (1, 3, 5)}
    for path in sorted(Path('artifacts/heads').glob('cifar*/**/shifts/report.json')):
        report = json.loads(path.read_text())
        if report['settings'] != 12:
            continue
        directory = path.parents[1]
        dataset, bone, protocol, seed, method = directory.relative_to(Path('artifacts/heads')).parts
        if seed != 'seed42' or method not in NAMES:
            continue
        if method == 'lfpi_global' and (directory.parent/'lfpi_global_1draw/shifts/report.json').exists():
            continue
        assert not report['refitting'] and report['dataset'] == dataset
        assert {r['setting'] for r in report['evaluation']} == expected
        weights = directory/'heads.pt'
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == report['heads_sha256']
        state = torch.load(weights, map_location='cpu', weights_only=True)
        heads = {**state['sparse']['budgets'], 'best_val': state['sparse']['best'],
                 'dense': state['dense']['head']}
        with np.load(directory/'dense_original.npz') as original:
            ids, labels = original['ids'], original['labels']
        assert len(ids) == 10000 and len(np.unique(ids)) == 10000
        assert len(report['evaluation']) == len(expected)*len(heads)
        for row in report['evaluation']:
            assert row['examples'] == 10000
            connections = nec(heads[row['head']]['weight'].numpy())
            for key in ('nec', 'nonzero_connections', 'per_class'):
                assert connections[key] == row[key]
            with np.load(directory/'shifts'/f"{row['setting']}_{row['head']}.npz") as saved:
                assert np.array_equal(saved['ids'], ids) and np.array_equal(saved['labels'], labels)
                assert classification_metrics(saved['logits'], labels)['accuracy'] == row['accuracy']
        groups.setdefault((dataset, bone, protocol), []).append({
            'method': method, 'source_report': str(path), 'heads_sha256': report['heads_sha256'],
            'refitting': False, 'images_per_setting': 10000, 'records': report['evaluation']})
    return groups


def main():
    torch.set_num_threads(3)
    font = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    groups = collect(); lines = ['# 官方 CIFAR-C：固定分类头下的实际 NEC 曲线', '']
    summary = []
    for (dataset, bone, protocol), methods in groups.items():
        references = controls(dataset, bone, protocol)
        title = f'{dataset.upper()} · {backbone_label(bone)} · ' + (
            '高斯噪声配对训练' if protocol == 'corruption_augmented' else '原图训练')
        lines += [f'## {title}', '']
        for severity in (1, 3, 5):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
            for ax, (kind, label) in zip(axes.flat, KINDS.items()):
                setting = f'{kind}_{severity}'
                for method in methods:
                    name = NAMES[method['method']]
                    selected = [r for r in method['records'] if r['setting'] == setting]
                    points = sorted((r['nec'], 100*r['accuracy']) for r in selected
                                    if r['head'] not in ('dense', 'best_val'))
                    ax.plot(*np.asarray(points).T, 'o-', color=COLORS[name], label=name, markersize=4)
                    dense = next(r for r in selected if r['head'] == 'dense')
                    ax.scatter(dense['nec'], 100*dense['accuracy'], marker='*', s=100,
                               color=COLORS[name], label=f'{name} 稠密头')
                if references:
                    values = [100*next(r['accuracy'] for r in reference['records']
                                      if r['setting'] == setting) for reference in references]
                    mean = float(np.mean(values))
                    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0
                    ax.axhline(mean, color=COLORS['非 CBM'], linestyle='--',
                               label=f'非 CBM（{len(values)} 个头种子）')
                    ax.axhspan(mean-std, mean+std, color=COLORS['非 CBM'], alpha=.12)
                ax.set_title(label); ax.set_ylim(0, 102)
                ax.set_xlabel('每类有效概念连接数（实际 NEC）'); ax.set_ylabel('分类准确率（%）')
                ax.grid(alpha=.18); ax.spines[['top', 'right']].set_visible(False)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center', ncol=3)
            fig.suptitle(title + f'\n严重度 {severity} · 每设置全部 10000 张测试图')
            filename = f'cifar_nec_{dataset}_{bone}_{protocol}_severity{severity}.png'
            fig.savefig(Path('reports')/filename, dpi=170); plt.close(fig)
            lines += [f'### 严重度 {severity}', '', f'![实际 NEC 与分类准确率]({filename})', '']
        summary.append({'dataset': dataset, 'backbone': bone, 'protocol': protocol,
                        'methods': methods, 'noncbm': references})
    write_metrics('reports/cifar_nec_summary.json', summary)
    Path('reports/CIFAR_NEC_RESULT.md').write_text('\n'.join(lines))
    print(json.dumps({'completed_groups': len(groups), 'verified_head_evaluations': sum(
        len(m['records']) for methods in groups.values() for m in methods)}))


if __name__ == '__main__':
    main()
