"""Evaluate frozen LF sparse/dense heads on aligned native RGB corruptions."""
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
from .report_ood_concepts import KINDS, DATASETS, read
from .report_utils import write_metrics


def evaluate(dataset):
    directory = Path('artifacts/ood') / dataset / 'resnet18/paired/seed42'
    protocol = json.loads((directory / 'protocol.json').read_text())
    source = Path('artifacts/matched/lf') / dataset / 'rn18_imagenet/paired_augmentation/seed42'
    selected = json.loads((source / 'report.json').read_text())['selected_indices']
    assert selected == protocol['concept_indices']
    settings = ['original_0'] + [f'{kind}_{severity}' for kind, _ in KINDS for severity in (1, 3, 5)]
    records, fingerprints = [], {}
    for method, arm in [('lfcbm', 'lfcbm'), ('lfpi_single', 'lfpi_global_1draw')]:
        head_directory = Path('artifacts/heads') / dataset / 'resnet18/paired/seed42' / arm
        report = json.loads((head_directory / 'report.json').read_text())
        assert report['selection_split'] == 'val' and not report['test_selection']
        assert report['concept_count'] == len(selected) and report['concept_field'] == 'concepts'
        assert report['train_views'] == ['original', 'changed'] and report['seed'] == 42
        expected_source = source if method == 'lfcbm' else source / 'single_draw'
        assert Path(report['source']).resolve() == expected_source.resolve()
        path = head_directory / 'heads.pt'
        fingerprints[method] = hashlib.sha256(path.read_bytes()).hexdigest()
        state = torch.load(path, map_location='cpu', weights_only=True)
        normalization = read(head_directory / 'normalization.npz')
        reference = read(head_directory / 'dense_original.npz')
        positions = {int(identifier): index for index, identifier in enumerate(reference['ids'])}
        assert len(positions) == len(reference['ids'])
        heads = {**state['sparse']['budgets'], 'best_val': state['sparse']['best'], 'dense': state['dense']['head']}
        for setting in settings:
            data = read(directory / f'{setting}_{method}.npz')
            assert np.array_equal(data['ids'], np.asarray(protocol['sample_ids']))
            assert len(np.unique(data['ids'])) == 256
            rows = [positions[int(identifier)] for identifier in data['ids']]
            assert np.array_equal(data['labels'], reference['labels'][rows])
            assert data['concepts'].shape == (256, len(selected)) and np.isfinite(data['concepts']).all()
            for name, head in heads.items():
                if name == 'dense':
                    mean, scale = state['dense']['mean'].numpy(), state['dense']['std'].numpy()
                else:
                    mean, scale = normalization['mean'], normalization['scale']
                inputs = torch.as_tensor((data['concepts'] - mean) / scale).float()
                logits = torch.nn.functional.linear(inputs, head['weight'], head['bias']).numpy()
                metrics = classification_metrics(logits, data['labels'])
                connections = nec(head['weight'].numpy())
                recorded = next(p for p in report['test_evaluations'] if p['head'] == name and p['view'] == 'original')
                assert connections['nec'] == recorded['nec']
                assert connections['nonzero_connections'] == recorded['nonzero_connections']
                assert connections['per_class'] == recorded['per_class']
                records.append({'method': method, 'setting': setting, 'head': name,
                                'actual_nec': connections['nec'], 'nonzero_connections': connections['nonzero_connections'],
                                'accuracy': metrics['accuracy'], 'balanced_accuracy': metrics['balanced_accuracy'],
                                'images': metrics['examples']})
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fingerprints[method]
    controls = {}
    for setting in settings:
        data = read(directory / f'{setting}_noncbm.npz')
        assert np.array_equal(data['ids'], np.asarray(protocol['sample_ids']))
        rows = [positions[int(identifier)] for identifier in data['ids']]
        assert np.array_equal(data['labels'], reference['labels'][rows])
        controls[setting] = classification_metrics(data['logits'], data['labels'])
    return {'dataset': dataset, 'images': 256, 'concept_indices': selected, 'head_fingerprints': fingerprints,
            'refitting': False, 'training_seed': 42, 'inference_draws': 1,
            'protocol_source': str(directory / 'protocol.json'), 'records': records, 'noncbm': controls}


def main():
    torch.set_num_threads(3)
    font = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    results = [evaluate(dataset) for dataset in DATASETS]
    lines = ['# LF-CBM／LF-Pi 图像扰动下的实际 NEC 曲线', '']
    for result in results:
        lines += [f'## {DATASETS[result["dataset"]]} · ImageNet ResNet18 · 背景增强训练', '']
        for severity in (1, 3, 5):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
            for ax, (kind, title) in zip(axes.flat, KINDS):
                setting = f'{kind}_{severity}'
                for method in ('lfcbm', 'lfpi_single'):
                    points = sorted((r['actual_nec'], 100*r['accuracy']) for r in result['records']
                                    if r['method'] == method and r['setting'] == setting and r['head'] not in ('dense', 'best_val'))
                    name = NAMES[method]
                    ax.plot(*np.asarray(points).T, 'o-', color=COLORS[name], label=name, markersize=4)
                    dense = next(r for r in result['records'] if r['method'] == method and r['setting'] == setting and r['head'] == 'dense')
                    ax.scatter(dense['actual_nec'], 100*dense['accuracy'], marker='*', s=100,
                               color=COLORS[name], label=f'{name} 稠密头')
                ax.axhline(100*result['noncbm'][setting]['accuracy'], color=COLORS['非 CBM'], linestyle='--', label='非 CBM')
                ax.set_title(title); ax.set_xlabel('每类有效概念连接数（实际 NEC）')
                ax.set_ylabel('分类准确率（%）'); ax.set_ylim(0, 102)
                ax.grid(alpha=.18); ax.spines[['top', 'right']].set_visible(False)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center', ncol=3)
            fig.suptitle(f'{DATASETS[result["dataset"]]} · 同一批 256 张测试图 · 严重度 {severity}\n固定分类头 · 单次噪声采样')
            filename = f'ood_nec_{result["dataset"]}_severity{severity}.png'
            fig.savefig(Path('reports') / filename, dpi=180); plt.close(fig)
            lines += [f'### 严重度 {severity}', '', f'![实际 NEC 与分类准确率]({filename})', '']
    write_metrics('reports/ood_nec_summary.json', results)
    Path('reports/OOD_NEC_RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'datasets': len(results), 'frozen_head_evaluations': sum(len(r['records']) for r in results)}))


if __name__ == '__main__':
    main()
