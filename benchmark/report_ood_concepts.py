"""Native concept quality and paired activation shifts on completed RGB tests."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .metrics import concept_metrics, paired_concept_metrics
from .report_generalization import NAMES, COLORS
from .report_utils import write_metrics


METHODS = ('lfcbm', 'lfpi_single', 'pini_mean', 'pini_noise')
KINDS = [('gaussian_noise', '高斯噪声'), ('gaussian_blur', '高斯模糊'),
         ('contrast', '降低对比度'), ('jpeg', 'JPEG 压缩')]
DATASETS = {'rival10': 'RIVAL10', 'cub': 'CUB', 'waterbirds': 'CUB–Places Waterbirds'}


def read(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def evaluate(dataset):
    directory = Path('artifacts/ood') / dataset / 'resnet18/paired/seed42'
    protocol = json.loads((directory / 'protocol.json').read_text())
    columns = np.asarray(protocol['concept_indices'], dtype=int)
    source = read(Path('artifacts/common') / dataset / 'rn18_imagenet/test_original.npz')
    positions = {int(identifier): index for index, identifier in enumerate(source['ids'])}
    ids = np.asarray(protocol['sample_ids'])
    if len(np.unique(ids)) != 256 or len(positions) != len(source['ids']):
        raise ValueError('Expected 256 unique subjects and unique source IDs')
    rows = np.asarray([positions[int(identifier)] for identifier in ids])
    expected = {'ids': ids, 'labels': source['labels'][rows],
                'targets': source['targets'][rows][:, columns],
                'known': source['known'][rows][:, columns]}
    lfroot = Path('artifacts/matched/lf') / dataset / 'rn18_imagenet/paired_augmentation/seed42'
    records = []
    for method in METHODS:
        if method == 'lfcbm':
            train_path = lfroot / 'lfcbm_train_original.npz'
        elif method == 'lfpi_single':
            train_path = lfroot / 'single_draw/lfpi_global_train_original.npz'
        else:
            arm = 'mean' if method == 'pini_mean' else 'noise'
            train_path = Path('artifacts/pini_matched') / dataset / 'exports' / arm / 'train_original.npz'
        train = read(train_path)
        train_concepts = train['concepts']
        if method.startswith('pini'):
            train_concepts = train_concepts[:, columns]
        common_train = read(Path('artifacts/common') / dataset / 'rn18_imagenet/train_original.npz')
        for key in ('ids', 'labels'):
            if not np.array_equal(train[key], common_train[key]):
                raise ValueError(f'Training reference mismatch: {train_path}: {key}')
        if train_concepts.shape[1] != len(columns) or not np.isfinite(train_concepts).all():
            raise ValueError(f'Invalid training concepts: {train_path}')
        clean = read(directory / f'original_0_{method}.npz')
        for setting in ['original_0'] + [f'{kind}_{severity}' for kind, _ in KINDS for severity in (1, 3, 5)]:
            path = directory / f'{setting}_{method}.npz'
            data = read(path)
            for key, value in expected.items():
                if not np.array_equal(data[key], value):
                    raise ValueError(f'Native subject/label mismatch: {path}: {key}')
            if data['concepts'].shape != expected['targets'].shape or not np.isfinite(data['concepts']).all():
                raise ValueError(f'Invalid test concepts: {path}')
            records.append({'method': method, 'setting': setting, 'source': str(path),
                            'train_reference': str(train_path),
                            'quality': concept_metrics(data['concepts'], expected['targets'], expected['known']),
                            'shift': paired_concept_metrics(clean['concepts'], data['concepts'], train_concepts)})
    return {'dataset': dataset, 'images': len(ids), 'concept_indices': columns.tolist(),
            'backbone': 'ImageNet ResNet18', 'training_views': protocol['training_views'],
            'training_seed': 42, 'noise_draws': 1, 'inference_noise_seed': protocol['inference_noise_seed'],
            'protocol_source': str(directory / 'protocol.json'), 'records': records}


def plot(result, metric, label, percent):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for ax, (kind, title) in zip(axes.flat, KINDS):
        for method in METHODS:
            lookup = {row['setting']: row for row in result['records'] if row['method'] == method}
            settings = ['original_0'] + [f'{kind}_{severity}' for severity in (1, 3, 5)]
            section, key = metric
            values = [lookup[setting][section][key] for setting in settings]
            if any(value is None for value in values):
                raise ValueError(f'Undefined metric: {result["dataset"]}/{method}/{metric}')
            name = NAMES[method]
            ax.plot([0, 1, 3, 5], np.asarray(values) * (100 if percent else 1),
                    'o-', color=COLORS[name], label=name, markersize=4)
        ax.set_title(title)
        ax.set_xticks([0, 1, 3, 5], ['原图', '1', '3', '5'])
        ax.set_xlabel('扰动严重度')
        ax.set_ylabel(label)
        if percent:
            ax.set_ylim(0, 100)
        elif metric[0] == 'shift':
            ax.set_ylim(bottom=0)
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4)
    fig.suptitle(f'{DATASETS[result["dataset"]]} · ImageNet ResNet18\n背景增强训练 · 同一批 256 张测试图 · 单次噪声采样')
    output = Path('reports') / f'ood_concepts_{result["dataset"]}_{metric[1]}.png'
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main():
    font = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    results = [evaluate(dataset) for dataset in DATASETS]
    lines = ['# 图像扰动下的概念预测与激活变化', '']
    for result in results:
        lines += [f'## {DATASETS[result["dataset"]]}', '']
        for metric, label, percent in [(('quality', 'mAP'), '概念 mAP（%）', True),
                                       (('quality', 'pearson_macro'), '概念相关系数', False),
                                       (('shift', 'standardized_mae'), '概念激活平均偏移（训练标准差单位）', False)]:
            path = plot(result, metric, label, percent)
            lines += [f'### {label}', '', f'![{label}]({path.name})', '']
    write_metrics('reports/ood_concepts_summary.json', results)
    Path('reports/OOD_CONCEPT_RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Verified {len(results)} datasets, {sum(len(r["records"]) for r in results)} aligned concept evaluations')


if __name__ == '__main__':
    main()
