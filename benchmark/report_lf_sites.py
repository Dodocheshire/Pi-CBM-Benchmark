"""Compare completed LF injection sites on the same native subjects/concepts."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .report_matched import evaluate, read
from .metrics import classification_metrics
from .report_utils import backbone_cache_folder, backbone_label, write_metrics

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'LF-CBM': '#2878B5', 'LF-Pi 全局层': '#E15759',
          'LF-Pi 图像层': '#9C755F', 'LF-Pi 内部层': '#76B7B2', '非 CBM': '#F28E2B'}


def collect():
    groups = {}
    for path in sorted((ROOT / 'artifacts/matched/lf_sites').rglob('report.json')):
        report = json.loads(path.read_text())
        site = report['site']; prefix = 'lfpi_image_' if site == 'target_image' else 'lfpi_internal_'
        directory = path.parent / 'single_draw'
        required = [directory / f'{prefix}{split}_{view}.npz'
                    for split in ['train', 'val', 'test'] for view in ['original', 'changed']]
        if not (directory / 'metadata.json').exists() or not all(p.exists() for p in required):
            continue
        metadata = json.loads((directory / 'metadata.json').read_text())
        assert metadata['inference_draws'] == 1 and not report['test_selection']
        assert metadata['checkpoint_sha256'] == hashlib.sha256((path.parent / 'models.pt').read_bytes()).hexdigest()
        reference = Path(report['reused_baseline']).parent
        assert report['reused_baseline_sha256'] == hashlib.sha256((reference / 'models.pt').read_bytes()).hexdigest()
        original = json.loads((reference / 'report.json').read_text())
        assert report['selected_indices'] == original['selected_indices']
        assert report['feature_provenance'] == original['feature_provenance']
        assert report['training_seed'] == original['training_seed'] and report['train_views'] == original['train_views']
        dataset = report['dataset']; backbone = report['feature_provenance'].get('backbone_name', 'resnet18')
        folder = backbone_cache_folder(backbone)
        common = ROOT / 'artifacts/common' / dataset / folder
        columns = report['selected_indices']
        for split in ['train', 'val', 'test']:
            for view in ['original', 'changed']:
                data = read(directory / f'{prefix}{split}_{view}.npz')
                native = read(common / f'{split}_{view}.npz')
                for key in ['ids', 'labels']:
                    assert np.array_equal(data[key], native[key])
                for key in ['targets', 'known']:
                    assert np.array_equal(data[key], native[key][:, columns])
                assert data['concepts'].shape == (len(native['ids']), len(columns)) and np.isfinite(data['concepts']).all()
                assert np.isfinite(data['logits']).all()
        test = read(common / 'test_original.npz')
        eligible = Path(report['feature_provenance']['cache_root']) / 'background_eligible.npy'
        mask = np.load(eligible)[test['ids']].astype(bool) if eligible.exists() else np.ones(len(test['ids']), bool)
        key = (dataset, backbone, tuple(report['train_views']), report['training_seed'])
        if key not in groups:
            groups[key] = {'dataset': dataset, 'backbone': backbone, 'train_views': report['train_views'],
                           'seed': report['training_seed'], 'shared_concepts': columns,
                           'test_subjects': int(mask.sum()), 'methods': {
                               'LF-CBM': evaluate(reference, 'lfcbm_', mask),
                               'LF-Pi 全局层': evaluate(reference / 'single_draw', 'lfpi_global_', mask)}}
            paired = len(report['train_views']) == 2
            control = ROOT / 'artifacts/noncbm' / f'{dataset}_{folder}{"_paired" if paired else ""}' / f'seed{report["training_seed"]}'
            if (control / 'changed_predictions.npz').exists():
                values = {}
                for view, filename in [('original', 'test_predictions.npz'), ('changed', 'changed_predictions.npz')]:
                    data = read(control / filename)
                    assert np.array_equal(data['ids'], test['ids']) and np.array_equal(data['labels'], test['labels'])
                    values[view] = {'classification': classification_metrics(data['logits'][mask], data['labels'][mask])}
                groups[key]['methods']['非 CBM'] = values
        name = 'LF-Pi 图像层' if site == 'target_image' else 'LF-Pi 内部层'
        groups[key]['methods'][name] = {**evaluate(directory, prefix, mask), 'source': str(directory)}
    return list(groups.values())


def main():
    rows = collect()
    if not rows:
        print('No completed site exports yet'); return
    font = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    lines = ['# LF-Pi 不同注入位置', '']
    for i, row in enumerate(rows):
        title = f'{row["dataset"].upper()} · {backbone_label(row["backbone"])} · ' + ('背景增强训练' if len(row['train_views']) == 2 else '原图训练')
        methods = row['methods']; names = [n for n in COLORS if n in methods]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
        for column, view, label in [(0, 'original', '原图'), (1, 'changed', '换背景')]:
            for axis, metric, selected_names, caption in [
                (axes[0, column], 'classification', names, '分类准确率'),
                (axes[1, column], 'concepts', [n for n in names if n != '非 CBM'], '概念 mAP')]:
                field = 'accuracy' if metric == 'classification' else 'mAP'
                values = [100*methods[name][view][metric][field] for name in selected_names]
                bars = axis.bar(selected_names, values, color=[COLORS[n] for n in selected_names])
                axis.bar_label(bars, fmt='%.1f', padding=3)
                axis.tick_params(axis='x', labelrotation=18)
                axis.set_ylim(0, 105); axis.set_ylabel('百分比（%）'); axis.set_title(label + caption)
                axis.spines[['top', 'right']].set_visible(False)
        fig.suptitle(title + f'\n同一批 {row["test_subjects"]} 张测试图 · 相同 {len(row["shared_concepts"])} 个概念 · 单次采样')
        filename = f'lf_sites_comparison_{i}.png'; fig.savefig(ROOT / 'reports' / filename, dpi=180); plt.close(fig)
        lines += [f'## {title}', '', f'![分类与概念识别]({filename})', '']
        conceptual = [name for name in names if name != '非 CBM']
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout='constrained')
        for axis, view, label in [(axes[0], 'original', '原图概念相关系数'), (axes[1], 'changed', '换背景概念相关系数')]:
            axis.bar(conceptual, [methods[n][view]['concepts']['pearson_macro'] for n in conceptual], color=[COLORS[n] for n in conceptual])
            axis.set_title(label); axis.set_ylim(-.1, 1.05)
        axes[2].bar(conceptual, [methods[n]['shift']['standardized_mae'] for n in conceptual], color=[COLORS[n] for n in conceptual])
        axes[2].set_title('换背景后的概念偏移'); axes[2].set_ylabel('训练标准差单位')
        for axis in axes:
            axis.tick_params(axis='x', labelrotation=20); axis.spines[['top', 'right']].set_visible(False)
        fig.suptitle(title)
        filename = f'lf_sites_shift_{i}.png'; fig.savefig(ROOT / 'reports' / filename, dpi=180); plt.close(fig)
        lines += [f'![相关性与概念偏移]({filename})', '']
    write_metrics('reports/lf_sites_summary.json', rows)
    (ROOT / 'reports/LF_SITE_RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Verified completed site groups: {len(rows)}')


if __name__ == '__main__':
    main()
