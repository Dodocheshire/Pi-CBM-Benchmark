"""Compact native-dataset overview from the audited result summaries."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .report_generalization import COLORS

METHODS = ['LF-CBM', 'LF-Pi', 'PiNI 均值', 'PiNI 均值＋噪声']
DATASETS = {'rival10': 'RIVAL10', 'cub': 'CUB', 'waterbirds': 'CUB–Places 水鸟'}
BACKBONES = {'resnet18': 'ResNet18', 'resnet50': 'ResNet50', 'clip_RN50': 'CLIP RN50'}


def heatmap(ax, values, methods, labels, title, *, correlation=False):
    values = np.asarray(values)
    assert np.isfinite(values).all()
    image = ax.imshow(values, vmin=-1 if correlation else 0, vmax=1 if correlation else 100,
                      cmap='RdBu_r' if correlation else 'YlGnBu', aspect='auto')
    ax.set_xticks(range(len(methods)), [m.replace('PiNI ', 'PiNI\n') for m in methods], fontsize=10)
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=8)
    for tick, method in zip(ax.get_xticklabels(), methods):
        tick.set_color(COLORS[method]); tick.set_fontweight('bold')
    ax.set_title(title, pad=52, fontsize=14)
    for (i, j), value in np.ndenumerate(values):
        rgba = image.cmap(image.norm(value))
        luminance = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
        ax.text(j, i, f'{value:.3f}' if correlation else f'{value:.2f}',
                ha='center', va='center', fontsize=11,
                color='white' if luminance < .5 else '#15202b')
    for boundary in (2.5, 5.5):
        ax.axhline(boundary, color='white', linewidth=3)
    ax.spines[:].set_visible(False)
    return image


def main():
    root = Path('reports')
    files = ['matched_summary.json', 'nec_summary.json', 'nec_noncbm_references.json']
    inputs = {name: json.loads((root/name).read_text()) for name in files}
    font = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    rows = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            matched = [r for r in inputs['matched_summary.json']
                       if r['dataset'] == dataset and r['backbone'] == backbone
                       and r['protocol'] == '背景增强训练' and r['training_seed'] == 42]
            assert len(matched) == 1
            matched = matched[0]
            group = f'artifacts/heads/{dataset}/{backbone}/paired/seed42'
            heads = {r['method']: r for r in inputs['nec_summary.json'] if r['group'] == group}
            controls = [r['references'] for r in inputs['nec_noncbm_references.json'] if r['group'] == group]
            assert len(controls) == 1 and all(m in heads for m in METHODS)
            row = {'dataset': dataset, 'backbone': backbone, 'head_group': group,
                   'shared_concepts': matched['concept_count'], 'dense_accuracy': {}, 'changed_concepts': {}}
            for view in ('original', 'changed'):
                row['dense_accuracy'][view] = {}
                for method in METHODS:
                    points = [p for p in heads[method]['test_evaluations']
                              if p['head'] == 'dense' and p['view'] == view]
                    assert len(points) == 1
                    row['dense_accuracy'][view][method] = points[0]['accuracy']
                references = controls[0][view]
                assert sorted(r['seed'] for r in references) == [42, 43, 44]
                row['dense_accuracy'][view]['非 CBM'] = float(np.mean([r['accuracy'] for r in references]))
            for method in METHODS:
                values = matched['methods'][method]['changed']['concepts']
                row['changed_concepts'][method] = {k: values[k] for k in ('mAP', 'pearson_macro')}
                assert all(np.isfinite(v) for v in row['changed_concepts'][method].values())
            rows.append(row)
    labels = [f"{DATASETS[r['dataset']]} · {BACKBONES[r['backbone']]}" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.8), layout='constrained')
    methods = METHODS + ['非 CBM']
    for ax, view, title in zip(axes, ('original', 'changed'), ('原图分类准确率（%）', '换背景分类准确率（%）')):
        image = heatmap(ax, [[100*r['dense_accuracy'][view][m] for m in methods] for r in rows],
                        methods, labels, title)
    fig.colorbar(image, ax=axes, shrink=.7, label='准确率（%）')
    fig.suptitle('统一稠密分类头 · 背景增强训练\n模型种子 42；非 CBM 为 3 个分类头种子的均值', fontsize=16)
    fig.savefig(root/'overview_dense_accuracy.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.8), layout='constrained')
    labels = [label+f" · K={r['shared_concepts']}" for label, r in zip(labels, rows)]
    for ax, key, title in zip(axes, ('mAP', 'pearson_macro'), ('概念 mAP（%）', '概念 Pearson 相关系数')):
        image = heatmap(ax, [[(100 if key == 'mAP' else 1)*r['changed_concepts'][m][key]
                              for m in METHODS] for r in rows], METHODS, labels, title,
                        correlation=key == 'pearson_macro')
        fig.colorbar(image, ax=ax, shrink=.7)
    fig.suptitle('换背景后的原生概念预测 · 背景增强训练 · 种子 42\n每行使用相同概念列与已知标签掩码；K 为该行共同概念数', fontsize=16)
    fig.savefig(root/'overview_native_concepts.png', dpi=170); plt.close(fig)
    result = {'training_protocol': 'paired', 'model_seed': 42, 'noncbm_head_seeds': [42, 43, 44],
              'classification_head': 'validation-selected dense head on frozen concept outputs',
              'concept_evaluation': 'native labels, matched columns and known masks per row',
              'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
              'rows': rows}
    (root/'overview_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'native_conditions': len(rows), 'figures': 2}))


if __name__ == '__main__':
    main()
