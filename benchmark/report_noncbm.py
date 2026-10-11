"""Summarize completed class-only controls; regenerate PNGs from saved metrics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from .report_utils import backbone_label


def main():
    root = Path('artifacts/noncbm'); out = Path('reports'); out.mkdir(exist_ok=True)
    font_path = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        font_name = 'sans-serif'
    plt.rcParams.update({'font.family': font_name, 'axes.unicode_minus': False})
    rows = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir(): continue
        reports = [json.loads(p.read_text()) for p in sorted(folder.glob('seed*/results.json'))]
        reports = [r for r in reports if Path(r['input_paths']['test']).resolve().is_relative_to(
            Path('artifacts/common').resolve())]
        if not reports: continue
        row = {'experiment': folder.name, 'training_seeds': [r['seed'] for r in reports],
               'train_views': reports[0].get('train_views', ['original']),
               'fit_images': reports[0]['fit_images'], 'feature_dimension': reports[0]['feature_dimension']}
        for view, key in [('original', 'test'), ('changed', 'changed'), ('gaussian_noise_s1', 'gaussian_noise_s1')]:
            if all(key in r for r in reports):
                a = np.array([r[key]['accuracy'] for r in reports])
                row[view] = {'mean_accuracy': float(a.mean()),
                             'seed_std': float(a.std(ddof=1)) if len(a)>1 else None,
                             'examples': reports[0][key]['examples']}
        rows.append(row)
    (out/'noncbm_summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    paired = [r for r in rows if 'changed' in r and '_rn18_imagenet' in r['experiment']]
    fig, ax = plt.subplots(figsize=(13, 4.8), layout='constrained')
    x = np.arange(len(paired))
    for view, offset, color, label in [('original', -.18, '#2878B5', '原图测试'),
                                      ('changed', .18, '#F28E2B', '换背景测试')]:
        values = [100*r[view]['mean_accuracy'] for r in paired]
        errors = [100*(r[view]['seed_std'] or 0) for r in paired]
        bars = ax.bar(x+offset, values, .34, yerr=errors, capsize=3, color=color, label=label)
        ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=10)
    ax.set_xticks(x, [r['experiment'].replace('_rn18_imagenet_paired','\n原图＋换背景训练')
                     .replace('_rn18_imagenet','\n原图训练').replace('rival10','RIVAL10')
                     .replace('cub','CUB') for r in paired])
    ax.set_ylabel('分类准确率（%）'); ax.set_ylim(0, 106)
    ax.set_title('非 CBM 对照：冻结 ImageNet ResNet18，直接拟合分类头')
    ax.legend(loc='upper left'); ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'noncbm_background.png', dpi=180); plt.close(fig)
    lines = ['# 非 CBM 对照', '', '使用冻结骨干输出直接训练线性分类头。四组固定学习率／权重衰减组合按验证集 NLL 选择，三个训练种子为 42、43、44。', '',
             '![原图与换背景准确率](noncbm_background.png)', '',
             '| 数据与骨干 | 训练视图 | 原图准确率 | 换背景准确率 | 高斯噪声准确率 |',
             '| --- | --- | --- | --- | --- |']
    for r in rows:
        def fmt(view):
            if view not in r:return '—'
            metric=r[view]
            suffix=f" ± {100*metric['seed_std']:.2f}" if metric['seed_std'] is not None else '（1 个种子）'
            return f"{100*metric['mean_accuracy']:.2f}%"+suffix
        dataset=r['experiment'].split('_')[0]
        labels={'cifar10':'CIFAR-10','cifar100':'CIFAR-100','cub':'CUB','rival10':'RIVAL10','waterbirds':'CUB–Places 水鸟二分类',
                'imagenet':'ImageNet 1000 类','places365':'Places365'}
        backbone='CLIP RN50' if 'clip' in r['experiment'].lower() else ('ResNet50' if '_rn50_' in r['experiment'] else 'ResNet18')
        if '_rn18_task_pretrained_train_val_selected' in r['experiment']:
            backbone=backbone_label('resnet18_rival10')
        elif '_rn18_task_pretrained_cub_train_val_selected' in r['experiment']:
            backbone=backbone_label('resnet18_cub_trainonly')
        views=('原图＋高斯噪声' if 'gaussian_noise_s1' in r['train_views'] else '原图＋换背景') if len(r['train_views'])>1 else '原图'
        lines.append(f"| {labels.get(dataset,dataset)}／{backbone} | {views} | {fmt('original')} | {fmt('changed')} | {fmt('gaussian_noise_s1')} |")
    lines += ['', '各组使用本轮共用的冻结骨干特征。误差为独立分类头训练种子间的标准差。', '']
    (out/'NONCBM_RESULT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
