"""Render matched conditions, shared concept columns and paired background tests."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .metrics import classification_metrics, concept_metrics, paired_concept_metrics
from .report_utils import write_metrics

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'LF-CBM':'#2878B5', 'LF-Pi':'#E15759', 'PiNI 均值':'#59A14F',
          'PiNI 均值＋噪声':'#B07AA1', '非 CBM':'#F28E2B'}


def read(path):
    with np.load(path) as z: return {k: np.array(z[k]) for k in z.files}


def evaluate(source, prefix, mask, columns=None):
    a, b, train = [read(source/f'{prefix}{split}_{view}.npz')
                   for split, view in [('test','original'), ('test','changed'), ('train','original')]]
    assert np.array_equal(a['ids'], b['ids']) and np.array_equal(a['labels'], b['labels'])
    assert len(mask)==len(a['ids'])
    result = {'test_subjects': int(mask.sum())}
    for view, data in [('original', a), ('changed', b)]:
        result[view] = {'classification': classification_metrics(data['logits'][mask], data['labels'][mask])}
        if 'concepts' in data:
            idx = np.arange(data['concepts'].shape[1]) if columns is None else columns
            result[view]['concepts'] = concept_metrics(data['concepts'][mask][:,idx],
                data['targets'][mask][:,idx], data['known'][mask][:,idx])
    if 'concepts' in a:
        result['shift'] = paired_concept_metrics(a['concepts'][mask][:,idx],
            b['concepts'][mask][:,idx], train['concepts'][:,idx])
    return result


def main():
    out = ROOT/'reports'; out.mkdir(exist_ok=True)
    font_path = Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    conditions = []
    for report_path in sorted((ROOT/'artifacts/matched/lf').rglob('report.json')):
        report = json.loads(report_path.read_text())
        if 'feature_provenance' not in report:
            continue
        if report.get('human_concept_annotations') is False:
            continue  # Continuous-teacher experiments have a separate report.
        dataset = report['dataset']; source = report_path.parent
        paired = report['train_views']==['original','changed']
        proto = '背景增强训练' if paired else '原图训练'
        backbone = report['feature_provenance'].get('backbone_name', 'resnet18')
        seed = report['training_seed']
        baseline = read(source/'lfcbm_test_original.npz')
        eligible_path = Path(report['feature_provenance']['cache_root'])/'background_eligible.npy'
        mask = np.load(eligible_path)[baseline['ids']].astype(bool) if eligible_path.exists() else np.ones(len(baseline['ids']),bool)
        row = {'dataset': dataset, 'backbone': backbone, 'training_seed': seed,
               'train_views': report['train_views'], 'protocol': proto,
               'concept_count': report['selected_concepts'], 'candidate_concepts': report['bank_concepts'],
               'shared_concept_indices': report['selected_indices'], 'methods': {}}
        for method, prefix in [('LF-CBM','lfcbm_'),('LF-Pi','lfpi_global_')]:
            single=source/'single_draw'
            complete=(single/'metadata.json').exists() and all((single/f'lfpi_global_{s}_{v}.npz').exists()
                for s in ['train','val','test'] for v in ['original','changed'])
            esource=single if method=='LF-Pi' and complete else source
            row['methods'][method] = evaluate(esource, prefix, mask)
            row['methods'][method]['inference_draws']=1 if method=='LF-CBM' or esource==single else 8
            row['methods'][method]['source']=str(esource)
        # Latest PiNI is intrinsically trained on paired views, with native targets.
        if paired and seed==42:
            for method, arm in [('PiNI 均值','mean'),('PiNI 均值＋噪声','noise')]:
                proot=ROOT/'artifacts/pini_matched'
                if backbone!='resnet18':proot=proot/backbone.lower()
                ps = proot/dataset/'exports'/arm
                if (ps/'results.json').exists():
                    row['methods'][method] = evaluate(ps, '', mask, report['selected_indices'])
                    row['methods'][method].update(inference_draws=1,source=str(ps))
        bfolder={'resnet18':'rn18_imagenet','resnet50':'rn50_imagenet','clip_RN50':'clip_RN50'}[backbone]
        noncbm = ROOT/'artifacts/noncbm'/f"{dataset}_{bfolder}{'_paired' if paired else ''}"/f'seed{seed}'
        if (noncbm/'changed_predictions.npz').exists():
            method = {}
            for view, file in [('original','test_predictions.npz'),('changed','changed_predictions.npz')]:
                a = read(noncbm/file); assert np.array_equal(a['ids'], baseline['ids'])
                method[view] = {'classification': classification_metrics(a['logits'][mask], a['labels'][mask])}
            row['methods']['非 CBM'] = method
        conditions.append(row)
    write_metrics(out/'matched_summary.json',conditions)
    lines = ['# 同条件评测结果', '', '所有图表按骨干、训练视图和训练种子分组。分类比较使用相同测试图像；概念比较使用该组 LF-CBM 保留的共同概念列。', '',
             'LF-CBM／LF-Pi 使用冻结 CLIP 概念相似度监督；PiNI 使用原方法的属性标签监督。CUB 评测使用逐图人工属性，已知标注按 certainty ≥ 3 筛选。', '']
    lines += ['本轮 Waterbirds 为完整 CUB 鸟类按陆鸟／水鸟分组、配合 Places 换背景的二分类任务。','']
    for i, row in enumerate(conditions):
        title = f"{row['dataset'].upper()} · {row['backbone']} · {row['protocol']} · 种子 {row['training_seed']}"
        names = list(row['methods']); x = np.arange(len(names)); fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for ax, view, label in zip(axes,['original','changed'],['原图测试','换背景测试']):
            values = [100*row['methods'][name][view]['classification']['accuracy'] for name in names]
            bars = ax.bar(x, values, color=[COLORS[n] for n in names],width=.65)
            ax.bar_label(bars,fmt='%.1f',padding=3); ax.set_xticks(x,names,rotation=15)
            ax.set_ylim(0,105);ax.set_ylabel('分类准确率（%）');ax.set_title(label)
            ax.spines[['top','right']].set_visible(False)
        fig.suptitle(title); file=f'matched_classification_{i}.png';fig.savefig(out/file,dpi=170);plt.close(fig)
        lines += [f'## {title}', '', f"共同概念 {row['concept_count']}／候选 {row['candidate_concepts']}。", '', f'![分类结果]({file})','']
        if row['methods']['LF-Pi']['inference_draws']!=1:
            lines += ['LF-Pi 为 8 次噪声采样平均；单次采样结果完成后更新主比较。','']
        cnames = [n for n in names if 'concepts' in row['methods'][n]['original']]
        fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for ax, view, label in zip(axes,['original','changed'],['原图概念 mAP','换背景概念 mAP']):
            values=[100*row['methods'][n][view]['concepts']['mAP'] for n in cnames]
            bars=ax.bar(np.arange(len(cnames)),values,color=[COLORS[n] for n in cnames],width=.65)
            ax.bar_label(bars,fmt='%.1f',padding=3);ax.set_xticks(np.arange(len(cnames)),cnames,rotation=15)
            ax.set_ylim(0,105);ax.set_ylabel('mAP（%）');ax.set_title(label);ax.spines[['top','right']].set_visible(False)
        fig.suptitle(title);file=f'matched_concepts_{i}.png';fig.savefig(out/file,dpi=170);plt.close(fig)
        lines += [f'![概念结果]({file})','']
        fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for view, offset, color in [('original',-.18,'#2878B5'),('changed',.18,'#F28E2B')]:
            values=[row['methods'][n][view]['concepts']['pearson_macro'] for n in cnames]
            bars=axes[0].bar(np.arange(len(cnames))+offset,values,.34,color=color,
                             label='原图' if view=='original' else '换背景')
            axes[0].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        axes[0].set_ylabel('Pearson r');axes[0].set_ylim(-.1,1.05);axes[0].legend(loc='upper left')
        axes[0].set_title('与逐图概念标签的相关性')
        values=[row['methods'][n]['shift']['standardized_mae'] for n in cnames]
        bars=axes[1].bar(np.arange(len(cnames)),values,color=[COLORS[n] for n in cnames],width=.65)
        axes[1].bar_label(bars,fmt='%.2f',padding=3);axes[1].set_ylabel('训练标准差为单位的平均偏移')
        axes[1].set_ylim(0,max(values)*1.25 if max(values)>0 else 1);axes[1].set_title('换背景后概念偏移（越低越好）')
        for ax in axes:
            ax.set_xticks(np.arange(len(cnames)),cnames,rotation=15);ax.spines[['top','right']].set_visible(False)
        fig.suptitle(title);file=f'matched_correlation_shift_{i}.png';fig.savefig(out/file,dpi=170);plt.close(fig)
        lines += [f'![相关性与概念偏移]({file})','']
    cross=[]
    for dataset in sorted({r['dataset'] for r in conditions}):
        for protocol in ['原图训练','背景增强训练']:
            group=[r for r in conditions if r['dataset']==dataset and r['protocol']==protocol and r['training_seed']==42]
            if len(group)<2:continue
            group=sorted(group,key=lambda r:['resnet18','resnet50','clip_RN50'].index(r['backbone']))
            common=set(group[0]['shared_concept_indices'])
            for row in group[1:]:common &= set(row['shared_concept_indices'])
            common=sorted(common)
            if not common:continue
            values=[];methods=sorted(set.union(*[set(r['methods']) for r in group])-{'非 CBM'})
            for row in group:
                local={bank:i for i,bank in enumerate(row['shared_concept_indices'])}
                take={local[i] for i in common}
                for method in methods:
                    if method not in row['methods']:continue
                    for view in ['original','changed']:
                        metrics=row['methods'][method][view]['concepts']
                        ap=[r['value'] for r in metrics['average_precision_per_concept'] if r['concept_index'] in take]
                        pearson=[r['value'] for r in metrics['pearson_per_concept'] if r['concept_index'] in take]
                        values.append({'backbone':row['backbone'],'method':method,'view':view,
                                       'mAP':float(np.mean(ap)) if ap else None,
                                       'pearson':float(np.mean(pearson)) if pearson else None})
            fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
            for ax,view,title in zip(axes,['original','changed'],['原图概念 mAP','换背景概念 mAP']):
                for method in methods:
                    points=[(i,100*v['mAP']) for i,row in enumerate(group) for v in values
                            if v['backbone']==row['backbone'] and v['method']==method and v['view']==view and v['mAP'] is not None]
                    if points:ax.plot(*np.array(points).T,'o-',color=COLORS[method],label=method)
                ax.set_xticks(range(len(group)),[r['backbone'] for r in group]);ax.set_ylabel('相同概念列 mAP（%）')
                ax.set_ylim(0,102);ax.set_title(title);ax.spines[['top','right']].set_visible(False)
            handles,labels=axes[0].get_legend_handles_labels()
            if handles:fig.legend(handles,labels,loc='outside lower center',ncol=4)
            title=f'{dataset.upper()} · {protocol} · 骨干间共同概念 {len(common)} 个'
            fig.suptitle(title);file=f'cross_backbone_{dataset}_{protocol}.png';fig.savefig(out/file,dpi=170);plt.close(fig)
            lines += [f'## {title}', '',f'![骨干间共同概念比较]({file})','']
            cross.append({'dataset':dataset,'protocol':protocol,'common_indices':common,'results':values})
    (out/'cross_backbone_summary.json').write_text(json.dumps(cross,indent=2)+'\n')
    if not conditions: lines += ['正式训练组仍在运行，结果文件完成后自动加入图表。','']
    (out/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__': main()
