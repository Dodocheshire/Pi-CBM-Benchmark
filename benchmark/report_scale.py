"""Render completed scale experiments using classification and CLIP fidelity."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .report_utils import write_metrics


def main():
    font=Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    out=Path('reports');rows=[];lines=['# 大规模数据集结果','']
    for dataset in ['imagenet_200_per_class','places365']:
        path=Path('artifacts/matched/lf')/dataset/'rn18_imagenet/seed42/report.json'
        if not path.exists():continue
        report=json.loads(path.read_text());manifest=report['feature_provenance']
        names=['LF-CBM','LF-Pi'];colors=['#2878B5','#E15759']
        values=[report['metrics'][f'{method}/test/original'] for method in ['lfcbm','lfpi_global']]
        controls=[]
        for file in sorted((Path('artifacts/noncbm')/f'{dataset}_rn18_imagenet').glob('seed*/results.json')):
            controls.append(json.loads(file.read_text())['test']['accuracy'])
        fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
        accuracy=[100*v['classification']['accuracy'] for v in values]
        if controls:
            accuracy.append(100*np.mean(controls));class_names=names+[f'非 CBM（{len(controls)} 个种子）']
        else:class_names=names
        bars=axes[0].bar(class_names,accuracy,color=colors+(['#F28E2B'] if controls else []))
        axes[0].bar_label(bars,fmt='%.2f',padding=3);axes[0].set_ylim(0,102)
        axes[0].set_title('测试分类准确率');axes[0].set_ylabel('准确率（%）')
        for ax,key,title in zip(axes[1:],['teacher_pearson_macro','teacher_cos_cubed_macro'],
                                ['CLIP 教师相关系数','CLIP 教师三次余弦']):
            bars=ax.bar(names,[v['CLIP_teacher'][key] for v in values],color=colors)
            ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_ylim(0,1.05);ax.set_title(title)
        for ax in axes:ax.spines[['top','right']].set_visible(False)
        title=f"{dataset} · ImageNet ResNet18 · 保留概念 {report['selected_concepts']}/{report['bank_concepts']}"
        fig.suptitle(title);name=f'scale_{dataset}.png';fig.savefig(out/name,dpi=170);plt.close(fig)
        lines += [f'## {dataset} · {manifest["classes"]} 类','',
                  f"训练／验证／测试：{manifest['split_counts']['train']}／{manifest['split_counts']['val']}／{manifest['split_counts']['test']} 张；训练种子 42，单次噪声推理。",'',
                  f'![分类与概念教师一致性]({name})','']
        rows.append({'dataset':dataset,'backbone':'resnet18','split_counts':manifest['split_counts'],
                     'retained_concepts':report['selected_concepts'],'candidate_concepts':report['bank_concepts'],
                     'inference_draws':1,'training_seed':42,'results':dict(zip(names,values)),
                     'noncbm_seeds':len(controls),'noncbm_accuracy_mean':float(np.mean(controls)) if controls else None})
    write_metrics(out/'scale_summary.json',rows)
    (out/'SCALE_RESULT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
