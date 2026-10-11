"""Render completed NEC paths and common RGB interventions in Chinese PNGs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from .metrics import classification_metrics
from .report_utils import write_metrics

NAMES={'lfcbm':'LF-CBM','lfpi_global':'LF-Pi（8次采样）','lfpi_global_1draw':'LF-Pi','lfpi':'LF-Pi（8次采样）','lfpi_single':'LF-Pi','pini_mean':'PiNI 均值',
       'pini_noise':'PiNI 均值＋噪声','noncbm':'非 CBM'}
COLORS={'LF-CBM':'#2878B5','LF-Pi':'#E15759','LF-Pi（8次采样）':'#E15759','PiNI 均值':'#59A14F',
        'PiNI 均值＋噪声':'#B07AA1','非 CBM':'#F28E2B'}


def shared_accuracy(directory, dataset, head, view):
    data=np.load(directory/f'{head}_{view}.npz')
    manifest=json.loads((Path('artifacts/common')/dataset/'rn18_imagenet/manifest.json').read_text())
    ep=Path(manifest.get('cache_root','artifacts/common/'+dataset+'/cache'))/'background_eligible.npy'
    mask=np.load(ep)[data['ids']].astype(bool) if ep.exists() else np.ones(len(data['ids']),bool)
    return classification_metrics(data['logits'][mask],data['labels'][mask])['accuracy']


def main():
    out=Path('reports');out.mkdir(exist_ok=True)
    font=Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    lines=['# 泛化与分类头容量', '']; groups={}
    for file in sorted(Path('artifacts/heads').rglob('report.json')):
        r=json.loads(file.read_text());group=file.parents[1];groups.setdefault(group,[]).append((file,r))
    for index,(group,reports) in enumerate(groups.items()):
        views=list(dict.fromkeys(p['view'] for _,r in reports for p in r['test_evaluations']))
        fig,axes=plt.subplots(1,len(views),figsize=(5.5*len(views),4.8),layout='constrained',squeeze=False)
        axes=axes[0]
        for file,r in reports:
            name=NAMES[file.parent.name];directory=file.parent
            for ax,view in zip(axes,views):
                label='原图测试' if view=='original' else '换背景测试'
                curve=[]
                for point in r['test_evaluations']:
                    if point['view']!=view or point['head'] in ['dense','best_val']:continue
                    accuracy=shared_accuracy(directory,r['dataset'],point['head'],view)
                    curve.append((point['nec'],100*accuracy))
                if curve:
                    curve=sorted(curve);ax.plot(*np.array(curve).T,'o-',color=COLORS[name],label=name,markersize=4)
                ax.set_xlabel('每类有效概念连接数（实际 NEC）');ax.set_ylabel('分类准确率（%）')
                ax.set_title(label);ax.set_ylim(0,102);ax.spines[['top','right']].set_visible(False)
            # Dense references are exported in JSON, kept separate from sparse curves.
        handles,labels=axes[0].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,loc='outside lower center',ncol=min(4,len(handles)))
        title=str(group.relative_to(Path('artifacts/heads')));fig.suptitle(title)
        file=f'nec_{index}.png';fig.savefig(out/file,dpi=170);plt.close(fig)
        labels=['稠密头原图准确率' if view=='original' else '稠密头换背景准确率' for view in views]
        lines += [f'## {title}', '', f'![NEC 曲线]({file})','',
                  '| 方法 | '+' | '.join(labels)+' | SAGA 达到步数上限的路径点 |',
                  '| '+' | '.join(['---']*(len(views)+2))+' |']
        for path,r in reports:
            cells=[f"{100*shared_accuracy(path.parent,r['dataset'],'dense',view):.2f}%"
                   if (path.parent/f'dense_{view}.npz').exists() else '—' for view in views]
            lines.append(f"| {NAMES[path.parent.name]} | "+' | '.join(cells)+f" | {r['unconverged_path_points']} |")
        lines += ['']
    ood=[]
    for directory in sorted(Path('artifacts/ood').glob('*/resnet18/paired/seed42')):
        dataset=directory.parents[2].name
        results={file.name.removesuffix('_COMPLETE.json'):json.loads(file.read_text()) for file in directory.glob('*_COMPLETE.json')}
        if not results:continue
        for setting,metrics in results.items():
            for method,value in metrics.items():
                with np.load(directory/(setting+'_'+method+'.npz')) as data:
                    value['classification']=classification_metrics(data['logits'],data['labels'])
        ood.append({'dataset':dataset,'results':results})
        methods=sorted(set.intersection(*[set(r) for r in results.values()]))
        if 'lfpi_single' in methods:methods.remove('lfpi')
        fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        for ax,kind,title in zip(axes.flat,['gaussian_noise','gaussian_blur','contrast','jpeg'],['高斯噪声','高斯模糊','降低对比度','JPEG 压缩']):
            for method in methods:
                points=[(s,100*results[f'{kind}_{s}'][method]['classification']['accuracy']) for s in [1,3,5] if f'{kind}_{s}' in results]
                if points:ax.plot(*np.array(points).T,'o-',color=COLORS[NAMES[method]],label=NAMES[method])
            ax.set_title(title);ax.set_xlabel('扰动强度档位');ax.set_xticks([1,3,5]);ax.set_ylim(0,102);ax.set_ylabel('分类准确率（%）');ax.spines[['top','right']].set_visible(False)
        handles,labels=axes[0,0].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,loc='outside lower center',ncol=3)
        n=next(iter(next(iter(results.values())).values()))['classification']['examples']
        fig.suptitle(f'{dataset.upper()} · 同批 {n} 张测试图 · 背景增强训练 · ResNet18')
        file=f'corruptions_{dataset}.png';fig.savefig(out/file,dpi=170);plt.close(fig)
        lines += [f'## {dataset.upper()} 图像扰动','',
                  '本组采用单次噪声采样。' if 'lfpi_single' in methods else '本组 LF-Pi 使用 8 次噪声采样，PiNI 使用单次采样。',
                  '',f'![扰动泛化]({file})','']
        masks=[k for k in ['original_0','foreground_only_0','background_only_0'] if k in results]
        fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');x=np.arange(len(masks))
        for i,method in enumerate(methods):
            values=[100*results[k][method]['classification']['accuracy'] for k in masks]
            ax.bar(x+(i-(len(methods)-1)/2)*.15,values,.14,color=COLORS[NAMES[method]],label=NAMES[method])
        labels={'original_0':'原图','foreground_only_0':'仅主体','background_only_0':'仅背景'}
        ax.set_xticks(x,[labels[k] for k in masks]);ax.set_ylim(0,102);ax.set_ylabel('分类准确率（%）')
        ax.set_title(f'{dataset.upper()} · 主体／背景干预 · 同批 {n} 张测试图')
        ax.spines[['top','right']].set_visible(False);fig.legend(loc='outside lower center',ncol=3)
        file=f'masked_{dataset}.png';fig.savefig(out/file,dpi=170);plt.close(fig)
        lines += [f'![主体与背景干预]({file})','']
        if dataset=='waterbirds':
            fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
            for i,method in enumerate(methods):
                values=[100*results[k][method]['classification']['balanced_accuracy'] for k in masks]
                ax.bar(x+(i-(len(methods)-1)/2)*.15,values,.14,color=COLORS[NAMES[method]],label=NAMES[method])
            ax.axhline(50,color='#777777',linestyle='--',linewidth=1)
            ax.set_xticks(x,[labels[k] for k in masks]);ax.set_ylim(0,102);ax.set_ylabel('两类平均准确率（%）')
            ax.set_title('CUB–Places 水鸟二分类 · 主体／背景干预')
            ax.spines[['top','right']].set_visible(False);fig.legend(loc='outside lower center',ncol=3)
            file='masked_waterbirds_balanced.png';fig.savefig(out/file,dpi=170);plt.close(fig)
            lines += [f'![水鸟两类平均准确率]({file})','']
    write_metrics(out/'generalization_summary.json',ood)
    (out/'GENERALIZATION_RESULT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
