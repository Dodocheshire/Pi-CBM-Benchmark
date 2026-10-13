"""Independent CIFAR reports: classification and continuous CLIP fidelity, no native AP."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from .lf_cifar import ROOT,CORRUPTIONS
from .metrics import classification_metrics,teacher_fidelity
from .report_utils import write_metrics,backbone_label

COLORS={'lfcbm':'#2878B5','lfpi_global':'#D84A49','pini_mean':'#2CA02C','pini_noise':'#9467BD','noncbm':'#FF8C00'}
METHODS={'lfcbm':'LF-CBM','lfpi_global':'LF-Pi','pini_mean':'PiNI 均值','pini_noise':'PiNI 噪声','noncbm':'非 CBM'}


def collect_pini_corruptions(dataset):
    directory=ROOT/'artifacts/pini_cifar_c'/dataset/'rn18_imagenet/seed42'
    marker=directory/'CIFAR_C_COMPLETE.json'
    if not marker.exists():return None
    completion=json.loads(marker.read_text())
    manifest=directory/'input_manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==completion['source_manifest_sha256']
    assert completion['settings']==12 and completion['rows_per_setting']==10000
    assert completion['arms']==2 and completion['draws']==1 and not completion['training_performed']
    report=json.loads((directory/'results.json').read_text())
    assert report['models_fixed'] and not report['head_retraining']
    assert report['inference_draws']==1 and report['inference_seed']==4242
    assert not report['native_concept_annotations']
    settings={f'{kind}_{severity}' for kind in CORRUPTIONS for severity in [1,3,5]}
    assert set(report['settings'])==settings
    with np.load(ROOT/'artifacts/common'/dataset/'rn18_imagenet/test_original.npz') as reference:
        ids,labels=reference['ids'],reference['labels']
    for setting,arms in report['settings'].items():
        assert set(arms)=={'mean','noise'}
        region=json.loads((directory/'regions'/setting/'COMPLETE.json').read_text())
        assert region['images']==10000
        for arm,metrics in arms.items():
            assert metrics['all_test_rows']==10000 and metrics['native_concept_mAP'] is None
            assert metrics['inference_draws']==1 and metrics['inference_seed']==4242
            with np.load(directory/'exports'/arm/f'test_{setting}.npz') as values:
                assert np.array_equal(values['ids'],ids) and np.array_equal(values['labels'],labels)
                assert not values['known'].any() and np.isfinite(values['concept_logits']).all()
                assert classification_metrics(values['logits'],labels)==metrics['classification']
    return {'source_report':str(directory/'results.json'),'metrics':report['settings']}


def collect(run_dir,dataset,teacher_dir,protocol):
    run_dir=Path(run_dir)
    if not run_dir.exists() or not (run_dir/'report.json').exists():return []
    candidates=list(run_dir.glob('*_test_original.npz'))
    # Prefer explicitly recorded equal-budget export for LF-Pi.
    single=run_dir/'single_draw/lfpi_global_test_original.npz'
    if single.exists():candidates=[p for p in candidates if not p.name.startswith('lfpi_global_')]+[single]
    names=protocol['names'];name_index={name:i for i,name in enumerate(names)}
    teacher=np.load(teacher_dir/'original_teacher.npy',mmap_mode='r');rows=[]
    report=json.loads((run_dir/'report.json').read_text()) if (run_dir/'report.json').exists() else {}
    for path in candidates:
        values=np.load(path)
        if not {'concepts','ids','labels','logits','concept_names'}.issubset(values.files):continue
        selected=[name_index[str(name)] for name in values['concept_names']]
        if values['known'].any():raise ValueError('CIFAR export incorrectly marks artificial concepts as native known')
        method=path.name.removesuffix('_test_original.npz')
        rows.append({'dataset':dataset,'backbone':report.get('backbone','rn18_imagenet'),'method':method,'source':str(path),'run':str(run_dir),
            'training_seed':report.get('training_seed',42),'train_views':report.get('train_views',['original']),
            'inference_draws':1,'candidate_concepts':len(names),'retained_concepts':len(selected),
            'bank_coverage':len(selected)/len(names),'selected_indices':selected,
            'human_annotations_available':False,'native_concept_mAP':None,
            'classification':classification_metrics(values['logits'],values['labels']),
            'CLIP_teacher':teacher_fidelity(values['concepts'],teacher[values['ids']][:,selected])})
    return rows


def run(out,pini_root=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];corruption_reports={}
    font=Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    for filename in ['clean.pdf','corruptions.pdf']:
        (out/filename).unlink(missing_ok=True)
    for dataset in ['cifar10','cifar100']:
        teacher_dir=ROOT/'artifacts/common'/dataset/'teacher'
        if not (teacher_dir/'metadata.json').exists():continue
        metadata=json.loads((teacher_dir/'metadata.json').read_text())
        lf=ROOT/'artifacts/matched/lf'/dataset/'rn18_imagenet/seed42'
        rows+=collect(lf,dataset,teacher_dir,metadata)
        rows+=collect(lf.parent/'gaussian_noise_augmented/seed42',dataset,teacher_dir,metadata)
        for bone in ['rn50_imagenet','clip_RN50']:
            wide=lf.parent.parent/bone/'seed42'
            rows+=collect(wide,dataset,teacher_dir,metadata)
            rows+=collect(wide.parent/'gaussian_noise_augmented/seed42',dataset,teacher_dir,metadata)
        if pini_root:
            for run_dir in (Path(pini_root)/dataset).glob('**/seed42'):rows+=collect(run_dir,dataset,teacher_dir,metadata)
            for relative,bone in [(Path(dataset),'rn18_imagenet'),(Path('resnet50')/dataset,'rn50_imagenet'),(Path('clip_rn50')/dataset,'clip_RN50')]:
                experiment=Path(pini_root)/relative
                if not (experiment/'CIFAR_COMPLETE.json').exists():continue
                for method in ['mean','noise']:
                    directory=experiment/'exports'/method
                    path=directory/'test_original.npz'
                    if not path.exists() or not (directory/'results.json').exists():continue
                    values=np.load(path);result=json.loads((directory/'results.json').read_text())
                    if values['known'].any():raise ValueError('PiNI CIFAR pseudo supervision is not native truth')
                    index=values['concept_indices'];teacher=np.load(teacher_dir/'original_teacher.npy',mmap_mode='r')
                    fidelity=teacher_fidelity(values['concept_logits'],teacher[values['ids']][:,index])
                    rows.append({'dataset':dataset,'backbone':bone,'method':'pini_'+method,'source':str(path),'run':str(directory),
                        'training_seed':result['training_seed'],'train_views':result['train_views'],
                        'inference_draws':result['inference_draws'],'inference_seed':result['inference_seed'],
                        'candidate_concepts':len(metadata['names']),'retained_concepts':len(index),
                        'bank_coverage':len(index)/len(metadata['names']),'selected_indices':index.tolist(),
                        'human_annotations_available':False,'native_concept_mAP':None,
                        'classification':classification_metrics(values['logits'],values['labels']),
                        'teacher_score_field':'concept_logits',
                        'CLIP_teacher':fidelity,'CLIP_teacher_probability_secondary':teacher_fidelity(values['concepts'],teacher[values['ids']][:,index])})
        for bone in ['rn18_imagenet','rn50_imagenet','clip_RN50']:
            for augmented in [False,True]:
                directory=lf.parent.parent/bone
                directory=directory/'gaussian_noise_augmented/seed42' if augmented else directory/'seed42'
                path=directory/'corruptions/report.json'
                if path.exists():
                    report=json.loads(path.read_text())
                    if len(report.get('settings', {}))!=12 or len(report.get('metrics', {}))!=24:
                        continue
                    report.setdefault('backbone',bone)
                    key=dataset+'/'+bone+('/gaussian_noise_s1_augmented' if augmented else '/clean')
                    report['source_report']=str(path)
                    if 'feature_provenance' in report:
                        report['feature_provenance']={k:v for k,v in report['feature_provenance'].items() if k!='splits'}
                    corruption_reports[key]=report
        pini_corruptions=collect_pini_corruptions(dataset)
        if pini_corruptions is not None:
            key=dataset+'/rn18_imagenet/gaussian_noise_s1_augmented'
            if key in corruption_reports:
                corruption_reports[key]['pini_corruptions']=pini_corruptions
    for dataset in ['cifar10','cifar100']:
        for paired in [False,True]:
            group=[r for r in rows if r['dataset']==dataset and (len(r['train_views'])>1)==paired]
            if not group:continue
            common=sorted(set.intersection(*[set(r['selected_indices']) for r in group]))
            teacher=np.load(ROOT/'artifacts/common'/dataset/'teacher/original_teacher.npy',mmap_mode='r')
            for row in group:
                values=np.load(row['source']);local={global_index:i for i,global_index in enumerate(row['selected_indices'])}
                row['training_protocol']='gaussian_noise_s1_augmented' if paired else 'clean'
                row['common_intersection_indices']=common;row['common_intersection_concepts']=len(common)
                score_field=row.get('teacher_score_field','concepts')
                row['CLIP_teacher_common_intersection']=teacher_fidelity(values[score_field][:,[local[i] for i in common]],teacher[values['ids']][:,common]) if common else None
    controls=[]
    for dataset in ['cifar10','cifar100']:
        for bone in ['rn18_imagenet','rn50_imagenet','clip_RN50']:
            for augmented in [False,True]:
                suffix='_gaussian_noise_augmented' if augmented else ''
                for seed in [42,43,44]:
                    path=ROOT/'artifacts/noncbm'/f'{dataset}_{bone}{suffix}'/f'seed{seed}'/'results.json'
                    if not path.exists():continue
                    result=json.loads(path.read_text());source=Path(result['input_paths']['test'])
                    if source.resolve()!=(ROOT/'artifacts/common'/dataset/bone/'test_original.npz').resolve():continue
                    predicted=np.load(path.parent/'test_predictions.npz');raw=np.load(source)
                    if not np.array_equal(predicted['ids'],raw['ids']) or not np.array_equal(predicted['labels'],raw['labels']):raise ValueError('NonCBM canonical test mismatch')
                    train_views=['original','gaussian_noise_s1'] if augmented else ['original']
                    controls.append({'dataset':dataset,'backbone':bone,'head_seed':seed,'train_views':train_views,
                        'source':str(path),'method':'noncbm','classification':classification_metrics(predicted['logits'],predicted['labels'])})
    shifted_controls=[]
    for control in controls:
        path=Path(control['source']).parent/'shifts/report.json'
        if not path.exists():continue
        report=json.loads(path.read_text())
        shifted_controls.append({'dataset':control['dataset'],'backbone':control['backbone'],'train_views':control['train_views'],
            'seed':control['head_seed'],'source':str(path),'refitting':report['refitting'],
            'evaluation':report['evaluation']})
    summary={'scope':'Fresh frozen-backbone full training seed42; inference1draw; compare identical training protocols; test excluded from selection',
             'concept_annotations':'No native per-image CIFAR concept truth; report only continuous frozen CLIP teacher fidelity',
             'native_concept_mAP':None,'records':rows,'non_cbm_controls':controls,
             'non_cbm_corruptions':shifted_controls,'corruptions':corruption_reports}
    precision_path=ROOT/'artifacts/cifar_feasibility/precision_audit.json'
    if precision_path.exists():summary['input_precision_audit']={'source':str(precision_path),'measured':json.loads(precision_path.read_text()),
        'shared_input_policy':'Same frozen weights and canonical resize224; record actual float32 intensity roundtrip and TF32 batch-shape errors; no false bitwise identity claim'}
    write_metrics(out/'summary.json',summary)
    fields=['dataset','method','backbone','training_seed','candidate_concepts','retained_concepts','bank_coverage','training_protocol','common_intersection_concepts','accuracy','teacher_pearson_macro','teacher_cos_cubed_macro']
    with (out/'clean.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in rows:
            fidelity=row['CLIP_teacher_common_intersection'] or row['CLIP_teacher']
            writer.writerow({key:row[key] for key in fields[:9]}|{'accuracy':row['classification']['accuracy'],
                'teacher_pearson_macro':fidelity['teacher_pearson_macro'],
                'teacher_cos_cubed_macro':fidelity['teacher_cos_cubed_macro']})
    if rows:
        groups=[(dataset,bone,paired,[r for r in rows if r['dataset']==dataset and r['backbone']==bone and (len(r['train_views'])>1)==paired])
                for dataset in ['cifar10','cifar100'] for bone in ['rn18_imagenet','rn50_imagenet','clip_RN50'] for paired in [False,True]]
        groups=[group for group in groups if group[3]]
        fig,axes=plt.subplots(len(groups),3,figsize=(14,3.5*len(groups)),layout='constrained',squeeze=False)
        for i,(dataset,bone,paired,selected) in enumerate(groups):
            methods=[METHODS.get(row['method'],row['method']) for row in selected]
            colors=[COLORS.get(row['method'],'#FF8C00') for row in selected]
            protocol='高斯噪声配对训练' if paired else '原图训练'
            metrics=[('测试准确率',[row['classification']['accuracy'] for row in selected]),
                     ('CLIP 教师相关系数（共同概念）',[(row['CLIP_teacher_common_intersection'] or row['CLIP_teacher'])['teacher_pearson_macro'] for row in selected]),
                     ('CLIP 教师三次余弦（共同概念）',[(row['CLIP_teacher_common_intersection'] or row['CLIP_teacher'])['teacher_cos_cubed_macro'] for row in selected])]
            for metric_index,(ax,(title,values)) in enumerate(zip(axes[i],metrics)):
                labels=list(methods);palette=list(colors);plotted=list(values)
                reference=[c['classification']['accuracy'] for c in controls if c['dataset']==dataset and c['backbone']==bone and (len(c['train_views'])>1)==paired]
                if metric_index==0 and reference:
                    labels.append(f'非 CBM\n{len(reference)} 个头种子');palette.append(COLORS['noncbm']);plotted.append(float(np.mean(reference)))
                bars=ax.bar(labels,plotted,color=palette);ax.bar_label(bars,fmt='%.3f',padding=3)
                if metric_index==0 and reference:ax.errorbar(len(labels)-1,float(np.mean(reference)),yerr=float(np.std(reference,ddof=1)) if len(reference)>1 else 0,color='black',capsize=3)
                ax.set_title(dataset.upper()+'｜'+backbone_label(bone)+'\n'+protocol+'\n'+title,fontsize=10);ax.set_ylabel('指标值');ax.tick_params(axis='x',rotation=20);ax.set_ylim(0,1)
        fig.suptitle('训练种子 42｜单次推理\n跨骨干按训练协议比较共同概念交集；CIFAR 无真实逐图概念标注')
        fig.savefig(out/'clean.png',dpi=170);plt.close(fig)
    if corruption_reports:
        fig,axes=plt.subplots(len(corruption_reports),4,figsize=(14,3.5*len(corruption_reports)),layout='constrained',squeeze=False)
        titles={'gaussian_noise':'高斯噪声','gaussian_blur':'高斯模糊','contrast':'对比度','jpeg_compression':'JPEG 压缩'}
        for i,(_,report) in enumerate(sorted(corruption_reports.items())):
            dataset=report['dataset'];bone=report.get('backbone','rn18_imagenet');protocol='高斯噪声配对训练' if len(report['training_protocol'])>1 else '原图训练'
            for ax,kind in zip(axes[i],CORRUPTIONS):
                for method in ['lfcbm','lfpi_global']:
                    points=[(severity,report['metrics'][f'{method}/{kind}_{severity}']['classification']['accuracy'])
                        for severity in [1,3,5] if f'{method}/{kind}_{severity}' in report.get('metrics',{})]
                    if points:ax.plot(*zip(*points),marker='o',label=METHODS[method],color=COLORS[method])
                if 'pini_corruptions' in report:
                    for arm in ['mean','noise']:
                        points=[(severity,report['pini_corruptions']['metrics'][f'{kind}_{severity}'][arm]['classification']['accuracy']) for severity in [1,3,5]]
                        method='pini_'+arm
                        ax.plot(*zip(*points),marker='o',label=METHODS[method],color=COLORS[method])
                controls_here=[c for c in shifted_controls if c['dataset']==dataset and c['backbone']==bone
                               and (len(c['train_views'])>1)==(len(report['training_protocol'])>1)]
                if controls_here:
                    points=[]
                    for severity in [1,3,5]:
                        values=[row['accuracy'] for c in controls_here for row in c['evaluation']
                                if row['setting']==f'{kind}_{severity}']
                        if values:points.append((severity,float(np.mean(values)),float(np.std(values,ddof=1)) if len(values)>1 else 0))
                    if points:
                        ax.errorbar([p[0] for p in points],[p[1] for p in points],yerr=[p[2] for p in points],
                                    marker='s',capsize=3,color=COLORS['noncbm'],label=f'非 CBM（{len(controls_here)} 个头种子）')
                ax.set_title(dataset.upper()+'｜'+backbone_label(bone)+'\n'+protocol+'\n'+titles[kind],fontsize=10);ax.set_xlabel('官方 CIFAR-C 腐蚀强度');ax.set_ylabel('准确率');ax.set_ylim(0,1);ax.set_xticks([1,3,5])
                if ax.lines:ax.legend()
        fig.suptitle('跨骨干｜官方原生 32×32 CIFAR-C 四类腐蚀子集｜每设置全部 10000 测试图｜单次推理')
        fig.savefig(out/'corruptions.png',dpi=170);plt.close(fig)
    print(json.dumps({'report':str(out/'summary.json'),'clean_runs':len(rows),'corruption_datasets':len(corruption_reports)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default=str(ROOT/'reports/cifar'))
    p.add_argument('--pini-root',default=str(ROOT/'artifacts/pini_matched'));a=p.parse_args();run(a.out,a.pini_root)
