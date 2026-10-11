"""Evaluate fresh LF checkpoints on official native32 CIFAR-C, all10000 test images."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .lf import upstream,predict_features,export_npz
from .lf_cifar import ROOT,CORRUPTIONS,load_encoders,encode,sha,teacher_metrics
from .metrics import classification_metrics,paired_concept_metrics


def run(dataset,device='cuda',batch=128,run_dir=None):
    upstream()
    from pi_cbm.pipeline import build_model,baseline_from_state
    base=ROOT/'artifacts/common'/dataset
    source=base/'rn18_imagenet';teacher_dir=base/'teacher'
    corruption_root=ROOT/'artifacts/common/cifar_c'/('CIFAR-10-C' if dataset=='cifar10' else 'CIFAR-100-C')
    original=np.load(source/'test_original.npz');train=np.load(source/'train_original.npz')
    ids=original['ids'];native_ids=ids-50000
    labels=np.load(corruption_root/'labels.npy',mmap_mode='r')
    if not np.array_equal(labels[:10000][native_ids],original['labels']):
        raise ValueError('Official corruption labels do not match clean canonical IDs')
    names=json.loads((teacher_dir/'metadata.json').read_text())['names']
    cached=source/'corruptions';cached.mkdir(exist_ok=True)
    out=Path(run_dir) if run_dir else ROOT/'artifacts/matched/lf'/dataset/'rn18_imagenet/seed42'
    result_dir=out/'corruptions';result_dir.mkdir(exist_ok=True)
    saved=torch.load(out/'models.pt',map_location='cpu',weights_only=True)
    selected=saved['selected'];config=saved['config'];selected_names=saved['concepts']
    text=torch.from_numpy(np.load(teacher_dir/'text.npy').copy())
    data={'splits':{'train':{'features':torch.from_numpy(train['features'].copy())}},'text':text}
    baseline_config=copy.deepcopy(config);baseline_config['injection']['site']='baseline'
    baseline=build_model(baseline_from_state(saved['baseline'],device),data,selected,baseline_config,device)
    pi_cbm=baseline_from_state(saved['pi_cbm'],device)
    pi=build_model(pi_cbm,data,selected,config,device)
    pi.generator.load_state_dict(saved['generator']);pi.eval().requires_grad_(False)
    baseline.eval().requires_grad_(False)
    backbone,teacher,teacher_text=load_encoders(device,names)
    report={'dataset':dataset,'checkpoint_sha256':sha(out/'models.pt'),'training_protocol':config['data']['benchmark_train_views'],
            'candidate_concepts':len(names),'selected_indices':selected.tolist(),'selected_concepts':len(selected),
            'inference_draws':1,'inference_seed':3000,'test_selection':False,'test_images_per_setting':10000,
            'official_source':'https://zenodo.org/records/'+('2535967' if dataset=='cifar10' else '3555552'),
            'severity_blocks':'official native32 array [(severity-1)*10000:severity*10000], then canonical test row IDs',
            'human_concept_annotations':False,'background_swap':'not applicable; foreground masks unavailable',
            'settings':{},'metrics':{}}
    for kind in CORRUPTIONS:
        marker=corruption_root/(kind+'.npy_COMPLETE.json')
        if not marker.exists():raise ValueError(f'Official corruption download incomplete: {kind}')
        official=np.load(corruption_root/(kind+'.npy'),mmap_mode='r')
        for severity in [1,3,5]:
            key=f'{kind}_{severity}';cache_path=cached/f'test_{key}.npz'
            if not cache_path.exists():
                raw=np.asarray(official[(severity-1)*10000+native_ids]).transpose(0,3,1,2)
                xs=np.empty((10000,512),dtype='float32');ys=np.empty((10000,len(names)),dtype='float32')
                for start,x,y in encode(raw,backbone,teacher,teacher_text,device,batch):
                    xs[start:start+len(x)]=x;ys[start:start+len(x)]=y
                np.savez_compressed(cache_path,features=xs,teacher=ys,labels=original['labels'],ids=ids,
                    targets=np.zeros_like(ys),known=np.zeros_like(ys,dtype=bool),concept_names=np.asarray(names))
            inputs=np.load(cache_path)
            split={'features':torch.from_numpy(inputs['features'].copy()),'labels':torch.from_numpy(inputs['labels'].copy()),
                   'indices':torch.from_numpy(inputs['ids'].copy())}
            for method,model in [('lfcbm',baseline),('lfpi_global',pi)]:
                pred=predict_features(model,split['features'],draws=1,seed=3000)
                path=result_dir/f'{method}_test_{key}.npz'
                export_npz(path,pred,split,selected_names,targets=inputs['targets'][:,selected.numpy()],known=inputs['known'][:,selected.numpy()])
                orig=np.load((out/'single_draw' if method=='lfpi_global' else out)/f'{method}_test_original.npz')
                train_pred=np.load((out/'single_draw' if method=='lfpi_global' else out)/f'{method}_train_original.npz')
                report['metrics'][f'{method}/{key}']={
                    'classification':classification_metrics(pred['logits'].numpy(),inputs['labels']),
                    'CLIP_teacher':teacher_metrics(pred['concepts'].numpy(),inputs['teacher'][:,selected.numpy()]),
                    'paired_clean_shift':paired_concept_metrics(orig['concepts'],pred['concepts'].numpy(),train_pred['concepts'])}
            report['settings'][key]={'cached_inputs':str(cache_path),'ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
                'official_array':str(corruption_root/(kind+'.npy')),'official_array_sha256':json.loads(marker.read_text())['sha256'],
                'severity':severity,'images':10000}
            (result_dir/'report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'dataset':dataset,'corruption':key,'test_images':10000,'completed':True}),flush=True)
    print(json.dumps({'dataset':dataset,'all_corruption_settings':12,'report':str(result_dir/'report.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True,choices=['cifar10','cifar100'])
    p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=128);p.add_argument('--run')
    a=p.parse_args();torch.set_num_threads(4);run(a.dataset,a.device,a.batch,a.run)
