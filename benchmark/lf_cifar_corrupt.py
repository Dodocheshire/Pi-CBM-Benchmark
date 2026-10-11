"""Evaluate fresh LF checkpoints on official native32 CIFAR-C, all10000 test images."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .lf import upstream,predict_features,export_npz
from .lf_cifar import ROOT,CORRUPTIONS,load_encoders,encode,resize224,sha,teacher_metrics
from .metrics import classification_metrics,paired_concept_metrics


def run(dataset,device='cuda',batch=128,run_dir=None,backbone_name='resnet18'):
    upstream()
    from pi_cbm.pipeline import build_model,baseline_from_state
    base=ROOT/'artifacts/common'/dataset
    folder={'resnet18':'rn18_imagenet','resnet50':'rn50_imagenet','clip_RN50':'clip_RN50'}[backbone_name]
    source=base/folder;teacher_dir=base/'teacher'
    corruption_root=ROOT/'artifacts/common/cifar_c'/('CIFAR-10-C' if dataset=='cifar10' else 'CIFAR-100-C')
    original=np.load(source/'test_original.npz');train=np.load(source/'train_original.npz')
    ids=original['ids'];native_ids=ids-50000
    labels=np.load(corruption_root/'labels.npy',mmap_mode='r')
    if not np.array_equal(labels[:10000][native_ids],original['labels']):
        raise ValueError('Official corruption labels do not match clean canonical IDs')
    names=json.loads((teacher_dir/'metadata.json').read_text())['names']
    cached=source/'corruptions';cached.mkdir(exist_ok=True)
    out=Path(run_dir) if run_dir else ROOT/'artifacts/matched/lf'/dataset/folder/'seed42'
    result_dir=out/'corruptions';result_dir.mkdir(exist_ok=True)
    saved=torch.load(out/'models.pt',map_location='cpu',weights_only=True)
    selected=saved['selected'];config=saved['config'];selected_names=saved['concepts']
    if config['backbone']['name'] != backbone_name:
        raise ValueError('Requested backbone does not match checkpoint; refusing cross-backbone cache use')
    feature_dim=int(train['features'].shape[1])
    text=torch.from_numpy(np.load(teacher_dir/'text.npy').copy())
    data={'splits':{'train':{'features':torch.from_numpy(train['features'].copy())}},'text':text}
    baseline_config=copy.deepcopy(config);baseline_config['injection']['site']='baseline'
    baseline=build_model(baseline_from_state(saved['baseline'],device),data,selected,baseline_config,device)
    pi_cbm=baseline_from_state(saved['pi_cbm'],device)
    pi=build_model(pi_cbm,data,selected,config,device)
    pi.generator.load_state_dict(saved['generator']);pi.eval().requires_grad_(False)
    baseline.eval().requires_grad_(False)
    if backbone_name=='resnet18':
        backbone,teacher,teacher_text=load_encoders(device,names)
    else:
        from pi_cbm.models.backbones import load_backbone
        backbone=load_backbone({'name':backbone_name,'precision':'float32'},device)
        backbone.eval().requires_grad_(False)
        if backbone.feature_dim!=feature_dim:raise ValueError('Frozen backbone dimension mismatches clean cache')
    encoder_digest=hashlib.sha256()
    for state_key,value in backbone.state_dict().items():
        encoder_digest.update(state_key.encode());encoder_digest.update(value.detach().cpu().numpy().tobytes())
    encoder_sha=encoder_digest.hexdigest()
    feature_provenance=json.loads((source/'manifest.json').read_text())
    if encoder_sha!=feature_provenance['backbone_sha256']:
        raise ValueError('Loaded corruption encoder weights mismatch original clean cache SHA256')
    print(json.dumps({'dataset':dataset,'backbone':folder,'encoder_sha256':encoder_sha,
                      'clean_cache_sha256_verified':True,'encoding_batch':batch}),flush=True)
    clip_norm=backbone_name=='clip_RN50'
    mean=torch.tensor([.48145466,.4578275,.40821073] if clip_norm else [.485,.456,.406],device=device)[None,:,None,None]
    std=torch.tensor([.26862954,.26130258,.27577711] if clip_norm else [.229,.224,.225],device=device)[None,:,None,None]
    report={'dataset':dataset,'backbone':folder,'backbone_name':backbone_name,'feature_dim':feature_dim,'encoding_batch':batch,'encoder_sha256':encoder_sha,
            'clean_cache_sha256_verified':True,
            'feature_provenance':json.loads((source/'manifest.json').read_text()),'checkpoint_sha256':sha(out/'models.pt'),'training_protocol':config['data']['benchmark_train_views'],
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
                xs=np.empty((10000,feature_dim),dtype='float32')
                if backbone_name=='resnet18':
                    ys=np.empty((10000,len(names)),dtype='float32')
                    for start,x,y in encode(raw,backbone,teacher,teacher_text,device,batch):
                        xs[start:start+len(x)]=x;ys[start:start+len(x)]=y
                else:
                    # Exactly the same full-bank frozen CLIP teacher for the same official images.
                    teacher_cache=base/'rn18_imagenet/corruptions'/f'test_{key}.npz'
                    with np.load(teacher_cache) as shared:
                        if not np.array_equal(shared['ids'],ids) or not np.array_equal(shared['labels'],original['labels']):
                            raise ValueError('Shared same-image corruption teacher IDs/labels mismatch')
                        ys=shared['teacher'].copy()
                    with torch.no_grad():
                        for start in range(0,len(raw),batch):
                            image=resize224(raw[start:start+batch],device)
                            x=backbone((image-mean)/std).flatten(1).float().cpu().numpy()
                            xs[start:start+len(x)]=x
                np.savez_compressed(cache_path,features=xs,teacher=ys,labels=original['labels'],ids=ids,
                    targets=np.zeros_like(ys),known=np.zeros_like(ys,dtype=bool),concept_names=np.asarray(names))
                cache_path.with_suffix('.metadata.json').write_text(json.dumps({
                    'backbone':folder,'encoder_sha256':encoder_sha,'feature_dim':feature_dim,'encoding_batch':batch,
                    'normalization':'CLIP RGB mean/std' if clip_norm else 'ImageNet RGB mean/std',
                    'preprocess':'native uint8 /255; bicubic antialias resize224 align_corners=False; clamp[0,1]',
                    'teacher_source':str(base/'rn18_imagenet/corruptions'/f'test_{key}.npz'),
                    'clean_cache_sha256_verified':True},indent=2))
            inputs=np.load(cache_path)
            if inputs['features'].shape!=(10000,feature_dim) or not np.array_equal(inputs['ids'],ids):
                raise ValueError('Corruption cache backbone dimension or canonical ID mismatch')
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
                'severity':severity,'images':10000,'feature_dim':feature_dim,'backbone':folder,
                'teacher_source':str(base/'rn18_imagenet/corruptions'/f'test_{key}.npz'),
                'encoder_sha256':encoder_sha,'cache_encoding':json.loads(cache_path.with_suffix('.metadata.json').read_text())
                if cache_path.with_suffix('.metadata.json').exists() else {'encoding_batch':None,'legacy_cache_metadata_unavailable':True}}
            (result_dir/'report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'dataset':dataset,'corruption':key,'test_images':10000,'completed':True}),flush=True)
    print(json.dumps({'dataset':dataset,'all_corruption_settings':12,'report':str(result_dir/'report.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True,choices=['cifar10','cifar100'])
    p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=128);p.add_argument('--run')
    p.add_argument('--backbone',default='resnet18',choices=['resnet18','resnet50','clip_RN50'])
    a=p.parse_args();torch.set_num_threads(4);run(a.dataset,a.device,a.batch,a.run,a.backbone)
