"""Fresh CIFAR global LF fits and official CIFAR-C evaluation, outside frozen core."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision import datasets

from .lf import upstream, fit_common, predict_features, export_npz
from .metrics import teacher_fidelity, classification_metrics, paired_concept_metrics

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = '/home/pengcheng/VLG-CBM/datasets'
CORRUPTIONS = ['gaussian_noise', 'gaussian_blur', 'contrast', 'jpeg_compression']


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): value.update(block)
    return value.hexdigest()


def resize224(raw, device):
    """Canonical float bicubic antialias resize shared by all CIFAR methods."""
    image = torch.from_numpy(np.array(raw)).to(device).float()/255
    return F.interpolate(image, size=(224,224), mode='bicubic', align_corners=False,
                         antialias=True).clamp(0,1)


def load_encoders(device, names):
    upstream()
    from pi_cbm.models.backbones import load_backbone
    backbone = load_backbone({'name':'resnet18','precision':'float32'}, device)
    teacher, _ = clip.load('/root/.cache/clip/ViT-B-16.pt', device=device, jit=False)
    teacher.eval().requires_grad_(False)
    with torch.no_grad():
        text = torch.cat([teacher.encode_text(tokens.to(device)).float()
                          for tokens in clip.tokenize(names).split(256)])
        text = F.normalize(text,dim=-1)
    return backbone, teacher, text


@torch.no_grad()
def encode(raw, backbone, teacher, text, device, batch=128):
    rn_mean = torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    rn_std = torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    clip_mean = torch.tensor([.48145466,.4578275,.40821073],device=device)[None,:,None,None]
    clip_std = torch.tensor([.26862954,.26130258,.27577711],device=device)[None,:,None,None]
    for start in range(0,len(raw),batch):
        images = resize224(raw[start:start+batch],device)
        x = backbone((images-rn_mean)/rn_std).flatten(1).float().cpu().numpy()
        z = F.normalize(teacher.encode_image((images-clip_mean)/clip_std).float(),dim=-1)
        yield start, x, (z@text.T).cpu().numpy()


def cache(dataset, device='cuda', batch=128):
    core = upstream()
    from pi_cbm.data import make_splits
    base = ROOT/'artifacts/common'/dataset
    cache_dir = base/'cache'; features_dir = base/'rn18_imagenet'; teacher_dir = base/'teacher'
    for directory in [cache_dir,features_dir,teacher_dir]:directory.mkdir(parents=True,exist_ok=True)
    concepts_path = core/f'configs/concepts/{dataset}_official.txt'
    names = concepts_path.read_text().splitlines()
    assert len(names)==len(set(names)) and all(names)
    config = yaml.safe_load((core/'configs/base.yaml').read_text())
    config['data'].update(dataset=dataset,root=DATA_ROOT,train_per_class=0,
                          val_per_class=500 if dataset=='cifar10' else 50,
                          test_per_class=0,split_seed=42)
    train,test,folds=make_splits(config)
    ids={name:np.asarray(values,dtype=np.int64)+(50000 if name=='test' else 0)
         for name,values in folds.items()}
    assert not set(ids['train'])&set(ids['val'])
    labels=np.asarray(train.targets+test.targets,dtype=np.int64)
    rgb_path=cache_dir/'rgb_original.npy'
    if not rgb_path.exists():np.save(rgb_path,np.concatenate([train.data,test.data]).transpose(0,3,1,2))
    np.save(cache_dir/'labels.npy',labels)
    (cache_dir/'concepts.json').write_text(json.dumps(names,indent=2))
    protocol={'dataset':dataset,'split_seed':42,'splits':{s:x.tolist() for s,x in ids.items()},
              'classes':train.classes,'class_count':len(train.classes),'candidate_concepts':len(names),
              'concepts_source':str(concepts_path),'concepts_sha256':sha(concepts_path),
              'ids':'train original rows 0..49999; official test original rows +50000',
              'rgb_native_shape':[60000,3,32,32],
              'preprocess':'uint8 native32 /255; torch bicubic antialias resize224 align_corners=False; clamp[0,1]; per-backbone normalization',
              'human_concept_annotations':False,'foreground_masks':'unavailable',
              'targets':'CLIP raw cosine is separate continuous teacher, never native concept truth',
              'train_protocol':'clean only; validation5k; official test10k excluded from model selection'}
    (cache_dir/'protocol.json').write_text(json.dumps(protocol,indent=2))
    finish=features_dir/'original_features_COMPLETE.json'
    if not finish.exists() or not (teacher_dir/'original_COMPLETE.json').exists():
        backbone,teacher,text=load_encoders(device,names)
        np.save(teacher_dir/'text.npy',text.cpu().numpy())
        rgb=np.load(rgb_path,mmap_mode='r')
        xs=np.lib.format.open_memmap(features_dir/'original_features.npy',mode='w+',dtype='float32',shape=(60000,512))
        ys=np.lib.format.open_memmap(teacher_dir/'original_teacher.npy',mode='w+',dtype='float32',shape=(60000,len(names)))
        for start,x,y in encode(rgb,backbone,teacher,text,device,batch):
            xs[start:start+len(x)]=x;ys[start:start+len(x)]=y
            if start%(batch*50)==0:print(json.dumps({'dataset':dataset,'stage':'clean_cache','images':start+len(x),'total':60000}),flush=True)
        xs.flush();ys.flush()
        state_digest=hashlib.sha256()
        for key,value in backbone.state_dict().items():
            state_digest.update(key.encode());state_digest.update(value.cpu().numpy().tobytes())
        finish.write_text(json.dumps({'images':60000,'backbone_sha256':state_digest.hexdigest()}))
        (teacher_dir/'original_COMPLETE.json').write_text(json.dumps({'images':60000,'concepts':len(names)}))
    features=np.load(features_dir/'original_features.npy',mmap_mode='r')
    teachers=np.load(teacher_dir/'original_teacher.npy',mmap_mode='r')
    for split,index in ids.items():
        np.savez_compressed(features_dir/f'{split}_original.npz',features=features[index],labels=labels[index],ids=index,
                            targets=np.zeros((len(index),len(names)),np.float32),
                            known=np.zeros((len(index),len(names)),bool),teacher=teachers[index],concept_names=np.asarray(names))
    manifest={**protocol,'backbone':'torchvision ResNet18 ImageNet1K_V1 independently frozen','backbone_name':'resnet18',
              'historical_checkpoint':False,'backbone_sha256':json.loads(finish.read_text())['backbone_sha256'],
              'rgb_original':str(rgb_path.resolve()),'rgb_sha256':sha(rgb_path),'feature_dim':512,
              'teacher':str(teacher_dir.resolve()),'protocol':str((cache_dir/'protocol.json').resolve())}
    (features_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (teacher_dir/'metadata.json').write_text(json.dumps({**protocol,'names':names,'phrases':names,
        'teacher':'CLIP ViT-B/16','teacher_checkpoint_sha256':sha('/root/.cache/clip/ViT-B-16.pt'),
        'text':'exact upstream candidate strings, no added template','image_source':str(rgb_path.resolve())},indent=2))
    print(json.dumps({'dataset':dataset,'cache_ready':str(features_dir),'candidateK':len(names)}),flush=True)


def teacher_metrics(prediction,teacher):
    return {**teacher_fidelity(prediction,teacher),
            'semantic_native_mAP':None,'human_annotations_available':False}


def cache_augmentation(dataset,device='cuda',batch=128):
    """Encode the exact independent-row Gaussian s1 RGB produced by PiNI adapter."""
    base=ROOT/'artifacts/common'/dataset;source=base/'rn18_imagenet';teacher_dir=base/'teacher';view='gaussian_noise_s1'
    if not (base/'cache'/f'{view}_COMPLETE.json').exists():raise ValueError('Shared artificial Gaussian s1 native RGB incomplete')
    teacher_meta=json.loads((teacher_dir/'metadata.json').read_text());names=teacher_meta['names']
    finish=source/f'{view}_features_COMPLETE.json'
    if not finish.exists() or not (teacher_dir/f'{view}_COMPLETE.json').exists():
        backbone,teacher,text=load_encoders(device,names)
        raw=np.load(base/'cache'/f'rgb_{view}.npy',mmap_mode='r')
        xs=np.lib.format.open_memmap(source/f'{view}_features.npy',mode='w+',dtype='float32',shape=(60000,512))
        ys=np.lib.format.open_memmap(teacher_dir/f'{view}_teacher.npy',mode='w+',dtype='float32',shape=(60000,len(names)))
        for start,x,y in encode(raw,backbone,teacher,text,device,batch):
            xs[start:start+len(x)]=x;ys[start:start+len(x)]=y
            if start%(batch*50)==0:print(json.dumps({'dataset':dataset,'stage':'augmentation_cache','images':start+len(x),'total':60000}),flush=True)
        xs.flush();ys.flush();info={'images':60000,'view':view,'artificial_corruption':True,
            'protocol':'CIFAR-C Gaussian severity1 formula sigma.04; RandomState(42+global_row); native HWC uint8'}
        finish.write_text(json.dumps(info));(teacher_dir/f'{view}_COMPLETE.json').write_text(json.dumps(info))
    xs=np.load(source/f'{view}_features.npy',mmap_mode='r');ys=np.load(teacher_dir/f'{view}_teacher.npy',mmap_mode='r')
    for split in ['train','val','test']:
        original=np.load(source/f'{split}_original.npz');index=original['ids']
        np.savez_compressed(source/f'{split}_{view}.npz',features=xs[index],teacher=ys[index],labels=original['labels'],ids=index,
            targets=original['targets'],known=original['known'],concept_names=original['concept_names'])
    print(json.dumps({'dataset':dataset,'augmentation_cache_complete':True}),flush=True)


def fit(dataset,device='cuda',seed=42,epochs=20,steps=1000,train_views='original',backbone='resnet18'):
    core=upstream()
    from pi_cbm.pipeline import build_model,baseline_from_state
    from pi_cbm.training.adapt import adapt
    from pi_cbm.randomness import seed_all
    folder={'resnet18':'rn18_imagenet','resnet50':'rn50_imagenet','clip_RN50':'clip_RN50'}[backbone]
    base=ROOT/'artifacts/common'/dataset;source=base/folder;teacher=base/'teacher'
    views=train_views.split(',')
    if views not in [['original'],['original','gaussian_noise_s1']]:raise ValueError('Unsupported CIFAR training views')
    out=ROOT/'artifacts/matched/lf'/dataset/folder
    if len(views)>1:out=out/'gaussian_noise_augmented'
    out=out/f'seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((teacher/'metadata.json').read_text());names=meta['names'];splits={};raw={}
    ys=np.load(teacher/'original_teacher.npy',mmap_mode='r')
    for s in ['train','val','test']:
        x=np.load(source/f'{s}_original.npz');raw[s]=x
        splits[s]={'features':torch.from_numpy(x['features'].copy()),'labels':torch.from_numpy(x['labels'].copy()),
                   'indices':torch.from_numpy(x['ids'].copy()),'teacher':torch.from_numpy(np.array(ys[x['ids']]))}
    canonical_splits={s:dict(x) for s,x in splits.items()}
    teacher_arrays={'original':ys}
    if len(views)>1:
        teacher_arrays['gaussian_noise_s1']=np.load(teacher/'gaussian_noise_s1_teacher.npy',mmap_mode='r')
        for s in ['train','val']:
            other=np.load(source/f'{s}_gaussian_noise_s1.npz');assert np.array_equal(other['ids'],raw[s]['ids'])
            augmented={'features':torch.from_numpy(other['features'].copy()),'labels':torch.from_numpy(other['labels'].copy()),
                'indices':torch.from_numpy(other['ids'].copy()),'teacher':torch.from_numpy(np.array(teacher_arrays['gaussian_noise_s1'][other['ids']]))}
            splits[s]={key:torch.cat([splits[s][key],augmented[key]]) for key in splits[s]}
    data={'splits':splits,'classes':meta['classes'],'concepts':names,
          'text':torch.from_numpy(np.load(teacher/'text.npy').copy())}
    config=yaml.safe_load((core/'configs/base.yaml').read_text())
    config['experiment'].update(name=f'matched_{dataset}_{backbone}',device=device,seed=seed,threads=4)
    config['data'].update(dataset=dataset,root=DATA_ROOT,concepts=str(core/f'configs/concepts/{dataset}_official.txt'),
                         train_per_class=0,val_per_class=500 if dataset=='cifar10' else 50,test_per_class=0,split_seed=42,
                         input_mode='cache',cache_internal=False,cache_images=False,benchmark_train_views=views)
    config['backbone']['name']=backbone;config['teacher']['name']='ViT-B/16'
    config['baseline'].update(include_adapter_val=False,validation_split='val',activation_cutoff=.25,
                              interpretability_cutoff=.45,projection_steps=steps)
    config['baseline']['head'].update(steps=steps,**{'lambda':.0007})
    config['training'].update(epochs=epochs,stage='finetune_projection',batch_size=256,lr=.0003,projection_lr=.00001)
    config['injection'].update(site='target_global',point='global')
    config['paths'].update(cache=str(out/'cache'),runs=str(out))
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    baseline,selected,baseline_fit=fit_common(data,config,device=device)
    if not len(selected):raise RuntimeError('No candidate retained under official thresholds; requires explicit val-only rerun config')
    baseline_state={k:v.cpu().clone() for k,v in baseline.cbm.state_dict().items()}
    cbm=baseline_from_state(baseline_state,device);seed_all(seed)
    pi=build_model(cbm,data,selected,config,device)
    with torch.no_grad():
        clean={s:torch.cat([cbm.concepts(batch.to(device)).cpu() for batch in x['features'].split(256)]) for s,x in splits.items()}
    history,refit=adapt(pi,data,clean,selected,config,device)
    selected_names=[names[i] for i in selected.tolist()]
    torch.save({'baseline':baseline_state,'pi_cbm':{k:v.cpu() for k,v in pi.cbm.state_dict().items()},
                'generator':{k:v.cpu() for k,v in pi.generator.state_dict().items()},'selected':selected,
                'config':config,'concepts':selected_names},out/'models.pt')
    report={'dataset':dataset,'backbone':folder,'training_seed':seed,'train_views':views,
            'underlying_subject_counts':{s:len(x['indices']) for s,x in canonical_splits.items()},
            'train_examples':len(splits['train']['labels']),'val_examples':len(splits['val']['labels']),
            'data_protocol':'Precomputed complete CIFAR; original train45000/val5000, official test10000; split_seed42; augmentation repeats only same-fold subject IDs',
            'optimization_device':device,'candidateK':len(names),'candidate_concepts':len(names),'bank_concepts':len(names),
            'selected_concepts':len(selected),'selected_indices':selected.tolist(),'selected_names':selected_names,
            'bank_coverage':len(selected)/len(names),'test_selection':False,
            'human_concept_annotations':False,'concept_evaluation':'Frozen CLIP continuous teacher Pearson/cos_cubed; no native mAP',
            'feature_provenance':json.loads((source/'manifest.json').read_text()),'teacher_provenance':meta,
            'baseline_fit':baseline_fit,'adapter_history':history,'adapter_refit':refit,
            'inference':{'lfcbm_draws':1,'lfpi_draws':1,'seed':3000},'metrics':{}}
    single=out/'single_draw';single.mkdir(exist_ok=True)
    single_metadata={'inference_draws':1,'inference_seed':3000,'batch_size':256,'device':device,
                     'checkpoint_sha256':sha(out/'models.pt'),'selected_indices':selected.tolist(),
                     'concept_names':selected_names,'source':str(source),'human_annotations_available':False,'input_mapping':{}}
    for method,model in [('lfcbm',baseline),('lfpi_global',pi)]:
        for split in canonical_splits:
            for view in views:
                inputs=np.load(source/f'{split}_{view}.npz')
                x={**canonical_splits[split],'features':torch.from_numpy(inputs['features'].copy())}
                pred=predict_features(model,x['features'],draws=1,seed=3000)
                destination=(single if method=='lfpi_global' else out)/f'{method}_{split}_{view}.npz'
                export_npz(destination,pred,x,selected_names,targets=inputs['targets'][:,selected.numpy()],known=inputs['known'][:,selected.numpy()])
                if method=='lfpi_global':
                    export_npz(out/f'{method}_{split}_{view}.npz',pred,x,selected_names,
                               targets=inputs['targets'][:,selected.numpy()],known=inputs['known'][:,selected.numpy()])
                    single_metadata['input_mapping'][split+'/'+view]={'source_file':str(source/f'{split}_{view}.npz'),
                        'source_sha256':sha(source/f'{split}_{view}.npz'),'ids_sha256':hashlib.sha256(x['indices'].numpy().tobytes()).hexdigest(),
                        'images':len(x['features']),'output_file':str(destination)}
                if split!='train':report['metrics'][f'{method}/{split}/{view}']={
                    'classification':classification_metrics(pred['logits'].numpy(),x['labels'].numpy()),
                    'CLIP_teacher':teacher_metrics(pred['concepts'].numpy(),teacher_arrays[view][x['indices'].numpy()][:,selected.numpy()])}
    single_metadata['metrics']={k:v for k,v in report['metrics'].items() if k.startswith('lfpi_global/')}
    (single/'metadata.json').write_text(json.dumps(single_metadata,indent=2))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'dataset':dataset,'fit_complete':str(out),'candidateK':len(names),'retainedK':len(selected)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['cache','cache-augmentation','fit']);parser.add_argument('--dataset',required=True,choices=['cifar10','cifar100'])
    parser.add_argument('--device',default='cuda');parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--train-views',default='original',choices=['original','original,gaussian_noise_s1'])
    parser.add_argument('--backbone',default='resnet18',choices=['resnet18','resnet50','clip_RN50'])
    args=parser.parse_args();torch.set_num_threads(4)
    if args.action!='fit' and args.backbone!='resnet18':parser.error('This cache command encodes RN18; wider caches use benchmark.cifar_backbones')
    if args.action=='cache':cache(args.dataset,args.device,args.batch)
    elif args.action=='cache-augmentation':cache_augmentation(args.dataset,args.device,args.batch)
    else:fit(args.dataset,args.device,args.seed,args.epochs,args.steps,args.train_views,args.backbone)


if __name__=='__main__':main()
