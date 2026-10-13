"""Thin wide PiNI evaluation adapter reusing frozen CIFAR-C inference/regions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import pini_cifar_corrupt as shared
from .pini_cifar_wide import Experiment as WideExperiment
from .pini_matched import ROOT


class CommonRouter:
    """Route the shared evaluator's backbone-cache name to the actual wide cache."""
    def __init__(self,root,folder):self.root=root;self.folder=folder
    def __truediv__(self,key):
        parts=Path(key).parts
        return self.root.joinpath(self.folder,*parts[1:]) if parts[0]=='rn18_imagenet' else self.root/key


class Experiment(WideExperiment):
    def __init__(self,dataset,backbone):
        super().__init__(dataset,backbone)
        self.common_root=self.common;self.common=CommonRouter(self.common_root,self.folder)
    def backbone(self):
        net=super().backbone()
        if not hasattr(net,'weight_sha256_aliases'):
            net.weight_sha256_aliases={'original_CLIP_full_encoder':net.weight_sha256}
        return net


def context(dataset,backbone,final_required):
    primary=ROOT/'artifacts/pini_cifar_c'/dataset/'rn18_imagenet/seed42'
    assert (primary/'CIFAR_C_COMPLETE.json').exists()
    original_manifest=json.loads((primary/'input_manifest.json').read_text())
    assert shared.sha(primary/'input_manifest.json')==json.loads((primary/'CIFAR_C_COMPLETE.json').read_text())['source_manifest_sha256']
    e=Experiment(dataset,backbone);out=ROOT/'artifacts/pini_cifar_c'/dataset/backbone/'seed42'
    protocol=json.loads((e.original/'cache/protocol.json').read_text());columns=np.load(e.original/'cache/retained_columns.npy');ids=np.asarray(protocol['splits']['test'],np.int64);labels=np.load(e.original/'cache/labels.npy',mmap_mode='r')[ids]
    assert len(ids)==10000 and np.array_equal(ids,np.asarray(original_manifest['ids']))
    assert np.array_equal(columns,np.asarray(original_manifest['concept_indices']))
    official=ROOT/'artifacts/common/cifar_c'/('CIFAR-10-C' if dataset=='cifar10' else 'CIFAR-100-C')
    files={'producer':Path(__file__),'shared_evaluator':Path(shared.__file__),'wide_adapter':ROOT/'benchmark/pini_cifar_wide.py','primary_input_manifest':primary/'input_manifest.json','primary_complete':primary/'CIFAR_C_COMPLETE.json','feature_manifest':e.feature_root/'manifest.json'}
    for name in ['core_forward','core_grounding','renderer']:
        record=original_manifest['files'][name];files[name]=Path(record['path'])
        assert shared.sha(files[name])==record['sha256']
    pending=[]
    if not (e.root/'CIFAR_COMPLETE.json').exists():pending.append(str(e.root/'CIFAR_COMPLETE.json'))
    else:files['trained_complete']=e.root/'CIFAR_COMPLETE.json'
    for arm in ['mean','noise']:
        p=e.root/'pair_explore/selection'/arm/'best.pt'
        if p.exists():files['checkpoint_'+arm]=p
        else:pending.append(str(p))
    for name,p in [('initialization_chain',e.root/'initialization_chain.json'),('private_original_features',e.root/'cache/original_FEATURE_COMPLETE.json'),('private_changed_features',e.root/'cache/changed_FEATURE_COMPLETE.json')]:
        if p.exists():files[name]=p
        else:pending.append(str(p))
    audit_batches=[]
    for kind in shared.CORRUPTIONS:
        files['official_'+kind]=official/(kind+'.npy')
        assert shared.sha(files['official_'+kind])==original_manifest['files']['official_'+kind]['sha256']
        for severity in [1,3,5]:
            key=f'{kind}_{severity}';region=primary/'regions'/key;receipt=json.loads((region/'COMPLETE.json').read_text())
            assert receipt['images']==10000 and receipt['producer_sha256']==original_manifest['files']['producer']['sha256']
            assert receipt['ordered_ids_sha256']==hashlib.sha256(ids.tobytes()).hexdigest()
            for name in ['features.npy','maps.npy','valid.npy','boxes.npy','scores.npy','COMPLETE.json']:
                files['region_'+key+'_'+name]=region/name
            p=e.feature_root/'corruptions'/f'test_{key}.npz';files['common_'+key]=p
            metadata=p.with_suffix('.metadata.json');files['common_metadata_'+key]=metadata
            audit_batches.append(int(json.loads(metadata.read_text())['encoding_batch']))
            with np.load(p) as z:
                assert np.array_equal(z['ids'],ids) and np.array_equal(z['labels'],labels) and z['features'].shape==(10000,e.feature_dim)
                assert z['teacher'].shape[1]>int(columns.max()) and not z['known'].any()
    assert len(set(audit_batches))==1,'Cache encoding batches differ; use per-setting audit rather than mixing backend shapes'
    if final_required:assert not pending,'Final wide training/checkpoints not complete: '+repr(pending)
    binding=dict(dataset=dataset,backbone=backbone,files={k:dict(path=str(p.resolve()),sha256=shared.sha(p)) for k,p in files.items()},ids=ids.tolist(),concept_indices=columns.tolist(),DINO_recomputed=False,regions_source=str(primary/'regions'),pixel_protocol='unchanged Native32IntensityReader CUDA q0*255 then original core /255=q1',draws=1,seed=4242,batch=64,feature_audit_batch=audit_batches[0],feature_audit_matches_original_cache_batch=True,training_performed=False,final_training_pending=pending)
    return e,out,official,ids,labels,columns,binding,primary


def audit(e,out,ids,columns,binding,primary):
    torch.manual_seed(42);cbm=e.model().eval();model_source='core inherited model interface with seed42 disposable weights; final checkpoints still pending'
    if not binding['final_training_pending']:
        state=torch.load(e.root/'pair_explore/selection/mean/best.pt',map_location='cpu',weights_only=True);cbm.load_state_dict(state['cbm']);model_source='actual final mean pair checkpoint'
    with np.load(e.feature_root/'corruptions/test_gaussian_noise_1.npz') as z:features=torch.from_numpy(z['features'][:3].copy())[:,:,None,None]
    with torch.no_grad():
        logits,c=cbm(features);reconstructed=F.linear((c-cbm.concept_center)/cbm.concept_scale,cbm.classifier.weight,cbm.classifier.bias)
    assert torch.equal(logits,reconstructed) and logits.shape==(3,e.classes) and c.shape==(3,e.k)
    region=primary/'regions/gaussian_noise_1';assert np.load(region/'features.npy',mmap_mode='r').shape==(10000,256,14,14);assert np.load(region/'maps.npy',mmap_mode='r').shape==(10000,8,14,14)
    net=e.backbone().cpu();manifest=json.loads((e.feature_root/'manifest.json').read_text());assert manifest['backbone_sha256'] in net.weight_sha256_aliases.values()
    record=dict(dataset=e.name,backbone=e.backbone_name,CPU_interface_verified=True,GPU_started=False,full_settings=12,images_per_setting=10000,feature_dim=e.feature_dim,head_reconstruction_max_abs=0.,model_source=model_source,final_training_pending=binding['final_training_pending'],binding=binding)
    shared.atomic(ROOT/'artifacts/audit'/f'{e.name}_pini_{e.backbone_name}_official_c_CPU.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='binding'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['audit','evaluate']);p.add_argument('--dataset',choices=['cifar10','cifar100'],default='cifar10');p.add_argument('--backbone',choices=['resnet50','clip_rn50'],required=True);a=p.parse_args();torch.set_num_threads(3)
    e,out,official,ids,labels,columns,binding,primary=context(a.dataset,a.backbone,a.action=='evaluate')
    if a.action=='audit':audit(e,out,ids,columns,binding,primary);return
    shared.bind(out,binding)
    # Independent local directories: evaluator precision receipts cannot modify
    # the original RN18 region directory; only immutable DINO arrays are linked.
    for key in [f'{k}_{s}' for k in shared.CORRUPTIONS for s in [1,3,5]]:
        folder=out/'regions'/key;folder.mkdir(parents=True,exist_ok=True)
        for name in ['features.npy','maps.npy','valid.npy','boxes.npy','scores.npy','COMPLETE.json']:
            target=folder/name;source=primary/'regions'/key/name
            if not target.exists():target.symlink_to(source.resolve())
            assert target.resolve()==source.resolve()
    shared.evaluate(e,out,official,ids,labels,columns,64,binding['feature_audit_batch'])

if __name__=='__main__':main()
