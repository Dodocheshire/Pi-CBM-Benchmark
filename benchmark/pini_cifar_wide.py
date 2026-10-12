"""Channel-only CIFAR PiNI adapters; frozen core and shared canonical inputs."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .pini_cifar import Experiment as BaseExperiment,export
from .pini_matched import ROOT,MatchedExperiment,ImageNetSpatialBackbone,CLIPRN50SpatialBackbone,initialize,write,digest
from .lf_cifar import resize224


class Experiment(BaseExperiment):
    def __init__(self,dataset,backbone):
        super().__init__(dataset)
        self.original=self.root;self.backbone_name=backbone
        self.feature_dim=2048 if backbone=='resnet50' else 1024
        self.folder='rn50_imagenet' if backbone=='resnet50' else 'clip_RN50'
        self.feature_root=self.common/self.folder
        self.root=ROOT/'artifacts/pini_matched'/backbone/dataset
    def model(self):return MatchedExperiment.model(self)
    def backbone(self):return ImageNetSpatialBackbone('resnet50') if self.backbone_name=='resnet50' else CLIPRN50SpatialBackbone()


def link(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source.resolve())
    assert target.resolve()==source.resolve()


def prepare_links(e):
    assert (e.original/'CIFAR_COMPLETE.json').exists()
    e.root.mkdir(parents=True,exist_ok=True)
    for name in ['labels.npy','attrs.npy','rgb_original.npy','rgb_changed.npy','foreground.npy','concepts.json','retained_columns.npy','thresholds.npy','BACKGROUND_COMPLETE.json','supervision_audit.json']:
        link(e.original/'cache'/name,e.root/'cache'/name)
    for view in ['original','changed']:
        assert (e.original/'regions'/view/'COMPLETE.json').exists()
        link(e.original/'regions'/view,e.root/'regions'/view)
    protocol=json.loads((e.original/'cache/protocol.json').read_text());manifest=json.loads((e.feature_root/'manifest.json').read_text())
    assert protocol['splits']==manifest['splits']
    protocol.update(backbone_name=e.backbone_name,feature_channels=e.feature_dim,feature_layout='native layer4 spatial; pooled only for frozen simple cache' if e.backbone_name=='resnet50' else 'original CLIP projected attention-pool global singleton',backbone_label_supervision='ImageNet1K only' if e.backbone_name=='resnet50' else 'original CLIP image/text pretraining',task_fit_sample_ids=[],wide_adaptation='channel dimensions only; inherited SimpleCBM and frozen literal allocation-adapted core simple',primary_pixel_source=str(e.original/'cache'),primary_regions_source=str(e.original/'regions'),human_concept_annotations=False,fit_split='train only',selection_split='val only')
    protocol['primary_RN18_feature_audit']=protocol.pop('feature_audit',None)
    protocol['feature_audit']='wide full-cache audit required before simple training'
    write(e.root/'cache/protocol.json',protocol)
    return protocol,manifest


def audit(e):
    protocol,manifest=prepare_links(e);net=e.backbone().cpu().eval().requires_grad_(False)
    aliases=getattr(net,'weight_sha256_aliases',{'original_CLIP_full_encoder':net.weight_sha256})
    matched=[k for k,v in aliases.items() if v==manifest['backbone_sha256']];assert matched,'Pretraining weight SHA mismatch'
    rows=np.asarray([protocol['splits'][s][0] for s in ['train','val','test']],np.int64);records=[]
    from benchmark.lf import upstream
    upstream()
    from pi_cbm.models.backbones import load_backbone
    reference_encoder=load_backbone({'name':'resnet50' if e.backbone_name=='resnet50' else 'clip_RN50','precision':'float32'},'cpu').eval().requires_grad_(False)
    for view,canonical in [('original','original'),('changed','gaussian_noise_s1')]:
        intensity=np.load(e.root/'cache'/('rgb_'+view+'.npy'),mmap_mode='r');raw=torch.from_numpy(np.array(intensity[rows])).float()/255
        if e.backbone_name=='resnet50':
            from spatial_noise.cub_aligned import normalize
            reference_input=normalize(raw)
        else:
            reference_input=(raw-raw.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None])/raw.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
        with torch.no_grad():
            features=net(raw).mean((2,3)).numpy();independent=reference_encoder(reference_input).numpy()
        assert np.allclose(features,independent,atol=1e-5,rtol=1e-5),'Independent LF encoder definition mismatch'
        assert features.shape==(3,e.feature_dim)
        record=dict(view=view,ids=rows.tolist(),shape=list(features.shape),independent_LF_same_q1_CPU_max_abs=float(np.max(np.abs(features-independent))),pixel_array_samefile=os.path.samefile(e.root/'cache'/('rgb_'+view+'.npy'),e.original/'cache'/('rgb_'+view+'.npy')),exact_features_claimed=False,CPU_vs_GPU_backend=True)
        existing=e.feature_root/(canonical+'_features.npy')
        if existing.exists():
            reference=np.load(existing,mmap_mode='r')[rows];record.update(relative_l2_vs_LF_q0_GPU=float(np.linalg.norm(features.astype('float64')-reference.astype('float64'))/np.linalg.norm(reference.astype('float64'))),max_abs_vs_LF_q0_GPU=float(np.max(np.abs(features-reference))))
        else:record.update(shared_changed_features_available=False,reference='independent LF original encoder definition on exact shared q1; full q0 GPU cache audit deferred to prepare')
        records.append(record)
        for split in ['train','val','test']:
            shared=e.feature_root/(split+'_'+canonical+'.npz')
            if not shared.exists():shared=e.feature_root/(split+'_original.npz')
            with np.load(shared) as cache:
                ids=np.asarray(protocol['splits'][split],np.int64);assert np.array_equal(ids,cache['ids'])
                assert np.array_equal(np.load(e.root/'cache/labels.npy',mmap_mode='r')[ids],cache['labels'])
                assert cache['features'].shape==(len(ids),e.feature_dim) and not cache['known'].any()
    del reference_encoder
    torch.manual_seed(42);model=e.model();x=raw[:1].detach().requires_grad_(True);y,c=model(net(x));targets=torch.from_numpy(np.array(np.load(e.root/'cache/attrs.npy',mmap_mode='r')[rows[:1]])).float();labels=torch.from_numpy(np.array(np.load(e.root/'cache/labels.npy',mmap_mode='r')[rows[:1]])).long()
    # Exercise inherited classification/concept interfaces and input gradient;
    # this is a disposable shape/backward audit, never a training replacement.
    loss=F.cross_entropy(y,labels)+F.binary_cross_entropy_with_logits(c,targets);loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 and all(p.grad is None for p in net.parameters())
    from spatial_noise import rn18_datasets,cub_aligned,cub_noise,noise_pair
    receipt=dict(dataset=e.name,backbone=e.backbone_name,feature_dim=e.feature_dim,weight_sha256=manifest['backbone_sha256'],hash_schema=matched,views=records,input_gradient_finite=True,input_gradient_abs_sum=float(x.grad.abs().sum()),backbone_frozen=True,core_model_type=type(model).__name__,concept_weight_shape=list(model.concepts.weight.shape),classifier_weight_shape=list(model.classifier.weight.shape),human_concept_annotations=False,training_started=False,GPU_used=False,core_functions_sha256={name:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for name,fn in [('simple',rn18_datasets.simple),('concept_logits',cub_aligned.SimpleCBM.concept_logits),('classify',cub_aligned.SimpleCBM.classify),('fit_classifier',cub_aligned.fit_classifier),('forward',cub_noise.forward),('warmup_train',cub_noise.train),('pair_train',noise_pair.train)]},training=dict(simple_steps=1000,warmup_epochs=10,warmup_batch=64,pair_epochs=5,pairs=32,core_seed=42,loss_or_prior_modified=False),producer_sha256=digest(__file__))
    receipt['source_sha256']={name:digest(e.original/'cache'/name) for name in ['concepts.json','retained_columns.npy','thresholds.npy','attrs.npy','labels.npy','protocol.json','original_RGB_FEATURE_COMPLETE.json','changed_RGB_FEATURE_COMPLETE.json']}
    receipt['region_complete_sha256']={view:digest(e.original/'regions'/view/'COMPLETE.json') for view in ['original','changed']}
    receipt['common_manifest_sha256']=digest(e.feature_root/'manifest.json')
    write(ROOT/'artifacts/audit'/f'{e.name}_pini_{e.backbone_name}_CPU.json',receipt);print(json.dumps(receipt),flush=True)


def prepare_features(e,batch=128):
    _,manifest=prepare_links(e);net=e.backbone().cuda().eval().requires_grad_(False);aliases=getattr(net,'weight_sha256_aliases',{'original_CLIP_full_encoder':net.weight_sha256});assert manifest['backbone_sha256'] in aliases.values()
    with torch.no_grad():
        for view,canonical in [('original','original'),('changed','gaussian_noise_s1')]:
            receipt=e.root/'cache'/(view+'_FEATURE_COMPLETE.json')
            if receipt.exists():continue
            source=np.load(e.root/'cache'/('rgb_'+view+'.npy'),mmap_mode='r');reference_path=e.feature_root/(canonical+'_features.npy');ref=np.load(reference_path,mmap_mode='r') if reference_path.exists() else None;native=np.load(e.common/'cache'/('rgb_'+canonical+'.npy'),mmap_mode='r') if ref is None else None;path=e.root/'cache'/('features_'+view+'.npy');feature=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(source),e.feature_dim,1,1));err2=0.;ref2=0.;maximum=0.;started=time.time()
            for start in range(0,len(source),batch):
                x=torch.as_tensor(np.array(source[start:start+batch]),device='cuda').float()/255;pooled=net(x).mean((2,3)).cpu().numpy();feature[start:start+len(pooled),:,0,0]=pooled
                reference=ref[start:start+len(pooled)] if ref is not None else net(resize224(native[start:start+batch],'cuda')).mean((2,3)).cpu().numpy()
                delta=pooled.astype('float64')-reference.astype('float64');err2+=float(np.square(delta).sum());ref2+=float(np.square(reference.astype('float64')).sum());maximum=max(maximum,float(np.abs(delta).max()))
            feature.flush();relative=(err2/ref2)**.5;assert relative<1e-3,'Approved q1-vs-LF-q0 precision gate failed'
            write(receipt,dict(shape=list(feature.shape),relative_l2=relative,max_abs=maximum,backbone_sha256=manifest['backbone_sha256'],shared_reference_sha256=digest(reference_path) if ref is not None else None,reference='existing shared LF q0 fullcache' if ref is not None else 'fresh q0 canonical GPU encoding; independent encoder definition already CPU audited',feature_sha256=digest(path),pixel_source_sha256=digest(e.root/'cache'/('rgb_'+view+'.npy')),batch=batch,seconds=time.time()-started,exact_feature_identity_claimed=False,approved_relative_l2_gate=1e-3))
    protocol=json.loads((e.root/'cache/protocol.json').read_text());protocol['feature_audit']={v:json.loads((e.root/'cache'/(v+'_FEATURE_COMPLETE.json')).read_text()) for v in ['original','changed']};write(e.root/'cache/protocol.json',protocol)
    del net;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['audit','prepare','train']);p.add_argument('--dataset',choices=['cifar10','cifar100'],required=True);p.add_argument('--backbone',choices=['resnet50','clip_rn50'],required=True);a=p.parse_args();torch.set_num_threads(4);e=Experiment(a.dataset,a.backbone)
    if a.action=='audit':audit(e);return
    prepare_features(e)
    if a.action=='prepare':return
    initialize(e,1000,10,64)
    from spatial_noise.noise_pair import train
    for method in ['mean','noise']:
        train(e,method,'selection',epochs=5,pairs=32);export(e,method)
    write(e.root/'CIFAR_COMPLETE.json',dict(dataset=e.name,backbone=e.backbone_name,methods=['mean','noise'],seed=42,human_concept_annotations=False,train_only=True))

if __name__=='__main__':main()
