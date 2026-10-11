"""Train frozen PiNI cores with ImageNet RN18 and train-only shared splits.

All algorithm, losses, generator priors, selection rules and checkpoint keys are
owned by the unmodified submodule. This wrapper supplies RGB/cache/backbone,
constructs the core's real initialization chain, and exports independent test.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import clip as official_clip  # Load installed official CLIP before core path registration.
from benchmark.pini import import_core, CORE
from benchmark.metrics import concept_metrics,classification_metrics,paired_concept_metrics,nec

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/pengcheng/Pi-CBM/PiNI/artifacts')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2))


class ImageNetSpatialBackbone(nn.Module):
    def __init__(self,name='resnet18'):
        super().__init__()
        from torchvision.models import resnet18,resnet50,ResNet18_Weights,ResNet50_Weights
        if name=='resnet18':net=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif name=='resnet50':net=resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:raise ValueError(name)
        self.output_channels=net.fc.in_features
        # Both cache generations use the same weights but different key schemas.
        fingerprint_model=nn.Module()
        fingerprint_model.prefix=nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool,net.layer1,net.layer2,net.layer3)
        fingerprint_model.suffix=nn.Sequential(net.layer4,net.avgpool,nn.Flatten(1))
        self.weight_sha256_aliases={}
        for schema,model in [('torchvision_full_including_fc',net),('lf_core_prefix_suffix_without_fc',fingerprint_model)]:
            h=hashlib.sha256()
            for key,value in model.state_dict().items():
                h.update(key.encode());h.update(value.numpy().tobytes())
            self.weight_sha256_aliases[schema]=h.hexdigest()
        self.weight_sha256=self.weight_sha256_aliases['torchvision_full_including_fc' if name=='resnet18' else 'lf_core_prefix_suffix_without_fc']
        self.spatial=nn.Sequential(*list(net.children())[:-2])
        self.eval().requires_grad_(False)

    def forward(self,rgb):
        from spatial_noise.cub_aligned import normalize
        return self.spatial(normalize(rgb))


class CLIPRN50SpatialBackbone(nn.Module):
    """Original CLIP attention-pool/projected embedding as global singleton.

    The unmodified SimpleCBM spatial average of [B,1024,1,1] is the original
    embedding. This is explicitly a global singleton, not a real spatial map.
    """
    def __init__(self):
        super().__init__()
        from benchmark.lf import upstream
        upstream()
        from pi_cbm.models.backbones import load_backbone
        self.encoder=load_backbone({'name':'clip_RN50','precision':'float32'},'cpu')
        h=hashlib.sha256()
        for key,value in self.encoder.state_dict().items():
            h.update(key.encode());h.update(value.numpy().tobytes())
        self.weight_sha256=h.hexdigest();self.output_channels=1024
        self.eval().requires_grad_(False)

    def forward(self,rgb):
        mean=rgb.new_tensor([.48145466,.4578275,.40821073])[None,:,None,None]
        std=rgb.new_tensor([.26862954,.26130258,.27577711])[None,:,None,None]
        return self.encoder((rgb-mean)/std)[:,:,None,None]


class MatchedExperiment:
    def __init__(self,dataset,output,backbone='resnet18'):
        import_core()
        from spatial_noise.rn18_datasets import RIVAL_PHRASES
        from spatial_noise.grounding import BIRD_PHRASES
        self.name=dataset;self.root=Path(output)/dataset
        self.backbone_name=backbone;self.feature_dim=512 if backbone=='resnet18' else (2048 if backbone=='resnet50' else 1024)
        self.original=SOURCE/(dataset+'_aligned_rn18')
        self.common=ROOT/'artifacts/common'/dataset/('rn18_imagenet' if backbone=='resnet18' else ('rn50_imagenet' if backbone=='resnet50' else 'clip_RN50'))
        self.k=18 if dataset=='rival10' else 112
        self.classes=10 if dataset=='rival10' else (2 if dataset=='waterbirds' else 200)
        self.phrases=RIVAL_PHRASES if dataset=='rival10' else BIRD_PHRASES
        self.detector_precision='fp16' if dataset=='rival10' else 'fp32'

    def model(self):
        from spatial_noise.cub_aligned import SimpleCBM
        if self.feature_dim==512:return SimpleCBM(self.k,self.classes)
        # Constructor is the original SimpleCBM allocation sequence with only
        # its feature-channel512 replaced by actual D. Preserve seed/RNG order.
        model=SimpleCBM.__new__(SimpleCBM);nn.Module.__init__(model)
        model.register_buffer('feature_center',torch.zeros(self.feature_dim))
        model.register_buffer('feature_scale',torch.ones(self.feature_dim))
        model.concepts=nn.Linear(self.feature_dim,self.k)
        model.register_buffer('concept_center',torch.zeros(self.k))
        model.register_buffer('concept_scale',torch.ones(self.k))
        model.classifier=nn.Linear(self.k,self.classes)
        return model

    def backbone(self):
        return CLIPRN50SpatialBackbone() if self.backbone_name=='clip_rn50' else ImageNetSpatialBackbone(self.backbone_name)

    def load(self):
        cache=self.root/'cache';protocol=json.loads((cache/'protocol.json').read_text())
        splits={key:np.asarray(value,np.int64) for key,value in protocol['splits'].items()}
        names=['labels','attrs','rgb_original','foreground','features_original']
        if (cache/'background_eligible.npy').exists():names.append('background_eligible')
        arrays={key:np.load(cache/(key+'.npy'),mmap_mode='r') for key in names}
        return arrays,splits,protocol


def prepare_waterbirds_shared():
    """Verify identical ordered CUB RGB/splits and reuse ImageNet global bank."""
    source=SOURCE/'waterbirds_aligned_rn18/cache'
    cubsource=SOURCE/'cub_aligned_rn18/cache'
    common=ROOT/'artifacts/common/waterbirds/rn18_imagenet'
    cubcommon=ROOT/'artifacts/common/cub/rn18_imagenet'
    common.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((source/'protocol.json').read_text())
    cubprotocol=json.loads((cubsource/'protocol.json').read_text())
    assert protocol['splits']==cubprotocol['splits']
    assert np.array_equal(np.load(source/'native_ids.npy'),np.load(cubsource/'native_ids.npy'))
    labels=np.load(source/'labels.npy');assert set(np.unique(labels))=={0,1}
    audit={}
    for view in ['original','changed']:
        a=source/('rgb_'+view+'.npy');b=cubsource/('rgb_'+view+'.npy')
        assert a.resolve()==b.resolve() or digest(a)==digest(b)
        audit[view]={'rgb_sha256':digest(a),'same_ordered_RGB_as_CUB':True}
        destination=common/(view+'_features.npy')
        if not destination.exists():destination.symlink_to((cubcommon/(view+'_features.npy')).resolve())
        for split,rows in protocol['splits'].items():
            raw=np.load(ROOT/'artifacts/pini/waterbirds'/(split+'_'+view+'.npz'))
            cub=np.load(cubcommon/(split+'_'+view+'.npz'))
            assert np.array_equal(raw['ids'],cub['ids']) and np.array_equal(raw['ids'],rows)
            assert np.array_equal(raw['labels'],labels[rows])
            path=common/(split+'_'+view+'.npz')
            if not path.exists():
                np.savez_compressed(path,ids=raw['ids'],labels=labels[rows],features=cub['features'],targets=raw['targets'],known=raw['known'],training_targets=raw['training_targets'])
    manifest=json.loads((cubcommon/'manifest.json').read_text())
    manifest.update(dataset='waterbirds',dataset_description=protocol['dataset'],classes=['landbird','waterbird'],class_count=2,cache_root=str(source),rgb_original=str(source/'rgb_original.npy'),rgb_changed=str(source/'rgb_changed.npy'),labels=str(source/'labels.npy'),split_protocol=str(source/'protocol.json'),source_original_manifest=str(ROOT/'artifacts/pini/waterbirds/manifest.json'),feature_reuse_audit=audit)
    write(common/'manifest.json',manifest)
    return common

def prepare(experiment,batch=128):
    e=experiment
    if e.name=='waterbirds' and e.backbone_name=='resnet18':prepare_waterbirds_shared()
    cache=e.root/'cache';cache.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((e.common/'manifest.json').read_text())
    if manifest['task_fit_sample_ids'] or manifest['historical_checkpoint']:
        raise ValueError('Matched experiment requires an independently frozen, non-taskfit backbone')
    # Link only raw/supervision inputs. Spatial features are freshly encoded.
    for source in (e.original/'cache').iterdir():
        if source.name.startswith('features_') or source.name=='protocol.json' or source.name=='COMPLETE.json':continue
        destination=cache/source.name
        if not destination.exists():destination.symlink_to(source.resolve())
    regions=e.root/'regions'
    if not regions.exists():regions.symlink_to((e.original/'regions').resolve(),target_is_directory=True)
    protocol=json.loads((e.original/'cache/protocol.json').read_text())
    sets={name:set(ids) for name,ids in protocol['splits'].items()}
    assert not(sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
    protocol.update(backbone=manifest['backbone'],backbone_sha256=manifest['backbone_sha256'],backbone_task_fit_sample_ids=[],backbone_preprocess=manifest['preprocess'],downstream_stage='selection_train_only',new_training=True)
    write(cache/'protocol.json',protocol)
    net=e.backbone().cuda()
    aliases=getattr(net,'weight_sha256_aliases',{'lf_core_clip_encoder':net.weight_sha256})
    matching_schemas=[schema for schema,sha in aliases.items() if sha==manifest['backbone_sha256']]
    assert matching_schemas,{'cache_sha256':manifest['backbone_sha256'],'live_aliases':aliases}
    backbone_hash_audit={'matched_schema':matching_schemas[0],'cache_sha256':manifest['backbone_sha256'],'live_aliases':aliases}
    protocol['backbone_hash_audit']=backbone_hash_audit
    audit={}
    if e.name=='waterbirds':
        for view in ['original','changed']:
            cub=e.root.parent/'cub/cache'
            target=cache/('features_'+view+'.npy')
            if not target.exists():target.symlink_to((cub/('features_'+view+'.npy')).resolve())
            record=json.loads((cub/(view+'_spatial_COMPLETE.json')).read_text())
            record.update(reused_from=str(cub/('features_'+view+'.npy')),ordered_RGB_and_splits_verified=True)
            write(cache/(view+'_spatial_COMPLETE.json'),record)
    for view in ['original','changed']:
        path=cache/('features_'+view+'.npy');complete=cache/(view+'_spatial_COMPLETE.json')
        if complete.exists():audit[view]=json.loads(complete.read_text());continue
        rgb=np.load(cache/('rgb_'+view+'.npy'),mmap_mode='r')
        global_features=np.load(e.common/(view+'_features.npy'),mmap_mode='r')
        feature=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(rgb),e.feature_dim,1 if e.backbone_name=='clip_rn50' else 7,1 if e.backbone_name=='clip_rn50' else 7))
        maxerror=0.;started=time.time()
        with torch.no_grad():
            for offset in range(0,len(rgb),batch):
                x=torch.as_tensor(np.array(rgb[offset:offset+batch]),device='cuda').float()/255
                f=net(x);pooled=f.mean((2,3)).cpu().numpy()
                maxerror=max(maxerror,float(np.abs(pooled-global_features[offset:offset+len(x)]).max()))
                feature[offset:offset+len(x)]=f.cpu().numpy()
                if offset%(50*batch)==0:print(json.dumps({'phase':'spatial_encode','dataset':e.name,'view':view,'done':offset+len(x),'total':len(rgb),'pid':os.getpid()}),flush=True)
        feature.flush();assert maxerror<1e-5,maxerror
        audit[view]={'shape':list(feature.shape),'dtype':'float32','spatial_sha256':digest(path),'shared_global_sha256':digest(e.common/(view+'_features.npy')),'global_meanpool_max_abs_error':maxerror,'backbone_sha256':manifest['backbone_sha256'],'backbone_hash_audit':backbone_hash_audit,'seconds':time.time()-started,'pid':os.getpid()}
        write(complete,audit[view])
    del net;torch.cuda.empty_cache()
    protocol['spatial_feature_audit']=audit;write(cache/'protocol.json',protocol)
    write(e.root/'input_manifest.json',{'dataset':e.name,'common':manifest,'spatial_audit':audit,'regions_source':str(e.original/'regions'),'concept_supervision':'RIVAL native binary image attributes' if e.name=='rival10' else 'Koh class-level majority-vote112 native attributes','backbone_label_supervision':'original CLIP pretrained image/text pairs; no task fitting' if e.backbone_name=='clip_rn50' else 'ImageNet1K only; no task image/class fitting','generator_region_supervision':'fixed text phrases and frozen pretrained GroundingDINO features/maps; no native foreground mask passed to generator','holdout_audited':True,'cache_backbone_matches_live':True,'backbone_name':e.backbone_name,'feature_channels':e.feature_dim,'feature_layout':'global singleton attention-pooled CLIP embedding' if e.backbone_name=='clip_rn50' else 'native layer4 spatial map','fit_split':'train only','selection_split':'val only','test_split':'never used for fitting or checkpoint selection','core_path':str(CORE),'core_revision':'eb2e4363ed30e4f5a6456ea781288c10cd4cbfcd','core_source_sha256':{name:digest(CORE/'spatial_noise'/name) for name in ['rn18_datasets.py','cub_aligned.py','cub_noise.py','noise_pair.py','grounding.py','model.py']},'pid':os.getpid()})


def channel_adapted_simple(experiment,steps=1000):
    """Literal one-allocation channel adaptation of frozen original trainer.

    In-memory function globals are copied; no module attribute/source is changed.
    Its sole dimension-specific allocation is 512. All equations, optimizer,
    seed, steps, selection, classifier fitting and saved outputs are identical.
    Model concept/classification functions themselves remain inherited core code.
    """
    import spatial_noise.rn18_datasets as core
    source=inspect.getsource(core.simple)
    needle='torch.empty((len(maps),512),device=\'cuda\')'
    assert source.count(needle)==1
    adapted=source.replace(needle,'torch.empty((len(maps),channels),device=\'cuda\')')
    namespace=dict(vars(core));namespace['channels']=experiment.feature_dim
    exec(compile(adapted,__file__+':channel_only_simple','exec'),namespace)
    write(experiment.root/'channel_adaptation.json',{'original_source_file':str(Path(core.__file__)),'original_source_sha256':digest(core.__file__),'original_function_sha256':hashlib.sha256(source.encode()).hexdigest(),'adapted_function_sha256':hashlib.sha256(adapted.encode()).hexdigest(),'literal_replacement':{'from':needle,'to':'torch.empty((len(maps),channels),device=\'cuda\')'},'channels':experiment.feature_dim,'loss_and_training_logic_unchanged':True,'extra_projection_or_truncation':False,'model_functions':'inherited frozen SimpleCBM.concept_logits/forward/classify and frozen fit_classifier'})
    return namespace['simple'](experiment,seed=42,steps=steps)

def initialize(experiment,steps=1000,epochs=10,batch=64):
    e=experiment
    from spatial_noise.rn18_datasets import simple
    from spatial_noise.cub_noise import train
    # simple() has an unconditional second trainval-refit arm. Its ordinary
    # resume guard allows us to skip that arm without running it or fabricating
    # any checkpoint. This marker records an intentionally skipped phase.
    skipped=e.root/'simple_refit/seed42/COMPLETE.json'
    if not skipped.exists():write(skipped,{'skipped':True,'reason':'matched protocol is train-only; no refit checkpoint is produced'})
    if e.feature_dim==512:simple(e,seed=42,steps=steps)
    else:channel_adapted_simple(e,steps)
    assert not(e.root/'simple_refit/seed42/best.pt').exists()
    train('mean',seed=42,epochs=epochs,batch=batch,refit=False,spatial_prior=False,experiment=e)
    path=e.root/'prior_off/seed42/selection/mean/best.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert set(['cbm','generator','stage','method']).issubset(state)
    assert state['stage']=='selection' and state['method']=='mean' and not state['spatial_noise_prior']
    write(e.root/'initialization_chain.json',{'simple_checkpoint':str(e.root/'simple/seed42/best.pt'),'simple_sha256':digest(e.root/'simple/seed42/best.pt'),'warmup_mean_checkpoint':str(path),'warmup_mean_sha256':digest(path),'simple_steps':steps,'warmup_mean_epochs':epochs,'warmup_mean_batch':batch,'warmup_mean_loss':'class_CE + weighted_concept_BCE + MSE(concept_prob,cached_original_prob) + .01*mean(mu^2+sigma^2)','pair_initialization':'unmodified noise_pair.initialize(e,selection) loads this actual warmup mean checkpoint for both arms','fit_split':'train','refit_phase_skipped':True})


@torch.no_grad()
def export(experiment,method,batch=64):
    e=experiment
    from spatial_noise.cub_noise import inputs,forward
    from spatial_noise.grounding import GroundedNoiseGenerator
    from spatial_noise.cub_aligned import tensor
    state=torch.load(e.root/'pair_explore/selection'/method/'best.pt',map_location='cpu',weights_only=True)
    cbm=e.model().cuda().eval();cbm.load_state_dict(state['cbm'])
    gen=GroundedNoiseGenerator(regions=len(e.phrases),mean_limit=state['mean_limit'],std_limit=state['std_limit'],spatial_noise_prior=state['spatial_noise_prior']).cuda().eval();gen.load_state_dict(state['generator'])
    net=e.backbone().cuda().eval();arrays,splits,protocol,grounded=inputs(experiment=e)
    destination=e.root/'exports'/method;destination.mkdir(parents=True,exist_ok=True)
    summary={};train_prob=None
    for split in ['train','val','test']:
        ids=splits[split];raw={view:np.load(e.common/(split+'_'+view+'.npz')) for view in ['original','changed']}
        split_prob={}
        for view in ['original','changed']:
            assert np.array_equal(raw[view]['ids'],ids)
            torch.manual_seed(4242);cs=[];ys=[]
            for offset in range(0,len(ids),batch):
                y,c,_,_=forward(net,cbm,gen,arrays,grounded,ids[offset:offset+batch],view,method)
                cs.append(c.cpu().numpy());ys.append(y.cpu().numpy())
            c=np.concatenate(cs);y=np.concatenate(ys);p=1/(1+np.exp(-np.clip(c,-80,80)));split_prob[view]=p
            eligible=np.asarray(arrays['background_eligible'][ids],bool) if view=='changed' and 'background_eligible' in arrays else np.ones(len(ids),bool)
            known=raw[view]['known'];targets=raw[view]['targets'];training_targets=raw[view]['training_targets']
            np.savez_compressed(destination/(split+'_'+view+'.npz'),ids=ids,labels=np.asarray(arrays['labels'][ids]),concepts=p,concept_logits=c,logits=y,targets=targets,known=known,training_targets=training_targets,eligible=eligible)
            metrics={'classification':classification_metrics(y[eligible],np.asarray(arrays['labels'][ids])[eligible]),'native_image_concepts':concept_metrics(p[eligible],targets[eligible],known[eligible]),'training_target_concepts':concept_metrics(p[eligible],training_targets[eligible]),'images':int(eligible.sum())}
            summary[split+'_'+view]=metrics
            print(json.dumps({'phase':'matched_export','dataset':e.name,'method':method,'split':split,'view':view,'accuracy':metrics['classification']['accuracy'],'native_mAP':metrics['native_image_concepts']['mAP'],'pid':os.getpid()}),flush=True)
        if split=='train':train_prob=split_prob['original']
        eligible=np.asarray(arrays['background_eligible'][ids],bool) if 'background_eligible' in arrays else np.ones(len(ids),bool)
        summary[split+'_shift']=paired_concept_metrics(split_prob['original'][eligible],split_prob['changed'][eligible],train_prob)
    weights=cbm.classifier.weight.cpu().numpy();np.save(destination/'classifier_weights.npy',weights)
    summary.update(nec=nec(weights),native_head_input='raw concept logits standardized with core concept_center/concept_scale; exported probabilities for interpretable concept metrics',selected_epoch=state['selected_epoch'],fit_split='train',selection_split='val',historical_checkpoint=False,algorithm_version='pair_consistency30_coverage_reward',checkpoint=str(e.root/'pair_explore/selection'/method/'best.pt'))
    write(destination/'results.json',summary)
    return summary


def run(dataset,output,steps,init_epochs,pair_epochs,batch,pairs,backbone='resnet18'):
    e=MatchedExperiment(dataset,output,backbone);e.root.mkdir(parents=True,exist_ok=True)
    prepare(e,batch=128)
    if e.feature_dim>512:
        required=14_000_000_000
        while torch.cuda.mem_get_info()[0]<required:
            print(json.dumps({'phase':'waiting_gpu_budget','required_free_bytes':required,'available_bytes':torch.cuda.mem_get_info()[0],'dataset':e.name,'backbone':e.backbone_name,'pid':os.getpid()}),flush=True)
            time.sleep(30)
    initialize(e,steps,init_epochs,batch)
    from spatial_noise.noise_pair import train
    results={}
    for method in ['mean','noise']:
        train(e,method,'selection',epochs=pair_epochs,pairs=pairs)
        results[method]=export(e,method,batch)
    write(e.root/'matched_results.json',results)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',default=['rival10','cub'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/pini_matched')
    parser.add_argument('--backbone',choices=['resnet18','resnet50','clip_rn50'],default='resnet18')
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--init-epochs',type=int,default=10)
    parser.add_argument('--pair-epochs',type=int,default=5)
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--pairs',type=int,default=32)
    args=parser.parse_args();torch.set_num_threads(3)
    if args.backbone!='resnet18':args.output=args.output/args.backbone
    print(json.dumps({'phase':'matched_start','pid':os.getpid(),'datasets':args.datasets,'fit':'train only','seed':42}),flush=True)
    for dataset in args.datasets:run(dataset,args.output,args.steps,args.init_epochs,args.pair_epochs,args.batch,args.pairs,args.backbone)


if __name__=='__main__':main()
