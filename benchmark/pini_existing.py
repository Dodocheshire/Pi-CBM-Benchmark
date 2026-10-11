"""Recompute and export historical PiNI checkpoints without modifying the core.

Historical runs carry different backbone/supervision/loss protocols. They are
explicitly tagged and cannot establish a matched new-training method comparison.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import average_precision_score
from benchmark.pini import import_core, CORE

DEFAULT_SOURCE=Path('/home/pengcheng/Pi-CBM/PiNI/artifacts')


def native_image_audit(source, root, dataset):
    """Exact CUB attribute-name mapping; certainty >=3, separate class targets."""
    attrs=np.load(root/'cache/attrs.npy',mmap_mode='r')
    if dataset=='rival10':
        return np.asarray(attrs),np.ones(attrs.shape,bool),{'type':'RIVAL10 native image attributes'}
    cubroot=Path(source)/'cub_aligned_rn18'
    species=np.load(cubroot/'cache/labels.npy')
    # CUB and derivative Waterbirds share identical ordered native images/attrs.
    assert np.array_equal(attrs,np.load(cubroot/'cache/attrs.npy',mmap_mode='r'))
    assert all(len(np.unique(attrs[species==k],axis=0))==1 for k in np.unique(species))
    local=Path(source).parent/'artifacts/datasets/local'
    names={line.split(' ',1)[1].strip():int(line.split(' ',1)[0]) for line in (local/'attributes.txt').read_text().splitlines()}
    concepts=json.loads((root/'cache/concepts.json').read_text())
    selected={names[name]:column for column,name in enumerate(concepts)}
    native_ids=np.load(root/'cache/native_ids.npy')
    row_of={int(value):row for row,value in enumerate(native_ids)}
    targets=np.zeros(attrs.shape,np.float32);certainty=np.zeros(attrs.shape,np.uint8)
    annotation=local/'CUB_200_2011/attributes/image_attribute_labels.txt'
    with annotation.open() as stream:
        for line in stream:
            fields=line.split();image,attribute=int(fields[0]),int(fields[1])
            if image in row_of and attribute in selected:
                row,column=row_of[image],selected[attribute]
                targets[row,column]=int(fields[2]);certainty[row,column]=int(fields[3])
    return targets,certainty>=3,{'type':'CUB native image attributes; exact same112 name-to-attribute-id mapping',
          'certainty_minimum':3,'training_labels':'Koh class-level majority-vote112, asserted constant per species',
          'attribute_ids':[names[name] for name in concepts],
          'annotation_sha256':hashlib.sha256(annotation.read_bytes()).hexdigest(),
          'attribute_names_sha256':hashlib.sha256((local/'attributes.txt').read_bytes()).hexdigest()}

def checkpoint_paths(source,dataset):
    root=Path(source)/(dataset+'_aligned_rn18')
    if dataset=='rival10':
        return root,{m:root/'pair_explore/refit'/m/'best.pt' for m in ['mean','noise']}
    return root,{'noise':root/'prior_off/seed42/refit/noise/best.pt'}


def concept_metrics(target,probabilities):
    """Macro AP and macro Pearson across nonconstant native concepts."""
    target=np.asarray(target,dtype=np.float64)
    probabilities=np.asarray(probabilities,dtype=np.float64)
    ap=[];pearson=[]
    for y,p in zip(target.T,probabilities.T):
        ap.append(float(average_precision_score(y,p)) if np.any(y>0) else None)
        pearson.append(float(np.corrcoef(y,p)[0,1]) if y.std()>0 and p.std()>0 else None)
    return {'mAP':float(np.mean([v for v in ap if v is not None])),
            'pearson_macro':float(np.mean([v for v in pearson if v is not None])),
            'concept_accuracy':float(((probabilities>=.5)==target.astype(bool)).mean()),
            'AP_per_concept':ap,'pearson_per_concept':pearson,
            'AP_valid_concepts':sum(v is not None for v in ap),
            'pearson_valid_concepts':sum(v is not None for v in pearson)}


def sigmoid(logits):
    x=np.asarray(logits,dtype=np.float64)
    return np.exp(-np.logaddexp(0,-x))


def metadata(source,dataset,root,checkpoint):
    protocol=json.loads((root/'cache/protocol.json').read_text())
    concepts=json.loads((root/'cache/concepts.json').read_text())
    version=('pair_consistency30_coverage_reward' if dataset=='rival10' else 'shared_noise_cached_reference_v1')
    return {'dataset':dataset,'historical_checkpoint':True,'algorithm_version':version,
            'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'core_path':str(CORE),'core_revision':'eb2e4363ed30e4f5a6456ea781288c10cd4cbfcd','concept_names':concepts,'protocol':protocol,
            'native_concept_supervision':'image native binary attributes' if dataset=='rival10' else 'Koh-112 class-level majority-vote attributes (not individual image visibility)',
            'foreground_supervision':'native masks used to construct diagnostic interventions; not supplied to generator',
            'original_id_source':str(Path(source)/'rival_vitl14/manifest.json') if dataset=='rival10' else str(root/'cache/native_ids.npy'),
            'grounding':'frozen Grounding DINO tiny, region features/maps cached from each corresponding RGB view',
            'backbone_selection_warning':('Historical selection used trainval-refit task-trained backbone; val scores are descriptive, not independent selection evidence' if dataset=='rival10' else 'CUB task pretrained backbone provenance must be independently audited before matched protocol conclusions'),
            'split_usage':'refit trained train+val; only test is a held-out report for the downstream fitting',
            'not_matched_new_training':True,'pid':os.getpid()}


def export_historical(source,output,datasets):
    """CPU reevaluation of historical saved test outputs, with original row IDs."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summary={}
    for dataset in datasets:
        root,checkpoints=checkpoint_paths(source,dataset)
        attrs=np.load(root/'cache/attrs.npy',mmap_mode='r')
        labels=np.load(root/'cache/labels.npy',mmap_mode='r')
        audit_targets,audit_known,audit_protocol=native_image_audit(source,root,dataset)
        protocol=json.loads((root/'cache/protocol.json').read_text())
        ids=np.asarray(protocol['splits']['test'],dtype=np.int64)
        for method,checkpoint in checkpoints.items():
            destination=output/dataset/method/'historical_saved_test';destination.mkdir(parents=True,exist_ok=True)
            meta=metadata(source,dataset,root,checkpoint);meta['native_image_audit_protocol']=audit_protocol;metrics={}
            for view in ['original','changed']:
                c=np.load(checkpoint.parent/(view+'_concepts.npy'));y=np.load(checkpoint.parent/(view+'_logits.npy'))
                assert len(c)==len(ids)==len(y)
                eligible=np.load(root/'cache/background_eligible.npy')[ids] if view=='changed' and (root/'cache/background_eligible.npy').exists() else np.ones(len(ids),bool)
                probabilities=sigmoid(c)
                np.savez(destination/(view+'.npz'),sample_ids=ids,native_ids=(np.load(root/'cache/native_ids.npy')[ids] if (root/'cache/native_ids.npy').exists() else ids),labels=labels[ids],native_concepts=attrs[ids],training_targets=attrs[ids],targets=audit_targets[ids],known=audit_known[ids],concept_logits=c,concept_probabilities=probabilities,class_logits=y,eligible=eligible)
                metrics[view]=concept_metrics(attrs[ids][eligible],probabilities[eligible])
                from benchmark.metrics import concept_metrics as shared_concept_metrics
                metrics[view]['native_image_audit']=shared_concept_metrics(probabilities[eligible],audit_targets[ids][eligible],audit_known[ids][eligible])
                metrics[view].update(class_accuracy=float((y.argmax(1)[eligible]==labels[ids][eligible]).mean()),images=int(eligible.sum()))
            metrics['concept_drift_mae']=float(np.abs(sigmoid(np.load(checkpoint.parent/'original_concepts.npy'))-sigmoid(np.load(checkpoint.parent/'changed_concepts.npy')))[eligible].mean())
            (destination/'metadata.json').write_text(json.dumps(meta,indent=2))
            (destination/'metrics.json').write_text(json.dumps(metrics,indent=2))
            summary[dataset+'/'+method]=metrics
            print(json.dumps({'phase':'cpu_recompute','dataset':dataset,'method':method,'pid':os.getpid(),'mAP':metrics['original']['mAP'],'pearson':metrics['original']['pearson_macro'],'accuracy':metrics['original']['class_accuracy']}),flush=True)
    (output/'historical_summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def export_live(source,output,datasets,splits,views,seeds,batch=64,max_images=None,online=False):
    """Call unmodified generator/backbone/CBM on RGB interventions and repeat seeds.

    Cached original/changed grounding is the real grounder output for that view.
    Masked interventions run the real FrozenGrounder online; reusing original
    maps for an altered image is forbidden. Noise seeds match paired RGB views.
    Batch size is recorded because the core's random draws depend on batching.
    """
    import_core()
    import torch
    from PIL import Image
    from spatial_noise.cub_aligned import SimpleCBM,BirdBackbone
    from spatial_noise.rn18_datasets import RivalBackbone,RIVAL_PHRASES
    from spatial_noise.grounding import GroundedNoiseGenerator,FrozenGrounder,BIRD_PHRASES
    torch.set_num_threads(3)
    for dataset in datasets:
        root,checkpoints=checkpoint_paths(source,dataset)
        protocol=json.loads((root/'cache/protocol.json').read_text())
        arrays={key:np.load(root/'cache'/(key+'.npy'),mmap_mode='r') for key in ['rgb_original','rgb_changed','foreground','attrs','labels']}
        audit_targets,audit_known,audit_protocol=native_image_audit(source,root,dataset)
        grounded={view:{key:np.load(root/'regions'/view/(key+'.npy'),mmap_mode='r') for key in ['features','maps']} for view in ['original','changed']}
        backbone=(RivalBackbone(root/'backbone/refit/best.pt') if dataset=='rival10' else BirdBackbone()).cuda().eval()
        grounder=None
        if online or any(v in ['foreground_only','background_only'] for v in views):
            # Resolve detector independently of the original relative working directory.
            detector=Path(source).parent.parent/'D-Pi/artifacts/models/grounding-dino-tiny'
            grounder=FrozenGrounder(phrases=RIVAL_PHRASES if dataset=='rival10' else BIRD_PHRASES,model_path=detector,precision='fp16' if dataset=='rival10' else 'fp32')
        for method,checkpoint in checkpoints.items():
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            k,classes=state['cbm']['concepts.weight'].shape[0],state['cbm']['classifier.weight'].shape[0]
            cbm=SimpleCBM(k,classes).cuda().eval();cbm.load_state_dict(state['cbm'])
            gen=GroundedNoiseGenerator(regions=len(RIVAL_PHRASES if dataset=='rival10' else BIRD_PHRASES),mean_limit=state['mean_limit'],std_limit=state['std_limit'],spatial_noise_prior=state['spatial_noise_prior']).cuda().eval();gen.load_state_dict(state['generator'])
            for split in splits:
                ids=np.asarray(protocol['splits'][split],dtype=np.int64)
                if max_images is not None and max_images<len(ids):ids=np.sort(np.random.default_rng(42).choice(ids,size=max_images,replace=False))
                native_ids=np.load(root/'cache/native_ids.npy')[ids] if (root/'cache/native_ids.npy').exists() else ids
                for seed in seeds:
                    for view in views:
                        destination=Path(output)/dataset/method/split/f'seed{seed}'/view;destination.mkdir(parents=True,exist_ok=True)
                        if (destination/'COMPLETE.json').exists():continue
                        started=time.time();torch.manual_seed(seed);cs=[];ys=[];fs=[]
                        with torch.no_grad():
                            for offset in range(0,len(ids),batch):
                                selected=ids[offset:offset+batch]
                                raw=np.array(arrays['rgb_changed' if view=='changed' else 'rgb_original'][selected])
                                if view in ['foreground_only','background_only']:
                                    mask=np.array(arrays['foreground'][selected])[:,None]
                                    if view=='background_only':mask=~mask
                                    raw=raw*mask # black fill, protocol records intervention
                                rgb=torch.as_tensor(raw,device='cuda').float()/255
                                if online or view in ['foreground_only','background_only']:
                                    region=grounder([Image.fromarray(im.transpose(1,2,0)) for im in raw])
                                else:
                                    region={key:torch.as_tensor(np.array(value[selected]),device='cuda').float() for key,value in grounded[view].items()}
                                perturbed,_,_=gen(rgb,region,method)
                                with torch.autocast('cuda',dtype=torch.float16):features=backbone(perturbed)
                                y,c=cbm(features);cs.append(c.cpu().numpy());ys.append(y.cpu().numpy());fs.append(features.float().mean((2,3)).cpu().numpy())
                        c=np.concatenate(cs);y=np.concatenate(ys);probabilities=sigmoid(c)
                        eligible=np.load(root/'cache/background_eligible.npy')[ids] if view=='changed' and (root/'cache/background_eligible.npy').exists() else np.ones(len(ids),bool)
                        np.savez(destination/'outputs.npz',sample_ids=ids,native_ids=native_ids,labels=arrays['labels'][ids],native_concepts=arrays['attrs'][ids],training_targets=arrays['attrs'][ids],targets=audit_targets[ids],known=audit_known[ids],concept_logits=c,concept_probabilities=probabilities,class_logits=y,eligible=eligible)
                        metrics=concept_metrics(arrays['attrs'][ids][eligible],probabilities[eligible]);
                        from benchmark.metrics import concept_metrics as shared_concept_metrics
                        metrics['native_image_audit']=shared_concept_metrics(probabilities[eligible],audit_targets[ids][eligible],audit_known[ids][eligible])
                        metrics.update(class_accuracy=float((y.argmax(1)[eligible]==arrays['labels'][ids][eligible]).mean()),images=int(eligible.sum()))
                        meta=metadata(source,dataset,root,checkpoint);meta.update(native_image_audit_protocol=audit_protocol,view=view,split=split,noise_seed=seed,batch_size=batch,grounding_mode='online_reencoded' if online or view in ['foreground_only','background_only'] else 'frozen_real_view_cache',masked_fill='black RGB zero' if 'only' in view else None,seconds=time.time()-started,max_images=max_images,subset_sampling='uniform without replacement seed42' if max_images is not None else None,core_generator_unmodified=True)
                        if Path(output).name=='pini_existing' and method=='noise' and seed==seeds[0] and view in ['original','changed'] and max_images is None:
                            shared=Path(output).parent/'pini'/dataset;shared.mkdir(parents=True,exist_ok=True)
                            # Raw frozen backbone features for matched non-CBM comparator.
                            cached=np.load(root/'cache'/('features_'+view+'.npy'),mmap_mode='r')
                            pooled=np.concatenate([np.asarray(cached[selected],dtype=np.float32).mean((2,3)) for selected in np.array_split(ids,max(1,(len(ids)+255)//256))])
                            np.savez(shared/(split+'_'+view+'.npz'),features=pooled,normalized_features=(pooled-cbm.feature_center.cpu().numpy())/cbm.feature_scale.cpu().numpy(),perturbed_features=np.concatenate(fs),labels=arrays['labels'][ids],ids=ids,native_ids=native_ids,targets=audit_targets[ids],known=audit_known[ids],training_targets=arrays['attrs'][ids],concepts=probabilities,concept_logits=c,logits=y,eligible=eligible)
                            common_manifest={'dataset':dataset,'cache_root':str(root/'cache'),'rgb_original':str(root/'cache/rgb_original.npy'),'rgb_changed':str(root/'cache/rgb_changed.npy'),'labels':str(root/'cache/labels.npy'),'split_protocol':str(root/'cache/protocol.json'),'concept_names':meta['concept_names'],'backbone':protocol['backbone'],'features':'unperturbed frozen backbone spatial mean; no LN; model_feature normalization stored separately','training_targets':'native image attributes' if dataset=='rival10' else 'Koh class-level112','targets':'native image attribute audit certainty>=3','known':'all true for RIVAL, native certainty>=3 for CUB/WB','feature_shape':[len(ids),512],'native_audit':audit_protocol,'historical_checkpoint':True,'algorithm_version':meta['algorithm_version']}
                            (shared/'manifest.json').write_text(json.dumps(common_manifest,indent=2))
                        (destination/'metadata.json').write_text(json.dumps(meta,indent=2));(destination/'metrics.json').write_text(json.dumps(metrics,indent=2));(destination/'COMPLETE.json').write_text(json.dumps({'pid':os.getpid(),'seconds':time.time()-started}))
                        print(json.dumps({'phase':'live','dataset':dataset,'method':method,'split':split,'view':view,'seed':seed,'pid':os.getpid(),'seconds':time.time()-started,'mAP':metrics['mAP'],'pearson':metrics['pearson_macro'],'accuracy':metrics['class_accuracy']}),flush=True)
        if grounder is not None:grounder.close()
        del backbone;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=Path('artifacts/pini_existing'))
    parser.add_argument('--phase',choices=['historical','live'],default='historical')
    parser.add_argument('--datasets',nargs='+',default=['rival10','cub','waterbirds'])
    parser.add_argument('--splits',nargs='+',default=['test','val','train'])
    parser.add_argument('--views',nargs='+',default=['original','changed'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[4242])
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--max-images',type=int)
    parser.add_argument('--online',action='store_true')
    args=parser.parse_args()
    if args.phase=='historical':export_historical(args.source,args.output,args.datasets)
    else:export_live(args.source,args.output,args.datasets,args.splits,args.views,args.seeds,args.batch,args.max_images,args.online)


if __name__=='__main__':main()
