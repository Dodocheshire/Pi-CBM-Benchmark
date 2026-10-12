"""Evaluate shared masked and corrupted RGB views with fresh online grounding."""
import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn

from .lf import upstream, predict_features
from .metrics import classification_metrics, concept_metrics


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def bind_full_export_inputs(out, paths, protocol):
    """Bind partial full exports to their original weights, inputs, and producer."""
    record={'files':{key:{'path':str(Path(path).resolve()),'sha256':file_sha256(path)} for key,path in paths.items()},'protocol':protocol}
    destination=out/'full_export_input_manifest.json'
    if destination.exists():
        if json.loads(destination.read_text())!=record:
            raise ValueError('Full export inputs changed during resume; use a new --out namespace.')
    elif any(out.glob('*_COMPLETE.json')):
        raise ValueError('Completed full exports lack an input binding; use a new --out namespace.')
    else:destination.write_text(json.dumps(record,indent=2))
    return record


def transform(raw, kind, severity, mask, seed):
    """Explicit corruption settings; outputs are uint8 CHW RGB at 224px."""
    if kind=='original': return raw.copy()
    if kind in ['foreground_only', 'background_only']:
        use = mask if kind=='foreground_only' else ~mask
        return raw*use[None]
    im = Image.fromarray(raw.transpose(1,2,0))
    level = {1:0,3:1,5:2}[severity]
    if kind=='gaussian_noise':
        noise = np.random.default_rng(seed).normal(0,[.03,.09,.18][level],raw.shape)
        return np.clip(raw.astype(float)+255*noise,0,255).round().astype('uint8')
    if kind=='gaussian_blur': im=im.filter(ImageFilter.GaussianBlur([.5,1.5,3.][level]))
    elif kind=='contrast': im=ImageEnhance.Contrast(im).enhance([.8,.4,.15][level])
    elif kind=='jpeg':
        buffer=BytesIO();im.save(buffer,format='JPEG',quality=[80,35,10][level]);buffer.seek(0)
        im=Image.open(buffer).convert('RGB')
    else: raise ValueError(kind)
    return np.asarray(im).transpose(2,0,1).copy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--images',type=int,default=256)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--export-full-concepts',action='store_true',help='Additionally save full PiNI raw concept logits and native labels in a new output namespace.')
    p.add_argument('--corruptions-only',action='store_true',help='Evaluate original plus the twelve corruption settings, excluding the two mask diagnostics.')
    a=p.parse_args();torch.set_num_threads(3);a.out.mkdir(parents=True,exist_ok=True)
    if a.export_full_concepts:
        for previous in a.out.glob('*_pini_*.npz'):
            with np.load(previous) as saved:
                if not {'concept_logits','concepts_full','targets_full','known_full'}.issubset(saved.files):
                    raise ValueError(f'{previous} is an old selected-only export; use a new --out namespace.')
        for completed in a.out.glob('*_COMPLETE.json'):
            for arm in ['mean','noise']:
                if not (a.out/(completed.name.removesuffix('_COMPLETE.json')+'_pini_'+arm+'.npz')).exists():
                    raise ValueError(f'{completed} lacks full PiNI exports; use a new --out namespace.')
        # Preserve the previously audited fallback after optional Hub kernels were installed.
        os.environ['USE_HUB_KERNELS']='NO'
    from .pini_matched import MatchedExperiment
    e=MatchedExperiment(a.dataset,Path('artifacts/pini_matched'))
    from spatial_noise.grounding import GroundedNoiseGenerator, FrozenGrounder
    upstream()
    from pi_cbm.pipeline import baseline_from_state, build_model
    source=Path('artifacts/matched/lf')/a.dataset/'rn18_imagenet/paired_augmentation/seed42'
    lf=torch.load(source/'models.pt',map_location='cpu',weights_only=True)
    common=e.common;report=json.loads((source/'report.json').read_text())
    raw_cache=e.root/'cache';protocol=json.loads((raw_cache/'protocol.json').read_text())
    ids=np.asarray(protocol['splits']['test'])
    ids=np.sort(np.random.default_rng(42).choice(ids,min(a.images,len(ids)),replace=False))
    test=np.load(common/'test_original.npz');positions={int(v):i for i,v in enumerate(test['ids'])}
    rows=np.array([positions[int(v)] for v in ids]);labels=test['labels'][rows]
    targets=test['targets'][rows];known=test['known'][rows]
    if a.export_full_concepts:assert targets.shape==(len(ids),e.k) and known.shape==targets.shape
    rgb=np.load(raw_cache/'rgb_original.npy',mmap_mode='r');masks=np.load(raw_cache/'foreground.npy',mmap_mode='r')
    if a.export_full_concepts:
        bind_full_export_inputs(a.out,{'producer':Path(__file__),'lf_models':source/'models.pt','pini_mean':e.root/'pair_explore/selection/mean/best.pt','pini_noise':e.root/'pair_explore/selection/noise/best.pt','noncbm':Path('artifacts/noncbm')/f'{a.dataset}_rn18_imagenet_paired/seed42/best.pt','common_manifest':common/'manifest.json','input_manifest':e.root/'input_manifest.json','train_arrays':common/'train_original.npz','test_arrays':common/'test_original.npz','teacher_text':Path('artifacts/common')/(a.dataset if a.dataset!='waterbirds' else 'cub')/'teacher/text.npy','rgb':raw_cache/'rgb_original.npy','foreground':raw_cache/'foreground.npy'}, {'dataset':a.dataset,'ids':ids.tolist(),'batch':a.batch,'sampling_seed':42,'noise_seed':4242,'pini_draws':1,'lfpi_draws':8,'corruptions_only':a.corruptions_only})
    train=np.load(common/'train_original.npz')['features'];teacher=Path('artifacts/common')/(a.dataset if a.dataset!='waterbirds' else 'cub')/'teacher'
    data={'splits':{'train':{'features':torch.from_numpy(train)}},'text':torch.from_numpy(np.load(teacher/'text.npy'))}
    upstream();baseline=baseline_from_state(lf['baseline'],'cuda')
    cfg=lf['config'];pi=build_model(baseline_from_state(lf['pi_cbm'],'cuda'),data,lf['selected'],cfg,'cuda')
    pi.generator.load_state_dict(lf['generator']);pi.eval()
    clean_cfg={**cfg,'injection':{**cfg['injection'],'site':'baseline'}}
    clean=build_model(baseline,data,lf['selected'],clean_cfg,'cuda').eval()
    net=e.backbone().cuda().eval();pini={}
    full_provenance={}
    for arm in ['mean','noise']:
        state=torch.load(e.root/'pair_explore/selection'/arm/'best.pt',map_location='cpu',weights_only=True)
        cbm=e.model().cuda().eval();cbm.load_state_dict(state['cbm'])
        gen=GroundedNoiseGenerator(regions=len(e.phrases),mean_limit=state['mean_limit'],std_limit=state['std_limit'],spatial_noise_prior=state['spatial_noise_prior']).cuda().eval()
        gen.load_state_dict(state['generator']);pini[arm]=(cbm,gen)
        if a.export_full_concepts:
            checkpoint=e.root/'pair_explore/selection'/arm/'best.pt'
            full_provenance[arm]={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'selected_epoch':state['selected_epoch']}
    state=torch.load(Path('artifacts/noncbm')/f'{a.dataset}_rn18_imagenet_paired/seed42/best.pt',weights_only=True)
    head=nn.Linear(512,state['head']['weight'].shape[0]).cuda();head.load_state_dict(state['head']);head.eval()
    detector=Path('/home/pengcheng/Pi-CBM/D-Pi/artifacts/models/grounding-dino-tiny')
    grounder=FrozenGrounder(e.phrases,detector,precision=e.detector_precision)
    if a.export_full_concepts:
        grounder.detector.set_use_kernels(False)
        assert grounder.detector.use_kernels is False
        attention=[m for m in grounder.detector.modules() if type(m).__name__=='MultiScaleDeformableAttention']
        assert attention and all(m.forward.__func__.__module__=='transformers.models.grounding_dino.modeling_grounding_dino' for m in attention)
        detector_digest=hashlib.sha256()
        for name,value in grounder.detector.state_dict().items():
            detector_digest.update(name.encode());detector_digest.update(value.cpu().numpy().tobytes())
    settings=[('original',0),('foreground_only',0),('background_only',0)]+[(kind,s) for kind in ['gaussian_noise','gaussian_blur','contrast','jpeg'] for s in [1,3,5]]
    if a.corruptions_only:settings=[setting for setting in settings if setting[0] not in ['foreground_only','background_only']]
    summary={};selected=np.asarray(report['selected_indices'])
    for kind,severity in settings:
        key=f'{kind}_{severity}'
        if (a.out/(key+'_COMPLETE.json')).exists():continue
        outputs={m:{'logits':[],'concepts':[]} for m in ['lfcbm','lfpi','pini_mean','pini_noise','noncbm']}
        if a.export_full_concepts:
            input_digest=hashlib.sha256()
            for arm in ['mean','noise']:outputs['pini_'+arm].update(concept_logits=[],concepts_full=[])
        torch.manual_seed(4242)
        with torch.no_grad():
            for start in range(0,len(ids),a.batch):
                batch_ids=ids[start:start+a.batch]
                raw=np.stack([transform(np.array(rgb[i]),kind,severity,np.asarray(masks[i],bool),42+int(i)) for i in batch_ids])
                if a.export_full_concepts:input_digest.update(raw.tobytes())
                images=torch.as_tensor(raw,device='cuda').float()/255
                region=grounder([Image.fromarray(im.transpose(1,2,0)) for im in raw])
                spatial=net(images);features=spatial.mean((2,3))
                for method,model in [('lfcbm',clean),('lfpi',pi)]:
                    pred=predict_features(model,features,draws=8 if method=='lfpi' else 1,seed=4242)
                    for field in ['logits','concepts']:outputs[method][field].append(pred[field].numpy())
                for arm,(cbm,gen) in pini.items():
                    noisy,_,_=gen(images,region,arm)
                    with torch.autocast('cuda',dtype=torch.float16): f=net(noisy)
                    y,c=cbm(f);outputs['pini_'+arm]['logits'].append(y.cpu().numpy())
                    outputs['pini_'+arm]['concepts'].append(c.sigmoid().cpu().numpy()[:,selected])
                    if a.export_full_concepts:
                        outputs['pini_'+arm]['concept_logits'].append(c.cpu().numpy())
                        outputs['pini_'+arm]['concepts_full'].append(c.sigmoid().cpu().numpy())
                y=head(((features.cpu()-state['mean'])/state['std']).cuda())
                outputs['noncbm']['logits'].append(y.cpu().numpy())
        metrics={}
        for method,value in outputs.items():
            y=np.concatenate(value['logits']);values={'ids':ids,'labels':labels,'logits':y}
            metrics[method]={'classification':classification_metrics(y,labels)}
            if value['concepts']:
                c=np.concatenate(value['concepts']);values.update(concepts=c,targets=targets[:,selected],known=known[:,selected])
                metrics[method]['concepts']=concept_metrics(c,targets[:,selected],known[:,selected])
            if a.export_full_concepts and method in ['pini_mean','pini_noise']:
                values.update(concept_logits=np.concatenate(value['concept_logits']),concepts_full=np.concatenate(value['concepts_full']),targets_full=targets,known_full=known,concept_indices_full=np.arange(targets.shape[1]),input_rgb_sha256=np.asarray(input_digest.hexdigest()))
            np.savez_compressed(a.out/(key+'_'+method+'.npz'),**values)
        (a.out/(key+'_COMPLETE.json')).write_text(json.dumps(metrics,indent=2));summary[key]=metrics
        print(json.dumps({'dataset':a.dataset,'setting':key,'accuracy':{k:v['classification']['accuracy'] for k,v in metrics.items()}}),flush=True)
    grounder.close()
    protocol_output={'dataset':a.dataset,'images':len(ids),'sample_ids':ids.tolist(),'sampling_seed':42,'inference_noise_seed':4242,'batch_size':a.batch,'grounding':'online recomputed for each changed RGB view','training_views':['original','changed'],'concept_indices':selected.tolist(),'masked_fill':'black','settings':settings,'gaussian_noise_std':[.03,.09,.18],'blur_radius':[.5,1.5,3],'contrast':[.8,.4,.15],'jpeg_quality':[80,35,10]}
    if a.export_full_concepts:
        protocol_output.update(export_full_concepts=True,full_concept_count=targets.shape[1],pini_draws=1,lfpi_draws=8,method_execution_order=['lfcbm','lfpi','pini_mean','pini_noise','noncbm'],grounder_operator='original PyTorch fallback',grounder_precision=e.detector_precision,detector_state_sha256=detector_digest.hexdigest(),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),pini_checkpoints=full_provenance,backbone_weight_sha256=net.weight_sha256,source_rgb_path=str(raw_cache/'rgb_original.npy'),source_input_manifest_sha256=hashlib.sha256((e.root/'input_manifest.json').read_bytes()).hexdigest(),full_label_source='native RIVAL18 or CUB same112 image attributes certainty>=3; species-level112 used only in training',fixed_heads_retrained=False)
    (a.out/'protocol.json').write_text(json.dumps(protocol_output,indent=2))


if __name__=='__main__':main()
