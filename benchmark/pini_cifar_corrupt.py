"""Fixed fresh PiNI checkpoints on all official CIFAR-C test rows; external adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['USE_HUB_KERNELS']='NO'
import numpy as np
from PIL import Image
import torch

from .lf_cifar import CORRUPTIONS,resize224
from .pini_cifar import Experiment,PHRASES,FROZEN_DINO_STATE_SHA256
from .pini_matched import ROOT
from .metrics import classification_metrics,teacher_fidelity


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)


class Native32IntensityReader:
    """Indexed native uint8 -> original CUDA canonical resize -> float32 s0.

    Core tensor(array, ids) obtains a batch, copies it to CUDA, then divides by
    255 to obtain q1. This preserves the existing s0/255 pixel protocol. No
    224px quantization or CPU interpolation substitutes for the original s0.
    """
    def __init__(self,source,source_rows,device='cuda'):
        self.source=source;self.source_rows=np.asarray(source_rows,np.int64);self.device=device
        self.shape=(len(self.source_rows),3,224,224);self.dtype=np.dtype('float32')
    def __len__(self):return self.shape[0]
    def __getitem__(self,selection):
        positions=np.arange(len(self))[selection];scalar=np.ndim(positions)==0
        rows=self.source_rows[np.atleast_1d(positions)]
        raw=np.array(self.source[rows]);raw=raw.transpose(0,3,1,2) if raw.shape[-1]==3 else raw
        with torch.no_grad(),torch.autocast('cuda',enabled=False):value=(resize224(raw,self.device)*255).cpu().numpy()
        return value[0] if scalar else value
    def __array__(self,dtype=None,copy=None):
        raise TypeError('RGB reader requires indexed access; whole-array materialization is disabled')


def context(dataset):
    e=Experiment(dataset);base=e.common;out=ROOT/'artifacts/pini_cifar_c'/dataset/'rn18_imagenet/seed42'
    if not (e.root/'CIFAR_COMPLETE.json').exists():raise ValueError('Formal fresh PiNI training incomplete')
    original=np.load(base/'rn18_imagenet/test_original.npz');ids=original['ids'];labels=original['labels']
    assert len(ids)==10000 and np.array_equal(np.sort(ids),np.arange(50000,60000))
    official=ROOT/'artifacts/common/cifar_c'/('CIFAR-10-C' if dataset=='cifar10' else 'CIFAR-100-C')
    native_labels=np.load(official/'labels.npy',mmap_mode='r');assert np.array_equal(native_labels[:10000][ids-50000],labels)
    columns=np.load(e.root/'cache/retained_columns.npy');names=json.loads((base/'teacher/metadata.json').read_text())['names'];concepts=json.loads((e.root/'cache/concepts.json').read_text());assert [names[i] for i in columns]==concepts
    paths={'producer':Path(__file__),'core_forward':ROOT/'methods/PiNI/spatial_noise/cub_noise.py','core_grounding':ROOT/'methods/PiNI/spatial_noise/grounding.py','renderer':ROOT/'benchmark/lf_cifar.py','feature_manifest':base/'rn18_imagenet/manifest.json','training_protocol':e.root/'cache/protocol.json','initialization_chain':e.root/'initialization_chain.json','concept_columns':e.root/'cache/retained_columns.npy','concept_bank':e.root/'cache/concepts.json','official_labels':official/'labels.npy'}
    for arm in ['mean','noise']:paths['checkpoint_'+arm]=e.root/'pair_explore/selection'/arm/'best.pt'
    for view in ['original','changed']:
        paths['primary_pixels_receipt_'+view]=e.root/'cache'/(view+'_RGB_FEATURE_COMPLETE.json')
    for kind in CORRUPTIONS:
        source=official/(kind+'.npy');marker=official/(kind+'.npy_COMPLETE.json');assert marker.exists();assert sha(source)==json.loads(marker.read_text())['sha256'];paths['official_'+kind]=source
        for severity in [1,3,5]:
            key=f'{kind}_{severity}';cached=base/'rn18_imagenet/corruptions'/f'test_{key}.npz';assert cached.exists();paths['teacher_cache_'+key]=cached
            with np.load(cached) as x:
                assert np.array_equal(x['ids'],ids) and np.array_equal(x['labels'],labels)
                assert x['teacher'].shape==(10000,len(names)) and np.array_equal(x['concept_names'],np.asarray(names))
                assert not x['known'].any() and not x['targets'].any()
    binding={'dataset':dataset,'files':{k:{'path':str(p.resolve()),'sha256':sha(p)} for k,p in paths.items()},'ids':ids.tolist(),'concept_indices':columns.tolist(),'protocol':{'fit':'none; fixed train-only/val-selected checkpoints','settings':[[k,s] for k in CORRUPTIONS for s in [1,3,5]],'all_test_rows':10000,'draws':1,'draw_seed':4242,'human_concept_annotations':False,'pixel_protocol':'original CUDA canonical native32 bicubic -> s0=q0*255 -> core q1=s0/255','foreground_GT':False}}
    return e,out,official,ids,labels,columns,binding


def bind(out,binding):
    out.mkdir(parents=True,exist_ok=True);path=out/'input_manifest.json'
    if path.exists():assert json.loads(path.read_text())==binding,'Inputs/producer changed during resume; choose a new namespace'
    else:
        assert not any(out.glob('regions/*/COMPLETE.json')),'Completed regions lack input binding'
        atomic(path,binding)


def pixel_replay_audit(e,out):
    """GPU-only canonical replay must equal existing primary s0 and q1 exactly."""
    rows=np.linspace(0,59999,256,dtype=np.int64);records=[]
    for view,name in [('original','original'),('changed','gaussian_noise_s1')]:
        native=np.load(e.common/'cache'/f'rgb_{name}.npy',mmap_mode='r');stored=np.load(e.root/'cache'/f'rgb_{view}.npy',mmap_mode='r');reader=Native32IntensityReader(native,rows)
        for batch in [8,64,128]:
            differences=0;max_error=0.
            for start in range(0,len(rows),batch):
                replay=reader[start:start+batch];reference=np.array(stored[rows[start:start+batch]])
                differences+=int(np.count_nonzero(replay.view('uint32')!=reference.view('uint32')));max_error=max(max_error,float(np.abs(replay-reference).max()))
                assert np.array_equal(replay/np.float32(255),reference/np.float32(255)),'q1 replay differs from formal training pixels'
            records.append({'view':view,'samples':256,'batch':batch,'intensity_bit_mismatches':differences,'max_intensity_error':max_error})
            atomic(out/'pixel_replay_audit.json',{'records':records,'passed':not any(x['intensity_bit_mismatches'] for x in records),'no_new_pixel_protocol':True})
            assert differences==0,'CUDA replay differs from formal s0; stop without changing pixel protocol'


def reader_for(official,kind,severity,ids):
    source=np.load(official/(kind+'.npy'),mmap_mode='r');assert source.shape==(50000,32,32,3) and source.dtype==np.uint8
    return Native32IntensityReader(source,(severity-1)*10000+ids-50000)


def prepare(e,out,official,ids,batch):
    from spatial_noise.grounding import FrozenGrounder
    pixel_replay_audit(e,out)
    grounder=FrozenGrounder(PHRASES,'/home/pengcheng/Pi-CBM/D-Pi/artifacts/models/grounding-dino-tiny',image_size=800,threshold=.25,precision='fp32')
    grounder.detector.set_use_kernels(False);assert grounder.detector.use_kernels is False
    attention=[m for m in grounder.detector.modules() if type(m).__name__=='MultiScaleDeformableAttention'];assert attention and all(m.forward.__func__.__module__=='transformers.models.grounding_dino.modeling_grounding_dino' for m in attention)
    h=hashlib.sha256()
    for key,value in grounder.detector.state_dict().items():h.update(key.encode());h.update(value.cpu().numpy().tobytes())
    assert h.hexdigest()==FROZEN_DINO_STATE_SHA256
    producer=sha(Path(__file__))
    atomic(out/'grounder_operator_audit.json',{'state_sha256':h.hexdigest(),'fallback_verified':True,'image_size':800,'threshold':.25,'precision':'fp32','phrases':PHRASES,'cache_features_dtype':'float16, same original core region cache','core_model_access_dtype':'float32 through original core tensor helper'})
    try:
        for kind in CORRUPTIONS:
            for severity in [1,3,5]:
                key=f'{kind}_{severity}';folder=out/'regions'/key;folder.mkdir(parents=True,exist_ok=True)
                if (folder/'COMPLETE.json').exists():continue
                reader=reader_for(official,kind,severity,ids);n=len(ids);native_sha=hashlib.sha256(np.array(reader.source[reader.source_rows]).tobytes()).hexdigest()
                progress_path=folder/'progress.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'completed_rows':0,'seconds':0.,'segments':[]}
                completed=int(progress['completed_rows'])
                if completed:assert progress['native_rgb_sha256']==native_sha and progress['producer_sha256']==producer and progress['batch']==batch
                mode='r+' if completed else 'w+';arrays={}
                for field,dtype,shape in [('features','float16',(n,256,14,14)),('maps','float32',(n,len(PHRASES),14,14)),('valid','bool',(n,len(PHRASES))),('boxes','float32',(n,len(PHRASES),4)),('scores','float32',(n,len(PHRASES)))]:arrays[field]=np.lib.format.open_memmap(folder/(field+'.npy'),mode=mode,dtype=dtype,shape=shape)
                detected=int(arrays['valid'][:completed].any(1).sum());began=time.time()
                for start in range(completed,n,batch):
                    end=min(start+batch,n);intensity=reader[start:end];result=grounder([Image.fromarray(im.astype('uint8').transpose(1,2,0)) for im in intensity])
                    for field,array in arrays.items():array[start:end]=result[field].cpu().numpy()
                    detected+=int(result['valid'].any(1).sum())
                    if (start-completed)//batch%50==0 or end==n:
                        for array in arrays.values():array.flush()
                        atomic(progress_path,{'completed_rows':end,'seconds':progress['seconds']+time.time()-began,'batch':batch,'producer_sha256':producer,'native_rgb_sha256':native_sha,'detector_sha256':h.hexdigest(),'segments':progress['segments']+[{'start':completed,'end':end,'batch':batch,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}]})
                    if start//batch%100==0:print(json.dumps({'phase':'official_CIFAR_C_DINO','dataset':e.name,'setting':key,'done':end,'total':n,'detected':detected,'seconds':progress['seconds']+time.time()-began,'pid':os.getpid()}),flush=True)
                # Replaying canonical pixels to hash them uses no random operations.
                intensity_digest=hashlib.sha256();q1_digest=hashlib.sha256();grounder_digest=hashlib.sha256()
                for start in range(0,n,128):
                    intensity=reader[start:start+128];intensity_digest.update(intensity.tobytes());q1_digest.update((intensity/np.float32(255)).tobytes());grounder_digest.update(intensity.astype('uint8').tobytes())
                atomic(folder/'COMPLETE.json',{'images':n,'detected_any_phrase':detected,'missing_all_phrases':n-detected,'yield':detected/n,'seconds':progress['seconds']+time.time()-began,'batch':batch,'image_size':800,'threshold':.25,'precision':'fp32','native_rgb_sha256':native_sha,'s0_intensity_sha256':intensity_digest.hexdigest(),'q1_model_input_sha256':q1_digest.hexdigest(),'uint8_grounder_input_sha256':grounder_digest.hexdigest(),'ordered_ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),'producer_sha256':producer,'detector_sha256':h.hexdigest(),'foreground_GT_available':False,'missing_masks':'actual valid=false and box maps zero; no full-image foreground substitution'})
    finally:grounder.close()


def evaluate(e,out,official,ids,labels,columns,batch,audit_batch):
    from spatial_noise.cub_noise import forward
    from spatial_noise.grounding import GroundedNoiseGenerator
    net=e.backbone().cuda().eval().requires_grad_(False);manifest=json.loads((e.common/'rn18_imagenet/manifest.json').read_text());aliases=net.weight_sha256_aliases;schemas=[key for key,value in aliases.items() if value==manifest['backbone_sha256']];assert schemas,'Frozen backbone SHA mismatch'
    clean=np.load(e.common/'teacher/original_teacher.npy',mmap_mode='r')[ids][:,columns];models={}
    for arm in ['mean','noise']:
        state=torch.load(e.root/'pair_explore/selection'/arm/'best.pt',map_location='cpu',weights_only=True);assert state['seed']==42 and state['method']==arm and not state['spatial_noise_prior']
        cbm=e.model().cuda().eval().requires_grad_(False);cbm.load_state_dict(state['cbm']);gen=GroundedNoiseGenerator(regions=len(PHRASES),mean_limit=state['mean_limit'],std_limit=state['std_limit'],spatial_noise_prior=state['spatial_noise_prior']).cuda().eval().requires_grad_(False);gen.load_state_dict(state['generator']);models[arm]=(cbm,gen)
    report={'dataset':e.name,'training_protocol':'corruption_augmented','models_fixed':True,'head_retraining':False,'backbone_hash_schema':schemas,'inference_draws':1,'inference_seed':4242,'inference_batch':batch,'native_concept_annotations':False,'settings':{}}
    with torch.no_grad():
        for kind in CORRUPTIONS:
            for severity in [1,3,5]:
                key=f'{kind}_{severity}';folder=out/'regions'/key;assert (folder/'COMPLETE.json').exists();reader=reader_for(official,kind,severity,ids);cached=np.load(e.common/'rn18_imagenet/corruptions'/f'test_{key}.npz');teacher=cached['teacher'][:,columns]
                # The same approved q1 vs LF-q0 fp32 backend gate, not an exact-feature claim.
                feature_parts=[]
                for start in range(0,len(ids),audit_batch):feature_parts.append(net(torch.as_tensor(reader[start:start+audit_batch],device='cuda')/255).mean((2,3)).float().cpu().numpy())
                features=np.concatenate(feature_parts);reference=cached['features'];relative=float(np.linalg.norm(features.astype('float64')-reference.astype('float64'))/np.linalg.norm(reference.astype('float64')));assert relative<1e-3,'q1 backbone audit failed; do not relax pixel precision'
                atomic(folder/'feature_precision_audit.json',{'all_rows':len(ids),'relative_l2':relative,'max_abs':float(np.max(np.abs(features-reference))),'approved_gate':1e-3,'batch':audit_batch,'backbone_sha256':manifest['backbone_sha256'],'source_weights_exact':True,'exact_feature_identity_claimed':False,'pixel_roundtrip_max_vs_q0':5.960464477539063e-8})
                arrays={'rgb_original':reader};grounded={'original':{field:np.load(folder/(field+'.npy'),mmap_mode='r') for field in ['features','maps']}};report['settings'][key]={}
                for arm,(cbm,gen) in models.items():
                    path=out/'exports'/arm/f'test_{key}.npz';path.parent.mkdir(parents=True,exist_ok=True);metric_path=path.with_suffix('.metrics.json')
                    if path.exists() and metric_path.exists():report['settings'][key][arm]=json.loads(metric_path.read_text());continue
                    torch.manual_seed(4242);ys=[];cs=[]
                    for start in range(0,len(ids),batch):
                        y,c,_,_=forward(net,cbm,gen,arrays,grounded,np.arange(start,min(start+batch,len(ids))),'original',arm);ys.append(y.cpu().numpy());cs.append(c.cpu().numpy())
                    y=np.concatenate(ys);c=np.concatenate(cs);prob=1/(1+np.exp(-np.clip(c,-80,80)));assert np.isfinite(y).all() and np.isfinite(c).all();unknown=np.zeros_like(prob,dtype=bool)
                    np.savez_compressed(path,ids=ids,labels=labels,logits=y,concept_logits=c,concepts=prob,targets=np.zeros_like(prob),known=unknown,teacher=teacher,clean_teacher=clean,concept_indices=columns,eligible=np.ones(len(ids),bool))
                    metrics={'classification':classification_metrics(y,labels),'CLIP_teacher_raw_logits':teacher_fidelity(c,teacher),'clean_teacher_invariance_raw_logits':teacher_fidelity(c,clean),'native_concept_mAP':None,'human_concept_annotations':False,'inference_draws':1,'inference_seed':4242,'setting':key,'all_test_rows':len(ids)};atomic(metric_path,metrics);report['settings'][key][arm]=metrics
                    print(json.dumps({'phase':'official_CIFAR_C_export','dataset':e.name,'setting':key,'method':arm,'images':len(ids),'accuracy':metrics['classification']['accuracy'],'pid':os.getpid()}),flush=True)
    assert len(report['settings'])==12 and all(len(x)==2 for x in report['settings'].values());atomic(out/'results.json',report);atomic(out/'CIFAR_C_COMPLETE.json',{'dataset':e.name,'settings':12,'rows_per_setting':10000,'arms':2,'draws':1,'training_performed':False,'source_manifest_sha256':sha(out/'input_manifest.json')})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['audit','prepare','evaluate','run']);p.add_argument('--dataset',choices=['cifar10','cifar100'],required=True);p.add_argument('--dino-batch',type=int,default=8);p.add_argument('--inference-batch',type=int,default=64);p.add_argument('--audit-batch',type=int,default=128);a=p.parse_args();torch.set_num_threads(4)
    e,out,official,ids,labels,columns,binding=context(a.dataset)
    if a.action=='audit':
        receipt={'dataset':a.dataset,'CPU_readiness_verified':True,'official_settings':12,'rows_per_setting':10000,'future_output':str(out),'CUDA_pixel_replay_still_requires_exact_verification':True,'GPU_jobs_started':False,'binding':binding}
        atomic(ROOT/'artifacts/audit'/f'{a.dataset}_pini_official_c_readiness.json',receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k!='binding'}),flush=True);return
    bind(out,binding)
    if a.action in ['prepare','run']:prepare(e,out,official,ids,a.dino_batch)
    if a.action in ['evaluate','run']:evaluate(e,out,official,ids,labels,columns,a.inference_batch,a.audit_batch)


if __name__=='__main__':main()
