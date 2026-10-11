"""Fresh CIFAR PiNI with frozen cores, train-only CLIP-Otsu proxy attrs."""
import argparse,json,os,time,hashlib
# Explicit official switch before Transformers is imported anywhere downstream.
# The independently audited fused operator changed threshold decisions.
os.environ['USE_HUB_KERNELS']='NO'
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from benchmark.pini import import_core,CORE
from benchmark.pini_matched import ImageNetSpatialBackbone,initialize,write,digest
from benchmark.pini_cifar_adapter import fit_otsu
from benchmark.pini_cifar_audit import PHRASES
from benchmark.lf_cifar import resize224
from benchmark.metrics import classification_metrics,concept_metrics,teacher_fidelity,nec
ROOT=Path(__file__).resolve().parents[1]
CORRUPTION_SOURCE='https://raw.githubusercontent.com/hendrycks/robustness/master/ImageNet-C/create_c/make_cifar_c.py'
FROZEN_DINO_STATE_SHA256='3c469281e4afb12b8f5ad68351060e3dec548adacd2fd9af36edf53cda586c26'


def atomic_write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    write(temporary,value)
    os.replace(temporary,path)


def gaussian_noise_s1(raw,row):
    """Official CIFAR-C Gaussian severity1 formula, independent row RNG."""
    rng=np.random.RandomState(42+int(row))
    hwc=np.asarray(raw).transpose(1,2,0)
    return (np.clip(hwc/255.+rng.normal(size=hwc.shape,scale=.04),0,1)*255).astype(np.uint8).transpose(2,0,1)


class Experiment:
    def __init__(self,dataset):
        import_core();self.name=dataset;self.root=ROOT/'artifacts/pini_matched'/dataset
        self.common=ROOT/'artifacts/common'/dataset;self.phrases=PHRASES
        self.feature_dim=512;self.detector_precision='fp32';self.classes=10 if dataset=='cifar10' else 100
        self.k=len(json.loads((self.root/'cache/concepts.json').read_text()))
    def load(self):
        cache=self.root/'cache';protocol=json.loads((cache/'protocol.json').read_text())
        return {k:np.load(cache/(k+'.npy'),mmap_mode='r') for k in ['labels','attrs','rgb_original','foreground','features_original']},{k:np.asarray(v,np.int64) for k,v in protocol['splits'].items()},protocol
    def model(self):
        from spatial_noise.cub_aligned import SimpleCBM
        return SimpleCBM(self.k,self.classes)
    def backbone(self):return ImageNetSpatialBackbone('resnet18')


def prepare(dataset,batch=4,rgb_only=False,prepare_only=False,region_views=None):
    import_core()
    from spatial_noise.grounding import FrozenGrounder
    base=ROOT/'artifacts/common'/dataset;root=ROOT/'artifacts/pini_matched'/dataset;cache=root/'cache';cache.mkdir(parents=True,exist_ok=True)
    assert (base/'teacher/original_COMPLETE.json').exists()
    protocol=json.loads((base/'cache/protocol.json').read_text());ids=np.asarray(protocol['splits']['train'],np.int64)
    teacher=np.load(base/'teacher/original_teacher.npy',mmap_mode='r');names=json.loads((base/'cache/concepts.json').read_text())
    thresholds,keep,stats=fit_otsu(teacher,ids)
    np.save(cache/'thresholds.npy',thresholds);np.save(cache/'retained_columns.npy',np.flatnonzero(keep))
    np.save(cache/'attrs.npy',teacher[:,keep]>thresholds[keep]);write(cache/'concepts.json',[n for n,k in zip(names,keep) if k])
    stats.update(raw_concept_count=len(names),retained_concept_count=int(keep.sum()),target_type='train-only Otsu binary CLIP proxy, no human concept annotations',native_known=False)
    write(cache/'supervision_audit.json',stats)
    print(json.dumps({'phase':'Otsu','dataset':dataset,'keep':int(keep.sum()),'raw':len(names),'pid':os.getpid()}),flush=True)
    labels=cache/'labels.npy'
    if not labels.exists():labels.symlink_to((base/'cache/labels.npy').resolve())
    raw=np.load(base/'cache/rgb_original.npy',mmap_mode='r');n=len(raw)
    changed_path=base/'cache/rgb_gaussian_noise_s1.npy'
    if not (base/'cache/gaussian_noise_s1_COMPLETE.json').exists():
        changed=np.lib.format.open_memmap(changed_path,mode='w+',dtype='uint8',shape=raw.shape)
        for row in range(n):changed[row]=gaussian_noise_s1(raw[row],row)
        changed.flush();write(base/'cache/gaussian_noise_s1_COMPLETE.json',{'shape':list(raw.shape),'sha256':digest(changed_path),'source':CORRUPTION_SOURCE,'formula':'clip(native32/255 + RandomState(42+global_row).normal(0,.04),0,1)*255 cast uint8','official_test_C_file_used':False,'generated_from':'each corresponding original raw row; no cross-split input','seed':42})
    changed=np.load(changed_path,mmap_mode='r')
    net=ImageNetSpatialBackbone().cuda();common=json.loads((base/'rn18_imagenet/manifest.json').read_text())
    matched=[s for s,h in net.weight_sha256_aliases.items() if h==common['backbone_sha256']];assert matched
    global_original=np.load(base/'rn18_imagenet/original_features.npy',mmap_mode='r')
    for view,native in [('original',raw),('changed',changed)]:
        finish=cache/(view+'_RGB_FEATURE_COMPLETE.json')
        if finish.exists():continue
        rgb=np.lib.format.open_memmap(cache/('rgb_'+view+'.npy'),mode='w+',dtype='float32',shape=(n,3,224,224))
        features=np.lib.format.open_memmap(cache/('features_'+view+'.npy'),mode='w+',dtype='float32',shape=(n,512,1,1))
        maxerr=0.;difference_squared=reference_squared=0.;started=time.time()
        with torch.no_grad():
            for start in range(0,n,128):
                images=resize224(native[start:start+128],'cuda');stored=images*255
                rgb[start:start+len(images)]=stored.cpu().numpy()
                pooled=net(stored/255).mean((2,3)).cpu().numpy()
                features[start:start+len(images),:,0,0]=pooled
                if view=='original':
                    reference=np.asarray(global_original[start:start+len(images)],dtype=np.float64)
                    difference=pooled.astype(np.float64)-reference
                    maxerr=max(maxerr,float(np.abs(difference).max()))
                    difference_squared+=float(np.square(difference).sum());reference_squared+=float(np.square(reference).sum())
                if start%6400==0:print(json.dumps({'phase':'RGB_features','dataset':dataset,'view':view,'done':start+len(images),'pid':os.getpid()}),flush=True)
        rgb.flush();features.flush()
        relative_l2=(difference_squared/max(reference_squared,1e-30))**.5 if view=='original' else None
        if view=='original':assert relative_l2<1e-3,{'relative_l2':relative_l2,'max_abs':maxerr}
        write(finish,{'shape':list(rgb.shape),'dtype':'float32 intensity0..255 for unchanged core /255','features_layout':'backbone global mean as N5121x1; live backbone uses native layer4 spatial','shared_global_max_abs_error':maxerr if view=='original' else None,'shared_global_relative_l2':relative_l2,'audit_batch':128,'verification_gate':'strict weight SHA plus relative L2 < 1e-3; audited TF32 sensitivity from core-required intensity roundtrip, not exact feature equality','pixel_roundtrip_max_abs':5.960464477539063e-08,'feature_sha256':digest(cache/('features_'+view+'.npy')),'rgb_sha256':digest(cache/('rgb_'+view+'.npy')),'seconds':time.time()-started})
    del net;torch.cuda.empty_cache()
    if rgb_only:
        # Initialization and RGB preparation may precede localization; no full
        # protocol/region COMPLETE is produced by this partial preparation.
        return Experiment(dataset)
    grounder=FrozenGrounder(phrases=PHRASES,model_path='/home/pengcheng/Pi-CBM/D-Pi/artifacts/models/grounding-dino-tiny',image_size=800,threshold=.25,precision='fp32')
    grounder.detector.set_use_kernels(False)
    assert grounder.detector.use_kernels is False
    attention=[m for m in grounder.detector.modules() if type(m).__name__=='MultiScaleDeformableAttention']
    assert attention and all(m.forward.__func__.__module__=='transformers.models.grounding_dino.modeling_grounding_dino' for m in attention)
    detector_digest=hashlib.sha256()
    for key,value in grounder.detector.state_dict().items():
        detector_digest.update(key.encode());detector_digest.update(value.cpu().numpy().tobytes())
    detector_sha=detector_digest.hexdigest()
    assert detector_sha==FROZEN_DINO_STATE_SHA256
    write(root/'grounder_operator_audit.json',{'USE_HUB_KERNELS':os.environ['USE_HUB_KERNELS'],'model_use_kernels':grounder.detector.use_kernels,'detector_state_sha256':detector_sha,'attention_forward_modules':sorted({m.forward.__func__.__module__ for m in attention}),'attention_layers':len(attention),'fused_kernel_enabled':False,'forward_code_sha256':hashlib.sha256(attention[0].forward.__func__.__code__.co_code).hexdigest(),'precision':'fp32','image_size':800,'threshold':.25,'failed_optimization_audit':str(ROOT/'artifacts/cifar_feasibility/kernel_audit.json')})
    for view in (region_views or ['original','changed']):
        folder=root/'regions'/view;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'COMPLETE.json').exists():continue
        rgb=np.load(cache/('rgb_'+view+'.npy'),mmap_mode='r')
        progress_path=folder/'progress.json'
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'completed_rows':0,'seconds':0.,'batches':[]}
        if progress['completed_rows']:
            assert progress['ordered_input_rgb_sha256']==json.loads((cache/(view+'_RGB_FEATURE_COMPLETE.json')).read_text())['rgb_sha256']
        completed=int(progress['completed_rows']);mode='r+' if completed else 'w+'
        features=np.lib.format.open_memmap(folder/'features.npy',mode=mode,dtype='float16',shape=(n,256,14,14))
        maps=np.lib.format.open_memmap(folder/'maps.npy',mode=mode,dtype='float32',shape=(n,len(PHRASES),14,14))
        valid=np.lib.format.open_memmap(folder/'valid.npy',mode=mode,dtype='bool',shape=(n,len(PHRASES)))
        boxes=np.lib.format.open_memmap(folder/'boxes.npy',mode=mode,dtype='float32',shape=(n,len(PHRASES),4))
        scores=np.lib.format.open_memmap(folder/'scores.npy',mode=mode,dtype='float32',shape=(n,len(PHRASES)))
        foreground=np.lib.format.open_memmap(cache/'foreground.npy',mode=mode,dtype='bool',shape=(n,224,224)) if view=='original' else None
        detected=int(valid[:completed].any(1).sum());began=time.time();previous_seconds=progress['seconds'];previous_batches=progress['batches']
        torch.cuda.reset_peak_memory_stats()
        input_sha=json.loads((cache/(view+'_RGB_FEATURE_COMPLETE.json')).read_text())['rgb_sha256']
        for start in range(completed,n,batch):
            # Grounder's unmodified PIL interface consumes byte RGB. Quantization
            # belongs only to detector preprocessing; classifier RGB stays float.
            images=[Image.fromarray(np.clip(x.transpose(1,2,0),0,255).astype(np.uint8)) for x in rgb[start:start+batch]]
            r=grounder(images);end=start+len(images)
            for key,array in [('features',features),('maps',maps),('valid',valid),('boxes',boxes),('scores',scores)]:array[start:end]=r[key].cpu().numpy()
            detected+=int(r['valid'].any(1).sum())
            if foreground is not None:
                predicted=r['maps'].amax(1).cpu().numpy()>0
                foreground[start:end]=predicted.repeat(16,1).repeat(16,2)
            if start%(batch*100)==0:print(json.dumps({'phase':'DINO','dataset':dataset,'view':view,'done':end,'total':n,'detected':detected,'seconds':time.time()-began,'pid':os.getpid()}),flush=True)
            if (start-completed)%(batch*100)==0:
                for array in [features,maps,valid,boxes,scores]:array.flush()
                if foreground is not None:foreground.flush()
                atomic_write(progress_path,{'completed_rows':end,'seconds':previous_seconds+time.time()-began,'batches':previous_batches+[{'start':completed,'end':end,'batch':batch,'input_sha256':input_sha,'detector_sha256':detector_sha,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}],'ordered_input_rgb_sha256':input_sha})
        for array in [features,maps,valid,boxes,scores]:array.flush()
        if foreground is not None:foreground.flush()
        atomic_write(folder/'COMPLETE.json',{'images':n,'detected_any_phrase':detected,'missing_all_phrases':n-detected,'yield':detected/n,'phrases':PHRASES,'threshold':.25,'image_size':800,'precision':'fp32','batch':batch,'batches':previous_batches+[{'start':completed,'end':n,'batch':batch,'input_sha256':input_sha,'detector_sha256':detector_sha,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}],'foreground_mask':'union of valid predicted DINO box maps, false for missing detections; diagnostic only, no GT available','seconds':previous_seconds+time.time()-began})
    grounder.close();del grounder;torch.cuda.empty_cache()
    if prepare_only:return Experiment(dataset)
    protocol.update(train_views=['original','artificial CIFAR-C Gaussian noise severity1'],changed_view='artificial_gaussian_noise_s1',official_CIFAR_C_test_used_for_fit=False,human_concept_annotations=False,concept_supervision='train-only Otsu CLIP pseudo-binary; not native attrs',foreground_masks='DINO predicted union only; missing detections all false; not native mask',backbone_label_supervision='ImageNet1K only, no task fitting',backbone_hash_schema=matched[0],task_fit_sample_ids=[],selection='val only',fit='train only',pixel_protocol='shared canonical float bicubic; PiNI core additionally uses float32 intensity *255 /255 roundtrip',feature_audit={v:json.loads((cache/(v+'_RGB_FEATURE_COMPLETE.json')).read_text()) for v in ['original','changed']},backend_batch_sensitivity_audit=str(ROOT/'artifacts/cifar_feasibility/precision_audit.json'))
    write(cache/'protocol.json',protocol);write(cache/'BACKGROUND_COMPLETE.json',{'view':'artificial gaussian noise severity1','background_intervention':False,'corruption_protocol':str(base/'cache/gaussian_noise_s1_COMPLETE.json')});write(root/'input_manifest.json',protocol)
    return Experiment(dataset)


@torch.no_grad()
def export(e,method,batch=64):
    from spatial_noise.cub_noise import inputs,forward
    from spatial_noise.grounding import GroundedNoiseGenerator
    arrays,splits,protocol,grounded=inputs(experiment=e)
    state=torch.load(e.root/'pair_explore/selection'/method/'best.pt',map_location='cpu',weights_only=True)
    cbm=e.model().cuda().eval();cbm.load_state_dict(state['cbm'])
    gen=GroundedNoiseGenerator(regions=len(PHRASES),mean_limit=state['mean_limit'],std_limit=state['std_limit'],spatial_noise_prior=False).cuda().eval();gen.load_state_dict(state['generator'])
    net=e.backbone().cuda().eval();teacher=np.load(e.common/'teacher/original_teacher.npy',mmap_mode='r');columns=np.load(e.root/'cache/retained_columns.npy')
    assert (e.common/'teacher/gaussian_noise_s1_COMPLETE.json').exists()
    changed_teacher=np.load(e.common/'teacher/gaussian_noise_s1_teacher.npy',mmap_mode='r')
    out=e.root/'exports'/method;out.mkdir(parents=True,exist_ok=True);results={}
    for split,ids in splits.items():
        for view in ['original','changed']:
            torch.manual_seed(4242);ys=[];cs=[]
            for start in range(0,len(ids),batch):
                y,c,_,_=forward(net,cbm,gen,arrays,grounded,ids[start:start+batch],view,method);ys.append(y.cpu().numpy());cs.append(c.cpu().numpy())
            y=np.concatenate(ys);c=np.concatenate(cs);prob=1/(1+np.exp(-np.clip(c,-80,80)))
            targets=np.zeros_like(prob);known=np.zeros_like(prob,dtype=bool);pseudo=np.asarray(arrays['attrs'][ids]);clean_t=np.asarray(teacher[ids][:,columns]);t=np.asarray((teacher if view=='original' else changed_teacher)[ids][:,columns])
            np.savez_compressed(out/(split+'_'+view+'.npz'),ids=ids,labels=arrays['labels'][ids],concepts=prob,concept_logits=c,logits=y,targets=targets,known=known,training_targets=pseudo,teacher=t,clean_teacher=clean_t,concept_indices=columns,eligible=np.ones(len(ids),bool))
            if view=='changed':
                alias=out/(split+'_gaussian_noise_s1.npz')
                alias.unlink(missing_ok=True)
                os.link(out/(split+'_changed.npz'),alias)
            metrics={'classification':classification_metrics(y,arrays['labels'][ids]),'CLIP_teacher_raw_logits':teacher_fidelity(c,t),'CLIP_teacher_probabilities':teacher_fidelity(prob,t),'clean_teacher_invariance_raw_logits':teacher_fidelity(c,clean_t),'CLIP_Otsu_proxy_AP':concept_metrics(prob,pseudo),'native_concept_mAP':None,'teacher_reference':'same-view cosine for primary fidelity; clean original cosine separately for invariance','human_concept_annotations':False}
            results[split+'_'+view]=metrics
            print(json.dumps({'phase':'CIFAR_export','dataset':e.name,'method':method,'split':split,'view':view,'accuracy':metrics['classification']['accuracy'],'pid':os.getpid()}),flush=True)
    np.save(out/'classifier_weights.npy',cbm.classifier.weight.cpu().numpy())
    results.update(nec=nec(cbm.classifier.weight.cpu().numpy()),concept_names=json.loads((e.root/'cache/concepts.json').read_text()),training_seed=42,inference_draws=1,inference_seed=4242,fit_split='train',selection_split='val',train_protocol='corruption_augmented',train_views=['original','gaussian_noise_s1'],changed_view={'canonical_name':'gaussian_noise_s1','core_alias':'changed','sigma':.04,'artificial_train_corruption':True,'background_intervention':False},native_annotations_available=False,algorithm_version='frozen latest pair consistency30 coverage_reward',checkpoint=str(e.root/'pair_explore/selection'/method/'best.pt'))
    write(out/'results.json',results)


def canonicalize_completed_exports(dataset,method):
    """Add explicit corruption names to finished exports from older live code."""
    out=ROOT/'artifacts/pini_matched'/dataset/'exports'/method
    report=out/'results.json'
    if not report.exists():return False
    for split in ['train','val','test']:
        source=out/(split+'_changed.npz');alias=out/(split+'_gaussian_noise_s1.npz')
        if not source.exists():return False
        if not alias.exists():os.link(source,alias)
        assert os.path.samefile(source,alias)
    try:result=json.loads(report.read_text())
    except json.JSONDecodeError:return False
    result.update(train_protocol='corruption_augmented',train_views=['original','gaussian_noise_s1'],changed_view={'canonical_name':'gaussian_noise_s1','core_alias':'changed','sigma':.04,'artificial_train_corruption':True,'background_intervention':False})
    temporary=out/'results.canonical.tmp'
    write(temporary,result);os.replace(temporary,report)
    write(out/'CANONICAL_EXPORT_COMPLETE.json',{'dataset':dataset,'method':method,'train_protocol':'corruption_augmented','splits':['train','val','test'],'views':['original','gaussian_noise_s1'],'compatibility_alias':'changed is same-inode hardlink of gaussian_noise_s1','human_concept_annotations':False})
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['cifar10','cifar100']);p.add_argument('--dino-batch',type=int,default=4);p.add_argument('--rgb-only',action='store_true');p.add_argument('--prepare-only',action='store_true');p.add_argument('--region-views',nargs='+',choices=['original','changed']);a=p.parse_args();torch.set_num_threads(4)
    if a.region_views and not a.prepare_only:p.error('--region-views requires --prepare-only')
    for dataset in a.datasets:
        e=prepare(dataset,a.dino_batch,rgb_only=a.rgb_only,prepare_only=a.prepare_only,region_views=a.region_views)
        if a.rgb_only or a.prepare_only:continue
        initialize(e,1000,10,64)
        from spatial_noise.noise_pair import train
        for method in ['mean','noise']:train(e,method,'selection',epochs=5,pairs=32);export(e,method)
        write(e.root/'CIFAR_COMPLETE.json',{'dataset':dataset,'methods':['mean','noise'],'seed':42,'human_concept_annotations':False,'train_only':True})

if __name__=='__main__':main()
