"""Run original supervised simple while independent DINO preparation continues."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from benchmark.pini import import_core
from benchmark.pini_cifar import Experiment,ROOT
from benchmark.pini_cifar_adapter import fit_otsu
from benchmark.pini_matched import ImageNetSpatialBackbone,write,digest
from benchmark.lf_cifar import resize224


def run(dataset,batch=128):
    import_core();torch.set_num_threads(4)
    base=ROOT/'artifacts/common'/dataset;out=ROOT/'artifacts/pini_matched'/dataset;cache=out/'cache';cache.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((base/'cache/protocol.json').read_text());ids=np.asarray(protocol['splits']['train'],np.int64)
    y=np.load(base/'teacher/original_teacher.npy',mmap_mode='r');names=json.loads((base/'cache/concepts.json').read_text());t,keep,stats=fit_otsu(y,ids)
    np.save(cache/'thresholds.npy',t);np.save(cache/'retained_columns.npy',np.flatnonzero(keep));np.save(cache/'attrs.npy',y[:,keep]>t[keep]);write(cache/'concepts.json',[n for n,k in zip(names,keep) if k]);write(cache/'supervision_audit.json',{**stats,'raw_concept_count':len(names),'retained_concept_count':int(keep.sum()),'native_known':False,'target_type':'train-only CLIP Otsu pseudo-binary'})
    path=cache/'simple_initialization_features.npy';finish=out/'simple_input_manifest.json'
    if not finish.exists():
        net=ImageNetSpatialBackbone().cuda();manifest=json.loads((base/'rn18_imagenet/manifest.json').read_text());assert manifest['backbone_sha256'] in net.weight_sha256_aliases.values()
        raw=np.load(base/'cache/rgb_original.npy',mmap_mode='r');common=np.load(base/'rn18_imagenet/original_features.npy',mmap_mode='r');features=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(raw),512,1,1));delta=reference=0.;maximum=0.;began=time.time()
        with torch.no_grad():
            for start in range(0,len(raw),batch):
                q=resize224(raw[start:start+batch],'cuda');f=net(q*255/255).mean((2,3)).cpu().numpy();features[start:start+len(f),:,0,0]=f
                original=np.asarray(common[start:start+len(f)],np.float64);d=f.astype(np.float64)-original;delta+=(d*d).sum();reference+=(original*original).sum();maximum=max(maximum,float(abs(d).max()))
                if start%10000==0:print(json.dumps({'phase':'simple_init_q1_features','done':start+len(f),'dataset':dataset,'pid':os.getpid()}),flush=True)
        features.flush();relative=float((delta/reference)**.5);assert relative<1e-3,relative
        write(finish,{'features':str(path.resolve()),'sha256':digest(path),'batch':batch,'source_manifest':str((base/'rn18_imagenet/manifest.json').resolve()),'backbone_sha_strict':True,'relative_L2_vs_common128':relative,'max_abs_vs_common128':maximum,'pixel_protocol':'same canonical bicubic plus required intensity*255/255','seconds':time.time()-began,'final_warmup_features':'main cache independently encoded batch128; numerical batch differences audited'})
        del net;torch.cuda.empty_cache()
    class SimpleOnlyExperiment(Experiment):
        def load(self):
            return {'features_original':np.load(path,mmap_mode='r'),'attrs':np.load(cache/'attrs.npy',mmap_mode='r'),'labels':np.load(base/'cache/labels.npy',mmap_mode='r')},{k:np.asarray(v,np.int64) for k,v in protocol['splits'].items()},protocol
    e=SimpleOnlyExperiment(dataset);write(out/'simple_refit/seed42/COMPLETE.json',{'skipped':True,'reason':'train-only no refit checkpoint'})
    from spatial_noise.rn18_datasets import simple
    print('Actual supervised simple PID',os.getpid(),flush=True);simple(e,seed=42,steps=1000)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--batch',type=int,default=128);a=p.parse_args();run(a.dataset,a.batch)
