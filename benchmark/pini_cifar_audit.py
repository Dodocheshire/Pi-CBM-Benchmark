"""External CIFAR detector-yield pilot; no localization ground truth is available."""
from pathlib import Path
import argparse,json,os,hashlib
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10,CIFAR100
from benchmark.pini import import_core

PHRASES=['object','animal','vehicle','person','head','body','wing','wheel']

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=32);p.add_argument('--batch',type=int,default=4);p.add_argument('--output',type=Path,default=Path('artifacts/cifar_feasibility/dino'));a=p.parse_args()
    import_core()
    from spatial_noise.grounding import FrozenGrounder
    a.output.mkdir(parents=True,exist_ok=True)
    detector=FrozenGrounder(phrases=PHRASES,model_path='/home/pengcheng/Pi-CBM/D-Pi/artifacts/models/grounding-dino-tiny',image_size=800,threshold=.25,precision='fp32')
    summary={'pid':os.getpid(),'seed':42,'phrases':PHRASES,'threshold':.25,'image_size':800,'precision':'fp32','limitations':'Detection yield only; CIFAR has no ground-truth boxes/masks, so localization accuracy is unknown. Predicted box maps are not native foreground annotations. Fixed phrase bank has no image label input.','datasets':{}}
    for name,cls in [('cifar10',CIFAR10),('cifar100',CIFAR100)]:
        ds=cls('/home/pengcheng/VLG-CBM/datasets',train=True,download=False)
        ids=np.sort(np.random.default_rng(42).choice(len(ds),a.samples,replace=False))
        summary['datasets'][name]={}
        for size in [32,224]:
            arrays={key:[] for key in ['features','maps','valid','scores','boxes']};times=[]
            for offset in range(0,len(ids),a.batch):
                images=[Image.fromarray(ds.data[i]).resize((size,size),Image.Resampling.BICUBIC) for i in ids[offset:offset+a.batch]]
                r=detector(images);times.append(r['detector_forward_s'])
                for key in arrays:arrays[key].append(r[key].cpu().numpy())
                print(json.dumps({'dataset':name,'input_size':size,'done':min(offset+a.batch,len(ids)),'pid':os.getpid()}),flush=True)
            arrays={key:np.concatenate(v) for key,v in arrays.items()};valid=arrays['valid']
            np.savez_compressed(a.output/(name+'_'+str(size)+'.npz'),ids=ids,**arrays)
            summary['datasets'][name][str(size)]={'samples':len(ids),'any_detection_rate':float(valid.any(1).mean()),'object_detection_rate':float(valid[:,0].mean()),'phrase_detection_rate':dict(zip(PHRASES,valid.mean(0).tolist())),'mean_valid_phrases':float(valid.sum(1).mean()),'mean_score':float(arrays['scores'].mean()),'detector_seconds':sum(times),'ordered_raw_rgb_sha256':hashlib.sha256(ds.data[ids].tobytes()).hexdigest(),'feature_shape':list(arrays['features'].shape),'map_shape':list(arrays['maps'].shape)}
            (a.output/'audit.json').write_text(json.dumps(summary,indent=2))
    detector.close()

if __name__=='__main__':main()
