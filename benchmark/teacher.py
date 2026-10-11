"""Cache a frozen CLIP teacher on the exact PiNI input images and concept bank."""
import argparse
import hashlib
import json
from pathlib import Path

import clip
import numpy as np
import torch
from torch.nn import functional as F


def phrases(names):
    result=[]
    for name in names:
        if '::' in name:
            group,value=name.split('::')
            group=group.removeprefix('has_').replace('_',' ')
            value=value.replace('_',' ')
            result.append(f'a bird with {value} {group}')
        else:
            result.append('an object with '+name.replace('-',' ').replace('_',' '))
    return result


def run(dataset,source,out,batch=128,device='cuda'):
    source,out=Path(source),Path(out);out.mkdir(parents=True,exist_ok=True)
    names=json.loads((source/'cache/concepts.json').read_text())
    words=phrases(names)
    model,_=clip.load('/root/.cache/clip/ViT-B-16.pt',device=device,jit=False)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        text=F.normalize(model.encode_text(clip.tokenize(words).to(device)).float(),dim=-1)
    np.save(out/'text.npy',text.cpu().numpy())
    protocol=json.loads((source/'cache/protocol.json').read_text())
    for view in ['original','changed']:
        finish=out/(view+'_COMPLETE.json')
        if finish.exists():continue
        rgb=np.load(source/'cache'/('rgb_'+view+'.npy'),mmap_mode='r')
        target=np.lib.format.open_memmap(out/(view+'_teacher.npy'),mode='w+',dtype='float32',shape=(len(rgb),len(names)))
        mean=torch.tensor([.48145466,.4578275,.40821073],device=device)[None,:,None,None]
        std=torch.tensor([.26862954,.26130258,.27577711],device=device)[None,:,None,None]
        with torch.no_grad():
            for start in range(0,len(rgb),batch):
                image=torch.tensor(np.array(rgb[start:start+batch]),device=device).float()/255
                features=F.normalize(model.encode_image((image-mean)/std).float(),dim=-1)
                target[start:start+len(image)]=(features@text.T).cpu().numpy()
                if start%(batch*50)==0:print(json.dumps({'dataset':dataset,'view':view,'images':start+len(image),'total':len(rgb)}),flush=True)
        target.flush()
        finish.write_text(json.dumps({'images':len(rgb),'concepts':len(names),'view':view})+'\n')
    metadata={'dataset':dataset,'teacher':'CLIP ViT-B/16','names':names,'phrases':words,
              'image_source':str(source.resolve()),'preprocess':'exact cached 224x224 PiNI RGB plus CLIP normalization; no second crop',
              'split_counts':{s:len(ids) for s,ids in protocol['splits'].items()},
              'concept_name_sha256':hashlib.sha256(json.dumps(names).encode()).hexdigest(),
              'targets':'frozen CLIP cosine similarity; not human concept labels'}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--source',required=True)
    p.add_argument('--out',required=True);p.add_argument('--batch',type=int,default=128);p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4);run(a.dataset,a.source,a.out,a.batch,a.device)


if __name__=='__main__':main()
