"""Class-only controls on all completed shared backbone caches."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .noncbm import load_split, run
from .metrics import classification_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backbones',nargs='+',default=['rn50_imagenet','clip_RN50'])
    p.add_argument('--datasets',nargs='+',default=['rival10','cub','waterbirds'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(3)
    for dataset in a.datasets:
        for backbone in a.backbones:
            source=Path('artifacts/common')/dataset/backbone
            if not (source/'manifest.json').exists():continue
            for paired in [False,True]:
                if paired:
                    inputs=source/'paired';inputs.mkdir(exist_ok=True)
                    for split in ['train','val']:
                        views=[load_split(source/f'{split}_{v}.npz') for v in ['original','changed']]
                        assert np.array_equal(views[0]['ids'],views[1]['ids'])
                        assert np.array_equal(views[0]['labels'],views[1]['labels'])
                        np.savez_compressed(inputs/f'{split}.npz',**{k:np.concatenate([v[k] for v in views]) for k in ['features','labels','ids']})
                    train,val=inputs/'train.npz',inputs/'val.npz'
                else:train,val=source/'train_original.npz',source/'val_original.npz'
                for seed in a.seeds:
                    out=Path('artifacts/noncbm')/f"{dataset}_{backbone}{'_paired' if paired else ''}"/f'seed{seed}'
                    if not (out/'results.json').exists():run(train,val,source/'test_original.npz',out,a.device,seed)
                    state=torch.load(out/'best.pt',weights_only=True)
                    changed=load_split(source/'test_changed.npz')
                    head=nn.Linear(changed['features'].shape[1],state['head']['weight'].shape[0]).to(a.device)
                    head.load_state_dict(state['head']);head.eval()
                    with torch.no_grad():
                        logits=np.concatenate([head(((x-state['mean'])/state['std']).to(a.device)).cpu().numpy()
                            for x in torch.as_tensor(changed['features']).float().split(2048)])
                    np.savez_compressed(out/'changed_predictions.npz',logits=logits,labels=changed['labels'],ids=changed['ids'])
                    report=json.loads((out/'results.json').read_text())
                    report.update(changed=classification_metrics(logits,changed['labels']),
                                  backbone=backbone,training_device=a.device,
                                  train_views=['original','changed'] if paired else ['original'],
                                  shared_backbone_manifest=str(source/'manifest.json'))
                    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps({'dataset':dataset,'backbone':backbone,'paired':paired,'seed':seed,'shift_accuracy':report['changed']['accuracy']}),flush=True)


if __name__=='__main__':main()
