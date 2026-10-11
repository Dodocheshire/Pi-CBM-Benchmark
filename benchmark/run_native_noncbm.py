"""Shared ImageNet RN18 class-only controls on native concept datasets."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .noncbm import run,load_split
from .metrics import classification_metrics

torch.set_num_threads(4)
for dataset in ['rival10','cub','waterbirds']:
    source=Path('artifacts/common')/dataset/'rn18_imagenet'
    for seed in [42,43,44]:
        out=Path('artifacts/noncbm')/(dataset+'_rn18_imagenet')/('seed'+str(seed))
        if not (out/'results.json').exists():
            run(source/'train_original.npz',source/'val_original.npz',source/'test_original.npz',out,'cuda',seed)
        state=torch.load(out/'best.pt',weights_only=True)
        changed=load_split(source/'test_changed.npz')
        original=load_split(source/'test_original.npz')
        assert np.array_equal(changed['ids'],original['ids'])
        assert np.array_equal(changed['labels'],original['labels'])
        head=nn.Linear(changed['features'].shape[1],state['head']['weight'].shape[0]).cuda()
        head.load_state_dict(state['head']);head.eval()
        logits=[]
        with torch.no_grad():
            for x in torch.as_tensor(changed['features']).float().split(2048):
                logits.append(head(((x-state['mean'])/state['std']).cuda()).cpu().numpy())
        logits=np.concatenate(logits)
        np.savez_compressed(out/'changed_predictions.npz',logits=logits,labels=changed['labels'],ids=changed['ids'])
        report=json.loads((out/'results.json').read_text());report['changed']=classification_metrics(logits,changed['labels'])
        report['shared_backbone_manifest']=str(source/'manifest.json')
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'dataset':dataset,'seed':seed,'original':report['test']['accuracy'],'changed':report['changed']['accuracy']}),flush=True)
