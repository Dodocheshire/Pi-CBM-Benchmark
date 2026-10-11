"""Resume class-label-only baselines on existing frozen CLIP RN50 feature caches."""
import json
from pathlib import Path
import torch
from .noncbm import run

root=Path('/home/pengcheng/Pi-CBM/LF-Pi/artifacts/mechanism_v1')
torch.set_num_threads(4)
for dataset in ['cifar10','cifar100']:
    for seed in [42,43,44]:
        out=Path('artifacts/noncbm')/(dataset+'_cliprn50')/('seed'+str(seed))
        if not (out/'results.json').exists():
            run(root/dataset/'clean_train.pt',root/dataset/'clean_val.pt',root/dataset/'clean_test.pt',out,'cuda',seed)
print(json.dumps({'noncbm_suite':'completed','datasets':['cifar10','cifar100'],'seeds':[42,43,44]}),flush=True)
