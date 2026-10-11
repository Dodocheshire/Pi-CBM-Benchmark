"""Class-only controls trained on the same paired background views as PiNI."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .metrics import classification_metrics
from .noncbm import load_split, run


def main():
    torch.set_num_threads(4)
    for dataset in ['rival10', 'cub', 'waterbirds']:
        source = Path('artifacts/common') / dataset / 'rn18_imagenet'
        paired = source / 'paired'
        paired.mkdir(exist_ok=True)
        for split in ['train', 'val']:
            views = [load_split(source / f'{split}_{v}.npz') for v in ['original', 'changed']]
            assert np.array_equal(views[0]['ids'], views[1]['ids'])
            assert np.array_equal(views[0]['labels'], views[1]['labels'])
            # Keep subject IDs unchanged so the trainer checks subject-level folds.
            np.savez_compressed(paired / f'{split}.npz', **{
                k: np.concatenate([v[k] for v in views]) for k in ['features', 'labels', 'ids']})
        for seed in [42, 43, 44]:
            out = Path('artifacts/noncbm') / f'{dataset}_rn18_imagenet_paired' / f'seed{seed}'
            if not (out/'results.json').exists():
                run(paired / 'train.npz', paired / 'val.npz', source / 'test_original.npz', out, 'cuda', seed)
            state = torch.load(out / 'best.pt', weights_only=True)
            changed = load_split(source / 'test_changed.npz')
            head = nn.Linear(changed['features'].shape[1], state['head']['weight'].shape[0]).cuda()
            head.load_state_dict(state['head']); head.eval()
            with torch.no_grad():
                logits = np.concatenate([head(((x-state['mean'])/state['std']).cuda()).cpu().numpy()
                    for x in torch.as_tensor(changed['features']).float().split(2048)])
            np.savez_compressed(out / 'changed_predictions.npz', logits=logits,
                                labels=changed['labels'], ids=changed['ids'])
            report = json.loads((out / 'results.json').read_text())
            report.update(changed=classification_metrics(logits, changed['labels']),
                          train_views=['original', 'changed'],
                          shared_backbone_manifest=str(source / 'manifest.json'))
            (out / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({'dataset': dataset, 'seed': seed, 'changed': report['changed']['accuracy']}), flush=True)


if __name__ == '__main__':
    main()
