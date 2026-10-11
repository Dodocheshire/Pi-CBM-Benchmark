"""Matched sparse and dense head evaluation on frozen exported concept vectors."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .lf import fit_nec_path
from .metrics import classification_metrics, nec
from .noncbm import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--field', default='concepts')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-views', default='original')
    parser.add_argument('--test-views', default='original,changed')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--saga-gradient-kernel', choices=['upstream','matmul'], default='upstream')
    parser.add_argument('--streaming-dense', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(3)
    args.out.mkdir(parents=True, exist_ok=True)
    views = args.train_views.split(',')
    test_views = args.test_views.split(',')
    existing=args.out/'report.json'
    if existing.exists() and (args.out/'heads.pt').exists():
        previous=json.loads(existing.read_text())
        if (previous.get('source')==str(args.source.resolve()) and
            previous.get('concept_field')==args.field and previous.get('prefix','')==args.prefix and
            previous.get('train_views')==views and previous.get('seed')==args.seed and
            previous.get('test_views',['original','changed'])==test_views and
            previous.get('dataset')==args.dataset and
            previous.get('saga_gradient_kernel',{}).get('name','upstream')==args.saga_gradient_kernel and
            previous.get('streaming_dense',False)==args.streaming_dense):
            print(json.dumps({'completed':str(existing),'action':'reuse completed matching fit'}))
            return
    def load(split, view):
        with np.load(args.source/f'{args.prefix}{split}_{view}.npz') as z:
            return {k: np.array(z[k]) for k in z.files}
    training = [load('train', v) for v in views]
    validation = [load('val', v) for v in views]
    if set(training[0]['ids']) & set(validation[0]['ids']):
        raise ValueError('Underlying train/val subjects overlap')
    for group in [training, validation]:
        for data in group[1:]:
            if not (np.array_equal(group[0]['ids'], data['ids']) and
                    np.array_equal(group[0]['labels'], data['labels'])):
                raise ValueError('Training views are not aligned')
    tr = np.concatenate([d[args.field] for d in training]).astype('float32')
    va = np.concatenate([d[args.field] for d in validation]).astype('float32')
    y = np.concatenate([d['labels'] for d in training])
    vy = np.concatenate([d['labels'] for d in validation])
    mean = tr.mean(0); scale = tr.std(0, ddof=1).clip(1e-5)
    x = torch.from_numpy((tr-mean)/scale); vx = torch.from_numpy((va-mean)/scale)
    labels = torch.as_tensor(y).long(); vlabels = torch.as_tensor(vy).long()
    classes = int(max(y.max(), vy.max()))+1
    matrix = yaml.safe_load(Path('configs/matrix.yaml').read_text())
    dataset_key = 'waterbirds_cub_places' if args.dataset=='waterbirds' else args.dataset
    targets = sorted({min(x.shape[1], b) for b in matrix['head_evaluation']['target_grids'][dataset_key]})
    report, heads = fit_nec_path(x, labels, vx, vlabels, classes, device=args.device,
                                seed=args.seed, steps=args.steps, targets=targets,
                                saga_gradient_kernel=args.saga_gradient_kernel)
    # The same dense fitting protocol used by the class-only control.
    train = {'features': tr, 'labels': y}; val = {'features': va, 'labels': vy}
    candidates = [fit(train, val, args.device, args.seed, lr, decay,streaming_dense=args.streaming_dense)
                  for lr in [.001, .01] for decay in [.0001, .01]]
    dense = min(candidates, key=lambda c: c['validation_nll'])
    evaluated = []
    for view in test_views:
        test = load('test', view)
        c = torch.from_numpy((test[args.field].astype('float32')-mean)/scale)
        for name, head in {'best_val': heads['best'], **heads['budgets']}.items():
            logits = torch.nn.functional.linear(c, head['weight'], head['bias']).numpy()
            np.savez_compressed(args.out/f'{name}_{view}.npz', logits=logits,
                                labels=test['labels'], ids=test['ids'])
            evaluated.append({'head': name, 'view': view,
                              **classification_metrics(logits, test['labels']),
                              **nec(head['weight'].numpy())})
        d = torch.as_tensor(test[args.field]).float()
        logits = torch.nn.functional.linear((d-dense['mean'])/dense['std'],
                        dense['head']['weight'], dense['head']['bias']).numpy()
        np.savez_compressed(args.out/f'dense_{view}.npz', logits=logits,
                            labels=test['labels'], ids=test['ids'])
        evaluated.append({'head': 'dense', 'view': view,
                          **classification_metrics(logits, test['labels']),
                          **nec(dense['head']['weight'].numpy())})
    report.update(test_evaluations=evaluated, source=str(args.source.resolve()),streaming_dense=args.streaming_dense,
                  prefix=args.prefix,
                  concept_field=args.field, train_views=views, dataset=args.dataset,
                  test_views=test_views,
                  normalization='mean/sample_std of training views only',
                  fitting_device=args.device,
                  dense_validation_nll=dense['validation_nll'])
    np.savez_compressed(args.out/'normalization.npz', mean=mean, scale=scale)
    torch.save({'sparse': heads, 'dense': dense}, args.out/'heads.pt')
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
