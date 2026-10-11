"""Audit and recompute historical LF caches; never label them matched new runs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .lf import upstream, predict_features, export_npz, fit_nec_path

DEFAULT_SOURCE = Path('/home/pengcheng/Pi-CBM/LF-Pi')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(source=DEFAULT_SOURCE):
    source = Path(source)
    entries = []
    for checkpoint in sorted((source / 'artifacts').glob('*/**/checkpoint.pt')):
        if 'source' in checkpoint.parts:
            continue
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        config = saved['config']
        entries.append({
            'path': str(checkpoint), 'dataset': config['data']['dataset'],
            'backbone': config['backbone'], 'site': config['injection']['site'],
            'seed': config['experiment']['seed'], 'data_protocol': config['data'],
            'baseline_protocol': config['baseline'],
            'classes': saved['cbm']['head.weight'].shape[0],
            'concept_count': saved['cbm']['head.weight'].shape[1],
            'feature_cache_exists': Path(config['paths']['cache']).exists(),
            'saved_baseline_concept_shapes': {
                key: list(value.shape) for key, value in saved.get('baseline_concepts', {}).items()
                if torch.is_tensor(value)
            },
            'saved_baseline_concepts_scope': 'Historical baseline representation; adapted checkpoint head must not be substituted without matching its representation.',
        })
    caches = []
    for dataset in ['cifar10', 'cifar100', 'cub']:
        directory = source / 'artifacts/mechanism_v1' / dataset
        for path in sorted(directory.rglob('*.pt')):
            if path.name in ['clean_train.pt', 'clean_val.pt', 'clean_test.pt',
                              'teacher_train.pt', 'teacher_val.pt', 'training_reference.pt',
                              'dataset.pt', 'predictions.pt']:
                x = torch.load(path, weights_only=True, map_location='cpu')
                if isinstance(x, torch.Tensor):
                    shapes = list(x.shape)
                else:
                    shapes = {k: list(v.shape) for k, v in x.items() if torch.is_tensor(v)}
                caches.append({'path': str(path), 'dataset': dataset, 'tensor_shapes': shapes})
    return {'historical_checkpoints': entries, 'existing_diagnostic_caches': caches,
            'scope': 'Historical inventory; not newly trained matched comparisons',
            'limitation': 'CIFAR/ImageNet/Places old concept selection used test; old ImageNet subsets and RGB-only CUB are not standard full-dataset protocols.'}


def export_cifar(dataset, output, *, source=DEFAULT_SOURCE, device='cpu', draws=8, run=None):
    source, output = Path(source), Path(output)
    upstream()
    from pi_cbm.pipeline import baseline_from_state
    from pi_cbm.evaluation.checkpoint import restore_model
    from pi_cbm.models.cbm import NoisyCBM
    from pi_cbm.injections import build_injection
    data = source / 'artifacts/mechanism_v1' / dataset
    baseline_paths = sorted((source / 'artifacts/cifar-nec-baselines').glob('*.pt'))
    classes = 10 if dataset == 'cifar10' else 100
    baseline_path = next(p for p in baseline_paths if
                         torch.load(p, weights_only=True)['cbm']['head.weight'].shape[0] == classes)
    saved = torch.load(baseline_path, weights_only=True)
    cbm = baseline_from_state(saved['cbm'], device)
    model = NoisyCBM(cbm, build_injection('baseline'), None, torch.nn.Identity())
    bank = (source / 'configs/concepts' / f'{dataset}_official.txt').read_text().splitlines()
    names = [bank[i] for i in saved['selected'].tolist()]
    output.mkdir(parents=True, exist_ok=True)
    splits = {s: torch.load(data / f'clean_{s}.pt', weights_only=True) for s in ['train', 'val', 'test']}
    for name, split in splits.items():
        pred = predict_features(model, split['features'])
        export_npz(output / f'lfcbm_{name}.npz', pred, split, names)
        teacher = data / 'matched_global_controls' / f'teacher_{name}.pt'
        if teacher.exists():
            torch.save({'teacher': torch.load(teacher, weights_only=True), 'indices': split['indices']},
                       output / f'teacher_{name}.pt')
            from .metrics import concept_metrics
            continuous = concept_metrics(pred['concepts'].numpy(), torch.load(teacher, weights_only=True).numpy())
            (output / f'lfcbm_{name}_clip_teacher_metrics.json').write_text(json.dumps({
                'target_type': 'CLIP cosine, not semantic annotations', **continuous}, indent=2))
    provenance = {'method': 'historical_lfcbm_recomputed', 'baseline': str(baseline_path),
                  'baseline_sha256': sha256(baseline_path), 'historical_fit': saved['metrics'],
                  'limits': 'Projection/concept selection used official test. Semantic annotations unavailable; known=false. CLIP teacher is a separate cosine target, not semantic ground truth.'}
    if run is not None:
        run = Path(run)
        saved_pi = torch.load(run / 'checkpoint.pt', weights_only=True)
        if saved_pi['config']['injection']['site'] != 'target_global':
            raise ValueError('Cached global features support target_global only; images/internal require image inference.')
        if saved_pi['concepts'] != names:
            raise ValueError('Historical LF-Pi concept bank differs from baseline')
        pi = restore_model(saved_pi, device)
        for name, split in splits.items():
            pred = predict_features(pi, split['features'], draws=draws, seed=3000)
            export_npz(output / f'lfpi_global_{name}.npz', pred, split, names)
        provenance['lfpi_run'] = str(run)
        provenance['lfpi_checkpoint_sha256'] = sha256(run / 'checkpoint.pt')
        provenance['lfpi_mc_draws'] = draws
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    return splits, cbm


def run_cifar_nec(dataset, output, *, source=DEFAULT_SOURCE, device='cpu', steps=120, points=12):
    output = Path(output)
    splits, cbm = export_cifar(dataset, output, source=source, device=device)
    with torch.no_grad():
        concepts = {s: cbm.concepts(v['features'].to(device)).cpu() for s, v in splits.items()}
    report, heads = fit_nec_path(concepts['train'], splits['train']['labels'], concepts['val'],
                               splits['val']['labels'], cbm.head.out_features,
                               device=device, steps=steps, points=points)
    # All choices are frozen before test evaluation.
    from .lf import head_nec
    evaluations = []
    selected_heads = {'best_val': heads['best'], **heads['budgets']}
    for label, head in selected_heads.items():
        logits = torch.nn.functional.linear(concepts['test'], head['weight'], head['bias'])
        from .metrics import classification_metrics
        evaluations.append({'selection': label, 'nec': head_nec(head['weight']),
                            **classification_metrics(logits.numpy(), splits['test']['labels'].numpy())})
        torch.save({'concepts': concepts['test'], 'logits': logits, 'labels': splits['test']['labels'],
                    'ids': splits['test']['indices'], 'head': head}, output / f'nec_{label}_test.pt')
    report.update(test_evaluations=evaluations, dataset=dataset,
                  limitation='New train-only head fits on historical frozen projection and concept bank previously selected with test; not a new leakage-free full-model experiment.')
    torch.save(heads, output / 'nec_heads.pt')
    (output / 'nec_report.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['inventory', 'export', 'nec'])
    p.add_argument('--source', default=str(DEFAULT_SOURCE))
    p.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--points', type=int, default=12)
    p.add_argument('--run')
    p.add_argument('--draws', type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    if a.action == 'inventory':
        result = inventory(a.source)
        Path(a.output).write_text(json.dumps(result, indent=2))
        print(f"Inventoried {len(result['historical_checkpoints'])} checkpoints", flush=True)
    elif a.action == 'export':
        export_cifar(a.dataset, a.output, source=a.source, device=a.device, run=a.run, draws=a.draws)
    else:
        print(json.dumps(run_cifar_nec(a.dataset, a.output, source=a.source,
                         device=a.device, steps=a.steps, points=a.points), indent=2), flush=True)


if __name__ == '__main__':
    main()
