"""Fit unmodified LF-CBM/LF-Pi on the full audited scale split caches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .lf import upstream, fit_common, predict_features, export_npz
from .metrics import classification_metrics, teacher_fidelity
from .report_utils import write_json


def fit(dataset, device, epochs, steps, seed):
    core = upstream()
    from pi_cbm.pipeline import build_model, baseline_from_state
    from pi_cbm.training.adapt import adapt
    from pi_cbm.randomness import seed_all
    base = Path('artifacts/common') / dataset
    source, teacher = base / 'rn18_imagenet', base / 'teacher'
    if not (source / 'manifest.json').exists() or not (teacher / 'metadata.json').exists():
        raise RuntimeError('Scale cache export has not completed')
    out = Path('artifacts/matched/lf') / dataset / 'rn18_imagenet' / f'seed{seed}'
    if (out / 'report.json').exists():
        print(json.dumps({'dataset': dataset, 'completed_report': str(out / 'report.json')}), flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((teacher / 'metadata.json').read_text())
    names = metadata['names']
    scores = np.load(teacher / 'original_teacher.npy', mmap_mode='r')
    splits = {}
    for split in ['train', 'val', 'test']:
        with np.load(source / f'{split}_original.npz') as values:
            ids = values['ids'].copy()
            splits[split] = {'features': torch.from_numpy(values['features'].copy()),
                             'labels': torch.from_numpy(values['labels'].copy()),
                             'indices': torch.from_numpy(ids),
                             'teacher': torch.from_numpy(np.array(scores[ids]))}
    if set(splits['train']['indices'].tolist()) & set(splits['val']['indices'].tolist()):
        raise ValueError('Training and validation IDs overlap')
    data = {'splits': splits, 'classes': metadata['class_names'], 'concepts': names,
            'text': torch.from_numpy(np.load(teacher / 'text.npy'))}
    config = yaml.safe_load((core / 'configs/base.yaml').read_text())
    config['experiment'].update(name=f'matched_{dataset}_rn18', device=device, seed=seed, threads=4)
    config['data'].update(dataset=dataset, input_mode='cache', cache_internal=False,
                          cache_images=False, benchmark_train_views=['original'])
    config['backbone']['name'] = 'resnet18'
    config['teacher']['name'] = 'ViT-B/16'
    config['baseline'].update(include_adapter_val=False, validation_split='val',
                              activation_cutoff=.25, interpretability_cutoff=.45,
                              projection_steps=steps)
    config['baseline']['head'].update(steps=steps, **{'lambda': .0007})
    config['training'].update(epochs=epochs, stage='finetune_projection', batch_size=256,
                              lr=.0003, projection_lr=.00001)
    config['injection'].update(site='target_global', point='global')
    config['paths'].update(cache=str(out / 'cache'), runs=str(out))
    (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    baseline, selected, baseline_fit = fit_common(data, config, device=device)
    if not len(selected):
        raise RuntimeError('No concepts passed original LF selection thresholds')
    baseline_state = {k: v.cpu().clone() for k, v in baseline.cbm.state_dict().items()}
    cbm = baseline_from_state(baseline_state, device)
    seed_all(seed)
    pi = build_model(cbm, data, selected, config, device)
    with torch.no_grad():
        clean = {split: torch.cat([cbm.concepts(batch.to(device)).cpu()
                                   for batch in values['features'].split(256)])
                 for split, values in splits.items()}
    history, refit = adapt(pi, data, clean, selected, config, device)
    selected_names = [names[i] for i in selected.tolist()]
    torch.save({'baseline': baseline_state,
                'pi_cbm': {k: v.cpu() for k, v in pi.cbm.state_dict().items()},
                'generator': {k: v.cpu() for k, v in pi.generator.state_dict().items()},
                'selected': selected, 'config': config, 'concepts': selected_names}, out / 'models.pt')
    report = {'dataset': dataset, 'training_seed': seed, 'train_views': ['original'],
              'bank_concepts': len(names), 'candidate_concepts': len(names),
              'selected_concepts': len(selected), 'selected_indices': selected.tolist(),
              'selected_names': selected_names, 'human_concept_annotations': False,
              'feature_provenance': json.loads((source / 'manifest.json').read_text()),
              'teacher_provenance': metadata, 'baseline_fit': baseline_fit,
              'adapter_history': history, 'adapter_refit': refit,
              'model_selection': 'training and validation only; no test model selection',
              'inference': {'lfcbm_draws': 1, 'lfpi_draws': 1, 'seed': 3000}, 'metrics': {}}
    single = out / 'single_draw'
    single.mkdir(exist_ok=True)
    with (out / 'models.pt').open('rb') as stream:
        fingerprint = hashlib.file_digest(stream, 'sha256').hexdigest()
    for name, model in [('lfcbm', baseline), ('lfpi_global', pi)]:
        for split, values in splits.items():
            prediction = predict_features(model, values['features'], draws=1, seed=3000)
            destination = single if name == 'lfpi_global' else out
            export_npz(destination / f'{name}_{split}_original.npz', prediction, values, selected_names)
            if split != 'train':
                report['metrics'][f'{name}/{split}/original'] = {
                    'classification': classification_metrics(prediction['logits'].numpy(), values['labels'].numpy()),
                    'CLIP_teacher': teacher_fidelity(prediction['concepts'].numpy(),
                                                    scores[values['indices'].numpy()][:, selected.numpy()])}
    write_json(single / 'metadata.json', {'inference_draws': 1, 'inference_seed': 3000,
                                          'checkpoint_sha256': fingerprint, 'selected_indices': selected.tolist(),
                                          'concept_names': selected_names, 'batch_size': 256})
    write_json(out / 'report.json', report)
    print(json.dumps({'dataset': dataset, 'fit_complete': str(out), 'retained_concepts': len(selected)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['imagenet_200_per_class', 'places365'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    fit(args.dataset, args.device, args.epochs, args.steps, args.seed)


if __name__ == '__main__':
    main()
