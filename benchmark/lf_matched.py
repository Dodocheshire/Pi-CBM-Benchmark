"""New LF-CBM/LF-Pi fits on the same images, raw backbone and concept bank as PiNI."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from .lf import upstream, fit_common, predict_features, export_npz, fit_nec_path
from .metrics import concept_metrics, classification_metrics, paired_concept_metrics


def encode_imagenet_rn18(source, output, device='cuda', batch_size=128, backbone='resnet18'):
    """Reencode exactly the shared PiNI RGB with an independently frozen backbone."""
    import torchvision
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original_manifest = json.loads((source / 'manifest.json').read_text())
    upstream()
    from pi_cbm.models.backbones import load_backbone
    # Core encoder preserves its original precision, architecture and pooling.
    model = load_backbone({'name':backbone,'precision':'float32'},device)
    fingerprint = hashlib.sha256()
    for key, value in model.state_dict().items():
        fingerprint.update(key.encode()); fingerprint.update(value.cpu().numpy().tobytes())
    encoder=model
    means,stds=([.48145466,.4578275,.40821073],[.26862954,.26130258,.27577711]) if backbone=='clip_RN50' else ([.485,.456,.406],[.229,.224,.225])
    mean = torch.tensor(means, device=device)[None,:,None,None]
    std = torch.tensor(stds, device=device)[None,:,None,None]
    for view in ['original', 'changed']:
        rgb = np.load(original_manifest[f'rgb_{view}'], mmap_mode='r')
        feature_path = output / f'{view}_features.npy'
        done = output / f'{view}_features_COMPLETE.json'
        if not done.exists():
            features = np.lib.format.open_memmap(feature_path, mode='w+', dtype='float32', shape=(len(rgb),model.feature_dim))
            with torch.no_grad():
                for start in range(0, len(rgb), batch_size):
                    image = torch.from_numpy(np.array(rgb[start:start+batch_size])).to(device).float()/255
                    prediction = encoder((image-mean)/std).flatten(1)
                    features[start:start+len(image)] = prediction.cpu().numpy()
                    if start % (batch_size*50) == 0:
                        print(json.dumps({'stage':'frozen_backbone_reencode','backbone':backbone,'view':view,'images':start+len(image),'total':len(rgb)}),flush=True)
            features.flush()
            done.write_text(json.dumps({'images':len(rgb),'backbone_sha256':fingerprint.hexdigest()}))
        features = np.load(feature_path,mmap_mode='r')
        for split in ['train','val','test']:
            destination = output / f'{split}_{view}.npz'
            if destination.exists(): continue
            original = np.load(source / f'{split}_{view}.npz')
            values = {key: original[key] for key in ['ids','labels','targets','known','training_targets'] if key in original.files}
            values['features'] = np.array(features[values['ids'].astype('int64')])
            np.savez_compressed(destination, **values)
    manifest = {**original_manifest, 'historical_checkpoint':False,
                'backbone':f'{backbone} independently frozen; no task fitting',
                'backbone_name':backbone,'feature_dim':model.feature_dim,
                'backbone_sha256':fingerprint.hexdigest(),'torchvision':torchvision.__version__,
                'preprocess':f'exact shared 224x224 uint8 RGB divided by255; normalize mean={means} std={stds}; no new crop',
                'features':'raw post avgpool vector, no LN or task normalization',
                'task_fit_sample_ids':[], 'source_original_manifest':str(source / 'manifest.json')}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def run(dataset, source, teacher, output, *, device='cuda', epochs=20,
        projection_steps=1000, saga_steps=1000, draws=8, seed=0, nec=False,backbone='resnet18',train_views='original'):
    source, teacher, output = map(Path, (source, teacher, output))
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((teacher / 'metadata.json').read_text())
    feature_manifest = json.loads((source / 'manifest.json').read_text())
    if feature_manifest.get('historical_checkpoint', False) or 'train+val' in feature_manifest.get('backbone', ''):
        raise ValueError('Matched selection requires independently frozen ImageNet or train-only backbone features; this feature manifest identifies a historical train+val checkpoint')
    names = metadata['names']
    for view in ['original', 'changed']:
        if not (teacher / f'{view}_COMPLETE.json').exists():
            raise ValueError(f'{view} teacher cache is not complete')
    teacher_arrays = {view: np.load(teacher / f'{view}_teacher.npy', mmap_mode='r')
                      for view in ['original', 'changed']}
    splits, raw = {}, {}
    for name in ['train', 'val', 'test']:
        original = np.load(source / f'{name}_original.npz')
        features = original['features']
        if features.ndim == 4:
            features = features.mean((2, 3))
        ids = original['ids'].astype('int64')
        splits[name] = {'features': torch.from_numpy(features.copy()).float(),
                        'labels': torch.from_numpy(original['labels'].copy()).long(),
                        'indices': torch.from_numpy(ids),
                        'teacher': torch.from_numpy(np.array(teacher_arrays['original'][ids])).float()}
        raw[name] = {key: original[key] for key in original.files}
    subjects={key:set(value['indices'].tolist()) for key,value in splits.items()}
    if any(subjects[a]&subjects[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Underlying train/val/test subject IDs overlap')
    canonical_splits={key:dict(value) for key,value in splits.items()}
    train_views=train_views.split(',') if isinstance(train_views,str) else train_views
    if train_views not in [['original'],['original','changed']]:
        raise ValueError('train_views must be original or original,changed')
    if 'changed' in train_views:
        for name in ['train','val']:
            changed=np.load(source/f'{name}_changed.npz')
            if not np.array_equal(changed['ids'],raw[name]['ids']):
                raise ValueError('Paired augmentation subject IDs must have identical order')
            x=changed['features']
            if x.ndim==4:x=x.mean((2,3))
            ids=changed['ids'].astype('int64')
            other={'features':torch.from_numpy(x.copy()).float(),
                   'labels':torch.from_numpy(changed['labels'].copy()).long(),
                   'indices':torch.from_numpy(ids),
                   'teacher':torch.from_numpy(np.array(teacher_arrays['changed'][ids])).float()}
            splits[name]={key:torch.cat([splits[name][key],other[key]]) for key in splits[name]}
    class_count = max(int(s['labels'].max()) for s in splits.values()) + 1
    data = {'splits': splits, 'classes': list(range(class_count)), 'concepts': names,
            'text': torch.from_numpy(np.load(teacher / 'text.npy').copy()).float()}
    core = upstream()
    config = yaml.safe_load((core / 'configs/base.yaml').read_text())
    config['experiment'].update(name=f'matched_{dataset}_{backbone}', device=device, seed=seed, threads=4)
    config['data'].update(dataset=dataset, input_mode='cache', cache_internal=False, cache_images=False)
    config['data']['benchmark_train_views']=train_views
    config['backbone']['name'] = backbone
    config['teacher']['name'] = 'ViT-B/16'
    # Official LF-CBM thresholds; projection and regularization selection only see val.
    config['baseline'].update(include_adapter_val=False, validation_split='val',
                              activation_cutoff=.25, interpretability_cutoff=.45,
                              projection_steps=projection_steps)
    config['baseline']['head'].update(steps=saga_steps, **{'lambda': .0007})
    config['training'].update(epochs=epochs, stage='finetune_projection', batch_size=256,
                              lr=.0003, projection_lr=.00001)
    config['injection'].update(site='target_global', point='global')
    config['paths'].update(cache=str(output / 'cache'), runs=str(output))
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    baseline, selected, baseline_fit = fit_common(data, config, device=device)
    if len(selected) == 0:
        raise RuntimeError('Official LF-CBM thresholds retain zero concepts on validation; choose and record validation-only thresholds explicitly before retrying')
    selected_names = [names[i] for i in selected.tolist()]
    baseline_state = {key: value.cpu().clone() for key, value in baseline.cbm.state_dict().items()}
    # Reuse the exact selected baseline for LF-Pi: avoid fitting a second projection.
    from pi_cbm.pipeline import build_model
    from pi_cbm.training.adapt import adapt
    from pi_cbm.randomness import seed_all
    from pi_cbm.pipeline import baseline_from_state
    cbm = baseline_from_state(baseline_state, device)
    seed_all(seed)
    pi = build_model(cbm, data, selected, config, device)
    with torch.no_grad():
        clean = {key: torch.cat([cbm.concepts(batch.to(device)).cpu()
                 for batch in split['features'].split(256)]) for key, split in splits.items()}
    history, refit = adapt(pi, data, clean, selected, config, device)
    torch.save({'baseline': baseline_state, 'selected': selected,
                'pi_cbm': {k: v.cpu() for k,v in pi.cbm.state_dict().items()},
                'generator': {k:v.cpu() for k,v in pi.generator.state_dict().items()},
                'config': config, 'concepts': selected_names}, output / 'models.pt')
    results = {'dataset': dataset, 'backbone': feature_manifest.get('backbone'),
               'optimization_device':device,
               'inference':{'lfcbm_draws':1,'lfpi_draws':draws,'seed':3000,
                            'concepts':'mean over noise draws','classification':'mean probabilities over noise draws; exported logits are log probabilities'},
               'training_seed':seed,'train_views':train_views,
               'underlying_subject_counts':{key:len(value) for key,value in subjects.items()},
               'augmentation':'Core loss unchanged; concatenate paired views in train/val; subject folds remain disjoint' if len(train_views)>1 else 'original only',
               'feature_provenance': feature_manifest, 'teacher_provenance': metadata,
               'bank_concepts': len(names), 'selected_concepts': len(selected),
               'bank_coverage': len(selected) / len(names),
               'selected_indices': selected.tolist(), 'selected_names': selected_names,
               'train_examples': len(splits['train']['labels']),
               'val_examples': len(splits['val']['labels']),
               'test_selection': False, 'baseline_fit': baseline_fit,
               'adapter_history': history, 'adapter_refit': refit, 'metrics': {}}
    predictions = {}
    for name in ['train', 'val', 'test']:
        views = ['original', 'changed'] if name == 'test' or len(train_views)>1 else ['original']
        for view in views:
            cached = raw[name] if view == 'original' else dict(np.load(source / f'{name}_{view}.npz'))
            features = cached['features']
            if features.ndim == 4:
                features = features.mean((2, 3))
            split = {**canonical_splits[name], 'features': torch.from_numpy(features.copy()).float()}
            targets = cached['targets'][:, selected.numpy()]
            known = cached.get('known', np.isfinite(cached['targets']))[:, selected.numpy()]
            for method, model in [('lfcbm', baseline), ('lfpi_global', pi)]:
                pred = predict_features(model, split['features'], draws=draws if method.startswith('lfpi') else 1, seed=3000)
                path = output / f'{method}_{name}_{view}.npz'
                export_npz(path, pred, split, selected_names, targets=targets, known=known)
                predictions[method, name, view] = pred
                if name != 'train':
                    full_teacher=concept_metrics(np.array(teacher_arrays[view][split['indices'].numpy()]),
                        cached['targets'],cached.get('known',np.isfinite(cached['targets'])))
                    semantic=concept_metrics(pred['concepts'].numpy(), targets, known)
                    denom=full_teacher['mAP_evaluable_concepts']
                    results['metrics'][f'{method}/{name}/{view}'] = {
                        'classification': classification_metrics(pred['logits'].numpy(), split['labels'].numpy()),
                        'semantic_concepts_common_selected_bank': semantic,
                        'full_candidate_bank_CLIP_teacher_semantics': full_teacher,
                        'coverage_weighted_native_mAP_missing_outputs_zero': semantic['mAP']*semantic['mAP_evaluable_concepts']/denom if denom and semantic['mAP'] is not None else None,
                        'CLIP_teacher': concept_metrics(pred['concepts'].numpy(),
                            np.array(teacher_arrays[view][split['indices'].numpy()])[:, selected.numpy()])}
    for method in ['lfcbm', 'lfpi_global']:
        results['metrics'][f'{method}/paired_shift'] = paired_concept_metrics(
            predictions[method, 'test', 'original']['concepts'].numpy(),
            predictions[method, 'test', 'changed']['concepts'].numpy(),
            predictions[method, 'train', 'original']['concepts'].numpy())
    if nec:
        for method, model in [('lfcbm', baseline), ('lfpi_global', pi)]:
            train_c=torch.cat([predictions[method,'train',view]['concepts'] for view in train_views])
            val_c=torch.cat([predictions[method,'val',view]['concepts'] for view in train_views])
            report, heads = fit_nec_path(train_c,
                                        splits['train']['labels'], val_c,
                                        splits['val']['labels'], class_count, device=device, steps=saga_steps,
                                        seed=seed)
            results[f'{method}_nec'] = report
            torch.save(heads, output / f'{method}_nec_heads.pt')
            evaluations=[]
            for label, head in {'best_val':heads['best'],**heads['budgets']}.items():
                for view in ['original','changed']:
                    c=predictions[method,'test',view]['concepts']
                    logits=torch.nn.functional.linear(c,head['weight'],head['bias'])
                    evaluations.append({'selection':label,'view':view,
                        **classification_metrics(logits.numpy(),splits['test']['labels'].numpy())})
            report['test_evaluations_after_val_selection']=evaluations
    (output / 'report.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({'output': str(output), 'selected_concepts': len(selected),
                      'bank_concepts': len(names)}, indent=2), flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True); p.add_argument('--source', required=True)
    p.add_argument('--teacher', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--projection-steps', type=int, default=1000)
    p.add_argument('--saga-steps', type=int, default=1000); p.add_argument('--nec', action='store_true')
    p.add_argument('--draws', type=int, default=8); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--encode-imagenet-rn18',action='store_true')
    p.add_argument('--encode-backbone',action='store_true')
    p.add_argument('--backbone',default='resnet18',choices=['resnet18','resnet50','clip_RN50'])
    p.add_argument('--train-views',default='original',choices=['original','original,changed'])
    a=p.parse_args(); torch.set_num_threads(4)
    if a.encode_imagenet_rn18 or a.encode_backbone:
        encode_imagenet_rn18(a.source,a.output,a.device,backbone=a.backbone); return
    run(a.dataset,a.source,a.teacher,a.output,device=a.device,epochs=a.epochs,
        projection_steps=a.projection_steps,saga_steps=a.saga_steps,draws=a.draws,seed=a.seed,nec=a.nec,backbone=a.backbone,train_views=a.train_views)


if __name__ == '__main__': main()
