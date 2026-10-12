"""Canonical-RGB image/internal LF-Pi adapters outside the frozen core.

Default action audits a few CPU samples only. Full prefix encoding and training
require separate explicit actions. Existing global runs are reference inputs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .lf import upstream, export_npz, predict_features
from .metrics import classification_metrics, concept_metrics, teacher_fidelity
from .task_backbones import ROOT, file_sha, state_sha


def context(dataset='rival10', protocol='original', seed=42):
    source = ROOT/'artifacts/common'/dataset/'rn18_imagenet'
    teacher = ROOT/'artifacts/common'/('cub' if dataset == 'waterbirds' else dataset)/'teacher'
    reference = ROOT/'artifacts/matched/lf'/dataset/'rn18_imagenet'
    if protocol == 'paired':
        reference /= 'paired_augmentation'
    reference /= f'seed{seed}'
    manifest = json.loads((source/'manifest.json').read_text())
    saved = torch.load(reference/'models.pt', map_location='cpu', weights_only=True)
    report = json.loads((reference/'report.json').read_text())
    assert saved['config']['backbone']['name'] == 'resnet18'
    assert report['test_selection'] is False and report['training_seed'] == seed
    assert report['train_views'] == (['original','changed'] if protocol == 'paired' else ['original'])
    names = json.loads((teacher/'metadata.json').read_text())['names']
    assert names == manifest['concept_names']
    assert saved['concepts'] == [names[i] for i in saved['selected'].tolist()]
    assert saved['selected'].tolist() == report['selected_indices']
    for view in ['original','changed']:
        assert (teacher/f'{view}_COMPLETE.json').exists()
        rgb = np.load(manifest[f'rgb_{view}'], mmap_mode='r')
        assert rgb.dtype == np.uint8 and rgb.shape[1:] == (3,224,224)
    folds = {s:np.load(source/f'{s}_original.npz')['ids'].copy() for s in ['train','val','test']}
    assert not any(set(folds[a]) & set(folds[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    return source, teacher, reference, manifest, saved, folds


def backbone(device, manifest):
    upstream()
    from pi_cbm.models.backbones import load_backbone
    model = load_backbone({'name':'resnet18','precision':'float32'},device)
    from torchvision.models import resnet18,ResNet18_Weights
    full = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # The original RN18 manifest hashes the complete torchvision network,
    # whereas the core split renames blocks and omits the unused classifier.
    assert state_sha(full.state_dict()) == manifest['backbone_sha256'], 'Shared encoder checkpoint differs'
    expected_prefix = torch.nn.Sequential(full.conv1,full.bn1,full.relu,full.maxpool,full.layer1,full.layer2,full.layer3)
    expected_suffix = torch.nn.Sequential(full.layer4,full.avgpool,torch.nn.Flatten(1))
    assert state_sha(model.prefix.state_dict()) == state_sha(expected_prefix.state_dict())
    assert state_sha(model.suffix.state_dict()) == state_sha(expected_suffix.state_dict())
    return model


def normalized(raw):
    value = torch.from_numpy(np.array(raw)).float()/255
    shape = (1,3,1,1) if value.ndim == 4 else (3,1,1)
    return (value-torch.tensor([.485,.456,.406]).reshape(shape))/torch.tensor([.229,.224,.225]).reshape(shape)


def internal_root(dataset):
    return ROOT/'artifacts/common'/dataset/'rn18_imagenet_pre_layer4'


class CanonicalInputs(Dataset):
    """Individual items for input_loader; batched iteration for channel_scale.

    No core ImagePaths transform is used: the shared crop is already canonical.
    """
    def __init__(self, groups, manifest, site, cache, calibration_batch=32, sample_arrays=None):
        self.site = site
        self.groups = groups
        self.ends = np.cumsum([len(ids) for view,ids in groups])
        self.calibration_batch = calibration_batch
        # sample_arrays is only used for the isolated two-item internal audit.
        self.arrays = dict(sample_arrays or {})
        for view,ids in groups:
            if view in self.arrays:
                continue
            if site == 'target_image':
                self.arrays[view] = np.load(manifest[f'rgb_{view}'],mmap_mode='r')
            else:
                receipt = json.loads((cache/f'{view}_COMPLETE.json').read_text())
                assert receipt['backbone_sha256'] == manifest['backbone_sha256']
                assert receipt['point'] == 'pre_layer4' and receipt['shape'][1:] == [256,14,14]
                assert receipt['rgb_sha256'] == file_sha(manifest[f'rgb_{view}'])
                self.arrays[view] = np.load(cache/f'internal_{view}.npy',mmap_mode='r')
                assert list(self.arrays[view].shape) == receipt['shape']

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        group = int(np.searchsorted(self.ends,index,side='right'))
        offset = int(self.ends[group-1]) if group else 0
        view,ids = self.groups[group]
        value = self.arrays[view][int(ids[index-offset])]
        return normalized(value) if self.site == 'target_image' else torch.from_numpy(np.array(value)).float()

    def __iter__(self):
        for start in range(0,len(self),self.calibration_batch):
            yield torch.stack([self[i] for i in range(start,min(start+self.calibration_batch,len(self)))])


def make_data(dataset, protocol, seed, site):
    source, teacher, reference, manifest, saved, folds = context(dataset,protocol,seed)
    key = 'images' if site == 'target_image' else 'internal'
    arrays = {v:np.load(teacher/f'{v}_teacher.npy',mmap_mode='r') for v in ['original','changed']}
    splits = {}
    for split,ids in folds.items():
        views = ['original','changed'] if protocol == 'paired' and split != 'test' else ['original']
        pieces = []
        for view in views:
            with np.load(source/f'{split}_{view}.npz') as z:
                assert np.array_equal(ids,z['ids'])
                pieces.append({'features':torch.from_numpy(z['features'].copy()).float(),
                               'labels':torch.from_numpy(z['labels'].copy()).long(),
                               'indices':torch.from_numpy(ids.copy()).long(),
                               'teacher':torch.from_numpy(np.array(arrays[view][ids])).float()})
        splits[split] = {field:torch.cat([p[field] for p in pieces]) for field in pieces[0]}
        for other in pieces[1:]:
            assert torch.equal(other['labels'],pieces[0]['labels'])
        splits[split][key] = CanonicalInputs([(v,ids) for v in views],manifest,site,internal_root(dataset))
        assert len(splits[split][key]) == len(splits[split]['labels'])
    data = {'splits':splits,'classes':list(range(saved['baseline']['head.weight'].shape[0])),
            'concepts':manifest['concept_names'], 'text':torch.from_numpy(np.load(teacher/'text.npy').copy()).float()}
    return data, (source,teacher,reference,manifest,saved,folds)


def configuration(saved, site, output, device, batch=None):
    config = copy.deepcopy(saved['config'])
    config['experiment']['device'] = device
    config['experiment']['threads'] = 4
    config['experiment']['name'] += f'_{site}'
    config['injection'].update(site=site,point='input' if site == 'target_image' else 'pre_layer4')
    config['data'].update(input_mode='cache',cache_images=site=='target_image',cache_internal=site=='target_internal')
    config['training']['cache_internal'] = False
    config['baseline']['head']['steps'] = 1000
    if batch is not None:
        config['training']['batch_size'] = batch
        config['data']['batch_size'] = batch
    config['paths'].update(cache=str(output/'cache'),runs=str(output))
    return config


def baseline_model(state,data,config,device):
    from pi_cbm.pipeline import baseline_from_state,build_model
    clean_config = copy.deepcopy(config)
    clean_config['injection'].update(site='baseline',point='global')
    return build_model(baseline_from_state(state,device),data,None,clean_config,device).eval().requires_grad_(False)


def encode_internal(dataset, device='cpu', batch=32):
    source,teacher,reference,manifest,saved,folds = context(dataset)
    output = internal_root(dataset)
    output.mkdir(parents=True,exist_ok=True)
    model = backbone(device,manifest)
    for view in ['original','changed']:
        final = output/f'internal_{view}.npy'
        marker = output/f'{view}_COMPLETE.json'
        if final.exists() or marker.exists():
            raise RuntimeError('Prefix cache already exists; refuse implicit overwrite or reencoding')
        rgb = np.load(manifest[f'rgb_{view}'],mmap_mode='r')
        pending = output/f'internal_{view}.pending.npy'
        values = np.lib.format.open_memmap(pending,mode='w+',dtype='float32',shape=(len(rgb),256,14,14))
        with torch.no_grad():
            for start in range(0,len(rgb),batch):
                inputs = normalized(rgb[start:start+batch]).to(device)
                internal = model.prefix(inputs)
                assert internal.shape[1:] == (256,14,14)
                values[start:start+len(inputs)] = internal.cpu().numpy()
                if start % (batch*50) == 0:
                    print(json.dumps({'view':view,'rows':start+len(inputs),'total':len(rgb)}),flush=True)
        values.flush();del values;pending.replace(final)
        marker.write_text(json.dumps({'point':'pre_layer4','shape':[len(rgb),256,14,14],
            'device':device,'encoding_batch':batch,'backbone_sha256':manifest['backbone_sha256'],
            'backbone_sha256_scope':'full torchvision state_dict; prefix/suffix tensors verified identical',
            'core_split_state_sha256':state_sha(model.state_dict()),
            'rgb_source':manifest[f'rgb_{view}'],'rgb_sha256':file_sha(manifest[f'rgb_{view}']),
            'no_new_crop':True,'normalization':'shared canonical uint8 /255 and ImageNet mean/std'},indent=2))


def audit(dataset='rival10', protocol='original', seed=42):
    source,teacher,reference,manifest,saved,folds = context(dataset,protocol,seed)
    model = backbone('cpu',manifest)
    ids = folds['train'][:2]
    evidence = {'dataset':dataset,'protocol':protocol,'seed':seed,'audit_scope':'CPU two training samples only; no fitting or full cache',
        'reference':str(reference),'reference_checkpoint_sha256':file_sha(reference/'models.pt'),
        'selected_indices':saved['selected'].tolist(),'backbone_sha256':manifest['backbone_sha256'],
        'full_cache_started':False,'formal_training_started':False,'gpu_used':False,'views':{},
        'core_split_state_sha256':state_sha(model.state_dict()),
        'full_torchvision_state_sha_matches_shared':True,'prefix_suffix_weights_exact_full_torchvision':True}
    upstream()
    from pi_cbm.pipeline import baseline_from_state,build_model
    from pi_cbm.randomness import seed_all
    from pi_cbm.inputs import input_loader
    small_maps,small_images = {},{}
    for view in ['original','changed']:
        raw = np.load(manifest[f'rgb_{view}'],mmap_mode='r')
        images = normalized(raw[ids])
        with torch.no_grad():
            internal = model.prefix(images)
            clean = model.suffix(internal)
            direct = model(images)
        with np.load(source/f'train_{view}.npz') as z:
            assert np.array_equal(z['ids'],folds['train'])
            global_reference = torch.from_numpy(z['features'][:2].copy()).float()
            sample_labels = torch.from_numpy(z['labels'][:2].copy()).long()
            sample_targets,sample_known = z['targets'][:2].copy(),z['known'][:2].copy()
        assert internal.shape == (2,256,14,14)
        small_images[view] = images
        small_maps[view] = {int(i):internal[j].numpy() for j,i in enumerate(ids)}
        assert torch.equal(clean,direct)
        delta = clean-global_reference
        rel = float(delta.norm()/global_reference.norm())
        cosine = torch.nn.functional.cosine_similarity(clean.double(),global_reference.double(),dim=1)
        # Shared GPU caches and CPU kernels need the already measured backend
        # tolerance; exact equality is required for the CPU split vs full path.
        assert rel < 1e-3 and float(cosine.min()) > .99999 and float(delta.abs().max()) < .03
        records = {'ids':ids.tolist(),'internal_shape':list(internal.shape),'global_shape':list(clean.shape),
            'canonical_rgb_source':manifest[f'rgb_{view}'],
            'canonical_sample_rgb_sha256':hashlib.sha256(np.array(raw[ids]).tobytes()).hexdigest(),
            'preprocess':'canonical uint8 /255; ImageNet normalization only; no resize/crop',
            'cpu_suffix_vs_full_path_max_abs':float((clean-direct).abs().max()),
            'shared_global_max_abs':float(delta.abs().max()),'shared_global_relative_l2':rel,
            'shared_global_cosine_min':float(cosine.min()),
            'shared_backend_tolerance':{'relative_l2':.001,'cosine_min':.99999,'max_abs':.03},'sites':{}}
        for site,tensor in [('target_image',images),('target_internal',internal)]:
            key = 'images' if site == 'target_image' else 'internal'
            samples = {view:{int(i):internal[j].numpy() for j,i in enumerate(ids)}} if site=='target_internal' else None
            adapter = CanonicalInputs([(view,ids)],manifest,site,internal_root(dataset),2,sample_arrays=samples)
            assert torch.equal(torch.stack([adapter[i] for i in range(2)]),tensor)
            assert torch.equal(next(iter(adapter)),tensor)
            split = {'features':global_reference,'labels':sample_labels,
                     'indices':torch.from_numpy(ids.copy()),'teacher':torch.from_numpy(np.array(np.load(teacher/f'{view}_teacher.npy',mmap_mode='r')[ids])),key:adapter}
            data = {'splits':{'train':split},'text':torch.from_numpy(np.load(teacher/'text.npy').copy()).float()}
            config = configuration(saved,site,ROOT/'artifacts/audit/lf_sites_unused','cpu',2)
            cbm = baseline_from_state(saved['baseline'],'cpu')
            seed_all(seed)
            noisy = build_model(cbm,data,saved['selected'],config,'cpu').eval().requires_grad_(False)
            leaf = tensor.clone().requires_grad_(True)
            logits,concepts,energy = noisy(leaf)
            (concepts.square().mean()+energy).backward()
            assert leaf.grad is not None and torch.isfinite(leaf.grad).all() and float(leaf.grad.abs().max()) > 0
            assert all(p.grad is None for p in noisy.parameters())
            loader = input_loader(split,noisy,config,split['labels'],batch_size=2)
            items = next(iter(loader));assert torch.equal(items[0],tensor)
            assert torch.equal(items[1],sample_labels)
            base = baseline_model(saved['baseline'],data,config,'cpu')
            base_prediction = predict_features(base,global_reference,draws=1,seed=3000,batch_size=2)
            with torch.no_grad():
                manual_concepts = base.cbm.concepts(global_reference)
                manual_probability = base.cbm.head(manual_concepts).softmax(-1)
            assert torch.equal(base_prediction['concepts'],manual_concepts)
            assert torch.equal(base_prediction['probabilities'],manual_probability)
            with torch.no_grad():
                clean_c = base.cbm.concepts(clean)
                clean_p = base.cbm.head(clean_c).softmax(-1)
            prediction = site_prediction(noisy,split,config,'cpu')
            sample_dir = ROOT/'artifacts/audit/lf_sites_samples'
            sample_dir.mkdir(exist_ok=True)
            prefix = f'{dataset}_{protocol}_{view}_{site}'
            for kind,pred in [('baseline',base_prediction),('site',prediction)]:
                path = sample_dir/f'{prefix}_{kind}.npz'
                export_npz(path,pred,split,saved['concepts'],targets=sample_targets[:,saved['selected']],known=sample_known[:,saved['selected']])
                with np.load(path) as exported:
                    assert np.array_equal(exported['ids'],ids)
                    assert np.array_equal(exported['labels'],sample_labels.numpy())
                    assert exported['concepts'].shape == (2,len(saved['selected']))
                    assert np.isfinite(exported['logits']).all()
            records['sites'][site] = {'original_input_loader_exact':True,'target_input_gradient_max_abs':float(leaf.grad.abs().max()),
                'gradient_crosses_frozen_suffix':True,'frozen_parameter_gradients_absent':True,
                'concept_shape':list(concepts.shape),'classifier_shape':list(logits.shape),
                'core_build_model_used':True,'generator_channels':len(noisy.generator.feature_scale),
                'lazy_canonical_input_adapter_exact':True,'baseline_wrapper_vs_direct_probability_max_abs':0.0,
                'clean_suffix_concepts_vs_shared_max_abs':float((clean_c-manual_concepts).abs().max()),
                'clean_suffix_probabilities_vs_shared_max_abs':float((clean_p-manual_probability).abs().max()),
                'clean_suffix_class_prediction_changes':int((clean_p.argmax(1)!=manual_probability.argmax(1)).sum()),
                'baseline_and_site_two_sample_npz_verified':True,'sample_export_prefix':str(sample_dir/prefix)}
        evidence['views'][view] = records
    evidence['paired_adapter_fixture'] = {}
    for site in ['target_image','target_internal']:
        samples = small_maps if site == 'target_internal' else None
        adapter = CanonicalInputs([(view,ids) for view in ['original','changed']],manifest,site,internal_root(dataset),2,sample_arrays=samples)
        expected = (torch.cat([small_images[v] for v in ['original','changed']]) if site == 'target_image'
                    else torch.cat([torch.from_numpy(np.stack([small_maps[v][int(i)] for i in ids])) for v in ['original','changed']]))
        assert torch.equal(torch.cat(list(iter(adapter))),expected)
        evidence['paired_adapter_fixture'][site] = {'views':['original','changed'],'underlying_subjects':2,
            'rows':len(adapter),'ordered_subject_ids':np.tile(ids,2).tolist(),'exact_normalized_input_parity':True,
            'scope':'input interface fixture only; no paired training or full cache'}
    evidence['core_source_sha256'] = {name:file_sha(ROOT/'methods/LF-Pi/pi_cbm'/name) for name in
        ['pipeline.py','training/adapt.py','inputs.py','injections/sites.py','models/backbones.py','evaluation/inference.py']}
    output = ROOT/f'artifacts/audit/lf_sites_{dataset}_{protocol}_cpu.json'
    output.write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2),flush=True)
    return evidence


@torch.no_grad()
def site_prediction(model,split,config,device):
    from pi_cbm.inputs import input_loader
    from pi_cbm.evaluation.inference import predict
    from pi_cbm.randomness import seeded
    from pi_cbm.precision import adapter_autocast
    probabilities,concepts = [],[]
    with seeded(3000):
        for (batch,) in input_loader(split,model,config,batch_size=config['data']['batch_size']):
            inputs = model.prepare_inputs(batch.to(device))
            with adapter_autocast(config):
                p,c = predict(model,inputs)
            probabilities.append(p.float().cpu());concepts.append(c.float().cpu())
    p,c = torch.cat(probabilities),torch.cat(concepts)
    return {'probabilities':p,'logits':p.clamp_min(1e-12).log(),'concepts':c}


def export(model,data,info,config,output,site):
    source,teacher,reference,manifest,saved,folds = info
    baseline = baseline_model(saved['baseline'],data,config,config['experiment']['device'])
    selected = saved['selected'].numpy();names = saved['concepts']
    output.mkdir(exist_ok=True,parents=True)
    metadata = {'site':site,'inference_draws':1,'inference_seed':3000,'inference_batch_size':config['data']['batch_size'],
                'reference_checkpoint_sha256':file_sha(reference/'models.pt'),'checkpoint_sha256':file_sha(output.parent/'models.pt'),
                'selected_indices':selected.tolist(),'concept_names':names,'input_mapping':{},'metrics':{}}
    for split,ids in folds.items():
        for view in ['original','changed']:
            with np.load(source/f'{split}_{view}.npz') as raw:
                assert np.array_equal(raw['ids'],ids)
                item = {'features':torch.from_numpy(raw['features'].copy()).float(),
                        'labels':torch.from_numpy(raw['labels'].copy()).long(),'indices':torch.from_numpy(ids.copy()).long()}
                item[model.injection.cache_key] = CanonicalInputs([(view,ids)],manifest,site,internal_root(config['data']['dataset']))
                pred = site_prediction(model,item,config,config['experiment']['device'])
                target,known = raw['targets'][:,selected],raw['known'][:,selected]
                prefix = 'lfpi_image' if site == 'target_image' else 'lfpi_internal'
                export_npz(output/f'{prefix}_{split}_{view}.npz',pred,item,names,targets=target,known=known)
                base = predict_features(baseline,item['features'],draws=1,seed=3000)
                export_npz(output.parent/f'lfcbm_{split}_{view}.npz',base,item,names,targets=target,known=known)
                metadata['input_mapping'][f'{split}/{view}'] = {'ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),
                    'rgb_source':manifest[f'rgb_{view}'],'source_npz_sha256':file_sha(source/f'{split}_{view}.npz')}
                if split != 'train':
                    teacher_values = np.load(teacher/f'{view}_teacher.npy',mmap_mode='r')[ids][:,selected]
                    metadata['metrics'][f'{split}/{view}'] = {'classification':classification_metrics(pred['logits'].numpy(),item['labels'].numpy()),
                        'semantic_concepts':concept_metrics(pred['concepts'].numpy(),target,known),
                        'teacher_fidelity':teacher_fidelity(pred['concepts'].numpy(),teacher_values)}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return metadata


def train(dataset,protocol,seed,site,device,batch=None):
    output = ROOT/'artifacts/matched/lf_sites'/dataset/'rn18_imagenet'/site
    if protocol == 'paired':
        output /= 'paired_augmentation'
    output /= f'seed{seed}'
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Site run already exists; refuse implicit restart or overwrite')
    data,info = make_data(dataset,protocol,seed,site)
    source,teacher,reference,manifest,saved,folds = info
    backbone(device,manifest)  # Verify encoder SHA before fitting anything.
    config = configuration(saved,site,output,device,batch)
    upstream()
    from pi_cbm.pipeline import baseline_from_state,build_model
    from pi_cbm.training.adapt import adapt
    from pi_cbm.randomness import seed_all
    cbm = baseline_from_state(saved['baseline'],device)
    seed_all(seed)
    model = build_model(cbm,data,saved['selected'],config,device)
    clean = {}
    with torch.no_grad():
        for name,split in data['splits'].items():
            clean[name] = torch.cat([cbm.concepts(b.to(device)).cpu() for b in split['features'].split(256)])
    output.mkdir(parents=True)
    import yaml
    (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    history,refit = adapt(model,data,clean,saved['selected'],config,device)
    torch.save({'baseline':saved['baseline'],'selected':saved['selected'],'pi_cbm':{k:v.cpu() for k,v in model.cbm.state_dict().items()},
        'generator':{k:v.cpu() for k,v in model.generator.state_dict().items()},'config':config,'concepts':saved['concepts']},output/'models.pt')
    exports = export(model,data,info,config,output/'single_draw',site)
    report = {'dataset':dataset,'site':site,'training_seed':seed,'train_views':['original','changed'] if protocol=='paired' else ['original'],
        'feature_provenance':manifest,'teacher_provenance':json.loads((teacher/'metadata.json').read_text()),
        'reused_baseline':str(reference/'models.pt'),'reused_baseline_sha256':file_sha(reference/'models.pt'),
        'baseline_retrained':False,'selected_indices':saved['selected'].tolist(),'selected_names':saved['concepts'],
        'bank_concepts':len(manifest['concept_names']),'selected_concepts':len(saved['selected']),
        'test_selection':False,'adapter_history':history,'adapter_refit':refit,'inference':exports}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'complete_site_run':str(output)}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['audit','encode-internal','train'],nargs='?',default='audit')
    p.add_argument('--dataset',choices=['rival10','cub','waterbirds'],default='rival10')
    p.add_argument('--protocol',choices=['original','paired'],default='original')
    p.add_argument('--site',choices=['target_image','target_internal'],default='target_image')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu');p.add_argument('--batch',type=int)
    args = p.parse_args();torch.set_num_threads(4)
    if args.batch is not None and args.batch < 1:
        p.error('batch must be positive')
    if args.action == 'audit':
        if args.device != 'cpu':
            p.error('Initial audit must use CPU')
        audit(args.dataset,args.protocol,args.seed)
    elif args.action == 'encode-internal':
        encode_internal(args.dataset,args.device,args.batch or 32)
    else:
        train(args.dataset,args.protocol,args.seed,args.site,args.device,args.batch)


if __name__ == '__main__':
    main()
