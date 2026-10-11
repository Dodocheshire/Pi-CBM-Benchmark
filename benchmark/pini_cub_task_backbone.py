"""PiNI external adapter for the independently audited task-trained CUB RN18.

Default prepare is CPU-only and requires the complete shared task feature cache.
Only the explicit train action calls unchanged PiNI training code. No core files
or shared backbone definitions are modified.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch

from benchmark.pini_matched import CORE, ROOT, MatchedExperiment, export, initialize, write
from benchmark.task_backbones import file_sha
from benchmark.cub_task_backbone import NAMESPACE, context, load_cub_task_backbone
from benchmark.cub_task_cache import CHECKPOINT


class CUBTaskMatchedExperiment(MatchedExperiment):
    def __init__(self):
        super().__init__('cub', ROOT / 'artifacts/pini_matched/resnet18_cub_trainonly', 'resnet18')
        self.backbone_name = 'resnet18_cub_trainonly'
        self.common = ROOT / 'artifacts/common/cub' / NAMESPACE

    def backbone(self):
        model = load_cub_task_backbone(CHECKPOINT, device='cpu')
        assert model.checkpoint_sha256 == '22cd942efcc814a30cbfae8c03beaf9ed02501a6d394a1b34255ffda14759635'
        assert model.state_sha256 == 'fad31642590c6eec95a2a1ef03dc6186668b7e988e5a5d47ad877dedc8443ee1'
        # Aliases describe the existing full-network hash; no weights are changed.
        model.output_channels = 512
        model.weight_sha256 = model.state_sha256
        model.weight_sha256_scope = 'full torchvision network state_dict, including fc'
        return model


def link(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    if destination.is_symlink():
        if destination.resolve() != source:
            raise ValueError(f'Existing link has another source: {destination}')
    elif destination.exists():
        raise ValueError(f'Refuse to replace existing file: {destination}')
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def cpu_audit(experiment):
    model = experiment.backbone()
    _, source, folds = context()
    ids = np.asarray([folds['train'][0], folds['val'][0]], dtype=np.int64)
    raw = np.load(source['rgb_original'], mmap_mode='r')
    rgb = torch.from_numpy(np.array(raw[ids])).float() / 255
    model.train(True)
    assert not model.training and not model.network.training
    with torch.no_grad():
        spatial = model(rgb)
        logits = model.network.fc(spatial.mean((2, 3)))
        original = model.network(model.normalize(rgb))
    torch.testing.assert_close(logits, original, rtol=1e-5, atol=1e-5)
    value = rgb[:1].clone().requires_grad_(True)
    model(value).mean((2, 3)).sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().max() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    evidence = {
        'pid': os.getpid(), 'device': 'cpu', 'gpu_jobs_started': False,
        'formal_training_started': False, 'subject_ids': ids.tolist(),
        'checkpoint_sha256': model.checkpoint_sha256,
        'backbone_sha256': model.state_sha256,
        'backbone_sha256_scope': model.weight_sha256_scope,
        'spatial_shape': list(spatial.shape), 'feature_channels': model.output_channels,
        'class_logits_vs_full_network_max_abs': float((logits-original).abs().max()),
        'input_gradient_max_abs': float(value.grad.abs().max()),
        'frozen_parameter_gradients_absent': True, 'batchnorm_frozen': True,
        'backbone_supervision': 'CUB200 species class labels on 4796 train images; 1198 validation images select epoch11; test excluded',
        'concept_supervision': 'Koh species-level112 binary attributes for training; unchanged core BCE; image-level112 certainty>=3 targets only for evaluation',
        'namespace': NAMESPACE,
    }
    write(experiment.root / 'adapter_cpu_audit.json', evidence)
    return evidence


def prepare(experiment):
    e = experiment
    required = [e.common / 'manifest.json']
    for view in ['original', 'changed']:
        required += [e.common / f'{view}_COMPLETE.json', e.common / f'spatial_{view}.npy', e.common / f'features_{view}.npy']
        required += [e.common / f'{split}_{view}.npz' for split in ['train', 'val', 'test']]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError('Shared task cache incomplete; no encoding or training started: ' + ', '.join(missing))
    manifest = json.loads((e.common / 'manifest.json').read_text())
    _, source, folds = context()
    protocol_path = Path(source['split_protocol'])
    model = e.backbone()
    assert manifest['backbone_sha256'] == model.state_sha256
    assert manifest['checkpoint_sha256'] == model.checkpoint_sha256
    assert manifest['evaluation_namespace'] == NAMESPACE
    assert manifest['task_fit_sample_ids'] == folds['train'].tolist()
    assert manifest['backbone_validation_selection_ids'] == folds['val'].tolist()
    assert manifest['counts'] == {'train': 4796, 'val': 1198, 'test': 5794}
    assert manifest['task_trained'] and not manifest['historical_checkpoint']
    assert manifest['gradient_exposure'] == {'train': 4796, 'val': 0, 'test': 0}
    assert manifest['selected_epoch'] == 11
    protocol = json.loads((e.original / 'cache/protocol.json').read_text())
    for split, ids in folds.items():
        assert np.array_equal(np.asarray(protocol['splits'][split], dtype=np.int64), ids)
    training_attrs = np.load(e.original / 'cache/attrs.npy', mmap_mode='r')
    class_labels = np.load(e.original / 'cache/labels.npy', mmap_mode='r')
    assert training_attrs.shape == (11788, 112)
    for species in np.unique(class_labels):
        values = training_attrs[class_labels == species]
        assert np.array_equal(values, np.broadcast_to(values[0], values.shape))
    n = sum(len(ids) for ids in folds.values())
    assert np.array_equal(np.sort(np.concatenate(list(folds.values()))), np.arange(n))
    cache = e.root / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    audit = {}
    # Row provenance comes from the frozen sequential producer, not a fabricated
    # historical IDs field (historical region COMPLETE has no RGB hash).
    from spatial_noise import cub_regions
    producer_source = inspect.getsource(cub_regions.regions)
    if 'rgb[offset:end]' not in producer_source or 'value[offset:end]' not in producer_source:
        raise ValueError('Unrecognized region producer row assignment')
    for view in ['original', 'changed']:
        receipt = json.loads((e.common / f'{view}_COMPLETE.json').read_text())
        assert receipt['images'] == n and receipt['spatial_float32']
        assert receipt['backbone_sha256'] == model.state_sha256
        assert receipt['checkpoint_sha256'] == model.checkpoint_sha256
        rgb_path = e.original / 'cache' / f'rgb_{view}.npy'
        rgb_sha = file_sha(rgb_path)
        assert rgb_sha == receipt['rgb_sha256']
        common_rgb = Path(manifest[f'rgb_{view}'])
        assert common_rgb.resolve() == Path(source[f'rgb_{view}']).resolve()
        if common_rgb.resolve() != rgb_path.resolve():
            assert file_sha(common_rgb) == rgb_sha
        raw = np.load(rgb_path, mmap_mode='r')
        assert raw.shape == (n, 3, 224, 224) and raw.dtype == np.uint8
        spatial_path = e.common / f'spatial_{view}.npy'
        global_path = e.common / f'features_{view}.npy'
        spatial = np.load(spatial_path, mmap_mode='r')
        pooled = np.load(global_path, mmap_mode='r')
        assert spatial.shape == (n, 512, 7, 7) and spatial.dtype == np.float32
        assert pooled.shape == (n, 512) and pooled.dtype == np.float32
        error = 0.
        for start in range(0, n, 128):
            value = torch.from_numpy(np.array(spatial[start:start+128])).mean((2, 3)).numpy()
            error = max(error, float(np.abs(value-pooled[start:start+128]).max()))
        assert error == 0, {'global_meanpool_max_abs': error}
        region_path = e.original / 'regions' / view
        region_receipt = json.loads((region_path / 'COMPLETE.json').read_text())
        assert region_receipt['done'] == n
        assert region_receipt.get('precision', 'fp32') == 'fp32'
        assert region_receipt['image_size'] == 800 and region_receipt['threshold'] == .25
        assert region_receipt['phrases'] == list(e.phrases)
        for key in ['features', 'maps', 'valid', 'boxes', 'scores']:
            assert len(np.load(region_path / f'{key}.npy', mmap_mode='r')) == n
        teacher = np.load(Path(manifest['teacher_root']) / f'{view}_teacher.npy', mmap_mode='r')
        for split, ids in folds.items():
            with np.load(e.common / f'{split}_{view}.npz') as values, np.load(Path(manifest['source_root']) / f'{split}_{view}.npz') as original:
                assert np.array_equal(values['ids'], ids)
                assert np.array_equal(values['features'], pooled[ids])
                assert np.array_equal(values['teacher'], teacher[ids])
                assert np.array_equal(values['concept_names'], np.asarray(manifest['concept_names']))
                for key in ['labels', 'targets', 'known', 'training_targets']:
                    assert np.array_equal(values[key], original[key])
                attrs = np.load(e.original / 'cache/attrs.npy', mmap_mode='r')
                labels = np.load(e.original / 'cache/labels.npy', mmap_mode='r')
                assert np.array_equal(values['training_targets'], attrs[ids])
                assert np.array_equal(values['labels'], labels[ids])
        sample_ids = np.asarray([folds['train'][0], folds['val'][0]], dtype=np.int64)
        with torch.no_grad():
            live = model(torch.from_numpy(np.array(raw[sample_ids])).float() / 255).numpy()
        reference = np.array(spatial[sample_ids])
        live_max = float(np.abs(live-reference).max())
        live_relative_l2 = float(np.linalg.norm(live-reference) / max(np.linalg.norm(reference), 1e-12))
        # CPU convolution batch/thread algorithms can round differently; the
        # immutable complete state SHA is the strict source-weight identity gate.
        assert live_relative_l2 < 1e-5, {'live_relative_l2': live_relative_l2}
        audit[view] = {
            'rgb_sha256': rgb_sha, 'spatial_sha256': file_sha(spatial_path),
            'shared_global_sha256': file_sha(global_path), 'global_meanpool_max_abs_error': error,
            'region_complete_sha256': file_sha(region_path / 'COMPLETE.json'),
            'region_rows': n, 'ordered_ids_sha256': hashlib.sha256(np.arange(n, dtype=np.int64).tobytes()).hexdigest(),
            'region_order_evidence': 'frozen producer assigns output[offset:end] from rgb[offset:end]; same source RGB full-file SHA, completed row count and canonical ordered split IDs verified',
            'region_producer_file_sha256': file_sha(cub_regions.__file__),
            'region_producer_function_sha256': hashlib.sha256(producer_source.encode()).hexdigest(),
            'historical_region_marker_contains_rgb_hash': False,
            'region_precision_evidence': 'core cub_regions default fp32; legacy marker may omit precision field',
            'checkpoint_sha256': model.checkpoint_sha256, 'backbone_sha256': model.state_sha256,
            'live_cpu_sample_ids': sample_ids.tolist(),
            'live_cpu_vs_cached_spatial_max_abs': live_max,
            'live_cpu_vs_cached_spatial_relative_l2': live_relative_l2,
            'live_cpu_exact_feature_identity_claimed': live_max == 0,
        }
        link(spatial_path, cache / f'features_{view}.npy')
        write(cache / f'{view}_spatial_COMPLETE.json', audit[view])
    for path in (e.original / 'cache').iterdir():
        if path.name.startswith('features_') or path.name in ['protocol.json', 'COMPLETE.json']:
            continue
        link(path, cache / path.name)
    link(e.original / 'regions', e.root / 'regions')
    protocol.update(backbone=manifest['backbone'], backbone_sha256=model.state_sha256,
                    backbone_task_fit_sample_ids=folds['train'].tolist(),
                    backbone_validation_selection_ids=folds['val'].tolist(),
                    backbone_preprocess=manifest['preprocess'], downstream_stage='selection_train_only',
                    new_training=True, spatial_feature_audit=audit)
    write(cache / 'protocol.json', protocol)
    write(e.root / 'input_manifest.json', {
        'dataset': e.name, 'backbone_name': e.backbone_name, 'common': manifest,
        'spatial_audit': audit, 'holdout_audited': True, 'cache_backbone_matches_live': True,
        'feature_channels': 512, 'feature_layout': 'native layer4 spatial map',
        'fit_split': 'train only', 'selection_split': 'val; also selected new backbone epoch11 from15 epochs',
        'test_split': 'never used for fitting or checkpoint selection',
        'concept_supervision': 'Koh species-level112 binary attributes for training; original core BCE; image-level112 certainty>=3 only for evaluation',
        'backbone_label_supervision': manifest['backbone_supervision'],
        'generator_region_supervision': 'unchanged frozen DINO12 bird phrases/800/.25/default fp32; no native foreground passed to generator',
        'core_revision': 'eb2e4363ed30e4f5a6456ea781288c10cd4cbfcd',
        'core_source_sha256': {name: file_sha(CORE / 'spatial_noise' / name) for name in ['rn18_datasets.py', 'cub_aligned.py', 'cub_noise.py', 'noise_pair.py']},
        'shared_cache_manifest_sha256': file_sha(e.common / 'manifest.json'),
        'split_protocol_sha256': file_sha(protocol_path), 'pid': os.getpid(),
        'prepared_on_cpu': True, 'formal_training_started': False,
    })
    return cpu_audit(e)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'prepare', 'train'], nargs='?', default='prepare')
    args = parser.parse_args()
    torch.set_num_threads(3)
    e = CUBTaskMatchedExperiment()
    if args.action == 'audit':
        result = cpu_audit(e)
    else:
        result = prepare(e)
        if args.action == 'train':
            # Explicit later action only: same initialization chain, seed42,
            # selection stage, losses, steps and pair count as ImageNetRN18.
            initialize(e, steps=1000, epochs=10, batch=64)
            from spatial_noise.noise_pair import train
            result = {}
            for method in ['mean', 'noise']:
                train(e, method, 'selection', epochs=5, pairs=32)
                result[method] = export(e, method, batch=64)
            write(e.root / 'matched_results.json', result)
    print(json.dumps({'action': args.action, 'pid': os.getpid(), 'root': str(e.root), 'result': result}), flush=True)


if __name__ == '__main__':
    main()
