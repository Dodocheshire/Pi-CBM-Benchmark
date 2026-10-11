"""Shared task-trained RN18 adapter outside both frozen model cores.

Audit is CPU-only. The explicit encode action is a later opt-in operation.
The adapter receives canonical RGB floats in [0,1] and normalizes internally.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path('/home/pengcheng/Pi-CBM/PiNI/artifacts/rival10_aligned_rn18/backbone/selection/best.pt')
NAMESPACE = 'rn18_task_pretrained_train_val_selected'
EXPECTED_CHECKPOINT_SHA = 'ac58a248f6cbbe09cf48c49c0662181037110738bbd6d87e3241a580cec6b8fc'


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class TaskRN18(nn.Module):
    """Frozen class-supervised backbone, returning unnormalized spatial maps."""
    feature_dim = 512

    def __init__(self, checkpoint=CHECKPOINT):
        super().__init__()
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = file_sha(self.checkpoint)
        if self.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA:
            raise ValueError('This registered namespace accepts only the audited RIVAL selection checkpoint')
        saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        if saved['stage'] != 'selection' or saved['fit_images'] != 18987:
            raise ValueError('Reject a trainval-refit or unregistered task checkpoint')
        self.network = resnet18(weights=None)
        self.network.fc = nn.Linear(512, 10)
        self.network.load_state_dict(saved['model'], strict=True)
        self.state_sha256 = state_sha(self.network.state_dict())
        if self.state_sha256 != state_sha(saved['model']):
            raise ValueError('Loaded network state differs from checkpoint state')
        self.metadata = {key: value for key, value in saved.items() if key != 'model'}
        self.eval().requires_grad_(False)

    def train(self, mode=True):
        # A frozen backbone must also freeze BatchNorm buffers under parent.train().
        return super().train(False)

    @staticmethod
    def normalize(rgb):
        if not rgb.is_floating_point():
            raise ValueError('Task adapter expects RGB float divided by255; uint8 is not normalized automatically')
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 224, 224):
            raise ValueError('Expected canonical N3x224x224 RGB float, no additional resize/crop')
        return (rgb - rgb.new_tensor([.485, .456, .406])[None, :, None, None]) / rgb.new_tensor([.229, .224, .225])[None, :, None, None]

    def forward(self, rgb):
        # Keep the exact torchvision computation and immutable BN statistics.
        network = self.network
        value = network.maxpool(network.relu(network.bn1(network.conv1(self.normalize(rgb)))))
        value = network.layer1(value)
        value = network.layer2(value)
        value = network.layer3(value)
        return network.layer4(value)

    def global_features(self, rgb):
        return self.forward(rgb).mean((2, 3))

    def class_logits(self, rgb):
        return self.network.fc(self.global_features(rgb))


def load_task_backbone(checkpoint=CHECKPOINT, device='cpu'):
    return TaskRN18(checkpoint).to(device).eval().requires_grad_(False)


def context():
    source = ROOT / 'artifacts/common/rival10/rn18_imagenet'
    source_manifest = json.loads((source / 'manifest.json').read_text())
    protocol_path = Path(source_manifest['split_protocol'])
    protocol = json.loads(protocol_path.read_text())
    folds = {}
    for split in ['train', 'val', 'test']:
        with np.load(source / f'{split}_original.npz') as values:
            ids = values['ids'].copy()
        if not np.array_equal(ids, np.asarray(protocol['splits'][split], dtype=np.int64)):
            raise ValueError('Current common IDs differ from task-backbone historical split evidence')
        folds[split] = ids
    if [len(folds[s]) for s in ['train', 'val', 'test']] != [18987, 2111, 5286]:
        raise ValueError('Unregistered RIVAL folds')
    if any(set(folds[a]) & set(folds[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Underlying subject IDs overlap across folds')
    return source, source_manifest, protocol_path, folds


def provenance(model, source_manifest, protocol_path, folds):
    evidence = ROOT / 'artifacts/audit/task_backbones.json'
    metadata = {
        'dataset': 'rival10', 'evaluation_namespace': NAMESPACE,
        'backbone_name': 'resnet18_rival10', 'core_global_model_backbone_name': 'resnet18', 'backbone': 'torchvision ResNet18, ImageNet-initialized then class-supervised on current RIVAL train only; frozen',
        'feature_dim': 512, 'feature_shape': [len(folds['test']), 512],
        'all_row_feature_shape': [sum(len(ids) for ids in folds.values()), 512], 'spatial_shape': [512, 7, 7],
        'features': 'raw spatial average; no LN or fitted feature normalization',
        'checkpoint': str(model.checkpoint), 'checkpoint_sha256': model.checkpoint_sha256,
        'backbone_sha256': model.state_sha256, 'backbone_sha256_scope': 'full torchvision network state_dict, including classifier',
        'checkpoint_metadata': model.metadata,
        'backbone_supervision': 'RIVAL10 class labels, cross entropy; no concept supervision',
        'task_fit_sample_ids': folds['train'].tolist(),
        'backbone_validation_selection_ids': folds['val'].tolist(),
        'task_fit_counts': {'train': 18987, 'val_gradient': 0, 'test_gradient': 0},
        'validation_role': 'current 2111 validation images selected backbone epoch14; not gradient-trained; subsequent CBM/head val selection uses this same development fold',
        'test_role': 'all5286 test subjects excluded from local backbone fitting and checkpoint selection',
        'preprocess': 'exact shared canonical224 uint8 RGB /255, no additional crop/resize; ImageNet mean/std inside adapter',
        'loader_input': 'N3x224x224 RGB float in [0,1]; forward outputs N512x7x7; global_features outputs N512',
        'split_protocol': str(protocol_path), 'split_protocol_sha256': file_sha(protocol_path),
        'counts': {split: len(ids) for split, ids in folds.items()},
        'splits': {split: ids.tolist() for split, ids in folds.items()},
        'source_imagenet_manifest': str(ROOT / 'artifacts/common/rival10/rn18_imagenet/manifest.json'),
        'historical_checkpoint': False, 'registered_existing_task_checkpoint': True,
        'checkpoint_history': 'Previously trained class-supervised selection checkpoint; independently audited train-only gradient IDs/val epoch selection; not a trainval-refit representation',
        'feature_cache_protocol': 'new encoding from registered checkpoint; never alias ImageNet or trainval-refit features',
        'teacher': 'same independently frozen CLIP ViT-B/16, exact shared candidate names/raw cosine arrays; teacher not refit',
        'audit_source': str(evidence), 'audit_source_sha256': file_sha(evidence),
    }
    for key in ['cache_root', 'rgb_original', 'rgb_changed', 'labels', 'concept_names', 'training_targets', 'targets', 'known', 'native_audit']:
        metadata[key] = source_manifest[key]
    teacher_root = ROOT / 'artifacts/common/rival10/teacher'
    metadata['teacher_root'] = str(teacher_root)
    metadata['teacher_inputs'] = {name: {'path': str(teacher_root / name), 'sha256': file_sha(teacher_root / name)}
                                 for name in ['original_teacher.npy', 'changed_teacher.npy', 'text.npy', 'metadata.json']}
    return metadata


def audit(checkpoint=CHECKPOINT):
    """No GPU use, no full cache creation, and no change to shared source arrays."""
    model = load_task_backbone(checkpoint, 'cpu')
    _, source_manifest, protocol_path, folds = context()
    ids = np.asarray([folds['train'][0], folds['val'][0]], dtype=np.int64)
    raw = np.load(source_manifest['rgb_original'], mmap_mode='r')
    if raw.dtype != np.uint8 or tuple(raw.shape[1:]) != (3, 224, 224):
        raise ValueError('Shared canonical RGB has unexpected dtype/shape')
    rgb = torch.from_numpy(np.array(raw[ids])).float() / 255
    with torch.no_grad():
        spatial = model(rgb)
        global_features = model.global_features(rgb)
        classifier = model.class_logits(rgb)
        network_classifier = model.network(model.normalize(rgb))
    assert tuple(spatial.shape) == (2, 512, 7, 7)
    assert tuple(global_features.shape) == (2, 512)
    assert tuple(classifier.shape) == (2, 10)
    assert torch.isfinite(spatial).all()
    torch.testing.assert_close(global_features, spatial.mean((2, 3)), rtol=0, atol=0)
    torch.testing.assert_close(classifier, network_classifier, rtol=1e-5, atol=1e-5)
    model.train(True)
    assert not model.training and not model.network.training
    differentiable_rgb = rgb[:1].clone().requires_grad_(True)
    model.global_features(differentiable_rgb).sum().backward()
    assert differentiable_rgb.grad is not None and torch.isfinite(differentiable_rgb.grad).all()
    assert differentiable_rgb.grad.abs().max() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    result = provenance(model, source_manifest, protocol_path, folds)
    result.update(
        audit_device='cpu', full_cache_started=False, gpu_jobs_started=False,
        audit_subject_rows=ids.tolist(), sample_rgb_bytes_sha256=hashlib.sha256(np.array(raw[ids]).tobytes()).hexdigest(),
        rgb_source_sha256=file_sha(source_manifest['rgb_original']),
        spatial_audit_shape=list(spatial.shape), global_audit_shape=list(global_features.shape),
        classifier_audit_shape=list(classifier.shape),
        class_logits_vs_full_network_max_abs=float((classifier-network_classifier).abs().max()),
        state_sha_verified=True, spatial_global_exact_parity=True, frozen_batchnorm_train_override_verified=True,
        input_gradient_supported=True, frozen_parameter_gradients_absent=True,
    )
    output = ROOT / 'artifacts/audit/rival_task_backbone_cpu.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'cpu_audit': str(output), 'namespace': NAMESPACE, 'shape': list(global_features.shape)}), flush=True)
    return result


def encode(checkpoint=CHECKPOINT, device='cpu', batch=64, spatial=False):
    """Explicit later action; full cache generation is never part of audit."""
    model = load_task_backbone(checkpoint, device)
    source, source_manifest, protocol_path, folds = context()
    output = ROOT / 'artifacts/common/rival10' / NAMESPACE
    output.mkdir(parents=True, exist_ok=True)
    metadata = provenance(model, source_manifest, protocol_path, folds)
    teacher_root = ROOT / 'artifacts/common/rival10/teacher'
    for view in ['original', 'changed']:
        if not (teacher_root / f'{view}_COMPLETE.json').exists():
            raise ValueError('Shared same-view teacher incomplete')
        raw = np.load(source_manifest[f'rgb_{view}'], mmap_mode='r')
        marker = output / f'{view}_COMPLETE.json'
        features_path = output / f'features_{view}.npy'
        if marker.exists():
            receipt = json.loads(marker.read_text())
            if receipt['checkpoint_sha256'] != model.checkpoint_sha256:
                raise ValueError('Existing task namespace has a different checkpoint')
            if spatial and not (output / f'spatial_{view}.npy').exists():
                raise ValueError('Existing cache lacks requested spatial maps; choose a fresh explicit namespace/protocol before reencoding')
        else:
            pending = output / f'features_{view}.pending.npy'
            features = np.lib.format.open_memmap(pending, mode='w+', dtype='float32', shape=(len(raw), 512))
            maps = None
            if spatial:
                maps = np.lib.format.open_memmap(output / f'spatial_{view}.pending.npy', mode='w+', dtype='float32', shape=(len(raw), 512, 7, 7))
            with torch.no_grad():
                for start in range(0, len(raw), batch):
                    rgb = torch.from_numpy(np.array(raw[start:start+batch])).to(device).float() / 255
                    values = model(rgb)
                    features[start:start+len(rgb)] = values.mean((2, 3)).float().cpu().numpy()
                    if maps is not None:
                        maps[start:start+len(rgb)] = values.float().cpu().numpy()
                    if start % (batch * 50) == 0:
                        print(json.dumps({'view': view, 'images': start+len(rgb), 'total': len(raw)}), flush=True)
            features.flush(); pending.replace(features_path)
            if maps is not None:
                maps.flush(); (output / f'spatial_{view}.pending.npy').replace(output / f'spatial_{view}.npy')
            marker.write_text(json.dumps({'view': view, 'images': len(raw), 'encoding_batch': batch,
                                         'device': device, 'checkpoint_sha256': model.checkpoint_sha256,
                                         'backbone_sha256': model.state_sha256, 'spatial_float32': spatial,
                                         'rgb_sha256': file_sha(source_manifest[f'rgb_{view}'])}, indent=2))
        features = np.load(features_path, mmap_mode='r')
        teacher = np.load(teacher_root / f'{view}_teacher.npy', mmap_mode='r')
        for split, ids in folds.items():
            with np.load(source / f'{split}_{view}.npz') as values:
                if not np.array_equal(values['ids'], ids):
                    raise ValueError('Shared ordered split/view IDs mismatch')
                fields = {key: values[key] for key in ['labels', 'ids', 'targets', 'known', 'training_targets']}
                fields['concept_names'] = np.asarray(source_manifest['concept_names'])
                destination = output / f'{split}_{view}.npz'
                pending = output / f'{split}_{view}.pending.npz'
                np.savez_compressed(pending, features=features[ids], teacher=teacher[ids], **fields)
                pending.replace(destination)
    metadata.update(encoding_device=device, encoding_batch=batch, spatial_float32=spatial,
                    teacher_root=str(teacher_root), source_root=str(source))
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'task_cache_complete': str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'encode'], nargs='?', default='audit')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--spatial', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == 'audit':
        if args.device != 'cpu':
            parser.error('Audit is deliberately CPU-only; full encode requires a separate explicit action')
        audit(args.checkpoint)
    else:
        if args.batch < 1:
            parser.error('Batch must be positive')
        encode(args.checkpoint, args.device, args.batch, args.spatial)


if __name__ == '__main__':
    main()
