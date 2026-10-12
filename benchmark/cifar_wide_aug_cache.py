"""Independent q0 Gaussian augmentation features; existing CIFAR inputs stay immutable."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .lf import upstream
from .lf_cifar import resize224
from .report_utils import write_json

ROOT = Path(__file__).resolve().parents[1]
VIEW = 'gaussian_noise_s1'
FOLDERS = {'resnet50': 'rn50_imagenet', 'clip_RN50': 'clip_RN50'}


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def normalize(images, backbone):
    values = ((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711)) if backbone == 'clip_RN50' else ((.485, .456, .406), (.229, .224, .225))
    mean = images.new_tensor(values[0])[None, :, None, None]
    std = images.new_tensor(values[1])[None, :, None, None]
    return (images - mean) / std


def context(dataset, backbone):
    base = ROOT / 'artifacts/common' / dataset
    output = base / FOLDERS[backbone]
    raw = base / 'cache' / f'rgb_{VIEW}.npy'
    teacher = base / 'teacher' / f'{VIEW}_teacher.npy'
    for marker in [base / 'cache' / f'{VIEW}_COMPLETE.json', base / 'teacher' / f'{VIEW}_COMPLETE.json']:
        if not marker.is_file():
            raise ValueError(f'Incomplete source: {marker}')
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['backbone_name'] == backbone
    rgb = np.load(raw, mmap_mode='r')
    scores = np.load(teacher, mmap_mode='r')
    names = json.loads((base / 'teacher/metadata.json').read_text())['names']
    assert rgb.shape == (60000, 3, 32, 32) and rgb.dtype == np.uint8
    assert scores.shape == (60000, len(names))
    groups = {}
    for split in ['train', 'val', 'test']:
        with np.load(output / f'{split}_original.npz') as saved:
            groups[split] = {key: saved[key] for key in ['ids', 'labels', 'targets', 'known', 'concept_names']}
        assert groups[split]['concept_names'].tolist() == names
        assert groups[split]['targets'].shape == (len(groups[split]['ids']), len(names))
        assert not groups[split]['known'].any(), 'CIFAR native concept labels unavailable'
    ids = np.concatenate([groups[s]['ids'] for s in ['train', 'val', 'test']])
    assert np.array_equal(np.sort(ids), np.arange(60000)), 'Splits must partition all exact global rows'
    return output, manifest, rgb, scores, groups, raw, teacher


def loader(backbone, device, manifest):
    upstream()
    from pi_cbm.models.backbones import load_backbone
    model = load_backbone({'name': backbone, 'precision': 'float32'}, device)
    digest = state_sha(model)
    assert digest == manifest['backbone_sha256'], 'Encoder weights differ from existing original cache'
    assert model.feature_dim == manifest['feature_dim']
    return model, digest


def audit(dataset, backbone):
    output, manifest, rgb, scores, groups, raw, teacher = context(dataset, backbone)
    originals = {str(p): file_sha(p) for p in [output / 'manifest.json', *[output / f'{s}_original.npz' for s in groups]]}
    model, digest = loader(backbone, 'cpu', manifest)
    ids = groups['train']['ids'][:2]
    native = np.array(rgb[ids])
    q0 = resize224(native, 'cpu')
    independently_resized = F.interpolate(torch.as_tensor(native).float() / 255, size=(224, 224), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    assert torch.equal(q0, independently_resized)
    x = normalize(q0, backbone)
    from pi_cbm.models.backbones import target_preprocess
    # Read only the official normalization transform; canonical pixels are not recropped.
    protocol = target_preprocess(backbone)
    normalization = protocol.transforms[-1] if hasattr(protocol, 'transforms') else protocol
    from torchvision.transforms.functional import normalize as official_normalize
    expected_normalized = official_normalize(q0.clone(), normalization.mean, normalization.std)
    assert torch.equal(x, expected_normalized)
    # Independent unsplit upstream architecture, rather than comparing a function to itself.
    if backbone == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        full = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval().requires_grad_(False)
        reference = torch.nn.Sequential(*list(full.children())[:-1], torch.nn.Flatten(1))
    else:
        import clip
        full, _ = clip.load('RN50', device='cpu', jit=False)
        reference = full.visual.float().eval().requires_grad_(False)
    with torch.no_grad():
        y = model(x).flatten(1).float()
        expected = reference(x).flatten(1).float()
    error = float((y - expected).abs().max())
    assert error < 1e-6 and tuple(y.shape) == (2, manifest['feature_dim'])
    assert torch.isfinite(y).all()
    assert all(file_sha(Path(path)) == value for path, value in originals.items())
    return {'dataset': dataset, 'backbone': backbone, 'sample_global_ids': ids.tolist(), 'feature_shape': list(y.shape), 'backbone_sha256': digest, 'manifest_sha256': originals[str(output / 'manifest.json')], 'independent_unsplit_max_abs': error, 'canonical_resize_max_abs': 0., 'official_normalization_max_abs': 0., 'input_shape': list(q0.shape), 'pixel_protocol': 'q0: native uint8/255 -> bicubic224 antialias align_corners=False -> clamp; no intensity255 roundtrip, no crop', 'teacher_shape': list(scores.shape), 'teacher_sample_sha256': hashlib.sha256(np.array(scores[ids]).tobytes()).hexdigest(), 'source_rgb': str(raw), 'source_rgb_sha256': file_sha(raw), 'teacher': str(teacher), 'teacher_sha256': file_sha(teacher), 'original_files_unchanged': True, 'native_concept_annotations': False, 'encoding_started': False, 'cuda_initialized': torch.cuda.is_initialized()}


def cache(dataset, backbone, batch, device):
    output, manifest, rgb, scores, groups, raw, teacher = context(dataset, backbone)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f'{VIEW}_q0_encoding.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model, digest = loader(backbone, device, manifest)
        marker = output / f'{VIEW}_features_COMPLETE.json'
        destination = output / f'{VIEW}_features.npy'
        producer_sha = file_sha(Path(__file__))
        if marker.exists():
            saved = json.loads(marker.read_text())
            assert saved['backbone_sha256'] == digest and saved['pixel_protocol'] == 'canonical_q0'
            assert saved['features_sha256'] == file_sha(destination)
        else:
            if destination.exists():
                raise ValueError('Feature file lacks completion binding; refuse overwrite')
            pending = output / f'{VIEW}_features.pending.npy'
            if pending.exists():
                raise ValueError('Partial feature cache exists; preserve it and resolve before a fresh encoding')
            features = np.lib.format.open_memmap(pending, mode='w+', dtype='float32', shape=(60000, model.feature_dim))
            with torch.no_grad():
                for start in range(0, 60000, batch):
                    q0 = resize224(rgb[start:start + batch], device)
                    result = model(normalize(q0, backbone)).flatten(1).float().cpu().numpy()
                    assert result.shape == (len(q0), model.feature_dim) and np.isfinite(result).all()
                    features[start:start + len(q0)] = result
                    if start % (batch * 100) == 0:
                        print(json.dumps({'dataset': dataset, 'backbone': backbone, 'encoded_rows': start + len(q0), 'total_rows': 60000}), flush=True)
            features.flush()
            del features
            pending.replace(destination)
            write_json(marker, {'images': 60000, 'feature_dim': model.feature_dim, 'backbone_name': backbone, 'backbone_sha256': digest, 'features_sha256': file_sha(destination), 'pixel_protocol': 'canonical_q0', 'preprocess': 'native32 uint8/255, bicubic224 antialias align_corners=False, clamp, original per-backbone normalization; no crop or roundtrip', 'encoding_batch': batch, 'device': device, 'dtype': 'float32', 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32, 'producer_sha256': producer_sha, 'source_rgb': str(raw), 'source_rgb_sha256': file_sha(raw), 'teacher_sha256': file_sha(teacher), 'original_manifest_sha256': file_sha(output / 'manifest.json')})
        features = np.load(destination, mmap_mode='r')
        assert features.shape == (60000, model.feature_dim)
        exports = []
        for split, group in groups.items():
            target = output / f'{split}_{VIEW}.npz'
            if not target.exists():
                pending = target.with_name(target.stem + '.pending.npz')
                np.savez_compressed(pending, **group, features=features[group['ids']], teacher=scores[group['ids']])
                pending.replace(target)
            with np.load(target) as saved:
                assert all(np.array_equal(saved[k], v) for k, v in group.items())
                assert np.array_equal(saved['features'], features[group['ids']])
                assert np.array_equal(saved['teacher'], scores[group['ids']])
            exports.append({'split': split, 'path': str(target), 'sha256': file_sha(target), 'rows': len(group['ids'])})
        write_json(output / f'{VIEW}_q0_exports_COMPLETE.json', {'dataset': dataset, 'backbone': backbone, 'view': VIEW, 'pixel_protocol': 'canonical_q0', 'backbone_sha256': digest, 'features_marker': str(marker), 'exports': exports, 'same_original_ids_labels_native_fields': True, 'teacher_by_global_ids_exact': True, 'original_files_untouched': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'cache'], nargs='?', default='audit')
    parser.add_argument('--datasets', nargs='+', choices=['cifar10', 'cifar100'], default=['cifar10', 'cifar100'])
    parser.add_argument('--backbones', nargs='+', choices=list(FOLDERS), default=list(FOLDERS))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == 'audit':
        rows = [audit(dataset, backbone) for dataset in args.datasets for backbone in args.backbones]
        write_json(ROOT / 'artifacts/audit/cifar_wide_aug_cache_cpu.json', {'audits': rows, 'all_passed': True, 'producer_sha256': file_sha(Path(__file__)), 'full_encoding_started': False})
    else:
        for dataset in args.datasets:
            for backbone in args.backbones:
                cache(dataset, backbone, args.batch, args.device)


if __name__ == '__main__':
    main()
