"""Add shared RN50/CLIP RN50 features on the exact common CIFAR input pixels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .lf import upstream
from .lf_cifar import resize224
from .report_utils import write_json


def cache(dataset, backbone, batch, device):
    upstream()
    from pi_cbm.models.backbones import load_backbone
    base = Path('artifacts/common') / dataset
    original = base / 'rn18_imagenet'
    folder = 'rn50_imagenet' if backbone == 'resnet50' else 'clip_RN50'
    output = base / folder
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'original_features_COMPLETE.json'
    if not marker.exists():
        model = load_backbone({'name': backbone, 'precision': 'float32'}, device)
        model.eval().requires_grad_(False)
        rgb = np.load(base / 'cache/rgb_original.npy', mmap_mode='r')
        features = np.lib.format.open_memmap(output / 'original_features.npy', mode='w+',
                                             dtype='float32', shape=(len(rgb), model.feature_dim))
        values = ([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711]) if backbone == 'clip_RN50' else ([.485, .456, .406], [.229, .224, .225])
        mean = torch.tensor(values[0], device=device)[None, :, None, None]
        std = torch.tensor(values[1], device=device)[None, :, None, None]
        with torch.no_grad():
            for start in range(0, len(rgb), batch):
                image = resize224(rgb[start:start+batch], device)
                features[start:start+len(image)] = model((image-mean)/std).flatten(1).float().cpu().numpy()
                if start % (batch*100) == 0:
                    print(json.dumps({'dataset': dataset, 'backbone': backbone,
                                      'images': start+len(image), 'total': len(rgb)}), flush=True)
        features.flush()
        sha = hashlib.sha256()
        for key, value in model.state_dict().items():
            sha.update(key.encode()); sha.update(value.cpu().numpy().tobytes())
        write_json(marker, {'images': len(rgb), 'feature_dim': model.feature_dim,
                            'backbone_sha256': sha.hexdigest()})
    features = np.load(output / 'original_features.npy', mmap_mode='r')
    for split in ['train', 'val', 'test']:
        with np.load(original / f'{split}_original.npz') as common:
            temporary=output / f'{split}_original.pending.npz'
            np.savez_compressed(temporary, features=features[common['ids']],
                                labels=common['labels'], ids=common['ids'],
                                targets=common['targets'],known=common['known'],
                                concept_names=common['concept_names'])
            temporary.replace(output / f'{split}_original.npz')
    manifest = json.loads((original / 'manifest.json').read_text())
    metadata = json.loads(marker.read_text())
    manifest.update(backbone_name=backbone, backbone=backbone+' independently frozen; no task fitting',
                    feature_dim=metadata['feature_dim'], backbone_sha256=metadata['backbone_sha256'],
                    teacher_input='same canonical resized RGB; shared raw cosine teacher across backbones')
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({'dataset': dataset, 'backbone': backbone, 'stage': 'cache_complete'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['cifar10', 'cifar100'], default=['cifar10', 'cifar100'])
    parser.add_argument('--backbones', nargs='+', choices=['resnet50', 'clip_RN50'], default=['resnet50', 'clip_RN50'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    for dataset in args.datasets:
        for backbone in args.backbones:
            cache(dataset, backbone, args.batch, args.device)


if __name__ == '__main__':
    main()
