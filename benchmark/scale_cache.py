"""Stream shared frozen RN18 and CLIP teacher caches from audited scale indices."""
import argparse
import hashlib
import json
from pathlib import Path

import clip
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .lf import upstream
from .report_utils import write_json


class Images(Dataset):
    def __init__(self, records):
        self.records = records
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224), transforms.ToTensor()])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index]['path']) as image:
            return self.transform(image.convert('RGB'))


def cache(dataset, batch, device, workers):
    core = upstream()
    from pi_cbm.models.backbones import load_backbone
    index_root = Path('artifacts/scale') / dataset
    index_manifest = json.loads((index_root / 'manifest.json').read_text())
    records = [json.loads(line) for line in (index_root / 'images.jsonl').read_text().splitlines()]
    common = Path('artifacts/common') / dataset
    source, teacher_root = common / 'rn18_imagenet', common / 'teacher'
    for path in [source, teacher_root]:
        path.mkdir(parents=True, exist_ok=True)
    bank = core / 'configs/concepts' / ('imagenet_official.txt' if dataset.startswith('imagenet') else 'places365.txt')
    names = bank.read_text().splitlines()
    finish = source / 'original_features_COMPLETE.json'
    if not finish.exists() or not (teacher_root / 'original_COMPLETE.json').exists():
        model = load_backbone({'name': 'resnet18', 'precision': 'float32'}, device)
        teacher, _ = clip.load('/root/.cache/clip/ViT-B-16.pt', device=device, jit=False)
        teacher.eval().requires_grad_(False)
        with torch.no_grad():
            text = F.normalize(torch.cat([teacher.encode_text(part.to(device)).float()
                                          for part in clip.tokenize(names).split(256)]), dim=-1)
        np.save(teacher_root / 'text.npy', text.cpu().numpy())
        features = np.lib.format.open_memmap(source / 'original_features.npy', mode='w+',
                                             dtype='float32', shape=(len(records), 512))
        targets = np.lib.format.open_memmap(teacher_root / 'original_teacher.npy', mode='w+',
                                            dtype='float32', shape=(len(records), len(names)))
        rn_mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
        rn_std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
        clip_mean = torch.tensor([.48145466, .4578275, .40821073], device=device)[None, :, None, None]
        clip_std = torch.tensor([.26862954, .26130258, .27577711], device=device)[None, :, None, None]
        loader = DataLoader(Images(records), batch_size=batch, num_workers=workers,
                            shuffle=False, pin_memory=True)
        start = 0
        with torch.no_grad():
            for images in loader:
                images = images.to(device, non_blocking=True)
                end = start + len(images)
                features[start:end] = model((images-rn_mean)/rn_std).float().cpu().numpy()
                embedding = F.normalize(teacher.encode_image((images-clip_mean)/clip_std).float(), dim=-1)
                targets[start:end] = (embedding @ text.T).cpu().numpy()
                start = end
                if start % (batch*100) == 0 or start == len(records):
                    print(json.dumps({'dataset': dataset, 'encoded': start, 'total': len(records)}), flush=True)
        features.flush(); targets.flush()
        sha = hashlib.sha256()
        for key, value in model.state_dict().items():
            sha.update(key.encode()); sha.update(value.cpu().numpy().tobytes())
        write_json(finish, {'images': len(records), 'backbone_sha256': sha.hexdigest(),
                            'index_sha256': index_manifest['index_sha256']})
        write_json(teacher_root / 'original_COMPLETE.json', {'images': len(records), 'concepts': len(names)})
    features = np.load(source / 'original_features.npy', mmap_mode='r')
    targets = np.load(teacher_root / 'original_teacher.npy', mmap_mode='r')
    for split in ['train', 'val', 'test']:
        ids = np.array([r['id'] for r in records if r['split'] == split], dtype=np.int64)
        labels = np.array([r['label'] for r in records if r['split'] == split], dtype=np.int64)
        np.savez_compressed(source / f'{split}_original.npz', features=features[ids],
                            labels=labels, ids=ids)
        np.save(common / f'{split}_ids.npy', ids)
    manifest = {**index_manifest, 'backbone_name': 'resnet18', 'feature_dim': 512,
                'backbone_sha256': json.loads(finish.read_text())['backbone_sha256'],
                'historical_checkpoint': False, 'task_fitting': False,
                'preprocess': 'shared RGB Resize256 CenterCrop224; model-specific normalization',
                'bank_concepts': len(names), 'human_concept_annotations': False}
    write_json(source / 'manifest.json', manifest)
    write_json(teacher_root / 'metadata.json', {**manifest, 'names': names, 'phrases': names,
                                               'teacher': 'CLIP ViT-B/16 raw cosine',
                                               'concept_bank_sha256': hashlib.sha256(bank.read_bytes()).hexdigest()})
    print(json.dumps({'dataset': dataset, 'stage': 'cache_complete', 'bank_concepts': len(names)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['imagenet_200_per_class', 'places365'], required=True)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cache(args.dataset, args.batch, args.device, args.workers)


if __name__ == '__main__':
    main()
