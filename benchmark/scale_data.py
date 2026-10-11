"""Index local ImageNet and Places images into disjoint scale-evaluation splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .report_utils import write_json


def publish(name, records, names, output, protocol):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    paths = [r['path'] for r in records]
    if len(set(paths)) != len(paths):
        raise ValueError(f'{name}: repeated images across splits')
    missing = [p for p in paths if not Path(p).is_file()]
    if missing:
        raise ValueError(f'{name}: missing {len(missing)} indexed images: {missing[:3]}')
    index = directory / 'images.jsonl'
    with index.open('w') as stream:
        for identity, record in enumerate(records):
            stream.write(json.dumps({'id': identity, **record}) + '\n')
    counts = {split: dict(sorted(Counter(r['label'] for r in records
                                        if r['split'] == split).items()))
              for split in ['train', 'val', 'test']}
    report = {'dataset': name, 'classes': len(names), 'class_names': names,
              'split_counts': {s: sum(c.values()) for s, c in counts.items()},
              'per_class_counts': counts, 'index': str(index.resolve()),
              'index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
              'all_indexed_files_exist': True, 'disjoint_image_paths': True,
              'protocol': protocol}
    write_json(directory / 'manifest.json', report)
    return report


def imagenet(base, output):
    source = base / 'imagenet/ILSVRC/Data/CLS-LOC'
    training = source / 'ImageNet_train_200_10_clean'
    names = sorted(p.name for p in (training / 'train').iterdir() if p.is_dir())
    records = []
    for label, name in enumerate(names):
        for split, folder in [('train', training / 'train'),
                              ('val', training / 'adapter_val'),
                              ('test', source / 'ImageNet_val')]:
            files = sorted(p for p in (folder / name).iterdir() if p.is_file())
            records.extend({'split': split, 'label': label, 'path': str(p.resolve())}
                           for p in files)
    return publish('imagenet_200_per_class', records, names, output,
                   {'train': 'existing clean 200 images/class training subset',
                    'val': 'existing independent 10 images/class adapter validation',
                    'test': 'official ILSVRC validation images',
                    'training_source': json.loads((training / 'source.json').read_text()),
                    'native_concept_annotations': False})


def places(base, output, train_count, val_count, seed):
    source = base / 'places365_torch'
    categories = [line.split() for line in (source / 'categories_places365.txt').read_text().splitlines()]
    names = [None] * len(categories)
    for name, label in categories:
        names[int(label)] = name
    # Reservoir sampling examines all existing listed training images, avoiding
    # a filename-order subset. Official validation images remain the test set.
    rng = np.random.default_rng(seed)
    size = train_count + val_count
    samples = [[] for _ in names]
    seen = np.zeros(len(names), dtype=np.int64)
    unavailable = 0
    with (source / 'places365_train_standard.txt').open() as stream:
        for line in stream:
            relative, label_text = line.split()
            label = int(label_text)
            path = source / 'data_256_standard' / relative.lstrip('/')
            if not path.is_file():
                unavailable += 1
                continue
            seen[label] += 1
            if len(samples[label]) < size:
                samples[label].append(str(path.resolve()))
            else:
                position = int(rng.integers(seen[label]))
                if position < size:
                    samples[label][position] = str(path.resolve())
    if any(len(group) < size for group in samples):
        raise ValueError('Places training images insufficient for balanced requested sample')
    records = []
    for label, paths in enumerate(samples):
        rng.shuffle(paths)
        records.extend({'split': 'train' if i < train_count else 'val',
                        'label': label, 'path': path} for i, path in enumerate(paths))
    with (source / 'places365_val.txt').open() as stream:
        for line in stream:
            relative, label = line.split()
            records.append({'split': 'test', 'label': int(label),
                            'path': str((source / 'val_256' / relative).resolve())})
    return publish('places365', records, names, output,
                   {'train_per_class': train_count, 'val_per_class': val_count,
                    'sampling': 'seeded uniform reservoir over existing official training files',
                    'split_seed': seed, 'test': 'all official validation images',
                    'available_training_images_per_class': seen.tolist(),
                    'missing_listed_training_files': unavailable,
                    'native_concept_annotations': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/home/pengcheng/VLG-CBM/datasets'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/scale'))
    parser.add_argument('--dataset', choices=['imagenet', 'places', 'both'], default='both')
    parser.add_argument('--places-train-per-class', type=int, default=200)
    parser.add_argument('--places-val-per-class', type=int, default=20)
    args = parser.parse_args()
    reports = []
    if args.dataset in ['imagenet', 'both']:
        reports.append(imagenet(args.data, args.out))
    if args.dataset in ['places', 'both']:
        reports.append(places(args.data, args.out, args.places_train_per_class,
                              args.places_val_per_class, 42))
    write_json(Path('reports/scale_data_inventory.json'), reports)
    for report in reports:
        print(json.dumps({'dataset': report['dataset'], 'classes': report['classes'],
                          'split_counts': report['split_counts']}), flush=True)


if __name__ == '__main__':
    main()
