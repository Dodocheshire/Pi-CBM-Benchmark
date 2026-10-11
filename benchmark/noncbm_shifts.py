"""Evaluate frozen class-only heads on cached, subject-aligned distribution shifts."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .metrics import classification_metrics
from .report_utils import write_json


def evaluate(heads, source, device='cpu'):
    saved = torch.load(heads / 'best.pt', map_location='cpu', weights_only=True)
    with np.load(heads / 'test_predictions.npz') as reference:
        ids, labels = reference['ids'].copy(), reference['labels'].copy()
    files = sorted(source.glob('test_*.npz'))
    if not files:
        raise ValueError('No cached test shifts found')
    output = heads / 'shifts'
    output.mkdir(exist_ok=True)
    rows = []
    weight, bias = (saved['head'][name].to(device) for name in ['weight', 'bias'])
    mean, std = saved['mean'].cpu().numpy(), saved['std'].cpu().numpy()
    with (heads / 'best.pt').open('rb') as stream:
        fingerprint = hashlib.file_digest(stream, 'sha256').hexdigest()
    for file in files:
        with np.load(file) as values:
            if not np.array_equal(values['ids'], ids) or not np.array_equal(values['labels'], labels):
                raise ValueError('Corrupted test subjects/classes differ from original test')
            features = values['features']
            if features.shape[1] != len(mean):
                raise ValueError('Frozen-head and shifted feature dimensions differ')
            with torch.no_grad():
                logits = np.concatenate([
                    F.linear(torch.as_tensor((features[i:i+2048] - mean) / std,
                                             device=device).float(), weight, bias).cpu().numpy()
                    for i in range(0, len(features), 2048)
                ])
        setting = file.stem.removeprefix('test_')
        np.savez_compressed(output / f'{setting}.npz', logits=logits, ids=ids, labels=labels)
        rows.append({'setting': setting, **classification_metrics(logits, labels)})
    report = {'method': 'non_cbm', 'concept_bottleneck': False, 'nec': None,
              'head_sha256': fingerprint, 'source': str(source.resolve()),
              'head_directory': str(heads.resolve()), 'refitting': False,
              'selection': saved['selection'], 'seed': saved['seed'],
              'settings': len(rows), 'evaluation': rows}
    write_json(output / 'report.json', report)
    print(f'{heads}: {len(rows)} completed shifts', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(3)
    evaluate(args.heads, args.source, args.device)


if __name__ == '__main__':
    main()
