"""Apply already fitted sparse/dense heads to aligned concept shift exports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .metrics import classification_metrics, nec
from .report_utils import write_json


def evaluate(directory, source, prefix, field, device):
    report = json.loads((directory / 'report.json').read_text())
    models = torch.load(directory / 'heads.pt', map_location='cpu', weights_only=True)
    norm = np.load(directory / 'normalization.npz')
    reference = np.load(directory / 'dense_original.npz')
    output = directory / 'shifts'
    output.mkdir(exist_ok=True)
    results = []
    files = sorted(source.glob(prefix + 'test_*.npz'))
    if not files:
        raise ValueError('No shifted test exports matched the requested prefix')
    with (directory / 'heads.pt').open('rb') as stream:
        fingerprint = hashlib.file_digest(stream, 'sha256').hexdigest()
    for path in files:
        setting = path.stem.removeprefix(prefix + 'test_')
        with np.load(path) as values:
            if not np.array_equal(values['ids'], reference['ids']) or not np.array_equal(values['labels'], reference['labels']):
                raise ValueError('Shift test subjects/classes differ from the original test evaluation')
            raw = values[field]
            if raw.shape[1] != len(norm['mean']):
                raise ValueError('Shift concept count differs from fitted head normalization')
            heads = {'best_val': models['sparse']['best'], **models['sparse']['budgets'],
                     'dense': models['dense']['head']}
            for name, head in heads.items():
                if name == 'dense':
                    mean, scale = models['dense']['mean'].numpy(), models['dense']['std'].numpy()
                else:
                    mean, scale = norm['mean'], norm['scale']
                logits = []
                weight, bias = head['weight'].to(device), head['bias'].to(device)
                with torch.no_grad():
                    for start in range(0, len(raw), 2048):
                        inputs = torch.as_tensor((raw[start:start+2048]-mean)/scale, device=device).float()
                        logits.append(F.linear(inputs, weight, bias).cpu().numpy())
                logits = np.concatenate(logits)
                np.savez_compressed(output / f'{setting}_{name}.npz', logits=logits,
                                    ids=values['ids'], labels=values['labels'])
                results.append({'setting': setting, 'head': name,
                                **classification_metrics(logits, values['labels']),
                                **nec(head['weight'].numpy())})
        print(json.dumps({'directory': str(directory), 'setting': setting, 'heads': len(heads)}), flush=True)
    write_json(output / 'report.json', {'dataset': report['dataset'], 'head_directory': str(directory.resolve()),
                                       'heads_sha256': fingerprint, 'source': str(source.resolve()),
                                       'concept_field': field, 'prefix': prefix, 'refitting': False,
                                       'settings': len(files), 'train_views': report['train_views'],
                                       'evaluation': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--field', default='concepts')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(3)
    evaluate(args.heads, args.source, args.prefix, args.field, args.device)


if __name__ == '__main__':
    main()
