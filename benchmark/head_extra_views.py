"""Evaluate additional views with existing heads and unchanged train normalization."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .metrics import classification_metrics, nec


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evaluate(directory, views):
    path = directory / 'report.json'
    original = path.read_bytes()
    report = json.loads(original)
    assert report['selection_split'] == 'val' and not report['test_selection']
    fingerprint = sha(directory / 'heads.pt')
    models = torch.load(directory / 'heads.pt', map_location='cpu', weights_only=True)
    norm = np.load(directory / 'normalization.npz')
    with np.load(directory / 'dense_original.npz') as reference:
        ids, labels = reference['ids'], reference['labels']
    heads = {'best_val': models['sparse']['best'], **models['sparse']['budgets'],
             'dense': models['dense']['head']}
    added, sources = [], {}
    for view in dict.fromkeys(['original', *views]):
        source = Path(report['source']) / f"{report['prefix']}test_{view}.npz"
        with np.load(source) as values:
            assert np.array_equal(values['ids'], ids) and np.array_equal(values['labels'], labels)
            raw = values[report['concept_field']].astype('float32')
        assert np.isfinite(raw).all() and raw.shape[1] == len(norm['mean'])
        sources[view] = {'path': str(source), 'sha256': sha(source)}
        for name, head in heads.items():
            if name == 'dense':
                mean, scale = models['dense']['mean'], models['dense']['std']
                inputs = (torch.from_numpy(raw) - mean) / scale
            else:
                inputs = torch.from_numpy((raw - norm['mean']) / norm['scale'])
            with torch.no_grad():
                logits = F.linear(inputs, head['weight'], head['bias']).numpy()
            target = directory / f'{name}_{view}.npz'
            row = {'head': name, 'view': view, **classification_metrics(logits, labels),
                   **nec(head['weight'].numpy())}
            if view in report['test_views']:
                with np.load(target) as previous:
                    assert np.array_equal(previous['ids'], ids) and np.array_equal(previous['labels'], labels)
                    assert np.allclose(previous['logits'], logits, atol=1e-5, rtol=1e-5)
                    assert np.array_equal(previous['logits'].argmax(1), logits.argmax(1))
                prior = next(x for x in report['test_evaluations'] if x['head'] == name and x['view'] == view)
                assert prior['accuracy'] == row['accuracy'] and prior['nec'] == row['nec']
            else:
                np.savez_compressed(target, logits=logits, labels=labels, ids=ids)
                added.append(row)
    assert sha(directory / 'heads.pt') == fingerprint and path.read_bytes() == original
    if added:
        backup = directory / 'report_before_extra_views.json'
        if not backup.exists():
            backup.write_bytes(original)
        report['test_views'] = list(dict.fromkeys(report['test_views'] + views))
        report['test_evaluations'].extend(added)
        report['inference_extension'] = {'heads_sha256': fingerprint, 'refitting': False,
            'original_report_sha256': hashlib.sha256(original).hexdigest(),
            'source_exports': sources, 'producer_sha256': sha(__file__)}
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(path)
    return {'directory': str(directory), 'heads_sha256': fingerprint,
            'refitting': False, 'original_predictions_reconstructed': True,
            'added_evaluations': len(added), 'test_views': report['test_views']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--views', default='gaussian_noise_s1')
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(evaluate(args.heads, args.views.split(','))), flush=True)


if __name__ == '__main__':
    main()
