"""Serial full-concept native OOD exports after sustained GPU headroom."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .report_utils import write_json

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ['original_0'] + [f'{kind}_{severity}' for kind in
                           ['gaussian_noise', 'gaussian_blur', 'contrast', 'jpeg'] for severity in (1, 3, 5)]


def verify(dataset, output):
    protocol = json.loads((output / 'protocol.json').read_text())
    assert protocol['export_full_concepts'] and protocol['batch_size'] == 8
    assert protocol['pini_draws'] == 1 and protocol['inference_noise_seed'] == 4242
    reference = ROOT / 'artifacts/ood' / dataset / 'resnet18/paired/seed42/protocol.json'
    previous = json.loads(reference.read_text())
    assert protocol['sample_ids'] == previous['sample_ids']
    source = ROOT / 'artifacts/common' / dataset / 'rn18_imagenet/test_original.npz'
    with np.load(source) as native:
        positions = {int(identifier): index for index, identifier in enumerate(native['ids'])}
        rows = [positions[int(identifier)] for identifier in protocol['sample_ids']]
        for setting in SETTINGS:
            assert (output / f'{setting}_COMPLETE.json').exists()
            for arm in ['mean', 'noise']:
                with np.load(output / f'{setting}_pini_{arm}.npz') as saved:
                    assert np.array_equal(saved['ids'], np.asarray(protocol['sample_ids']))
                    for field, target in [('labels', 'labels'), ('targets_full', 'targets'), ('known_full', 'known')]:
                        assert np.array_equal(saved[field], native[target][rows])
                    assert saved['concept_logits'].shape == saved['targets_full'].shape
                    assert np.isfinite(saved['concept_logits']).all() and np.isfinite(saved['logits']).all()
                    assert np.array_equal(saved['concepts'], saved['concepts_full'][:, protocol['concept_indices']])
                    assert np.array_equal(saved['concept_indices_full'], np.arange(protocol['full_concept_count']))
    return {'images': len(rows), 'settings': len(SETTINGS), 'full_concepts': protocol['full_concept_count'],
            'native_fields_exact': True, 'selected_probabilities_exact': True}


def main():
    audit = ROOT / 'artifacts/audit'; audit.mkdir(parents=True, exist_ok=True)
    with (audit / 'ood_full_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = json.loads((audit / 'pini_ood_full_export_cpu_audit.json').read_text())
        assert expected['stochastic_and_model_call_AST_exact'] and expected['changed_weight_resume_rejected']
        producer = ROOT / 'benchmark/ood.py'
        assert hashlib.sha256(producer.read_bytes()).hexdigest() == expected['producer_sha256']
        record = {'supervisor_pid': os.getpid(), 'state': 'waiting_gpu', 'minimum_free_MiB': 16384,
                  'gate_samples': 3, 'gate_interval_seconds': 30, 'completed': [], 'batch_size': 8}
        status = audit / 'ood_full_queue_status.json'
        for dataset in ['rival10', 'cub', 'waterbirds']:
            output = ROOT / 'artifacts/ood_fullraw' / dataset / 'resnet18/paired/seed42'
            record.update(dataset=dataset, output=str(output), state='waiting_gpu', samples=[])
            if (output / 'protocol.json').exists():
                record['completed'].append({'dataset': dataset, **verify(dataset, output)})
                write_json(status, record); continue
            while True:
                free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                                   '--format=csv,noheader,nounits'], text=True).strip())
                record['samples'] = (record['samples'] + [{'time': time.time(), 'free_MiB': free}])[-3:]
                write_json(status, record)
                if len(record['samples']) == 3 and all(sample['free_MiB'] >= 16384 for sample in record['samples']):
                    break
                time.sleep(30)
            assert hashlib.sha256(producer.read_bytes()).hexdigest() == expected['producer_sha256']
            command = [sys.executable, '-u', '-m', 'benchmark.ood', '--dataset', dataset, '--images', '256',
                       '--batch', '8', '--out', str(output), '--export-full-concepts', '--corruptions-only']
            with (ROOT / 'logs' / f'ood-fullraw-{dataset}.log').open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                record.update(state='running', child_pid=child.pid, command=command); write_json(status, record)
                code = child.wait()
            record.update(exit_code=code)
            if code:
                record.update(state='failed'); write_json(status, record)
                raise RuntimeError(f'{dataset} full OOD export exited {code}')
            record['completed'].append({'dataset': dataset, **verify(dataset, output)})
            record.pop('child_pid', None); write_json(status, record)
        record.update(state='complete'); write_json(status, record)


if __name__ == '__main__':
    main()
