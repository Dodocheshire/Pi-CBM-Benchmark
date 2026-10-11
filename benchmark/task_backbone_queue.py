"""Validate the new task cache, then run at most two CPU training processes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from .task_backbones import ROOT, NAMESPACE, EXPECTED_CHECKPOINT_SHA, file_sha


def validate_cache():
    source = ROOT / 'artifacts/common/rival10' / NAMESPACE
    original = ROOT / 'artifacts/common/rival10/rn18_imagenet'
    teacher = ROOT / 'artifacts/common/rival10/teacher'
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['checkpoint_sha256'] == EXPECTED_CHECKPOINT_SHA
    assert manifest['backbone_name'] == 'resnet18_rival10'
    audit = {'source': str(source), 'checkpoint_sha256': EXPECTED_CHECKPOINT_SHA,
             'manifest_sha256': file_sha(source / 'manifest.json'), 'checks': {},
             'global_spatial_max_abs': {}, 'training_started_before_cache_validation': False}
    for view in ['original', 'changed']:
        receipt = json.loads((source / f'{view}_COMPLETE.json').read_text())
        assert receipt['device'] == 'cpu' and receipt['encoding_batch'] == 32 and receipt['spatial_float32']
        assert receipt['checkpoint_sha256'] == EXPECTED_CHECKPOINT_SHA
        features = np.load(source / f'features_{view}.npy', mmap_mode='r')
        spatial = np.load(source / f'spatial_{view}.npy', mmap_mode='r')
        assert features.shape == (26384, 512) and spatial.shape == (26384, 512, 7, 7)
        max_error = 0.0
        for start in range(0, len(features), 32):
            maps = torch.from_numpy(np.array(spatial[start:start+32]))
            pooled = maps.mean((2, 3)).numpy()
            max_error = max(max_error, float(np.abs(pooled-features[start:start+32]).max()))
            assert np.isfinite(pooled).all()
        assert max_error == 0.0
        audit['global_spatial_max_abs'][view] = max_error
        targets = np.load(teacher / f'{view}_teacher.npy', mmap_mode='r')
        for split in ['train', 'val', 'test']:
            path = source / f'{split}_{view}.npz'
            with np.load(path) as new, np.load(original / f'{split}_{view}.npz') as old:
                for field in ['ids', 'labels', 'targets', 'known', 'training_targets']:
                    assert np.array_equal(new[field], old[field]), f'{split}/{view}/{field}'
                ids = new['ids']
                assert np.array_equal(new['teacher'], targets[ids]), f'{split}/{view}/teacher'
                assert np.array_equal(new['features'], features[ids]), f'{split}/{view}/features'
                assert np.array_equal(new['concept_names'], np.asarray(manifest['concept_names']))
                audit['checks'][split+'/'+view] = {'rows': len(ids), 'fields_exact_source_parity': True,
                    'teacher_exact_same_view_parity': True, 'feature_dim': 512,
                    'ids_int64_bytes_sha256': hashlib.sha256(ids.tobytes()).hexdigest()}
    path = ROOT / 'artifacts/audit/rival_task_cache_verified.json'
    path.write_text(json.dumps(audit, indent=2))
    print(json.dumps({'task_cache_verified': str(path)}), flush=True)
    return source, teacher


def logged(arguments, name):
    with (ROOT / 'logs' / name).open('w') as log:
        subprocess.run([sys.executable, '-m', *arguments], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    torch.set_num_threads(4)
    source = ROOT / 'artifacts/common/rival10' / NAMESPACE
    while not (source / 'manifest.json').exists():
        time.sleep(15)
    source, teacher = validate_cache()
    command = [sys.executable, '-m', 'benchmark.run_backbone_noncbm', '--datasets', 'rival10',
               '--backbones', NAMESPACE, '--protocols', 'original', 'paired', '--seeds', '42', '43', '44', '--device', 'cpu']
    terminal = {'device': 'cpu', 'max_new_concurrent_training_processes': 2, 'runs': []}
    with (ROOT / 'logs/rival-task-noncbm.log').open('w') as log:
        controls = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        terminal['noncbm_pid'] = controls.pid
        try:
            for paired in [False, True]:
                root = ROOT / 'artifacts/matched/lf/rival10' / NAMESPACE
                output = root / 'paired_augmentation/seed42' if paired else root / 'seed42'
                tag = 'paired' if paired else 'original'
                if not (output / 'report.json').exists():
                    logged(['benchmark.lf_matched', '--dataset', 'rival10', '--source', str(source),
                            '--teacher', str(teacher), '--output', str(output), '--device', 'cpu',
                            '--backbone', 'resnet18', '--epochs', '20', '--projection-steps', '1000',
                            '--saga-steps', '1000', '--seed', '42', '--draws', '1',
                            '--train-views', 'original,changed' if paired else 'original'],
                           f'rival-task-lf-{tag}.log')
                logged(['benchmark.lf_single_draw', '--run', str(output), '--source', str(source),
                        '--teacher', str(teacher), '--device', 'cpu'], f'rival-task-single-draw-{tag}.log')
                terminal['runs'].append({'protocol': tag, 'run': str(output), 'fit_and_single_draw_exit_code': 0})
                print(json.dumps(terminal['runs'][-1]), flush=True)
        finally:
            # A failed LF run must not terminate independently authorized controls.
            terminal['noncbm_exit_code'] = controls.wait()
            (ROOT / 'artifacts/audit/rival_task_training_terminal.json').write_text(json.dumps(terminal, indent=2))
        if terminal['noncbm_exit_code'] != 0:
            raise RuntimeError('Task nonCBM controls failed; inspect their independent log')
    print(json.dumps({'task_training_complete': True, **terminal}), flush=True)


if __name__ == '__main__':
    main()
