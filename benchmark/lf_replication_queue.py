"""Run the remaining registered native-data LF seeds with bounded CPU workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def jobs():
    spec = json.loads((ROOT / 'configs/lf_matched_runs.json').read_text())
    result = []
    for dataset in spec['datasets']:
        for bone in spec['backbones']:
            folder = spec['backbone_cache_names'][bone]
            source = ROOT / 'artifacts/common' / dataset / folder
            teacher_dataset = 'cub' if dataset == 'waterbirds' else dataset
            teacher = ROOT / 'artifacts/common' / teacher_dataset / 'teacher'
            for paired in [False, True]:
                parent = ROOT / 'artifacts/matched/lf' / dataset / folder
                if paired:
                    parent /= 'paired_augmentation'
                reference = parent / 'seed42/report.json'
                original = json.loads(reference.read_text())
                views = ['original', 'changed'] if paired else ['original']
                assert original['training_seed'] == 42 and original['train_views'] == views
                assert original['test_selection'] is False
                assert original['teacher_provenance'] == json.loads((teacher / 'metadata.json').read_text())
                # Original ImageNet RN18 manifests predate backbone_name;
                # only that registered folder may use the legacy default.
                registered = original['feature_provenance'].get('backbone_name')
                assert registered == bone or (registered is None and bone == 'resnet18' and folder == 'rn18_imagenet')
                for seed in spec['training_seeds']:
                    if seed == 42:
                        continue
                    result.append({'dataset': dataset, 'backbone': bone, 'seed': seed,
                                   'source': str(source), 'teacher': str(teacher),
                                   'output': str(parent / f'seed{seed}'), 'train_views': views,
                                   'reference': str(reference), 'reference_sha256': digest(reference),
                                   'source_manifest_sha256': digest(source / 'manifest.json'),
                                   'teacher_metadata_sha256': digest(teacher / 'metadata.json'),
                                   'core_training': spec['core_training']})
    return result


def validate_completed(job):
    output = Path(job['output'])
    if not (output / 'report.json').exists():
        return False
    report = json.loads((output / 'report.json').read_text())
    assert report['dataset'] == job['dataset'] and report['training_seed'] == job['seed']
    assert report['train_views'] == job['train_views'] and report['test_selection'] is False
    registered = report['feature_provenance'].get('backbone_name')
    assert registered == job['backbone'] or (registered is None and job['backbone'] == 'resnet18' and Path(job['source']).name == 'rn18_imagenet')
    assert report['feature_provenance'] == json.loads((Path(job['source']) / 'manifest.json').read_text())
    for arm in ['lfcbm', 'lfpi_global']:
        for split in ['train', 'val', 'test']:
            views = ['original', 'changed'] if split == 'test' or len(job['train_views']) == 2 else ['original']
            for view in views:
                assert (output / f'{arm}_{split}_{view}.npz').is_file()
    return True


def execute(job):
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    # Hold this lock through fit and export; a second launcher cannot overwrite
    # an in-flight replicate or create simultaneous fits of the same seed.
    with (output / 'replication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = {**job, 'supervisor_pid': os.getpid(), 'device': 'cpu',
                   'state': 'checking', 'core_logic_changed': False}
        status = output / 'replication_status.json'
        status.write_text(json.dumps(receipt, indent=2))
        assert digest(Path(job['source']) / 'manifest.json') == job['source_manifest_sha256']
        assert digest(Path(job['teacher']) / 'metadata.json') == job['teacher_metadata_sha256']
        if not validate_completed(job):
            if (output / 'models.pt').exists() or (output / 'config.yaml').exists():
                raise RuntimeError(f'Incomplete previous fit requires explicit recovery: {output}')
            core = job['core_training']
            command = [sys.executable, '-m', 'benchmark.lf_matched', '--dataset', job['dataset'],
                       '--source', job['source'], '--teacher', job['teacher'], '--output', str(output),
                       '--device', 'cpu', '--backbone', job['backbone'], '--seed', str(job['seed']),
                       '--epochs', str(core['epochs']), '--projection-steps', str(core['projection_steps']),
                       '--saga-steps', str(core['saga_steps']), '--draws', '1',
                       '--train-views', ','.join(job['train_views'])]
            with (output / 'replication_fit.log').open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                receipt.update(state='fitting', child_pid=process.pid, command=command)
                status.write_text(json.dumps(receipt, indent=2))
                code = process.wait()
            receipt.update(fit_exit_code=code)
            if code:
                receipt.update(state='failed')
                status.write_text(json.dumps(receipt, indent=2))
                return receipt
            assert validate_completed(job)
        single = output / 'single_draw/metadata.json'
        if not single.exists():
            command = [sys.executable, '-m', 'benchmark.lf_single_draw', '--run', str(output),
                       '--source', job['source'], '--teacher', job['teacher'], '--device', 'cpu']
            with (output / 'replication_export.log').open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                receipt.update(state='exporting', child_pid=process.pid, command=command)
                status.write_text(json.dumps(receipt, indent=2))
                code = process.wait()
            if code:
                receipt.update(state='failed', export_exit_code=code)
                status.write_text(json.dumps(receipt, indent=2))
                return receipt
        metadata = json.loads(single.read_text())
        assert metadata['inference_draws'] == 1 and not metadata['retrained']
        assert metadata['checkpoint_sha256'] == digest(output / 'models.pt')
        for split in ['train', 'val', 'test']:
            for view in ['original', 'changed']:
                assert (single.parent / f'lfpi_global_{split}_{view}.npz').is_file()
        receipt.update(state='complete', fit_exit_code=0, export_exit_code=0,
                       checkpoint_sha256=metadata['checkpoint_sha256'])
        status.write_text(json.dumps(receipt, indent=2))
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    pending = jobs()
    directory = ROOT / 'artifacts/audit'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'lf_replication_plan.json').write_text(json.dumps({
        'supervisor_pid': os.getpid(), 'workers': args.workers, 'device': 'cpu',
        'seed42_reference_unchanged': True, 'jobs': pending}, indent=2))
    print(json.dumps({'planned_jobs': len(pending), 'workers': args.workers}), flush=True)
    if args.plan_only:
        return
    # Report every terminal job immediately; failures leave other seeds running.
    failures = []
    def run(job):
        try:
            result = execute(job)
        except Exception as error:
            result = {**job, 'state': 'failed', 'error': str(error)}
        print(json.dumps(result), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run, pending):
            if result['state'] != 'complete':
                failures.append(result)
    (directory / 'lf_replication_terminal.json').write_text(json.dumps({
        'supervisor_pid': os.getpid(), 'jobs': len(pending), 'failed': failures,
        'state': 'failed' if failures else 'complete'}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
