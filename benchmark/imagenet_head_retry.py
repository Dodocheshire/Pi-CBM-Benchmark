"""Retry the terminal ImageNet head OOM only after sustained GPU headroom."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--efficient-head', action='store_true')
    parser.add_argument('--wait-for-cub', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'artifacts/heads/imagenet_200_per_class/resnet18/original/seed42/lfpi_global_1draw'
    source = ROOT / 'artifacts/matched/lf/imagenet_200_per_class/rn18_imagenet/seed42/single_draw'
    error_log = ROOT / 'logs/head-imagenet-lfpi.log'
    if (out / 'report.json').exists():
        raise RuntimeError('Head results already exist; no retry required')
    if 'torch.OutOfMemoryError: CUDA out of memory' not in error_log.read_text():
        raise RuntimeError('Retry requires the observed terminal OOM evidence')
    # Verify the previous actual process is gone; elapsed time is not a reason
    # to restart a live fitting process.
    previous = 2545230
    if Path(f'/proc/{previous}/cmdline').exists():
        raise RuntimeError('Prior fitting process still exists; do not restart')
    with error_log.open('rb') as stream:
        failed_log_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    command = [sys.executable, '-m', 'benchmark.head_suite', '--source', str(source),
               '--prefix', 'lfpi_global_', '--dataset', 'imagenet_200_per_class',
               '--out', str(out), '--test-views', 'original', '--device', 'cuda']
    if args.efficient_head:
        audit = json.loads((ROOT/'artifacts/audit/optional_head_regression.json').read_text())
        assert audit['saga_regression']['same_best_validation_index']
        assert audit['dense_regression']['same_selected_lr_decay']
        for file, expected in audit['source_sha256'].items():
            assert hashlib.sha256((ROOT/'benchmark'/file).read_bytes()).hexdigest() == expected
        command += ['--saga-gradient-kernel', 'matmul', '--streaming-dense']
    minimum_free = 12288 if args.efficient_head else 22528
    path = ROOT / 'artifacts/audit/imagenet_lfpi_head_retry.json'
    record = {'state': 'waiting_gpu', 'supervisor_pid': os.getpid(),
              'previous_terminal_pid': previous, 'previous_exit_code': 1,
              'reason': 'confirmed terminal CUDA OOM; failed output directory contained no results',
              'failed_log': str(error_log), 'failed_log_sha256': failed_log_sha,
              'command': command, 'minimum_free_MiB': minimum_free,
              'efficient_head': args.efficient_head, 'wait_for_cub': args.wait_for_cub,
              'required_consecutive_samples': 3, 'interval_seconds': 30,
              'steps_seed_solver_grid_and_inference_unchanged': True,
              'existing_lfcbm_fit_untouched': True, 'samples': []}
    def save():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2)); temp.replace(path)
    save()
    while True:
        if args.wait_for_cub:
            terminal = ROOT/'artifacts/pini_matched/resnet18_cub_trainonly/cub/task_training_terminal.json'
            cub = json.loads(terminal.read_text()) if terminal.exists() else {}
            if cub.get('state') != 'succeeded' or cub.get('returncode') != 0 or cub.get('missing_exports') != []:
                record.update(state='waiting_for_cub', cub_state=cub.get('state', 'not_started'))
                save(); time.sleep(30); continue
            record.update(state='waiting_gpu', cub_state='succeeded')
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                process_args = (proc / 'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if b'benchmark.head_suite' in process_args and b'--out' in process_args:
                target = process_args[process_args.index(b'--out')+1].decode()
                if (ROOT / target).resolve() == out:
                    raise RuntimeError(f'Another process owns the retry head: {proc.name}')
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                                           '--format=csv,noheader,nounits'], text=True).strip())
        record['samples'].append({'unix_time': time.time(), 'free_MiB': free})
        record['samples'] = record['samples'][-3:]
        save()
        if len(record['samples']) == 3 and all(s['free_MiB'] >= minimum_free for s in record['samples']):
            break
        time.sleep(30)
    with (ROOT / 'logs/head-imagenet-lfpi-retry.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(state='running', child_pid=child.pid); save()
        code = child.wait()
    complete = (out/'report.json').is_file() and (out/'heads.pt').is_file()
    record.update(state='complete' if code == 0 and complete else 'failed', exit_code=code,
                  completed_head_files=complete); save()
    if code or not complete:
        raise RuntimeError('Retry failed; inspect its separate log and terminal evidence')


if __name__ == '__main__':
    main()
