"""Resume the unchanged CUB GPU gate after formal RIVAL task PiNI success."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PINI = ROOT / 'artifacts/pini_matched/resnet18_rival10/rival10'
COMMON = ROOT / 'artifacts/common/rival10/rn18_task_pretrained_train_val_selected'
AUDIT = ROOT / 'artifacts/audit/cub_task_after_pini_scheduler.json'


def complete_exports():
    terminal = PINI / 'task_training_terminal.json'
    if not terminal.exists():
        return None
    status = json.loads(terminal.read_text())
    if (status.get('state') != 'succeeded' or status.get('returncode') != 0
            or status.get('missing_exports') != []):
        return None
    result = PINI / 'matched_results.json'
    if not result.exists() or not {'mean', 'noise'} <= json.loads(result.read_text()).keys():
        return None
    checked = []
    for arm in ('mean', 'noise'):
        out = PINI / 'exports' / arm
        report = out / 'results.json'
        if not report.exists():
            return None
        metrics = json.loads(report.read_text())
        for split in ('train', 'val', 'test'):
            for view in ('original', 'changed'):
                key = f'{split}_{view}'
                path = out / f'{key}.npz'
                if key not in metrics or not path.exists():
                    return None
                with np.load(path) as z, np.load(COMMON / f'{key}.npz') as source:
                    for field in ('ids', 'labels', 'targets', 'known'):
                        assert np.array_equal(z[field], source[field]), (path, field)
                    assert z['concepts'].shape == source['targets'].shape
                    assert np.isfinite(z['concepts']).all() and np.isfinite(z['logits']).all()
                checked.append(str(path))
    return {'pini_training_terminal': str(terminal), 'terminal_status': status,
            'formal_exports_verified': checked}


def main():
    command = [sys.executable, '-m', 'benchmark.cub_task_gpu_queue']
    record = {'scheduler_pid': os.getpid(), 'state': 'waiting_for_rival_task_pini',
              'cub_command': command, 'minimum_free_MiB': 10240,
              'required_consecutive_samples': 3,
              'checkpoint_or_training_formula_changes': False}
    while True:
        evidence = complete_exports()
        record.update(last_checked_unix_time=time.time())
        AUDIT.write_text(json.dumps(record, indent=2))
        if evidence is not None:
            break
        time.sleep(30)
    # The previous waiting watcher was stopped and audited before this scheduler.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        assert b'benchmark.cub_task_gpu_queue' not in args, f'existing CUB gate pid {proc.name}'
    record.update(evidence, state='restarting_unchanged_cub_gpu_gate')
    AUDIT.write_text(json.dumps(record, indent=2))
    with (ROOT / 'logs/cub-task-gpu-queue-resumed.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(state='cub_gpu_gate_running', cub_gate_pid=child.pid)
        AUDIT.write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
        code = child.wait()
    record.update(state='complete' if code == 0 else 'failed', cub_gate_exit_code=code)
    AUDIT.write_text(json.dumps(record, indent=2))
    if code:
        raise RuntimeError(f'CUB gate/train exited {code}; inspect its terminal audit')


if __name__ == '__main__':
    main()
