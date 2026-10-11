"""Wait for sustained GPU headroom before the authorized CUB train-only fit."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_MIB = 10 * 1024


def main():
    command = [sys.executable, '-m', 'benchmark.cub_task_backbone', 'train', '--device', 'cuda']
    evidence = ROOT / 'artifacts/audit/cub_task_gpu_gate.json'
    evidence.parent.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'artifacts/task_backbones/cub/rn18_task_pretrained_cub_train_val_selected/seed42'
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('CUB run already exists; refuse implicit restart or repeated test')
    samples = []
    while True:
        free = int(subprocess.check_output(
            ['nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True,
        ).strip())
        samples.append({'unix_time': time.time(), 'free_MiB': free})
        samples = samples[-3:]
        gate = {'command': command, 'minimum_free_MiB': MIN_FREE_MIB,
                'required_consecutive_samples': 3, 'sample_interval_seconds': 30,
                'samples': samples, 'state': 'waiting', 'training_started': False}
        evidence.write_text(json.dumps(gate, indent=2))
        if len(samples) == 3 and all(x['free_MiB'] >= MIN_FREE_MIB for x in samples):
            break
        time.sleep(30)
    # Both the command and measured free memory are logged before GPU allocation.
    gate.update(state='starting')
    evidence.write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate), flush=True)
    with (ROOT / 'logs/cub-task-backbone-train.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        gate.update(state='running', training_started=True, training_pid=process.pid)
        evidence.write_text(json.dumps(gate, indent=2))
        print(json.dumps({'training_pid': process.pid, 'command': command, 'gate_samples': samples}), flush=True)
        code = process.wait()
    gate.update(state='complete' if code == 0 else 'failed', exit_code=code)
    evidence.write_text(json.dumps(gate, indent=2))
    terminal = ROOT / 'artifacts/audit/cub_task_gpu_terminal.json'
    terminal.write_text(json.dumps(gate, indent=2))
    print(json.dumps({'terminal': str(terminal), 'exit_code': code}), flush=True)
    if code:
        raise RuntimeError(f'CUB task training failed with exit code {code}; prior datasets/checkpoints untouched')


if __name__ == '__main__':
    main()
