"""Fit remaining native-data replicate heads as their upstream exports finish."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .lf_replication_queue import ROOT, jobs, validate_completed, digest
from .report_utils import write_json


def reserved_outputs():
    reserved = set()
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            arguments = path.read_bytes().decode().split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if 'benchmark.head_queue' in arguments and '--include' in arguments:
            for argument in arguments[arguments.index('--include') + 1:]:
                if not argument or argument.startswith('--'):
                    break
                reserved.add(str((ROOT / argument).resolve()))
        if 'benchmark.head_suite' in arguments and '--out' in arguments:
            reserved.add(str((ROOT / arguments[arguments.index('--out') + 1]).resolve()))
    return reserved


def ready(job):
    parent = Path(job['upstream']['output'])
    status = parent / 'replication_status.json'
    if not status.exists() or json.loads(status.read_text()).get('state') != 'complete':
        return False
    if not validate_completed(job['upstream']):
        return False
    metadata = json.loads((parent / 'single_draw/metadata.json').read_text())
    assert metadata['inference_draws'] == 1 and not metadata['retrained']
    assert metadata['checkpoint_sha256'] == digest(parent / 'models.pt')
    return True


def run(job):
    out = Path(job['out']); out.mkdir(parents=True, exist_ok=True)
    if str(out) in reserved_outputs():
        raise RuntimeError(f'Head output acquired another live owner: {out}')
    upstream = job['upstream']
    source = Path(upstream['output'])
    prefix = 'lfcbm_'
    if job['arm'] == 'lfpi_global_1draw':
        source /= 'single_draw'; prefix = 'lfpi_global_'
    command = [sys.executable, '-m', 'benchmark.head_suite', '--source', str(source),
               '--prefix', prefix, '--dataset', upstream['dataset'], '--out', str(out),
               '--seed', str(upstream['seed']), '--train-views', ','.join(upstream['train_views']),
               '--test-views', 'original,changed', '--device', 'cpu']
    with (out / 'replicate_head.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write_json(out / 'running.json', {'pid': child.pid, 'command': command})
        code = child.wait()
    write_json(out / 'terminal.json', {'pid': child.pid, 'exit_code': code})
    if code or not (out / 'report.json').exists():
        raise RuntimeError(f'Head fit failed: {out}, exit={code}')
    print(json.dumps({'output': str(out), 'exit_code': code}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    audit = ROOT / 'artifacts/audit'; audit.mkdir(parents=True, exist_ok=True)
    with (audit / 'native_replication_heads.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reserved = reserved_outputs()
        pending = []
        for upstream in jobs():
            protocol = 'paired' if len(upstream['train_views']) == 2 else 'original'
            for arm in ['lfcbm', 'lfpi_global_1draw']:
                out = ROOT / 'artifacts/heads' / upstream['dataset'] / upstream['backbone'] / protocol / f'seed{upstream["seed"]}' / arm
                if str(out) in reserved or (out / 'report.json').exists():
                    continue
                pending.append({'upstream': upstream, 'arm': arm, 'out': str(out)})
        write_json(audit / 'native_replication_heads_plan.json', {'supervisor_pid': os.getpid(),
                   'workers': args.workers, 'device': 'cpu', 'reserved_outputs': sorted(reserved), 'jobs': pending})
        print(json.dumps({'planned_heads': len(pending), 'workers': args.workers}), flush=True)
        if args.plan_only:
            return
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            active = {}
            while pending or active:
                for job in pending[:]:
                    if len(active) >= args.workers:
                        break
                    if ready(job):
                        active[pool.submit(run, job)] = job
                        pending.remove(job)
                if active:
                    finished, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                    for future in finished:
                        future.result(); del active[future]
                else:
                    time.sleep(30)
        write_json(audit / 'native_replication_heads_terminal.json', {'state': 'complete', 'supervisor_pid': os.getpid()})


if __name__ == '__main__':
    main()
