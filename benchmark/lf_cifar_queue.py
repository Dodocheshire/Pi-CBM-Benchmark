"""Run the remaining CIFAR LF fits sequentially without competing GPU jobs."""
from pathlib import Path
import json
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def run_logged(arguments, log_name):
    log_path = ROOT / 'logs' / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w') as log:
        subprocess.run(
            [sys.executable, '-m', *arguments], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )


def main():
    report = ROOT / 'artifacts/matched/lf/cifar100/rn18_imagenet/gaussian_noise_augmented/seed42/report.json'
    while not report.exists():
        time.sleep(15)
    corruption_report = report.parent / 'corruptions/report.json'
    # Completed corruption reports are immutable; avoid needless recomputation
    # when restarting the queue after the four RN18 protocols have completed.
    corruption_complete = False
    if corruption_report.exists():
        completed = json.loads(corruption_report.read_text())
        corruption_complete = len(completed.get('settings', {})) == 12 and len(completed.get('metrics', {})) == 24
    if not corruption_complete:
        run_logged(
            ['benchmark.lf_cifar_corrupt', '--dataset', 'cifar100',
             '--run', str(report.parent), '--device', 'cuda'],
            'lf-cifar100-augmented-corrupt.log',
        )
    for backbone, folder in [('resnet50', 'rn50_imagenet'), ('clip_RN50', 'clip_RN50')]:
        for dataset in ['cifar10', 'cifar100']:
            target = ROOT / 'artifacts/matched/lf' / dataset / folder / 'seed42/report.json'
            if not target.exists():
                print('START', dataset, backbone, flush=True)
                run_logged(
                    ['benchmark.lf_cifar', 'fit', '--dataset', dataset,
                     '--backbone', backbone, '--seed', '42', '--device', 'cuda'],
                    f'lf-{dataset}-{backbone}-fit.log',
                )
            print('DONE', dataset, backbone, flush=True)
            run_logged(
                ['benchmark.report_cifar', '--pini-root', 'artifacts/pini_matched'],
                'report-cifar.log',
            )
    print('COMPLETE FOUR WIDE CIFAR CLEAN FITS', flush=True)


if __name__ == '__main__':
    main()
