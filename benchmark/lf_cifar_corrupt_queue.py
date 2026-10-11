"""Sequential wide-backbone CIFAR-C evaluation after each full fit completes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def complete(path):
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    return len(report.get('settings', {})) == 12 and len(report.get('metrics', {})) == 24


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    # The independently launched first job owns this directory until completion.
    first = ROOT / 'artifacts/matched/lf/cifar10/rn50_imagenet/seed42/corruptions/report.json'
    while not complete(first):
        time.sleep(15)
    for backbone, folder, dataset in [
        ('resnet50', 'rn50_imagenet', 'cifar100'),
        ('clip_RN50', 'clip_RN50', 'cifar10'),
        ('clip_RN50', 'clip_RN50', 'cifar100'),
    ]:
        directory = ROOT / 'artifacts/matched/lf' / dataset / folder / 'seed42'
        while not (directory / 'report.json').exists():
            time.sleep(15)
        if not complete(directory / 'corruptions/report.json'):
            print('START', dataset, backbone, flush=True)
            with (ROOT / 'logs' / f'lf-{dataset}-{backbone}-corrupt.log').open('w') as log:
                subprocess.run(
                    [sys.executable, '-m', 'benchmark.lf_cifar_corrupt', '--dataset', dataset,
                     '--backbone', backbone, '--batch', str(args.batch), '--device', args.device],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                )
        print('DONE', dataset, backbone, flush=True)
        with (ROOT / 'logs/report-cifar-wide-corruption.log').open('w') as log:
            subprocess.run(
                [sys.executable, '-m', 'benchmark.report_cifar', '--pini-root', 'artifacts/pini_matched'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
    print('COMPLETE FOUR WIDE CIFAR-C EVALUATIONS', flush=True)


if __name__ == '__main__':
    main()
