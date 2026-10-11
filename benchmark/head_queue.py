"""Run completed concept exports through a bounded common head-fit queue."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--exclude',nargs='*',default=[])
    p.add_argument('--include',nargs='*',default=None,
                   help='Only run these output paths, for additions after a queue was started')
    p.add_argument('--inference-draws',type=int,choices=[1,8],default=1)
    args=p.parse_args();jobs=[]
    for file in sorted(Path('artifacts/matched/lf').rglob('report.json')):
        r=json.loads(file.read_text())
        if not {'feature_provenance','train_views','training_seed'}.issubset(r):continue
        bone=r['feature_provenance'].get('backbone_name','resnet18')
        proto=('paired' if r['train_views']==['original','changed'] else
               'corruption_augmented' if len(r['train_views'])>1 else 'original')
        test_views=['original'] if r['dataset'] in ['cifar10','cifar100','imagenet_200_per_class','places365'] else ['original','changed']
        for arm in ['lfcbm','lfpi_global']:
            source=file.parent;head_arm=arm
            if arm=='lfpi_global' and args.inference_draws==1:
                source=source/'single_draw';head_arm=arm+'_1draw'
                if not (source/'metadata.json').exists():continue
                if not all((source/f'{arm}_{s}_{v}.npz').exists() for s in ['train','val','test'] for v in set(r['train_views']+test_views)):continue
            out=Path('artifacts/heads')/r['dataset']/bone/proto/f"seed{r['training_seed']}"/head_arm
            jobs.append({'source':str(source),'prefix':arm+'_','field':'concepts',
                'dataset':r['dataset'],'out':str(out),'train_views':','.join(r['train_views']), 'seed':r['training_seed'],
                'test_views':','.join(test_views)})
    for file in sorted(Path('artifacts/pini_matched').rglob('exports/*/results.json')):
        dataset=file.parents[2].name;arm=file.parent.name
        bone='resnet18' if file.parents[3].name=='pini_matched' else file.parents[3].name
        if bone=='clip_rn50':bone='clip_RN50'
        r=json.loads(file.read_text())
        if dataset in ['cifar10','cifar100']:
            train_views='original,gaussian_noise_s1'
            test_views='original,gaussian_noise_s1'
            proto='corruption_augmented'
        else:
            train_views='original,changed';test_views='original,changed';proto='paired'
        required=[file.parent/f'{split}_{view}.npz' for split in ['train','val','test']
                  for view in train_views.split(',')]
        if not all(path.exists() for path in required):continue
        seed=r.get('training_seed',42)
        out=Path('artifacts/heads')/dataset/bone/proto/f'seed{seed}'/('pini_'+arm)
        jobs.append({'source':str(file.parent),'prefix':'','field':'concept_logits',
                     'dataset':dataset,'out':str(out),'train_views':train_views,'test_views':test_views,'seed':seed})
    jobs=[j for j in jobs if not (Path(j['out'])/'report.json').exists() and j['out'] not in args.exclude]
    if args.include is not None:
        jobs=[j for j in jobs if j['out'] in args.include]
    Path('logs').mkdir(exist_ok=True)
    def run(j):
        out=Path(j['out']);out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-m','benchmark.head_suite','--source',j['source'],'--prefix',j['prefix'],
             '--field',j['field'],'--dataset',j['dataset'],'--out',j['out'],'--train-views',j['train_views'],
             '--seed',str(j['seed'])]
        cmd += ['--test-views',j.get('test_views','original,changed')]
        log=out/'run.log'
        with log.open('w') as stream:
            process=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
            (out/'running.json').write_text(json.dumps({'pid':process.pid,'command':cmd},indent=2))
            code=process.wait()
        (out/'terminal.json').write_text(json.dumps({'pid':process.pid,'exit_code':code,'log':str(log)},indent=2))
        print(json.dumps({'output':str(out),'exit_code':code}),flush=True)
        return code
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes=list(pool.map(run,jobs))
    if any(codes):raise SystemExit(1)


if __name__=='__main__':main()
