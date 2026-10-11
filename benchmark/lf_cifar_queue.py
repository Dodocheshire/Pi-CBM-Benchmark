from pathlib import Path
import subprocess,sys,time
root=Path.cwd();report=root/'artifacts/matched/lf/cifar100/rn18_imagenet/gaussian_noise_augmented/seed42/report.json'
while not report.exists():time.sleep(15)
with open('logs/lf-cifar100-augmented-corrupt.log','w') as log:
 subprocess.run([sys.executable,'-m','benchmark.lf_cifar_corrupt','--dataset','cifar100','--run',str(report.parent),'--device','cuda'],stdout=log,stderr=subprocess.STDOUT,check=True)
for bone in ['resnet50','clip_RN50']:
 for dataset in ['cifar10','cifar100']:
  folder={'resnet50':'rn50_imagenet','clip_RN50':'clip_RN50'}[bone]
  target=root/'artifacts/matched/lf'/dataset/folder/'seed42/report.json'
  if not target.exists():
   print('START',dataset,bone,flush=True)
   with open(f'logs/lf-{dataset}-{bone}-fit.log','w') as log:
    subprocess.run([sys.executable,'-m','benchmark.lf_cifar','fit','--dataset',dataset,'--backbone',bone,'--seed','42','--device','cuda'],stdout=log,stderr=subprocess.STDOUT,check=True)
  print('DONE',dataset,bone,flush=True)
  with open('logs/report-cifar.log','w') as log:subprocess.run([sys.executable,'-m','benchmark.report_cifar','--pini-root','artifacts/pini_matched'],stdout=log,stderr=subprocess.STDOUT,check=True)
print('COMPLETE FOUR WIDE CIFAR CLEAN FITS',flush=True)
