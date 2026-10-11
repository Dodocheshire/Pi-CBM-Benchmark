"""Frozen-backbone linear classifier: class labels only, no concept bottleneck."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from .metrics import classification_metrics


def load_split(path):
    path=Path(path)
    if path.suffix=='.npz':
        with np.load(path) as a:return {k:np.array(a[k]) for k in ['features','labels','ids'] if k in a}
    a=torch.load(path,map_location='cpu',weights_only=True)
    return {k:(v.numpy() if isinstance(v,torch.Tensor) else v) for k,v in a.items()}


def fit(train,val,device,seed,lr,decay,epochs=100):
    torch.manual_seed(seed)
    x=torch.as_tensor(train['features'],device=device).float()
    y=torch.as_tensor(train['labels'],device=device).long()
    vx=torch.as_tensor(val['features'],device=device).float()
    vy=torch.as_tensor(val['labels'],device=device).long()
    mean=x.mean(0);std=x.std(0).clamp_min(1e-5)
    x=(x-mean)/std;vx=(vx-mean)/std
    head=nn.Linear(x.shape[1],int(y.max())+1).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=decay)
    best=float('inf');stale=0;history=[];best_state=None
    for epoch in range(epochs):
        order=torch.randperm(len(x),device=device)
        head.train()
        for ids in order.split(2048):
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(head(x[ids]),y[ids]);loss.backward();optimizer.step()
        head.eval()
        with torch.no_grad():
            logits=head(vx);nll=nn.functional.cross_entropy(logits,vy).item()
            accuracy=(logits.argmax(1)==vy).float().mean().item()
        history.append({'epoch':epoch+1,'val_nll':nll,'val_accuracy':accuracy})
        if nll<best-1e-6:
            best=nll;stale=0;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};selected_epoch=epoch+1
        else:stale+=1
        if stale>=15:break
    return {'head':best_state,'mean':mean.cpu(),'std':std.cpu(),'validation_nll':best,
            'selected_epoch':selected_epoch,'history':history,'lr':lr,'weight_decay':decay,'seed':seed}


def run(train_path,val_path,test_path,out,device='cuda',seed=42):
    started=time.time();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    train,val,test=map(load_split,[train_path,val_path,test_path])
    train_ids=np.asarray(train.get('ids',train.get('indices',[])))
    val_ids=np.asarray(val.get('ids',val.get('indices',[])))
    if len(train_ids) and len(val_ids) and set(train_ids)&set(val_ids):raise ValueError('train/val overlap')
    candidates=[fit(train,val,device,seed,lr,decay) for lr in [.001,.01] for decay in [.0001,.01]]
    chosen=min(candidates,key=lambda c:c['validation_nll'])
    chosen.update(method='non_cbm',selection='minimum validation NLL; four fixed LR/decay configurations',
                  train_path=str(Path(train_path).resolve()),validation_path=str(Path(val_path).resolve()),
                  test_path=str(Path(test_path).resolve()),concept_bottleneck=False,nec=None)
    torch.save(chosen,out/'best.pt')
    head=nn.Linear(test['features'].shape[1],chosen['head']['weight'].shape[0]).to(device)
    head.load_state_dict(chosen['head']);head.eval()
    logits=[]
    with torch.no_grad():
        for x in torch.as_tensor(test['features']).float().split(2048):
            logits.append(head(((x-chosen['mean'])/chosen['std']).to(device)).cpu().numpy())
    logits=np.concatenate(logits)
    np.savez_compressed(out/'test_predictions.npz',logits=logits,labels=test['labels'],
                        ids=test.get('ids',test.get('indices',np.arange(len(logits)))))
    result={'method':'non_cbm','test':classification_metrics(logits,test['labels']),
            'validation_nll':chosen['validation_nll'],'selected_epoch':chosen['selected_epoch'],
            'selection':chosen['selection'],'fit_images':len(train['labels']),
            'validation_images':len(val['labels']),'feature_dimension':train['features'].shape[1],
            'head_parameters':sum(v.numel() for v in chosen['head'].values()),'nec':None,
            'seed':seed,'seconds':time.time()-started,
            'candidates':[{'lr':c['lr'],'weight_decay':c['weight_decay'],'val_nll':c['validation_nll']} for c in candidates],
            'input_paths':{k:str(Path(p).resolve()) for k,p in zip(['train','val','test'],[train_path,val_path,test_path])}}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True);p.add_argument('--val',required=True);p.add_argument('--test',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();torch.set_num_threads(4)
    run(a.train,a.val,a.test,a.out,a.device,a.seed)


if __name__=='__main__':main()
