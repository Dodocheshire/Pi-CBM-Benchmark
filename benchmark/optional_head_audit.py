"""Isolated opt-in head regressions; never restarts a benchmark supervisor."""
import inspect
import json
import zipfile

import numpy as np
import torch

from .lf import fit_nec_path, upstream
from .noncbm import fit
from .saga_outer_audit import difference
from .task_backbones import ROOT, file_sha


def balanced(labels, count):
    return np.concatenate([np.flatnonzero(labels == c)[:count] for c in range(10)])


def actual_imagenet_batch():
    path = ROOT/'artifacts/matched/lf/imagenet_200_per_class/rn18_imagenet/seed42/single_draw/lfpi_global_train_original.npz'
    with zipfile.ZipFile(path) as archive, archive.open('concepts.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(stream)
                                if version == (1, 0) else np.lib.format.read_array_header_2_0(stream))
        assert not fortran
        offset = stream.tell()
        rows = np.linspace(0, shape[0]-1, 16, dtype=np.int64)
        samples = []
        for index in rows:
            stream.seek(offset+int(index)*shape[1]*dtype.itemsize)
            samples.append(np.frombuffer(stream.read(shape[1]*dtype.itemsize), dtype=dtype).copy())
    with np.load(path) as z:
        labels = z['labels'][rows].copy()
    return np.stack(samples), labels, str(path), rows


def main():
    torch.set_num_threads(3)
    out = ROOT/'artifacts/audit/optional_head_regression'
    out.mkdir(exist_ok=True)
    # Both CUDA reference and candidate runs use identical full precision.
    old_precision = torch.get_float32_matmul_precision()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        result = run(out)
        result.update(formal_task_deployed=False, gpu_device=torch.cuda.get_device_name(),
                      matmul_precision='highest', matmul_allow_tf32=False, cudnn_allow_tf32=False,
                      existing_supervisor_restarted=False, core_modified=False,
                      source_sha256={name:file_sha(ROOT/'benchmark'/name) for name in
                                     ['saga_kernel.py','lf.py','noncbm.py','head_suite.py','optional_head_audit.py']})
        (ROOT/'artifacts/audit/optional_head_regression.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn


def run(out):
    source = ROOT/'artifacts/matched/lf/rival10/rn18_task_pretrained_train_val_selected/seed42/single_draw'
    with np.load(source/'lfpi_global_train_original.npz') as z:
        tr = {key:z[key].copy() for key in ['concepts','labels','ids']}
    with np.load(source/'lfpi_global_val_original.npz') as z:
        va = {key:z[key].copy() for key in ['concepts','labels','ids']}
    train_ids, val_ids = balanced(tr['labels'], 16), balanced(va['labels'], 8)
    raw = tr['concepts'][train_ids].astype('float32')
    mean, std = raw.mean(0), raw.std(0, ddof=1).clip(1e-5)
    x, vx = torch.from_numpy((raw-mean)/std), torch.from_numpy((va['concepts'][val_ids]-mean)/std)
    y, vy = torch.from_numpy(tr['labels'][train_ids]).long(), torch.from_numpy(va['labels'][val_ids]).long()
    paths, heads = {}, {}
    upstream()
    from pi_cbm.vendor.lfcbm import elasticnet
    original_function = elasticnet.train_saga
    original_flags = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    for mode in ['upstream','matmul']:
        paths[mode], heads[mode] = fit_nec_path(x,y,vx,vy,10,device='cpu',seed=42,
            steps=500,batch_size=64,points=6,epsilon=.001,targets=[1,3,5],saga_gradient_kernel=mode)
        assert elasticnet.train_saga is original_function
        assert original_flags == (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        (out/f'saga_{mode}.json').write_text(json.dumps(paths[mode],indent=2))
    path_diff=[]
    for index,(a,b) in enumerate(zip(heads['upstream']['path'], heads['matmul']['path'])):
        p=torch.nn.functional.linear(vx,a['weight'],a['bias']);q=torch.nn.functional.linear(vx,b['weight'],b['bias'])
        path_diff.append({'index':index,'lambda_reference':paths['upstream']['path'][index]['lambda'],
            'lambda_candidate':paths['matmul']['path'][index]['lambda'],
            'raw_nec_reference':paths['upstream']['path'][index]['raw_nec'],
            'raw_nec_candidate':paths['matmul']['path'][index]['raw_nec'],
            'weight_difference':difference(a['weight'],b['weight']), 'bias_max_abs':float((a['bias']-b['bias']).abs().max()),
            'validation_logits_difference':difference(p,q), 'validation_prediction_changes':int((p.argmax(1)!=q.argmax(1)).sum())})
    budgets=[]
    for name,a in heads['upstream']['budgets'].items():
        b=heads['matmul']['budgets'][name]
        budgets.append({'target':name,'weight_difference':difference(a['weight'],b['weight']),
                        'connection_mask_changes':int(((a['weight'].abs()>1e-5)!=(b['weight'].abs()>1e-5)).sum())})
    saga={'source':str(source),'train_sample_ids':tr['ids'][train_ids].tolist(),'val_sample_ids':va['ids'][val_ids].tolist(),
          'classes':10,'concepts':raw.shape[1],'all_path_points_computed':len(path_diff),
          'same_best_validation_index':paths['upstream']['best_val_index']==paths['matmul']['best_val_index'],
          'selected_lambda_reference':paths['upstream']['path'][paths['upstream']['best_val_index']]['lambda'],
          'selected_lambda_candidate':paths['matmul']['path'][paths['matmul']['best_val_index']]['lambda'],
          'path_comparison':path_diff,'budget_comparison':budgets,'runtime_function_and_precision_restored':True,
          'unconverged_points_reference':paths['upstream']['unconverged_path_points'],
          'unconverged_points_candidate':paths['matmul']['unconverged_path_points']}
    # >2048 training rows verifies identical GPU randperm and a partial tail batch.
    dense_rows=balanced(tr['labels'],240)
    training={'features':tr['concepts'][dense_rows], 'labels':tr['labels'][dense_rows]}
    validation={'features':va['concepts'][val_ids], 'labels':va['labels'][val_ids]}
    candidates={False:[],True:[]};dense_diff=[]
    for streaming in [False,True]:
        for lr in [.001,.01]:
            for decay in [.0001,.01]:
                torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                trained=fit(training,validation,'cuda',42,lr,decay,streaming_dense=streaming)
                trained['peak_cuda_allocated_MiB']=torch.cuda.max_memory_allocated()/2**20
                candidates[streaming].append(trained)
    for a,b in zip(candidates[False],candidates[True]):
        p=torch.nn.functional.linear((torch.as_tensor(validation['features'])-a['mean'])/a['std'],a['head']['weight'],a['head']['bias'])
        q=torch.nn.functional.linear((torch.as_tensor(validation['features'])-b['mean'])/b['std'],b['head']['weight'],b['head']['bias'])
        dense_diff.append({'lr':a['lr'],'decay':a['weight_decay'],'selected_epoch_reference':a['selected_epoch'],
            'selected_epoch_candidate':b['selected_epoch'],'val_nll_reference':a['validation_nll'],'val_nll_candidate':b['validation_nll'],
            'weight_difference':difference(a['head']['weight'],b['head']['weight']),
            'bias_max_abs':float((a['head']['bias']-b['head']['bias']).abs().max()),
            'mean_exact':torch.equal(a['mean'],b['mean']),'std_exact':torch.equal(a['std'],b['std']),
            'validation_logits_difference':difference(p,q),'validation_prediction_changes':int((p.argmax(1)!=q.argmax(1)).sum()),
            'peak_cuda_allocated_MiB_reference':a['peak_cuda_allocated_MiB'],
            'peak_cuda_allocated_MiB_candidate':b['peak_cuda_allocated_MiB']})
    best=[min(range(4),key=lambda i:candidates[mode][i]['validation_nll']) for mode in [False,True]]
    dense={'train_rows':len(dense_rows),'batch_size':2048,'tail_batch_rows':len(dense_rows)%2048,
           'same_selected_lr_decay':best[0]==best[1],'selected_configuration_indices':best,
           'comparisons':dense_diff,'seed':42,'optimizer':'unchanged AdamW','max_epochs':100,'early_stop_stale':15}
    torch.save({'saga':heads,'dense_candidates':candidates},out/'heads.pt')
    # Large-class actual ImageNet data, but a small batch (no full training).
    features, labels, image_source, positions = actual_imagenet_batch()
    from pi_cbm.vendor.lfcbm.elasticnet import soft_threshold_with_shrinkage
    cuda_records=[]
    for dtype in [torch.float64,torch.float32]:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.manual_seed(42)
        t=torch.from_numpy(features).to(device='cuda',dtype=dtype);n,k=t.shape;c=1000
        weight=torch.randn(c,k,device='cuda',dtype=dtype)*.001
        target=torch.nn.functional.one_hot(torch.from_numpy(labels).long(),c).to(device='cuda',dtype=dtype)
        a=torch.softmax(torch.nn.functional.linear(t,weight),-1)-target
        previous=torch.softmax(torch.randn(n,c,device='cuda',dtype=dtype),-1)-target
        average=torch.randn_like(weight)*.001
        direct=(a.unsqueeze(2)*t.unsqueeze(1)).mean(0);prior=(previous.unsqueeze(2)*t.unsqueeze(1)).mean(0)
        gemm=a.T@t/n;gemm_prior=previous.T@t/n
        saga_a=direct-prior+average;saga_b=gemm-gemm_prior+average
        old=soft_threshold_with_shrinkage(weight-.1*saga_a,.1*.0007*.99,.1*.0007*.01)
        new=soft_threshold_with_shrinkage(weight-.1*saga_b,.1*.0007*.99,.1*.0007*.01)
        torch.cuda.synchronize()
        cuda_records.append({'dtype':str(dtype),'gradient':difference(direct,gemm),'previous_gradient':difference(prior,gemm_prior),
            'one_step_upstream_prox':difference(old,new),'connection_mask_changes':int(((old.abs()>1e-5)!=(new.abs()>1e-5)).sum()),
            'peak_cuda_allocated_MiB':torch.cuda.max_memory_allocated()/2**20})
        del t,weight,target,a,previous,average,direct,prior,gemm,gemm_prior,saga_a,saga_b,old,new
    return {'saga_regression':saga,'dense_regression':dense,'cuda_imagenet':{
        'source':image_source,'row_positions':positions.tolist(),'labels':labels.tolist(),
        'batch':16,'classes':1000,'concepts':features.shape[1],'records':cuda_records}}


if __name__ == '__main__':
    main()
