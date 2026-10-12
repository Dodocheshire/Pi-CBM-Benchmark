"""Prepare or profile one real LF-Pi site batch, never a formal experiment.

Default prepare is CPU-only and materializes two samples. Explicit run requires
--device cuda. Its first B training subjects calibrate noise only for profiling;
the formal site adapter calibrates on the full training fold instead.
"""
import argparse
import json
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .lf import upstream
from .lf_sites import context,backbone,normalized,internal_root,configuration
from .task_backbones import ROOT,file_sha


def inputs(dataset,protocol,seed,site,view,count):
    source,teacher,reference,manifest,saved,folds = context(dataset,protocol,seed)
    ids = folds['train'][:count]
    assert len(ids) == count
    raw = np.load(manifest[f'rgb_{view}'],mmap_mode='r')
    images = normalized(raw[ids])
    receipt = internal_root(dataset)/f'{view}_COMPLETE.json'
    description = {'site':site,'view':view,'subject_ids':ids.tolist(),'rgb_source':manifest[f'rgb_{view}'],
                   'normalization':'shared canonical uint8 /255 and ImageNet mean/std; no resize/crop'}
    if site == 'target_image':
        target = images
        description['input_source'] = 'canonical normalized RGB'
    elif receipt.exists():
        cache = json.loads(receipt.read_text())
        assert cache['point'] == 'pre_layer4' and cache['shape'][1:] == [256,14,14]
        assert cache['backbone_sha256'] == manifest['backbone_sha256']
        assert cache['rgb_sha256'] == file_sha(manifest[f'rgb_{view}'])
        value = np.load(internal_root(dataset)/f'internal_{view}.npy',mmap_mode='r')
        assert list(value.shape) == cache['shape']
        target = torch.from_numpy(np.array(value[ids])).float()
        description.update(input_source='completed real pre-layer4 cache',cache_receipt=cache)
    else:
        # Resource preparation is independent of a concurrently running producer.
        # This is a small private batch and is never written into its namespace.
        encoder = backbone('cpu',manifest)
        with torch.no_grad():
            target = torch.cat([encoder.prefix(x) for x in images.split(32)])
        assert target.shape == (count,256,14,14)
        description.update(input_source='private CPU frozen pre-layer4 batch; full cache not complete',
                           prefix_compute_device='cpu',prefix_compute_batch=32,full_cache_created=False)
    with np.load(source/f'train_{view}.npz') as z:
        assert np.array_equal(z['ids'],folds['train'])
        features = torch.from_numpy(z['features'][:count].copy()).float()
        labels = torch.from_numpy(z['labels'][:count].copy()).long()
    teacher_array = np.load(teacher/f'{view}_teacher.npy',mmap_mode='r')
    teacher_values = torch.from_numpy(np.array(teacher_array[ids])).float()
    assert len(target) == len(features) == len(labels) == len(teacher_values)
    description.update(input_shape=list(target.shape),clean_global_shape=list(features.shape),
                       raw_teacher_shape=list(teacher_values.shape),selected_indices=saved['selected'].tolist(),
                       reference_checkpoint=str(reference/'models.pt'),reference_sha256=file_sha(reference/'models.pt'),
                       backbone_sha256=manifest['backbone_sha256'])
    return target,features,labels,teacher_values,description,(source,teacher,reference,manifest,saved,folds)


def prepare(args):
    before = torch.cuda.is_initialized()
    target,features,labels,teacher,description,info = inputs(args.dataset,args.protocol,args.seed,args.site,args.view,2)
    upstream()
    from pi_cbm.pipeline import baseline_from_state,build_model
    from pi_cbm.inputs import input_loader
    from pi_cbm.training.lfcbm import cos_cubed
    from pi_cbm.precision import adapter_autocast
    from pi_cbm.randomness import seed_all
    # Import interfaces but do not build a stochastic model: its seeded context
    # can initialize CUDA RNG state even for a CPU model on a CUDA host.
    cbm = baseline_from_state(info[4]['baseline'],'cpu').eval().requires_grad_(False)
    with torch.no_grad():
        anchor = cbm.concepts(features)
    assert anchor.shape == (2,len(info[4]['selected'])) and torch.isfinite(anchor).all()
    config = configuration(info[4],args.site,ROOT/'artifacts/audit/lf_sites_profile_unused','cuda',args.batch)
    record = {'state':'cpu_prepared','gpu_profile_started':False,'formal_training_started':False,
              'requested_profile_batch':args.batch,'materialized_cpu_samples':2,'threads':4,
              'input':description,'anchor_shape':list(anchor.shape),'training_settings':config['training'],
              'core_interfaces_imported':['build_model','input_loader','cos_cubed','adapter_autocast','seed_all'],
              'cuda_initialized_before':before,'cuda_initialized_after':torch.cuda.is_initialized(),
              'lf_sites_source_sha256':file_sha(ROOT/'benchmark/lf_sites.py')}
    assert not record['cuda_initialized_after']
    path = ROOT/f'artifacts/audit/lf_sites_profile_{args.dataset}_{args.protocol}_{args.site}_{args.view}_prepare.json'
    path.write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)


def run(args):
    # Gather CPU data before resetting CUDA peaks, so prefix fallback is clearly
    # separated from GPU model construction/calibration and one training step.
    start = time.perf_counter()
    target,features,labels,teacher,description,info = inputs(args.dataset,args.protocol,args.seed,args.site,args.view,args.batch)
    preparation_seconds = time.perf_counter()-start
    saved = info[4]
    output = ROOT/f'artifacts/audit/lf_sites_profile_{args.dataset}_{args.protocol}_{args.site}_{args.view}_batch{args.batch}.json'
    upstream()
    from pi_cbm.pipeline import baseline_from_state,build_model
    from pi_cbm.inputs import input_loader
    from pi_cbm.training.lfcbm import cos_cubed
    from pi_cbm.precision import adapter_autocast
    from pi_cbm.randomness import seed_all
    config = configuration(saved,args.site,ROOT/'artifacts/audit/lf_sites_profile_unused',args.device,args.batch)
    key = 'images' if args.site == 'target_image' else 'internal'
    split = {'features':features,'labels':labels,'indices':torch.as_tensor(description['subject_ids']),
             'teacher':teacher,key:target}
    data = {'splits':{'train':split},'text':torch.from_numpy(np.load(info[1]/'text.npy').copy()).float()}
    free = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
    record = {'state':'starting_gpu_resource_profile','formal_experiment':False,'optimizer_steps':1,'seed':args.seed,
              'site':args.site,'protocol_reference':args.protocol,'input':description,'batch':args.batch,
              'calibration_scope':f'only first {args.batch} real train subjects in {args.view} view; NOT full-fold formal calibration',
              'preparation_seconds':preparation_seconds,'free_MiB_before':int(free),'training_settings':config['training'],
              'classification_head_refit_executed':False,'epochs_executed':0,'checkpoint_saved':False,
              'core_adapt_source_sha256':file_sha(ROOT/'methods/LF-Pi/pi_cbm/training/adapt.py'),
              'lf_sites_source_sha256':file_sha(ROOT/'benchmark/lf_sites.py')}
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    gpu_start = time.perf_counter();phase='model_build'
    try:
        cbm = baseline_from_state(saved['baseline'],args.device)
        with torch.no_grad():
            anchor = cbm.concepts(features.to(args.device)).cpu()
        seed_all(args.seed)
        model = build_model(cbm,data,saved['selected'],config,args.device)
        settings = config['training']
        model.cbm.requires_grad_(False)
        groups = [{'params':list(model.generator.parameters()),'lr':settings['lr']}]
        if settings['stage'] == 'finetune_projection':
            model.cbm.projection.requires_grad_(True)
            groups.append({'params':model.cbm.projection.parameters(),'lr':settings['projection_lr']})
        optimizer = torch.optim.Adam(groups)
        trainable = [p for group in groups for p in group['params']]
        loader = input_loader(split,model,config,labels,anchor,teacher[:,saved['selected']],
                              batch_size=args.batch,shuffle=False)
        model.train();phase='forward'
        batch, y, clean_anchor, raw_teacher = next(iter(loader))
        batch = model.prepare_inputs(batch.to(args.device,non_blocking=True))
        y=y.to(args.device);clean_anchor=clean_anchor.to(args.device);raw_teacher=raw_teacher.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        with adapter_autocast(config):
            logits,concepts,energy = model(batch)
            classification = F.cross_entropy(logits,y)
            classification_value = float(classification.detach())
            anchor_loss = F.mse_loss(concepts,clean_anchor)
            loss = classification
            loss += settings['concept_anchor_weight']*anchor_loss
            loss += settings['noise_energy_weight']*energy
            alignment = None
            if settings['stage'] == 'finetune_projection':
                alignment = 1-cos_cubed(raw_teacher,concepts).mean()
                loss += settings['teacher_alignment_weight']*alignment
        phase='backward';loss.backward()
        grads = [p.grad for p in trainable if p.grad is not None]
        grad_finite = all(torch.isfinite(g).all().item() for g in grads)
        assert grad_finite and grads
        norm = torch.nn.utils.clip_grad_norm_(trainable,settings['grad_clip_norm'])
        assert torch.isfinite(norm)
        phase='optimizer_step';optimizer.step();torch.cuda.synchronize()
        record.update(state='complete',grad_finite=grad_finite,gradient_tensors=len(grads),
            nonzero_gradient_tensors=sum(bool(g.count_nonzero()) for g in grads),
            clip_grad_norm_before=float(norm),loss=float(loss.detach()),
            loss_components={'classification':classification_value,'anchor_mse':float(anchor_loss.detach()),
                             'noise_energy':float(energy.detach()),'teacher_alignment':float(alignment.detach()) if alignment is not None else None},
            model_input_shape=list(batch.shape),anchor_shape=list(clean_anchor.shape),teacher_selected_shape=list(raw_teacher.shape),
            generator_channels=len(model.generator.feature_scale),prefix_attached=model.prefix is not None,
            frozen_suffix_parameter_gradients_absent=all(p.grad is None for p in model.suffix.parameters()))
    except Exception as error:
        record.update(state='failed',phase=phase,error_type=type(error).__name__,error=str(error))
        raise
    finally:
        record.update(gpu_elapsed_seconds=time.perf_counter()-gpu_start,
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
            current_allocated_MiB=torch.cuda.memory_allocated()/2**20)
        output.write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'],nargs='?',default='prepare')
    p.add_argument('--site',choices=['target_image','target_internal'],default='target_image')
    p.add_argument('--dataset',choices=['rival10','cub','waterbirds'],default='rival10')
    p.add_argument('--protocol',choices=['original','paired'],default='original')
    p.add_argument('--view',choices=['original','changed'],default='original')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--batch',type=int,default=256)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args();torch.set_num_threads(4)
    if args.batch < 1:
        p.error('batch must be positive')
    if args.action == 'run':
        if args.device != 'cuda':
            p.error('GPU resource profiling requires explicit --device cuda; default action only prepares CPU samples')
        run(args)
    else:
        if args.device != 'cpu':
            p.error('prepare must remain CPU-only')
        prepare(args)


if __name__ == '__main__':
    main()
