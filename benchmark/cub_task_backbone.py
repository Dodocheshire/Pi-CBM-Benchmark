"""New CUB train-only class-supervised RN18 registry, outside frozen cores.

Default audit verifies the ImageNet initialization interface on CPU only. It is
not a trained CUB checkpoint or a task-feature cache. Training requires an
explicit separate train action; all selection uses current validation IDs.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

from .metrics import classification_metrics
from .task_backbones import ROOT, TaskRN18, file_sha, state_sha

NAMESPACE = 'rn18_task_pretrained_cub_train_val_selected'
BACKBONE_NAME = 'resnet18_cub_trainonly'
IMAGENET = Path('/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
PLAN = {
    'seed': 42, 'epochs': 15, 'batch': 128, 'threads': 4,
    'optimizer': 'AdamW', 'lr': .0003, 'weight_decay': .0001,
    'scheduler': 'CosineAnnealingLR', 'scheduler_T_max': 15,
    'loss': 'class-label cross entropy; 200 species; no concept supervision',
    'augmentation': 'train-only horizontal flip p=.5; original canonical224RGB; no additional resize/crop',
    'normalization': 'ImageNet RGB mean/std',
    'selection': 'strictly higher val accuracy; ties keep earliest epoch; train all15 epochs',
    'test_protocol': 'test only once after loading selected frozen checkpoint; never choose epoch or hyperparameter on test',
    'train_subjects': 4796, 'val_subjects': 1198, 'test_subjects': 5794,
}


class CUBTaskRN18(TaskRN18):
    """Same raw RGB→spatial/global API as RIVAL, with a separate registry."""
    def __init__(self, checkpoint=None):
        nn.Module.__init__(self)
        self.network = resnet18(weights=None)
        if checkpoint is None:
            if not IMAGENET.exists():
                raise FileNotFoundError('Audited existing ImageNet initialization missing; no implicit download')
            self.network.load_state_dict(torch.load(IMAGENET, map_location='cpu', weights_only=True))
            self.network.fc = nn.Linear(512, 200)
            self.checkpoint = None
            self.checkpoint_sha256 = None
            self.task_trained = False
            self.metadata = {'initialization': 'ImageNet1K_V1 resnet18-f37072fd',
                             'initialization_file_sha256': file_sha(IMAGENET),
                             'task_checkpoint_trained': False, 'classifier_seed': 42}
        else:
            self.checkpoint = Path(checkpoint).resolve()
            saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
            if saved['dataset'] != 'cub' or saved['backbone_name'] != BACKBONE_NAME or not saved['training_complete']:
                raise ValueError('Reject unknown, historical external CUB, or incomplete checkpoint')
            _, _, folds = context()
            expected_exposure = {'train':4796, 'val':0, 'test':0}
            if saved['config'].get('gradient_exposure') != expected_exposure:
                raise ValueError('CUB task checkpoint gradient exposure differs from registered train-only protocol')
            if saved.get('test_evaluation_count') != 1 or not 1 <= saved['selected_epoch'] <= 15:
                raise ValueError('CUB task checkpoint final selection/evaluation metadata invalid')
            for split in folds:
                if not np.array_equal(np.asarray(saved['split_ids'][split]), folds[split]) or not np.array_equal(np.asarray(saved['config']['exact_split_ids'][split]),folds[split]):
                    raise ValueError('CUB registered checkpoint uses different current folds')
            self.network.fc = nn.Linear(512, 200)
            self.network.load_state_dict(saved['model'], strict=True)
            self.checkpoint_sha256 = file_sha(self.checkpoint)
            self.task_trained = True
            self.metadata = {key: value for key, value in saved.items() if key not in ['model', 'split_ids']}
        self.state_sha256 = state_sha(self.network.state_dict())
        if checkpoint is not None and self.state_sha256 != saved['backbone_state_sha256']:
            raise ValueError('CUB task checkpoint state digest mismatch')
        self.eval().requires_grad_(False)


def load_cub_task_backbone(checkpoint=None, device='cpu'):
    # Registered initialization has a reproducible fresh 200-class head.
    if checkpoint is None:
        torch.manual_seed(42)
    return CUBTaskRN18(checkpoint).to(device).eval().requires_grad_(False)


def context():
    source = ROOT / 'artifacts/common/cub/rn18_imagenet'
    manifest = json.loads((source / 'manifest.json').read_text())
    protocol = json.loads(Path(manifest['split_protocol']).read_text())
    folds = {}
    for split in ['train', 'val', 'test']:
        with np.load(source / f'{split}_original.npz') as values:
            folds[split] = values['ids'].copy()
        if not np.array_equal(folds[split], np.asarray(protocol['splits'][split], dtype=np.int64)):
            raise ValueError('CUB current fold mismatch')
    if [len(folds[s]) for s in ['train', 'val', 'test']] != [4796, 1198, 5794]:
        raise ValueError('Unregistered CUB fold sizes')
    groups = [set(ids) for ids in folds.values()]
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError('Underlying CUB folds overlap')
    if set(np.concatenate(list(folds.values()))) != set(range(11788)):
        raise ValueError('CUB folds do not cover ordered source RGB')
    return source, manifest, folds


def protocol(manifest, folds):
    old_cache = Path(manifest['cache_root'])
    native = np.load(old_cache / 'native_ids.npy')
    return {
        'dataset': 'cub', 'evaluation_namespace': NAMESPACE,
        'backbone_name': BACKBONE_NAME, 'core_global_model_backbone_name': 'resnet18',
        'feature_dim': 512, 'spatial_shape': [512, 7, 7],
        'loader_input': 'raw canonical N3x224x224 RGB float[0,1]; adapter normalizes internally; spatial/global/class_logits methods',
        'source_common_manifest': str(ROOT / 'artifacts/common/cub/rn18_imagenet/manifest.json'),
        'source_common_manifest_sha256': file_sha(ROOT / 'artifacts/common/cub/rn18_imagenet/manifest.json'),
        'rgb_original': manifest['rgb_original'], 'rgb_changed': manifest['rgb_changed'],
        'split_protocol': manifest['split_protocol'], 'split_protocol_sha256': file_sha(manifest['split_protocol']),
        'exact_split_ids': {s: ids.tolist() for s, ids in folds.items()},
        'exact_native_split_ids': {s: native[ids].tolist() for s, ids in folds.items()},
        'native_ids_path': str(old_cache / 'native_ids.npy'), 'native_ids_sha256': file_sha(old_cache / 'native_ids.npy'),
        'training_plan': PLAN,
        'backbone_supervision': 'train4796 species class labels; val1198 only epoch selection; no concept loss',
        'gradient_exposure': {'train': 4796, 'val': 0, 'test': 0},
        'validation_role': '1198 selected best backbone epoch; no validation gradient',
        'test_role': '5794 entirely excluded until one final frozen-selected-model class evaluation',
        'existing_external_cub_checkpoint_used': False,
        'future_shared_teacher': str(ROOT / 'artifacts/common/cub/teacher'),
        'future_cache_fields': ['features', 'labels', 'ids', 'targets', 'known', 'training_targets', 'concept_names', 'teacher'],
        'historical_checkpoint': False,
    }


def audit():
    source, manifest, folds = context()
    model = load_cub_task_backbone(device='cpu')
    ids = np.asarray([folds['train'][0], folds['val'][0]])
    raw = np.load(manifest['rgb_original'], mmap_mode='r')
    if raw.shape != (11788, 3, 224, 224) or raw.dtype != np.uint8:
        raise ValueError('CUB canonical RGB must be existing uint8 N3x224x224')
    rgb = torch.from_numpy(np.array(raw[ids])).float() / 255
    with torch.no_grad():
        maps = model(rgb)
        features = model.global_features(rgb)
        logits = model.class_logits(rgb)
        full_logits = model.network(model.normalize(rgb))
    assert maps.shape == (2, 512, 7, 7) and features.shape == (2, 512) and logits.shape == (2, 200)
    torch.testing.assert_close(features, maps.mean((2, 3)), rtol=0, atol=0)
    torch.testing.assert_close(logits, full_logits, rtol=1e-5, atol=1e-5)
    model.train(True)
    assert not model.training and not model.network.training
    input_rgb = rgb[:1].clone().requires_grad_(True)
    model.global_features(input_rgb).sum().backward()
    assert input_rgb.grad is not None and torch.isfinite(input_rgb.grad).all() and input_rgb.grad.abs().max() > 0
    assert all(p.grad is None for p in model.parameters())
    result = protocol(manifest, folds)
    result.update(audit_device='cpu', audit_scope='ImageNet initialization interface only; NOT trained CUB task features',
                  task_checkpoint_trained=False, task_training_started=False, gpu_jobs_started=False,
                  gradient_exposure_is_plan=True, actual_gradient_exposure={'train':0, 'val':0, 'test':0},
                  initialized_state_sha256=model.state_sha256, initialization=model.metadata,
                  spatial_audit_shape=list(maps.shape), global_audit_shape=list(features.shape),
                  classifier_audit_shape=list(logits.shape),
                  class_logits_vs_full_network_max_abs=float((logits-full_logits).abs().max()),
                  spatial_global_exact_parity=True, frozen_batchnorm_verified=True, input_gradient_supported=True,
                  sample_global_ids=ids.tolist())
    output = ROOT / 'artifacts/audit/cub_task_backbone_initialization_cpu.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'initialization_only_cpu_audit': str(output), 'task_trained': False}), flush=True)


@torch.no_grad()
def evaluate(network, raw, labels, ids, device, batch):
    logits = []
    network.eval()
    for start in range(0, len(ids), batch):
        selection = ids[start:start+batch]
        rgb = torch.from_numpy(np.array(raw[selection])).to(device).float() / 255
        logits.append(network(TaskRN18.normalize(rgb)).float().cpu())
    values = torch.cat(logits)
    target = torch.from_numpy(labels[ids]).long()
    return {'accuracy': float((values.argmax(1)==target).float().mean()),
            'nll': float(nn.functional.cross_entropy(values, target))}, values.numpy()


def train(device='cpu'):
    source, manifest, folds = context()
    output = ROOT / 'artifacts/task_backbones/cub' / NAMESPACE / 'seed42'
    if output.exists() and any(output.iterdir()):
        raise ValueError('Registered CUB run already started; no implicit restart/overwrite or repeated test evaluation; inspect and explicitly recover an interrupted run')
    output.mkdir(parents=True, exist_ok=True)
    model = load_cub_task_backbone(device=device)
    network = model.network.requires_grad_(True)
    raw = np.load(manifest['rgb_original'], mmap_mode='r')
    labels = np.load(manifest['labels'])
    if raw.shape != (11788, 3, 224, 224) or not np.array_equal(np.unique(labels),np.arange(200)):
        raise ValueError('Unexpected CUB canonical source or class labels')
    config = protocol(manifest, folds)
    config.update(initialization=model.metadata, initialized_state_sha256=model.state_sha256,
                  device=device, torch_version=str(torch.__version__),
                  cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    optimizer = torch.optim.AdamW(network.parameters(), lr=PLAN['lr'], weight_decay=PLAN['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PLAN['epochs'])
    generator = torch.Generator(device=device).manual_seed(42)
    use_amp = str(device).startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=256)
    best = -float('inf'); history = []; started = time.time()
    for epoch in range(1, PLAN['epochs']+1):
        network.train()
        order = torch.randperm(len(folds['train']), generator=generator, device=device).cpu().numpy()
        ids = folds['train'][order]
        total_loss = 0.0
        for start in range(0, len(ids), PLAN['batch']):
            selection = ids[start:start+PLAN['batch']]
            rgb = torch.from_numpy(np.array(raw[selection])).to(device).float() / 255
            flips = torch.rand(len(rgb), generator=generator, device=device)<.5
            rgb[flips] = rgb[flips].flip(-1)
            target = torch.from_numpy(labels[selection]).to(device).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda' if use_amp else 'cpu', dtype=torch.float16, enabled=use_amp):
                loss = nn.functional.cross_entropy(network(TaskRN18.normalize(rgb)), target)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += float(loss.detach())*len(rgb)
        validation, _ = evaluate(network, raw, labels, folds['val'], device, PLAN['batch'])
        entry = {'epoch': epoch, 'train_nll': total_loss/len(ids), 'val_accuracy': validation['accuracy'],
                 'val_nll': validation['nll'], 'seconds': time.time()-started}
        history.append(entry)
        if validation['accuracy'] > best:
            best = validation['accuracy']
            torch.save({'model': {k:v.cpu().clone() for k,v in network.state_dict().items()},
                        'dataset': 'cub', 'backbone_name': BACKBONE_NAME, 'training_complete': False,
                        'selected_epoch': epoch, 'seed': 42, 'fit_images': 4796,
                        'split_ids': config['exact_split_ids'], 'config': config}, output / 'best.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(entry), flush=True); scheduler.step()
    saved = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    network.load_state_dict(saved['model']);network.eval().requires_grad_(False)
    (output / 'test_started.json').write_text(json.dumps({'selected_epoch':saved['selected_epoch'],
        'protocol':'one frozen-selected-model evaluation only; test never selected training',
        'selected_state_sha256':state_sha(network.state_dict())},indent=2))
    test, logits = evaluate(network, raw, labels, folds['test'], device, PLAN['batch'])
    np.savez_compressed(output / 'test_predictions.npz', logits=logits, labels=labels[folds['test']], ids=folds['test'])
    saved.update(training_complete=True, test_evaluation_count=1, backbone_state_sha256=state_sha(network.state_dict()))
    torch.save(saved, output / 'best.pt')
    terminal = {'selected_epoch': saved['selected_epoch'], 'val_accuracy': best,
                'test_evaluation_count': 1, 'test': classification_metrics(logits, labels[folds['test']]),
                'checkpoint_sha256': file_sha(output / 'best.pt'), 'backbone_state_sha256': saved['backbone_state_sha256'],
                'seed':42, 'test_selection': False, 'actual_epochs': 15, 'device': device}
    (output / 'COMPLETE.json').write_text(json.dumps(terminal, indent=2))
    print(json.dumps({'task_train_complete': str(output), **terminal}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['audit', 'train'], nargs='?', default='audit')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args();torch.set_num_threads(4)
    if args.action == 'audit':
        if args.device != 'cpu':parser.error('Initialization audit is CPU-only')
        audit()
    else:
        train(args.device)


if __name__ == '__main__':
    main()
