"""Encode the newly trained CUB task backbone without changing model cores."""
import json
import numpy as np
import torch
from .task_backbones import ROOT, file_sha
from .cub_task_backbone import NAMESPACE, BACKBONE_NAME, load_cub_task_backbone, context, protocol
CHECKPOINT_ROOT = ROOT / 'artifacts/task_backbones/cub' / NAMESPACE / 'seed42'
CHECKPOINT = CHECKPOINT_ROOT / 'best.pt'


def provenance(model, manifest, folds):
    terminal=json.loads((CHECKPOINT_ROOT/'COMPLETE.json').read_text())
    history=json.loads((CHECKPOINT_ROOT/'history.json').read_text())
    started=json.loads((CHECKPOINT_ROOT/'test_started.json').read_text())
    assert len(history)==15 and [h['epoch'] for h in history]==list(range(1,16))
    selected=max(history,key=lambda h:h['val_accuracy'])['epoch']
    assert selected==terminal['selected_epoch']==model.metadata['selected_epoch']==started['selected_epoch']
    assert terminal['test_evaluation_count']==1 and terminal['test_selection'] is False
    assert terminal['checkpoint_sha256']==model.checkpoint_sha256
    assert terminal['backbone_state_sha256']==model.state_sha256==started['selected_state_sha256']
    with np.load(CHECKPOINT_ROOT/'test_predictions.npz') as z:
        assert np.array_equal(z['ids'],folds['test']) and len(z['ids'])==5794
    result=protocol(manifest,folds)
    result.update(backbone='torchvision ResNet18 ImageNet-initialized, newly class-supervised on current CUB train only; frozen',
        checkpoint=str(CHECKPOINT),checkpoint_sha256=model.checkpoint_sha256,backbone_sha256=model.state_sha256,
        checkpoint_metadata=model.metadata,checkpoint_history='New seed42 15epoch training; train4796 gradient only, val1198 epoch selection, test5794 frozen evaluation once',
        task_trained=True,selected_epoch=selected,backbone_validation_selection_ids=folds['val'].tolist(),
        task_fit_sample_ids=folds['train'].tolist(),splits={s:ids.tolist() for s,ids in folds.items()},
        counts={s:len(ids) for s,ids in folds.items()},features='raw spatial average; no fitted feature normalization',
        preprocess=manifest['preprocess'],teacher_root=str(ROOT/'artifacts/common/cub/teacher'))
    for key in ['cache_root','labels','concept_names','training_targets','targets','known','native_audit']:
        result[key]=manifest[key]
    result['teacher_inputs']={name:{'path':str(ROOT/'artifacts/common/cub/teacher'/name),'sha256':file_sha(ROOT/'artifacts/common/cub/teacher'/name)} for name in ['original_teacher.npy','changed_teacher.npy','text.npy','metadata.json']}
    audit=ROOT/'artifacts/audit/cub_task_checkpoint_verified.json'
    audit.write_text(json.dumps({'checkpoint_sha256':model.checkpoint_sha256,'state_sha256':model.state_sha256,'selected_epoch':selected,'history_epochs':15,'test_evaluation_count':1,'fold_counts':result['counts'],'gradient_exposure':model.metadata['config']['gradient_exposure'],'split_ids_exact':True,'test_prediction_ids_exact':True,'source':str(CHECKPOINT_ROOT)},indent=2))
    return result


def encode(checkpoint=CHECKPOINT, device='cpu', batch=32, spatial=True):
    """Explicit later action; full cache generation is never part of audit."""
    model = load_cub_task_backbone(checkpoint, device)
    source, source_manifest, folds = context()
    output = ROOT / 'artifacts/common/cub' / NAMESPACE
    output.mkdir(parents=True, exist_ok=True)
    metadata = provenance(model, source_manifest, folds)
    teacher_root = ROOT / 'artifacts/common/cub/teacher'
    for view in ['original', 'changed']:
        if not (teacher_root / f'{view}_COMPLETE.json').exists():
            raise ValueError('Shared same-view teacher incomplete')
        raw = np.load(source_manifest[f'rgb_{view}'], mmap_mode='r')
        marker = output / f'{view}_COMPLETE.json'
        features_path = output / f'features_{view}.npy'
        if marker.exists():
            receipt = json.loads(marker.read_text())
            if receipt['checkpoint_sha256'] != model.checkpoint_sha256:
                raise ValueError('Existing task namespace has a different checkpoint')
            if spatial and not (output / f'spatial_{view}.npy').exists():
                raise ValueError('Existing cache lacks requested spatial maps; choose a fresh explicit namespace/protocol before reencoding')
        else:
            pending = output / f'features_{view}.pending.npy'
            features = np.lib.format.open_memmap(pending, mode='w+', dtype='float32', shape=(len(raw), 512))
            maps = None
            if spatial:
                maps = np.lib.format.open_memmap(output / f'spatial_{view}.pending.npy', mode='w+', dtype='float32', shape=(len(raw), 512, 7, 7))
            with torch.no_grad():
                for start in range(0, len(raw), batch):
                    rgb = torch.from_numpy(np.array(raw[start:start+batch])).to(device).float() / 255
                    values = model(rgb)
                    features[start:start+len(rgb)] = values.mean((2, 3)).float().cpu().numpy()
                    if maps is not None:
                        maps[start:start+len(rgb)] = values.float().cpu().numpy()
                    if start % (batch * 50) == 0:
                        print(json.dumps({'view': view, 'images': start+len(rgb), 'total': len(raw)}), flush=True)
            features.flush(); pending.replace(features_path)
            if maps is not None:
                maps.flush(); (output / f'spatial_{view}.pending.npy').replace(output / f'spatial_{view}.npy')
            marker.write_text(json.dumps({'view': view, 'images': len(raw), 'encoding_batch': batch,
                                         'device': device, 'checkpoint_sha256': model.checkpoint_sha256,
                                         'backbone_sha256': model.state_sha256, 'spatial_float32': spatial,
                                         'rgb_sha256': file_sha(source_manifest[f'rgb_{view}'])}, indent=2))
        features = np.load(features_path, mmap_mode='r')
        teacher = np.load(teacher_root / f'{view}_teacher.npy', mmap_mode='r')
        for split, ids in folds.items():
            with np.load(source / f'{split}_{view}.npz') as values:
                if not np.array_equal(values['ids'], ids):
                    raise ValueError('Shared ordered split/view IDs mismatch')
                fields = {key: values[key] for key in ['labels', 'ids', 'targets', 'known', 'training_targets']}
                fields['concept_names'] = np.asarray(source_manifest['concept_names'])
                destination = output / f'{split}_{view}.npz'
                pending = output / f'{split}_{view}.pending.npz'
                np.savez_compressed(pending, features=features[ids], teacher=teacher[ids], **fields)
                pending.replace(destination)
    metadata.update(encoding_device=device, encoding_batch=batch, spatial_float32=spatial,
                    teacher_root=str(teacher_root), source_root=str(source))
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps({'task_cache_complete': str(output)}), flush=True)



if __name__ == '__main__':
    torch.set_num_threads(4)
    encode()
