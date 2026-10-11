"""Compact public metrics and atomic report writes."""
import json
import os
from pathlib import Path
import tempfile


BACKBONE_CACHE_FOLDERS = {
    'resnet18': 'rn18_imagenet',
    'resnet50': 'rn50_imagenet',
    'clip_RN50': 'clip_RN50',
    'resnet18_rival10': 'rn18_task_pretrained_train_val_selected',
}


def backbone_cache_folder(name):
    return BACKBONE_CACHE_FOLDERS.get(name, name)


def backbone_order(name):
    return (list(BACKBONE_CACHE_FOLDERS).index(name), name) if name in BACKBONE_CACHE_FOLDERS else (len(BACKBONE_CACHE_FOLDERS), name)


def backbone_label(name):
    return {
        'resnet18': 'ImageNet ResNet18', 'rn18_imagenet': 'ImageNet ResNet18',
        'resnet50': 'ImageNet ResNet50', 'rn50_imagenet': 'ImageNet ResNet50',
        'clip_RN50': 'CLIP RN50',
        'resnet18_rival10': 'RIVAL10 任务预训练 ResNet18',
        'rn18_task_pretrained_train_val_selected': 'RIVAL10 任务预训练 ResNet18',
    }.get(name, name)


def compact(value):
    if isinstance(value,dict):
        return {k:compact(v) for k,v in value.items() if k not in
                {'average_precision_per_concept','pearson_per_concept',
                 'teacher_pearson_per_concept','teacher_cos_cubed_per_concept'}}
    if isinstance(value,list):return [compact(v) for v in value]
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False,encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');temporary=f.name
    os.replace(temporary,path)


def write_metrics(path,value):
    path=Path(path)
    write_json(Path('artifacts/report_details')/path.name,value)
    write_json(path,compact(value))
