"""Compact public metrics and atomic report writes."""
import json
import os
from pathlib import Path
import tempfile


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
