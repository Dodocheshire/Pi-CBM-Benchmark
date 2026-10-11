"""Reuse verified identical CUB RGB encodings with Waterbirds binary labels."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    original=Path('artifacts/common/waterbirds/rn18_imagenet/manifest.json')
    wb=json.loads(original.read_text());cache=Path(wb['cache_root'])
    labels=np.load(cache/'labels.npy')
    for backbone in ['rn50_imagenet','clip_RN50']:
        source=Path('artifacts/common/cub')/backbone
        if not (source/'manifest.json').exists():continue
        cub=json.loads((source/'manifest.json').read_text());audit={}
        for view in ['original','changed']:
            wb_rgb=cache/f'rgb_{view}.npy';cub_rgb=Path(cub[f'rgb_{view}'])
            a,b=sha(wb_rgb),sha(cub_rgb)
            if a!=b:raise ValueError('CUB and derived Waterbirds RGB are not identical')
            audit[view]={'waterbirds_rgb_sha256':a,'cub_rgb_sha256':b}
        destination=Path('artifacts/common/waterbirds')/backbone;destination.mkdir(exist_ok=True)
        for view in ['original','changed']:
            for suffix in ['_features.npy','_features_COMPLETE.json']:
                path=destination/(view+suffix)
                if not path.exists():path.symlink_to((source/(view+suffix)).resolve())
            for split in ['train','val','test']:
                with np.load(source/f'{split}_{view}.npz') as z:
                    values={k:np.array(z[k]) for k in z.files}
                ids=np.load(Path('artifacts/common/waterbirds/rn18_imagenet')/f'{split}_{view}.npz')['ids']
                if not np.array_equal(ids,values['ids']):raise ValueError('Subject fold mismatch')
                values['labels']=labels[ids]
                path=destination/f'{split}_{view}.npz'
                if not path.exists():np.savez_compressed(path,**values)
        cub.update(dataset='waterbirds',dataset_description=wb['dataset_description'],classes=wb['classes'],
                   class_count=2,cache_root=wb['cache_root'],labels=wb['labels'],split_protocol=wb['split_protocol'],
                   rgb_original=wb['rgb_original'],rgb_changed=wb['rgb_changed'],
                   feature_reuse_audit=audit,source_original_manifest=str(original.resolve()))
        (destination/'manifest.json').write_text(json.dumps(cub,indent=2)+'\n')
        print(json.dumps({'dataset':'waterbirds','backbone':backbone,'features_reused_from':str(source)}),flush=True)


if __name__=='__main__':main()
