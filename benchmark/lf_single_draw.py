"""Export one-draw LF-Pi predictions from completed checkpoints without retraining."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .lf import upstream,predict_features,export_npz
from .metrics import classification_metrics,concept_metrics


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def run(run_dir,source,teacher,device=None):
    run_dir,source,teacher=map(Path,(run_dir,source,teacher))
    saved=torch.load(run_dir/'models.pt',weights_only=True,map_location='cpu')
    config=saved['config'];device=device or config['experiment']['device']
    if config['injection']['site']!='target_global':
        raise ValueError('This exporter supports cached target_global inputs only')
    upstream()
    from pi_cbm.pipeline import baseline_from_state,build_model
    selected=saved['selected'].numpy();names=saved['concepts']
    original=np.load(source/'train_original.npz')
    train=torch.from_numpy(original['features'].copy()).float()
    if train.ndim==4:train=train.mean((2,3))
    data={'splits':{'train':{'features':train}},
          'text':torch.from_numpy(np.load(teacher/'text.npy').copy()).float()}
    cbm=baseline_from_state(saved['pi_cbm'],device)
    model=build_model(cbm,data,saved['selected'],config,device)
    model.generator.load_state_dict(saved['generator'])
    model.eval().requires_grad_(False)
    output=run_dir/'single_draw';output.mkdir(exist_ok=True)
    metadata={'method':'lfpi_global','inference_draws':1,'inference_seed':3000,
              'batch_size':256,'device':device,'retrained':False,
              'checkpoint':str(run_dir/'models.pt'),'checkpoint_sha256':digest(run_dir/'models.pt'),
              'source':str(source),'teacher_text_sha256':digest(teacher/'text.npy'),
              'selected_indices':selected.tolist(),'concept_names':names,
              'prior_eight_draw_exports':str(run_dir),'input_mapping':{},'metrics':{}}
    for split_name in ['train','val','test']:
        for view in ['original','changed']:
            path=source/f'{split_name}_{view}.npz'
            if not path.exists():continue
            raw=np.load(path)
            x=torch.from_numpy(raw['features'].copy()).float()
            if x.ndim==4:x=x.mean((2,3))
            if x.shape[1]!=cbm.projection.in_features:
                raise ValueError('Feature width differs from checkpoint projection')
            split={'features':x,'labels':torch.from_numpy(raw['labels'].copy()).long(),
                   'indices':torch.from_numpy(raw['ids'].copy()).long()}
            prediction=predict_features(model,x,draws=1,seed=3000,batch_size=256)
            targets=raw['targets'][:,selected]
            known=raw['known'][:,selected] if 'known' in raw.files else np.isfinite(targets)
            destination=output/f'lfpi_global_{split_name}_{view}.npz'
            export_npz(destination,prediction,split,names,targets=targets,known=known)
            key=f'{split_name}/{view}'
            metadata['input_mapping'][key]={'source_file':str(path),'source_sha256':digest(path),
                'ids_sha256':hashlib.sha256(split['indices'].numpy().tobytes()).hexdigest(),
                'images':len(x),'output_file':str(destination)}
            if split_name!='train':
                metadata['metrics'][key]={'classification':classification_metrics(
                    prediction['logits'].numpy(),split['labels'].numpy()),
                    'semantic_concepts':concept_metrics(prediction['concepts'].numpy(),targets,known)}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'single_draw_output':str(output),'device':device},indent=2),flush=True)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--source',required=True)
    parser.add_argument('--teacher',required=True);parser.add_argument('--device')
    args=parser.parse_args();torch.set_num_threads(4)
    run(args.run,args.source,args.teacher,args.device)


if __name__=='__main__':main()
