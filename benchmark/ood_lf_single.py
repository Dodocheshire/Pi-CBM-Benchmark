"""Add single-draw LF-Pi to existing RGB intervention results without rerunning DINO."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .lf import upstream,predict_features
from .metrics import classification_metrics,concept_metrics
from .ood import transform


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);a=p.parse_args();torch.set_num_threads(3)
    directory=Path('artifacts/ood')/a.dataset/'resnet18/paired/seed42'
    protocol=json.loads((directory/'protocol.json').read_text())
    from .pini_matched import MatchedExperiment
    e=MatchedExperiment(a.dataset,Path('artifacts/pini_matched'));upstream()
    from pi_cbm.pipeline import baseline_from_state,build_model
    source=Path('artifacts/matched/lf')/a.dataset/'rn18_imagenet/paired_augmentation/seed42'
    state=torch.load(source/'models.pt',map_location='cpu',weights_only=True)
    teacher=Path('artifacts/common')/('cub' if a.dataset=='waterbirds' else a.dataset)/'teacher'
    train=np.load(e.common/'train_original.npz')['features']
    data={'splits':{'train':{'features':torch.from_numpy(train)}},'text':torch.from_numpy(np.load(teacher/'text.npy'))}
    model=build_model(baseline_from_state(state['pi_cbm'],'cuda'),data,state['selected'],state['config'],'cuda')
    model.generator.load_state_dict(state['generator']);model.eval();net=e.backbone().cuda().eval()
    rgb=np.load(e.root/'cache/rgb_original.npy',mmap_mode='r');masks=np.load(e.root/'cache/foreground.npy',mmap_mode='r')
    ids=np.asarray(protocol['sample_ids']);batch=protocol['batch_size']
    for kind,severity in protocol['settings']:
        key=f'{kind}_{severity}';marker=directory/(key+'_COMPLETE.json')
        metrics=json.loads(marker.read_text())
        if 'lfpi_single' in metrics:continue
        reference=np.load(directory/(key+'_lfpi.npz'));cs=[];ys=[]
        with torch.no_grad():
            for start in range(0,len(ids),batch):
                raw=np.stack([transform(np.array(rgb[i]),kind,severity,np.asarray(masks[i],bool),42+int(i)) for i in ids[start:start+batch]])
                features=net(torch.as_tensor(raw,device='cuda').float()/255).mean((2,3))
                pred=predict_features(model,features,draws=1,seed=4242)
                cs.append(pred['concepts'].numpy());ys.append(pred['logits'].numpy())
        c,y=np.concatenate(cs),np.concatenate(ys)
        values={k:np.array(reference[k]) for k in ['ids','labels','targets','known']}
        values.update(concepts=c,logits=y);np.savez_compressed(directory/(key+'_lfpi_single.npz'),**values)
        metrics['lfpi_single']={'classification':classification_metrics(y,values['labels']),
                              'concepts':concept_metrics(c,values['targets'],values['known'])}
        marker.write_text(json.dumps(metrics,indent=2))
        print(json.dumps({'dataset':a.dataset,'setting':key,'single_draw_accuracy':metrics['lfpi_single']['classification']['accuracy']}),flush=True)
    protocol.update(lfpi_primary_inference_draws=1,lfpi_reference_inference_draws=8,
                    single_draw_inference_seed=4242,single_draw_backbone='same frozen ImageNet ResNet18')
    (directory/'protocol.json').write_text(json.dumps(protocol,indent=2))


if __name__=='__main__':main()
