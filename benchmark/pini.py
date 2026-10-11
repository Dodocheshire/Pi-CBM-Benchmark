"""External adapters for the frozen PiNI submodule; no loss/generator changes."""
from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / 'methods/PiNI'


def import_core():
    core = str(CORE)
    if core not in sys.path:
        sys.path.insert(0, core)
    return importlib.import_module('spatial_noise.cub_noise')


class ConfigurableExperiment:
    """Core-compatible data/model/backbone interface with explicit fit stage.

    Caches contain RGB uint8 NCHW, labels int64, native binary attributes,
    foreground boolean NHW, RN features N512HW; regions/features are N25614x14,
    regions/maps are NR14x14. Backbones take RGB float [0,1] and return N512HW.
    `backbone_factory` is `module:function`; normalization belongs to that factory.
    Selection caches must come from a backbone that did not fit validation IDs.
    """
    def __init__(self, config: dict, stage: str):
        if stage not in ('selection', 'refit'):
            raise ValueError(stage)
        import_core()
        self.config, self.stage = config, stage
        self.name = config['dataset']
        self.root = Path(config['output_root']).resolve()
        self.cache_root = Path(config['cache_by_stage'][stage]).resolve()
        self.phrases = list(config['region_phrases'])
        self.k, self.classes = int(config['concept_count']), int(config['class_count'])
        self.detector_precision = config.get('detector_precision', 'fp32')
        self._validate()

    def _validate(self):
        arrays, splits, protocol = self.load()
        sets = {key:set(map(int, ids)) for key,ids in splits.items()}
        if any(sets[a] & sets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
            raise ValueError('Data splits overlap')
        if arrays['attrs'].shape[1] != self.k:
            raise ValueError('Concept count does not match native attribute columns')
        provenance = self.config['backbone_provenance_by_stage'][self.stage]
        fitted = set(map(int, provenance.get('fit_sample_ids', [])))
        forbidden = sets['test'] | (sets['val'] if self.stage == 'selection' else set())
        if fitted & forbidden:
            raise ValueError('Backbone fit overlaps evaluation holdout')
        if self.stage == 'selection' and provenance.get('trained_on_trainval', False):
            raise ValueError('Selection must not use a trainval-refitted backbone')
        if not provenance.get('cache_backbone_matches_live', False):
            raise ValueError('Explicit audit that cached features and live backbone use the same checkpoint is required')
        spec=self.config['backbone_by_stage'][self.stage]
        if spec['factory']=='spatial_noise.rn18_datasets:RivalBackbone':
            import torch
            checkpoint=torch.load(spec['kwargs']['path'],map_location='cpu',weights_only=True)
            if checkpoint.get('stage')!=self.stage:
                raise ValueError('RIVAL backbone checkpoint stage mismatch')
            expected=len(splits['train'])+(len(splits['val']) if self.stage=='refit' else 0)
            if checkpoint.get('fit_images')!=expected:
                raise ValueError('RIVAL backbone fit_images mismatches stage training partition')
        if not provenance.get('holdout_audited', False):
            raise ValueError('Explicit backbone holdout provenance audit is required')
        shape = arrays['features_original'].shape
        if len(shape) != 4 or shape[1] != 512:
            raise ValueError(f'Frozen SimpleCBM requires N512HW features; got {shape}')
        self.protocol = protocol

    def prepare_links(self):
        """Expose external cache/regions at paths expected by unmodified core."""
        self.root.mkdir(parents=True, exist_ok=True)
        targets = {'cache': self.cache_root,
                   'regions': Path(self.config['regions_by_stage'][self.stage]).resolve()}
        for name, source in targets.items():
            destination = self.root / name
            if destination.exists() or destination.is_symlink():
                if destination.resolve() != source:
                    raise ValueError(f'{destination} must point to {source}; use a separate output_root per stage')
            else:
                destination.symlink_to(source, target_is_directory=True)

    def load(self):
        protocol = json.loads((self.cache_root/'protocol.json').read_text())
        splits = {k:np.asarray(v,dtype=np.int64) for k,v in protocol['splits'].items()}
        names = ['labels','attrs','rgb_original','foreground','features_original']
        if (self.cache_root/'background_eligible.npy').exists():
            names.append('background_eligible')
        return ({k:np.load(self.cache_root/(k+'.npy'),mmap_mode='r') for k in names},splits,protocol)

    def model(self):
        from spatial_noise.cub_aligned import SimpleCBM
        return SimpleCBM(self.k,self.classes)

    def backbone(self):
        spec = self.config['backbone_by_stage'][self.stage]
        module, function = spec['factory'].split(':')
        model = getattr(importlib.import_module(module),function)(**spec.get('kwargs',{}))
        return model.eval().requires_grad_(False)


def train_external(config: dict, stage='selection', method='noise', epochs=10, batch=64, seed=42, algorithm='shared_noise', pairs=32):
    """Invoke the original single-view training loss through an Experiment wrapper.

    Initialize output_root/simple[/_refit]/seedN/best.pt with the desired initial
    SimpleCBM checkpoint before calling. Stage-specific cache roots keep refit
    feature banks out of selection. Pair core retains its own fixed init protocol.
    """
    core = import_core()
    experiment = ConfigurableExperiment(config,stage)
    experiment.prepare_links()
    if algorithm=='pair':
        if method not in ('mean','noise') or seed!=42:
            raise ValueError('Frozen pair core supports mean/noise and seed42 only')
        pair_core=importlib.import_module('spatial_noise.noise_pair')
        # Core initializes from prior_off/seed42/<stage>/mean/best.pt.
        return pair_core.train(experiment,method,stage,epochs=epochs,pairs=pairs)
    if algorithm!='shared_noise':
        raise ValueError(algorithm)
    return core.train(method,seed=seed,epochs=epochs,batch=batch,
                      refit=stage=='refit',spatial_prior=False,experiment=experiment)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=['selection','refit'],default='selection')
    parser.add_argument('--method',choices=['clean','mean','noise'],default='noise')
    parser.add_argument('--algorithm',choices=['shared_noise','pair'],default='shared_noise')
    parser.add_argument('--pairs',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.validate_only:
        experiment=ConfigurableExperiment(config,args.stage)
        print(json.dumps({'validated':True,'stage':args.stage,'concepts':experiment.k,'classes':experiment.classes,'cache':str(experiment.cache_root)}))
    else:
        train_external(config,args.stage,args.method,args.epochs,args.batch,args.seed,args.algorithm,args.pairs)


if __name__=='__main__':main()
