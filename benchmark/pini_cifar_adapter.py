"""Train-only CLIP proxy binarization for unmodified PiNI BCE supervision.

This supplies data only: native annotations remain unknown. The raw cosine
teacher is retained independently for the shared concept fidelity comparison.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np


def fit_otsu(teacher,train_ids,bins=256,min_positive=.01,min_variance=1e-6):
    train=np.asarray(teacher[train_ids],dtype=np.float32)
    if not np.isfinite(train).all():raise ValueError('Nonfinite teacher')
    thresholds=np.zeros(train.shape[1],np.float32)
    variance=train.var(0,dtype=np.float64)
    for j in range(train.shape[1]):
        lo,hi=float(train[:,j].min()),float(train[:,j].max())
        if hi==lo:thresholds[j]=lo;continue
        counts,edges=np.histogram(train[:,j],bins=bins,range=(lo,hi))
        centers=(edges[:-1]+edges[1:])/2
        w=np.cumsum(counts).astype(np.float64);s=np.cumsum(counts*centers)
        other=len(train)-w;valid=(w>0)&(other>0)
        between=np.full(bins,-np.inf)
        between[valid]=w[valid]*other[valid]*(s[valid]/w[valid]-(s[-1]-s[valid])/other[valid])**2
        thresholds[j]=edges[int(np.argmax(between))+1]
    positive=(train>thresholds).mean(0)
    keep=(variance>=min_variance)&(positive>=min_positive)&(positive<=1-min_positive)
    return thresholds,keep,{'variance':variance.tolist(),'positive_rate':positive.tolist(),'retained_columns':np.flatnonzero(keep).tolist(),'filter_rule':{'min_train_variance':min_variance,'min_train_positive_rate':min_positive,'max_train_positive_rate':1-min_positive},'otsu_bins':bins,'ties':'first maximum between-class variance threshold','comparison':'teacher > threshold','fit_ids_sha256':hashlib.sha256(np.asarray(train_ids,dtype=np.int64).tobytes()).hexdigest(),'fit_images':len(train_ids)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',type=Path,required=True);p.add_argument('--train-ids',type=Path,required=True);p.add_argument('--concepts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    teacher=np.load(a.teacher,mmap_mode='r');ids=np.load(a.train_ids)
    names=json.loads(a.concepts.read_text()) if a.concepts.suffix=='.json' else a.concepts.read_text().splitlines()
    assert teacher.ndim==2 and len(names)==teacher.shape[1]
    threshold,keep,stats=fit_otsu(teacher,ids)
    a.output.mkdir(parents=True,exist_ok=True)
    np.save(a.output/'thresholds.npy',threshold)
    np.save(a.output/'retained_columns.npy',np.flatnonzero(keep))
    out=np.lib.format.open_memmap(a.output/'attrs.npy',mode='w+',dtype='bool',shape=(len(teacher),int(keep.sum())))
    for start in range(0,len(teacher),2048):out[start:start+2048]=teacher[start:start+2048,keep]>threshold[keep]
    out.flush()
    (a.output/'concepts.json').write_text(json.dumps([n for n,k in zip(names,keep) if k]))
    stats.update(target_type='CLIP train-only Otsu binary proxy; no human concept labels',native_concept_known=False,raw_teacher=str(a.teacher.resolve()),native_mAP='unavailable',proxy_mAP='must be explicitly named CLIP Otsu proxy AP',primary_concept_metrics=['Pearson against original continuous cosine','cos_cubed against original continuous cosine'],raw_concept_count=len(names),retained_concept_count=int(keep.sum()),test_or_val_used_for_threshold_or_filter=False)
    (a.output/'supervision_audit.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps({'retained':int(keep.sum()),'raw':len(names),'fit_images':len(ids)}))

if __name__=='__main__':main()
