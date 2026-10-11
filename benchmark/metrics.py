"""Shared concept, classification and paired-shift metrics with explicit masks."""
import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score


def concept_metrics(scores, targets, known=None):
    scores=np.asarray(scores,dtype=np.float64)
    targets=np.asarray(targets,dtype=np.float64)
    if scores.shape!=targets.shape or scores.ndim!=2:
        raise ValueError('scores and targets must share [images, concepts] shape')
    valid=np.isfinite(scores)&np.isfinite(targets)
    if known is not None:
        if np.asarray(known).shape!=scores.shape:raise ValueError('known mask shape mismatch')
        valid &= np.asarray(known,dtype=bool)
    ap,pearson,spearman=[],[],[]
    for k in range(scores.shape[1]):
        x,y=scores[valid[:,k],k],targets[valid[:,k],k]
        row={'concept_index':k,'known_images':len(y)}
        binary=np.all(np.isin(y,[0,1]))
        if len(y)>1 and binary and len(np.unique(y))==2:
            row['value']=float(average_precision_score(y,x));ap.append(row)
        if len(y)>2 and np.std(x)>1e-12 and np.std(y)>1e-12:
            pearson.append({'concept_index':k,'value':float(np.corrcoef(x,y)[0,1])})
            spearman.append({'concept_index':k,'value':float(spearmanr(x,y).statistic)})
    average=lambda rows:float(np.mean([r['value'] for r in rows])) if rows else None
    return {'mAP':average(ap),'pearson_macro':average(pearson),'spearman_macro':average(spearman),
            'concepts':scores.shape[1],'images':scores.shape[0],'known_pairs':int(valid.sum()),
            'mAP_evaluable_concepts':len(ap),'correlation_evaluable_concepts':len(pearson),
            'average_precision_per_concept':ap,'pearson_per_concept':pearson}


def classification_metrics(logits, labels, probabilities=False):
    logits=np.asarray(logits,dtype=np.float64);labels=np.asarray(labels,dtype=np.int64)
    if logits.ndim!=2 or len(labels)!=len(logits):raise ValueError('classification shape mismatch')
    if not np.isfinite(logits).all():raise ValueError('nonfinite predictions')
    p=logits if probabilities else softmax(logits,axis=1)
    if probabilities and (np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-5)):
        raise ValueError('invalid probabilities')
    correct=p.argmax(1)==labels
    confidence=p.max(1);ece=0.
    bins=np.minimum((confidence*15).astype(int),14)
    for b in range(15):
        use=bins==b
        if use.any():ece+=use.mean()*abs(correct[use].mean()-confidence[use].mean())
    return {'accuracy':float(correct.mean()),'nll':float(-np.log(p[np.arange(len(p)),labels].clip(1e-12)).mean()),
            'ece_15_bins':float(ece),'examples':len(labels),'correct':int(correct.sum())}


def nec(weights, threshold=1e-5):
    w=np.asarray(weights)
    if w.ndim!=2:raise ValueError('head weights must be [classes, concepts]')
    counts=(np.abs(w)>threshold).sum(1)
    return {'nec':float(counts.mean()),'nonzero_connections':int(counts.sum()),
            'per_class':counts.tolist(),'threshold':threshold,'concept_pool':w.shape[1]}


def paired_concept_metrics(a,b,train_scores=None):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape:raise ValueError('pair alignment required')
    finite=np.isfinite(a)&np.isfinite(b)
    difference=b-a
    result={'raw_mae':float(np.abs(difference[finite]).mean()),
            'raw_mse':float(np.square(difference[finite]).mean()),'images':len(a)}
    if train_scores is not None:
        scale=np.asarray(train_scores,dtype=float).std(0)
        active=np.isfinite(scale)&(scale>1e-8)
        normalized=difference[:,active]/scale[active]
        result.update(standardized_mae=float(np.abs(normalized).mean()),
                      standardized_mse=float(np.square(normalized).mean()),
                      normalized_concepts=int(active.sum()))
    return result


def paired_accuracy_interval(a,b,labels,seed=42,replicates=2000):
    y=np.asarray(labels)
    d=(np.asarray(a).argmax(1)==y).astype(float)-(np.asarray(b).argmax(1)==y).astype(float)
    rng=np.random.default_rng(seed);samples=[]
    for start in range(0,replicates,50):
        draw=rng.integers(len(y),size=(min(50,replicates-start),len(y)))
        samples.extend(d[draw].mean(1))
    return {'accuracy_difference':float(d.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'unit':'paired image','replicates':replicates,'seed':seed}
