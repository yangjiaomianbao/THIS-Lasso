"""Exact continuous-score metrics with implicit huge zero-score populations."""
import math
import numpy as np
from sklearn.metrics import average_precision_score


def sparse_metrics(truth,scores,universe,threshold=1e-8):
    truth=set(truth);scores={k:float(v) for k,v in scores.items() if v!=0}
    if universe<len(truth|scores.keys()) or any(not np.isfinite(v) or v<0 for v in scores.values()):raise ValueError('Invalid universe/scores')
    predicted={k for k,v in scores.items() if v>threshold};tp=len(predicted&truth);fp=len(predicted-truth);fn=len(truth-predicted);tn=universe-tp-fp-fn
    denom=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    y=[k in truth for k in scores];s=list(scores.values());w=[1]*len(s)
    missing_true=len(truth-scores.keys());missing_false=universe-len(truth)-len(scores.keys()-truth)
    for label,num in [(True,missing_true),(False,missing_false)]:
        if num:y.append(label);s.append(0.);w.append(num)
    valid=0<len(truth)<universe
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        AP=float(average_precision_score(y,s,sample_weight=w)) if valid else None,
        AP_reason='' if valid else 'no positive or no negative candidates',
        MCC=(tp*tn-fp*fn)/denom if denom else None,MCC_reason='' if denom else 'zero confusion-matrix denominator',
        Jaccard=tp/(tp+fp+fn) if tp+fp+fn else None,Recall=tp/len(truth) if truth else None,
        Precision=tp/(tp+fp) if tp+fp else 0.,universe=universe)


def screening_metrics(truth,pairs,scores,ranking,k,full_count):
    truth=set(truth);pool=set(map(tuple,pairs));chosen=set(map(tuple,pairs[ranking[:k]]));hits=len(chosen&truth)
    return dict(original_candidates=full_count,pool_size=len(pool),selected_count=len(chosen),retained_true=hits,
        pool_recall=len(pool&truth)/len(truth) if truth else None,
        Recall_at_K=hits/len(truth) if truth else None,Precision_at_K=hits/len(chosen) if chosen else None,
        compression_ratio=len(chosen)/full_count,
        pool_AP=sparse_metrics(truth&pool,dict(zip(map(tuple,pairs),scores)),len(pool))['AP'],
        full_pipeline_screen_AP=sparse_metrics(truth,dict(zip(map(tuple,pairs),scores)),full_count)['AP'])
