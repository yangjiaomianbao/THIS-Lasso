"""Exact directed hyperedge scores and original AP/strict-threshold metrics."""
import itertools
import math
import numpy as np
from sklearn.metrics import average_precision_score


def binary_metrics(truth, scores, threshold=1e-8):
    truth, scores = np.asarray(truth, bool), np.asarray(scores, float)
    if truth.ndim != 1 or scores.shape != truth.shape or not np.isfinite(scores).all():
        raise ValueError('Require equal finite 1D candidate arrays')
    predicted = scores > threshold
    tp = int(np.sum(truth & predicted));fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted));tn = int(np.sum(~truth & ~predicted))
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    both = bool(np.any(truth) and np.any(~truth))
    reason = '' if both else ('no_positive_truth' if not np.any(truth) else 'no_negative_truth')
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, candidate_count=len(truth),
        positive_count=int(np.sum(truth)), negative_count=int(np.sum(~truth)),
        precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else float('nan'),
        F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        MCC=(tp*tn-fp*fn)/denom if denom else float('nan'),
        AUPRC_AP=float(average_precision_score(truth, scores)) if both else float('nan'),
        AUPRC_status='OK' if both else 'NA', AUPRC_reason=reason,
        MCC_status='OK' if denom else 'NA', threshold=float(threshold))



def edge_scores(coefficients, d, order):
    C, d = np.asarray(coefficients), np.asarray(d)
    n, p = C.shape
    if d.shape[0] != p or order not in (2, 3):
        raise ValueError('Invalid feature dimensions/order')
    keys = [(i, *sources) for i in range(1, n+1)
            for sources in itertools.combinations([j for j in range(1, n+1) if j != i], order-1)]
    scores = dict.fromkeys(keys, 0.)
    for feature, row in enumerate(d):
        sources = tuple(sorted(int(x) for x in row if x != 0))
        if len(sources) != order-1 or len(set(sources)) != len(sources):
            continue
        for target in range(1, n+1):
            if target not in sources:
                key = (target, *sources)
                scores[key] = max(scores[key], abs(float(C[target-1, feature])))
    return keys, np.array([scores[key] for key in keys])

