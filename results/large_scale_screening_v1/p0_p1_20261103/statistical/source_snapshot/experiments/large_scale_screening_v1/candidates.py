"""Label-free canonical candidate streaming and identical shared pools."""
import itertools
import math
import numpy as np


def count(n): return math.comb(n-1,2)


def blocks(n,target,block_size):
    it=itertools.combinations([j for j in range(n) if j!=target],2)
    while True:
        block=list(itertools.islice(it,block_size))
        if not block:return
        yield np.array(block,int)


def validate(pairs,n,target):
    pairs=np.asarray(pairs,int)
    if pairs.ndim!=2 or pairs.shape[1]!=2 or np.any(pairs<0) or np.any(pairs>=n):raise ValueError('Invalid pairs')
    pairs=np.sort(pairs,axis=1)
    if np.any(pairs[:,0]==pairs[:,1]) or np.any(pairs==target): raise ValueError('Target or duplicate source')
    if len(np.unique(pairs,axis=0))!=len(pairs):raise ValueError('Repeated candidates')
    return pairs


def shared_pool(n,target,size,seed):
    total=count(n);size=min(size,total)
    if size==total:return np.concatenate(list(blocks(n,target,max(size,1))))
    rng=np.random.default_rng(np.random.SeedSequence([seed,n,target,44]));seen=set()
    # Uniform exploration is independent of all observed correlations and labels.
    sources=np.array([j for j in range(n) if j!=target])
    while len(seen)<size: seen.add(tuple(sorted(rng.choice(sources,2,replace=False).tolist())))
    return np.array(sorted(seen),int)


def ranked_pairs(pairs,scores):
    if len(pairs)!=len(scores) or not np.isfinite(scores).all(): raise ValueError('Nonfinite/mismatched score')
    return np.lexsort((pairs[:,1],pairs[:,0],-scores))
