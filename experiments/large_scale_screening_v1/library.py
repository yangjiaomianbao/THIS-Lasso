"""Sparse native-compatible cubic monomials; one candidate maps to four columns."""
import numpy as np
from .candidates import validate


def lower_terms(n,target):
    terms={()}
    for j in range(n):terms.update([(j,),(j,j),(j,j,j)])
    for j in range(n):
        if j!=target:terms.update([tuple(sorted(t)) for t in [(target,j),(target,target,j),(target,j,j)]])
    return terms


def terms_and_groups(n,target,pairs):
    pairs=validate(pairs,n,target);terms=lower_terms(n,target);groups={}
    for j,k in pairs:
        group=[(j,k),tuple(sorted((j,j,k))),tuple(sorted((j,k,k))),tuple(sorted((target,j,k)))]
        terms.update(group);groups[(int(j),int(k))]=group
    ordered=sorted(terms);lookup={t:i for i,t in enumerate(ordered)}
    groups={p:[lookup[t] for t in ts] for p,ts in groups.items()}
    assert len(ordered)==6*n-2+4*len(pairs)
    return ordered,groups


def evaluate(x,terms):
    theta=np.ones((len(x),len(terms)))
    for col,t in enumerate(terms):
        if t: theta[:,col]=np.prod(x[:,t],axis=1)
    return theta


def native_indices(terms):
    # Native d writes variables in descending order, zero padded on the left.
    return np.array([[0]*(3-len(t))+[j+1 for j in reversed(t)] for t in terms],int)


def decode(coef,terms,target):
    out={2:{},3:{}}
    for value,t in zip(coef,terms):
        if len(t) in (1,2) and len(set(t))==len(t) and target not in t:
            out[len(t)+1][(target,*t)]=abs(float(value))
    return out
