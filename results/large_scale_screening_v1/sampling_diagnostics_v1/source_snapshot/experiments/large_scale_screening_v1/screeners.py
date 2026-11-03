"""Six comparable screeners; scores select candidates and are not recovered edges."""
import time
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
from .common import Monitor
from .candidates import validate,ranked_pairs

STAT_METHODS=('pearson','scc','scc_ols','fwl','gc_cmi')


def standardized(x):
    x=np.asarray(x,float);s=x.std(axis=0);s=np.where(s>1e-14,s,1.)
    return (x-x.mean(axis=0))/s


def gaussianize(x): return standardized(ndtri((rankdata(x,axis=0,method='average')-.5)/len(x)))


class Context:
    def __init__(self,x,y):
        started=time.perf_counter();self.x=np.asarray(x,float);self.y=np.asarray(y,float)
        if self.x.ndim!=2 or self.y.shape!=self.x.shape or not np.isfinite(self.x).all() or not np.isfinite(self.y).all():raise ValueError('Finite same-shape [samples,nodes] X,dX required')
        self.z=standardized(x);self.yz=standardized(y);self.median=np.median(self.z,axis=0)
        b=np.column_stack([np.ones(len(x)),self.z]);u,s,_=np.linalg.svd(b,full_matrices=False)
        self.rank=int(np.sum(s>max(b.shape)*np.finfo(float).eps*s[0]));self.q=u[:,:self.rank]
        self.residual=self.y-self.q@(self.q.T@self.y)
        self.rs=standardized(self.residual);self.marginal=np.abs(self.z.T@self.yz/len(x))
        self.seconds=time.perf_counter()-started;self.gc=None
    def copula(self):
        if self.gc is None:
            z=gaussianize(self.z);r=gaussianize(self.residual)
            self.gc=(z.T@z/len(z),z.T@r/len(z))
        return self.gc


def group_stat(x,y,group,kind,min_group):
    n=group.sum(axis=0);safe=np.maximum(n,1)
    sx=np.sum(x*group,axis=0);sy=np.sum(y*group,axis=0)
    cov=np.sum(x*y*group,axis=0)-sx*sy/safe
    vx=np.maximum(np.sum(x*x*group,axis=0)-sx*sx/safe,0)
    vy=np.maximum(np.sum(y*y*group,axis=0)-sy*sy/safe,0)
    denom=np.sqrt(vx*vy) if kind=='corr' else vx
    valid=(n>=min_group)&(denom>1e-12)
    result=np.divide(cov,denom,out=np.zeros(len(n)),where=valid)
    if kind=='corr':result=np.clip(result,-1,1)
    return result,~valid


class Screener:
    def __init__(self,method,params,context=None,model=None,phase=None,groups=None):
        if method not in (*STAT_METHODS,'triadic_ndformer'):raise ValueError('Unknown screener')
        self.method=method;self.params=dict(params);self.context=context;self.model=model;self.phase=phase;self.groups=groups
    def score(self,X,dX,target_node,candidate_pairs):
        context=self.context if self.context is not None else Context(X,dX)
        if X.shape!=context.x.shape or dX.shape!=context.y.shape:raise ValueError('Context shape mismatch')
        pairs=validate(candidate_pairs,X.shape[1],target_node);scores=[];degenerate=0;block=self.params['block_size']
        with Monitor() as monitor:
            for start in range(0,len(pairs),block):
                p=pairs[start:start+block];j,k=p.T;z=context.z
                if self.method=='pearson':
                    value=np.sqrt(context.marginal[j,target_node]*context.marginal[k,target_node])
                elif self.method in ('scc','scc_ols'):
                    kind='corr' if self.method=='scc' else 'slope';parts=[]
                    for source,moderator in [(j,k),(k,j)]:
                        g=z[:,moderator]>context.median[moderator]
                        hi,bad1=group_stat(z[:,source],context.rs[:,[target_node]],g,kind,self.params['min_group'])
                        lo,bad2=group_stat(z[:,source],context.rs[:,[target_node]],~g,kind,self.params['min_group'])
                        # A missing/zero-variance group cannot define a group difference.
                        valid=~(bad1|bad2)
                        parts.append(np.where(valid,abs(hi-lo),0));degenerate+=int(bad1.sum()+bad2.sum())
                    value=.5*(parts[0]+parts[1])
                elif self.method=='fwl':
                    products=z[:,j]*z[:,k];projection=context.q.T@products
                    original_norm=np.linalg.norm(products,axis=0)
                    # Explicit block residual is essential near the projected space:
                    # ||z||²-||Q'z||² loses relative precision by cancellation.
                    products-=context.q@projection
                    norm=np.sum(products*products,axis=0)
                    cross=context.residual[:,target_node]@products
                    valid=norm>(1e-12*np.maximum(original_norm,1.))**2
                    value=np.divide(cross*cross,norm,out=np.zeros(len(p)),where=valid);degenerate+=int(np.sum(~valid))
                elif self.method=='gc_cmi':
                    zz,zr=context.copula();shrink=1-self.params['copula_shrinkage']
                    a=zr[j,target_node]*shrink;b=zr[k,target_node]*shrink;c=zz[j,k]*shrink
                    def mi(rho):return -.5*np.log1p(-np.clip(rho,-1+1e-10,1-1e-10)**2)
                    pcj=(a-b*c)/np.sqrt(np.maximum((1-b*b)*(1-c*c),1e-14))
                    pck=(b-a*c)/np.sqrt(np.maximum((1-a*a)*(1-c*c),1e-14))
                    value=np.maximum(.5*(mi(pcj)-mi(a)+mi(pck)-mi(b)),0)
                else:
                    if self.model is None or self.phase is None:raise ValueError('Neural checkpoint and phases required')
                    from .neural import infer
                    value=infer(self.model,self.phase,dX[:,target_node],self.groups,target_node,p)
                scores.extend(value.tolist())
        scores=np.array(scores);ranking=ranked_pairs(pairs,scores)
        return dict(pairs=pairs,scores=scores,ranking=ranking,method=self.method,
            parameters=self.params,experimental=self.method=='gc_cmi',degenerate_groups_or_products=degenerate,
            **monitor.fields())
