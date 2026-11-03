"""Trainable symmetric temporal triad scorer and lightweight architecture control.

This is a new experimental adaptation, not the original NDformer checkpoint.
Neither architecture adds a hand-written third-moment score to its logits.
"""
import itertools
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .common import save,table,digest
from .data import generate


class TriadicModel(nn.Module):
    def __init__(self,architecture='light',d_model=16,nhead=2,layers=1,points=32):
        super().__init__();self.spec=dict(architecture=architecture,d_model=d_model,nhead=nhead,layers=layers,points=points)
        self.phase_embed=nn.Sequential(nn.Linear(2,d_model),nn.GELU(),nn.Linear(d_model,d_model))
        self.response_embed=nn.Sequential(nn.Linear(1,d_model),nn.GELU(),nn.Linear(d_model,d_model))
        if architecture=='transformer':
            layer=nn.TransformerEncoderLayer(d_model,nhead,2*d_model,dropout=0,batch_first=True)
            self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        elif architecture=='light':self.encoder=nn.Identity()
        else:raise ValueError('architecture must be light or transformer')
        self.fuse=nn.Sequential(nn.Linear(4*d_model,d_model),nn.GELU(),nn.Linear(d_model,1))
    def forward(self,phase,response):
        # phase[B,time,3]: target,j,k. Shared node temporal encoder.
        state=torch.stack([phase.sin(),phase.cos()],dim=-1)
        b,t,_,_=state.shape;h=self.phase_embed(state).permute(0,2,1,3).reshape(3*b,t,-1)
        # Fixed positional feature carries temporal order in both architectures.
        pos=torch.linspace(-1,1,t,device=h.device)[None,:,None]
        h=self.encoder(h+.01*pos).reshape(b,3,t,-1)
        a,j,k=h[:,0],h[:,1],h[:,2];r=self.response_embed(response[:,:,None])
        fused=torch.cat([a,j+k,j*k,r],dim=-1)
        return self.fuse(fused).mean(dim=1).squeeze(-1)


def windows(phase,response,groups,target,pairs,points):
    """Use equally spaced positions per whole training trajectory; never mix splits."""
    result=[]
    scale=response.std();response=(response-response.mean())/(scale if scale>1e-14 else 1)
    for ident in np.unique(groups):
        ids=np.flatnonzero(groups==ident);ids=ids[np.linspace(0,len(ids)-1,min(points,len(ids)),dtype=int)]
        for j,k in pairs:
            result.append((phase[ids][:,[target,j,k]],response[ids]))
    return result


@torch.no_grad()
def infer(model,phase,response,groups,target,pairs):
    model.eval();points=model.spec['points'];examples=windows(phase,response,groups,target,pairs,points)
    predictions=[]
    for start in range(0,len(examples),64):
        batch=examples[start:start+64]
        x=torch.from_numpy(np.stack([r[0] for r in batch]).astype('float32'))
        y=torch.from_numpy(np.stack([r[1] for r in batch]).astype('float32'))
        predictions.extend(torch.sigmoid(model(x,y)).cpu().numpy().tolist())
    # windows iterates trajectories first, then the identical candidate list.
    return np.array(predictions).reshape(-1,len(pairs)).mean(axis=0)


def example_set(data,points):
    train=data['splits']=='train';p=data['phase'][train];dy=data['derivative'][train];g=data['groups'][train]
    n=p.shape[1];truth=set(map(tuple,data['net']['triples']));rows=[];labels=[]
    for i in range(n):
        candidates=list(itertools.combinations([j for j in range(n) if j!=i],2))
        items=windows(p,dy[:,i],g,i,candidates,points);rows.extend(items)
        labels.extend([float((i,j,k) in truth) for _ in np.unique(g) for j,k in candidates])
    return np.stack([r[0] for r in rows]).astype('float32'),np.stack([r[1] for r in rows]).astype('float32'),np.array(labels,'float32')


def train(out,cfg,steps=3,sizes=(8,),topologies=2):
    """Fit both architectures; only independent validation topologies select checkpoints."""
    out=Path(out);out.mkdir(parents=True,exist_ok=False);seed=cfg['seeds']['neural'];torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);rng=np.random.default_rng(seed)
    # Disjoint purpose namespaces and explicit graph hashes; benchmark seeds separate.
    sets={'train':[],'validation':[],'test':[]};provenance={}
    for split,offset,num in [('train',10000,topologies),('validation',20000,1),('test',30000,1)]:
        records=[]
        for index in range(num):
            n=sizes[index%len(sizes)];top=seed+offset+index;traj=seed+offset+1000+index
            data=generate(n,top,traj,cfg['data'],[256,256,256]);sets[split].append(example_set(data,cfg['neural']['points']))
            records.append(dict(n=n,topology_seed=top,trajectory_seed=traj,topology_hash=data['diagnostics']['topology_sha256']))
        provenance[split]=records
    hashes=[r['topology_hash'] for records in provenance.values() for r in records]
    assert len(hashes)==len(set(hashes))
    arrays={split:tuple(np.concatenate([v[k] for v in values]) for k in range(3)) for split,values in sets.items()}
    save(out/'topology_split.json',provenance);summaries=[]
    for architecture in ['light','transformer']:
        torch.manual_seed(seed);spec=dict(cfg['neural']);model=TriadicModel(architecture,**{k:spec[k] for k in ['d_model','nhead','layers','points']})
        optimizer=torch.optim.AdamW(model.parameters(),lr=spec['lr']);x,y,label=arrays['train'];positive=float(label.sum());negative=len(label)-positive
        criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative/max(positive,1)))
        history=[];best=float('inf');started=time.perf_counter();checkpoint=out/(architecture+'.pt')
        for step in range(1,steps+1):
            ids=rng.choice(len(x),min(spec['batch_size'],len(x)),replace=False)
            model.train();optimizer.zero_grad();loss=criterion(model(torch.from_numpy(x[ids]),torch.from_numpy(y[ids])),torch.from_numpy(label[ids]))
            loss.backward();optimizer.step();model.eval()
            with torch.no_grad():
                vx,vy,vl=arrays['validation'];vlosses=[]
                for st in range(0,len(vx),128):
                    logits=model(torch.from_numpy(vx[st:st+128]),torch.from_numpy(vy[st:st+128]))
                    # Unweighted validation loss; labels are only validation topology labels.
                    vlosses.append(float(nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(vl[st:st+128]),reduction='sum')))
                val=sum(vlosses)/len(vx)
            history.append(dict(step=step,train_loss=float(loss.detach()),validation_loss=val))
            if val<best:
                best=val;torch.save(dict(state_dict=model.state_dict(),spec=model.spec,step=step,
                    topology_provenance=provenance,seed=seed,smoke_only=steps<=3),checkpoint)
        fitted=load(checkpoint);tx,ty,tl=arrays['test']
        with torch.no_grad():
            prediction=np.concatenate([torch.sigmoid(fitted(torch.from_numpy(tx[st:st+128]),torch.from_numpy(ty[st:st+128]))).numpy() for st in range(0,len(tx),128)])
        from sklearn.metrics import average_precision_score
        summaries.append(dict(architecture=architecture,steps=steps,smoke_only=steps<=3,
            parameters=sum(p.numel() for p in model.parameters()),pretraining_seconds=time.perf_counter()-started,
            best_validation_loss=best,test_AP=float(average_precision_score(tl,prediction)),
            checkpoint=str(checkpoint.name),checkpoint_sha256=digest(checkpoint),device='cpu',
            calibration='sigmoid output, class-weighted training; not calibrated probabilities',
            train_positive=positive,train_negative=negative,positive_weight=negative/max(positive,1)))
        table(out/(architecture+'_history.csv'),history)
    save(out/'summary.json',summaries);return summaries


def load(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=TriadicModel(**checkpoint['spec']);model.load_state_dict(checkpoint['state_dict'])
    model.smoke_only=checkpoint['smoke_only'];model.topology_provenance=checkpoint['topology_provenance']
    return model.eval()
