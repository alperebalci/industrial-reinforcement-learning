"""Learn probabilistic dynamics from fixed logs; compare risk-neutral and risk-aware MPC."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def plant(state,action,demand,noise,efficiency=1.0):
    """Used ONLY for data generation/evaluation, never inside a learned planner."""
    return state+efficiency*(.85+.1*np.tanh(2*state))*action-demand+noise-.03*np.maximum(state,0)


def cost(state,action):
    return .3*action+.5*np.maximum(state,0)+4*np.maximum(-state,0)


def logged_data(seed,n=400):
    g=np.random.default_rng(seed)
    x=g.uniform(-1,2,n);u=g.uniform(0,1,n);d=g.uniform(.2,.8,n)
    X=np.column_stack([x,u,d]);y=plant(x,u,d,g.normal(0,.07,n))-x
    return X,y


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(3,32),nn.Tanh(),nn.Linear(32,2))
    def forward(self,x):
        z=self.net(x);return z[...,0],z[...,1].clamp(-7,2)


def train_ensemble(X,y,seed=0,members=3,epochs=100):
    X=np.asarray(X,float);y=np.asarray(y,float)
    if X.ndim!=2 or X.shape[1]!=3 or y.shape!=(len(X),) or len(X)<2:
        raise ValueError('Need logged (state,action,demand) rows and delta targets')
    if not np.isfinite(X).all() or not np.isfinite(y).all() or min(members,epochs)<1:
        raise ValueError('Invalid data or training budget')
    g=np.random.default_rng(seed);models=[]
    for k in range(members):
        torch.manual_seed(seed+k);m=Dynamics();o=torch.optim.Adam(m.parameters(),lr=.015)
        idx=g.integers(0,len(X),len(X));a=torch.tensor(X[idx],dtype=torch.float32)
        b=torch.tensor(y[idx],dtype=torch.float32)
        for _ in range(epochs):
            mu,lv=m(a);loss=(.5*((b-mu)**2*torch.exp(-lv)+lv)).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite model loss')
            o.zero_grad();loss.backward();o.step()
        m.eval();models.append(m)
    return models


def plan(models,state,forecast,seed=0,risk=0.0,population=48,particles=9,rounds=3):
    if (not models or not np.isfinite(state) or not np.isfinite(forecast).all()
        or len(forecast)<1 or min(population,particles,rounds)<1 or risk<0):
        raise ValueError('Invalid planner arguments')
    g=np.random.default_rng(seed);H=len(forecast)
    mean=np.full(H,.5);sd=np.full(H,.35);best=None
    # Fixed model per particle (TS-infinity); common random numbers across plans.
    member=g.integers(0,len(models),particles)
    eps=g.normal(size=(H,particles))
    for _ in range(rounds):
        U=np.clip(g.normal(mean,sd,(population,H)),0,1)
        S=np.full((population,particles),state);C=np.zeros_like(S)
        with torch.no_grad():
            for t in range(H):
                Z=np.stack([S,np.broadcast_to(U[:,t,None],S.shape),
                            np.full_like(S,forecast[t])],axis=-1)
                a=torch.tensor(Z,dtype=torch.float32);nxt=np.empty_like(S)
                for k,m in enumerate(models):
                    mu,lv=m(a);mu,lv=mu.numpy(),lv.numpy();mask=member==k
                    nxt[:,mask]=S[:,mask]+mu[:,mask]+np.exp(.5*lv[:,mask])*eps[t,mask]
                S=nxt;C+=cost(S,U[:,t,None])
        score=C.mean(axis=1)+risk*C.std(axis=1)
        elite=U[np.argsort(score)[:max(2,population//8)]]
        mean=elite.mean(axis=0);sd=np.maximum(elite.std(axis=0),.05)
        best=U[int(score.argmin())]
    return float(best[0])


def known_model_plan(state,forecast,efficiency=1.0,seed=0,population=48,particles=9,rounds=3):
    """Information-advantaged reference with the SAME continuous CEM search budget."""
    g=np.random.default_rng(seed);H=len(forecast)
    mean=np.full(H,.5);sd=np.full(H,.35);eps=g.normal(0,.07,(H,particles))
    for _ in range(rounds):
        U=np.clip(g.normal(mean,sd,(population,H)),0,1)
        S=np.full((population,particles),state);C=np.zeros_like(S)
        for t,d in enumerate(forecast):
            S=plant(S,U[:,t,None],d,eps[t],efficiency);C+=cost(S,U[:,t,None])
        score=C.mean(axis=1);elite=U[np.argsort(score)[:max(2,population//8)]]
        mean=elite.mean(axis=0);sd=np.maximum(elite.std(axis=0),.05);best=U[score.argmin()]
    return float(best[0])


def evaluate(models,seed=500,episodes=8,horizon=10,efficiency=1.0):
    rows=[];g=np.random.default_rng(seed)
    demands=g.uniform(.3,.7,(episodes,horizon+3));noises=g.normal(0,.07,(episodes,horizon))
    for method in ('base_stock','learned_mean','learned_risk','known_model_mpc'):
        totals=[];stockout=[];actions=[]
        for e in range(episodes):
            s=.4;total=0.;bad=0
            for t in range(horizon):
                f=demands[e,t:t+3]
                if method=='base_stock':u=float(np.clip(f[0]+.4-s,0,1))
                elif method=='known_model_mpc':u=known_model_plan(s,f,efficiency,seed+100*e+t)
                else:u=plan(models,s,f,seed+100*e+t,risk=1. if method=='learned_risk' else 0.)
                s=float(plant(s,u,f[0],noises[e,t],efficiency));total+=cost(s,u);bad+=s<0;actions.append(u)
            totals.append(float(total));stockout.append(bad/horizon)
        rows.append({'method':method,'mean_cost':float(np.mean(totals)),
                     'episode_costs':totals,'stockout_rate':float(np.mean(stockout)),
                     'action_bound_violation':int(sum(not 0<=u<=1 for u in actions))})
    return rows


def benchmark():
    torch.set_num_threads(1);out=[]
    for seed in (7,8,9):
        X,y=logged_data(seed);m=train_ensemble(X,y,seed=seed)
        out.append({'training_seed':seed,'logged_transitions':len(X),
                    'nominal':evaluate(m),'efficiency_shift':evaluate(m,efficiency=.8)})
    return {'scope':'logged-data ensemble MPC; mean+std is NOT a safety certificate',
            'known_model_reference_has_privileged_dynamics':True,
            'sac_comparison':'not implemented; no model-free superiority claim','results':out}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results.json')
    args=p.parse_args();Path(args.output).write_text(json.dumps(benchmark(),indent=2))
