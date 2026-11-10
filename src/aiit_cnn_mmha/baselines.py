"""Reference baselines for comparison with the CNN–MMHA main architecture.

These are trainable model-family baselines, not copies of unpublished conference code.
"""
from __future__ import annotations
import torch
from torch import nn


class MLPBaseline(nn.Module):
    def __init__(self,features=9,hidden=64):
        super().__init__();self.network=nn.Sequential(nn.Linear(features,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self,x):return self.network(x.flatten(0,1).reshape(x.size(0),x.size(1),-1).mean(dim=1)).squeeze(-1)


class RecurrentBaseline(nn.Module):
    def __init__(self,features=9,hidden=64,bidirectional=False):
        super().__init__();self.rnn=nn.LSTM(features,hidden,bidirectional=bidirectional,batch_first=True)
        self.out=nn.Linear(hidden*(2 if bidirectional else 1),1)
    def forward(self,x):
        s,_=self.rnn(x);return self.out(s[:,-1]).squeeze(-1)


class ConvRecurrentBaseline(nn.Module):
    def __init__(self,features=9,hidden=64,bidirectional=False):
        super().__init__();self.conv=nn.Sequential(nn.Conv1d(features,hidden,3,padding=1),nn.ReLU())
        self.rnn=nn.LSTM(hidden,hidden,bidirectional=bidirectional,batch_first=True)
        self.out=nn.Linear(hidden*(2 if bidirectional else 1),1)
    def forward(self,x):
        z=self.conv(x.transpose(1,2)).transpose(1,2)
        z,_=self.rnn(z);return self.out(z[:,-1]).squeeze(-1)


def make_baseline(name,features=9,hidden=64)->nn.Module:
    names={
       'mlp':lambda:MLPBaseline(features,hidden),
       'lstm':lambda:RecurrentBaseline(features,hidden),
       'bilstm':lambda:RecurrentBaseline(features,hidden,bidirectional=True),
       'cnn_lstm':lambda:ConvRecurrentBaseline(features,hidden),
       'cnn_bilstm':lambda:ConvRecurrentBaseline(features,hidden,bidirectional=True),
    }
    if name not in names:raise ValueError(f'unknown baseline: {name}')
    return names[name]()
