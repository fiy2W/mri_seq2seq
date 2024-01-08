import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('./src/')

from seq2seq.models.hyperconv import hyperConv
from tsf.models.attention import ChannelAttention, SpatialAttention


class TSF_seq2seq(nn.Module):
    def __init__(self, args):
        super().__init__()

        # define TSF-seq2seq model
        self.ndims = args['seq2seq']['ndims']
        self.feat_len = args['seq2seq']['c_lstm']
        self.c_dec = args['TSF_seq2seq']['c_enc']
        self.k_dec = args['TSF_seq2seq']['k_enc']
        self.s_dec = args['TSF_seq2seq']['s_enc']
        self.seq_in = args['TSF_seq2seq']['c_s_source']
        self.seq_out = args['TSF_seq2seq']['c_s_target']
        self.style_dim = self.seq_in + self.seq_out
        self.weight_dim = args['TSF_seq2seq']['c_w']

        norm = args['TSF_seq2seq']['norm']
        self.norm = getattr(nn, '%s%dd' % (norm, self.ndims)) if norm!='None' else None
        ReflectionPad = getattr(nn, 'ReflectionPad%dd' % self.ndims)

        c_pre = self.seq_in * self.feat_len

        self.fc_p = nn.Sequential(
            nn.Linear(self.style_dim, self.seq_in),
            nn.Sigmoid()
        )
        
        self.pads = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.cas = nn.ModuleList()
        self.sas = nn.ModuleList()
        self.norms = nn.ModuleList()
        for c, k, s in zip(self.c_dec, self.k_dec, self.s_dec):
            self.pads.append(ReflectionPad((k-1)//2))
            self.convs.append(hyperConv(self.style_dim, c_pre, c, ksize=k, stride=s, padding=0, weight_dim=self.weight_dim, ndims=self.ndims))
            self.cas.append(ChannelAttention(c, weight_dim=self.weight_dim, style_dim=self.style_dim))
            self.sas.append(SpatialAttention(weight_dim=self.weight_dim, style_dim=self.style_dim))
            if self.norm is not None:
                self.norms.append(nn.Sequential(self.norm(c), nn.LeakyReLU(0.2, True)))
            else:
                self.norms.append(nn.LeakyReLU(0.2, True))
            c_pre = c

        if 'segmentor' in args:
            self.segmentor = Segmentor2d(args)
        
        if 'classifier' in args:
            self.classifier = Classifier2d(args)
            
    def tsp_attention(self, x, seq_code, skip_attn=False, eps=1e-5):
        res = x
        seq_in = seq_code[:, :self.seq_in].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        param = self.fc_p(seq_code).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + eps # + eps to avoid dividing 0
        x = torch.stack(x.chunk(self.seq_in, dim=1), dim=1)
        x = torch.sum(x * seq_in * param, dim=1) / torch.sum(seq_in*param)

        if skip_attn:
            return x

        for pad, conv, norm, ca, sa in zip(self.pads, self.convs, self.norms, self.cas, self.sas):
            res = norm(conv(pad(res), seq_code))
            res = ca(res, seq_code) * res
            res = sa(res, seq_code) * res
            x = x + res
            res = x

        return x
        
    @torch.no_grad()
    def encode(self, seq2seq, xs, source_seqs):
        x0 = xs[0]
        f0 = seq2seq.enc(x0)

        feats = []
        for i in range(self.seq_in):
            feats.append(torch.zeros_like(f0, device=x0.device))
        
        feats[source_seqs[0]] = f0

        for xi, si in zip(xs[1:], source_seqs[1:]):
            fi = seq2seq.enc(xi)
            feats[si] = fi
        
        feats = torch.cat(feats, dim=2)
        return feats
    
    def forward(self, seq2seq, xs, source_seqs, target_seq, n_outseq=1, task='rec', task_attn=True, skip_attn=False):
        feats = self.encode(seq2seq, xs, source_seqs)

        source_seqs = torch.from_numpy(np.array([1 if i in source_seqs else 0 for i in range(self.seq_in)])).reshape((1,self.seq_in)).to(device=feats.device, dtype=torch.float32)
        target_s = target_seq[:,:self.seq_out]
        if not task_attn:
            target_s = torch.zeros_like(target_s).to(device=target_seq.device)

        seqs = torch.cat([source_seqs, target_s], dim=1)
        feats = self.tsp_attention(feats[:,0], seqs, skip_attn=skip_attn)

        if task=='rec':
            y = seq2seq.dec(xs[0].shape, feats.unsqueeze(1), target_seq, n_outseq=n_outseq)
        elif task=='seg':
            y = self.segmentor(feats)
        elif task=='cls':
            y = self.classifier(feats)
        return y