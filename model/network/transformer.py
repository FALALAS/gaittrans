import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, encoder, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.generator = generator

    def forward(self, src, mask):
        return self.generator(self.encoder(src, mask))


class Generator(nn.Module):
    def __init__(self, d_model, d_out, dropout):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, d_out)

    def forward(self, x):
        output = self.linear(x)
        return F.relu(output)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layer = clones(layer, N)
        self.Norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return self.Norm(x)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(self.norm(sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, pe, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.pe = pe

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda y: self.self_attn(x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, pe, dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.pe = pe
        self.query_linears = nn.Parameter(torch.ones(d_model, self.d_k))
        self.key_linears = nn.Parameter(torch.ones(d_model, self.d_k))

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = x.size(1)
        features = x.split(1, 0)

        outputs = list()
        for feature in features:
            query = feature.squeeze(0)
            query = torch.einsum('nsc, ck-> nsck', query, self.query_linears)
            query = self.pe(query.view(nbatches, -1, self.h, self.d_k, self.d_k).sum(3)).transpose(1, 2)
            key = feature.squeeze(0)
            key = torch.einsum('nsc, ck -> nsck', key, self.key_linears)
            key = self.pe(key.view(nbatches, -1, self.h, self.d_k, self.d_k).sum(3)).transpose(1, 2)
            value = feature.squeeze(0).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            output, self.attn = attention(query, key, value, mask=mask)
            output = output.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
            outputs.append(output.unsqueeze(0))
        next_feature = torch.cat(outputs, 0)
        return next_feature


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, act_layer=nn.GELU):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.act = act_layer()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.drop(self.act(self.w_1(x))))


class PositionEncoding(nn.Module):
    def __init__(self, d_model, h, dropout, max_len=50000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) *
                             -(math.log(10000) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.d_k = d_model // h
        self.pe = pe.view(1, self.max_len, h, self.d_k)

    def forward(self, x):
        device = torch.cuda.current_device()
        self.pe = self.pe.to(device)
        x = x + self.pe[:, :x.size(1)]
        return x


def make_model(N=2, d_model=128, d_ff=256, d_out=256, h=8, dropout=0.4):
    c = copy.deepcopy
    position = PositionEncoding(d_model, h, dropout)
    attn = MultiHeadAttention(h, d_model, position, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = Transformer(
        Encoder(EncoderLayer(d_model, position, c(attn), c(ff), dropout), N),
        Generator(d_model, d_out, dropout)
    )

    return model
