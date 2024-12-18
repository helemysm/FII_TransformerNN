
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os
import time
import tqdm
import pandas as pd



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1) -> None:
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x) -> torch.Tensor:
        print('with PE', x.shape)
        seq_len = x.shape[1]
        x = math.sqrt(self.d_model) * x
        x = x + self.pe[:, :seq_len].requires_grad_(False)
        #return x
        return self.dropout(x)

class PositionalEncodingOld(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1, max_len=5000):
        super(PositionalEncodingOld, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        print('pe in old before unsqueeze ', pe.shape)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        print('pe in old ', pe.shape)
        self.register_buffer('pe', pe)
       
    def forward(self, x):
        print('shape before positional encoding ', x.shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class PositionalEncodingTime(nn.Module):
    def __init__(self, d_model, base=10000.0):
        super(PositionalEncodingTime, self).__init__()
        self.d_model = d_model
        self.base = base

    def forward(self, x, times):
        
        batch_size, seq_len = times.size()
        #frequencies = torch.arange(0, self.d_model, 2, dtype=torch.float32)
        #frequencies = torch.pow(self.base, -frequencies / self.d_model)
        frequencies = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        
        pos_encodings = torch.zeros(batch_size, seq_len, self.d_model)
        pos_encodings[:, :, 0::2] = torch.sin(times.unsqueeze(-1) * frequencies)
        pos_encodings[:, :, 1::2] = torch.cos(times.unsqueeze(-1) * frequencies)

        pos_encodings = pos_encodings.to(x.device)
        print('pos_encodings',pos_encodings.shape)

        encoded_x = x + pos_encodings

        return encoded_x