
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from positional_encodings import PositionalEncoding, PositionalEncodingTime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.multiclass import unique_labels
import numpy as np
import math
import os
import time
import tqdm
import pandas as pd

from tqdm import tqdm


import logging


from cnn_embedding import CNNEmbeddingSimple, CNNEmbeddingBottonUp, CNNEmbeddingTopDown
from cnn_embedding import CNNEmbedding3LayerBottonUp, CNNEmbedding3LayerTopDown, CNNEmbeddingLayerBottonUpPooling




class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

    

class ResidualBlock(nn.Module):
    def __init__(self, layer: nn.Module, embed_dim: int, n_heads, is_att, p=0.1) -> None:
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.isatt = is_att
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_weights = None
        self.n_heads = n_heads

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, seq, padding_value=0, num_heads=1):
        """Create a mask from the input sequence."""
        
        mask = (seq == padding_value).unsqueeze(1).repeat(1, seq.size(1), 1)
        # Repeat the mask for all heads
        mask = mask.repeat(num_heads, 1, 1)
        return mask

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        :param x: [N, seq_len, features]
        :return: [N, seq_len, features]
        """
            
        
        if self.isatt:
            
            batch_size, seq_size, dim = x.size()
            device = x.device
            src =  x.transpose(0, 1)    
            
            if not mask==None:
                """
                add mask ...... 
                """
                masks = self.create_mask(mask, num_heads=self.n_heads).to(device)
                output, attention_weights = self.layer(src, src, src, attn_mask = masks) #with mask
                output = output.transpose(0, 1)     
            else:
                
                output, attention_weights = self.layer(src, src, src) #w/o mask
                output = output.transpose(0, 1)  
                
        else:
            output, attention_weights = self.layer(x)
            
            
        output = self.dropout(output)  # same to x = self.layer_norm1(x + self.dropout(x)) # transformer model -> encoder
        output = self.norm(x + output)
        return output, attention_weights


    
class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation"
    def __init__(self, d_model, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        d_ff=256#5121024
        self.w_1 = nn.Linear(d_model, d_ff)#d_model*2)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = NewGELUActivation()
            
    def forward(self, x):
        x = self.w_1(x)
        x = self.activation(x) # according ViT

        x = self.w_2(x)
        x = self.dropout(x)
        #F.relu(self.w_1(x))#self.dropout(F.relu(self.w_1(x)))

        return x, []


class EncoderBlock(nn.Module):
    def __init__(self, config, embed_dim: int, num_head: int, dropout_rate=0.1) -> None:
        super(EncoderBlock, self).__init__()
        
        self.window_size =   config.modelparam.window_size
        self.num_random =   config.modelparam.num_random
        self.config = config
        
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        
        self.attention = ResidualBlock(
            nn.MultiheadAttention(embed_dim, num_head), embed_dim, self.config.modelparam.n_heads, True, p=dropout_rate)
        
        self.ffn = PositionWiseFeedForward(embed_dim)#ResidualBlock(PositionWiseFeedForward(embed_dim), embed_dim, self.config.modelparam.n_heads, False, p=dropout_rate)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        
        x, attn_weights = self.attention(self.layernorm_1(x), mask)
        #x = x + output
        #return x, attn_weights
        
        #x = self.layer_norm1(x + self.dropout(x)) # transformer model -> encoder
        
        #x_ffn, _ = self.ffn(self.layernorm_2(x)) # or add this for transformer model -> encoder
        x_ffn, _ = self.ffn(x) #remove this in aug 2024
        
        # Skip connection
        x = x + x_ffn
        #x = self.layer_norm2(x + self.dropout(feed_forward))  # transformer model -> encoder
        
        return x, attn_weights
                

class ClassificationModule(nn.Module):
    
    """
    Classification module. This module reduces the input dimensionality
    and outputs the final class predictions.
    """
    
    
    def __init__(self, d_model: int, seq_len: int, factor: int, num_class: int) -> None:
        super(ClassificationModule, self).__init__()
        self.sequence_length = seq_len
        self.d_model = d_model
        self.factor = factor
        self.num_class = num_class

        #self.fc = nn.Linear(int(d_model/2 * self.sequence_length), 1) # for sigmoid
        self.fc = nn.Linear(d_model, num_class) 

        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)        
        x = torch.sigmoid(out)
        
        return x


class EncoderLayerForModel(nn.Module):
    
    """
    Encoder layer for the model, it combines convolutional embedding with positional encoding, 
    and stacked encoder blocks with multi-head attention.

    Parameters:
    -----------
    config : object
        Configuration object containing hyperparameters, including kernel sizes for embeddings
    input_features : int
        Number of input feature dimensions
    seq_len : int
        Length of the input sequence
    n_heads : int
        Number of attention heads in each encoder
    n_layers : int
        Number of encoder blocks to stack.
    d_model : int
        Dimensionality of the model (embedding dimension)
    """
    
    
    def __init__(self, config, input_features, seq_len, n_heads, n_layers, d_model, dropout_rate=0.1) -> None:
        super(EncoderLayerForModel, self).__init__()
        self.d_model = d_model
        self.seq_size = seq_len
        self.config = config
        #self.input_embedding = nn.Conv1d(input_features, d_model, 1)
        self.input_embedding = CNNEmbeddingLayerBottonUpPooling(input_features, d_model, config.embedding.kernel_sizes)
        
        
        self.positional_encoding = PositionalEncoding(d_model, self.seq_size)
            
        self.blocks = nn.ModuleList([
            EncoderBlock(self.config, d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def generate_positional_sequence(self, sz, seq_sz):
        position = torch.arange(0, seq_sz, dtype=torch.int64).unsqueeze(0).repeat(sz, 1)
        return position
 
    def generate_positional_sequence_time(self, sz, seq_sz):
        position = torch.arange(0, seq_sz, dtype=torch.int64).unsqueeze(0).repeat(sz, 1)
        return position

    
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor, mask, x_time) -> torch.Tensor:
        
        x = self.input_embedding(x) # cnn embedding
        x = x.transpose(1, 2)
        
        device = x.device
        batch_size, seq_size, dim = x.size()
        if self.config.type_pe.pe:
            x = self.positional_encoding(x)#, x_time) # for encoder
            
        """
        add mask, and remove if it is not necesa
        """
        
        all_attn_weights = []
        
        for l in self.blocks:
            x , attn_weights= l(x, mask)
            all_attn_weights.append(np.array(attn_weights.cpu().detach().numpy()))

            
        return x, all_attn_weights#attn_weights
    


class MinPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MinPool1d, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        
    def forward(self, x):
        return -self.maxpool(-x)
    
    
    
class model_clf(nn.Module):
    
    """
    Here we define tha class that takes in hyperparameters and produces the full model.
    """
     
    def __init__(
            self, config) -> None:
        super(model_clf, self).__init__()
        
        self.input_features = config.timeseries.in_feature
        self.seq_len = config.timeseries.seq_lenght
        self.n_heads = config.modelparam.n_heads
        self.n_layers = config.modelparam.layers
        self.d_model = config.modelparam.emb_dim
        self.dropout_rate = config.modelparam.dropout_rate
        self.factor = config.modelparam.factor
        self.n_class = config.modelparam.n_class
        self.if_mask = config.modelparam.mask
        
        self.config = config
        self.encoder = EncoderLayerForModel(self.config, self.input_features, self.seq_len, self.n_heads, self.n_layers, self.d_model, self.dropout_rate)
        
        self.pooling_type = self.config.modelparam.pooling_layer
        
        self.create_pooling_layer(self.pooling_type)        
        
        self.clf = ClassificationModule(self.d_model, self.seq_len, self.factor, self.n_class)
    
    #def forward(self, x: torch.Tensor, x_corrected: torch.Tensor) -> torch.Tensor: this is for two inputs
    
    def create_pooling_layer(self, pooling_type):
        kernel_size = 2
        stride=2
        if pooling_type == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size, stride=stride)
        elif pooling_type == 'avg':
            self.pooling_layer = nn.AvgPool1d(kernel_size, stride=stride)
        elif pooling_type == 'min':
            #self.pooling_layer = lambda x: -nn.MaxPool1d(kernel_size, stride=stride)(-x)
            self.pooling_layer = MinPool1d(kernel_size, stride=stride)
        elif pooling_type == 'mean':
            self.pooling_layer = None
            
    def create_input_for_embedding(self, x, x_cen, x_bkg) -> torch.Tensor: #here include background
        
        if self.config.modelparam.centroids and not self.config.modelparam.background:
            x_input = torch.cat([x, x_cen], -1)
            return x_input
        
        if not self.config.modelparam.centroids and self.config.modelparam.background:
            x_input = torch.cat([x, x_bkg], -1) # for now, only centroids instead background... --> change
            return x_input
        
        if self.config.modelparam.centroids and self.config.modelparam.background:
            x_input = torch.cat([x, x_cen, x_bkg], -1) # for now, only centroids... check again when background is ready
            return x_input
        
        if not self.config.modelparam.centroids and not self.config.modelparam.background:
            return x # without changes
        
    
    def forward(self, x: torch.Tensor, x_cen: torch.Tensor, x_bkg: torch.Tensor, x_time, mask=None) -> torch.Tensor: 
        
        device = x.device

        x_temporal = self.create_input_for_embedding(x, x_cen, x_bkg)
        ####x = x.transpose(1, 2)
        x = x_temporal.transpose(1, 2)
        
        x, attn_weights = self.encoder(x, mask, x_time) # b, len, dim256
        
        
        if not self.pooling_layer == None:
            #x = self.pooling_layer(x)
            out = torch.mean(x, dim=1)
        
        else:
            
            if not mask==None:
                # Apply the mask to the data tensor 
                masked_data = x.cpu() * mask.unsqueeze(-1)  # broadcasting the mask to match data_tensor shape
                #=============================================================
                #mean_pooling_value considering the mask values
                out = (torch.sum(masked_data, dim=1) / mask.sum(dim=1, keepdim=True)).to(device)
                #=============================================================
            else:
                out = torch.mean(x, dim=1)
        
        x = self.clf(out)
        
        
        normalized_embeddings = F.normalize(out, p=2, dim=1)
        
        return x, attn_weights, normalized_embeddings.unsqueeze(1)#out.unsqueeze(1)
        