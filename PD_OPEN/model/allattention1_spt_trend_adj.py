import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from lib.utils import norm_Adj
import pdb

def clones(module, N):
    '''
    Produce N identical layers.
    :param module: nn.Module
    :param N: int
    :return: torch.nn.ModuleList
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    '''
    mask out subsequent positions.
    :param size: int
    :return: (1, size, size)
    '''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0   # 1 means reachable; 0 means unreachable


class spatialGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(spatialGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class GCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(GCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, F_in)
        :return: (batch_size, N, F_out)
        '''
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)))  # (N,N)(b,N,in)->(b,N,in)->(b,N,out)


class Spatial_Attention_layer(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        score = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)

        score = self.dropout(F.softmax(score, dim=-1))  # the sum of each row is 1; (b*t, N, N)

        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))


class Spatial_Attention_layer_cross(nn.Module):
    '''
    compute spatial attention scores
    '''
    def __init__(self, dropout=.0):
        super(Spatial_Attention_layer_cross, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x2):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, T, N, N)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)
        x2 = x2.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))

        score = torch.matmul(x, x2.transpose(1, 2)) / math.sqrt(in_channels)  # (b*t, N, F_in)(b*t, F_in, N)=(b*t, N, N)

        score = self.dropout(F.softmax(score, dim=-1))  # the sum of each row is 1; (b*t, N, N)

        return score.reshape((batch_size, num_of_timesteps, num_of_vertices, num_of_vertices))


class spatialAttentionGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''

        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        spatial_attention = self.SAt(x)  # (batch, T, N, N)

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)

        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))
        # (b*t, n, f_in)->(b*t, n, f_out)->(b,t,n,f_out)->(b,n,t,f_out)


class spatialAttentionScaledGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionScaledGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt = Spatial_Attention_layer(dropout=dropout)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        spatial_attention = self.SAt(x) / math.sqrt(in_channels)  # scaled self attention: (batch, T, N, N)

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        # (b, n, t, f)-permute->(b, t, n, f)->(b*t,n,f_in)

        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)

#         # (b*t, n, f_in)->(b*t, n, f_out)->(b,t,n,f_out)->(b,n,t,f_out)
        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))
        # return F.relu(self.Theta(torch.matmul(spatial_attention, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))


class spatialAttentionScaledGCN_cross(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels, dropout=.0):
        super(spatialAttentionScaledGCN_cross, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta1 = nn.Linear(in_channels, out_channels, bias=False)
        self.Theta2 = nn.Linear(in_channels, out_channels, bias=False)
        self.SAt_cross = Spatial_Attention_layer_cross(dropout=dropout)

    def forward(self, x, x2):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        spatial_attention = self.SAt_cross(x, x2) / math.sqrt(in_channels)  # scaled self attention: (batch, T, N, N)

        x_ = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        x2_ = x2.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))
        # (b, n, t, f)-permute->(b, t, n, f)->(b*t,n,f_in)

        spatial_attention = spatial_attention.reshape((-1, num_of_vertices, num_of_vertices))  # (b*T, n, n)
        x = F.relu(self.Theta1(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention), x2_)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))
        x2 = F.relu(self.Theta2(torch.matmul(self.sym_norm_Adj_matrix.mul(spatial_attention.transpose(1, 2)), x_)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))
        # x = F.relu(self.Theta1(torch.matmul(spatial_attention, x2_)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))
        # x2 = F.relu(self.Theta2(torch.matmul(spatial_attention.transpose(1, 2), x_)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))
        # (b*t, n, f_in)->(b*t, n, f_out)->(b,t,n,f_out)->(b,n,t,f_out)

        return x, x2


class MultiHeadAttentionSpatialAttention(nn.Module):  # spatial attention
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttentionSpatialAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :return: (batch, N, T, d_model)
        '''

        nbatches, N, T, d_model = x.shape
        
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0, 4, 1, 3, 2)->(batch, T, h, N, d_k)
#         x_trend = self.conv1Ds_aware_temporal_context(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 4, 1, 3, 2)

        key = self.linears[1](x).view(nbatches, N, -1, self.h, self.d_k).permute(0, 2, 3, 1, 4)
        query = self.linears[2](x).view(nbatches, N, -1, self.h, self.d_k).permute(0, 2, 3, 1, 4)  # (batch, N, T, h, d_k)-> (batch, T, h, N, d_k)

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, T, h, N, d_k)
        value = self.linears[0](x).view(nbatches, N, -1, self.h, self.d_k).permute(0, 2, 3, 1, 4)

        # apply attention on all the projected vectors in batch 
        x, self.attn = attention(key, query, value, dropout=self.dropout)
        # x:(batch, T, h, N, d_k)
        # attn:(batch, T, h, N, N)

        x = x.permute(0, 3, 1, 2, 4).contiguous()  # (batch, N, T, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T, d_model)
        return self.linears[-1](x)
    
    
class MultiHeadAttentionAwareTemporalContex_qc_kc(nn.Module):  # key causal; query causal;
    def __init__(self, nb_head, d_model, kernel_size=3,
                 dropout=.0):
        '''
        :param nb_head:
        :param d_model:
        :param kernel_size:
        :param dropout:
        '''
        super(MultiHeadAttentionAwareTemporalContex_qc_kc, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = kernel_size - 1
        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2)  # # 2 causal conv: 1  for query, 1 for key
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)

        query, key = [
            l(x.permute(0, 3, 1, 2))[:, :, :, :-self.padding].contiguous().view(nbatches, self.h, self.d_k, N,
                                                                                -1).permute(0, 3, 1, 4, 2) for l, x
            in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class MultiHeadAttentionAwareTemporalContex_q1d_k1d(nn.Module):  # 1d conv on query, 1d conv on key
    def __init__(self, nb_head, d_model, kernel_size=3, dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_q1d_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.padding = (kernel_size - 1) // 2

        self.conv1Ds_aware_temporal_context = clones(
            nn.Conv2d(d_model, d_model, (1, kernel_size), padding=(0, self.padding)),
            2)  # # 2 causal conv: 1  for query, 1 for key

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)
        query, key = [
            l(x.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2) for
            l, x in zip(self.conv1Ds_aware_temporal_context, (query, key))]

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class MultiHeadAttentionAwareTemporalContex_qc_k1d(nn.Module):  # query: causal conv; key 1d conv
    def __init__(self, nb_head, d_model, kernel_size=3,
                 dropout=.0):
        super(MultiHeadAttentionAwareTemporalContex_qc_k1d, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 2)  # 2 linear layers: 1  for W^V, 1 for W^O
        self.causal_padding = kernel_size - 1
        self.padding_1D = (kernel_size - 1) // 2
        self.query_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size),
                                                              padding=(0, self.causal_padding))
        self.key_conv1Ds_aware_temporal_context = nn.Conv2d(d_model, d_model, (1, kernel_size),
                                                            padding=(0, self.padding_1D))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask:  (batch, T, T)
        :return: (batch, N, T, d_model)
        '''

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # deal with key and query: temporal conv
        # (batch, N, T, d_model)->permute(0, 3, 1, 2)->(batch, d_model, N, T) -conv->(batch, d_model, N, T)-view->(batch, h, d_k, N, T)-permute(0,3,1,4,2)->(batch, N, h, T, d_k)
        query = self.query_conv1Ds_aware_temporal_context(query.permute(0, 3, 1, 2))[:, :, :,
                :-self.causal_padding].contiguous().view(nbatches, self.h, self.d_k, N, -1).permute(0, 3, 1, 4, 2)
        key = self.key_conv1Ds_aware_temporal_context(key.permute(0, 3, 1, 2)).contiguous().view(nbatches, self.h,
                                                                                                 self.d_k, N,
                                                                                                 -1).permute(0, 3, 1, 4,
                                                                                                             2)

        # deal with value:
        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        value = self.linears[0](value).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3)

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class SpatialPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_of_vertices, dropout):
        super(SpatialPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = torch.nn.Embedding(num_of_vertices, d_model)

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = torch.LongTensor(torch.arange(num_of_vertices)).to(x.device)  # (N,)
        embed = self.embedding(x_indexs).unsqueeze(0)  # (N, d_model)->(1,N,d_model)
        x = x + embed.unsqueeze(2)  # (B, N, T, d_model)+(1, N, 1, d_model)
        return self.dropout(x)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        # computing the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T_max, d_model)
        self.register_buffer('pe', pe)
        # register_buffer:
        # Adds a persistent buffer to the module.
        # This is typically used to register a buffer that should not to be considered a model parameter.

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        if self.lookup_index is not None:
            x = x + self.pe[:, :, self.lookup_index, :]  # (batch_size, N, T, F_in) + (1,1,T,d_model)
        else:
            x = x + self.pe[:, :, :x.size(2), :]

        return self.dropout(x.detach())


class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm
    '''
    def __init__(self, size, dropout, residual_connection, use_LayerNorm):
        super(SublayerConnection, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        '''
        :param x: (batch, N, T, d_model)
        :param sublayer: nn.Module
        :return: (batch, N, T, d_model)
        '''
        if self.residual_connection and self.use_LayerNorm:
            return x + self.dropout(sublayer(self.norm(x)))
        if self.residual_connection and (not self.use_LayerNorm):
            return x + self.dropout(sublayer(x))
        if (not self.residual_connection) and self.use_LayerNorm:
            return self.dropout(sublayer(self.norm(x)))


class SublayerConnection2(nn.Module):
    '''
    A residual connection followed by a layer norm
    '''

    def __init__(self, size, dropout, residual_connection, use_LayerNorm):
        super(SublayerConnection2, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.dropout = nn.Dropout(dropout)
        if self.use_LayerNorm:
            self.norm = nn.LayerNorm(size)
            self.norm2 = nn.LayerNorm(size)

    def forward(self, x, x2, sublayer):
        '''
        :param x: (batch, N, T, d_model)
        :param sublayer: nn.Module
        :return: (batch, N, T, d_model)
        '''
        if self.residual_connection and self.use_LayerNorm:
            x_, x2_ = sublayer(self.norm(x), self.norm2(x2))
            return x + self.dropout(x_), x2 + self.dropout(x2_)
        if self.residual_connection and (not self.use_LayerNorm):
            x_, x2_ = sublayer(x, x2)
            return x + self.dropout(x_), x2 + self.dropout(x2_)
        if (not self.residual_connection) and self.use_LayerNorm:
            x_, x2_ = sublayer(self.norm(x), self.norm2(x2))
            return self.dropout(x_), self.dropout(x2_)


def attention(query, key, value, mask=None, dropout=None):
    '''

    :param query:  (batch, N, h, T1, d_k)
    :param key: (batch, N, h, T2, d_k)
    :param value: (batch, N, h, T2, d_k)
    :param mask: (batch, 1, 1, T2, T2)
    :param dropout:
    :return: (batch, N, h, T1, d_k), (batch, N, h, T1, T2)
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores: (batch, N, h, T1, T2)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)  # -1e9 means attention scores=0
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # p_attn: (batch, N, h, T1, T2)

    return torch.matmul(p_attn, value), p_attn  # (batch, N, h, T1, d_k), (batch, N, h, T1, T2)


class MultiHeadAttention(nn.Module):
    def __init__(self, nb_head, d_model, dropout=.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nb_head == 0
        self.d_k = d_model // nb_head
        self.h = nb_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        :param query: (batch, N, T, d_model)
        :param key: (batch, N, T, d_model)
        :param value: (batch, N, T, d_model)
        :param mask: (batch, T, T)
        :return: x: (batch, N, T, d_model)
        '''
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, T, T), same mask applied to all h heads.

        nbatches = query.size(0)

        N = query.size(1)

        # (batch, N, T, d_model) -linear-> (batch, N, T, d_model) -view-> (batch, N, T, h, d_k) -permute(2,3)-> (batch, N, h, T, d_k)
        query, key, value = [l(x).view(nbatches, N, -1, self.h, self.d_k).transpose(2, 3) for l, x in
                             zip(self.linears, (query, key, value))]

        # apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x:(batch, N, h, T1, d_k)
        # attn:(batch, N, h, T1, T2)

        x = x.transpose(2, 3).contiguous()  # (batch, N, T1, h, d_k)
        x = x.view(nbatches, N, -1, self.h * self.d_k)  # (batch, N, T1, d_model)
        return self.linears[-1](x)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_dense, src_dense2, trg_dense, trg_dense2, generator, generator2, DEVICE):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_dense
        self.src_embed2 = src_dense2
        self.trg_embed = trg_dense
        self.trg_embed2 = trg_dense2
        self.prediction_generator = generator
        self.prediction_generator2 = generator2
        self.to(DEVICE)

    def forward(self, src, trg):
        '''
        src:  (batch_size, N, T_in, F_in)
        trg: (batch, N, T_out, F_out)
        '''
        encoder_output, encoder_output2 = self.encode(
            src)  # (batch_size, N, T_in, d_model) [4, 307, 12, 1]->[4, 307, 12, 64]

        return self.decode(trg, encoder_output, encoder_output2)

    def encode(self, src):
        '''
        src: (batch_size, N, T_in, F_in)
        '''
        h = self.src_embed(src[...,[0]].repeat(1, 1, 1, 64))
        h2 = self.src_embed2(src[...,[1]].repeat(1, 1, 1, 64))
        return self.encoder(h, h2)

    def decode(self, trg, encoder_output, encoder_output2):
        pick, drop = self.decoder(self.trg_embed(trg[...,[0]].repeat(1, 1, 1, 64)), self.trg_embed(trg[...,[1]].repeat(1, 1, 1, 64)), encoder_output, encoder_output2)
        pick = self.prediction_generator(pick)
        drop = self.prediction_generator2(drop)
        return torch.cat((pick, drop), -1)

    
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, gcn, gcn_cross, dropout, residual_connection=True, use_LayerNorm=True):
        super(EncoderLayer, self).__init__()
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        self.self_attn = clones(self_attn, 4)
        self.feed_forward_gcn = clones(gcn, 2)
        self.feed_forward_gcn_cross = gcn_cross
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 8)
        self.sublayer2 = SublayerConnection2(size, dropout, residual_connection, use_LayerNorm)
        self.size = size

    def forward(self, x, x2):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[6](x, self.feed_forward_gcn[0])
            x2 = self.sublayer[7](x2, self.feed_forward_gcn[1])
            x, x2 = self.sublayer2(x, x2, self.feed_forward_gcn_cross)
            x = self.sublayer[2](x, lambda x: self.self_attn[0](x, x2, x2))
            x = self.sublayer[3](x, lambda x: self.self_attn[1](x, x, x))
            x2 = self.sublayer[4](x2, lambda x2: self.self_attn[2](x2, x, x))
            x2 = self.sublayer[5](x2, lambda x2: self.self_attn[3](x2, x2, x2))
            return x, x2
        else:
            x = self.tcn(x)
            x = self.self_attn(x, x, x)
            return self.feed_forward_gcn(x)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        '''
        :param layer:  EncoderLayer
        :param N:  int, number of EncoderLayers
        '''
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        self.norm2 = nn.LayerNorm(layer.size)

    def forward(self, x, x2):
        '''
        :param x: src: (batch_size, N, T_in, F_in)
        :return: (batch_size, N, T_in, F_in)
        '''
        for layer in self.layers:
            x, x2 = layer(x, x2)
        return self.norm(x), self.norm2(x2)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, gcn, gcn_cross, dropout, residual_connection=True, use_LayerNorm=True):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = clones(self_attn, 4)
        self.src_attn = clones(src_attn, 4)
        self.feed_forward_gcn = clones(gcn, 2)
        self.feed_forward_gcn_cross = gcn_cross
        self.residual_connection = residual_connection
        self.use_LayerNorm = use_LayerNorm
        if residual_connection or use_LayerNorm:
            self.sublayer = clones(SublayerConnection(size, dropout, residual_connection, use_LayerNorm), 10)
        self.sublayer2 = SublayerConnection2(size, dropout, residual_connection, use_LayerNorm)

    def forward(self, x, x2, m, m2):
        '''
        :param x: (batch_size, N, T', F_in)
        :param memory: (batch_size, N, T, F_in)
        :return: (batch_size, N, T', F_in)
        '''
        tgt_mask = subsequent_mask(x.size(-2)).to(m.device)  # (1, T', T')
        if self.residual_connection or self.use_LayerNorm:
            x = self.sublayer[8](x, self.feed_forward_gcn[0])
            x2 = self.sublayer[9](x2, self.feed_forward_gcn[1])
            x, x2 = self.sublayer2(x, x2, self.feed_forward_gcn_cross)

            x = self.sublayer[2](x, lambda x: self.self_attn[0](x, x, x, tgt_mask))  # output: (batch, N, T', d_model)
            # x = self.sublayer[3](x, lambda x: self.src_attn[0](x, m2, m2))  # output: (batch, N, T', d_model)
            x = self.sublayer[3](x, lambda x: self.self_attn[2](x, x2, x2, tgt_mask))  # output: (batch, N, T', d_model)
            x = self.sublayer[4](x, lambda x: self.src_attn[1](x, m, m))  # output: (batch, N, T', d_model)

            x2 = self.sublayer[5](x2, lambda x: self.self_attn[1](x, x, x, tgt_mask))  # output: (batch, N, T', d_model)
            x2 = self.sublayer[6](x2, lambda x2: self.self_attn[3](x2, x, x, tgt_mask))  # output: (batch, N, T', d_model)
            # x2 = self.sublayer[6](x2, lambda x: self.src_attn[2](x, m, m))  # output: (batch, N, T', d_model)
            x2 = self.sublayer[7](x2, lambda x2: self.src_attn[3](x2, m2, m2))  # output: (batch, N, T', d_model)

            return x, x2
        else:
            x = self.tcn(x)
            x = self.self_attn(x, x, x, tgt_mask)  # output: (batch, N, T', d_model)
            x = self.src_attn(x, m, m)  # output: (batch, N, T', d_model)
            return self.feed_forward_gcn(x)  # output:  (batch, N, T', d_model)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)
        self.norm2 = nn.LayerNorm(layer.size)

    def forward(self, x, x2, memory, memory2):
        '''

        :param x: (batch, N, T', d_model)
        :param memory: (batch, N, T, d_model)
        :return:(batch, N, T', d_model)
        '''
        for layer in self.layers:
            x, x2 = layer(x, x2, memory, memory2)
        return self.norm(x), self.norm2(x2)


def make_model(DEVICE, num_layers, encoder_input_size, decoder_output_size, d_model, adj_mx, nb_head, num_of_weeks,
               num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=.0, aware_temporal_context=True,
               ScaledSAt=True, SE=True, TE=True, kernel_size=3, smooth_layer_num=0, residual_connection=True,
               use_LayerNorm=True):
    # LR rate means: graph Laplacian Regularization

    c = copy.deepcopy

    adj_mx = np.ones(adj_mx.shape)/adj_mx.shape[0]# + np.eye(adj_mx.shape[0])
#     norm_Adj_matrix = torch.from_numpy(adj_mx).type(torch.FloatTensor).to(DEVICE)
    norm_Adj_matrix = torch.from_numpy(norm_Adj(adj_mx)).type(torch.FloatTensor).to(DEVICE)  # 通过邻接矩阵，构造归一化的拉普拉斯矩阵
    num_of_vertices = norm_Adj_matrix.shape[0]

    attention_gcn = spatialAttentionScaledGCN(norm_Adj_matrix, d_model, d_model)
    attention_gcn_cross = spatialAttentionScaledGCN_cross(norm_Adj_matrix, d_model, d_model)

    # encoder temporal position embedding
    max_len = 12
    h_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    en_lookup_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    print('TemporalPositionalEncoding max_len:', max_len)
    print('h_index:', h_index)
    print('en_lookup_index:', en_lookup_index)

    attn_ss = MultiHeadAttentionAwareTemporalContex_q1d_k1d(nb_head, d_model, kernel_size,
                                                            dropout=dropout)  # encoder的trend-aware attention用一维卷积
    attn_st = MultiHeadAttentionAwareTemporalContex_qc_k1d(nb_head, d_model, kernel_size,
                                                           dropout=dropout)
    attn_tt = MultiHeadAttentionAwareTemporalContex_qc_kc(nb_head, d_model, kernel_size,
                                                         dropout=dropout)  # decoder的trend-aware attention用因果卷积

    encode_temporal_position = TemporalPositionalEncoding(d_model, dropout, max_len, en_lookup_index)  # decoder temporal position embedding
    decode_temporal_position = TemporalPositionalEncoding(d_model, dropout, num_for_predict)
    spatial_position = SpatialPositionalEncoding(d_model, num_of_vertices, dropout)
    encoder_embedding = nn.Sequential(c(encode_temporal_position), c(spatial_position))
    decoder_embedding = nn.Sequential(c(decode_temporal_position), c(spatial_position))

    encoderLayer = EncoderLayer(d_model, attn_ss, c(attention_gcn), c(attention_gcn_cross), dropout,
                                residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)
    encoder = Encoder(encoderLayer, num_layers)

    decoderLayer = DecoderLayer(d_model, attn_tt, attn_st, c(attention_gcn), c(attention_gcn_cross), dropout,
                                residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)
    decoder = Decoder(decoderLayer, num_layers)

    generator = nn.Linear(d_model, decoder_output_size)

    model = EncoderDecoder(encoder,
                          decoder,
                          encoder_embedding,
                          c(encoder_embedding),
                          decoder_embedding,
                          c(decoder_embedding),
                          generator,
                          c(generator),
                          DEVICE)

    # param init
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
