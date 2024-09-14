# MIT License
#
# Copyright (c) 2020 Bin Li
# Copyright (c) 2024 Hossein Jafarinia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
import math

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class BClassifier(nn.Module):
    def __init__(self, encoder, num_classes, input_size: int):
        super(BClassifier, self).__init__()
        self.encoder = encoder
        self.linear = nn.Linear(input_size, num_classes)
        self.feats_size = input_size
        self.num_class = num_classes

    def forward(self, x, c):
        "Pass the input (and mask) through each layer in turn."
        x, attentions = self.encoder(x, c)

        return self.linear(x.mean(dim=1)), attentions


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, c):
        "Pass the input (and mask) through each layer in turn."
        for i, layer in enumerate(self.layers):
            x, attntions = layer(x, c, i)
        return self.norm(x), attntions


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, c, top_big_lambda_indices, random_indices, num_batch, feats_size, num_classes, k,
                mode):
        "Apply residual connection to any sublayer with the same size."
        if mode == 'attn':
            top_big_lambdas = torch.gather(x, 1, top_big_lambda_indices.unsqueeze(-1).expand(-1, -1, feats_size).long())
            random_big_lambda = torch.gather(x, 1, random_indices.unsqueeze(-1).expand(-1, -1, feats_size).long())
            top_big_lambda = torch.cat((top_big_lambdas, random_big_lambda), dim=1)
            multiheadedattn = sublayer(self.norm(x))
            return top_big_lambda + self.dropout(multiheadedattn[0]), multiheadedattn[1]
        elif mode == 'ff':
            return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, num_class, dropout, big_lambda, random_patch_share):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.big_lambda = big_lambda
        self.random_patch_share = random_patch_share
        self.top_big_lambda_share = 1.0 - random_patch_share
        self.num_classes = num_class

    def forward(self, x, c, current_layer):

        "Follow Figure 1 (left) for connections."
        ref_dim = np.inf
        top_big_lambdas_indices_list = []

        _, m_indices = torch.sort(c, 1, descending=True)
        top_big_lambda_share_indices = m_indices[:, 0:math.ceil(self.big_lambda * self.top_big_lambda_share),
                                       :].flatten(start_dim=1, end_dim=-1)
        for i in range(top_big_lambda_share_indices.shape[0]):
            top_big_lambdas_indices_list.append(torch.unique(top_big_lambda_share_indices[i, :]))
            ref_dim = min(ref_dim, len(top_big_lambdas_indices_list[i]))

        ref_dim = min(ref_dim, x.shape[1] - ref_dim)
        topk_indices = torch.zeros(x.shape[0], ref_dim)
        for i in range(x.shape[0]):
            topk_indices[i, :] = top_big_lambdas_indices_list[i][:ref_dim]

        topk_indices = topk_indices.to(device)
        top_big_lambdas = torch.gather(x, 1, topk_indices.unsqueeze(-1).expand(-1, -1, self.size).long())

        random_indices = torch.zeros(x.shape[0], ref_dim)
        for i in range(x.shape[0]):
            remaining_indices = list(set(range(x.shape[1])) - set(top_big_lambdas_indices_list[i].tolist()))
            random_indices[i, :] = torch.from_numpy(
                np.random.choice(remaining_indices, ref_dim, replace=False))

        random_indices = random_indices.to(device)
        random_big_lambda = torch.gather(x, 1, random_indices.unsqueeze(-1).expand(-1, -1, self.size).long())

        top_big_lambda = torch.cat((top_big_lambdas, random_big_lambda), dim=1)

        x_big_lambda, attentions = self.sublayer[0](x, lambda x: self.self_attn(x, top_big_lambda, x), c, topk_indices,
                                                    random_indices, c.shape[1], self.size, self.num_classes,
                                                    self.big_lambda, 'attn')

        selected_indices = torch.cat((topk_indices, random_indices), dim=1)
        y = x.clone()
        y[torch.arange(x.shape[0]).unsqueeze(1), selected_indices.long(), :] = x_big_lambda

        return self.sublayer[1](y, self.feed_forward, c, None, None, c.shape[1], self.size, self.num_classes,
                                self.big_lambda, 'ff'), attentions


def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_big_lambda = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_big_lambda)
    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn.transpose(-2, -1), value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_big_lambda = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Figure 2"
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_big_lambda).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(  # be in topk bedam
            query, key, value, dropout=self.dropout
        )
        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_big_lambda)
        )
        del query
        del key
        del value
        return self.linears[-1](x), self.attn


class PositionwiseFeedForward(nn.Module):  # mikham
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, activation, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        activation_dictionary = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leakyrelu': nn.LeakyReLU(),
            'selu': nn.SELU()
        }
        self.activation = activation_dictionary[activation]

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A = self.b_classifier(feats, classes)

        return classes, prediction_bag, A
