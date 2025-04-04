from abc import abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    checkpoint,
    zero_module,
    normalization,
    count_flops_attn,
    gamma_embedding
)

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class EmbedBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        # Apply the module to 'x' given 'emb' embeddings


class EmbedSequential(nn.Sequential, EmbedBlock):
    #
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    #
    def __init__(self, channels, use_conv, out_channel=None):
        super().__init__()
        