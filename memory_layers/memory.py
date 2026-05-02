# Copyright (c) Meta Platforms, Inc. and affiliates.
# Simplified for single GPU - no distributed training

from logging import getLogger
import math
from typing import Optional
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from memory_layers.xformer_embeddingbag import xFormerEmbeddingBag

logger = getLogger()


@dataclass
class ProductKeyArgs:
    is_enabled: bool = False
    layers: str = ""  # Which layers to have memory (example "6,12,18")
    mem_n_keys: int = 1024
    mem_heads: int = 4
    mem_knn: int = 32
    mem_share_values: bool = True
    mem_k_dim: int = 512
    mem_v_dim: int = -1
    swilu_projection: bool = True
    value_fixed_lr: Optional[float] = 0.001
    mem_gated: bool = False
    peer_variant: bool = False


class HashingMemory(nn.Module):
    """
    Simplified HashingMemory for single GPU training.
    All distributed training code removed.
    """

    VALUES = None
    EVAL_MEMORY = True

    def __init__(
        self,
        input_dim,
        output_dim,
        value_fixed_lr=0.001,
        mem_k_dim=512,
        mem_v_dim=-1,
        mem_heads=4,
        mem_knn=32,
        mem_share_values=True,
        mem_n_keys=1024,
        mem_query_bias=True,
        mem_query_batchnorm=False,
        mem_gated=False,
        mem_input_dropout=0.0,
        mem_query_dropout=0.0,
        mem_value_dropout=0.0,
        peer_variant=False,
        swilu_projection=True,
    ):
        # Check parameters
        assert mem_k_dim >= 2
        assert 0 <= mem_input_dropout < 1
        assert 0 <= mem_query_dropout < 1
        assert 0 <= mem_value_dropout < 1
        assert not (peer_variant and mem_v_dim > 0)
        assert mem_k_dim % 2 == 0
        assert mem_heads >= 2

        super().__init__()
        self.use_peer_variant = peer_variant

        # Global parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.size = mem_n_keys ** 2
        self.k_dim = mem_k_dim
        self.v_dim = mem_v_dim if mem_v_dim > 0 else output_dim

        # Values initialization
        self.swilu_proj = swilu_projection
        self.v_proj = mem_v_dim > 0 or self.swilu_proj
        self.heads = mem_heads
        self.knn = mem_knn

        # Dropout
        self.input_dropout = mem_input_dropout
        self.query_dropout = mem_query_dropout
        self.value_dropout = mem_value_dropout

        # Initialize keys
        self.keys = nn.Parameter(
            torch.empty(2 * self.heads * int(self.size ** 0.5), self.k_dim // 2)
        )

        # Optionally use the same values for all memories
        self.mem_share_values = mem_share_values
        self.original = not self.mem_share_values or HashingMemory.VALUES is None

        # Initialize the values
        if self.original:
            if not self.use_peer_variant:  # PK
                self.values = xFormerEmbeddingBag(self.size, self.v_dim)
                HashingMemory.VALUES = self.values
            else:  # PEER
                self.values_u = nn.Embedding(self.size, self.v_dim)
                self.values_v = nn.Embedding(self.size, self.v_dim)
                HashingMemory.VALUES = self.values_u, self.values_v
        else:
            if not self.use_peer_variant:
                self.values = None
            else:
                self.values_u = None
                self.values_v = None
        
        self.value_fixed_lr = value_fixed_lr

        if self.v_proj:
            proj_input = mem_v_dim
            if self.swilu_proj and proj_input < 0:
                proj_input = output_dim
            self.value_proj = torch.nn.Linear(proj_input, output_dim)
        if self.swilu_proj:
            self.swilu_projection = torch.nn.Linear(self.input_dim, proj_input)
        
        # Gated memory
        self.gating = None
        if mem_gated:
            self.gating = torch.nn.Linear(input_dim, 1)

        # Query network
        l_sizes = (self.input_dim, self.heads * self.k_dim)
        self.query_proj = QueryMLP(
            self.input_dim,
            self.heads,
            self.k_dim,
            l_sizes,
            bias=mem_query_bias,
            batchnorm=mem_query_batchnorm,
        )

    def reset_parameters(self, init_std=None, factor=1.0):
        # Keys
        bound = 1 / math.sqrt(self.k_dim)
        nn.init.uniform_(self.keys, a=-bound, b=bound)
        
        # Values
        if self.original:
            if not self.use_peer_variant:
                nn.init.normal_(self.values.weight, mean=0, std=self.v_dim ** -0.5)
            else:
                nn.init.normal_(self.values_u.weight, mean=0, std=self.v_dim ** -0.5)
                nn.init.normal_(self.values_v.weight, mean=0, std=self.v_dim ** -0.5)
        
        # Queries
        nn.init.xavier_uniform_(self.query_proj.query_mlps[0].weight)
        
        # Value projection
        if self.v_proj:
            nn.init.normal_(self.value_proj.weight, mean=0, std=self.output_dim ** -0.5)
        if self.swilu_proj:
            nn.init.normal_(
                self.swilu_projection.weight, mean=0, std=self.output_dim ** -0.5
            )
        
        # Fixed learning rate
        if self.original:
            if self.use_peer_variant:
                for p in self.values_u.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
                for p in self.values_v.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
            else:
                for p in self.values.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
        
        if self.gating is not None:
            nn.init.normal_(self.gating.weight, mean=0, std=self.input_dim ** -0.5)

    def forward(self, input):
        """Read from the memory."""
        B, T, C = input.shape
        input = input.view(-1, self.input_dim)

        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]

        # Compute query
        bs = np.prod(prefix_shape)
        input = F.dropout(input, p=self.input_dropout, training=self.training)
        query = self.query_proj(input)
        query = F.dropout(query, p=self.query_dropout, training=self.training)
        assert query.shape == (bs * self.heads, self.k_dim)

        # Get indices
        knn = self.knn
        scores, indices = self.get_indices(query, knn)

        # Store indices/scores (eval mode only)
        if not self.training and HashingMemory.EVAL_MEMORY:
            self.last_indices = indices.view(bs, self.heads, knn).detach().cpu()
            self.last_scores = scores.view(bs, self.heads, knn).detach().cpu().float()

        # Re-scoring
        scores = F.softmax(scores.float(), dim=-1).type_as(scores)

        # Merge heads / knn
        indices = indices.view(bs, self.heads * knn)
        scores = scores.view(bs, self.heads * knn)

        if not self.use_peer_variant:
            scores = scores.to(self.values.weight.dtype)
            output = self.values(indices, scores)
            if self.v_proj and not self.swilu_proj:
                if output.dtype != self.value_proj.weight.dtype:
                    output = output.to(self.value_proj.weight.dtype)
                output = self.value_proj(output)
            if self.swilu_proj:
                proj_input = output * F.silu(self.swilu_projection(input))
                if proj_input.dtype != self.value_proj.weight.dtype:
                    proj_input = proj_input.to(self.value_proj.weight.dtype)
                output = self.value_proj(proj_input)
        else:
            u = self.values_u(indices)
            x = torch.einsum("bh, blh->bl", input, u)
            x = F.gelu(x)
            v = self.values_v(indices)
            x = x * scores
            output = torch.einsum("bl, blh->bh", x, v)

        output = F.dropout(output, p=self.value_dropout, training=self.training)

        # Reshape output
        if len(prefix_shape) >= 2:
            output = output.view(prefix_shape + (self.v_dim,))

        if self.gating:
            output = F.sigmoid(self.gating(input)) * output
        
        output = output.view(B, T, -1)
        return output

    def get_indices(self, query, knn):
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = len(query) // self.heads
        query = query.view(-1, self.heads, self.k_dim)
        half = self.k_dim // 2
        
        # Keys: (heads, 2, n_keys, half)
        keys = self.keys.view(self.heads, 2, -1, half)
        keys1 = keys[:, 0, :, :]
        keys2 = keys[:, 1, :, :]
        n_keys = len(keys[0][0])

        # Split query for product quantization
        q1 = query[:, :, :half]
        q2 = query[:, :, half:]

        # Compute indices with associated scores
        scores1 = torch.einsum("blh, lkh->blk", q1, keys1)
        scores2 = torch.einsum("blh, lkh->blk", q2, keys2)

        scores1, indices1 = scores1.topk(knn, dim=2, largest=True)
        scores2, indices2 = scores2.topk(knn, dim=2, largest=True)

        # Cartesian product on best candidate keys
        all_scores = (
            scores1.view(bs, self.heads, knn, 1).expand(bs, self.heads, knn, knn)
            + scores2.view(bs, self.heads, 1, knn).expand(bs, self.heads, knn, knn)
        ).view(bs, self.heads, -1)
        
        all_indices = (
            indices1.view(bs, self.heads, knn, 1).expand(bs, self.heads, knn, knn) * n_keys
            + indices2.view(bs, self.heads, 1, knn).expand(bs, self.heads, knn, knn)
        ).view(bs, self.heads, -1)

        # Select overall best scores and indices
        scores, best_indices = torch.topk(all_scores, k=knn, dim=2, largest=True, sorted=True)
        indices = all_indices.gather(2, best_indices)

        assert scores.shape == indices.shape == (bs, self.heads, knn)
        return scores.view(bs * self.heads, knn), indices.view(bs * self.heads, knn)


class QueryMLP(nn.Module):
    def __init__(self, input_dim, heads, k_dim, sizes, bias=False, batchnorm=False):
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.k_dim = k_dim
        self.sizes = sizes
        assert sizes[0] == input_dim
        assert sizes[-1] == (heads * k_dim)

        sizes_ = list(sizes)
        self.query_mlps = QueryMLP.mlp(sizes_, bias=bias, batchnorm=batchnorm)

    @staticmethod
    def mlp(sizes, bias=True, batchnorm=True):
        """Generate a feedforward neural network."""
        assert len(sizes) >= 2
        pairs = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        layers = []

        for i, (dim_in, dim_out) in enumerate(pairs):
            layers.append(nn.Linear(dim_in, dim_out, bias=bias))
            if batchnorm:
                layers.append(nn.BatchNorm1d(dim_out))
            if i < len(pairs) - 1:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, input):
        """Compute queries."""
        assert input.shape[-1] == self.input_dim
        input = input.contiguous().view(-1, self.input_dim) if input.dim() > 2 else input
        bs = len(input)

        query = self.query_mlps(input)

        assert query.shape == (bs, self.heads * self.k_dim)
        return query.view(bs * self.heads, self.k_dim)
