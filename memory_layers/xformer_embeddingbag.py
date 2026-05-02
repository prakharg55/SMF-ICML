# Copyright (c) Meta Platforms, Inc. and affiliates.
# Simplified for single GPU - no Triton, no distributed training
import torch
import torch.nn as nn
import torch.nn.functional as F


class xFormerEmbeddingBag(nn.Module):
    """
    Simplified embedding bag for single GPU training.
    Uses float32 internally, auto-casts to match input precision.
    """
    def __init__(self, size, dim):
        super().__init__()
        # Use float32 for stability, will auto-cast during forward
        self.weight = nn.Parameter(torch.randn(size, dim))
        self.size = size
        self.dim = dim

    def forward(self, indices, scores):
        """
        Args:
            indices: [batch_size, bag_size] - indices into embedding table
            scores: [batch_size, bag_size] - weights for each embedding
        
        Returns:
            [batch_size, dim] - weighted sum of embeddings
        """
        # Auto-cast weight to match scores dtype (fp16/bf16/fp32)
        weight = self.weight
        if scores.dtype != weight.dtype:
            weight = weight.to(scores.dtype)
        
        output = F.embedding_bag(
            indices, 
            weight, 
            per_sample_weights=scores, 
            mode="sum"
        )
        return output