# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from https://github.com/facebookresearch/XLM/blob/main/xlm/model/memory/__init__.py


from .memory import HashingMemory, ProductKeyArgs
from .xformer_embeddingbag import xFormerEmbeddingBag
from .callbacks import MemoryLayerMonitorAndCheckpoint
from .data import load_and_process_dataset
from .data import load_hellaswag_dataset
from .evaluation import ModelEvaluator

__all__ = [
    'HashingMemory', 
    'ProductKeyArgs', 
    'xFormerEmbeddingBag',
    'MemoryLayerMonitorAndCheckpoint',
    'load_and_process_dataset',
    'load_hellaswag_dataset'
    'ModelEvaluator'
]