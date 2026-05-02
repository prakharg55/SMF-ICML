from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from smf_rebuild.memory_model import get_memory_module


class MemoryAccessLogger:
    """Counts memory value rows touched by a HashingMemory layer."""

    def __init__(self, values_module):
        self.module = values_module
        self.mem_size = values_module.weight.shape[0]
        self.device = values_module.weight.device
        self.batch_counts = torch.zeros(self.mem_size, dtype=torch.long, device=self.device)
        self.handle = self.module.register_forward_hook(self._hook)

    def reset(self) -> None:
        self.batch_counts.zero_()

    def close(self) -> None:
        self.handle.remove()

    def _hook(self, module, inputs, output) -> None:
        indices = inputs[0].reshape(-1).detach()
        with torch.no_grad():
            counts = torch.bincount(indices.to(self.batch_counts.device), minlength=self.mem_size)
            self.batch_counts.add_(counts[: self.mem_size])


@dataclass
class SparseSelectionConfig:
    top_t: int = 256
    scoring: str = "kl"
    background_max_batches: int = 200


class SparseMemoryController:
    def __init__(self, model, layers: Iterable[int], config: SparseSelectionConfig, device: torch.device | str):
        self.model = model
        self.layers = tuple(layers)
        self.config = config
        self.device = torch.device(device)
        self.loggers = {
            idx: MemoryAccessLogger(get_memory_module(model, idx).values)
            for idx in self.layers
        }
        self.mem_sizes = {idx: logger.mem_size for idx, logger in self.loggers.items()}
        self.bg_df = {
            idx: torch.zeros(self.mem_sizes[idx], dtype=torch.long, device=self.device)
            for idx in self.layers
        }
        self.bg_access_counts = {
            idx: torch.zeros(self.mem_sizes[idx], dtype=torch.long, device=self.device)
            for idx in self.layers
        }
        self.bg_num_batches = {idx: 0 for idx in self.layers}
        self.slot_train_masks = {
            idx: torch.zeros(self.mem_sizes[idx], dtype=torch.bool, device=self.device)
            for idx in self.layers
        }
        self._hook_handles = []

    def reset_batch_counts(self) -> None:
        for logger in self.loggers.values():
            logger.reset()

    def close(self) -> None:
        for logger in self.loggers.values():
            logger.close()
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def compute_background_df(self, dataset, data_collator, batch_size: int = 1, num_workers: int = 0) -> None:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=num_workers,
        )
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(loader, desc="Background DF")):
                if step >= self.config.background_max_batches:
                    break
                self.reset_batch_counts()
                batch = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                    if key in {"input_ids", "attention_mask"}
                }
                self.model(**batch)
                for idx, logger in self.loggers.items():
                    self.bg_df[idx].add_((logger.batch_counts > 0).to(self.bg_df[idx].dtype))
                    self.bg_access_counts[idx].add_(logger.batch_counts.to(self.bg_access_counts[idx].dtype))
                    self.bg_num_batches[idx] += 1
        self.model.train(was_training)

    def attach_gradient_hooks(self) -> None:
        for idx in self.layers:
            values_param = get_memory_module(self.model, idx).values.weight
            self._hook_handles.append(values_param.register_hook(self._make_grad_hook(idx)))

    def _make_grad_hook(self, layer_idx: int):
        def hook(grad):
            mask = self.slot_train_masks[layer_idx].to(grad.device)
            return grad * mask.unsqueeze(-1).to(grad.dtype)

        return hook

    def update_masks_from_current_batch(self) -> None:
        for idx in self.layers:
            counts = self.loggers[idx].batch_counts
            total = counts.sum()
            mask = self.slot_train_masks[idx]
            mask.zero_()
            if total == 0:
                continue

            if self.config.scoring == "tfidf":
                tf = counts.float() / total.float()
                df = self.bg_df[idx].float()
                n = float(max(self.bg_num_batches[idx], 1))
                score = tf * torch.log((n + 1.0) / (df + 1.0))
            elif self.config.scoring == "kl":
                p_batch = counts.float() / total.float()
                bg = self.bg_access_counts[idx].float() + 1.0
                p_bg = bg / bg.sum().clamp_min(1.0)
                eps = 1e-8
                score = p_batch * torch.log((p_batch + eps) / (p_bg + eps))
            else:
                raise ValueError("--sparse-scoring must be tfidf or kl")

            score = score.masked_fill(counts == 0, float("-inf"))
            active = int((counts > 0).sum().item())
            k = min(self.config.top_t, active)
            if k > 0:
                _, top_idx = torch.topk(score, k=k)
                mask[top_idx] = True
