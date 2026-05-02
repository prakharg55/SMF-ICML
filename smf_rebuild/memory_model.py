from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_layers import HashingMemory


@dataclass
class MemoryConfig:
    layers: tuple[int, ...] = (6, 12, 18)
    mem_n_keys: int = 128
    mem_heads: int = 4
    mem_knn: int = 16
    mem_k_dim: int = 256
    mem_v_dim: int = -1
    swilu_projection: bool = True
    value_fixed_lr: float = 0.001
    mem_share_values: bool = False
    mode: str = "replacement"
    memory_scale_init: float = 0.01

    @property
    def num_slots(self) -> int:
        return self.mem_n_keys * self.mem_n_keys

    @classmethod
    def from_json(cls, path: str | Path) -> "MemoryConfig":
        data = json.loads(Path(path).read_text())
        data["layers"] = tuple(data["layers"])
        return cls(**data)

    def save_json(self, path: str | Path) -> None:
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def parse_layers(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def dtype_from_name(name: str) -> torch.dtype:
    aliases = {
        "auto": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}. Use auto, bf16, fp16, or fp32.") from exc


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


class AdditiveMemoryMLP(nn.Module):
    def __init__(self, base_mlp: nn.Module, memory: HashingMemory, scale_init: float = 0.01):
        super().__init__()
        self.base_mlp = base_mlp
        self.memory = memory
        self.memory_scale = nn.Parameter(torch.tensor(float(scale_init)))

    @property
    def values(self):
        return self.memory.values

    def forward(self, hidden_states):
        memory_out = self.memory(hidden_states).to(hidden_states.dtype)
        return self.base_mlp(hidden_states) + self.memory_scale.to(hidden_states.dtype) * memory_out


def get_memory_module(model, layer_idx: int) -> HashingMemory:
    mlp = model.model.layers[layer_idx].mlp
    return mlp.memory if hasattr(mlp, "memory") else mlp


def _make_memory_layer(hidden_dim: int, memory_config: MemoryConfig) -> HashingMemory:
    memory_layer = HashingMemory(
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        mem_n_keys=memory_config.mem_n_keys,
        mem_heads=memory_config.mem_heads,
        mem_knn=memory_config.mem_knn,
        mem_k_dim=memory_config.mem_k_dim,
        mem_v_dim=memory_config.mem_v_dim,
        swilu_projection=memory_config.swilu_projection,
        value_fixed_lr=memory_config.value_fixed_lr,
        mem_share_values=memory_config.mem_share_values,
    )
    memory_layer.reset_parameters()
    return memory_layer


def patch_qwen_mlp_with_memory(model, memory_config: MemoryConfig) -> None:
    hidden_dim = model.config.hidden_size
    for layer_idx in memory_config.layers:
        memory_layer = _make_memory_layer(hidden_dim, memory_config)
        if memory_config.mode == "replacement":
            model.model.layers[layer_idx].mlp = memory_layer
        elif memory_config.mode == "additive":
            base_mlp = model.model.layers[layer_idx].mlp
            model.model.layers[layer_idx].mlp = AdditiveMemoryMLP(
                base_mlp=base_mlp,
                memory=memory_layer,
                scale_init=memory_config.memory_scale_init,
            )
        else:
            raise ValueError("memory mode must be 'replacement' or 'additive'")


def load_qwen_with_memory(
    model_name_or_path: str,
    memory_config: MemoryConfig,
    dtype: torch.dtype,
    device: str | torch.device | None = None,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    patch_qwen_mlp_with_memory(model, memory_config)
    model.to(dtype=dtype)
    if device is not None:
        model.to(device)
    return model


def load_compatible_state_dict(model, checkpoint_dir: str | Path) -> tuple[list[str], list[str]]:
    checkpoint_dir = Path(checkpoint_dir)
    safetensors_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        state_dict = load_file(str(safetensors_path))
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}")

    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return list(missing), skipped + list(unexpected)


def freeze_for_memory_training(model, layers: Iterable[int], scope: str = "memory") -> None:
    for name, param in model.named_parameters():
        in_memory_layer = False
        in_value_table = False
        for idx in layers:
            mlp = model.model.layers[idx].mlp
            if hasattr(mlp, "memory"):
                memory_prefix = f"model.layers.{idx}.mlp.memory."
                scale_name = f"model.layers.{idx}.mlp.memory_scale"
                in_memory_layer = name.startswith(memory_prefix) or name == scale_name
                in_value_table = name == f"{memory_prefix}values.weight"
            else:
                memory_prefix = f"model.layers.{idx}.mlp."
                in_memory_layer = name.startswith(memory_prefix)
                in_value_table = name == f"{memory_prefix}values.weight"
            if in_memory_layer:
                break
        if scope == "memory":
            param.requires_grad = in_memory_layer
        elif scope == "values":
            param.requires_grad = in_memory_layer and name.endswith("values.weight")
        elif scope == "values_scale":
            param.requires_grad = (in_memory_layer and name.endswith("values.weight")) or name.endswith("memory_scale")
        else:
            raise ValueError("scope must be 'memory', 'values', or 'values_scale'")


def cast_trainable_params(model, dtype: torch.dtype) -> None:
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype)


def count_trainable_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_experiment_metadata(output_dir: str | Path, memory_config: MemoryConfig, base_model: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_config.save_json(output_dir / "memory_config.json")
    (output_dir / "base_model.txt").write_text(base_model + "\n")
