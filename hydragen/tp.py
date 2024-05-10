# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Optional, List, Union

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol

from hydragen.llama import (
    HydragenLlamaAttention,
    LlamaMLP,
    HydragenLlamaModel,
    HydragenLlamaForCausalLM,
)

from hydragen.utils import get_rank, get_world_size

from transformers import LlamaConfig

from accelerate import init_empty_weights

from pathlib import Path


def _apply_tp_linear(
    linear: nn.Linear, style: str, weight_splits: List[int] = []
) -> None:
    rank = get_rank()
    world_size = get_world_size()

    # Linear's weight matrix is transposed, and is of shape
    # (linear.out_features, linear.in_features)
    dim_lookup = {"colwise": (0, "out_features"), "rowwise": (1, "in_features")}
    assert style in dim_lookup
    shard_dim, size_attr = dim_lookup[style]

    # ensure we can shard evenly
    assert getattr(linear, size_attr) % world_size == 0

    def shard(x, dim):
        assert x.size(dim=dim) % world_size == 0
        return torch.tensor_split(x, world_size, dim=dim)[rank].clone()

    def shard_qkv(qkv, dim, weight_splits):
        q, k, v = qkv.split(weight_splits, dim=dim)
        q = shard(q, dim)
        k = shard(k, dim)
        v = shard(v, dim)
        return torch.cat((q, k, v), dim=dim)

    # shard
    if weight_splits:
        # attention
        assert len(weight_splits) == 3

        sharded_weight = shard_qkv(linear.weight, shard_dim, weight_splits)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard_qkv(linear.scales, 0, weight_splits)
    else:
        sharded_weight = shard(linear.weight, shard_dim)
        if hasattr(linear, "scales") and style == "colwise":
            linear.scales = shard(linear.scales, 0)

    linear.weight = nn.Parameter(sharded_weight, requires_grad=False)
    setattr(linear, size_attr, getattr(linear, size_attr) // world_size)


def _apply_tp_ffn(mlp: LlamaMLP) -> None:
    assert hasattr(mlp, "gate_proj")
    assert hasattr(mlp, "up_proj")
    assert hasattr(mlp, "down_proj")

    _apply_tp_linear(mlp.gate_proj, "colwise")
    _apply_tp_linear(mlp.up_proj, "colwise")
    _apply_tp_linear(mlp.down_proj, "rowwise")

    world_size = get_world_size()
    mlp.register_forward_hook(
        lambda _module, _input, output: funcol.all_reduce(
            output, "sum", list(range(world_size))
        )
    )


def _apply_tp_attn(attn: HydragenLlamaAttention) -> None:
    assert hasattr(attn, "q_proj")
    assert hasattr(attn, "k_proj")
    assert hasattr(attn, "v_proj")
    assert hasattr(attn, "o_proj")

    _apply_tp_linear(attn.q_proj, "colwise")
    _apply_tp_linear(attn.k_proj, "colwise")
    _apply_tp_linear(attn.v_proj, "colwise")
    _apply_tp_linear(attn.o_proj, "rowwise")

    # overwrite
    world_size = get_world_size()
    attn.hidden_size = attn.hidden_size // world_size
    attn.num_heads = attn.num_heads // world_size
    attn.head_dim = attn.hidden_size // attn.num_heads
    attn.num_key_value_heads = attn.num_key_value_heads // world_size

    attn.register_forward_hook(
        lambda _module, _input, output: funcol.all_reduce(
            output, "sum", list(range(world_size))
        )
    )


def _apply_tp_Transformer(Transformer: HydragenLlamaModel) -> None:
    # overwrite config before Transformer.setup_cache is called
    world_size = get_world_size()
    Transformer.config.num_attention_heads = (
        Transformer.config.num_attention_heads // world_size
    )
    Transformer.config.num_key_value_heads = (
        Transformer.config.num_key_value_heads // world_size
    )
    Transformer.config.hidden_size = Transformer.config.hidden_size // world_size


def apply_tp(model: HydragenLlamaModel) -> None:
    _apply_tp_Transformer(model)
    for block in model.layers:
        # Apply to MLP
        _apply_tp_ffn(block.mlp)
        _apply_tp_attn(block.self_attn)


def from_pretrained_tp(
    model_name: str, load_dir: Union[Path, str], dtype: Optional[torch.dtype] = None
):
    """
    Loads a `HydragenLlamaForCausalLM` model for inference with tensor parallelism.

    Args:
        model_name: huggingface model to load, must use the `LlamaForCausalLM` class
        load_dir: where to load the model partitions
        dtype: data type for model
    """
    
    config: LlamaConfig = LlamaConfig.from_pretrained(model_name)

    world_size = get_world_size()
    rank = get_rank()

    device = f"cuda:{rank}"

    with init_empty_weights(include_buffers=False):
        model = HydragenLlamaForCausalLM(
            config,
        )
        apply_tp(model.model)

    part_files = list(sorted(Path(load_dir).glob("*.pt")))

    assert len(part_files) == world_size, f"{len(part_files)} != {world_size}"

    file = part_files[rank]
    print(f"Rank {rank} loading {file} (device {device})")

    sd = torch.load(file, map_location=device)

    model.load_state_dict(sd, assign=True)
    model.to(device)
    model.device = device
    if dtype is None or dtype == "auto":
        model.dtype = model.parameters().__next__().dtype
    else:
        model.dtype = dtype
        model.to(dtype=dtype)

    torch.manual_seed(1234)  # make sampling consistent across gpus

    return model
