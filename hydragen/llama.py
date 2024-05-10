from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    LlamaConfig,
    LlamaForCausalLM,
)

from einops import rearrange

import torch
from torch import nn, Tensor
import os

from typing import Optional, List, Union

from tqdm import tqdm

from hydragen.attention import (
    hydragen_attention,
)

from hydragen.flash import flash_attention, flash_attention_seqlen

from dataclasses import dataclass
from accelerate import init_empty_weights


def repeat_to_batch_size(tensors: list[Tensor], target_batch_size: int | None = None):
    if target_batch_size is None:
        target_batch_size = max(tensor.shape[0] for tensor in tensors)

    repeated_tensors = []
    for tensor in tensors:
        assert target_batch_size % tensor.shape[0] == 0
        repeated_tensor = tensor.repeat_interleave(
            target_batch_size // tensor.shape[0], dim=0
        )
        repeated_tensors.append(repeated_tensor)

    return repeated_tensors


class HydragenLlamaRotaryEmbedding(LlamaRotaryEmbedding):
    cos_cached: Tensor
    sin_cached: Tensor

    def forward(self, x: Tensor, seq_len=None):
        return (
            self.cos_cached.to(dtype=x.dtype),
            self.sin_cached.to(dtype=x.dtype),
        )


class SharedCache(nn.Module):
    k_cache: Tensor
    v_cache: Tensor
    seq_lens: Tensor
    cumsum_lengths: Tensor

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        k_cache = torch.zeros(
            (max_batch_size * max_seq_length, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        v_cache = torch.zeros_like(k_cache)

        seq_lens = torch.zeros(
            (max_batch_size,), dtype=torch.int32, device=device
        )

        cumsum_lengths = torch.zeros(
            (max_batch_size + 1,), dtype=torch.int32, device=device
        )

        self.register_buffer("k_cache", k_cache, persistent=False)
        self.register_buffer("v_cache", v_cache, persistent=False)
        self.register_buffer("seq_lens", seq_lens, persistent=False)
        self.register_buffer("cumsum_lengths", cumsum_lengths, persistent=False)

        # max batch size and sequence length
        # we can fit in the cache.
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_seq_length

        # the number of sequences currently in the cache
        self.current_batch_size = 0

        # TODO: currently the varlen flash-attn kernel
        # is slower with bs=1 than the non-varlen kernel.
        # therefore, we use the non-varlen kernel whenever
        # possible (i.e. when there is no padding). This involves
        # slicing the varlen KV cache to extract the relevant part,
        # which can lead to CUDA graph invalidations when varlen is
        # off and the length of the shared prompt changes (see GraphedHydragenLlamaModel).
        # It seems like flash-attn has been updated with faster
        # varlen kernels, hopefully we can upgrade versions,
        # always use varlen, and get rid of this extra messiness.
        self.use_varlen = False
        self.sliced_sequence_length = None

    def get_current_batch_size(self):
        return self.current_batch_size

    def fill(self, key_states: Tensor, value_states: Tensor, seq_lens: Tensor):
        bs = key_states.shape[0]

        if bs > self.max_batch_size:
            raise ValueError(
                f"Batch size {bs} exceeds max batch size {self.max_batch_size}"
            )

        if key_states.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {key_states.shape[1]} exceeds max sequence length {self.max_sequence_length}"
            )

        shared_key_states = torch.cat(
            [
                key_state[:sequence_length]
                for key_state, sequence_length in zip(key_states, seq_lens)
            ],
            dim=0,
        )
        shared_value_states = torch.cat(
            [
                value_state[:sequence_length]
                for value_state, sequence_length in zip(value_states, seq_lens)
            ],
            dim=0,
        )

        cumsum_lengths = torch.zeros(
            (bs + 1,), dtype=torch.int32, device=key_states.device
        )
        cumsum_lengths[1 : bs + 1] = seq_lens.cumsum(0).to(torch.int32)

        self.k_cache[: shared_key_states.shape[0]] = shared_key_states
        self.v_cache[: shared_value_states.shape[0]] = shared_value_states
        self.seq_lens[:bs] = seq_lens
        self.cumsum_lengths[: bs + 1] = cumsum_lengths

        self.use_varlen: bool = (
            seq_lens.max().item() != seq_lens.min().item()
        )

        if not self.use_varlen:
            self.sliced_sequence_length = seq_lens[0].item()
        else:
            self.sliced_sequence_length = None

        self.current_batch_size = bs

    def get_used_cumsum_lengths(self):
        return self.cumsum_lengths[: self.current_batch_size + 1]


class PerLayerKVCache(nn.Module):
    per_completion_k_cache: Tensor
    per_completion_v_cache: Tensor

    def __init__(
        self,
        max_unique_batch_size: int,
        max_unique_seq_length: int,
        max_shared_batch_sizes: list[int],
        max_shared_seq_lengths: list[int],
        n_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        per_completion_cache_shape = (
            max_unique_batch_size,
            max_unique_seq_length,
            n_kv_heads,
            head_dim,
        )

        self.register_buffer(
            "per_completion_k_cache",
            torch.zeros(per_completion_cache_shape, dtype=dtype, device=device),
        )

        self.register_buffer(
            "per_completion_v_cache",
            torch.zeros(per_completion_cache_shape, dtype=dtype, device=device),
        )

        self.shared_caches = nn.ModuleList(
            [
                SharedCache(
                    max_batch_size=max_shared_batch_size,
                    max_seq_length=max_shared_seq_length,
                    num_heads=n_kv_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                )
                for max_shared_batch_size, max_shared_seq_length in zip(
                    max_shared_batch_sizes, max_shared_seq_lengths
                )
            ]
        )
        self.num_used_shared_caches = 0

    def empty_shared_cache(self):
        self.truncate_shared_caches(0)

    def get_num_total_shared_caches(self):
        return len(self.shared_caches)

    def truncate_shared_caches(self, new_num_shared_caches: int):
        assert (
            new_num_shared_caches <= self.get_num_total_shared_caches()
        ), f"{new_num_shared_caches} {self.get_num_total_shared_caches()}"

        self.num_used_shared_caches = new_num_shared_caches

    def update_per_completion_kvs(
        self, input_pos: Tensor, k_val: Tensor, v_val: Tensor
    ):
        """
        input_pos: (bs, seq_len)
        k_val: (bs, seq_len, n_heads, head_dim)
        v_val: (bs, seq_len, n_heads, head_dim)
        """
        assert input_pos.shape[1] == k_val.shape[1], f"{input_pos.shape} {k_val.shape}"

        bs, sl, n_heads, head_dim = k_val.shape
        k_out = self.per_completion_k_cache
        v_out = self.per_completion_v_cache

        expanded_input_pos = (
            input_pos[:, :, None, None]
            .expand(bs, -1, n_heads, head_dim)
            .to(torch.int64)
        )
        
        k_out.scatter_(1, expanded_input_pos, k_val)
        v_out.scatter_(1, expanded_input_pos, v_val)

        return (
            k_out[:bs],
            v_out[:bs],
        )

    @torch.no_grad()
    def copy_shared_to_unique(self, total_num_sequences: int):
        assert (
            self.num_used_shared_caches == 1
        ), "Cannot copy shared without exactly one active shared cache"

        shared_cache: SharedCache = self.shared_caches[0]
        shared_batch_size = shared_cache.get_current_batch_size()

        assert total_num_sequences % shared_batch_size == 0

        repetition_factor = total_num_sequences // shared_batch_size

        for i in range(shared_batch_size):
            seq_len = shared_cache.seq_lens[i]

            self.per_completion_k_cache[
                i * repetition_factor : (i + 1) * repetition_factor, :seq_len
            ] = (
                shared_cache.k_cache[
                    shared_cache.cumsum_lengths[i] : shared_cache.cumsum_lengths[i + 1]
                ]
                .unsqueeze(0)
                .repeat_interleave(repetition_factor, 0)
            )

            self.per_completion_v_cache[
                i * repetition_factor : (i + 1) * repetition_factor, :seq_len
            ] = (
                shared_cache.v_cache[
                    shared_cache.cumsum_lengths[i] : shared_cache.cumsum_lengths[i + 1]
                ]
                .unsqueeze(0)
                .repeat_interleave(repetition_factor, 0)
            )

    @torch.no_grad()
    def repeat_per_completion_cache_for_num_samples(
        self, current_size: int, num_samples: int
    ):
        if num_samples == 1:
            return

        self.per_completion_k_cache[: current_size * num_samples] = (
            self.per_completion_k_cache[:current_size].repeat_interleave(num_samples, 0)
        )
        self.per_completion_v_cache[: current_size * num_samples] = (
            self.per_completion_v_cache[:current_size].repeat_interleave(num_samples, 0)
        )

    def get_used_shared_caches(self) -> list[SharedCache]:
        return list(self.shared_caches)[: self.num_used_shared_caches]

    def get_shared_len(self, final_batch_size: int):
        if self.num_used_shared_caches == 0:
            return torch.zeros(
                (final_batch_size,), dtype=torch.long, device=self.per_completion_k_cache.device
            )
        else:
            used_caches = self.get_used_shared_caches()
            lens = []
            for shared_cache in used_caches:
                shared_cache: SharedCache
                lens.append(shared_cache.seq_lens)

            repeated = repeat_to_batch_size(lens, final_batch_size)
            return sum(repeated)

    def has_shared(self):
        return self.num_used_shared_caches > 0

    def append_shared(
        self, key_states: Tensor, value_states: Tensor, seq_lens: Tensor
    ):
        if self.num_used_shared_caches >= self.get_num_total_shared_caches():
            raise ValueError(
                f"No more available shared caches: {self.num_used_shared_caches} {self.get_num_total_shared_caches()}"
            )

        next_cache: SharedCache = self.shared_caches[self.num_used_shared_caches]
        next_cache.fill(key_states, value_states, seq_lens)

        self.num_used_shared_caches += 1


class AttentionMode:
    SHARED_PREFILL = "shared-prefill"
    UNIQUE_PREFILL = "unique-prefill"
    DECODE = "decode"


def hydragen_attention_on_caches(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    shared_caches: list[SharedCache],
    seq_len: Optional[Tensor] = None,
):

    keys = []
    values = []
    cu_seqlens = []
    max_seqlens = []
    use_varlens = []

    for shared_cache in shared_caches:

        if shared_cache.use_varlen:
            keys.append(shared_cache.k_cache)
            values.append(shared_cache.v_cache)
            cu_seqlens.append(shared_cache.get_used_cumsum_lengths())
            max_seqlens.append(shared_cache.max_sequence_length)
        else:
            slice_to = (
                shared_cache.get_current_batch_size()
                * shared_cache.sliced_sequence_length
            )
            sliced_keys = rearrange(
                shared_cache.k_cache[:slice_to],
                "(b s) h d -> b s h d",
                b=shared_cache.get_current_batch_size(),
                s=shared_cache.sliced_sequence_length,
            )
            sliced_values = rearrange(
                shared_cache.v_cache[:slice_to],
                "(b s) h d -> b s h d",
                b=shared_cache.get_current_batch_size(),
                s=shared_cache.sliced_sequence_length,
            )
            assert sliced_keys.is_contiguous()
            assert sliced_values.is_contiguous()

            keys.append(sliced_keys)
            values.append(sliced_values)

            cu_seqlens.append(None)
            max_seqlens.append(None)

        use_varlens.append(shared_cache.use_varlen)

    return hydragen_attention(
        q,
        k,
        v,
        shared_ks=keys,
        shared_vs=values,
        shared_cu_seq_lens=cu_seqlens,
        shared_max_seq_lens=max_seqlens,
        use_varlens=use_varlens,
        seq_lens=seq_len,
    )


class HydragenLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.disable_hydragen = False

        # skips attention entirely -
        # not functionally correct but
        # useful for calculating a throughput
        # upper bound.
        self.disable_attention = False

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # Configured by upper level modules
        self.kv_cache: Optional[PerLayerKVCache] = None
        self.mode: Optional[str] = None
        self.rotary_emb: Optional[HydragenLlamaRotaryEmbedding] = None

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
    ):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        cos, sin = self.rotary_emb(value_states)

        if self.disable_hydragen:
            unique_position_ids = position_ids
        else:
            unique_position_ids = position_ids - self.kv_cache.get_shared_len(
                position_ids.shape[0]
            ).unsqueeze(-1)

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            position_ids=position_ids,
            unsqueeze_dim=2,  # unsqueeze dim = head dim on q/k
        )

        if self.disable_attention:
            attn_output = query_states

        elif self.mode == AttentionMode.SHARED_PREFILL:
            if not self.kv_cache.has_shared():

                attn_output, _ = flash_attention(
                    query_states, key_states, value_states, causal=True
                )

            else:
                shared_caches = self.kv_cache.get_used_shared_caches()

                attn_output = hydragen_attention_on_caches(
                    query_states,
                    key_states,
                    value_states,
                    shared_caches,
                )

            self.kv_cache.append_shared(
                key_states, value_states, unique_position_ids.max(1).values + 1
            )

        elif self.mode == AttentionMode.UNIQUE_PREFILL:
            if self.disable_hydragen:

                key_states, value_states = self.kv_cache.update_per_completion_kvs(
                    unique_position_ids, key_states, value_states
                )

                # Note the slicing is needed so that there is no zero padding for the
                # decoded tokens in the key/value status. This will not work for shared
                # caches with batch size > 1 where the sequence length differs.
                attn_output, _ = flash_attention(
                    query_states,
                    key_states[:, : (unique_position_ids.max().item() + 1)],
                    value_states[:, : (unique_position_ids.max().item() + 1)],
                    causal=True,
                )

            else:
                if not self.kv_cache.has_shared():
                    attn_output, _ = flash_attention(
                        query_states, key_states, value_states, causal=True
                    )

                else:
                    shared_caches = self.kv_cache.get_used_shared_caches()

                    attn_output = hydragen_attention_on_caches(
                        query_states,
                        key_states,
                        value_states,
                        shared_caches,
                    )

                key_states, value_states = self.kv_cache.update_per_completion_kvs(
                    unique_position_ids, key_states, value_states
                )

        elif self.mode == AttentionMode.DECODE:
            key_states, value_states = self.kv_cache.update_per_completion_kvs(
                unique_position_ids, key_states, value_states
            )

            seq_lens = unique_position_ids.squeeze(-1) + 1

            if not self.kv_cache.has_shared() or self.disable_hydragen:
                attn_output, _ = flash_attention_seqlen(
                    query_states,
                    key_states,
                    value_states,
                    seq_len=seq_lens,
                )
            else:
                shared_caches = self.kv_cache.get_used_shared_caches()

                attn_output = hydragen_attention_on_caches(
                    query_states,
                    key_states,
                    value_states,
                    shared_caches,
                    seq_len=seq_lens,
                )

        else:
            raise ValueError(f"Unknown mode {self.mode}")

        attn_output = rearrange(attn_output, "b n h d -> b n (h d)")

        attn_output = self.o_proj(attn_output)
        return attn_output


class HydragenLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.self_attn = HydragenLlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class HydragenLlamaModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id if config.pad_token_id is not None else 0
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [HydragenLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self._init_rope()

        for layer in self.layers:
            layer: HydragenLlamaDecoderLayer
            layer.self_attn.rotary_emb = self.rotary_emb

    def set_disable_hydragen(self, disable: bool = True):
        for layer in self.layers:
            layer: HydragenLlamaDecoderLayer
            layer.self_attn.disable_hydragen = disable

    def get_disable_hydragen(self):
        first_layer: HydragenLlamaDecoderLayer = self.layers[0]
        return first_layer.self_attn.disable_hydragen

    def set_disable_attention(self, disable: bool = True):
        for layer in self.layers:
            layer: HydragenLlamaDecoderLayer
            layer.self_attn.disable_attention = disable

    def get_disable_attention(self):
        first_layer: HydragenLlamaDecoderLayer = self.layers[0]
        return first_layer.self_attn.disable_attention

    def copy_shared_cache_to_unique(self, total_num_sequences: int):
        for layer in self.layers:
            layer: HydragenLlamaDecoderLayer
            layer.self_attn.kv_cache.copy_shared_to_unique(total_num_sequences)

    def get_shared_batch_sizes(self):
        first_layer: HydragenLlamaDecoderLayer = self.layers[0]
        batch_sizes = []
        for shared_cache in first_layer.self_attn.kv_cache.get_used_shared_caches():
            shared_cache: SharedCache
            batch_sizes.append(shared_cache.get_current_batch_size())

        return batch_sizes

    def get_shared_varlens(self):
        first_layer: HydragenLlamaDecoderLayer = self.layers[0]
        varlens = []
        for shared_cache in first_layer.self_attn.kv_cache.get_used_shared_caches():
            shared_cache: SharedCache
            varlens.append(shared_cache.use_varlen)

        return varlens

    def get_shared_slice_seq_lens(self):
        first_layer: HydragenLlamaDecoderLayer = self.layers[0]
        slice_lengths = []
        for shared_cache in first_layer.self_attn.kv_cache.get_used_shared_caches():
            shared_cache: SharedCache
            slice_lengths.append(shared_cache.sliced_sequence_length)

        return slice_lengths

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = HydragenLlamaRotaryEmbedding(
                self.config.hidden_size // self.config.num_attention_heads,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.config.hidden_size // self.config.num_attention_heads,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.config.hidden_size // self.config.num_attention_heads,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
    ):
        """
        Unique shape: [layers, 2 batch, seq, heads, dim]
        Shared shape: [layers, 2, seq, heads, dim]
        """
        inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


@dataclass
class CaptureData:
    graph: torch.cuda.CUDAGraph
    static_input_ids: Tensor
    static_position_ids: Tensor
    static_hidden: Tensor
    captured_shared_batch_sizes: list[int]
    captured_use_varlens: list[bool]
    captured_sliced_seq_lens: list[int | None]
    captured_disable_hydragen: bool
    captured_disable_attention: bool


class GraphedHydragenLlamaModel(nn.Module):
    def __init__(self, model: HydragenLlamaModel):
        super().__init__()
        self.model = model

        self.capture_data: Optional[CaptureData] = None

    def invalidate(self):
        self.capture_data = None

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
    ):

        if self.capture_data is not None and (
            input_ids.shape != self.capture_data.static_input_ids.shape
            or position_ids.shape != self.capture_data.static_position_ids.shape
            or self.model.get_shared_batch_sizes()
            != self.capture_data.captured_shared_batch_sizes
            or self.model.get_disable_hydragen()
            != self.capture_data.captured_disable_hydragen
            or self.model.get_disable_attention()
            != self.capture_data.captured_disable_attention
            or self.model.get_shared_varlens() != self.capture_data.captured_use_varlens
            or self.model.get_shared_slice_seq_lens()
            != self.capture_data.captured_sliced_seq_lens
        ):
            self.invalidate()

        if self.capture_data is None:
            self.capture(
                input_ids=input_ids,
                position_ids=position_ids,
            )

        self.capture_data.static_input_ids.copy_(input_ids)
        self.capture_data.static_position_ids.copy_(position_ids)

        self.capture_data.graph.replay()

        return self.capture_data.static_hidden

    def capture(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
    ):
        """
        Following the procedure:
        https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
        """
        static_input_ids = input_ids.clone()
        static_position_ids = position_ids.clone()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            # warmup
            for _ in tqdm(range(3)):
                self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                )

        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_hidden = self.model(
                input_ids=static_input_ids,
                position_ids=static_position_ids,
            )

        self.capture_data = CaptureData(
            graph=g,
            static_input_ids=static_input_ids,
            static_position_ids=static_position_ids,
            static_hidden=static_hidden,
            captured_shared_batch_sizes=self.model.get_shared_batch_sizes(),
            captured_disable_hydragen=self.model.get_disable_hydragen(),
            captured_disable_attention=self.model.get_disable_attention(),
            captured_use_varlens=self.model.get_shared_varlens(),
            captured_sliced_seq_lens=self.model.get_shared_slice_seq_lens(),
        )


class SharedCacheOp:
    WIPE = "wipe"
    EXTEND = "extend"
    PRESERVE = "preserve"


class HydragenLlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ):
        super().__init__()
        self.config = config

        self.model = HydragenLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.kv_cache_allocated: bool = False

        self.graphed_model: Optional[GraphedHydragenLlamaModel] = None
        self.mode: Optional[str] = None

    def set_mode(self, mode):
        self.mode = mode
        for module in self.modules():
            if isinstance(module, HydragenLlamaAttention):
                module.mode = mode

    def graph(self, do_graph: bool = True):
        """
        Controls whether to use CUDA graphs during decoding.
        """

        if do_graph:
            if self.graphed_model is None:
                self.graphed_model = GraphedHydragenLlamaModel(self.model)
        else:
            self.graphed_model = None

    def maybe_invalidate(self):
        if self.graphed_model is not None:
            self.graphed_model.invalidate()

    def get_num_heads(self):
        if hasattr(self.config, "num_heads"):
            return self.config.num_heads
        elif hasattr(self.config, "num_attention_heads"):
            return self.config.num_attention_heads
        else:
            raise ValueError("config needs to specify num heads")

    def setup_caches(
        self,
        max_unique_batch_size: int,
        max_unique_seq_length: int,
        max_shared_batch_sizes: list[int],
        max_shared_seq_lengths: list[int],
    ):
        """
        Allocates the non-shared KV cache at every layer.

        Args:
            max_batch_size: The maximum batch size that the model will be run with.
            max_unique_seq_length: The maximum number of tokens that will be unique per sequence. This includes any generated tokens as well as prompts that are not shared across multiple sequences. Shared prompts do not count against this length.
        """

        # changing the kv cache objects invalidates the graph
        self.maybe_invalidate()

        # round up to nearest 16
        max_unique_seq_length = (max_unique_seq_length + 15) // 16 * 16

        for layer in self.model.layers:
            layer: HydragenLlamaDecoderLayer
            layer.self_attn.kv_cache = PerLayerKVCache(
                max_unique_batch_size=max_unique_batch_size,
                max_unique_seq_length=max_unique_seq_length,
                max_shared_batch_sizes=max_shared_batch_sizes,
                max_shared_seq_lengths=max_shared_seq_lengths,
                n_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.hidden_size // self.get_num_heads(),
                device=self.lm_head.weight.device,
                dtype=self.lm_head.weight.dtype,
            )

        self.kv_cache_allocated = True

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        seq_lens: Tensor | None = None,
        use_graph: bool = False,
        full_logits: bool = False,
    ):
        if use_graph:
            assert self.graphed_model is not None
            model = self.graphed_model
        else:
            model = self.model

        hidden_states = model(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        # generally we don't want to apply the LM head
        # to every token in the sequence during prefill, since it uses
        # lots of memory (and is wasted compute since we only use
        # the last token's logits)
        
        if full_logits:
            to_lm_head = hidden_states
        elif seq_lens is not None:
            # TODO replace with torch.gather?
            to_lm_head = hidden_states[
                torch.arange(hidden_states.shape[0], device=hidden_states.device),
                seq_lens - 1,
            ].unsqueeze(1)
        else:
            to_lm_head = hidden_states[:, -1:]

        logits = self.lm_head(to_lm_head)

        # hf does this
        logits = logits.float()

        return logits

    def apply_top_p(
        self,
        logits: torch.FloatTensor,
        top_p: float,
        min_tokens_to_keep: int = 1,
        filter_value: float = -float("Inf"),
    ) -> torch.FloatTensor:
        """
        Modified from HF TopPLogitsWarper
        """

        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        filled_logits = logits.masked_fill(indices_to_remove, filter_value)
        return filled_logits

    def sample_from_logits(
        self,
        logits: Tensor,
        temperature: float,
        num_samples: int = 1,
        top_p: Optional[float] = None,
    ):
        if top_p is not None:
            logits = self.apply_top_p(logits, top_p)

        if temperature == 0:
            assert logits.ndim == 2
            next_tokens = logits.argmax(dim=-1, keepdim=True).repeat_interleave(
                num_samples, dim=-1
            )
        else:
            probs = nn.functional.softmax(logits / temperature, dim=-1)
            next_tokens = torch.multinomial(
                probs, num_samples=num_samples, replacement=True
            )

        return next_tokens

    def empty_shared_cache(self):
        for layer in self.model.layers:
            layer: HydragenLlamaDecoderLayer
            layer.self_attn.kv_cache.empty_shared_cache()

    def truncate_shared_caches(self, new_num_shared_caches: int):
        """
        Removes levels of shared caches.

        Args:
            new_num_shared_caches: The number of shared caches to keep. If this is 0, all shared caches are removed.
        """

        for layer in self.model.layers:
            layer: HydragenLlamaDecoderLayer
            layer.self_attn.kv_cache.truncate_shared_caches(new_num_shared_caches)

    def get_shared_cache_len(self, batch_size):
        first_layer: HydragenLlamaDecoderLayer = self.model.layers[0]
        return first_layer.self_attn.kv_cache.get_shared_len(batch_size)

    def get_num_used_shared_caches(self):
        first_layer: HydragenLlamaDecoderLayer = self.model.layers[0]
        return first_layer.self_attn.kv_cache.num_used_shared_caches

    @torch.no_grad()
    def append_shared(
        self, input_ids: Tensor, seq_lens: Optional[Tensor] = None, full_logits: Optional[bool] = False
    ) -> Tensor:
        """
        Adds a new level of shared cache to the model.

        Args:
            input_ids: The tokens ids for the new shared prompt. Any padding must be done on the right.

            seq_lens: The true length of each sequence in input_ids. If None, assumes no padding.
        """

        self.set_mode(AttentionMode.SHARED_PREFILL)

        input_seq_len = input_ids.shape[1]
        shared_lens = self.get_shared_cache_len(input_ids.shape[0])

        position_ids = torch.stack(
            [
                torch.arange(
                    shared_len,
                    shared_len + input_seq_len,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                for shared_len in shared_lens
            ]
        ).repeat(input_ids.shape[0] // len(shared_lens), 1)

        if seq_lens is not None:
            for batch_idx, sequence_length in enumerate(seq_lens):
                position_ids[batch_idx, sequence_length:] = position_ids[
                    batch_idx, sequence_length - 1
                ]

        out = self(
            input_ids=input_ids,
            position_ids=position_ids,
            seq_lens=seq_lens,
            full_logits=full_logits,
        )

        return out

    @torch.no_grad()
    def process_unique(
        self,
        input_ids: Tensor,
        seq_lens: Tensor | None = None,
    ) -> Tensor:
        self.set_mode(AttentionMode.UNIQUE_PREFILL)

        input_seq_len = input_ids.shape[1]
        shared_lens = self.get_shared_cache_len(input_ids.shape[0])

        position_ids = torch.stack(
            [
                torch.arange(
                    shared_len,
                    shared_len + input_seq_len,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                for shared_len in shared_lens
            ]
        ).repeat(input_ids.shape[0] // len(shared_lens), 1)

        return self(
            input_ids=input_ids,
            position_ids=position_ids,
            seq_lens=seq_lens,
        )

    def repeat_per_completion_cache_for_num_samples(
        self, current_size: int, num_samples: int
    ):
        for layer in self.model.layers:
            layer: HydragenLlamaDecoderLayer
            layer.self_attn.kv_cache.repeat_per_completion_cache_for_num_samples(
                current_size, num_samples
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[Union[Tensor, list[Tensor]]] = None,
        seq_lens: Optional[Union[Tensor, list[Tensor]]] = None,
        starting_logits: Optional[Tensor] = None,
        num_return_sequences: int = 1,
        max_new_tokens: int = 5,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        return_logits: bool = False,
        shared_cache_op: str = SharedCacheOp.PRESERVE,
        disable_hydragen: bool = False,
        disable_attention: bool = False,
        disable_hierarchy: bool = False,
        token_overrides: Optional[Tensor] = None,
    ):
        """
        Generates text from the model.

        Args:
            input_ids: Either a single tensor of shape [batch_size, seq_length], or a list of such tensors. If a list of tensors is provided (corresponding to a hierarchy of prompts), each id tensor can have a unique batch size, however all id batch sizes must divide the batch size of the final tensor. Can be None if starting_logits is provided. Any padding must be done on the right hand side of each id tensor, and seq_lens must be provided to indicate the true length of each sequence.

            seq_lens: Either a single integer tensor of shape [batch_size], or a list of such tensors (must match the structure of input_ids). Can be None, corresponding to no padding in the input_ids tensors.

            starting_logits: A tensor of shape [batch_size, vocab_size], corresponding to the logits that should be used for sampling the first token. This can be useful if one wants to make multiple calls to generate without re-processing the same prompt.

            num_return_sequences: The number of completions to generate for each input.

            max_new_tokens: The maximum number of tokens to generate for each completion.

            temperature: The temperature to use for sampling from the logits.

            top_p: The top-p value to use for sampling from the logits.

            eos_token_id: The token id to use as the end of sequence token. If None, no end of sequence token is used.

            return_logits: If True, returns the logits for each token generated.

            shared_cache_op: Controls how the shared cache should be modified before and after generation. Can be one of:
                - "wipe": clears any existing shared cache before generation.
                - "preserve": keeps any existing shared cache levels and removes any new cache levels added during generation.
                - "extend": removes nothing, keeping any existing shared cache levels and leaving any new cache levels added during generation.


        Testing / Benchmarking Args:

            disable_hydragen: Controls disabling Hydragen attention entirely.

            disable_attention: Skips all attention operations.

            disable_hierarchy: Restricts Hydragen attention to a single prefix-suffix decomposition (i.e. no hierarchy).

            token_overrides: A tensor of shape [batch_size, max_new_tokens], corresponding to the token ids that should be chosen for each position in the generated sequence. This can be useful for testing against a reference implementation, to avoid the issue where small numerical differences can cause implementations to choose different tokens, causing a cascade of differences in the generated sequence.
        """
        assert self.kv_cache_allocated
        assert (input_ids is None) or (starting_logits is None)
        assert not (input_ids is None and starting_logits is None)

        if input_ids is None:
            input_ids = []

        if temperature < 0:
            raise ValueError(
                f"temperature must be non-negative, {temperature} is invalid"
            )

        if disable_attention:
            self.model.set_disable_attention(True)

        if shared_cache_op == SharedCacheOp.WIPE:
            self.empty_shared_cache()

        og_num_shared_caches = self.get_num_used_shared_caches()

        if isinstance(input_ids, Tensor):
            input_ids = [input_ids]

        num_new_levels = len(input_ids)
        if num_return_sequences > 1:
            num_new_levels += 1

        total_levels = og_num_shared_caches + num_new_levels

        # limited support for the baselines we run in the paper
        if disable_hydragen:
            # For the FlashAttention baseline
            # we only support two levels of hierarchy, i.e.
            # prefix + many completions
            # or prefix + suffix + 1 completion
            assert total_levels == 2

            # shared cache must have a batch size of 1
            if num_new_levels == 2: assert input_ids[0].shape[0] == 1
            else: self.model.layers[0].self_attn.kv_cache.shared_caches[0].max_batch_size == 1
        if disable_hierarchy:
            # For the one-level Hydragen baseline, we support
            # prefix + suffix + many completions
            assert total_levels == 3 and num_return_sequences > 1

        if isinstance(seq_lens, Tensor):
            seq_lens = [seq_lens]
        elif seq_lens is None:
            seq_lens = [
                torch.full(
                    (input_id.shape[0],),
                    input_id.shape[1],
                    device=input_id.device,
                    dtype=torch.long,
                )
                for input_id in input_ids
            ]

        if len(input_ids) > 0:
            total_batch_size = input_ids[-1].shape[0] * num_return_sequences
        else:
            total_batch_size = starting_logits.shape[0] * num_return_sequences

        # suffix ids are for any prompts we need to run prefill on that
        # go in the unique-per-sequence KV cache.
        if num_return_sequences > 1 and not (disable_hierarchy or disable_hydragen):
            shared_ids = input_ids
            shared_seq_lens = seq_lens
            suffix_ids = None
            suffix_seq_lens = None
        elif len(input_ids) > 0:
            shared_ids = input_ids[:-1]
            shared_seq_lens = seq_lens[:-1]
            suffix_ids = input_ids[-1]
            suffix_seq_lens = seq_lens[-1]
        else: # passing starting logits instead of ids
            shared_ids,shared_seq_lens,suffix_ids,suffix_seq_lens = [],[],None,None


        if starting_logits is not None:
            starting_logits = starting_logits.unsqueeze(1)

        for sid, slen in zip(shared_ids, shared_seq_lens):
            starting_logits = self.append_shared(sid, slen)

        if disable_hydragen:
            self.model.set_disable_hydragen(True)
            if self.get_num_used_shared_caches() > 0:
                self.model.copy_shared_cache_to_unique(total_batch_size)

        if suffix_ids is not None:
            starting_logits = self.process_unique(suffix_ids, suffix_seq_lens)

            # needed for disable baselines
            self.repeat_per_completion_cache_for_num_samples(
                suffix_ids.shape[0], num_return_sequences
            )

        prefill_logits = starting_logits[:, -1]
        raw_first_token_ids = self.sample_from_logits(
            starting_logits[:, -1],
            temperature=temperature,
            num_samples=num_return_sequences,
            top_p=top_p,
        )
        first_token_ids = rearrange(
            raw_first_token_ids,
            "num_seq num_completions -> (num_seq num_completions) 1",
        )

        if return_logits:
            logits_to_return = [
                prefill_logits.repeat_interleave(num_return_sequences, 0)
            ]

        starting_position_ids = self.get_shared_cache_len(first_token_ids.shape[0])[
            :, None
        ]

        if suffix_seq_lens is not None:
            starting_position_ids = (
                starting_position_ids
                + suffix_seq_lens.repeat_interleave(num_return_sequences, 0)[:, None]
            )

        if eos_token_id is not None:
            finished_sequences = first_token_ids == eos_token_id
        else:
            finished_sequences = None

        decoded_tokens = [first_token_ids]

        if token_overrides is None:
            current_token_ids = first_token_ids
        else:
            current_token_ids = token_overrides[:, 0:1]

        self.set_mode(AttentionMode.DECODE)

        for generated_token_idx in range(max_new_tokens - 1):
            position_ids = starting_position_ids + generated_token_idx

            decode_logits = self(
                input_ids=current_token_ids,
                position_ids=position_ids,
                use_graph=self.graphed_model is not None,
            )

            if return_logits:
                logits_to_return.append(decode_logits[:, -1])

            current_token_ids = self.sample_from_logits(
                decode_logits[:, -1], temperature=temperature, top_p=top_p
            )

            if finished_sequences is not None:
                finished_sequences = torch.logical_or(
                    finished_sequences, current_token_ids == eos_token_id
                )

                if torch.all(finished_sequences):
                    break

            decoded_tokens.append(current_token_ids)

            if token_overrides is not None:
                current_token_ids = token_overrides[
                    :, generated_token_idx + 1 : generated_token_idx + 2
                ]

        cat_decoded_ids = torch.cat(decoded_tokens, dim=-1)

        if shared_cache_op == SharedCacheOp.PRESERVE:
            self.truncate_shared_caches(og_num_shared_caches)

        if disable_hydragen:
            self.model.set_disable_hydragen(False)

        if disable_attention:
            self.model.set_disable_attention(False)

        if return_logits:
            return cat_decoded_ids, logits_to_return

        return cat_decoded_ids

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs,
    ):
        hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path, **kwargs)

        if hf_model.dtype != torch.float16 and hf_model.dtype != torch.bfloat16:
            raise ValueError(
                f"Model must be in float16 or bfloat16, not {hf_model.dtype}"
            )

        with init_empty_weights(include_buffers=False):
            model = cls(
                hf_model.config,
            )

        model.load_state_dict(hf_model.state_dict(), assign=True)
        model.to(hf_model.device)

        model.device = hf_model.device
        model.dtype = hf_model.dtype

        return model
