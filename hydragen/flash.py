"""
Interfaces for the high-performance 
attention kernels that we use to implement hydragen.

We rely on two kernels: flash-attn for attention
over keys and values that do not change across decoding
steps (i.e. shared KV caches), and a slightly modified kernel from 
xformers to compute attention over KVs that do change across
iterations (i.e. the unique-per-sequence KV cache). The reason
we need the latter kernel is because we want to use CUDA
graphs to reduce CPU overhead, which imposes constraints
on how tensors can be updated/accessed across decoding steps.

We've made some changes to the original xformers triton kernel,
e.g. to support many queries, remove options we don't need, etc.
"""

from hydragen.xformers_stuff import _fwd_kernel_splitK, _strides, unroll_varargs


import triton
import triton.language as tl

import torch
from torch import Tensor

import math

from einops import rearrange

from flash_attn.flash_attn_interface import (
    _flash_attn_forward,
    _flash_attn_varlen_forward,
)


def pick_split_k(
    B: int, H: int, M: int, BLOCK_M: int, Mk: int, BLOCK_N: int, sm_count: int
):
    """
    From https://github.com/Dao-AILab/flash-attention/blob/9356a1c0389660d7e231ff3163c1ac17d9e3824a/csrc/flash_attn/flash_api.cpp#L215
    """
    num_m_blocks = (M + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (Mk + BLOCK_N - 1) // BLOCK_N

    batch_nheads_mblocks = B * H * num_m_blocks

    if batch_nheads_mblocks >= 0.8 * sm_count:
        return 1

    max_splits = min(sm_count, num_n_blocks, 128)

    is_split_eligible = lambda num_splits: (
        num_splits == 1
        or math.ceil(num_n_blocks / num_splits)
        != math.ceil(num_n_blocks / (num_splits - 1))
    )

    efficiency = []

    for num_splits in range(1, max_splits + 1):
        if not is_split_eligible(num_splits):
            efficiency.append(0.0)
        else:
            n_waves = batch_nheads_mblocks * num_splits / sm_count
            eff = n_waves / math.ceil(n_waves)
            efficiency.append(eff)

    max_efficiency = max(efficiency)

    for num_splits, eff in enumerate(efficiency, 1):
        if eff >= 0.85 * max_efficiency:
            return num_splits


@triton.jit
def _splitK_reduce(
    Out_splitK,  # [B, H, split_k, Mq, K]
    Metadata,  # [B, H, 2, split_k, M_ceil] contains [mi, li]
    Out,  # [B, H, M, K]
    LSE,  # [B, H, M]
    split_k,
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_osk_k,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_mm,
    stride_oz,
    stride_oh,
    stride_og,
    stride_om,
    stride_ok,
    stride_lse_zhg,
    stride_lse_m,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
):
    """
    Like the xformers version, but we use two grid dimensions to avoid
    overflowing the 16-bit limit.
    """

    off_zhg = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_h = (off_zhg // G) % H
    off_g = off_zhg % G
    off_m = tl.program_id(2)

    Out_splitK_ptr = (
        Out_splitK
        + stride_osk_zhg * off_zhg
        + stride_osk_m * off_m
        + tl.arange(0, BLOCK_SIZE)
    )
    Metadata_ptr = Metadata + stride_mzhg * off_zhg + off_m
    m = tl.load(Metadata_ptr)
    l_sum = tl.load(Metadata_ptr + stride_m2)
    acc = tl.load(Out_splitK_ptr)

    for split_k_idx in range(1, split_k):
        Metadata_ptr = Metadata_ptr + stride_ms
        Out_splitK_ptr = Out_splitK_ptr + stride_osk_s

        m_k = tl.load(Metadata_ptr)
        l_k = tl.load(Metadata_ptr + stride_m2)
        acc_k = tl.load(Out_splitK_ptr)

        m_new = tl.maximum(m, m_k)
        if m_k < m:
            # Scale incoming values
            alpha = tl.math.exp2(m_k - m_new)
            acc_k = acc_k * alpha
            l_k = l_k * alpha
        else:
            # Scale our values
            alpha = tl.math.exp2(m - m_new)
            acc = acc * alpha
            l_sum = l_sum * alpha

        m = m_new
        l_sum = l_sum + l_k
        acc = acc + acc_k

    acc = acc / l_sum
    Out_ptr = (
        Out
        + stride_oz * off_z
        + stride_oh * off_h
        + stride_og * off_g
        + stride_om * off_m
        + tl.arange(0, BLOCK_SIZE)
    )
    tl.store(Out_ptr, acc)

    l_ptrs = LSE + off_zhg * stride_lse_zhg + off_m
    tl.store(l_ptrs, (m + tl.math.log2(l_sum)) / 1.44269504)


def flash_attention_seqlen(raw_q: Tensor, raw_k: Tensor, raw_v: Tensor, seq_len=None):
    """
    q shape: [batch, qseq_len, qheads, dim]
    k shape: [batch, kseq_len, kheads, dim]
    v shape: [batch, kseq_len, kheads, dim]
    """

    qheads = raw_q.shape[-2]
    kheads = raw_k.shape[-2]

    qheads_per_k = qheads // kheads

    # for GQA fold extra query heads into the seq len
    q = rearrange(raw_q, "b q (kh qh) d -> b (q qh) kh 1 d", kh=kheads, qh=qheads_per_k)
    k = rearrange(raw_k, "b k kh d -> b k kh 1 d")
    v = rearrange(raw_v, "b k kh d -> b k kh 1 d")

    Lk = k.shape[-1]
    PACKED_PER_VAL = 1

    B, Mk, G, H, Kkv = k.shape
    B, M, G, H, Kq = q.shape

    assert Lk == Kq, f"Keys have head dim {Lk} but queries have head dim {Kq}"

    next_pow2 = 2 ** math.ceil(math.log2(M))
    BLOCK_M = max(16, min(next_pow2, 128))
    BLOCK_N = 64
    num_warps = min(max(max(BLOCK_M, BLOCK_N) // 32, 1), 4)

    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    split_k = pick_split_k(
        B=B, H=H, M=M, BLOCK_M=BLOCK_M, Mk=Mk, BLOCK_N=BLOCK_N, sm_count=sm_count
    )

    M_ceil = (M + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    o_splitk = torch.empty(
        [B * G * H, split_k, M_ceil, Kq], dtype=torch.float32, device=q.device
    )
    metadata = torch.empty(
        [B * G * H, 2, split_k, M_ceil], dtype=torch.float32, device=q.device
    )
    grid = (triton.cdiv(M, BLOCK_M), B * G * H, split_k)

    split_size = (Mk + split_k - 1) // split_k

    use_seq_len = seq_len is not None

    _fwd_kernel_splitK_unrolled = unroll_varargs(_fwd_kernel_splitK, N=1)

    _fwd_kernel_splitK_unrolled[grid](
        Q=q,
        K=k,
        V=v,
        sm_scale=q.shape[-1] ** -0.5,
        Out_splitK=o_splitk,
        Metadata=metadata,
        Seq_len=seq_len.to(torch.int32),
        **_strides(q, "qz", "qm", "qg", "qh", "qk"),
        **_strides(k, "kz", "kn", "kg", "kh", "kk"),
        **_strides(v, "vz", "vn", "vg", "vh", "vk"),
        **_strides(o_splitk, "osk_zhg", "osk_s", "osk_m", "osk_k"),
        **_strides(metadata, "mzhg", "m2", "ms", "mm"),
        Z=B,
        H=H,
        G=G,
        N_CTX_Q=M,
        N_CTX_K=Mk,
        BLOCK_N_PER_SPLIT=split_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=Lk,
        BOUNDS_CHECKS_N=(split_size % BLOCK_N) > 0 or use_seq_len,
        USE_SEQ_LEN=use_seq_len,
        PACKED_PER_VAL=PACKED_PER_VAL,
        N_GROUPS=1,
        num_warps=num_warps,
        num_stages=1,
    )

    # Merge together
    grid = (B * G * H, 1, M)

    # Handling case where we exceed limits of each CUDA grid axis.
    if grid[0] >= 2**16:
        big_dim = grid[0]
        # TODO make cleaner
        candidate_divisors = [2**i for i in range(3, 10)] + [B, G, H]
        max_divisor = max([i for i in candidate_divisors if big_dim % i == 0])
        grid = (big_dim // max_divisor, max_divisor, M)

    out = torch.empty((B, M, G, H, Kq), device=q.device, dtype=q.dtype)
    lse = torch.empty((B * G * H, M), device=q.device, dtype=torch.float32)

    _splitK_reduce[grid](
        o_splitk,
        metadata,
        out,
        lse,
        split_k=split_k,
        **_strides(o_splitk, "osk_zhg", "osk_s", "osk_m", "osk_k"),
        **_strides(metadata, "mzhg", "m2", "ms", "mm"),
        **_strides(out, "oz", "om", "og", "oh", "ok"),
        **_strides(lse, "lse_zhg", "lse_m"),
        BLOCK_SIZE=out.shape[-1],
        G=G,
        H=H,
        # TODO: Tune num_warps
    )

    final_out = rearrange(
        out, "b (q qh) kh 1 d -> b q (kh qh) d", kh=kheads, qh=qheads_per_k
    )

    final_lse = rearrange(
        lse, "(b 1 kh) (q qh) -> b q (kh qh)", kh=kheads, qh=qheads_per_k
    )

    return final_out, final_lse


def flash_attention(
    q: Tensor, k: Tensor, v: Tensor, causal: bool = False
) -> tuple[Tensor, Tensor]:
    """
    q: [b, seq q, qheads, dim]
    k: [b, seq k, kheads, dim]
    v: [b, seq k, kheads, dim]
    """

    softmax_scale = q.shape[-1] ** -0.5

    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
        q,
        k,
        v,
        dropout_p=0.0,
        causal=causal,
        softmax_scale=softmax_scale,
        window_size=(-1, -1),
        return_softmax=False,
    )

    return out, softmax_lse


def flash_attention_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    q: [b*seq q, qheads, dim]
    k: [b*seq k, kheads, dim]
    v: [b*seq k, kheads, dim]
    """

    softmax_scale = q.shape[-1] ** -0.5

    (
        out,
        q,
        k,
        v,
        out_padded,
        softmax_lse,
        S_dmask,
        rng_state,
    ) = _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        causal=causal,
        softmax_scale=softmax_scale,
        window_size=(-1, -1),
        return_softmax=False,
    )

    return out, softmax_lse