"""
Implementation of the hydragen attention operation. 
"""

import torch
from torch import Tensor

from typing import Optional, List

from hydragen.flash import (
    flash_attention,
    flash_attention_varlen,
    flash_attention_seqlen,
)
from einops import rearrange

import triton
import triton.language as tl


def combine_lse_torch(
    outs: list[Tensor],
    lses: list[Tensor],
):
    """
    Torch implementation of 'combine_lse'. See 'combine_lse' function
    for docstring.
    """

    outs = torch.stack(outs)
    lses = torch.stack(lses)

    max_lse = lses.max(0).values

    adjust_factors = (lses - max_lse[None]).exp()

    new_denominator = adjust_factors.sum(0)

    aggregated = (
        (outs * adjust_factors.unsqueeze(-1)).sum(0)
    ) / new_denominator.unsqueeze(-1)

    return aggregated.to(outs.dtype)


@triton.jit
def combine_lse_kernel(
    out1_ptr,
    out2_ptr,
    lse1_ptr,
    lse2_ptr,
    aggregated_ptr,
    bsh_stride,
    bsh,
    hdim,
    BLOCK_SIZE_BSH: tl.constexpr,
    BLOCK_SIZE_HDIM: tl.constexpr,
):
    # Compute linear index
    bsh_idx = tl.program_id(0)

    bsh_range = tl.arange(0, BLOCK_SIZE_BSH)
    hdim_range = tl.arange(0, BLOCK_SIZE_HDIM)

    # Compute global memory offset
    lse_start = bsh_idx * BLOCK_SIZE_BSH
    lse_offs = lse_start + bsh_range

    lse1_ptrs = lse1_ptr + lse_offs
    lse2_ptrs = lse2_ptr + lse_offs

    lse1 = tl.load(lse1_ptrs, mask=lse_offs < bsh, other=0.0)
    lse2 = tl.load(lse2_ptrs, mask=lse_offs < bsh, other=0.0)

    max_lse = tl.maximum(lse1, lse2)

    adjust_factor1 = tl.exp(lse1 - max_lse)
    adjust_factor2 = tl.exp(lse2 - max_lse)

    new_denominator = adjust_factor1 + adjust_factor2

    out_start = bsh_idx * BLOCK_SIZE_BSH * bsh_stride
    out_offs = out_start + (bsh_range[:, None] * bsh_stride + hdim_range[None, :])

    out1_ptrs = out1_ptr + out_offs
    out2_ptrs = out2_ptr + out_offs
    agg_ptrs = aggregated_ptr + out_offs

    for i in range(0, tl.cdiv(hdim, BLOCK_SIZE_HDIM)):
        mask = out_offs + i * BLOCK_SIZE_HDIM < hdim * bsh
        out1 = tl.load(out1_ptrs, mask=mask, other=0.0)
        out2 = tl.load(out2_ptrs, mask=mask, other=0.0)

        agg = (
            out1 * adjust_factor1[:, None] + out2 * adjust_factor2[:, None]
        ) / new_denominator[:, None]

        tl.store(agg_ptrs, agg, mask=mask)

        out1_ptrs += BLOCK_SIZE_HDIM
        out2_ptrs += BLOCK_SIZE_HDIM
        agg_ptrs += BLOCK_SIZE_HDIM


def combine_lse_triton(out1: Tensor, lse1: Tensor, out2: Tensor, lse2: Tensor):
    """
    Triton version of 'combine_lse'.

    Args:
        out1, out2: Tensors of shape [batch, seq_len, qheads, hdim].
        lse1, lse2: Tensors of shape [batch, seq_len, qheads].
    """

    BATCH, SEQ_LEN, QHEADS, HDIM = out1.shape

    out1_flat = rearrange(out1, "b s h d -> (b s h) d")
    out2_flat = rearrange(out2, "b s h d -> (b s h) d")

    # Allocate output tensor
    aggregated_flat = torch.empty_like(out1_flat)

    assert out1_flat.is_contiguous()
    assert out2_flat.is_contiguous()
    assert lse1.is_contiguous()
    assert lse2.is_contiguous()
    assert aggregated_flat.is_contiguous()

    BSH = BATCH * SEQ_LEN * QHEADS

    grid = lambda META: (triton.cdiv(BSH, META["BLOCK_SIZE_BSH"]),)

    # Launch kernel
    combine_lse_kernel[grid](
        out1_flat,
        out2_flat,
        lse1,
        lse2,
        aggregated_flat,
        out1_flat.stride(0),
        BSH,
        HDIM,
        BLOCK_SIZE_BSH=32,
        BLOCK_SIZE_HDIM=64,
        num_warps=2,
    )

    aggregated = rearrange(
        aggregated_flat, "(b s h) d -> b s h d", b=BATCH, s=SEQ_LEN, h=QHEADS
    )

    return aggregated


def combine_lse(
    outs: list[Tensor],
    lses: list[Tensor],
    enable_triton: bool = True,
):
    """
    Merge attention results using log-sum-exp metadata.

    Args:
        outs: Attention results, list of [batch, seq_len, qheads, hdim]
        lses: Log-sum-exps of outs, corresponding list of [batch, seq_len, qheads]
        enable_triton: If True, uses triton kernel for combination. Only works for
            merging two attention results.
    """

    if enable_triton and len(outs) == 2:
        out1, out2 = outs
        lse1, lse2 = lses
        return combine_lse_triton(out1, lse1, out2, lse2)
    else:
        return combine_lse_torch(outs, lses)


def hydragen_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    shared_ks: list[Tensor],
    shared_vs: list[Tensor],
    shared_cu_seq_lens: list[Tensor | None],
    shared_max_seq_lens: list[int | None],
    use_varlens: list[bool],
    seq_lens: Tensor | None = None,
):
    """
    Computes Hydragen attention (e.g. applies attention decomposition + inter-sequence batching).

    This function allows shared keys and values to be provided in two different ways,
    depending on whether padding is needed. If shared cache level i does not need padding,
    then pass use_varlens[i] = False and pass shared_ks[i], shared_vs[i] with shape
    [sbatch, slen, kvheads, head_dim] (shared_cu_seq_lens[i] and shared_max_seq_lens[i]
    will be ignored). If padding is required, then pass use_varlens[i] = True and pass
    the keys and values using the flattened format that the flash-attn packages for
    their varlen kernels.

    If you don't need padding for any shared cache level, consider hydragen_attention_nopad
    for a simpler interface.

    Args:
        q: attention queries, shape [batch, qlen, qheads, head_dim]
        k: unique-per-sequence keys, shape [batch, kvlen, kvheads, head_dim]
        v: unique-per-sequence values, shape [batch, kvlen, kvheads, head_dim]

        shared_ks: list of shared keys. if use_varlens[i] = True, then shared_ks[i]
        has shape [total_slen, kvheads, head_dim]. Otherwise, it has shape
        [sbatch, slen, kvheads, head_dim]. sbatch can be any batch size that
        divides the batch size of q/k/v.

        shared_vs: list of shared values with shapes matching shared_ks.

        shared_cu_seq_lens: list of result of applying cumsum to slens, each of shape [sbatch+1] (first index
        is 0). For example, with sbatch = 2, max_slen = 4 where the sequence lengths of the shared prompts are 2 and 4,
        shared_prompt_cu_seqlens entry should be torch.tensor([0, 2, 6]).

        shared_max_seq_lens: list of max sequence length for shared keys/values.
        for previous example, entry should be 4

        use_varlens: list indicating if varlen kernel should be used, should be False if all items
        in shared batch have same sequence length, true otherwise

        seq_lens: sequence lengths for the unique-per-sequence KVs. If None, assumes no padding.
        padding must be on the right.
    """

    assert q.ndim == 4, f"{q.shape}"
    assert k.ndim == 4, f"{k.shape}"
    assert v.ndim == 4, f"{v.shape}"

    assert k.shape == v.shape
    assert (
        len(shared_ks)
        == len(shared_vs)
        == len(shared_cu_seq_lens)
        == len(shared_max_seq_lens)
        == len(use_varlens)
    )

    for shared_prompt_key, shared_prompt_value in zip(shared_ks, shared_vs):
        assert (
            shared_prompt_key.shape == shared_prompt_value.shape
        ), f"{shared_prompt_key.shape} {shared_prompt_value.shape}"

    b, nq, hq, d = q.shape

    outs, lses = [], []

    for sk, sv, scu, smax, use_varlen in zip(
        shared_ks,
        shared_vs,
        shared_cu_seq_lens,
        shared_max_seq_lens,
        use_varlens,
    ):

        # the varlen kernel seems to perform poorly at low
        # batch sizes, so when we don't need the padding,
        # we use the non-varlen kernel.
        if not use_varlen:
            num_shared_sequences = sk.shape[0]

            batched_q = rearrange(
                q,
                "(num_shared_seq seq_per_shared) nq hq d -> num_shared_seq (seq_per_shared nq) hq d",
                num_shared_seq=num_shared_sequences,
            )

            shared_out, shared_lse = flash_attention(batched_q, sk, sv)

            shared_out = shared_out.view(b, nq, hq, d)
            if k.shape[1] == 0 and len(shared_ks) == 1:
                return shared_out

            shared_lse = rearrange(
                shared_lse,
                "num_shared_seq h (seq_per_shared nq) -> (num_shared_seq seq_per_shared) nq h",
                nq=nq,
            ).contiguous()

        else:
            num_shared_sequences = scu.shape[0] - 1

            assert b % num_shared_sequences == 0, f"{b} {num_shared_sequences}"

            sequences_per_shared = b // num_shared_sequences
            queries_per_shared = sequences_per_shared * nq

            batched_q = rearrange(
                q,
                "b nq hq d -> (b nq) hq d",
            )

            qlens_to_cumsum = torch.cat(
                [
                    torch.zeros(
                        (1,),
                        dtype=torch.int32,
                        device=q.device,
                    ),
                    torch.full(
                        (num_shared_sequences,),
                        queries_per_shared,
                        dtype=torch.int32,
                        device=q.device,
                    ),
                ]
            )

            q_culens = qlens_to_cumsum.cumsum(0, dtype=torch.int32)

            shared_out, shared_lse = flash_attention_varlen(
                batched_q,
                sk,
                sv,
                cu_seqlens_q=q_culens,
                cu_seqlens_k=scu,
                max_seqlen_q=queries_per_shared,
                max_seqlen_k=smax,
            )

            shared_out = rearrange(
                shared_out,
                "(ngroups groupsize nq) hq d -> (ngroups groupsize) nq hq d",
                ngroups=num_shared_sequences,
                groupsize=sequences_per_shared,
            ).contiguous()

            if k.shape[1] == 0 and len(shared_ks) == 1:
                return shared_out

            shared_lse = rearrange(
                shared_lse,
                "ngroups h (groupsize nq) -> (ngroups groupsize) nq h",
                ngroups=num_shared_sequences,
                groupsize=sequences_per_shared,
            ).contiguous()

        outs.append(shared_out)
        lses.append(shared_lse)

    if seq_lens is None:
        unique_out, unique_lse = flash_attention(q, k, v, causal=True)
        unique_lse = rearrange(unique_lse, "b h q -> b q h").contiguous()
    else:
        unique_out, unique_lse = flash_attention_seqlen(q, k, v, seq_len=seq_lens)

    outs.append(unique_out)
    lses.append(unique_lse)

    aggregated = combine_lse(outs, lses, enable_triton=True)

    return aggregated


def hydragen_attention_nopad(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    shared_ks: list[Tensor],
    shared_vs: list[Tensor],
    seq_len: Optional[Tensor] = None,
):
    """
    Computes Hydragen attention (e.g. applies attention decomposition + inter-sequence batching).
    Assumes no padding in the shared KVs - is that is required, check out hydragen_attention.

    Args:
        q: attention queries, shape [batch, qlen, qheads, head_dim]
        k: unique-per-sequence keys, shape [batch, kvlen, kvheads, head_dim]
        v: unique-per-sequence values, shape [batch, kvlen, kvheads, head_dim]

        shared_prompt_keys: list of shared keys, each of shape [sbatch, slen, kvheads, head_dim].
        sbatch can be any batch size that divides the total batch size of q/k/v.

        shared_prompt_values: list of shared values matching shared_prompt_keys.

        seq_len: sequence lengths for the unique-per-sequence KVs. If None, assumes no padding.
        padding must be on the right.
    """
    return hydragen_attention(
        q,
        k,
        v,
        shared_ks=shared_ks,
        shared_vs=shared_vs,
        shared_cu_seq_lens=[None] * len(shared_ks),
        shared_max_seq_lens=[None] * len(shared_ks),
        use_varlens=[False] * len(shared_ks),
        seq_lens=seq_len,
    )
