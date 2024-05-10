"""
Internals from the xformers repo that are needed
to run their triton attention kernel.

Code taken from:
https://github.com/facebookresearch/xformers/blob/042abc8aa47d1f5bcc2e82df041811de218924ba/xformers/triton/vararg_kernel.py
https://github.com/facebookresearch/xformers/blob/042abc8aa47d1f5bcc2e82df041811de218924ba/xformers/ops/fmha/triton_splitk.py

"""

import ast
import copy
import functools
import linecache
import sys
from typing import Any, Dict, List

import triton
import triton.language as tl

import torch


def _strides(x: torch.Tensor, *stride_names: str):
    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}


class _ForLoopUnroller(ast.NodeTransformer):
    def __init__(self, target, inline_variables, loop_iter):
        self.loop_iter = loop_iter
        self.target = target
        self.inline_variables = inline_variables

    def visit_Name(self, node):
        if node.id != self.target:
            return node
        return ast.Name(str(self.loop_iter))

    def visit_Subscript(self, node):
        # Pattern-matching `value[slice]`
        if (
            isinstance(node.slice, ast.Name)
            and node.slice.id == self.target
            and isinstance(node.value, ast.Name)
            and node.value.id in self.inline_variables
        ):
            return ast.Name(f"{node.value.id}{self.loop_iter}")
        return node


class _VisitorUnrollKernel(ast.NodeTransformer):
    def __init__(self, N):
        self.inline_variables = set()
        self.N = N

    def visit_AnnAssign(self, node):
        # Pattern-matching:
        # var_name: "VAR_ARGS_ARRAY"
        if (
            node.value is None
            and node.simple == 1
            and isinstance(node.target, ast.Name)
            and isinstance(node.annotation, ast.Constant)
            and node.annotation.value == "VAR_ARGS_ARRAY"
        ):
            self.inline_variables.add(node.target.id)
            return []
        if node.value is not None:
            node.value = self.visit(node.value)
        if node.annotation is not None:
            node.annotation = self.visit(node.annotation)
        if node.target is not None:
            node.target = self.visit(node.target)
        return node

    def visit_arguments(self, node):
        # Replace `args` annotated with `VAR_ARGS_ARRAY`
        new_args = []
        for arg in node.args:
            if (
                arg.annotation is not None
                and isinstance(arg.annotation, ast.Constant)
                and arg.annotation.value == "VAR_ARGS_ARRAY"
            ):
                self.inline_variables.add(arg.arg)
                new_args += [ast.arg(f"{arg.arg}{i}") for i in range(self.N)]
                continue
            new_args.append(arg)
        if node.vararg is not None:
            self.inline_variables.add(node.vararg.arg)
            new_args += [ast.arg(f"{node.vararg.arg}{i}") for i in range(self.N)]
            node.vararg = None
            new_args += node.kwonlyargs
            node.kwonlyargs = []
        node.args = new_args
        return node

    def visit_For(self, node):
        if (
            not isinstance(node.iter, ast.Call)
            or node.iter.func.id != "range"
            or len(node.iter.args) != 1
            or not isinstance(node.iter.args[0], ast.Call)
            or node.iter.args[0].func.id != "len"
            or len(node.iter.args[0].args) != 1
            or node.iter.args[0].args[0].id not in self.inline_variables
        ):
            node.body = [self.visit(x) for x in node.body]
            return node
        # We know we have to modify this loop
        new_nodes = []
        for i in range(self.N):
            unroller = _ForLoopUnroller(
                target=node.target.id,
                inline_variables=self.inline_variables,
                loop_iter=i,
            )
            for body in node.body:
                body = copy.deepcopy(body)
                new_node = ast.fix_missing_locations(unroller.visit(body))
                new_node = self.visit(new_node)
                new_nodes.append(new_node)
        return new_nodes


# Hackfix to get access to get source-code for
# `exec`-created functions - see https://stackoverflow.com/a/69668999
_getlines_orig = None
_FILENAME_TO_SRC: Dict[str, str] = {}


def _monkey_patched_getlines(filename, module_globals=None):
    if filename in _FILENAME_TO_SRC:
        return _FILENAME_TO_SRC[filename]
    else:
        return _getlines_orig(filename, module_globals)  # type: ignore


@functools.lru_cache(None)
def unroll_varargs(kernel, N: int):
    """
    Specializes a triton kernel with variable number of inputs
    to a specific number of inputs `N`.
    NOTE: Because it's quite costly to call `triton.jit`,
    we cache the returned value with `lru_cache`
    """
    global _FILENAME_TO_SRC, _getlines_orig

    k = triton.JITFunction(kernel.fn)
    parsed = ast.parse(k.src)
    nodeVisitor = _VisitorUnrollKernel(N=N)
    parsed = nodeVisitor.visit(parsed)
    parsed = ast.fix_missing_locations(parsed)

    # NOTE: `ast.unparse` requires python 3.9+
    if (sys.version_info.major, sys.version_info.minor) <= (3, 8):
        raise RuntimeError("Error: This functionality requires python 3.9 or above")
    new_src = ast.unparse(parsed)  # type: ignore

    # Now we want to `eval` the function, but we need all this
    # boilerplate code to make sure triton can run `inspect.getsource`

    fn_filename = f"<unroll_varargs-{kernel.fn.__name__}-{N}>"

    # Create function given source
    code = compile(new_src, fn_filename, "exec")

    _locals: Dict[str, Any] = {}
    exec(code, kernel.fn.__globals__, _locals)
    assert len(_locals) == 1, len(_locals)
    fn = next(iter(_locals.values()))
    # Patch `getlines` only the first time
    if not _FILENAME_TO_SRC:
        _getlines_orig = linecache.getlines
        linecache.getlines = _monkey_patched_getlines
    _FILENAME_TO_SRC[fn_filename] = new_src

    jitted_fn = triton.jit(fn)
    jitted_fn.src = new_src
    return jitted_fn


# Note: just import this to make mypy happy
# when annotating variables with `VAR_ARGS_ARRAY`
VAR_ARGS_ARRAY = List[Any]


@triton.jit
def _fwd_kernel_splitK(
    Q,
    K,
    V,
    sm_scale,
    Out_splitK,  # [B, H, split_k, Mq, K]
    Metadata,  # [B, H, 2, split_k, M_ceil] contains [mi, li]
    Seq_len,
    stride_qz,
    stride_qm,
    stride_qg,
    stride_qh,
    stride_qk,
    stride_kz,
    stride_kn,
    stride_kg,
    stride_kh,
    stride_kk,
    stride_vz,
    stride_vn,
    stride_vg,
    stride_vh,
    stride_vk,
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_osk_k,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_mm,
    Z,
    N_CTX_Q,
    N_CTX_K,
    BLOCK_N_PER_SPLIT,
    H: tl.constexpr,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BOUNDS_CHECKS_N: tl.constexpr,
    USE_SEQ_LEN: tl.constexpr,
    PACKED_PER_VAL: tl.constexpr = 1,
    N_GROUPS: tl.constexpr = 1,
):
    """This kernel can accept non-quantized or int4-quantized keys/values.
    PACKED_PER_VAL determines the quantization type:
        - PACKED_PER_VAL == 1 means no quantization
        - PACKED_PER_VAL == 8 means 4-bit quantization (8 packed quantized values inside one int32)
    For the quantized case K/V should be int32 tensors.
    Quantization can be row-wise (when N_GROUPS = 1) or group-wise with N_GROUPS = 2, 4, or 8.
    Quantization coefficients are stored at the beginning of the row along the last dimension of K/V
    So K[B, H, M, :] has a form
    [   quant_coef0, quant_coef1, ...|
        group0_quant_value0, group0_quant_value1,... |
        group1_quant_value0, group1_quant_value1,...]
    where each quant_coef is an int32 which should be interpreted as 2 packed float16: scale and offset.

    Note: this kernel needs to be processed by xformers.triton.vararg_kernel.unroll_varargs
    before compilation. That will unroll variables marked with "VAR_ARGS_ARRAY" into lists.
    See how FwOp.apply does it below.
    """
    tl.static_assert(
        (PACKED_PER_VAL == 1 and tl.constexpr(K.dtype.element_ty != tl.int32))
        or (PACKED_PER_VAL == 8 and tl.constexpr(K.dtype.element_ty == tl.int32)),
        f"Only 4-bit quantization is supported, K/V should have dtype int32 in "
        f"the quantized case: {PACKED_PER_VAL=} {tl.constexpr(K.dtype)=} {tl.constexpr(K.dtype.element_ty)=}",
    )
    tl.static_assert(
        (((N_GROUPS == 1 or N_GROUPS == 2) or N_GROUPS == 4) or N_GROUPS == 8),
        "Number of quantization groups can be 1 (row-wise quantization), 2, 4, or 8.",
    )

    QUANTIZED: tl.constexpr = PACKED_PER_VAL > 1
    PACKED_D_PER_GROUP: tl.constexpr = BLOCK_DMODEL // PACKED_PER_VAL // N_GROUPS
    D_PER_GROUP: tl.constexpr = BLOCK_DMODEL // N_GROUPS

    start_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_h = (off_zhg // G) % H
    off_g = off_zhg % G
    splitk_idx = tl.program_id(2)

    lo = splitk_idx * BLOCK_N_PER_SPLIT
    if USE_SEQ_LEN:
        kv_len = tl.load(Seq_len + off_z)
    else:
        kv_len = N_CTX_K
    hi = tl.minimum((splitk_idx + 1) * BLOCK_N_PER_SPLIT, kv_len)

    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_h * stride_qh + off_z * stride_qz + off_g * stride_qg,
        shape=(N_CTX_Q, D_PER_GROUP),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_PER_GROUP),
        order=(1, 0),
    )

    k_base = K + off_h * stride_kh + off_z * stride_kz + off_g * stride_kg
    # Additional shift by 1 along the last dimension in the quantized case, since
    # the first element along that dim contains packed quantization coefficients.
    K_block_ptr = tl.make_block_ptr(
        base=k_base + stride_kk * QUANTIZED * N_GROUPS,
        shape=(PACKED_D_PER_GROUP, hi),
        strides=(stride_kk, stride_kn),
        offsets=(0, lo),
        block_shape=(PACKED_D_PER_GROUP, BLOCK_N),
        order=(0, 1),
    )
    v_base = V + off_h * stride_vh + off_z * stride_vz + off_g * stride_vg
    V_block_ptr = tl.make_block_ptr(
        base=v_base + stride_vk * QUANTIZED * N_GROUPS,
        shape=(hi, PACKED_D_PER_GROUP),
        strides=(stride_vn, stride_vk),
        offsets=(lo, 0),
        block_shape=(BLOCK_N, PACKED_D_PER_GROUP),
        order=(1, 0),
    )

    if QUANTIZED:
        # Pointers to quantization coefficients. Even those they are 1D,
        # we have to use block pointers, since usual pointers
        # don't support boundary checks
        K_scale_shift_block_ptr = tl.make_block_ptr(
            base=k_base,
            shape=(1, hi),
            strides=(stride_kk, stride_kn),
            offsets=(0, lo),
            block_shape=(1, BLOCK_N),
            order=(0, 1),
        )
        V_scale_shift_block_ptr = tl.make_block_ptr(
            base=v_base,
            shape=(hi, 1),
            strides=(stride_vn, stride_vk),
            offsets=(lo, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )
    else:
        K_scale_shift_block_ptr = None
        V_scale_shift_block_ptr = None

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Before compilation, this kernel will be processed by xformers.triton.vararg_kernel.unroll_varargs.
    # That turns tensors annotated as the one below into lists of tensors of length N_GROUPS.
    # This is a solution for Triton native lack of support for lists of tensors.
    acc: "VAR_ARGS_ARRAY"  # noqa: F821

    for i in range(len(acc)):  # noqa: F821
        acc[i] = tl.zeros([BLOCK_M, D_PER_GROUP], dtype=tl.float32)  # noqa: F821
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q: "VAR_ARGS_ARRAY"  # noqa: F821
    for i in range(len(acc)):  # noqa: F821
        q[i] = tl.load(  # noqa: F821
            tl.advance(Q_block_ptr, (0, i * D_PER_GROUP)), boundary_check=(0,)
        )
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        k: "VAR_ARGS_ARRAY"  # noqa: F821
        v: "VAR_ARGS_ARRAY"  # noqa: F821
        for i in range(len(acc)):  # noqa: F821
            k[i], v[i] = load_dequantize_k_v_group(  # noqa: F821
                K_block_ptr,
                V_block_ptr,
                K_scale_shift_block_ptr,
                V_scale_shift_block_ptr,
                BOUNDS_CHECKS_N,
                PACKED_PER_VAL,
                PACKED_D_PER_GROUP,
                Q.dtype.element_ty,
                i,
            )

        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for i in range(len(acc)):  # noqa: F821
            qk += tl.dot(q[i], k[i])  # noqa: F821
        qk *= qk_scale

        # TODO: This is slow, and only needed at the last iteration.
        # Maybe we can unroll the last iteration instead?
        if BOUNDS_CHECKS_N:
            qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n, qk, float("-inf"))
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        p = p.to(Q.dtype.element_ty)

        # -- scale and update acc --
        for i in range(len(acc)):  # noqa: F821
            acc[i] *= alpha[:, None]  # noqa: F821
            acc[i] += tl.dot(p, v[i])  # noqa: F821
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if PACKED_PER_VAL > 1:
            K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr, (0, BLOCK_N))
            V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr, (BLOCK_N, 0))

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out_splitK + off_zhg * stride_osk_zhg + splitk_idx * stride_osk_s,
        shape=(N_CTX_Q, D_PER_GROUP),
        strides=(stride_osk_m, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_PER_GROUP),
        order=(1, 0),
    )
    for i in range(len(acc)):  # noqa: F821
        tl.store(
            tl.advance(O_block_ptr, (0, i * D_PER_GROUP)),
            acc[i],  # noqa: F821
            boundary_check=(0,),
        )
    # Write metadata for split-K reduction
    Metadata_ptr = (
        Metadata
        + off_zhg * stride_mzhg
        + splitk_idx * stride_ms
        + start_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )
    tl.store(Metadata_ptr, m_i)
    tl.store(Metadata_ptr + stride_m2, l_i)


@triton.jit
def load_dequantize_k_v_group(
    K_block_ptr,
    V_block_ptr,
    K_scale_shift_block_ptr,
    V_scale_shift_block_ptr,
    BOUNDS_CHECKS_N: tl.constexpr,
    PACKED_PER_VAL: tl.constexpr,
    PACKED_D_PER_GROUP: tl.constexpr,
    dtype: tl.constexpr,
    group_id: tl.constexpr,
):
    """Load K/V for a given block. In case of int4-quantized K/V, dequantize them after loading.
    If quantization is group-wise, use group_id to advance the pointers to the current group.
    """
    # Advance to the current quantization group
    K_block_ptr = tl.advance(K_block_ptr, (PACKED_D_PER_GROUP * group_id, 0))
    V_block_ptr = tl.advance(V_block_ptr, (0, PACKED_D_PER_GROUP * group_id))

    # -- load k, v --
    k = tl.load(K_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ())
    v = tl.load(V_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ())

    if PACKED_PER_VAL > 1:
        # K/V are quantized, load quantization coefficients and dequantize

        K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr, (group_id, 0))
        V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr, (0, group_id))

        k_scale_shift = tl.load(
            K_scale_shift_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ()
        )
        v_scale_shift = tl.load(
            V_scale_shift_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ()
        )

        k_scale, k_shift = cast_uint32_to_half2(k_scale_shift)
        v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
        v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL).to(dtype)
        k_t = dequantize(
            tl.trans(k),
            tl.trans(k_scale),
            tl.trans(k_shift),
            PACKED_PER_VAL,
        ).to(dtype)
        k = tl.trans(k_t)
    return k, v


@triton.jit
def cast_uint32_to_half2(scale_shift):
    """Extract two float16 packed into one int32"""
    scale = scale_shift & 0xFFFF
    shift = scale_shift >> 16
    scale = scale.to(tl.uint16).to(tl.float16, bitcast=True)
    shift = shift.to(tl.uint16).to(tl.float16, bitcast=True)
    return scale, shift


@triton.jit
def dequantize(
    x_,
    scale,
    shift,
    PACKED_PER_VAL: tl.constexpr = 8,
):
    """PACKED_PER_VAL is the number of values packed into each element x_.
    For example, for int4 quantization and x_ of type int32, PACKED_PER_VAL is 8.
    """

    # Axis along which offsets are applied matters here
    # It would be natural to have offsets in shape (BLOCK_N, D // PACKED_PER_VAL, PACKED_PER_VAL)
    # and expand K/V to that shape before applying offsets
    # However, Triton for some reason considers dim=1 as contiguous when doing tl.view below, and not dim=2
    # Note that tl.view doesn't guarantee the order of elements in the result - thus the code below depends
    # on the implementation details which might change in the future.
    # Ideally we would like to use tl.reshape, but it's not implemented yet.
    # See https://github.com/openai/triton/blob/9055af1a5dadc576804b38dd77ee91dc42af0bf7/python/triton/language/semantic.py#L541 # noqa: E501

    # x_ : (BLOCK_N, D // PACKED_PER_VAL)
    # scale: (BLOCK_N, 1)
    # offsets: (PACKED_PER_VAL,)
    BLOCK_N: tl.constexpr = x_.shape[0]
    BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[1]
    offsets = tl.arange(0, PACKED_PER_VAL) * 4
    quant_offset = (
        x_[:, None, :] >> offsets[None, :, None]
    )  # (BLOCK_N, PACKED_PER_VAL, D // PACKED_PER_VAL)

    quant_offset = tl.view(
        quant_offset, (BLOCK_N, BLOCK_DMODEL_PACKED * PACKED_PER_VAL)
    )
    # Trick - instead of converting int4 to float16 we view it as float16
    # and then multiply by 32768 * 512 == 2**24
    quant_offset = (quant_offset & 0xF).to(tl.uint16).to(tl.float16, bitcast=True)
    quant_offset = (quant_offset * 32768.0).to(tl.float16)
    scale_512 = scale * 512

    dequant = quant_offset * scale_512 + shift
    return dequant
