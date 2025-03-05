# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union

import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
import flash_attn_3_cuda

# isort: on


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


# torch.compile() support is only enabled for pytorch >= 2.4
# The reason for this is that we are using the new custom_op and register_fake
# APIs, which support inplace modification of inputs in the function itself
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:

    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


def round_multiple(x, m):
    return (x + m - 1) // m * m


def get_max_headdim():
    """
    Translates the inline C++ function get_max_headdim() into Python.
    We simulate the #ifndef checks by passing in booleans.
    """
    return 256

def round_up_headdim(head_size):
    """
    Translates the inline C++ function round_up_headdim(int head_size) into Python.
    """
    if head_size <= 64:
        return 64
    if head_size <= 96:
        return 96
    if head_size <= 128:
        return 128
    if head_size <= 192:
        return 192
    if head_size <= 256:
        return 256
    return 256

# Reader: this Python function is written in this unusual way on purpose,
# so that aligns as closely as possible the with variable names, logic, and
# style of the c++ function ~100 lines of flash_api.cpp::mha_bwd. This is in order
# to make it easy to update this function when the c++ function changes,
# and easy to verify it is _exactly_ identical. Please do not rewrite this
# function to make it look like readable python
def mha_bwd_shapes(
    q,                 # (b, s_q, h, d) or (total_q, h, d) if varlen
    k,                 # (b, s_k, h_k, d) or (total_k, h_k, d) if varlen
    softmax_lse,       # (b, h, s_q) or (h, total_q) if varlen
    cu_seqlens_q=None, # optional tensor (b+1)
    cu_seqlens_k=None, # optional tensor (b+1)
    seqused_q=None,    # optional
    seqused_k=None,    # optional
    max_seqlen_q=None,
    max_seqlen_k=None,
    is_causal=False,
    window_size_left=0,
    window_size_right=0,
    softcap=0.0,
):
    """
    Translates the C++ function bwd_shapes(...) into Python.

    Returns a list [seqlen_q_rounded, seqlen_k_rounded, total_q_padded_rounded,
                    total_k_padded_rounded, head_size_rounded].
    """

    # Determine whether we have "varlen" input
    is_varlen_q = cu_seqlens_q is not None
    is_varlen_k = cu_seqlens_k is not None
    is_varlen   = is_varlen_q or is_varlen_k or (seqused_q is not None) or (seqused_k is not None)

    # Extract shapes
    if not is_varlen_q:
        # (b, s_q, h, d)
        batch_size = q.size(0)
        seqlen_q   = q.size(1)
        total_q    = batch_size * seqlen_q
    else:
        # (total_q, h, d)
        batch_size = cu_seqlens_q.size(0) - 1
        seqlen_q   = max_seqlen_q
        total_q    = q.size(0)

    num_heads   = q.size(-2)
    head_size   = q.size(-1)

    if not is_varlen_k:
        # (b, s_k, h_k, d)
        seqlen_k = k.size(1)
        total_k  = batch_size * seqlen_k
    else:
        # (total_k, h_k, d)
        seqlen_k = max_seqlen_k
        total_k  = k.size(0)

    num_heads_k = k.size(-2)

    # Adjust window sizes & causal
    if window_size_left >= seqlen_k - 1:
        window_size_left = -1
    if window_size_right >= seqlen_q - 1:
        window_size_right = -1
    if is_causal:
        window_size_right = 0

    is_causal = window_size_left < 0 and window_size_right == 0

    major, minor = torch.cuda.get_device_capability()
    arch = 10 * major + minor


    # Round up head_size
    head_size_rounded = round_up_headdim(head_size)

    # If not causal but window_size_left < 0, window_size_right == 0,
    # the backend might forcibly treat it as causal. For simplicity, keep the same logic:
    is_local = (window_size_left >= 0 or window_size_right >= 0) and not is_causal

    # Derive kBlockM, kBlockN based on architecture and head_size_rounded

    # Boolean equivalent
    is_local = ((window_size_left >= 0) or (window_size_right >= 0)) and (not is_causal)

    # kBlockM_sm90
    kBlockM_sm90 = (
        (96 if (is_causal and softcap > 0.0) else 128) if head_size_rounded <= 64 else
        (
            64 if head_size_rounded <= 96 else
            (
                (64 if (is_causal or is_local or softcap > 0.0) else 80) if head_size_rounded <= 128 else 64
            )
        )
    )

    # kBlockM_sm80
    kBlockM_sm80 = 128 if head_size_rounded <= 64 else 64

    # kBlockM_sm86
    kBlockM_sm86 = 64 if head_size_rounded <= 192 else 32

    # kBlockM
    kBlockM = (
        kBlockM_sm90 if arch >= 90 else
        (kBlockM_sm86 if arch == 86 or arch == 89 else kBlockM_sm80)
    )

    # kBlockN_sm90
    kBlockN_sm90 = (
        128 if head_size_rounded <= 128 else
        (96 if head_size_rounded <= 192 else 80)
    )

    # kBlockN_sm80
    kBlockN_sm80 = (
        128 if head_size_rounded <= 128 else
        (80 if head_size_rounded <= 192 else 64)
    )

    # kBlockN_sm86
    kBlockN_sm86 = (
        128 if head_size_rounded <= 64 else
        (
            128 if head_size_rounded <= 96 else
            (
                96 if head_size_rounded <= 128 else
                (
                    64 if head_size_rounded <= 192 else 64
                )
            )
        )
    )

    # kBlockN
    kBlockN = (
        kBlockN_sm90 if arch >= 90 else
        (kBlockN_sm86 if (arch == 86 or arch == 89) else kBlockN_sm80)
    )

    # Rounding calculations
    seqlen_q_rounded         = round_multiple(seqlen_q, kBlockM)
    seqlen_k_rounded         = round_multiple(seqlen_k, kBlockN)
    total_q_padded_rounded   = round_multiple(total_q + batch_size * kBlockM, kBlockM)
    total_k_padded_rounded   = round_multiple(total_k + batch_size * kBlockN, kBlockN)

    return seqlen_q_rounded, seqlen_k_rounded, total_q_padded_rounded, total_k_padded_rounded, head_size_rounded


@_torch_custom_op_wrapper("flash_attn_interface::_flash_attn_forward", mutates_args=("out",), device_types="cuda")
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor],
    v_new: Optional[torch.Tensor],
    qv: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cu_seqlens_k_new: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    page_table: Optional[torch.Tensor],
    kv_batch_idx: Optional[torch.Tensor],
    leftpad_k: Optional[torch.Tensor],
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    sink_token_length: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Ensure the relevant inputs are contiguous
    q, k, v, k_new, v_new = [maybe_contiguous(x) if x is not None else None for x in (q, k, v, k_new, v_new)]

    # Call the actual CUDA kernel.
    # Note: the kernel expects the two window_size values separately.
    out, softmax_lse, out_accum, softmax_lse_accum = flash_attn_3_cuda.fwd(
        q,
        k,
        v,
        k_new,
        v_new,
        qv,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        sink_token_length,
        softcap,
        rotary_interleaved,
        num_splits,
        pack_gqa,
        sm_margin,
    )
    return out, softmax_lse, out_accum, softmax_lse_accum


@_torch_register_fake_wrapper("flash_attn_interface::_flash_attn_forward")
def _flash_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor],
    v_new: Optional[torch.Tensor],
    qv: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cu_seqlens_k_new: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    page_table: Optional[torch.Tensor],
    kv_batch_idx: Optional[torch.Tensor],
    leftpad_k: Optional[torch.Tensor],
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    sink_token_length: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fake wrapper for torch.compile.
    Returns dummy tensors with shapes that match the CUDA kernel's outputs.
    For the normal non-varlen case:
      - out: (batch_size, seqlen_q, num_heads, head_size_v)
      - softmax_lse: (batch_size, num_heads, seqlen_q) in float32.
      - For num_splits > 1, dummy accumulators are produced.
    """
    out_dtype = torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype

    # if all of these are none, then we are in the normal non-varlen case (!varlen)
    if cu_seqlens_q is None and cu_seqlens_k is None and seqused_q is None and seqused_k is None and leftpad_k is None:
        batch_size, seqlen_q, num_heads, _ = q.shape
        head_size_v = v.shape[-1]
        if out is None:
            out = torch.empty((batch_size, seqlen_q, num_heads, head_size_v), device=q.device, dtype=out_dtype)
        softmax_lse = torch.empty((batch_size, num_heads, seqlen_q), device=q.device, dtype=torch.float32)
    else:
        total_q, num_heads, _ = q.shape
        head_size_v = v.shape[-1]
        if out is None:
            out = torch.empty((total_q, num_heads, head_size_v), device=q.device, dtype=out_dtype)
            softmax_lse = torch.empty((num_heads, total_q), device=q.device, dtype=torch.float32)

    if num_splits > 1:
        if cu_seqlens_q is None and cu_seqlens_k is None and seqused_q is None and seqused_k is None and leftpad_k is None:
            batch_size, seqlen_q, num_heads, _ = q.shape
            head_size_v = v.shape[-1]
            out_accum = torch.empty((num_splits, batch_size, num_heads, seqlen_q, head_size_v), device=q.device, dtype=torch.float32)
            softmax_lse_accum = torch.empty((num_splits, batch_size, num_heads, seqlen_q), device=q.device, dtype=torch.float32)
        else:
            total_q, num_heads, _ = q.shape
            head_size_v = v.shape[-1]
            out_accum = torch.empty((num_splits, num_heads, total_q, head_size_v), device=q.device, dtype=torch.float32)
            softmax_lse_accum = torch.empty((num_splits, num_heads, total_q), device=q.device, dtype=torch.float32)

        return out, softmax_lse, out_accum, softmax_lse_accum
    else:
        return out, softmax_lse, torch.Tensor(), torch.Tensor()


if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_forward = torch.ops.flash_attn_interface._flash_attn_forward
else:
    _wrapped_flash_attn_forward = _flash_attn_forward


@_torch_custom_op_wrapper("flash_attn_interface::_flash_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    sink_token_length: int = 0,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert sink_token_length == 0, "sink_token_length not supported yet"
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d, softmax_lse_log2, dq_accum, dk_accum, dv_accum = flash_attn_3_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        sink_token_length,
        softcap,
        deterministic,
        sm_margin,
    )
    return softmax_d, softmax_lse_log2, dq_accum, dk_accum, dv_accum


@_torch_register_fake_wrapper("flash_attn_interface::_flash_attn_backward")
def _flash_attn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int],
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    sink_token_length: int = 0,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    seqlen_q_rounded, seqlen_k_rounded, total_q_padded_rounded, total_k_padded_rounded, head_size_rounded = mha_bwd_shapes(
            q=q,
            k=k,
            softmax_lse=softmax_lse,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            softcap=softcap,
        )
    if cu_seqlens_q is None and cu_seqlens_k is None and seqused_q is None and seqused_k is None:
        batch_size, seqlen_q, num_heads, _ = q.shape
        softmax_d = torch.empty((batch_size, num_heads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        softmax_lse_log2 = torch.empty((batch_size, num_heads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
        dq_accum = torch.empty((batch_size, num_heads, seqlen_q_rounded * head_size_rounded), device=q.device, dtype=torch.float32)
    else:
        batch_size, num_heads, _ = q.shape
        softmax_d = torch.empty((num_heads, total_q_padded_rounded), device=q.device, dtype=torch.float32)
        softmax_lse_log2 = torch.empty((num_heads, total_q_padded_rounded), device=q.device, dtype=torch.float32)
        dq_accum = torch.empty((num_heads, total_q_padded_rounded * head_size_rounded), device=q.device, dtype=torch.float32)

    # compare the q and k num_heads to test if we have MQA/GQA
    if k.shape[-2] != q.shape[-2]:
        if cu_seqlens_q is None and cu_seqlens_k is None and seqused_q is None and seqused_k is None:
            batch_size, seqlen_k, num_heads_k, _ = k.shape
            dk_accum = torch.empty((batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded), device=k.device, dtype=torch.float32)
            dv_accum = torch.empty((batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded), device=v.device, dtype=torch.float32)
        else:
            batch_size, num_heads_k, _ = k.shape
            dk_accum = torch.empty((num_heads_k, total_k_padded_rounded, head_size_rounded), device=k.device, dtype=torch.float32)
            dv_accum = torch.empty((num_heads_k, total_k_padded_rounded, head_size_rounded), device=v.device, dtype=torch.float32)

        return softmax_d, softmax_lse_log2, dq_accum, dk_accum, dv_accum
    else:
        return softmax_d, softmax_lse_log2, dq_accum, torch.Tensor(), torch.Tensor()


if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_backward = torch.ops.flash_attn_interface._flash_attn_backward
else:
    _wrapped_flash_attn_backward = _flash_attn_backward


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        softmax_scale,
        causal,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        sink_token_length=0,
        softcap=0.0,
        deterministic=False,
        num_heads_q=None,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        if qkv.dim() == 5:
            assert qkv.shape[-3] == 3
            q, k, v = qkv.unbind(dim=-3)
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert num_heads_k * 2 + num_heads_q == qkv.shape[2]
            q, k, v = qkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)
        out, q, k, v, out_padded, softmax_lse = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            softmax_scale,
            causal=causal,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            sink_token_length=sink_token_length,
            softcap=softcap,
        )
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.sink_token_length = sink_token_length
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.ndim = qkv.dim()
        # return out, softmax_lse
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        if ctx.ndim == 5:
            qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.unbind(dim=-3)
        else:
            num_heads_q = q.shape[2]
            num_heads_k = k.shape[2]
            qkv_shape = q.shape[:-2] + (num_heads_q + num_heads_k * 2, *q.shape[-1:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)
        _wrapped_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.sink_token_length,
            ctx.softcap,
            ctx.deterministic,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        sink_token_length=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
    ):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        # out, q, k, v, out_padded, softmax_lse = _flash_attn_forward(
        out, softmax_lse, *rest = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            qv,  # qv
            None,  # out
            None,
            None,
            None,  # cu_seqlens_q/k/k_new
            None,
            None,  # seqused_q/k
            None,
            None,  # max_seqlen_q/k
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,  # rotary_cos/sin
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            sink_token_length=sink_token_length,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.sink_token_length = sink_token_length
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _wrapped_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,  # cu_seqlens_q, cu_seqlens_k,
            None,
            None,  # sequed_q, sequed_k,
            None,
            None,  # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.sink_token_length,
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        sink_token_length=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
    ):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        # out, q, k, v, out_padded, softmax_lse = _flash_attn_varlen_forward(
        out, softmax_lse, *rest = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            qv,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # cu_seqlens_k_new
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,  # rotary_cos/sin
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            sink_token_length=sink_token_length,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.sink_token_length = sink_token_length
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _wrapped_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.sink_token_length,
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def flash_attn_qkvpacked_func(
    qkv,
    softmax_scale=None,
    causal=False,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    sink_token_length=0,
    softcap=0.0,
    deterministic=False,
    num_heads_q=None,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
            the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        softmax_scale,
        causal,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        sink_token_length,
        softcap,
        deterministic,
        num_heads_q,
    )


def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    sink_token_length=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        sink_token_length,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    sink_token_length=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
):
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        sink_token_length,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
    )


def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None):
    return flash_attn_3_cuda.fwd_combine(out_partial, lse_partial, out, out_dtype)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    sink_token_length=0,
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a page_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim_v) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim_v) if there's a page_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim_v). Similar to k.
        qv [optional]: (batch_size, seqlen, nheads, headdim_v)
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        page_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_softmax_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert sink_token_length == 0
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full((k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device)
        cache_seqlens = maybe_contiguous(cache_seqlens)
    out, softmax_lse, *rest = _wrapped_flash_attn_forward(
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        None,  # out
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens,
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        cache_batch_idx,
        cache_leftpad,
        rotary_cos,
        rotary_sin,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal=causal,
        window_size=window_size,
        sink_token_length=sink_token_length,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )
    # return (out, softmax_lse) if return_softmax_lse else out
    return (out, softmax_lse, *rest) if return_softmax_lse else out
