# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .solve_tril import solve_tril
from .utils import FLA_CHUNK_SIZE, SUPPRESS_LEVEL, input_guard
from .wy_fast import recompute_w_u_fwd


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,
):
    g = chunk_local_cumsum(
        g, chunk_size=FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        core_attn_out=core_attn_out,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        core_attn_out: torch.Tensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        if core_attn_out is not None:
            assert not torch.is_grad_enabled(), (
                "core_attn_out buffer reuse is only supported for inference"
            )
            assert q.dtype == o.dtype, "Incompatible dtype for inplace computation"
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    core_attn_out: torch.Tensor | None = None,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) Gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, V, K, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."

    # [FA2/SM70 -- НАТИВНАЯ ВЕТКА СКАНА, БЕЗ ПИТОН-ПРОКЛАДОК ПОВЕРХ ИМПОРТОВ]
    # Прежде наш скан вставал шимом (sitecustomize + подмена имени поверх импортов) -- это
    # прокладка, и курс проекта её запрещает. Ветка живёт ЗДЕСЬ, в самом определении: путь
    # один, искать нечего. Голова цепочки (l2norm + внутричанковый кумсум) остаётся
    # трионовской (0.8% скана, ни одного матричного умножения); дальше -- наши HMMA-ядра
    # (kkt -> solve_tril -> сшитое h+o+wu, fa2_sm70/gdn_scan.cu). Включение: FA2SM70_GDN=1.
    # Откат ТОЛЬКО по форме; нехватка памяти отдаётся движку (запасной Triton-путь ТЯЖЕЛЕЕ
    # основного и включается ровно когда памяти нет -- урок 11.08).
    import os as _os
    if _os.environ.get("FA2SM70_GDN", "0") == "1":
        try:
            from fa2_sm70 import gdn as _fa2_gdn
            if not _fa2_gdn.supported(q, k, v, g, beta, initial_state, cu_seqlens):
                raise NotImplementedError("форма вне инстанцированных")
            _scale = k.shape[-1] ** -0.5 if scale is None else scale
            _qn, _kn = (l2norm_fwd(q), l2norm_fwd(k)) if use_qk_l2norm_in_kernel else (q, k)
            _gg = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
            _o, _ht = _fa2_gdn.chunk_gated_delta_rule_volta(
                _qn.to(torch.float16).contiguous(), _kn.to(torch.float16).contiguous(),
                v, _gg, beta, float(_scale), initial_state, bool(output_final_state),
                cu_seqlens, v.dtype)
            if core_attn_out is not None:
                # [FA2/SM70] upstream contract: o is written into the caller's buffer.
                _dst = core_attn_out.view(-1)[: _o.numel()].view_as(_o)
                _dst.copy_(_o)
                _o = _dst
            global _FA2SM70_SCAN_OURS
            try:
                _FA2SM70_SCAN_OURS += 1
            except NameError:
                _FA2SM70_SCAN_OURS = 1
            if _FA2SM70_SCAN_OURS & (_FA2SM70_SCAN_OURS - 1) == 0:
                import sys as _sys
                print(f"[fa2_sm70 gdn] НАТИВНЫЙ СКАН отработал: {_FA2SM70_SCAN_OURS}",
                      file=_sys.stderr, flush=True)
            return _o, _ht
        except torch.OutOfMemoryError:
            raise
        except Exception as _e:  # noqa: BLE001
            if _os.environ.get("FA2SM70_GDN_STRICT", "0") == "1":
                raise RuntimeError(
                    f"[fa2_sm70] СТРОГИЙ РЕЖИМ: скан GDN не пошёл нашим ядром: "
                    f"{type(_e).__name__}: {_e}") from _e
            global _FA2SM70_SCAN_FB
            try:
                _FA2SM70_SCAN_FB += 1
            except NameError:
                _FA2SM70_SCAN_FB = 1
            if _FA2SM70_SCAN_FB == 1:
                import sys as _sys
                print(f"[fa2_sm70 gdn] нативный скан: откат на Triton: "
                      f"{type(_e).__name__}: {_e}", file=_sys.stderr, flush=True)

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        use_qk_l2norm_in_kernel,
        core_attn_out,
    )
    return o, final_state
