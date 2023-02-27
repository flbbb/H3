import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
except ImportError:
    flash_attn_unpadded_qkvpacked_func, flash_attn_unpadded_kvpacked_func = None, None

try:
    from flash_attn.ops.flash_attn_triton import (
        flash_attn_qkvpacked_func,
        flash_attn_kvpacked_func,
    )
except ImportError:
    flash_attn_qkvpacked_func, flash_attn_kvpacked_func = None, None

try:
    from flash_attn.ops.fused_dense import (
        FusedDense,
        ColumnParallelLinear,
        RowParallelLinear,
    )
except ImportError:
    FusedDense, ColumnParallelLinear, RowParallelLinear = None, None, None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

try:
    import ft_attention
except ImportError:
    ft_attention = None
from flash_attn.modules.mha import (
    LinearResidual,
    FlashCrossAttention,
    CrossAttention,
    _update_kv_cache,
)


class MHACrossAttn(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        dropout=0.0,
        softmax_scale=None,
        layer_idx=None,
        dwconv=False,
        rotary_emb_dim=0,
        rotary_emb_scale_base=0,
        fused_bias_fc=False,
        use_flash_attn=False,
        return_residual=False,
        checkpointing=False,
        device=None,
        dtype=None,
    ) -> None:
        """
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx
        self.dwconv = dwconv
        self.rotary_emb_dim = rotary_emb_dim
        self.use_flash_attn = use_flash_attn
        self.return_residual = return_residual
        self.checkpointing = checkpointing

        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads

        if self.rotary_emb_dim > 0:
            assert (
                False
            ), "MHA with rotary embedding does not support cross-attention yet"

        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        linear_resid_cls = (
            LinearResidual
            if not fused_bias_fc
            else partial(FusedDense, return_residual=True)
        )
        inner_cross_attn_cls = FlashCrossAttention if use_flash_attn else CrossAttention
        self.Wq = linear_cls(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        if not self.return_residual:
            self.Wkv = linear_cls(embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)
        else:
            self.Wkv = linear_resid_cls(
                embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs
            )
        if self.dwconv:
            self.dwconv_q = nn.Conv1d(
                embed_dim, embed_dim, kernel_size=3, padding=2, groups=embed_dim
            )
            self.dwconv_kv = nn.Conv1d(
                2 * embed_dim,
                2 * embed_dim,
                kernel_size=3,
                padding=2,
                groups=2 * embed_dim,
            )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=False, softmax_scale=softmax_scale, attention_dropout=dropout
        )
        # output projection always have the bias (for now)
        self.out_proj = linear_cls(embed_dim, embed_dim, **factory_kwargs)

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert not self.dwconv, "Generation does not support dwconv yet"
        assert (
            self.layer_idx is not None
        ), "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def forward(
        self,
        x,
        x_kv,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.use_flash_attn
            assert not self.dwconv
            assert self.rotary_emb_dim == 0
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
            assert not self.use_flash_attn
        if inference_params is not None:
            assert key_padding_mask is None
            assert cu_seqlens is None and max_seqlen is None
            assert not self.dwconv

        kwargs = (
            {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen, **kwargs}
            if self.use_flash_attn
            else {"key_padding_mask": key_padding_mask, **kwargs}
        )
        if not self.return_residual:
            q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
            kv = self.Wkv(x_kv if x_kv is not None else x)
        else:
            if x_kv is not None:
                kv, x_kv = self.Wkv(x_kv)
            else:
                kv, x = self.Wkv(x)
            q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(kv, "... (two h d) -> ... two h d", two=2, d=self.head_dim)
        if self.dwconv:
            q = rearrange(
                self.dwconv_q(rearrange(q, "b s d -> b d s"))[..., :-2],
                "b d s -> b s d",
            ).contiguous()
            kv = rearrange(
                self.dwconv_kv(rearrange(kv, "b s d -> b d s"))[..., :-2],
                "b d s -> b s d",
            ).contiguous()
        if inference_params is None:
            if not self.checkpointing:
                context = self.inner_cross_attn(q, kv, **kwargs)
            else:
                context = torch.utils.checkpoint.checkpoint(
                    self.inner_cross_attn, q, kv, **kwargs
                )
        else:
            kv = self._update_kv_cache(kv)
            context = self.inner_cross_attn(q, kv, causal=False)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)
