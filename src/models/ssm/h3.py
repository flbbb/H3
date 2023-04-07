import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from src.models.ssm.ss_kernel import SSKernel, SSKernelExpand

try:
    from src.ops.fftconv import fftconv_func
except ImportError:
    fftconv_func = None


@torch.jit.script
def mul_sum(q, y):
    return (q * y).sum(dim=1)


def make_bidirectional(ssm_kernel, L):
    k_direct, k_reversed = torch.tensor_split(ssm_kernel, 2, dim=0)

    return (
        F.pad(k_direct.contiguous(), (0, L)).contiguous()
        + torch.roll(
            F.pad(k_reversed.flip(-1).contiguous(), (L, 0)), 1, dims=-1
        ).contiguous()
    )


class H3(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        l_max=None,
        num_heads=1,
        use_fast_fftconv=True,
        bidirectional=False,
        dropout=0.0,  # Just to absorb the kwarg
        layer_idx=None,
        device=None,
        dtype=None,
        # SSM Kernel arguments
        **kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L. Set l_max=None to always use a global kernel

        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert (num_heads == 8) or (num_heads == 1)
        assert d_model % num_heads == 0
        self.bidirectional = bidirectional
        self.H = d_model // num_heads
        self.N = d_state
        self.L = l_max
        self.layer_idx = layer_idx
        self.use_fast_fftconv = use_fast_fftconv
        if self.use_fast_fftconv:
            assert fftconv_func is not None, "Need to install fftconv"

        self.q_proj = nn.Linear(self.d_model, self.d_model, **factory_kwargs)
        self.k_proj = nn.Linear(self.d_model, self.d_model, **factory_kwargs)
        self.v_proj = nn.Linear(self.d_model, self.d_model, **factory_kwargs)

        # TODO: SSKernel doesn't take device argument yet
        if self.bidirectional:
            channels = 2
        else:
            channels = 1
        self.ssm_k_kernel = SSKernel(
            self.d_model,
            N=d_state,
            L=self.L,
            mode="shift",
            channels=channels,
            lr=kernel_args.get("lr", None),
        )
        self.ssm_k_D = nn.Parameter(torch.randn(self.d_model))
        # S4D Kernel
        self.kernel = SSKernel(
            self.H, N=self.N, L=self.L, channels=channels, **kernel_args
        )
        self.D = nn.Parameter(torch.randn(self.H, **factory_kwargs))

        # Pointwise
        # position-wise output transform to mix features
        # Don't use FusedDense since the layout is H first
        self.output_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, u):
        """
        u: (B L H)

        Returns: same shape as u
        """
        L_og = u.size(-2)
        if self.use_fast_fftconv and L_og % 2 != 0:
            u = F.pad(u, (0, 0, 0, 1))
        L = u.size(-2)

        use_fast_fftconv = self.use_fast_fftconv

        state_k, state = None, None

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, self.L)
        ssm_kernel, k_state = self.kernel(
            L=L_kernel, state=state, rate=1.0
        )  # (C H L) (B C H L)

        u = rearrange(u, "b l h -> (b l) h")
        dtype = (
            self.q_proj.weight.dtype
            if not torch.is_autocast_enabled()
            else torch.get_autocast_gpu_dtype()
        )
        q = self.q_proj.weight @ u.T + self.q_proj.bias.to(dtype).unsqueeze(-1)
        k = self.k_proj.weight @ u.T + self.k_proj.bias.to(dtype).unsqueeze(-1)
        v = self.v_proj.weight @ u.T + self.v_proj.bias.to(dtype).unsqueeze(-1)
        q, k, v = [rearrange(x, "h (b l) -> b h l", l=L) for x in [q, k, v]]

        k_og = k
        ssm_k_kernel, _ = self.ssm_k_kernel(
            L=L_kernel, state=state_k, rate=1.0
        )  # (C H L) (B C H L)
        if self.bidirectional:
            ssm_k_kernel = make_bidirectional(ssm_k_kernel, L)
            ssm_kernel = make_bidirectional(ssm_kernel, L)
        ssm_k_kernel = rearrange(ssm_k_kernel, "1 h l -> h l")
        ssm_kernel = rearrange(ssm_kernel, "1 h l -> h l")

        if not use_fast_fftconv:
            fft_size = L_kernel + L

            ssm_k_kernel_f = torch.fft.rfft(ssm_k_kernel, n=fft_size)  # (H 2L)
            k_f = torch.fft.rfft(k.to(ssm_kernel.dtype), n=fft_size)  # (B H 2L)
            shift_k_out = torch.fft.irfft(ssm_k_kernel_f * k_f, n=fft_size)[..., :L]
            k = shift_k_out + rearrange(self.ssm_k_D, "h -> h 1") * k
        else:
            dropout_mask = None
            # No GeLU after the SSM
            # We want output_hbl=True so that k has the same layout as q and v for the next
            # fftconv
            k = fftconv_func(
                k, ssm_k_kernel, self.ssm_k_D, dropout_mask, False, False, True
            )
            # This line below looks like it doesn't do anything, but it gets the stride right
            # for the case batch_size=1. In that case k has stride (L, L, 1), but q and v has
            # stride (H * L, L, 1). The two strides are equivalent because batch_size=1, but
            # the C++ code doesn't like that.
            k = rearrange(rearrange(k, "b h l -> h b l"), "h b l -> b h l")

        if not use_fast_fftconv:
            fft_size = L_kernel + L
            # kv = k * v
            kv = rearrange(
                k, "b (h d1) l -> b d1 1 h l", d1=self.num_heads
            ) * rearrange(
                v, "b (h d2) l -> b 1 d2 h l", d2=self.num_heads
            )  # b d1 d2 h l
            kv_f = torch.fft.rfft(kv.to(dtype=ssm_kernel.dtype), n=fft_size) / fft_size
            ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)  # h L+1
            y = torch.fft.irfft(kv_f * ssm_kernel_f, n=fft_size, norm="forward")[
                ..., :L
            ]  # b d1 d2 h l
            y = y + kv * self.D.unsqueeze(-1)  # b d1 d2 h l
            q = rearrange(q, "b (h d1) l -> b d1 1 h l", d1=self.num_heads)
            # einsum is way slower than multiply and then sum.
            if self.num_heads > 1:
                y = mul_sum(y, q)
                y = rearrange(y, "b d h l -> b (d h) l")
            else:
                y = rearrange(y * q, "b 1 1 h l -> b h l")
        else:
            dropout_mask = None
            # No GeLU after the SSM
            # Set output_hbl_layout=True since we'll be doing a matmul right after
            y = fftconv_func(
                k,
                ssm_kernel,
                self.D,
                dropout_mask,
                False,
                torch.is_autocast_enabled(),
                True,
                v,
                self.num_heads,
                q,
            )

        y = rearrange(y, "b h l -> b l h")

        # y could be in fp32 because of the SSMs
        if not torch.is_autocast_enabled():
            y = y.to(dtype=self.output_linear.weight.dtype)
        y = self.output_linear(y)
        if L_og < L:
            y = y[:, :L_og, :]

        return y


class H3Expand(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        n_reconstructs=8,
        l_max=None,
        num_heads=1,
        use_fast_fftconv=False,
        dropout=0.0,  # Just to absorb the kwarg
        layer_idx=None,
        device=None,
        dtype=None,
        # SSM Kernel arguments
        **kernel_args,
    ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum kernel length, also denoted by L. Set l_max=None to always use a global kernel

        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.H = d_model // num_heads
        self.N = d_state
        self.L = l_max
        self.layer_idx = layer_idx
        self.use_fast_fftconv = use_fast_fftconv
        self.n_reconstructs = n_reconstructs
        if self.use_fast_fftconv:
            assert fftconv_func is not None, "Need to install fftconv"

        self.k_proj = nn.Linear(self.d_model, self.d_model, **factory_kwargs)

        # TODO: SSKernel doesn't take device argument yet
        self.ssm_k_kernel = SSKernel(
            self.d_model,
            N=d_state,
            L=self.L,
            mode="shift",
            lr=kernel_args.get("lr", None),
        )
        self.ssm_k_D = nn.Parameter(torch.randn(self.d_model))
        # S4D Kernel
        self.kernel_expand = SSKernelExpand(
            self.H,
            N=self.N,
            L=self.L,
            n_reconstructs=n_reconstructs,
            channels=1,
            num_heads=self.num_heads,
            **kernel_args,
        )

        # Pointwise
        # position-wise output transform to mix features
        # Don't use FusedDense since the layout is H first
        # self.output_linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, u):
        """
        u: (B L H)

        No need for inference params in encoder.

        Returns: same shape as u
        """
        L_og = u.size(-2)
        if self.use_fast_fftconv and L_og % 2 != 0:
            u = F.pad(u, (0, 0, 0, 1))
        L = u.size(-2)

        use_fast_fftconv = self.use_fast_fftconv

        state_k, state = None, None

        # Compute SS Kernel
        L_kernel = L if self.L is None else min(L, self.L)
        # ssm_kernel, k_state = self.kernel(L=L_kernel, state=state, rate=1.0) # (C H L) (B C H L)
        # ssm_kernel = rearrange(ssm_kernel, '1 h l -> h l')

        u = rearrange(u, "b l h -> (b l) h")
        dtype = (
            self.k_proj.weight.dtype
            if not torch.is_autocast_enabled()
            else torch.get_autocast_gpu_dtype()
        )

        k = self.k_proj.weight @ u.T + self.k_proj.bias.to(dtype).unsqueeze(-1)
        k = rearrange(k, "h (b l) -> b h l", l=L)
        # rearrange(k, "b (h d1) l -> b d1 1 h l", d1=self.num_heads)
        ssm_k_kernel, _ = self.ssm_k_kernel(
            L=L_kernel, state=state_k, rate=1.0
        )  # (C H L) (B C H L)
        ssm_k_kernel = rearrange(ssm_k_kernel, "1 h l -> h l")
        if not use_fast_fftconv:
            fft_size = L_kernel + L
            ssm_k_kernel_f = torch.fft.rfft(ssm_k_kernel, n=fft_size)  # (H 2L)
            k_f = torch.fft.rfft(k.to(ssm_k_kernel.dtype), n=fft_size)  # (B H 2L)
            shift_k_out = torch.fft.irfft(ssm_k_kernel_f * k_f, n=fft_size)[..., :L]
            k = shift_k_out + rearrange(self.ssm_k_D, "h -> h 1") * k
        else:
            dropout_mask = None
            # No GeLU after the SSM
            # We want output_hbl=True so that k has the same layout as q and v for the next
            # fftconv
            k = fftconv_func(
                k, ssm_k_kernel, self.ssm_k_D, dropout_mask, False, False, True
            )
            # This line below looks like it doesn't do anything, but it gets the stride right
            # for the case batch_size=1. In that case k has stride (L, L, 1), but q and v has
            # stride (H * L, L, 1). The two strides are equivalent because batch_size=1, but
            # the C++ code doesn't like that.
            k = rearrange(rearrange(k, "b h l -> h b l"), "h b l -> b h l")

        k = rearrange(k, "b (h d1) l -> b d1 h l", d1=self.num_heads)
        with torch.autocast(enabled=False, device_type="cuda"):
            hidden_state = self.kernel_expand(u=k.float(), rate=1.0, L=L)  # (B r H)

        # hidden_state could be in fp32 because of the SSMs
        # if not torch.is_autocast_enabled():
        #     hidden_state = hidden_state.to(dtype=self.output_linear.weight.dtype)
        # hidden_state = self.output_linear(hidden_state.mT).mT

        return hidden_state


# class H3DecoderLayer(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         d_state=64,
#         l_max=None,
#         num_heads=1,
#         use_fast_fftconv=False,
#         use_flash_attn=False,
#         dropout=0.0,  # Just to absorb the kwarg
#         layer_idx=None,
#         device=None,
#         dtype=None,
#         # SSM Kernel arguments
#         **kernel_args,
#     ):
#         self.h3 = H3(
#             d_model=d_model,
#             d_state=d_state,
#             l_max=l_max,
#             num_heads=num_heads,
#             use_fast_fftconv=use_fast_fftconv,
#             layer_idx=layer_idx,
#             device=device,
#             dtype=dtype,
#             **kernel_args
#         )
#         if use_flash_attn:
#             self.attn_layer = None
