# Copyright (c) 2023, Tri Dao, Dan Fu.

import math
from functools import partial


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers import PreTrainedModel

from flash_attn.modules.mlp import Mlp, FusedMLP
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import GPT2Embeddings
from src.models.ssm.cross_attn import MHACrossAttn
from src.models.ssm_config import SSMConfig
from src.utils.utils import shift_tokens_right

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

from src.models.ssm.h3 import H3, H3Expand
from src.models.custom_block import HiddenStateBlock


def create_mixer_cls(
    ssm_cls=H3,
    ssm_cfg=None,
    attn_layer_idx=None,
    num_heads=1,
    layer_idx=None,
):
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        mixer_cls = partial(
            MHACrossAttn,
            layer_idx=layer_idx,
            num_heads=num_heads,
        )
    else:
        mixer_cls = partial(
            ssm_cls,
            layer_idx=layer_idx,
            num_heads=num_heads,
            **(ssm_cfg if ssm_cfg is not None else {}),
        )
    return mixer_cls


def create_mlp_cls(d_model, d_inner=None, fused_mlp=False):
    inner_dim = d_inner if d_inner is not None else 4 * d_model
    if not fused_mlp:
        mlp_cls = partial(
            Mlp,
            hidden_features=inner_dim,
            activation=partial(F.gelu, approximate="tanh"),
        )
    else:
        mlp_cls = partial(FusedMLP, hidden_features=inner_dim)
    return mlp_cls


def create_block(
    config: SSMConfig,
    attn_layer_idx=None,
    layer_idx=None,
    last_layer=False,
    first_layer=False,
):
    if last_layer:
        ssm_cls = H3Expand
        ssm_cfg = {
            "disc": config.disc,
            "d_state": config.d_state,
            "use_fast_fftconv": config.use_fast_fftconv,
            "n_reconstructs": config.n_reconstructs,
        }
    else:
        ssm_cls = H3
        ssm_cfg = {
            "disc": config.disc,
            "d_state": config.d_state,
            "use_fast_fftconv": config.use_fast_fftconv,
        }
    mixer_cls = create_mixer_cls(
        ssm_cls=ssm_cls,
        layer_idx=layer_idx,
        ssm_cfg=ssm_cfg,
        attn_layer_idx=attn_layer_idx,
    )
    mlp_cls = create_mlp_cls(
        config.d_model, d_inner=config.d_inner, fused_mlp=config.fused_mlp
    )
    norm_cls = partial(nn.LayerNorm, eps=config.layer_norm_epsilon)
    resid_dropout1 = config.embed_dropout if first_layer else config.resid_dropout
    if last_layer:
        block = HiddenStateBlock(
            config.d_model,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            resid_dropout1=resid_dropout1,
            resid_dropout2=config.resid_dropout,
        )
    else:
        block = Block(
            config.d_model,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            prenorm=True,
            resid_dropout1=resid_dropout1,
            resid_dropout2=config.resid_dropout,
            fused_dropout_add_ln=config.fused_dropout_add_ln,
            residual_in_fp32=config.residual_in_fp32,
        )
    block.layer_idx = layer_idx
    block.is_last_layer = last_layer
    block.is_first_layer = first_layer
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    glu_act=False,
):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer)
                )
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if not glu_act:
                    nn.init.normal_(
                        p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer)
                    )
                else:
                    out_features = p.shape[0]
                    # Multiplying the first half of the matrix by 2 since sigmoid scales it down by 0.5
                    # on average.
                    nn.init.normal_(
                        p[: out_features // 2],
                        mean=0.0,
                        std=initializer_range / math.sqrt(2 * n_layer) * 2,
                    )


class SSMEncoderModel(nn.Module):
    def __init__(
        self,
        config: SSMConfig,
        embeddings,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.residual_in_fp32 = config.residual_in_fp32

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.fused_dropout_add_ln = config.fused_dropout_add_ln
        if self.fused_dropout_add_ln and dropout_add_layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")
        list_modules = [
            create_block(
                config,
                layer_idx=i,
                first_layer=i == 0,
                last_layer=False,
            )
            for i in range(config.n_layer - 1)
        ]
        last_layer = create_block(
            config,
            last_layer=True,
            layer_idx=config.n_layer - 1,
        )
        list_modules.append(last_layer)
        self.layers = nn.ModuleList(list_modules)

        self.drop_f = nn.Dropout(config.resid_dropout)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
            )
        )

    def forward(
        self,
        input_ids=None,
        embeddings=None,
        position_ids=None,
        attention_mask=None,
        **kwargs,  # To absorb the kwargs of generation module.
    ):
        if input_ids is not None and embeddings is None:
            hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        elif input_ids is None and embeddings is not None:
            hidden_states = embeddings
        else:
            raise AttributeError("Either input_ids or embeddings ha to be supplied.")
        residual = None
        mixer_kwargs = None

        # Bad but we need to make sure that attention mask is supplied for correct zero-padding.
        assert attention_mask is not None
        for layer in self.layers:
            # Left-zero padding.
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            if layer.is_last_layer:
                hidden_states = layer(
                    hidden_states, residual=None, mixer_kwargs=mixer_kwargs
                )
                residual = None
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, mixer_kwargs=mixer_kwargs
                )
        if not self.fused_dropout_add_ln:
            dropped = self.drop_f(hidden_states)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = dropout_add_layer_norm(
                hidden_states,
                residual,
                self.ln_f.weight,
                self.ln_f.bias,
                self.drop_f.p if self.training else 0.0,
                self.ln_f.eps,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=hidden_states,
        )


class SSMDecoderModel(nn.Module):
    def __init__(
        self,
        config: SSMConfig,
        embeddings,
    ) -> None:
        super().__init__()
        self.embeddings = embeddings
        self.config = config

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.fused_dropout_add_ln = config.fused_dropout_add_ln
        if self.fused_dropout_add_ln and dropout_add_layer_norm is None:
            raise ImportError("dropout_add_layer_norm is not installed")
        self.attn_layer_idx = [
            i for i in range(1, 2 * self.config.n_layer, 2)
        ]  # One attention layer between each SSM.
        self.residual_in_fp32 = config.residual_in_fp32

        list_modules = [
            create_block(
                self.config,
                attn_layer_idx=self.attn_layer_idx,
                # resid_dropout1=embed_dropout if i == 0 else resid_dropout,
                layer_idx=i,
                first_layer=i == 0,
                last_layer=False,  # Last layer is standard in the decoder.
            )
            for i in range(
                2 * self.config.n_layer
            )  # Per block one SSM + reconstructed attention
        ]
        self.layers = nn.ModuleList(list_modules)

        self.drop_f = nn.Dropout(config.resid_dropout)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
            )
        )

    def forward(
        self,
        input_ids=None,
        embeddings=None,
        encoder_hidden_state=None,
        position_ids=None,
    ):
        if input_ids is not None and embeddings is None:
            hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        elif input_ids is None and embeddings is not None:
            hidden_states = embeddings
        else:
            raise AttributeError("Either input_ids or embeddings have to be supplied.")
        residual = None

        for layer in self.layers:
            if layer.layer_idx in self.attn_layer_idx:
                mixer_kwargs = {"x_kv": encoder_hidden_state}
            else:
                mixer_kwargs = {}
            hidden_states, residual = layer(
                hidden_states, residual, mixer_kwargs=mixer_kwargs
            )
        if not self.fused_dropout_add_ln:
            dropped = self.drop_f(hidden_states)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = dropout_add_layer_norm(
                hidden_states,
                residual,
                self.ln_f.weight,
                self.ln_f.bias,
                self.drop_f.p if self.training else 0.0,
                self.ln_f.eps,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


class SSMPretrainedModel(PreTrainedModel):
    config_class = SSMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False


class SSMModel(SSMPretrainedModel):
    def __init__(self, config: SSMConfig) -> None:
        super().__init__(config)
        self.config = config
        if config.shared_embeddings:
            if self.config.use_positional_embeddings:
                args_position_embeddings = self.config.max_position_embeddings
            else:
                args_position_embeddings = 0
            shared_embeddings = GPT2Embeddings(
                self.config.d_model,
                vocab_size=self.config.vocab_size,
                max_position_embeddings=args_position_embeddings,
            )
            encoder_embeddings = shared_embeddings
            decoder_embeddings = shared_embeddings
        else:
            encoder_embeddings = GPT2Embeddings(
                self.config.d_model,
                vocab_size=self.config.vocab_size,
                max_position_embeddings=args_position_embeddings,
            )
            decoder_embeddings = GPT2Embeddings(
                self.config.d_model,
                vocab_size=self.config.vocab_size,
                max_position_embeddings=args_position_embeddings,
            )

        self.encoder = SSMEncoderModel(
            config,
            encoder_embeddings,
        )
        self.decoder = SSMDecoderModel(
            config,
            decoder_embeddings,
        )

    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        if decoder_input_ids is None:
            if (input_ids is None) and (encoder_outputs is None):
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)

        decoder_output = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_state=encoder_outputs.last_hidden_state,
        )
        return decoder_output, encoder_outputs


class SSMForConditionalGeneration(SSMPretrainedModel):
    def __init__(self, config: SSMConfig):
        super().__init__(config)
        self.config = config
        self.model = SSMModel(config=config)
        self.register_buffer("final_logits_bias", torch.zeros((1, config.vocab_size)))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def pad_inputs(self, input_ids, pad_token_id, L=None):
        """Padding for autoregressive generation."""
        if L is None:
            L = self.config.max_position_embeddings
        gap = L - input_ids.shape[1]
        return F.pad(input_ids, (0, gap), mode="constant", value=pad_token_id)

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method.
        """

        return {
            "input_ids": None,
            "decoder_input_ids": input_ids,
            "encoder_outputs": kwargs["encoder_outputs"],
        }

    def forward(
        self,
        input_ids=None,
        labels=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        if labels is not None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        decoder_outputs, encoder_hidden_state = self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
        )

        lm_logits = self.lm_head(decoder_outputs) + self.final_logits_bias

        if input_ids is None:
            # for generation
            lm_logits = lm_logits

        masked_lm_loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            encoder_last_hidden_state=encoder_hidden_state,
        )
