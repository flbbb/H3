from transformers import PretrainedConfig


class SSMConfig(PretrainedConfig):
    model_type = "ssm"

    def __init__(
        self,
        d_model=320,
        d_state=32,
        n_layer=4,
        d_inner=1280,
        vocab_size=96103,
        num_heads=4,
        n_reconstructs=8,
        max_position_embeddings=8192,
        embed_dropout=0.1,
        resid_dropout=0.0,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=1,
        forced_eos_token_id=1,
        decoder_start_token_id=0,
        shared_embeddings=True,
        use_positional_embeddings=False,
        fused_dropout_add_ln=False,
        use_fast_fftconv=False,
        fused_mlp=False,
        residual_in_fp32=False,
        bidirectional=False,
        use_cross_attention=False,
        layer_norm_epsilon=1e-5,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_state = d_state
        self.n_layer = n_layer
        self.max_position_embeddings = max_position_embeddings
        self.d_inner = d_inner  # 4 * d_model in reference implementation
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_reconstructs = n_reconstructs
        self.shared_embeddings = shared_embeddings
        self.embed_dropout = embed_dropout
        self.resid_dropout = resid_dropout
        self.use_positional_embeddings = use_positional_embeddings
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.residual_in_fp32 = residual_in_fp32
        self.fused_mlp = fused_mlp
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_fast_fftconv = use_fast_fftconv
        self.disc = "zoh"
        self.bidirectional = bidirectional
        self.use_cross_attention=use_cross_attention

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
        self.is_encoder_decoder = True
