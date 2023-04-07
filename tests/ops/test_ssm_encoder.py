from src.models.ssm_seq import (
    SSMEncoderModel,
    SSMDecoderModel,
    SSMModel,
    SSMForConditionalGeneration,
)
from src.models.ssm_config import SSMConfig
from flash_attn.modules.embedding import GPT2Embeddings
import torch
from argparse import ArgumentParser
from torch.cuda.amp import autocast

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-L_enc", "--encoder_sequence_length", type=int, default=4096)
    parser.add_argument("-L_dec", "--decoder_sequence_length", type=int, default=910)
    parser.add_argument("-H", "--d_model", type=int, default=720)
    parser.add_argument("-N", "--ssm_dim", type=int, default=256)
    parser.add_argument("-B", "--batch_size", type=int, default=16)
    parser.add_argument("-n", "--n_layer", type=int, default=3)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_reconstructs", type=int, default=8)

    args = parser.parse_args()
    l_decoder = args.decoder_sequence_length
    l_encoder = args.encoder_sequence_length
    H = args.d_model
    N = args.ssm_dim
    B = args.batch_size
    u = torch.randn(B, l_encoder, H, requires_grad=True)
    input_ids = torch.randint(0, args.vocab_size, (B, l_encoder))
    decoder_input_ids = torch.randint(0, args.vocab_size, (B, l_decoder))
    n_layer = args.n_layer
    n_heads = args.n_heads

    config = SSMConfig(
        d_model=H,
        d_inner=4 * H,
        d_state=N,
        n_layer=args.n_layer,
        use_fast_fftconv=True,
        vocab_size=args.vocab_size,
        n_reconstructs=args.n_reconstructs,
        num_heads=n_heads,
    )
    embeddings = GPT2Embeddings(
        config.d_model,
        vocab_size=config.vocab_size,
        max_position_embeddings=0,
    )

    pos_mask = torch.randint(0, l_encoder, (B, 1))
    attention_mask = torch.ones(B, l_encoder, dtype=torch.float32)
    indices = torch.arange(0, l_encoder).unsqueeze(0)
    attention_mask[indices <= pos_mask] = 0.0
    attention_mask = attention_mask.cuda()

    encoder = SSMEncoderModel(config, embeddings=embeddings)

    decoder = SSMDecoderModel(config, embeddings=embeddings)

    model = SSMModel(config)

    encoder.cuda()
    decoder.cuda()
    model.cuda()
    u = u.cuda()
    input_ids = input_ids.cuda()
    decoder_input_ids = decoder_input_ids.cuda()

    # encoder_output = encoder(embeddings=u, attention_mask=attention_mask)
    # encoder_output.last_hidden_state.mean().backward()
    # decoder_hidden_state = decoder(
    #     embeddings=u, encoder_hidden_state=encoder_output.last_hidden_state
    # )

    # model_decoder_output, model_encoder_output = model(
    #     input_ids=input_ids,
    #     decoder_input_ids=decoder_input_ids,
    #     attention_mask=attention_mask,
    # )
    # print("Encoder:", encoder_output.last_hidden_state.shape)
    # print("Decoder:", decoder_hidden_state.shape)
    # decoder_hidden_state.mean().backward()
    # model_decoder_output.mean().backward()

    input_ids = torch.randint(0, args.vocab_size, (B, l_encoder))
    decoder_input_ids = torch.randint(0, args.vocab_size, (B, l_decoder))

    input_ids = input_ids.cuda()
    decoder_input_ids = decoder_input_ids.cuda()

    model_lm = SSMForConditionalGeneration(config)
    try:
        # default to fused AdamW if apex is installed
        # based on this benchmark https://github.com/huggingface/transformers/issues/22101
        from apex.optimizers import FusedAdam

        optimizer_cls = FusedAdam
    except:
        from transformers import AdamW

        optimizer_cls = AdamW
    optimizer = optimizer_cls(
        [p for p in model_lm.parameters()],
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    model_lm.cuda()
    import ipdb

    ipdb.set_trace()
    with autocast(dtype=torch.bfloat16):
        logits = model_lm(
        input_ids=input_ids,
        labels=decoder_input_ids,
        attention_mask=attention_mask,
    )
    print(logits.encoder_last_hidden_state.last_hidden_state.shape)
    logits.loss.backward()
    optimizer.step()
