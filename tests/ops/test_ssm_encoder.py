from src.models.ssm_seq import SSMEncoderModel, SSMDecoderModel, SSMModel
from src.models.ssm_config import SSMConfig
from flash_attn.modules.embedding import GPT2Embeddings
import torch
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-L_enc", "--encoder_sequence_length", type=int, default=128)
    parser.add_argument("-L_dec", "--decoder_sequence_length", type=int, default=32)
    parser.add_argument("-H", "--d_model", type=int, default=64)
    parser.add_argument("-N", "--ssm_dim", type=int, default=32)
    parser.add_argument("-B", "--batch_size", type=int, default=4)
    parser.add_argument("-n", "--n_layer", type=int, default=3)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_reconstructs", type=int, default=4)

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
        use_fast_fftconv=True,
        vocab_size=args.vocab_size,
        n_reconstructs=args.n_reconstructs,
    )
    embeddings = GPT2Embeddings(
        config.d_model,
        vocab_size=config.vocab_size,
        max_position_embeddings=0,
    )

    encoder = SSMEncoderModel(config, embeddings=embeddings)

    decoder = SSMDecoderModel(config, embeddings=embeddings)

    model = SSMModel(config)

    encoder.cuda()
    decoder.cuda()
    model.cuda()
    u = u.cuda()
    input_ids = input_ids.cuda()
    decoder_input_ids = decoder_input_ids.cuda()

    encoder_output = encoder(embeddings=u)
    decoder_hidden_state = decoder(
        embeddings=u, encoder_hidden_state=encoder_output.last_hidden_state
    )

    model_output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    print("Encoder:", encoder_output.last_hidden_state.shape)
    print("Decoder:", decoder_hidden_state.shape)
    decoder_hidden_state.mean().backward()
    model_output.mean().backward()
