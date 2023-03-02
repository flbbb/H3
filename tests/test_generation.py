from src.models.ssm_seq import SSMForConditionalGeneration, SSMModel
from src.models.ssm_config import SSMConfig
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
    model = SSMForConditionalGeneration(config)

    model.cuda()
    input_ids = input_ids.cuda()
    generation = model.generate(input_ids=input_ids, num_beams=3)
    print(generation)
