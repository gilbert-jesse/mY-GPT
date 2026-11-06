from data_cleanup import load_and_clean_chat
import torch


def tokenizer(filepath):

    text = load_and_clean_chat(filepath)

    # Build character vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Character <-> integer mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encoder and decoder
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # Encode entire dataset
    data = torch.tensor(
        encode(text),
        dtype=torch.long
    )

    # Sanity check
    assert decode(encode(text)) == text

    return data, encode, decode, vocab_size