import torch


def get_causal_mask(seq_len, device):
    """For masking out the subsequent info, i.e., masked self-attention.
    In our case, True means item kept, and False means item masked.
    """

    subsequent_mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    # [seq_len, seq_len]
    return subsequent_mask.unsqueeze(dim=0)  # [1, seq_len, seq_len]
