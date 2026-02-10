import torch


def get_attn_key_pad_mask(seq_k, seq_q, padding = 0):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(padding)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1, -1)  # b x lq x lk
    return padding_mask


def get_causal_mask_original(seq, dim=2):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()[:2]
    subsequent_mask = torch.triu(
        torch.ones((dim, len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1).permute(1, 2, 0)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1, -1)  # b x ls x ls
    return subsequent_mask


def get_causal_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()[:2]
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device = seq.device, dtype = torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)        # b x ls x ls
    return subsequent_mask