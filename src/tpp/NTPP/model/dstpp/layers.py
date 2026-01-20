import torch.nn as nn

from src.tpp.tpp_models.dstpp.self_attn import MultiheadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, device, dropout = 0.1, normalize_before = True):
        super(EncoderLayer, self).__init__()
        self.device = device

        self.slf_attn = MultiheadAttention(
            n_head, d_model, d_k, d_v, dropout = dropout, device = self.device)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, device = self.device)


    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)               # [batch_size, seq_len, d_model] + # [batch_size, n_head, seq_len, seq_len]
        enc_output *= non_pad_mask.unsqueeze(dim = -1)                         # [batch_size, seq_len, d_model]

        enc_output = self.pos_ffn(enc_output)                                  # [batch_size, seq_len, d_model]
        enc_output *= non_pad_mask.unsqueeze(dim = -1)                         # [batch_size, seq_len, d_model]

        return enc_output, enc_slf_attn