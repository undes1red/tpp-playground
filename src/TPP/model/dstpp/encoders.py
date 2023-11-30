import math
import torch
import torch.nn as nn

from einops import repeat
from src.TPP.model.dstpp.layers import EncoderLayer
from src.TPP.model.dstpp.utils import *


class Encoder_ST(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, device, loc_dim):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.loc_dim = loc_dim

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device = self.device)

        # event loc embedding
        self.event_emb_temporal = nn.Sequential(
            nn.Linear(1, d_model, device = self.device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device = self.device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device = self.device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device = self.device),
        )

        self.event_emb_loc = nn.Sequential(
            nn.Linear(self.loc_dim, d_model, device = self.device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device = self.device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device = self.device),
            nn.ReLU(),
            nn.Linear(d_model, d_model, device = self.device),
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout = dropout, device = self.device)
            for _ in range(n_layers)])

        self.layer_stack_loc = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout = dropout, device = self.device)
            for _ in range(n_layers)])

        self.layer_stack_temporal = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout = dropout, device = self.device)
            for _ in range(n_layers)])


    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        
        non_pad_mask = non_pad_mask.unsqueeze(dim = -1)                        # [batch_size, seq_len, 1]
        result = time.unsqueeze(-1) / self.position_vec                        # [batch_size, seq_len, d_model]
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask


    def forward(self, event_loc, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        _, seq_len = event_time.shape[:2]

        self_attn_mask_subseq = get_subsequent_mask(event_loc)                 # [batch_size, seq_len, seq_len]
        self_attn_mask_keypad = torch.ones_like(non_pad_mask, device = self.device) - non_pad_mask
                                                                               # [batch_size, seq_len]
        self_attn_mask_keypad = repeat(self_attn_mask_keypad, 'b s -> b s_1 s', s_1 = seq_len)
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask = (self_attn_mask_subseq + self_attn_mask_keypad).gt(0) # [batch_size, seq_len, seq_len]

        enc_output_temporal = self.temporal_enc(event_time, non_pad_mask)      # [batch_size, seq_len, d_model]
        enc_output_loc = self.event_emb_loc(event_loc)                         # [batch_size, seq_len, d_model]

        enc_output = enc_output_temporal + enc_output_loc
        
        for index in range(len(self.layer_stack)):
            enc_output_loc, _ = self.layer_stack_loc[index](
                enc_output_loc,
                non_pad_mask = non_pad_mask,
                slf_attn_mask = self_attn_mask)                                # [batch_size, seq_len, d_model]

            enc_output_temporal, _ = self.layer_stack_temporal[index](
                enc_output_temporal,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=self_attn_mask)                                  # [batch_size, seq_len, d_model]

            enc_output, _ = self.layer_stack[index](
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=self_attn_mask)                                  # [batch_size, seq_len, d_model]
        
        return enc_output, enc_output_temporal, enc_output_loc


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn, device):
        super().__init__()
        self.device = device

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True, device = self.device)
        self.projection = nn.Linear(d_rnn, d_model, device = self.device)


    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.long().sum(dim = -1)
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)             # [batch_size, *, d_model]
        temp = self.rnn(pack_enc_output)[0]                                    # [batch_size, *, d_rnn]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]      # [batch_size, seq_len, d_rnn]
        out = self.projection(out)                                             # [batch_size, seq_len, d_model]

        return out


class Transformer_ST(nn.Module):
    """ 
    A sequence to sequence model with attention mechanism. 
    The original codebase contains class Transformer but not uses it anywhere, so we remove
    it here.
    """

    def __init__(
            self, device, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, loc_dim=2):
        super().__init__()
        self.device = device

        self.encoder = Encoder_ST(
            d_model = d_model,
            d_inner = d_inner,
            n_layers = n_layers,
            n_head = n_head,
            d_k = d_k,
            d_v = d_v,
            dropout = dropout,
            device = device,
            loc_dim = loc_dim,
        )

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1, device = self.device))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0, device = self.device))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn, device = self.device)
        self.rnn_temporal = RNN_layers(d_model, d_rnn, device = self.device)
        self.rnn_spatial = RNN_layers(d_model, d_rnn, device = self.device)


    def forward(self, event_loc, event_time, non_pad_mask):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_loc: batch*seq_len*2;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim
        """
        
        enc_output, enc_output_temporal, enc_output_loc = self.encoder(event_loc, event_time, non_pad_mask)
                                                                               # 3 * [batch_size, seq_len, d_model]
        # Might be unneeded.
        # assert (enc_output != enc_output_temporal).any() & (enc_output != enc_output_loc).any() & (enc_output_loc != enc_output_temporal).any()
        
        enc_output = self.rnn(enc_output, non_pad_mask)                        # [batch_size, seq_len, d_model]
        enc_output_temporal = self.rnn_temporal(enc_output_temporal, non_pad_mask)
                                                                               # [batch_size, seq_len, d_model]
        enc_output_loc = self.rnn_spatial(enc_output_loc, non_pad_mask)        # [batch_size, seq_len, d_model]

        enc_output_all = torch.cat((enc_output_temporal, enc_output_loc, enc_output), dim = -1)
                                                                               # [batch_size, seq_len, 3 * d_model]

        return enc_output_all
