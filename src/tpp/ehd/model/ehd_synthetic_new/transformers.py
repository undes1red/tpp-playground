import torch
import torch.nn as nn
from einops import repeat

from src.ehd.model.ehd_perplexity.layers import TransformerDecoder
from src.ehd.model.ehd_perplexity.utils import *
from src.ehd.model.ehd_perplexity.position import PositionalEmbedding


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, device, d_input, d_rnn, d_hidden, n_layers_encoder, n_layers_decoder, n_head, d_qk, d_v, dropout):
        super(Transformer, self).__init__()
        self.device = device
        self.transformer_module = TransformerModule(
            d_input = d_input, d_hidden = d_hidden, n_layers_encoder = n_layers_encoder, \
            n_layers_decoder = n_layers_decoder, n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device
        )
        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_input, d_rnn, device = self.device)


    def forward(self, input_x):
        """
        Return intensity functions' values for all events and time and events, if possible, predictions.
        Args:
        1. event_time: the length of all time intervals between two adjacent events. shape: [batch_size, seq_len]
        2. event_type: vectors containing the information about each event. shape: [batch_size, seq_len]
        3. non_pad_mask: padding mask. 1 refers to the existence of an event, while 0 means a dummy event. shape: [batch_size, seq_len]
        """

        enc_output = self.transformer_module(input_x)                          # [batch_size, seq_len, d_input]
        enc_output = self.rnn(enc_output)                                      # [batch_size, seq_len, d_input]

        return enc_output


class TransformerModule(nn.Module):
    """ A encoder model with self attention mechanism. """
    def __init__(self, d_input, d_hidden, n_layers_encoder, n_layers_decoder, n_head, d_qk, d_v, dropout, device):
        super(TransformerModule, self).__init__()
        self.device = device
        self.d_input = d_input

        # position vector, used for temporal encoding
        # FIXME: set max_len during runtime, current max_len = 4096
        self.position_emb = PositionalEmbedding(d_input, max_len = 4096, device = self.device)

        self.decoder = TransformerDecoder(d_input, d_hidden, n_head, d_qk, d_v, \
                                          n_layers_decoder, device = self.device, dropout = dropout)


    def forward(self, input_x):
        """
        Encode event sequences via masked self-attention.
        Args:
        1. event_type: 
        2. event_time: input time intervals. shape: [batch_size, seq_len]
        3. non_pad_mask: pad mask tensor. shape: [batch_size, seq_len]
        """
        # prepare attention masks
        # self_attn_mask is where we cannot look, i.e., the future and the padding
        # Until now, we do not use any self attention masks.

        emb_input_x = self.position_emb(input_x)                               # [batch_size, seq_len, d_input]
        input_x = input_x + emb_input_x                                        # [batch_size, seq_len, d_input]
        output = self.decoder(input_x, non_pad_mask = None, self_attn_mask = None)
                                                                               # [batch_size, seq_len, d_input]

        return output


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn, device):
        super(RNN_layers, self).__init__()
        self.device = device

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True, device = self.device)
        self.projection = nn.Linear(d_rnn, d_model, device = self.device)

    def forward(self, data):
        out = self.rnn(data)[0]                                                # [batch_size, seq_len, d_rnn]

        out = self.projection(out)                                             # [batch_size, seq_len, d_model]
        return out