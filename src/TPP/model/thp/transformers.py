import math, torch
import torch.nn as nn

from .layers import TransformerLayer
from .utils import *


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_input, d_hidden,
            n_layers, n_head, d_qk, d_v, dropout, 
            device):
        super(Encoder, self).__init__()
        self.device = device
        self.d_input = d_input

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_input) for i in range(d_input)],
            device=self.device)

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_input, padding_idx=PAD, device = self.device)

        self.layer_stack = nn.ModuleList([
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,\
                             d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_input.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_type, event_time, non_pad_mask):
        """
        Encode event sequences via masked self-attention.
        Args:
        1. event_type: 
        2. event_time: input time intervals. shape: [batch_size, seq_len]
        3. non_pad_mask: pad mask tensor. shape: [batch_size, seq_len, 1]
        """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        self_attn_mask_subseq = get_subsequent_mask(event_time)
        self_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_time, seq_q=event_time)
        self_attn_mask_keypad = self_attn_mask_keypad.type_as(self_attn_mask_subseq)
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0) # [batch_size, seq_len, seq_len]

        time_emb = self.temporal_enc(event_time, non_pad_mask)                 # [batch_size, seq_len, d_input]

        if event_type != None:
            events_emb = self.event_emb(event_type)                            # [batch_size, seq_len, d_input]
        else:
            events_emb = torch.zeros_like(time_emb, device = self.device)      # [batch_size, seq_len, d_input]

        for enc_layer in self.layer_stack:
            time_emb += events_emb                                             # [batch_size, seq_len, d_input]
            time_emb, _ = enc_layer(
                time_emb,
                non_pad_mask = non_pad_mask,
                self_attn_mask = self_attn_mask)                               # [batch_size, seq_len, d_input]
        return time_emb


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types, device):
        super(Predictor, self).__init__()
        self.device = device

        self.linear = nn.Linear(dim, num_types, bias=False, device = self.device)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)                                                # [batch_size, seq_len, num_types]
        out = out * non_pad_mask
        return out


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


class TransformerTPP(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, num_types, device, d_input, d_rnn, d_hidden,
            n_layers, n_head, d_qk, d_v, dropout):
        super(TransformerTPP, self).__init__()
        self.device = device

        self.num_types = num_types if num_types > 0 else 1

        self.encoder = Encoder(
            num_types = self.num_types,
            d_input = d_input,
            d_hidden = d_hidden,
            n_layers = n_layers,
            n_head = n_head,
            d_qk = d_qk,
            d_v = d_v,
            dropout = dropout,
            device = self.device
        )

        # convert hidden vectors into valid intensity function values.
        self.linear = nn.Linear(d_input, num_types, device = self.device)

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_input, d_rnn, device = self.device)

        # prediction of next time stamp
        self.time_predictor = Predictor(d_input, 1, device = self.device)

        # prediction of next event type
        if num_types > 1:
            self.type_predictor = Predictor(d_input, num_types, device = self.device)
        else:
            self.type_predictor = None

    def forward(self, event_time, event_type):
        """
        Return intensity functions' values for all events and time and events, if possible, predictions.
        Args:
        1. event_time: the length of all time intervals between two adjacent events. shape: [batch_size, seq_len]
        2. event_type: 
        """

        non_pad_mask = get_non_pad_mask(event_time)                            # [batch_size, seq_len, 1]

        enc_output = self.encoder(event_type, event_time, non_pad_mask)        # [batch_size, seq_len, d_input]
        enc_output = self.rnn(enc_output)                                      # [batch_size, seq_len, d_input]
        intensity_output = self.linear(enc_output)                             # [batch_size, seq_len, num_types]

        time_prediction = self.time_predictor(enc_output, non_pad_mask)        # [batch_size, seq_len, 1]

        if self.type_predictor:
            type_prediction = self.type_predictor(enc_output, non_pad_mask)    # [batch_size, seq_len, num_types]
        else:
            type_prediction = None

        return intensity_output, (type_prediction, time_prediction)
