import torch
import torch.nn as nn
from einops import repeat

from src.ehd.model.ehd_synthetic.layers import TransformerEncoder, TransformerDecoder
from src.ehd.model.ehd_synthetic.utils import *
from src.ehd.model.ehd_synthetic.position import BiasedPositionalEmbedding


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, num_events, device, d_input, d_rnn, d_hidden, n_layers_encoder, n_layers_decoder, n_head, d_qk, d_v, dropout):
        super(Transformer, self).__init__()
        self.device = device
        self.num_events = num_events if num_events > 0 else 1
        self.transformer_module = TransformerModule(
            num_events = self.num_events, d_input = d_input, d_hidden = d_hidden,
            n_layers_encoder = n_layers_encoder, n_layers_decoder = n_layers_decoder,
            n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device
        )
        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_input, d_rnn, device = self.device)


    def forward(self, events_history, events_future, time_history, time_future, mask_history, mask_future):
        """
        Return intensity functions' values for all events and time and events, if possible, predictions.
        Args:
        1. event_time: the length of all time intervals between two adjacent events. shape: [batch_size, seq_len]
        2. event_type: vectors containing the information about each event. shape: [batch_size, seq_len]
        3. non_pad_mask: padding mask. 1 refers to the existence of an event, while 0 means a dummy event. shape: [batch_size, seq_len]
        """

        enc_output = self.transformer_module(events_history, events_future, time_history, time_future, mask_history, mask_future)
                                                                               # [batch_size, seq_len_h, d_input]
        enc_output = self.rnn(enc_output)                                      # [batch_size, seq_len_h, d_input]

        return enc_output


class TransformerModule(nn.Module):
    """ A encoder model with self attention mechanism. """
    def __init__(self, num_events, d_input, d_hidden, n_layers_encoder, n_layers_decoder, n_head, d_qk, d_v, dropout, device):
        super(TransformerModule, self).__init__()
        self.device = device
        self.d_input = d_input
        self.num_events = num_events

        # position vector, used for temporal encoding
        # FIXME: set max_len during runtime, current max_len = 4096
        self.position_emb = BiasedPositionalEmbedding(d_input, max_len = 4096, device = self.device)

        # event type embedding
        self.event_emb = nn.Embedding(num_events + 1, d_input, padding_idx = num_events, device = self.device)

        self.encoder = TransformerEncoder(d_input, d_hidden, n_head, d_qk, d_v, \
                                          n_layers_encoder, device = self.device, dropout = dropout)

        self.decoder = TransformerDecoder(d_input, d_hidden, n_head, d_qk, d_v, \
                                          n_layers_decoder, device = self.device, dropout = dropout)


    def forward(self, events_history, events_future, time_history, time_future, mask_history, mask_future):
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
        '''
        self_attn_mask_keypad = torch.ones_like(mask_future, device = self.device) - mask_future
                                                                               # [batch_size, seq_len_x]
        seq_len_x = self_attn_mask_keypad.shape[-1]
        self_attn_mask_keypad = repeat(self_attn_mask_keypad, 'b s -> b s_1 s', s_1 = seq_len_x)
                                                                               # [batch_size, seq_len_x, seq_len_x]
        self_attn_mask = self_attn_mask_keypad.gt(0)                           # [batch_size, seq_len_x, seq_len_x]
        '''

        # Time Embedding
        time_emb_history = self.position_emb(events_history, time_history)     # [batch_size, seq_len_h, d_input]
        time_emb_future = self.position_emb(events_future, time_future)        # [batch_size, seq_len_x, d_input]

        # Event Embedding
        events_emb_history = self.event_emb(events_history)                    # [batch_size, seq_len_h, d_input]
        mingled_emb_history = time_emb_history + events_emb_history            # [batch_size, seq_len_h, d_input]
        events_emb_future = self.event_emb(events_future)                      # [batch_size, seq_len_x, d_input]
        mingled_emb_future = time_emb_future + events_emb_future               # [batch_size, seq_len_x, d_input]

        reference = self.encoder(mingled_emb_future, non_pad_mask = mask_future, self_attn_mask = None)
                                                                               # [batch_size, seq_len_x, d_input]

        output = self.decoder(mingled_emb_history, reference, non_pad_mask = mask_history, self_attn_mask = None)
                                                                               # [batch_size, seq_len_h, d_input]

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