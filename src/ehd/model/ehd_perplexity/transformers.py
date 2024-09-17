import torch.nn as nn
from einops import repeat

from src.toolbox.position_embedding import BiasedPositionalEmbedding
from src.toolbox.transformer import TransformerLayer


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """
    def __init__(self, seq_len_x, seq_len_h, num_events, d_input, d_rnn, \
                 d_hidden, n_layers_encoder, n_layers_decoder, n_head, d_qk, d_v, dropout, device):
        super(Transformer, self).__init__()
        self.device = device
        self.num_events = num_events
        self.seq_len_h = seq_len_h
        self.seq_len_x = seq_len_x

        self.embedding = nn.Embedding(num_events + 1, d_input, padding_idx = num_events, device = self.device)
        self.position_embedding = BiasedPositionalEmbedding(d_input, max_len = 4096, device = self.device)

        self.encoder = nn.ModuleList([
                TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head, \
                                 d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device) for _ in range(n_layers_encoder)
            ])

        self.decoder = nn.ModuleList([
                TransformerDecoderLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head, \
                                 d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device) for _ in range(n_layers_decoder)
            ])

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
        events_history_embedding = self.embedding(events_history)              # [batch_size, seq_len_h, d_input]
        events_future_embedding = self.embedding(events_future)                # [batch_size, seq_len_x, d_input]

        time_history_embedding = self.position_embedding(self.seq_len_h, time_history)
                                                                               # [batch_size, seq_len_h, d_input]
        time_future_embedding = self.position_embedding(self.seq_len_x, time_future, position_start_index = self.seq_len_h)
                                                                               # [batch_size, seq_len_x, d_input]

        history_embedding = events_history_embedding + time_history_embedding  # [batch_size, seq_len_h, d_input]
        future_embedding = events_future_embedding + time_future_embedding     # [batch_size, seq_len_x, d_input]
        
        for enc_layer in self.encoder:
            future_embedding, _ = enc_layer(future_embedding)                  # [batch_size, seq_len_x, d_input]
        
        for dec_layer in self.decoder:
            history_embedding = dec_layer(history_embedding, future_embedding) # [batch_size, seq_len_h, d_input]

        outputs = self.rnn(history_embedding)                                  # [batch_size, seq_len_h, d_input]

        return outputs



class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_input, d_hidden, n_head, d_qk, d_v, dropout, device):
        super(TransformerDecoderLayer, self).__init__()
        self.device = device

        # Do self-attention on the representation of following events.
        self.self_attention = TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device)
        # Do cross-attention on the representation of following events and historical events.
        self.cross_attention = TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device)


    def forward(self, history_representation, future_representation):
        # Unlike the vanilla transformer, here we do not need any masks.
        # All attention modules can freely access all input events. 
        history, _ = self.self_attention(history_representation)               # [batch_size, seq_len_h, d_input]
        output, _ = self.cross_attention(q = history, k = future_representation, v = future_representation)
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