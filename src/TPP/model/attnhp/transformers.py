import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.toolbox.transformer import TransformerLayer
from src.toolbox.subsequent_mask import get_subsequent_mask
from src.toolbox.position_embedding import BiasedPositionalEmbedding


class TransformerEncoder(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, num_types, device, d_input, d_rnn, d_hidden,
                 n_layers, n_head, d_qk, d_v, dropout, integration_sample_rate):
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.num_types = num_types if num_types > 0 else 1
        self.integration_sample_rate = integration_sample_rate

        self.encoder = Encoder(
            num_types = self.num_types,
            d_input = d_input,
            d_hidden = d_hidden,
            n_layers = n_layers,
            n_head = n_head,
            d_qk = d_qk,
            d_v = d_v,
            dropout = dropout,
            integration_sample_rate = integration_sample_rate,
            device = self.device
        )


    def forward(self, time_history, sample_time, events_history, non_pad_mask, custom_events_history = False):
        """
        Return intensity functions' values for all events and time and events, if possible, predictions.
        Args:
        1. event_time: the length of all time intervals between two adjacent events. shape: [batch_size, seq_len]
        2. event_type: vectors containing the information about each event. shape: [batch_size, seq_len]
        3. non_pad_mask: padding mask. 1 refers to the existence of an event, while 0 means a dummy event. shape: [batch_size, seq_len]
        """

        enc_output = self.encoder(time_history, sample_time, events_history, non_pad_mask, custom_events_history)
                                                                               # [batch_size, seq_len, d_input]

        return enc_output


    def get_event_embedding(self, input_event):
        return self.encoder.get_event_embedding(input_event)                   # [batch_size, seq_len, d_input]


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """
    def __init__(self, num_types, d_input, d_hidden, integration_sample_rate,
                 n_layers, n_head, d_qk, d_v, dropout, device):
        super(Encoder, self).__init__()
        self.device = device
        self.d_input = d_input
        self.num_types = num_types
        self.integration_sample_rate = integration_sample_rate

        # position vector, used for temporal encoding
        # FIXME: set max_len during runtime, current max_len = 4096
        self.position_emb = BiasedPositionalEmbedding(d_input, max_len = 4096, device = self.device)

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_input, padding_idx = num_types, device = self.device)

        self.layer_stack = nn.ModuleList([
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,\
                             d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device)
            for _ in range(n_layers)])


    def forward(self, time_history, sample_time, events_history, non_pad_mask, custom_events_history):
        """
        Encode event sequences via masked self-attention.
        Args:
        1. time_history: input time intervals. shape: [batch_size, seq_len]
        2. sample_time: shape: [..., batch_size, seq_len, sample_rate]
        3. events_history: shape: [batch_size, seq_len]
        4. non_pad_mask: pad mask tensor. shape: [batch_size, seq_len]
        """
        batch_size, seq_len = events_history.shape
        
        sample_time = rearrange(sample_time, '... b s sr -> ... sr () b s')    # [..., sample_rate, num_event, batch_size, seq_len]

        sample_event = torch.arange(self.num_types, device = self.device)      # [num_event]
        sample_event = repeat(sample_event, 'ne -> sr ne b s', sr = self.integration_sample_rate, b = batch_size, s = seq_len)
                                                                               # [sample_rate, num_event, batch_size, seq_len]
        
        time_history = repeat(time_history, 'b s -> sr () b s', sr = self.integration_sample_rate)
                                                                               # [sample_rate, num_event, batch_size, seq_len]
        if len(sample_time.shape) > 4:
            einop = '... -> '
            parameter_dict = {}
            for idx, val in enumerate(sample_time.shape[:-4]):
                einop += f'a{idx} '
                parameter_dict[f'a{idx}'] = val
            
            einop += '...'
            time_history = repeat(time_history, einop, **parameter_dict)       # [..., sample_rate, num_event, batch_size, seq_len]
            
        events_history = repeat(events_history, 'b s -> sr ne b s', sr = self.integration_sample_rate, ne = self.num_types)
                                                                               # [sample_rate, num_event, batch_size, seq_len]

        # Connect history with samples, further we perform masked attention on this sequence.
        connected_event_seq = torch.cat((events_history, sample_event), dim = -1)
                                                                               # [sample_rate, num_event, batch_size, 2 * seq_len]
        '''
        Prepare attention masks
        AttNHP's attention mask should be carefully handled. It should ensure:
        1. each sample_event only sees itself and history events it should consider.
        2. each event in history only sees eariler events. It should not know the existence of sample_event.
        3. padding events and EOS are invisible to history_events and sample_events.
        '''
        self_attn_mask_from_history_to_history = get_subsequent_mask(seq_len, device = self.device)
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask_from_history_to_sample = torch.zeros_like(self_attn_mask_from_history_to_history)
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask_from_sample_to_history = self_attn_mask_from_history_to_history
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask_from_sample_to_sample = torch.eye(seq_len, dtype = torch.uint8, device = self.device)
        self_attn_mask_from_sample_to_sample = rearrange(self_attn_mask_from_sample_to_sample, 's s1 -> () s s1')
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask_from_history_all = torch.cat((self_attn_mask_from_history_to_history, self_attn_mask_from_history_to_sample), dim = -1)
                                                                               # [batch_size, seq_len, seq_len * 2]
        self_attn_mask_from_sample_all = torch.cat((self_attn_mask_from_sample_to_history, self_attn_mask_from_sample_to_sample), dim = -1)
                                                                               # [batch_size, seq_len, seq_len * 2]
        self_attn_mask = torch.cat((self_attn_mask_from_history_all, self_attn_mask_from_sample_all), dim = -2)
                                                                               # [batch_size, seq_len * 2, seq_len * 2]
        
        non_pad_mask = torch.cat((non_pad_mask, torch.ones_like(non_pad_mask)), dim = -1)
                                                                               # [batch_size, seq_len * 2]
        non_pad_mask_with_sample = rearrange(non_pad_mask, 'b s -> b () s')    # [batch_size, seq_len * 2, seq_len * 2]

        self_attn_mask = self_attn_mask & non_pad_mask_with_sample             # [batch_size, seq_len * 2, seq_len * 2]

        # Time Embedding
        time_history_emb = self.position_emb(seq_len, time_history)            # [..., sample_rate, num_events, batch_size, seq_len, d_input]
        sample_time_emb = self.position_emb(seq_len, sample_time, position_start_index = 1)
                                                                               # [..., sample_rate, num_events, batch_size, seq_len, d_input]
        time_emb = torch.cat((time_history_emb, sample_time_emb), dim = -2)    # [..., sample_rate, num_events, batch_size, seq_len * 2, d_input]

        # Event Embedding
        if events_history != None:
            if custom_events_history:
                events_emb = events_history
            else:
                events_emb = self.event_emb(connected_event_seq)               # [sample_rate, num_event, batch_size, seq_len * 2, d_input]
                einop = f'... -> {"() " * (len(time_emb.shape) - 5)}...'
                events_emb = rearrange(events_emb, einop)                      # [..., sample_rate, num_event, batch_size, seq_len * 2, d_input]
        else:
            events_emb = torch.zeros_like(time_emb, device = self.device)      # [..., sample_rate, num_event, batch_size, seq_len * 2, d_input]

        mingled_emb = time_emb + events_emb                                    # [..., sample_rate, num_event, batch_size, seq_len * 2, d_input]

        for enc_layer in self.layer_stack:
            mingled_emb, _ = enc_layer(mingled_emb, non_pad_mask = non_pad_mask, self_attn_mask = self_attn_mask)
                                                                               # [..., sample_rate, batch_size, seq_len * 2, num_event, d_input]

        return mingled_emb
    

    def get_event_embedding(self, input_event):
        return self.event_emb(input_event)                                     # [batch_size, seq_len, d_input]