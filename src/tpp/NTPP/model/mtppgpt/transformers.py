import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

from src.toolbox.transformer import TransformerLayer
from src.toolbox.position_embedding import BiasedPositionalEmbedding
from src.toolbox.subsequent_mask import get_subsequent_mask


class TransEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self,
                 num_events, d_input, d_hidden,
                 n_layers, n_head, d_qk, d_v, dropout, device):
        super(TransEncoder, self).__init__()
        self.device = device
        self.d_input = d_input
        self.num_events = num_events

        # position vector, used for temporal encoding
        # FIXME: set max_len during runtime, current max_len = 4096
        self.position_emb = BiasedPositionalEmbedding(d_input, max_len = 4096, device = self.device)

        # event type embedding
        self.event_emb = nn.Embedding(num_events + 1, d_input, padding_idx = num_events, device = self.device)

        # history time encoder
        self.history_time_emb = nn.Linear(1, d_input, device = self.device)

        self.encoder = nn.ModuleList([
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,\
                             d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device)
            for _ in range(n_layers)])


    def forward(self, events_history, time_history, non_pad_mask):
        """
        Encode event sequences via masked self-attention.
        Args:
        1. events_history: historical events.        shape: [batch_size, seq_len]
        2. time_history: historical time intervals.  shape: [batch_size, seq_len]
        3. non_pad_mask: pad mask tensor.            shape: [batch_size, seq_len]
        """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        _, seq_len = time_history.shape
        
        self_attn_mask_subseq = get_subsequent_mask(events_history)            # [batch_size, seq_len, seq_len]
        self_attn_mask_keypad = torch.ones_like(non_pad_mask, device = self.device) - non_pad_mask
                                                                               # [batch_size, seq_len]
        self_attn_mask_keypad = repeat(self_attn_mask_keypad, 'b s -> b s1 s', s1 = seq_len)
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0) # [batch_size, seq_len, seq_len]

        # Mark Embeddings
        events_emb = self.event_emb(events_history)                            # [batch_size, seq_len, d_input]
        # Time Embeddings
        time_emb = self.position_emb(seq_len, time_history)             # [batch_size, seq_len, d_input]
        output = events_emb + time_emb                                         # [batch_size, seq_len, d_input]

        for enc_layer in self.encoder:
            '''
            history event sequence
            '''
            output, _ = enc_layer(
                output, output, output,
                non_pad_mask = non_pad_mask,
                self_attn_mask = self_attn_mask)                               # [batch_size, seq_len, d_input]

        return output