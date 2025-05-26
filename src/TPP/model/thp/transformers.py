import torch
import torch.nn as nn
from einops import rearrange

from src.toolbox.transformer import TransformerLayer
from src.toolbox.position_embedding import BiasedPositionalEmbedding
from src.toolbox.subsequent_mask import get_subsequent_mask


class Encoder(nn.Module):
    def __init__(self,
                 num_types, d_input, d_hidden,
                 n_layers, n_head, d_qk, d_v, dropout, 
                 device):
        '''
        This function builds a Transformer encoder.
        
        ### Args
          * ```int``` num_types
            The number of all possible marks.
          * ```torch.device``` device
            The device where we place this transformer encoder.
          * ```int``` d_input
            The dimension of the Transformer input tensor.
          * ```int``` d_hidden
              The dimension of the FFN module in the Transformer.  
          * ```int``` n_layers
              The number of self attention + FFN layers in the Transformer.  
          * ```int``` n_head
            The number of head in self attention.
          * ```int``` d_qk
            The dimension of matrices Q and K.
          * ```int``` d_v
            The dimension of metrix V.
          * ```float``` dropout
            Dropout rate for the history encoder.
        '''
        super(Encoder, self).__init__()
        self.device = device
        self.d_input = d_input
        self.num_types = num_types

        # position vector, used for temporal encoding
        # FIXME: set max_len during runtime, current max_len = 4096
        self.position_emb = BiasedPositionalEmbedding(d_input, max_len = 4096, device = self.device)

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_input, padding_idx = num_types, device = self.device)

        self.layer_stack = nn.ModuleList([
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,\
                             d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device)
            for _ in range(n_layers)])


    def forward(self, event_type, event_time, non_pad_mask):
        '''
        Encode the input continuous-time event stream using Transformer.
        
        ### Args
          * ```torch.tensor``` event_time
            shape: ```[batch_size, seq_len]```
            The length of all time intervals between two adjacent events.
          * ```torch.tensor``` event_type
            shape: ```[batch_size, seq_len]```
            Vectors containing the information about each event. 
          * ```torch.tensor``` non_pad_mask
            shape: ```[batch_size, seq_len]```
            Padding mask. 1 refers to the existence of an event, while 0 means a dummy event. 
        ### Outputs
            * ```torch.tensor``` output
              shape: ```[batch_size, seq_len, d_input]```
              The representation of the original input.
        '''
        # prepare attention masks
        # self_attn_mask is where we cannot look, i.e., the future and the padding
        seq_len = event_type.shape[-1]
        self_attn_mask_subseq = get_subsequent_mask(seq_len, device = self.device)
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask_keypad = rearrange(non_pad_mask, 'b s -> b () s')       # [batch_size, seq_len, seq_len]
        self_attn_mask = self_attn_mask_keypad & self_attn_mask_subseq         # [batch_size, seq_len, seq_len]

        # Time Embedding
        time_emb = self.position_emb(seq_len, event_time)                      # [batch_size, seq_len, d_input]

        if event_type != None:
            events_emb = self.event_emb(event_type)                            # [batch_size, seq_len, d_input]
        else:
            events_emb = torch.zeros_like(time_emb, device = self.device)      # [batch_size, seq_len, d_input]

        output = time_emb + events_emb                                         # [batch_size, seq_len, d_input]
        for enc_layer in self.layer_stack:
            output, _ = enc_layer(
                output,
                non_pad_mask = non_pad_mask,
                self_attn_mask = self_attn_mask)                               # [batch_size, seq_len, d_input]
        return output


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """
    def __init__(self, d_model, d_rnn, device):
        '''
        This function builds a RNN module.
        
        ### Args:
          * ```int``` d_model
            The dimension of the RNN input.
          * ```int``` d_rnn
            The dimension of RNN's hidden state.
          * ```torch.device``` device
            The device where we place this RNN module.
        '''
        super(RNN_layers, self).__init__()
        self.device = device

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True, device = self.device)
        self.projection = nn.Linear(d_rnn, d_model, device = self.device)


    def forward(self, data):
        '''
        Use the RNN module to transform the input data.
        
        ### Args:
          * ```torch.tensor``` data
            shape: [batch_size, seq_len, d_model]
            The RNN module input.
        
        ### Outputs
          * ```torch.tensor``` out
            shape: [batch_size, seq_len, d_model]
            The RNN module output. 
        '''
        out = self.rnn(data)[0]                                                # [batch_size, seq_len, d_rnn]

        out = self.projection(out)                                             # [batch_size, seq_len, d_model]
        return out


class TransformerTPP(nn.Module):
    def __init__(
            self, num_types, device, d_input, d_rnn, d_hidden,
            n_layers, n_head, d_qk, d_v, dropout):
        '''
        This function builds a Transformer encoder.
        
        ### Args
          * ```int``` num_types
            The number of all possible marks.
          * ```torch.device``` device
            The device where we place this transformer encoder.
          * ```int``` d_input
            The dimension of the Transformer input tensor.
          * ```int``` d_rnn
            The dimension of RNN's hidden state.
          * ```int``` d_hidden
              The dimension of the FFN module in the Transformer.  
          * ```int``` n_layers
              The number of self attention + FFN layers in the Transformer.  
          * ```int``` n_head
            The number of head in self attention.
          * ```int``` d_qk
            The dimension of matrices Q and K.
          * ```int``` d_v
            The dimension of metrix V.
          * ```float``` dropout
            Dropout rate for the history encoder.
        '''
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

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_input, d_rnn, device = self.device)


    def forward(self, event_time, event_type, non_pad_mask):
        '''
        Encode the input continuous-time event stream using Transformer.
        
        ### Args
          * ```torch.tensor``` event_time
            shape: ```[batch_size, seq_len]```
            The length of all time intervals between two adjacent events.
          * ```torch.tensor``` event_type
            shape: ```[batch_size, seq_len]```
            Vectors containing the information about each event. 
          * ```torch.tensor``` non_pad_mask
            shape: ```[batch_size, seq_len]```
            Padding mask. 1 refers to the existence of an event, while 0 means a dummy event. 
        ### Outputs
            * ```torch.tensor``` enc_output
              shape: ```[batch_size, seq_len, d_input]```
              The representation of the original input.
        '''
        enc_output = self.encoder(event_type, event_time, non_pad_mask)        # [batch_size, seq_len, d_input]
        enc_output = self.rnn(enc_output)                                      # [batch_size, seq_len, d_input]

        return enc_output
