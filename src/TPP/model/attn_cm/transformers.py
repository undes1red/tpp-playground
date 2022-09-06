import math, torch
import torch.nn as nn

from .layers import TransformerLayer
from .utils import *
from .nonneg import NonNegLinear


class TransEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_input, d_hidden,
            n_layers, n_head, d_qk, d_v, dropout,
            event_toggle, wq_nonneg, wk_nonneg, wv_nonneg,
            device):
        super(TransEncoder, self).__init__()
        self.device = device
        self.d_input = d_input
        self.event_toggle = event_toggle
        self.num_types = num_types

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_input) for i in range(d_input)],
            device=self.device)

        # history time encoder
        self.history_time_emb = nn.Linear(1, d_input, device = self.device)

        self.event_encoder = nn.ModuleList([
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,\
                             d_qk = d_qk, d_v = d_v, dropout = dropout, wq_nonneg = wq_nonneg, \
                             wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg, device = self.device)
            for _ in range(n_layers)])

        self.time_encoder = nn.ModuleList([
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,\
                             d_qk = d_qk, d_v = d_v, dropout = dropout, wq_nonneg = wq_nonneg, \
                             wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg, device = self.device)
            for _ in range(n_layers)])

    def encode_position_idx(self, idx):
        """
        Input:  [seq_len]
        Output: [batch_size, seq_len, d_input]
        """

        result = idx.unsqueeze(-1) / self.position_vec
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:, 1::2])
        return result

    def forward(self, events_emb, event_time, non_pad_mask):
        """
        Encode event sequences via masked self-attention.
        Args:
        1. event_type: 
        2. event_time: input time intervals. shape: [batch_size, seq_len, 1]
        3. non_pad_mask: pad mask tensor. shape: [batch_size, seq_len, 1]
        """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        seq_idx = torch.arange(non_pad_mask.shape[1], device = self.device)
        
        self_attn_mask_subseq = get_subsequent_mask(event_time)
        self_attn_mask_keypad = torch.ones_like(non_pad_mask, device = self.device) - non_pad_mask
                                                                               # [batch_size, seq_len, 1]
        self_attn_mask_keypad = self_attn_mask_keypad.repeat(1, 1, self_attn_mask_keypad.shape[-2])
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0) # [batch_size, seq_len, seq_len]

        idx_emb = self.encode_position_idx(seq_idx).unsqueeze(dim = 0)         # [seq_len, d_input]
        time = event_time                                                      # [batch_size, seq_len, 1]

        if self.event_toggle:
            events_emb = events_emb + idx_emb                                  # [batch_size, seq_len, d_input]
            for enc_layer in self.event_encoder:
                '''
                history event sequence
                '''
                events_emb, _ = enc_layer(
                    events_emb, events_emb, events_emb,
                    non_pad_mask = non_pad_mask,
                    self_attn_mask = self_attn_mask)                           # [batch_size, seq_len, d_input]
            
            time_emb = self.history_time_emb(time) + idx_emb                   # [batch_size, seq_len, d_input]
            time_emb, _ = self.time_encoder[0](
                    events_emb, time_emb, time_emb,
                    non_pad_mask = non_pad_mask,
                    self_attn_mask = self_attn_mask)                           # [batch_size, seq_len, d_input]
            for enc_layer in self.time_encoder[1:]:
                '''
                history event sequence
                '''
                time_emb, _ = enc_layer(
                    time_emb, time_emb, time_emb,
                    non_pad_mask = non_pad_mask,
                    self_attn_mask = self_attn_mask)                           # [batch_size, seq_len, d_input]
            
            time_emb += events_emb                                             # [batch_size, seq_len, d_input]
        else:
            time_emb = self.history_time_emb(time) + idx_emb                   # [batch_size, seq_len, d_input]
            for enc_layer in self.time_encoder:
                '''
                history event sequence
                '''
                time_emb, _ = enc_layer(
                    time_emb, time_emb, time_emb,
                    non_pad_mask = non_pad_mask,
                    self_attn_mask = self_attn_mask)                           # [batch_size, seq_len, d_input]

        return time_emb


class TransDecoder(nn.Module):
    '''
    Decode the intensity integral based on the history and time.
    '''
    def __init__(self, num_types, d_input, d_hidden, d_qk, d_v, integral_module_layers, n_head, dropout, event_toggle, device):
        super(TransDecoder, self).__init__()
        self.device = device
        self.num_types = num_types
        self.d_input = d_input
        self.event_toggle = event_toggle

        self.time_encoder = NonNegLinear(1, d_input, bias = False, device = device)

        self.decoder = nn.ModuleList(
            [
                TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head, d_qk = d_qk,
                                 d_v = d_v, dropout = dropout, wq_nonneg = True, wk_nonneg = True, wv_nonneg = True,
                                 device = device)
                for _ in range(integral_module_layers + 1)
            ]
        )

        self.integral = NonNegLinear(d_input, 1, device = device)
        self.nonneg_activate = nn.Softplus()

    
    def forward(self, history, time_next, mask_next):
        '''
        1. history:         [batch_size, seq_len, d_input] if self.event_toggle = False else [batch_size, seq_len, num_events, d_history]
           The history information encoded by Transformers.
        2. time_next:       [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]
           For relative time calculation.
        3. mask_next:       [batch_size, seq_len]
           mask matrix to filter out padding events from the original sequences. 0 means should be masked out.
        '''
        # batch_size, seq_len = history.shape[:2]

        self_attn_mask_subseq = get_subsequent_mask(time_next)
        self_attn_mask_keypad = (torch.ones_like(mask_next, device = self.device) - mask_next).unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, 1]
        self_attn_mask_keypad = self_attn_mask_keypad.repeat(1, 1, self_attn_mask_keypad.shape[-2])
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0) # [batch_size, seq_len, seq_len]
        
        if self.event_toggle:
            time_next = time_next.transpose(-1, -2).contiguous()               # [batch_size, num_events, seq_len]
            emb_next = time_next.unsqueeze(dim = -1)                           # [batch_size, num_events, seq_len, 1]
        else:
            emb_next = time_next                                               # [batch_size, seq_len, 1]

        emb_next = self.time_encoder(emb_next)                                 # [batch_size, num_events, seq_len, d_input] if we need events else [batch_size, seq_len, d_input]

        # The first attention layer
        # Q: history, K: emb_next, V: emb_next
        first_layer = self.decoder[0]
        if self.event_toggle:
            history = history.transpose(-2, -3).contiguous()                   # [batch_size, num_events, seq_len, d_history]
            self_attn_mask = self_attn_mask.unsqueeze(dim = 1)
            mask_next = mask_next.unsqueeze(dim = 1)

        output, _ = first_layer(history, emb_next, emb_next, self_attn_mask = self_attn_mask, non_pad_mask = mask_next.unsqueeze(dim = -1))
                                                                               # [batch_size, num_events, seq_len, d_input] if we need events else [batch_size, seq_len, d_input]
        for layer in self.decoder[1:]:
            output, _ = layer(output, output, output, self_attn_mask = self_attn_mask, non_pad_mask = mask_next.unsqueeze(dim = -1))
                                                                               # [batch_size, num_events, seq_len, d_input] if we need events else [batch_size, seq_len, d_input]

        # Integral
        integral = self.nonneg_activate(self.integral(output)).squeeze(dim = -1)
                                                                               # [batch_size, num_events, seq_len] if we need events else [batch_size, seq_len]
        if self.event_toggle:
            integral = integral.transpose(-1, -2)                              # [batch_size, seq_len, num_events]

        return integral                                                        # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]


    def probe(self, history, history_time, result, mask, resolution):
        '''
        1. history:  [batch_size, seq_len, history_length, d_input]
           The history information encoded by Transformers.
        2. history_time: [batch_size, seq_len, history_length]
           For relative time calculation.
        3. result:   [batch_size, seq_len]
           Current timestamp for calculating the integral
        4. mask:     [batch_size, seq_len, history_length]
           mask matrix to filter out padding events from the original sequences. 0 means should be masked out.
        5. resolution: int
           As discribed in PyTorch documentation, we would insert (resolution - 2) points in each time interval.
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        batch_size, seq_len, _, _ = history.shape
        result = result.unsqueeze(dim = -1) * time_multiplier                  # [batch_size, seq_len, resolution]
        result.requires_grad = True

        # We preprocess the mask as the sequence here is supposed to be 1 item longer than the history.
        mask = torch.cat(
            (mask, torch.ones(batch_size, seq_len, 1, device = self.device)),
            dim = -1
        )                                                                      # [batch_size, seq_len, history_length + 1]
        self_attn_mask_keypad = (torch.ones_like(mask, device = self.device) - mask).unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, history_length + 1, 1]
        self_attn_mask_keypad = self_attn_mask_keypad.repeat(1, 1, 1, self_attn_mask_keypad.shape[-2])
                                                                               # [batch_size, seq_len, history_length + 1, history_length + 1]
        self_attn_mask = self_attn_mask_keypad.gt(0)                           # [batch_size, seq_len, history_length + 1, history_length + 1]
        self_attn_mask = self_attn_mask.unsqueeze(dim = 2).repeat(1, 1, resolution, 1, 1)
                                                                               # [batch_size, seq_len, resolution, history_length + 1, history_length + 1]

        # Preparing the attention input
        cls_ori = self.cls.repeat(batch_size, seq_len, 1, 1)                   # [batch_size, seq_len, 1, d_input]
        new_history = torch.cat((history, cls_ori), dim = -2)                  # [batch_size, seq_len, history_length + 1, d_input]
        expand_new_history = new_history.unsqueeze(dim = 2).repeat(1, 1, resolution, 1, 1)
                                                                               # [batch_size, seq_len, resolution, history_length + 1, d_input]
        relative_time = torch.cat(
            (history_time.unsqueeze(dim = 2).repeat(1, 1, resolution, 1), result.unsqueeze(dim = -1)),
            dim = -1
        )                                                                      # [batch_size, seq_len, resolution, history_length + 1]
        interval_from_current_to_history = (torch.sum(relative_time, dim = -1, keepdim = True) - torch.cumsum(relative_time, dim = -1)).float()
                                                                               # [batch_size, seq_len, resolution, history_length + 1]
        # Tackle the floatpoint number precision tolerance which may lead to tiny computation errors in interval_from_current_to_history
        interval_from_current_to_history[:, :, :, -1] = result
        interval_from_current_to_history[:, :, :, :-1] = 0

        emb_time_1 = self.time_scaler_function_1(self.nonneg_activate(self.time_scaler_parameter_1) * interval_from_current_to_history)
                                                                               # [batch_size, seq_len, resolution, history_length + 1]

        # Time activation function may be required here before feeded into the time encoder.
        emb_time = self.time_encoder(emb_time_1.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, resolution, history_length + 1, d_input]

        emb_for_integral = self.nonneg_activate(expand_new_history) + emb_time # [batch_size, seq_len, resolution, history_length + 1, d_input]

        for monotonic_layer in self.decoder:
            emb_for_integral, _ = monotonic_layer(
                emb_for_integral, emb_for_integral, emb_for_integral,
                non_pad_mask = mask.unsqueeze(dim = -1).unsqueeze(dim = 2),
                self_attn_mask = self_attn_mask)                               # [batch_size, seq_len, resolution, history_length + 1, d_input]
        
        cls_embedding = emb_for_integral[:, :, :, -1, :]                       # [batch_size, seq_len, resolution, d_input]

        # Integral
        integral = self.nonneg_activate(self.integral(cls_embedding)).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution]

        # Intensity values and their sum.
        intensity = torch.autograd.grad(
            outputs = integral,
            inputs = result,
            grad_outputs = torch.ones_like(integral),
            create_graph = True,
        )[0]                                                                   # [batch_size, seq_len, resolution]

        integral = integral.reshape(batch_size, -1)                            # [batch_size, seq_len * resolution]
        intensity = intensity.reshape(batch_size, -1)                          # [batch_size, seq_len * resolution]

        return integral, intensity