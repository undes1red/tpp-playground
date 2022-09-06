import math, torch
import torch.nn as nn

from .layers import TransformerLayer
from .utils import *
from .nonneg import NonNegLinear
from .activate import sym_Log


class TransEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_input, d_hidden,
            n_layers, n_head, d_qk, d_v, dropout,
            event_toggle, wq_nonneg, wk_nonneg, wv_nonneg,
            device, seq_mask = False):
        super(TransEncoder, self).__init__()
        self.device = device
        self.d_input = d_input
        self.event_toggle = event_toggle
        self.num_types = num_types
        self.seq_mask = seq_mask

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_input) for i in range(d_input)],
            device=self.device)

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 2, d_input, padding_idx = num_types, device = self.device)

        # history time encoder
        # Devided into two parts: relative time & absolute time
        # The first event should always be excluded. As the absolute time is the cumsum of relative time, so
        # the first (not-counted) dummy event would always start at time 0.
        # Which one should we introduce here, sinusoidal embedding or plain linear Transformation?

        self.history_encoder = nn.ModuleList([
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,
                             d_qk = d_qk, d_v = d_v, dropout = dropout, 
                             wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg,
                             device = self.device)
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

    def forward(self, history_event, history_time, non_pad_mask):
        """
        Encode event sequences via masked self-attention.
        1. subseq_mask can be None, implying free attention across all available items, this feature is controlled by
           option seq_mask

        Args:
        1. history_event:[batch_size, seq_len, history_length]
        history event sequences for history encoder (model can decide if it should use it by event_toggle)
        2. history_time: [batch_size, seq_len, history_length, 1]
        history time sequences for history encoder
        3. non_pad_mask: pad mask tensor. shape: [batch_size, seq_len, history_length, 1]
        """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding        
        if self.seq_mask:
            self_attn_mask_subseq = get_subsequent_mask(history_time)
        self_attn_mask_keypad = torch.ones_like(non_pad_mask, device = self.device) - non_pad_mask
                                                                               # [batch_size, seq_len, history_length, 1]
        self_attn_mask_keypad = self_attn_mask_keypad.repeat(1, 1, 1, self_attn_mask_keypad.shape[-2])
                                                                               # [batch_size, seq_len, history_length, history_length]
        if self.seq_mask:
            self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0)
                                                                               # [batch_size, seq_len, history_length, history_length]
        else:
            self_attn_mask = self_attn_mask_keypad.gt(0)                       # [batch_size, seq_len, history_length, history_length]

        # relative and absolute timestamps for this event sequence.
        time = history_time                                                    # [batch_size, seq_len, history_length, 1]
        cum_time = history_time.cumsum(dim = -2)                               # [batch_size, seq_len, history_length, 1]

        time_emb = self.encode_position_idx(time.squeeze(dim = -1))            # [batch_size, seq_len, history_length, d_input]
        cum_time_emb = self.encode_position_idx(cum_time.squeeze(dim = -1))    # [batch_size, seq_len, history_length, d_input]

        if self.event_toggle:
            events_emb = self.event_emb(history_event)                         # [batch_size, seq_len, history_length, d_input]
            time_emb = time_emb + cum_time_emb                                 # [batch_size, seq_len, history_length, d_input]
            history_emb = events_emb + time_emb                                # [batch_size, seq_len, history_length, d_input]
            for enc_layer in self.history_encoder:
                '''
                Encoding history time+event sequences using self-attention
                '''
                history_emb, _ = enc_layer(
                    history_emb, history_emb, history_emb,
                    non_pad_mask = non_pad_mask,
                    self_attn_mask = self_attn_mask)                           # [batch_size, seq_len, history_length, d_input]
        else:
            history_emb = time_emb + cum_time_emb                              # [batch_size, seq_len, history_length, d_input]
            for enc_layer in self.history_encoder:
                '''
                history event sequence
                '''
                history_emb, _ = enc_layer(
                    history_emb, history_emb, history_emb,
                    non_pad_mask = non_pad_mask,
                    self_attn_mask = self_attn_mask)                           # [batch_size, seq_len, history_length, d_input]

        return history_emb                                                     # [batch_size, seq_len, history_length, d_input]


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

        # Trainable Embedding of the dummy token, a little bit similar to the [CLS] token in large language models
        # However, this embedding should be placed at the end of the input sequence.
        self.cls = nn.Parameter(torch.randn(self.d_input, dtype = torch.float32, requires_grad = True, device = device))

        # Should we introduce more complex time_scalar based on our prior knowledge about TPP?
        self.time_scaler_function_1 = nn.Tanh()
        self.time_scaler_parameter_1 = nn.Parameter(torch.tensor(0., device = device, requires_grad=True))

        self.time_encoder = NonNegLinear(1, d_input, bias = False, device = device)

        self.decoder = nn.ModuleList(
            [
                TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head, d_qk = d_qk,
                                 d_v = d_v, dropout = dropout, wq_nonneg = True, wk_nonneg = True, wv_nonneg = True,
                                 device = device)
                for _ in range(integral_module_layers)
            ]
        )

        self.integral = NonNegLinear(d_input, 1, device = device)
        self.nonneg_activate = nn.Softplus()

        # Event prediction
        # Available when event_toggle is True.
        if self.event_toggle:
            self.event_predictor = nn.Linear(d_input, num_types, device = device)

    
    def forward(self, history, history_time, result, mask):
        '''
        1. history:  [batch_size, seq_len, history_length, d_input]
           The history information encoded by Transformers.
        2. history_time: [batch_size, seq_len, history_length]
           For relative time calculation.
        3. result:   [batch_size, seq_len]
           Current timestamp for calculating the integral
        4. mask:     [batch_size, seq_len, history_length]
           mask matrix to filter out padding events from the original sequences. 0 means should be masked out.
        '''
        batch_size, seq_len, _, _ = history.shape

        # We preprocess the mask as the sequence here is supposed to be 1 item longer than the history.
        mask = torch.cat(
            (mask, (mask.sum(dim = -1) > 0).int().unsqueeze(dim = -1)),
            dim = -1
        )                                                                      # [batch_size, seq_len, history_length + 1]
        self_attn_mask_keypad = (torch.ones_like(mask, device = self.device) - mask).unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, history_length + 1, 1]
        self_attn_mask_keypad = self_attn_mask_keypad.repeat(1, 1, 1, self_attn_mask_keypad.shape[-2])
                                                                               # [batch_size, seq_len, history_length + 1, history_length + 1]
        self_attn_mask = self_attn_mask_keypad.gt(0)                           # [batch_size, seq_len, history_length + 1, history_length + 1]

        cls_ori = self.cls.repeat(batch_size, seq_len, 1, 1)                   # [batch_size, seq_len, 1, d_input]
        new_history = torch.cat((history, cls_ori), dim = -2)                  # [batch_size, seq_len, history_length + 1, d_input]

        relative_time = torch.cat(
            (history_time, result.unsqueeze(dim = -1)),
            dim = -1
        )                                                                      # [batch_size, seq_len, history_length + 1]
        interval_from_current_to_history = (torch.sum(relative_time, dim = -1, keepdim = True) - torch.cumsum(relative_time, dim = -1)).float()
                                                                               # [batch_size, seq_len, history_length + 1]
        # Tackle the Pytorch precision tolerance which may lead to tiny computation errors in interval_from_current_to_history
        # Tuning: Remove the relative time information part 2 in the history.
        interval_from_current_to_history[:, :, -1] = result

        emb_time_1 = self.time_scaler_function_1(self.nonneg_activate(self.time_scaler_parameter_1) * interval_from_current_to_history)
                                                                               # [batch_size, seq_len, history_length + 1]
        
        # Time activation function may be required here before feeded into the time encoder.
        emb_time = self.time_encoder(emb_time_1.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, history_length + 1, d_input]

        emb_for_integral = self.nonneg_activate(new_history) + emb_time
                                                                               # [batch_size, seq_len, history_length + 1, d_input]

        for monotonic_layer in self.decoder:
            emb_for_integral, _ = monotonic_layer(
                emb_for_integral, emb_for_integral, emb_for_integral,
                non_pad_mask = None,
                self_attn_mask = self_attn_mask)                               # [batch_size, seq_len, history_length + 1, d_input]
        
        cls_embedding = emb_for_integral[:, :, -1, :]                          # [batch_size, seq_len, d_input]

        # Integral
        integral = self.nonneg_activate(self.integral(cls_embedding)).squeeze()# [batch_size, seq_len]

        # Event prediction if expected.
        if self.event_toggle:
            # event prediction
            event = self.event_predictor(cls_embedding).squeeze()              # [batch_size, seq_len, num_event]
            return integral, event                                             # [batch_size, seq_len] + [batch_size, seq_len, num_event]
        else:
            return integral                                                    # [batch_size, seq_len]


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
            (mask, (mask.sum(dim = -1) > 0).int().unsqueeze(dim = -1)),
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

        emb_time_1 = self.time_scaler_function_1(self.nonneg_activate(self.time_scaler_parameter_1) * interval_from_current_to_history)
                                                                               # [batch_size, seq_len, resolution, history_length + 1]

        # Time activation function may be required here before feeded into the time encoder.
        emb_time = self.time_encoder(emb_time_1.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, resolution, history_length + 1, d_input]

        emb_for_integral = self.nonneg_activate(expand_new_history) + emb_time
                                                                               # [batch_size, seq_len, resolution, history_length + 1, d_input]

        for monotonic_layer in self.decoder:
            emb_for_integral, _ = monotonic_layer(
                emb_for_integral, emb_for_integral, emb_for_integral,
                non_pad_mask = mask.unsqueeze(dim = -1).unsqueeze(dim = 2),
                self_attn_mask = self_attn_mask)                               # [batch_size, seq_len, resolution, history_length + 1, d_input]
        
        cls_embedding = emb_for_integral[:, :, :, -1, :]                       # [batch_size, seq_len, resolution, d_input]

        # Integral offset
        integral_original = self.nonneg_activate(self.integral(cls_embedding)).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution]
        integral_zero = integral_original[:, :, 0].detach()                    # [batch_size, seq_len]
        integral = integral_original - integral_zero.unsqueeze(dim = -1)       # [batch_size, seq_len, resolution]


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