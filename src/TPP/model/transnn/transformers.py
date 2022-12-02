import math, torch
import torch.nn as nn
from einops import repeat, rearrange, reduce

from .layers import TransformerLayer, MultiEventDecodeLayer
from .utils import *
from .nonneg import NonNegLinear
from .activate import sym_Log


class TransHisEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_events, d_input, d_hidden,
            n_layers, n_head, d_qk, d_v, dropout,
            event_toggle, wq_nonneg, wk_nonneg, wv_nonneg,
            device):
        super(TransHisEncoder, self).__init__()
        self.device = device
        self.d_input = d_input
        self.event_toggle = event_toggle
        self.num_events = num_events

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
                [math.pow(10000.0, 2.0 * (i // 2) / d_input) for i in range(d_input)],
            device=self.device)
        self.time_emb = nn.Linear(d_input, d_input, device = self.device)

        # event type embedding
        self.event_emb = nn.Embedding(num_events + 2, d_input, padding_idx = num_events, device = self.device)

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

        result = rearrange(idx, '... -> ... 1') / self.position_vec
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:, 1::2])
        return result

    def forward(self, events_history, time_history_normed, non_pad_mask):
        """
        Encode event sequences via masked self-attention.
        1. subseq_mask can be None, implying free attention across all available items, this feature is controlled by
           option seq_mask

        Args:
        1. events_history:      [batch_size, seq_len]
        history event sequences for history encoder (model can decide if it should use it by event_toggle)
        2. time_history_normed: [batch_size, seq_len]
        history time sequences for history encoder
        3. non_pad_mask:        [batch_size, seq_len]
        pad mask tensor.
        """
        _, seq_len = events_history.shape

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        self_attn_mask_subseq = get_subsequent_mask(time_history_normed)       # [batch_size, seq_len, seq_len]
        self_attn_mask_keypad = torch.ones_like(non_pad_mask, device = self.device) - non_pad_mask
                                                                               # [batch_size, seq_len]
        self_attn_mask_keypad = repeat(self_attn_mask_keypad, 'b s -> b s s1', s1 = seq_len)
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0) # [batch_size, seq_len, seq_len]

        # relative and absolute timestamps for this event sequence.
        # we need to evaluate different time encoder here, including the naive sin/cos, or trainable vectors.
        time_emb = self.time_emb(self.encode_position_idx(time_history_normed))# [batch_size, seq_len, d_input]

        if self.event_toggle:
            events_emb = self.event_emb(events_history)                        # [batch_size, seq_len, d_input]
            history_emb = events_emb + time_emb                                # [batch_size, seq_len, d_input]
        else:
            history_emb = time_emb                                             # [batch_size, seq_len, d_input]

        for enc_layer in self.history_encoder:
            '''
            history event sequence
            '''
            history_emb, _ = enc_layer(
                history_emb, history_emb, history_emb,
                non_pad_mask = non_pad_mask,
                self_attn_mask = self_attn_mask)                               # [batch_size, seq_len, d_input]

        return history_emb


class TransTPPDecoder(nn.Module):
    '''
    Decode the intensity integral based on the history and time.
    '''
    def __init__(self, num_events, d_input, d_hidden, d_qk, d_v, dropout, event_toggle, device):
        super(TransTPPDecoder, self).__init__()
        self.device = device
        self.num_events = num_events
        self.d_input = d_input
        self.event_toggle = event_toggle
        self.d_qk = d_qk
        self.d_v = d_v

        # Position vector, used for temporal encoding
        self.position_vec = torch.tensor(
                [math.pow(10000.0, 2.0 * (i // 2) / d_input) for i in range(d_input)],
            device=self.device)
        # For time embedding
        self.time_emb = NonNegLinear(d_input, d_input, embedding_like = True, device = self.device)

        self.decoder = MultiEventDecodeLayer(d_input = d_input, d_hidden = d_hidden, n_head = num_events, d_qk = d_qk,
                                             d_v = d_v, dropout = dropout, device = device)

        self.integral = NonNegLinear(d_v, 1, bias = False, device = device)
        self.nonneg_activate = nn.Softplus()
    
    def forward(self, history, time_history, time_next, mask_next, mean, var):
        '''
        1. history:      [batch_size, seq_len, d_input]
           The history information encoded by Transformers.
        2. time_history: [batch_size, seq_len]
           For relative time calculation.
        3. time_next:    [batch_size, seq_len]
           Current timestamp for calculating the integral
        4. mask_next:    [batch_size, seq_len]
           mask matrix to filter out padding events from the original sequences. 0 means should be masked out.
        '''
        batch_size, seq_len, _ = history.shape

        # We preprocess the mask as the sequence here is supposed to be 1 item longer than the history.
        self_attn_mask_subseq = get_subsequent_mask(time_next)                 # [batch_size, seq_len, seq_len]
        self_attn_mask_keypad = torch.ones_like(mask_next, device = self.device) - mask_next
                                                                               # [batch_size, seq_len]
        self_attn_mask_keypad = repeat(self_attn_mask_keypad, 'b s -> b s s1', s1 = seq_len)
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0) # [batch_size, seq_len, seq_len]

        if self.event_toggle:
            history = repeat(history, 'b s di -> b ne s di', ne = self.num_events)
                                                                               # [batch_size, num_events, seq_len, d_input]
            time_next = repeat(time_next, 'b s -> b ne s', ne = self.num_events)
                                                                               # [batch_size, num_events, seq_len]
        
        # FIXME: embedding-like time embedding module when num_events > 1.
        time_emb = self.time_emb(time_next)                                    # [batch_size, seq_len, num_events, d_input] if we need events else [batch_size, seq_len, d_input]
        mixed_information = history + time_emb                                 # [batch_size, seq_len, num_events, d_input] if we need events else [batch_size, seq_len, d_input]
        
        integral_vector, _, = self.decoder(
            mixed_information, mixed_information, mixed_information,
            self_attn_mask = self_attn_mask, non_pad_mask = mask_next
        )                                                                      # [batch_size, seq_len, num_events, d_input] if we need events else [batch_size, seq_len, d_input]

        cls_ori = self.cls.repeat(batch_size, seq_len, 1, 1)                   # [batch_size, seq_len, 1, d_input]
        new_history = torch.cat((history, cls_ori), dim = -2)                  # [batch_size, seq_len, history_length + 1, d_input]

        result = result.unsqueeze(dim = -1).repeat(1, 1, self.num_types)       # [batch_size, seq_len, num_types]
        result.requires_grad = True
        
        result_1 = result.unsqueeze(dim = -2)                                  # [batch_size, seq_len, 1, num_types]
        relative_time = torch.cat(
            (history_time.unsqueeze(dim = -1).repeat(1, 1, 1, self.num_types), result_1),
            dim = -2
        )                                                                      # [batch_size, seq_len, history_length + 1, num_types]
        # Tackle the Pytorch precision tolerance which may lead to tiny computation errors in interval_from_current_to_history
        # Tuning: Remove the relative time information part 2 in the history.
        interval_from_current_to_history[:, :, -1, :] = result                 # [batch_size, seq_len, history_length + 1, num_types]

        emb_history_time = self.nonneg_activate(self.time_scaler_parameter_1) * interval_from_current_to_history
                                                                               # [batch_size, seq_len, history_length + 1, num_types]
 
        # Time activation functions are required here before feeded into the time encoder.
        emb_time = emb_history_time.unsqueeze(dim = -1) * self.nonneg_activate(self.history_time_encoder)
                                                                               # [batch_size, seq_len, history_length + 1, num_types, d_v]

        new_history = self.history_expand(new_history).reshape(*new_history.shape[:3], self.num_types, -1)
                                                                               # [batch_size, seq_len, history_length + 1, num_types, d_v]
        # emb_for_integral = self.nonneg_activate(new_history) + self.nonneg_activate(emb_time)

        new_history[:, :, -1, :, :] = -1e9
        emb_for_integral = self.nonneg_activate(new_history) + emb_time
                                                                               # [batch_size, seq_len, history_length + 1, num_types, d_v]

        for monotonic_layer in self.decoder:
            emb_for_integral, _ = monotonic_layer(
                emb_for_integral, emb_for_integral[:, :, -1, :, :].unsqueeze(dim = -3), emb_for_integral,
                non_pad_mask = None,
                self_attn_mask = self_attn_mask)                               # [batch_size, seq_len, num_event, d_v]
        
        # Integral
        integral = self.integral(emb_for_integral).squeeze(dim = -1)           # [batch_size, seq_len, num_event]

        # Intensity values.
        intensity = torch.autograd.grad(
            outputs = integral,
            inputs = result,
            grad_outputs = torch.ones_like(integral),
            create_graph = True,
        )[0]                                                                   # [batch_size, seq_len, num_events]
        
        return integral, intensity


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
        result = result.unsqueeze(dim = -1).repeat(1, 1, 1, self.num_types)    # [batch_size, seq_len, resolution, num_types]
        result.requires_grad = True

        # We preprocess the mask as the sequence here is supposed to be 1 item longer than the history.
        mask = torch.cat(
            (mask, (mask.sum(dim = -1) > 0).int().unsqueeze(dim = -1)),
            dim = -1
        )                                                                      # [batch_size, seq_len, history_length + 1]
        self_attn_mask_keypad = (torch.ones_like(mask, device = self.device) - mask).unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, history_length + 1, 1]
        self_attn_mask = self_attn_mask_keypad.gt(0)                           # [batch_size, seq_len, history_length + 1, 1]
        self_attn_mask = self_attn_mask.unsqueeze(dim = 2).repeat(1, 1, resolution, 1, 1)
                                                                               # [batch_size, seq_len, resolution, history_length + 1, 1]

        # Preparing the attention input
        # History part
        cls_ori = self.cls.repeat(batch_size, seq_len, 1, 1)                   # [batch_size, seq_len, 1, d_input]
        new_history = torch.cat((history, cls_ori), dim = -2)                  # [batch_size, seq_len, history_length + 1, d_input]
        expand_new_history = new_history.unsqueeze(dim = 2).repeat(1, 1, resolution, 1, 1)
                                                                               # [batch_size, seq_len, resolution, history_length + 1, d_input]
        
        # Time part
        result_1 = result.unsqueeze(dim = -2)                                  # [batch_size, seq_len, resolution, 1, num_types]
        history_time = history_time.unsqueeze(dim = 2).repeat(1, 1, resolution, 1)
                                                                               # [batch_size, seq_len, resolution, history_length]
        relative_time = torch.cat(
            (history_time.unsqueeze(dim = -1).repeat(1, 1, 1, 1, self.num_types), result_1),
            dim = -2
        )                                                                      # [batch_size, seq_len, resolution, history_length + 1, num_types]
        interval_from_current_to_history = (torch.sum(relative_time, dim = -2, keepdim = True) - torch.cumsum(relative_time, dim = -2)).float()
                                                                               # [batch_size, seq_len, resolution, history_length + 1, num_types]
        # Tackle the floatpoint number precision tolerance which may lead to tiny computation errors in interval_from_current_to_history
        interval_from_current_to_history[:, :, :, -1, :] = result

        emb_time_1 = self.nonneg_activate(self.time_scaler_parameter_1) * interval_from_current_to_history
                                                                               # [batch_size, seq_len, resolution, history_length + 1, num_types]
        # Time activation function may be required here before feeded into the time encoder.
        emb_time = emb_time_1.unsqueeze(dim = -1) * self.nonneg_activate(self.history_time_encoder)
                                                                               # [batch_size, seq_len, resolution, history_length + 1, num_types, d_v]
        expand_new_history = self.history_expand(expand_new_history).reshape(*expand_new_history.shape[:4], self.num_types, -1)
                                                                               # [batch_size, seq_len, resolution, history_length + 1, num_types, d_v]
        
        expand_new_history[:, :, :, -1, :, :] = -1e9
        # emb_for_integral = self.nonneg_activate(new_history) + self.nonneg_activate(emb_time)
        emb_for_integral = self.nonneg_activate(expand_new_history) + emb_time
                                                                               # [batch_size, seq_len, resolution, history_length + 1, num_types, d_v]

        for monotonic_layer in self.decoder:
            emb_for_integral, _ = monotonic_layer(
                emb_for_integral, emb_for_integral[:, :, :, -1, :, :].unsqueeze(dim = -3), emb_for_integral,
                non_pad_mask = None,
                self_attn_mask = self_attn_mask)                               # [batch_size, seq_len, resolution, num_event, d_v]
        
        # Integral offset
        integral = self.integral(emb_for_integral).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution, num_event]

        # Intensity values and their sum.
        intensity = torch.autograd.grad(
            outputs = integral,
            inputs = result,
            grad_outputs = torch.ones_like(integral),
            create_graph = True,
        )[0]                                                                   # [batch_size, seq_len, resolution, num_event]
        result.requires_grad = False

        integral = integral.sum(dim = -1).reshape(batch_size, -1)              # [batch_size, seq_len, resolution]
        intensity = intensity.sum(dim = -1).reshape(batch_size, -1)            # [batch_size, seq_len, resolution]

        return integral, intensity

    def model_probe(self, history, history_time, result, mask, resolution):
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
        result = result.unsqueeze(dim = -1).repeat(1, 1, 1, self.num_types)    # [batch_size, seq_len, resolution, num_types]
        result.requires_grad = True

        # We preprocess the mask as the sequence here is supposed to be 1 item longer than the history.
        mask = torch.cat(
            (mask, (mask.sum(dim = -1) > 0).int().unsqueeze(dim = -1)),
            dim = -1
        )                                                                      # [batch_size, seq_len, history_length + 1]
        self_attn_mask_keypad = (torch.ones_like(mask, device = self.device) - mask).unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, history_length + 1, 1]
        self_attn_mask = self_attn_mask_keypad.gt(0)                           # [batch_size, seq_len, history_length + 1, 1]
        self_attn_mask = self_attn_mask.unsqueeze(dim = 2).repeat(1, 1, resolution, 1, 1)
                                                                               # [batch_size, seq_len, resolution, history_length + 1, 1]

        # Preparing the attention input
        # History part
        cls_ori = self.cls.repeat(batch_size, seq_len, 1, 1)                   # [batch_size, seq_len, 1, d_input]
        new_history = torch.cat((history, cls_ori), dim = -2)                  # [batch_size, seq_len, history_length + 1, d_input]
        expand_new_history = new_history.unsqueeze(dim = 2).repeat(1, 1, resolution, 1, 1)
                                                                               # [batch_size, seq_len, resolution, history_length + 1, d_input]
        
        # Time part
        result_1 = result.unsqueeze(dim = -2)                                  # [batch_size, seq_len, resolution, 1, num_types]
        history_time = history_time.unsqueeze(dim = 2).repeat(1, 1, resolution, 1)
                                                                               # [batch_size, seq_len, resolution, history_length]
        relative_time = torch.cat(
            (history_time.unsqueeze(dim = -1).repeat(1, 1, 1, 1, self.num_types), result_1),
            dim = -2
        )                                                                      # [batch_size, seq_len, resolution, history_length + 1, num_types]
        interval_from_current_to_history = (torch.sum(relative_time, dim = -2, keepdim = True) - torch.cumsum(relative_time, dim = -2)).float()
                                                                               # [batch_size, seq_len, resolution, history_length + 1, num_types]
        # Tackle the floatpoint number precision tolerance which may lead to tiny computation errors in interval_from_current_to_history
        interval_from_current_to_history[:, :, :, -1, :] = result

        # emb_time_1 = self.time_scaler_function_1(self.nonneg_activate(self.time_scaler_parameter_1) * interval_from_current_to_history)
        emb_time_1 = self.nonneg_activate(self.time_scaler_parameter_1) * interval_from_current_to_history
                                                                               # [batch_size, seq_len, resolution, history_length + 1, num_types]

        # Time activation function may be required here before feeded into the time encoder.
        emb_time = emb_time_1.unsqueeze(dim = -1) * self.nonneg_activate(self.history_time_encoder)
                                                                               # [batch_size, seq_len, resolution, history_length + 1, num_types, d_v]
        expand_new_history = self.history_expand(expand_new_history).reshape(*expand_new_history.shape[:4], self.num_types, -1)
                                                                               # [batch_size, seq_len, resolution, history_length + 1, num_types, d_v]

        expand_new_history[:, :, :, -1, :, :] = -1e9
        # emb_for_integral = self.nonneg_activate(new_history) + self.nonneg_activate(emb_time)
        emb_for_integral = self.nonneg_activate(expand_new_history) + emb_time
                                                                               # [batch_size, seq_len, resolution, history_length + 1, num_types, d_v]

        for monotonic_layer in self.decoder:
            emb_for_integral, attn = monotonic_layer(
                emb_for_integral, emb_for_integral[:, :, :, -1, :, :].unsqueeze(dim = -3), emb_for_integral,
                non_pad_mask = None,
                self_attn_mask = self_attn_mask)                               # [batch_size, seq_len, resolution, num_event, d_v] + [batch_size, seq_len, resolution, n_head, seq_len, 1]

        # Integral offset
        integral = self.integral(emb_for_integral).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution, num_event]

        # Intensity values and their sum.
        intensity = torch.autograd.grad(
            outputs = integral,
            inputs = result,
            grad_outputs = torch.ones_like(integral),
            create_graph = True,
        )[0]                                                                   # [batch_size, seq_len, resolution, num_event]
        result.requires_grad = False
        
        intensity_for_all_events = {}
        integral_for_all_events = {}
        loss = {}
        for i in range(self.num_types):
            intensity_for_all_events[f'event_intensity_ {i}'] = intensity[:, :, :, i].reshape(batch_size, -1)
                                                                               # [batch_size, seq_len * resolution]
            integral_for_all_events[f'event_integral_{i}'] = integral[:, :, :, i].reshape(batch_size, -1)
                                                                               # [batch_size, seq_len * resolution]
            loss[f'event_loss_{i}'] = (-torch.log(intensity[:, :, :, i]) + integral[:, :, :, i]).reshape(batch_size, -1)
                                                                               # [batch_size, seq_len * resolution]

        integral = integral.sum(dim = -1).reshape(batch_size, -1)              # [batch_size, seq_len * resolution]
        intensity = intensity.sum(dim = -1).reshape(batch_size, -1)            # [batch_size, seq_len * resolution]

        result = {
            'integral_sum': integral,
            'intensity_sum': intensity,
            **intensity_for_all_events,
            **integral_for_all_events,
            **loss,
            'loss': -torch.log(intensity) + integral
        }

        additional_result = {
            'attn': attn
        }

        return result, additional_result