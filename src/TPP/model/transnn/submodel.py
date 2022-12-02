import torch.nn as nn
import torch
from einops import rearrange, reduce, repeat

from .nonneg import NonNegLinear
from .activate import *
from .transformers import TransHisEncoder, TransTPPDecoder


TA = {
    # Vanilla Softplus harms the algorithm by shifting the entire distribution into the non-nrgative area.
    # That is to say, each scalar in the output vector is bigger than log(2) if all hidden layer weights only
    # have positive numbers like what FullyNN does.
    # We have vanilla version and symmetrical version of softplus
    'softplus': nn.Softplus,
    'sym_softplus': sym_softplus,

    # Some papers have pointed out that Tanh introduces significant gradient vanishment when the input time is too big. After theoretical
    # analysis, we argue that this feature is required by approaches like FullyNN to fit long-tail functions like Hawkes intensity function.
    'tanh': nn.Tanh,
    # Yet another function that has small gradients when it has big inputs. But as the log function is not bounded above, the hard integral bound introduced
    # by tanh can be alleviated.
    'log': sym_Log,
    # This activation can perfectly show why FullyNN needs tanh to attain a trade-off between intensity function regression ability and extrapolation 
    'identity': nn.Identity,
    # Might be the redeemer, but I'm not sure.
    'ploy': sym_Polynomial
}

class TransNN(nn.Module):
    '''
    This is our implementation of Omi's paper: Fully Neural Network based Model for General Temporal Point Processes
    Hope it can work properly.

    Currently, normalization is disabled.
    Update: 2022-01-19: Now you can use data normalization via synthetic dataloader.

    Following Babylon's paper, we would check the performance of FullyNN with integral offsets.
    '''

    def __init__(self, d_history, d_intensity, num_events, dropout, history_module_layers, d_qk, 
                 integral_module_layers, mlp_layers, nonlinear, event_toggle, n_head, wq_nonneg, wk_nonneg, wv_nonneg, device):
        super(TransNN, self).__init__()
        self.device = device
        self.num_events = num_events
        self.event_toggle = event_toggle

        # For some reasons, we force that d_qk = d_v in TransDecoder
        self.d_qk = d_qk
        self.d_v = self.d_qk

        #　Maybe we can decompose self.hidden_x into the multiplication of two smaller matrices.
        self.hidden_x = nn.Parameter(
            torch.zeros((self.num_events if self.event_toggle else 1, d_intensity), \
                        device = self.device, \
                        requires_grad = True)
        )

        self.history_encoder = TransHisEncoder(num_events = num_events, d_input = d_history, d_hidden = 4 * d_history, \
                                            n_layers = history_module_layers, n_head = n_head, d_qk = d_history, \
                                            d_v = d_history, dropout = dropout, event_toggle = event_toggle, \
                                            wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg, device = device)

        self.intensity_integral_solver = TransTPPDecoder(num_events = num_events, d_input = d_history, \
                                                      d_hidden = 4 * d_history, d_qk = self.d_qk, d_v = self.d_v, \
                                                      dropout = dropout, event_toggle = event_toggle, device = device)

    def forward(self, time_history, events_history, mask_history, mask_next, time_next, mean, var):
        '''
        Args:
        1. time_history: [batch_size, seq_len]
           history time sequences for history encoder
        2. event_history:[batch_size, seq_len]
           history event sequences for history encoder (model can decide if it should use it by event_toggle)
        3. time_next:    [batch_size, seq_len]
           the value of t-t_l
        4. mask_history: [batch_size, seq_len]
           mask matrix to filter out padding events from history event sequences. 0 means the corresponding event should be masked.
        5. mask_next:    [batch_size, seq_len]
           mask matrix to filter out padding events from calculating the loss of padded sequences. 0 means
           the corresponding event should be masked.
        6. mean:         int
        7. var:          int
           For data normalization.
        '''
        # Input data normalization
        time_history_normed = (time_history - mean) / var                      # [batch_size, seq_len]
        time_history = time_history / var                                      # [batch_size, seq_len]
        time_next = time_next / var                                            # [batch_size, seq_len]

        zero = torch.zeros_like(time_next, device = self.device, dtype = torch.float32)

        history_output = self.history_encoder(events_history, time_history_normed, mask_history)
                                                                               # [batch_size, seq_len, d_history]
        
        # Integral shift ported from Attn-CM
        integral_original, intensity \
            = self.intensity_integral_solver(history = history_output, time_history = time_history,\
                                             time_next = time_next, mask_next = mask_next, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_event] * 2
        integral_zero, _ = self.intensity_integral_solver(history = output, history_time = history_time, result = zero, mask = mask, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_event]
        
        assert (integral_zero == 0).all()
        # integral_zero = integral_zero.detach()

        return integral_original, intensity

    def integral_intensity(self, history_time, history_event, result, resolution, mean, var, mask):
        '''
        Probing the learned intensity function
        Args:
        1. history_time: [batch_size, seq_len, history_length]
           history time sequences for history encoder
        2. history_event:[batch_size, seq_len, history_length]
           history event sequences for history encoder (model can decide if it should use it by event_toggle)
        3. result:       [batch_size, seq_len]
           the value of t-t_l
        4. resolution:   int
           How many interpolating points do we have in each time interval?
        5. mask:         [batch_size, seq_len, history_length]
           mask matrix to filter out padding events from the original sequences. 0 means should be masked.
        6. mean:         int
        7. var:          int
           For data normalization.
        '''
        batch_size, seq_len, history_length = history_time.shape
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = result.unsqueeze(dim = -1) * time_multiplier    # [batch_size, seq_len, resolution]

        '''
        Restore the original timestamp
        '''
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        # Input data normalization
        history_time_norm = (history_time - mean) / var                        # [batch_size, seq_len, history_length]
        history_time = (history_time) / var                                    # [batch_size, seq_len, history_length]
        result = (result) / var                                                # [batch_size, seq_len, history_length]
        
        '''
        History part
        '''
        if self.event_toggle and self.history_module == 'lstm':
            events_embeddings = self.events(history_event)                     # [batch_size, seq_len, history_length, d_history]
            history = torch.cat(
                (events_embeddings, history_time_norm.unsqueeze(dim = -1)), dim = -1
            )                                                                  # [batch_size, seq_len, history_length, d_history + 1]
        else:
            history = history_time_norm                                        # [batch_size, seq_len, history_length, d_history + 1] if we need events else [batch_size, seq_len, history_length]
        
        # Reshape hidden output for full connection layers.
        if self.history_module == 'lstm':
            output, (_, _) = self.his_encoder(history)                         # [batch_size, seq_len, history_length, d_history]
        elif self.history_module == 'transformers':
            output = self.his_encoder(history_event, history_time_norm.unsqueeze(dim = -1), mask.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, history_length, d_history]
        
        '''
        Final output
        '''
        integral, intensity = self.intensity_integral_solver.probe(
            history = output, history_time = history_time, result = result, mask = mask, resolution = resolution
        )                                                                      # 2 * [batch_size, seq_len * resolution]

        
        return integral, intensity, timestamp

    def model_probe_function(self, history_time, history_event, result, resolution, mean, var, mask):
        '''
        We use this function to dive into the fullynn and find the reason of abrupt gradient drop around 0
        Args:
        1. history_time: [batch_size, seq_len, history_length]
           history time sequences for history encoder
        2. history_event:[batch_size, seq_len, history_length]
           history event sequences for history encoder (model can decide if it should use it by event_toggle)
        3. result:       [batch_size, seq_len]
           the value of t-t_l
        4. resolution:   int
           How many interpolating points do we have in each time interval?
        5. mask:         [batch_size, seq_len, history_length]
           mask matrix to filter out padding events from the original sequences. 0 means should be masked.
        6. mean:         int
        7. var:          int
           For data normalization.
        '''
        batch_size, seq_len, history_length = history_time.shape
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = result.unsqueeze(dim = -1) * time_multiplier    # [batch_size, seq_len, resolution]

        '''
        Restore the original timestamp
        '''
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        # Input data normalization
        history_time_norm = (history_time - mean) / var                        # [batch_size, seq_len, history_length]
        history_time = (history_time) / var                                    # [batch_size, seq_len, history_length]
        result = (result) / var                                                # [batch_size, seq_len, history_length]
        
        '''
        History part
        '''
        if self.event_toggle and self.history_module == 'lstm':
            events_embeddings = self.events(history_event)                     # [batch_size, seq_len, history_length, d_history]
            history = torch.cat(
                (events_embeddings, history_time_norm.unsqueeze(dim = -1)), dim = -1
            )                                                                  # [batch_size, seq_len, history_length, d_history + 1]
        else:
            history = history_time_norm                                        # [batch_size, seq_len, history_length, d_history + 1] if we need events else [batch_size, seq_len, history_length]
        
        # Reshape hidden output for full connection layers.
        if self.history_module == 'lstm':
            output, (_, _) = self.his_encoder(history)                         # [batch_size, seq_len, history_length, d_history]
        elif self.history_module == 'transformers':
            output = self.his_encoder(history_event, history_time_norm.unsqueeze(dim = -1), mask.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, history_length, d_history]
        
        '''
        Final output
        '''
        result_dict, additional_dict = self.intensity_integral_solver.model_probe(
            history = output, history_time = history_time, result = result, mask = mask, resolution = resolution
        )                                                                      # 2 * [batch_size, seq_len * resolution]

        result = {
            **result_dict
        }

        additional_plot = {
            'heatmap': []
        }
        '''
        Additional plots
        '''
        '''
        1. Attention map for each events
        '''
        attn_matrix = additional_dict['attn']                                  # [batch_size, seq_len, resolution, n_head, seq_len, 1]
        # WIP: heatmap in a gif
        # select one specific event as an example.
        choosed_attn_matrix = attn_matrix[0, -1, -1, :, :, :].squeeze()        # [n_head, seq_len]
        additional_plot['heatmap'].append([
            'attn',
            {
                'data': choosed_attn_matrix.detach().cpu().numpy(),
                'cmap': "YlGnBu",
                'vmin': 0
            }
        ])

        return result, additional_plot, timestamp