import torch
import torch.nn as nn
from .special import EI
from einops import rearrange, reduce, repeat

class RMTPPModule(nn.Module):
    def __init__(self, input_size, hidden_size, history_encoder_layers, dropout, num_events, output_size, limited_history_norm, 
                 original_mark_generation, device):
        super(RMTPPModule, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_events = num_events
        self.limited_history_norm = limited_history_norm

        # Use this hyperparameter to select the marker prediction module.
        # True: The event prediction module proposed in the RMTPP paper, extracting markers directly from the hidden state.
        # False: The event prediction module following the MTPP paradigm.
        self.original_mark_generation = original_mark_generation

        self.time_embedding = nn.Linear(1, input_size, device = self.device)
        self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = history_encoder_layers, batch_first = True, \
                            dropout = dropout, device = self.device)
        self.project = nn.Linear(hidden_size, output_size, device = self.device)
        
        # Mark related
        if self.num_events > 1:
            self.event_embedding = nn.Embedding(num_embeddings = num_events + 1, embedding_dim = input_size,\
                                                padding_idx = num_events, device = self.device)
            self.event_mapper = nn.Linear(output_size, self.num_events, device = self.device)
            self.event_decider = nn.Softmax(dim = -1)

        self.non_neg_activation = nn.Softplus()

        # intensity related
        if self.original_mark_generation:
            self.intensity = nn.Linear(output_size, 1, device = self.device)
            self.time_scalar = nn.Linear(output_size, 1, device = self.device)
            self.base_intensity = nn.Linear(output_size, 1, device = self.device)
        else:
            self.intensity = nn.Linear(output_size, self.num_events, device = self.device)
            self.time_scalar = nn.Linear(output_size, self.num_events, device = self.device)
            self.base_intensity = nn.Linear(output_size, self.num_events, device = self.device)


    def forward(self, events_history, time_history, time_next, mean, var):
        '''
        This implementation is in fact an advanced RMTPP with history-event-related time scaler and base intensity.
        '''
        time_history = (time_history) / var
        time_next = (time_next) / var

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        if self.num_events > 1:
            events_vec = self.event_embedding(events_history)                  # [batch_size, seq_len, input_size]
            input_vec = time_vec + events_vec
        else:
            input_vec = time_vec                                               # [batch_size, seq_len, input_size]

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]
        history_output = torch.relu(history_output)                            # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]

        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]

        constant = history_part * torch.exp(self.base_intensity(history_output))
                                                                               # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]
        time_scalar = self.time_scalar(history_output)                         # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]

        # time_scalar can not be zero.
        time_scalar_sign = (time_scalar >= 0).int() - (time_scalar < 0).int()  # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]
        shifted_time_scalar_abs_value = torch.abs(time_scalar).clamp(min = 1e-4)
                                                                               # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]
        time_scalar = shifted_time_scalar_abs_value * time_scalar_sign         # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]

        # Get the intensity function and corresponding integral.
        intensity_events = torch.exp(time_scalar * time_next) * constant       # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]
        integral_events = (intensity_events - constant) / time_scalar          # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]

        if self.original_mark_generation:
            intensity, integral = intensity_events.sum(-1, keepdim = True),\
                                  integral_events.sum(-1, keepdim = True)      # [batch_size, seq_len, 1]
            if self.num_events > 1:
                mark = self.event_decider(self.event_mapper(history_output))   # [batch_size, seq_length, num_events]
            else:
                mark = None
            return intensity, integral, mark, history_part
        else:
            mark = intensity_events / (intensity_events.sum(dim = -1, keepdim = True) + 1e-6)
                                                                               # [batch_size, seq_len, num_events]
            return intensity_events, integral_events, mark, history_part
        
        # Perhaps, I get this expression by wolframalpha, too.
        # expectation = (- torch.exp(constant/ time_scalar) * self.ei.apply(-constant / time_scalar) / time_scalar)
                                                                               # [batch_size, seq_len, num_events]
        # expectation = expectation * var                                      # [batch_size, seq_len, num_events]

    def intensity_integral(self, events_history, time_history, time_next, resolution, mean, var, sum = True):
        time_history = time_history / var

        if len(time_next.shape) == 3:
            '''
            Normal Mode
            '''
            async_time = False
        else:
            async_time = True

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)

        original_time_expand = time_next * time_multiplier                     # [batch_size, seq_len, resolution]
        original_time_expand_normed = original_time_expand / var               # [batch_size, seq_len, resolution]

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        if self.num_events > 1:
            events_vec = self.event_embedding(events_history)                  # [batch_size, seq_len, input_size]
            input_vec = time_vec + events_vec
        else:
            input_vec = time_vec                                               # [batch_size, seq_len, input_size]

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]
        history_output = torch.relu(history_output)                            # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]
        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, num_events] if self.original_mark_generation == False else [batch_size, seq_len, 1]

        constant = (history_part * torch.exp(self.base_intensity(history_output))).unsqueeze(-1)
                                                                               # [batch_size, seq_len, num_events, 1] if self.original_mark_generation == False else [batch_size, seq_len, 1, 1]

        time_scalar = self.time_scalar(history_output).unsqueeze(dim = -1)     # [batch_size, seq_len, num_events, 1] if self.original_mark_generation == False else [batch_size, seq_len, 1, 1]

        # time_scalar can not be zero.
        time_scalar_sign = (time_scalar >= 0).int() - (time_scalar < 0).int()  # [batch_size, seq_len, num_events, 1] if self.original_mark_generation == False else [batch_size, seq_len, 1]
        shifted_time_scalar_abs_value = torch.abs(time_scalar).clamp(min = 1e-4)
                                                                               # [batch_size, seq_len, num_events, 1] if self.original_mark_generation == False else [batch_size, seq_len, 1]
        time_scalar = shifted_time_scalar_abs_value * time_scalar_sign         # [batch_size, seq_len, num_events, 1] if self.original_mark_generation == False else [batch_size, seq_len, 1]

        if async_time:
            sum = False
            expanded_time = original_time_expand_normed                        # [batch_size, seq_len, num_events, resolution]
        else:
            expanded_time = original_time_expand_normed.unsqueeze(-2)          # [batch_size, seq_len, 1, resolution]
        
        intensity_events = torch.exp(time_scalar * expanded_time) * constant   # [batch_size, seq_len, num_events, resolution] if self.original_mark_generation == False else [batch_size, seq_len, 1, resolution]
        integral_events = (intensity_events - constant) / time_scalar          # [batch_size, seq_len, num_events, resolution] if self.original_mark_generation == False else [batch_size, seq_len, 1, resolution]

        if sum:
            intensity, integral = intensity_events.sum(dim = -2), integral_events.sum(dim = -2)
                                                                               # [batch_size, seq_len, resolution]
                    
            intensity = rearrange(intensity, 'b s r -> b (s r)')               # [batch_size, seq_len * resolution]
            integral = rearrange(integral, 'b s r -> b (s r)')                 # [batch_size, seq_len * resolution]
        else:
            intensity = rearrange(intensity_events, 'b s ne r -> b (s r) ne')  # [batch_size, seq_len * resolution, num_events]
            integral = rearrange(integral_events, 'b s ne r -> b (s r) ne')    # [batch_size, seq_len * resolution, num_events]

        # aggregated timestamp
        if async_time:
            batch_size, seq_len, _, _ = original_time_expand.shape
            timestamp = torch.cat(
                (torch.zeros((batch_size, seq_len, self.num_events, 1), device = self.device), original_time_expand.diff(dim = -1)),
                dim = -1)                                                      # [batch_size, seq_len, resolution]
            timestamp = rearrange(timestamp, 'b s ne r -> b (s r) ne')         # [batch_size, seq_len * resolution]
        else:
            batch_size, seq_len, _ = original_time_expand.shape
            timestamp = torch.cat(
                (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
                dim = -1)                                                      # [batch_size, seq_len, resolution]
            timestamp = rearrange(timestamp, 'b s r -> b (s r)')               # [batch_size, seq_len * resolution]

        return intensity, integral, timestamp