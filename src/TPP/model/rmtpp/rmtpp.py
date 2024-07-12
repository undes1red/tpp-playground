import torch
import torch.nn as nn

from einops import rearrange, reduce, repeat


class RMTPPModule(nn.Module):
    def __init__(self, input_size, hidden_size, history_encoder_layers, dropout, event_toggle, 
                 num_events, output_size, limited_history_norm, time_scalar_min, device):
        super(RMTPPModule, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.limited_history_norm = limited_history_norm
        self.time_scalar_min = time_scalar_min

        self.time_embedding = nn.Linear(1, input_size, device = self.device)
        self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = history_encoder_layers, batch_first = True, \
                            dropout = dropout, device = self.device)
        self.project = nn.Linear(hidden_size, output_size, device = self.device)
        
        # Mark related
        self.event_embedding = nn.Embedding(num_embeddings = num_events + 1, embedding_dim = input_size,\
                                            padding_idx = num_events, device = self.device)
        self.event_mapper = nn.Linear(output_size, self.num_events, device = self.device)
        self.event_decider = nn.Softmax(dim = -1)

        self.non_neg_activation = nn.Softplus()

        self.intensity = nn.Linear(output_size, 1, device = self.device)
        self.time_scalar = nn.Linear(output_size, 1, device = self.device)
        self.base_intensity = nn.Linear(output_size, 1, device = self.device)


    def clamp_time_scalar(self, time_scalar):
        time_scalar_sign = (time_scalar >= 0).int() - (time_scalar < 0).int()  # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        shifted_time_scalar_abs_value = torch.abs(time_scalar).clamp(min = self.time_scalar_min)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        time_scalar = shifted_time_scalar_abs_value * time_scalar_sign         # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        return time_scalar


    def forward(self, events_history, time_history, time_next, mean, std, custom_events_history = False):
        '''
        This implementation is in fact an advanced RMTPP with history-event-related time scaler and base intensity.
        '''
        time_history = (time_history - mean) / std

        time_history = time_history.unsqueeze(dim = -1)                        # [batch_size, seq_len, 1]

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        if self.event_toggle:
            if custom_events_history:
                events_vec = events_history                                    # [batch_size, seq_len, input_size]
            else:
                events_vec = self.event_embedding(events_history)              # [batch_size, seq_len, input_size]
            input_vec = time_vec + events_vec
        else:
            input_vec = time_vec                                               # [batch_size, seq_len, input_size]

        hidden_history, (_, _) = self.rnn(input_vec)                           # [batch_size, seq_len, hidden_size]
        hidden_history = self.project(hidden_history)                          # [batch_size, seq_len, output_size]
        hidden_history = torch.relu(hidden_history)                            # [batch_size, seq_len, output_size]

        history_part = self.intensity(hidden_history)                          # [batch_size, seq_len, 1]

        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, 1]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, 1]

        constant = history_part * torch.exp(self.base_intensity(hidden_history))
                                                                               # [batch_size, seq_len, 1]
        time_scalar = self.time_scalar(hidden_history)                         # [batch_size, seq_len, 1]

        # time_scalar can not be zero.
        time_scalar = self.clamp_time_scalar(time_scalar)                      # [batch_size, seq_len, 1]

        time_next = (time_next) / std
        time_next = time_next.unsqueeze(dim = -1)                              # [..., batch_size, seq_len, 1]

        # reshape the parameters.
        ein_ops = f'... -> {"() " * (len(time_next.shape) - len(time_scalar.shape))}...'
        time_scalar = rearrange(time_scalar, ein_ops)                          # [..., batch_size, seq_len, 1]
        constant = rearrange(constant, ein_ops)                                # [..., batch_size, seq_len, 1]

        # Get the intensity function and corresponding integral.
        intensity = torch.exp(time_scalar * time_next) * constant              # [..., batch_size, seq_len, 1]
        integral = (intensity - constant) / time_scalar * std                  # [..., batch_size, seq_len, 1]

        mark = None
        if self.event_toggle:
            intensity, integral = intensity.sum(dim = -1), integral.sum(dim = -1)
                                                                               # [..., batch_size, seq_len]
            mark = self.event_decider(self.event_mapper(hidden_history))       # [batch_size, seq_length, num_events]


        return integral, intensity, mark, history_part
    

    def get_event_embedding(self, input_event):
        return self.event_embedding(input_event)                               # [batch_size, seq_len, input_size]


    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, resolution, mean, std):
        time_history = ((time_history - mean) / std).unsqueeze(dim = -1)

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        if self.num_events > 1:
            events_vec = self.event_embedding(events_history)                  # [batch_size, seq_len, input_size]
            input_vec = time_vec + events_vec
        else:
            input_vec = time_vec                                               # [batch_size, seq_len, input_size]

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]
        history_output = torch.relu(history_output)                            # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, 1]
        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, 1]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, 1]

        constant = (history_part * torch.exp(self.base_intensity(history_output))).unsqueeze(-1)
                                                                               # [batch_size, seq_len, 1, 1]

        time_scalar = self.time_scalar(history_output).unsqueeze(dim = -1)     # [batch_size, seq_len, 1, 1]

        # time_scalar can not be zero.
        time_scalar = self.clamp_time_scalar(time_scalar)                      # [batch_size, seq_len, 1, 1]


        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        original_time_expand = time_next.unsqueeze(dim = -1) * time_multiplier # [batch_size, seq_len, resolution]
        original_time_expand_normed = original_time_expand / std               # [batch_size, seq_len, resolution]
        expanded_time = original_time_expand_normed.unsqueeze(dim = -2)        # [batch_size, seq_len, 1, resolution]
        
        intensity_events = torch.exp(time_scalar * expanded_time) * constant   # [batch_size, seq_len, 1, resolution]
        integral_events = (intensity_events - constant) / time_scalar * std    # [batch_size, seq_len, 1, resolution]

        # aggregated timestamp
        batch_size, seq_len, _ = original_time_expand.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        
        intensity = rearrange(intensity_events, 'b s ne r -> b s r ne')        # [batch_size, seq_len, resolution, 1]
        integral = rearrange(integral_events, 'b s ne r -> b s r ne')          # [batch_size, seq_len, resolution, 1]

        return integral, intensity, timestamp


    def model_probe_function(self, events_history, time_history, time_next, resolution, mean, std):
        time_history = ((time_history - mean) / std).unsqueeze(dim = -1)

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        if self.num_events > 1:
            events_vec = self.event_embedding(events_history)                  # [batch_size, seq_len, input_size]
            input_vec = time_vec + events_vec
        else:
            input_vec = time_vec                                               # [batch_size, seq_len, input_size]

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]
        history_output = torch.relu(history_output)                            # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, 1]
        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, 1]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, 1]

        constant = (history_part * torch.exp(self.base_intensity(history_output))).unsqueeze(-1)
                                                                               # [batch_size, seq_len, 1, 1]

        time_scalar = self.time_scalar(history_output).unsqueeze(dim = -1)     # [batch_size, seq_len, 1, 1]

        # time_scalar can not be zero.
        time_scalar = self.clamp_time_scalar(time_scalar)                      # [batch_size, seq_len, 1, 1]


        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        original_time_expand = time_next.unsqueeze(dim = -1) * time_multiplier # [batch_size, seq_len, resolution]
        original_time_expand_normed = original_time_expand / std               # [batch_size, seq_len, resolution]
        expanded_time = original_time_expand_normed.unsqueeze(dim = -2)        # [batch_size, seq_len, 1, resolution]
        
        intensity_events = torch.exp(time_scalar * expanded_time) * constant   # [batch_size, seq_len, 1, resolution]
        integral_events = (intensity_events - constant) / time_scalar * std    # [batch_size, seq_len, 1, resolution]

        # aggregated timestamp
        batch_size, seq_len, _ = original_time_expand.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        
        intensity = rearrange(intensity_events, 'b s ne r -> b s r ne')        # [batch_size, seq_len, resolution, 1]
        integral = rearrange(integral_events, 'b s ne r -> b s r ne')          # [batch_size, seq_len, resolution, 1]

        '''
        Here we start constructing data dict.
        '''
        data = {}
        data['expand_intensity_for_each_event'] = intensity                    # [batch_size, seq_len, resolution, 1]
        data['expand_integral_for_each_event'] = integral                      # [batch_size, seq_len, resolution, 1]


        return data, timestamp