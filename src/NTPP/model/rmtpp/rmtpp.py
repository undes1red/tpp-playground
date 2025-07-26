import torch
import torch.nn as nn

from einops import rearrange, reduce, repeat


class RMTPPModule(nn.Module):
    def __init__(self, input_size, hidden_size, history_encoder_layers, dropout, 
                 num_events, output_size, limited_history_norm, time_scalar_min, device):
        '''
        This function creates a MRMTPP model.
        
        ### Args
            * ```int``` input_size
              The dimension of the event representation that is fed into the history encoder.
            * ```int``` hidden_size
              The dimension of the history representation.
            * ```int``` history_encoder_layers
              How many layer of RNN our model will have?
            * ```float``` dropout
              Dropout rate for the history encoder. Only works when history_encoder_layers > 1.
            * ```int``` output_size
              The dimension of the intensity representation.
            * ```bool``` limited_history_norm
              If true, we will normalize the intensity representation by tanh()
            * ```float``` time_scalar_min
              The integral of the intensity function has the reciprocal of the time_scalar. This means the time scalar
              can not be 0. This parameter sets how small the time_scalar can be.
            * ```torch.device``` device
              Running models on GPU or CPU?
        '''
        super(RMTPPModule, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_events = num_events
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
        '''
        This function clamp the time_scalar so that the integral won't be affected by divided-by-zero issue.
        
        ### Args
            * ```torch.tensor``` time_scalar
              shape: [batch_size, seq_len, num_events]
              The original time_scalar.
              
        ### Outputs
            * ```torch.tensor``` time_scalar
              shape: [batch_size, seq_len, num_events]
              The clamped time_scalar with gradients attached.
        '''
        time_scalar_sign = (time_scalar >= 0).int() - (time_scalar < 0).int()  # [batch_size, seq_len, num_events]
        shifted_time_scalar_abs_value = torch.abs(time_scalar).clamp(min = self.time_scalar_min)
                                                                               # [batch_size, seq_len, num_events]
        time_scalar = shifted_time_scalar_abs_value * time_scalar_sign         # [batch_size, seq_len, num_events]
        return time_scalar


    def forward(self, events_history, time_history, time_next, mean, std, custom_events_history = False):
        '''
        MRMTPP's forwardpropagation function for training.
        
        ### Args
            * ```torch.tensor``` events_history
              shape: ```[batch_size, seq_len]```
              Historical event sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next event will happen.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.
            * ```bool``` custom_events_history
              when true, the events_history will be the mark embedding of historical events.
        ### Outputs
            * ```torch.tensor``` integral_all_events
              shape: ```[..., batch_size, seq_len, num_events]```
              The value of \\Lambda^*(m, t) on [t_{i-1}, t_i).
            * ```torch.tensor``` intensity_all_events
              shape: ```[..., batch_size, seq_len, num_events]```
              The value of \\lambda^*(m, t) on at t_i.
            * ```torch.tensor``` history_part
              shape: ```[batch_size, seq_len, num_events]```
        '''
        time_history = (time_history - mean) / std

        time_history = time_history.unsqueeze(dim = -1)                        # [batch_size, seq_len, 1]

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        if custom_events_history:
            events_vec = events_history                                        # [batch_size, seq_len, input_size]
        else:
            events_vec = self.event_embedding(events_history)                  # [batch_size, seq_len, input_size]
        input_vec = time_vec + events_vec

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

        intensity, integral = intensity.sum(dim = -1), integral.sum(dim = -1)  # [..., batch_size, seq_len]
        mark = self.event_decider(self.event_mapper(hidden_history))           # [batch_size, seq_length, num_events]

        return integral, intensity, mark, history_part
    

    def get_event_embedding(self, input_event):
        return self.event_embedding(input_event)                               # [batch_size, seq_len, input_size]


    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, resolution, mean, std):
        '''
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks share the sampled timestmaps, so the dimension of time_next does not include num_event.
        
        ### Args
            * ```torch.tensor``` events_history
              shape: ```[batch_size, seq_len]```
              Historical event sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next event will happen.
            * ```int``` resolution
              The number of interpolated points in a time interval between two adjoint events for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.
        ### Outputs
            * ```torch.tensor``` integral
              shape: ```[..., batch_size, seq_len, resolution, num_events]```
              The value of \\Lambda^*(m, t) at sampled times.
            * ```torch.tensor``` intensity
              shape: ```[..., batch_size, seq_len, resolution, num_events]```
              The value of \\lambda^*(m, t) at sampled times.
            * ```torch.tensor``` original_time_expand
              shape: ```[..., batch_size, seq_len, resolution]```
              The value of sampled times.
        '''
        time_history = ((time_history - mean) / std).unsqueeze(dim = -1)

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        events_vec = self.event_embedding(events_history)                      # [batch_size, seq_len, input_size]
        input_vec = time_vec + events_vec

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

        intensity = rearrange(intensity_events, 'b s ne r -> b s r ne')        # [batch_size, seq_len, resolution, 1]
        integral = rearrange(integral_events, 'b s ne r -> b s r ne')          # [batch_size, seq_len, resolution, 1]

        return integral, intensity, original_time_expand