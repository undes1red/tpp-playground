import torch
import torch.nn as nn
from .special import EI

class RMTPPModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_events, output_size, device, increase):
        super(RMTPPModule, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_events = num_events

        # TODO: do we add additional dummy events here?
        self.time_embedding = nn.Linear(1, input_size, device = self.device)
        self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, \
                            dropout = dropout, device = self.device)
        self.project = nn.Linear(hidden_size, output_size, device = self.device)
        
        # Mark related
        if self.num_events > 1:
            self.event_embedding = nn.Embedding(num_embeddings = num_events + 1, embedding_dim = input_size, device = self.device)
            self.event_decider = nn.Softmax()
        self.non_neg_activation = nn.Softplus()
        self.increase_or_decrease = 1 if increase else -1

        self.ei = EI
        # intensity related
        self.intensity = nn.Linear(output_size, self.num_events, device = self.device)
        self.time_scalar = nn.Linear(output_size, self.num_events, device = self.device)
        self.base_intensity = nn.Linear(output_size, self.num_events, device = self.device)

        # self.time_scalar = nn.parameter.Parameter(torch.tensor(0., device = self.device))
        # self.base_intensity = nn.parameter.Parameter(torch.tensor(.1, device = self.device))

    def forward(self, event_history, time_history, time_next):
        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        if self.num_events > 1:
            event_vec = self.event_embedding(event_history)                    # [batch_size, seq_len, input_size]
            input_vec = time_vec + event_vec
        else:
            input_vec = time_vec                                               # [batch_size, seq_len, input_size]

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, num_events]
        history_part = torch.tanh(history_part)                                # [batch_size, seq_len, num_events]
        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, num_events]

        time_scalar = self.non_neg_activation(self.time_scalar(history_output)) * self.increase_or_decrease
                                                                               # [batch_size, seq_len, num_events]
        constant = history_part * torch.exp(self.base_intensity(history_output))
                                                                               # [batch_size, seq_len, num_events]
        intensity_events = torch.exp(time_scalar * time_next) * constant       # [batch_size, seq_len, num_events]
        integral_events = (intensity_events - constant) / time_scalar          # [batch_size, seq_len, num_events]
        intensity, integral = intensity_events.sum(-1, keepdim = True),\
                              integral_events.sum(-1, keepdim = True)          # [batch_size, seq_len, 1]
        expectation = (- torch.exp(constant/ time_scalar) * self.ei.apply(-constant / time_scalar) / time_scalar)
                                                                               # [batch_size, seq_len, num_events]

        # For event, we need (batch, seq_length, num_event)
        if self.num_events > 1:
            mark = self.event_decider(intensity_events)                        # [batch_size, seq_length, num_events]
        else:
            mark = None

        return intensity, integral, mark, expectation, history_part

    def intensity_integral(self, event_history, time_history, time_next, resolution):
        batch_size, seq_len, _ = time_history.shape
        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        if self.num_events > 1:
            event_vec = self.event_embedding(event_history)                    # [batch_size, seq_len, input_size]
            input_vec = time_vec + event_vec
        else:
            input_vec = time_vec                                               # [batch_size, seq_len, input_size]

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, num_events]
        history_part = torch.tanh(history_part)                                # [batch_size, seq_len, num_events]
        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, num_events]

        time_scalar = (self.non_neg_activation(self.time_scalar(history_output)) * self.increase_or_decrease).unsqueeze(-1)
                                                                               # [batch_size, seq_len, num_events, 1]
        constant = (history_part * torch.exp(self.base_intensity(history_output))).unsqueeze(-1)
                                                                               # [batch_size, seq_len, num_events, 1]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = (time_next * time_multiplier).unsqueeze(-2)            # [batch_size, seq_len, 1, resolution]
        intensity_events = torch.exp(time_scalar * expanded_time) * constant   # [batch_size, seq_len, num_events, resolution]
        integral_events = (intensity_events - constant) / time_scalar          # [batch_size, seq_len, num_events, resolution]
        intensity, integral = intensity_events.sum(dim = -2), integral_events.sum(dim = -2)
                                                                               # [batch_size, seq_len, resolution]

        # For event, we need (batch, seq_length, num_event)
        # mark = self.event_decider(self.marker(history_output))               # [batch, seq_length, num_event]

        intensity, integral,= intensity.reshape(batch_size, -1), integral.reshape(batch_size, -1)
                                                                               # [batch_size, seq_len * resolution]

        # aggregated timestamp
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_time.squeeze(-2).diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        return intensity, integral, timestamp