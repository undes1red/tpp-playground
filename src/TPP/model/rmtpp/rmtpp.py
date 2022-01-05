import torch
import torch.nn as nn
import torch.nn.functional as F
from .special import EI

class RMTPP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_events, output_size, device):
        super(RMTPP, self).__init__()
        self.device = device

        # TODO: do we add additional dummy events here?
        self.event_embedding = nn.Embedding(num_embeddings = num_events, embedding_dim = input_size)
        self.time_embedding = nn.Linear(1, input_size)
        self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, \
                            dropout = dropout)
        self.project = nn.Linear(hidden_size, output_size)
        
        # Mark related
        self.marker = nn.Linear(output_size, num_events)
        self.decide = nn.Softmax(dim = -1)

        # intensity related
        self.intensity = nn.Linear(output_size, 1)
        self.time_scalar = torch.tensor(.1)
        self.base_intensity = torch.tensor(.1)

    def forward(self, event, time):
        # time_norm = ((time - mean)/var).float()
        event_vec = self.event_embedding(event.long())
        time_vec = self.time_embedding(time.unsqueeze(-1))
        output, (_, _) = self.rnn(input = time_vec + event_vec)
        # output shape: (batch, seq_length, H_out)
        # We need (batch, seq_length)

        history_part = self.intensity(torch.tanh(output)).squeeze()
        intensity = torch.exp(history_part + self.time_scalar * time + self.base_intensity)
        integral = (intensity - torch.exp(history_part + self.base_intensity)) / self.time_scalar
        # For event, we need (batch, seq_length, num_event)
        mark = self.decide(self.marker(output))

        return intensity, integral, mark

class RMTPPModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_events, output_size, device, increase):
        super(RMTPPModule, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        # TODO: do we add additional dummy events here?
        self.event_embedding = nn.Embedding(num_embeddings = num_events + 1, embedding_dim = input_size, device = self.device)
        self.time_embedding = nn.Linear(1, input_size, device = self.device)
        self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, \
                            dropout = dropout, device = self.device)
        self.project = nn.Linear(hidden_size, output_size, device = self.device)
        
        # Mark related
        self.marker = nn.Linear(output_size, num_events, device = self.device)
        self.event_decider = nn.Softmax(dim = -1)
        self.non_neg_activation = nn.Softplus()
        self.increase_or_decrease = 1 if increase else -1

        self.ei = EI
        # intensity related
        self.intensity = nn.Linear(output_size, 1, device = self.device)
        self.time_scalar = nn.Linear(output_size, 1, device = self.device)
        self.base_intensity = nn.Linear(output_size, 1, device = self.device)

        # self.time_scalar = nn.parameter.Parameter(torch.tensor(0., device = self.device))
        # self.base_intensity = nn.parameter.Parameter(torch.tensor(.1, device = self.device))

    def forward(self, event_history, time_history, time_next):
        event_vec = self.event_embedding(event_history)                        # [batch_size, seq_len, input_size]
        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        input_vec = time_vec + event_vec

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, 1]
        history_part = torch.tanh(history_part)                                # [batch_size, seq_len, 1]
        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, 1]

        time_scalar = self.non_neg_activation(self.time_scalar(history_output)) * self.increase_or_decrease
                                                                               # [batch_size, seq_len, 1]
        constant = history_part * torch.exp(self.base_intensity(history_output))
                                                                               # [batch_size, seq_len, 1]
        intensity = torch.exp(time_scalar * time_next) * constant              # [batch_size, seq_len, 1]
        integral = (intensity - constant) / time_scalar                        # [batch_size, seq_len, 1]
        expectation = - torch.exp(constant/ time_scalar) * self.ei.apply(-constant/ time_scalar) / time_scalar
                                                                               # [batch_size, seq_len, 1]

        # For event, we need (batch, seq_length, num_event)
        mark = self.event_decider(self.marker(history_output))                 # [batch, seq_length, num_event]

        return intensity, integral, mark, expectation, history_part

    def intensity_integral(self, event_history, time_history, time_next, resolution):
        batch_size, seq_len = event_history.shape
        event_vec = self.event_embedding(event_history)                        # [batch_size, seq_len, input_size]
        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        input_vec = time_vec + event_vec

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, 1]
        history_part = torch.tanh(history_part)                                # [batch_size, seq_len, 1]
        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, 1]

        time_scalar = self.non_neg_activation(self.time_scalar(history_output)) * self.increase_or_decrease
                                                                               # [batch_size, seq_len, 1]
        constant = history_part * torch.exp(self.base_intensity(history_output))
                                                                               # [batch_size, seq_len, 1]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = time_next * time_multiplier                            # [batch_size, seq_len, resolution]
        intensity = torch.exp(time_scalar * expanded_time) * constant          # [batch_size, seq_len, resolution]
        integral = (intensity - constant) / time_scalar                        # [batch_size, seq_len, resolution]

        # For event, we need (batch, seq_length, num_event)
        # mark = self.event_decider(self.marker(history_output))               # [batch, seq_length, num_event]

        intensity, integral,= intensity.reshape(batch_size, -1), integral.reshape(batch_size, -1)
                                                                               # [batch_size, seq_len * resolution]

        # aggregated timestamp
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        return intensity, integral, timestamp