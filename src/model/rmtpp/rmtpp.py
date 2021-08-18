import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_events, output_size):
        super(Model, self).__init__()

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
        self.time_scalar = torch.tensor(2.)
        self.base_intensity = torch.tensor(0.2)

    def forward(self, event, time):
        event_vec = self.event_embedding(event.long())
        time_vec = self.time_embedding(time.unsqueeze(-1))
        output, (_, _) = self.rnn(input = time_vec + event_vec)
        # output shape: (batch, seq_length, H_out)
        # We need (batch, seq_length)
        intensity = torch.exp(self.intensity(output).squeeze() + self.time_scalar * time + self.base_intensity)
        integral = (intensity - torch.exp(self.intensity(output).squeeze() + self.base_intensity)) / self.time_scalar
        # For event, we need (batch, seq_length, num_event)
        mark = self.decide(self.marker(output))

        return intensity, integral, mark