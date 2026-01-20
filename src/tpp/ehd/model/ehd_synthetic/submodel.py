import torch.nn as nn
import torch


class EHD_backend(nn.Module):
    '''
    This is our implementation of Omi's paper: Fully Neural Network based Model for General Temporal Point Processes
    Hope it can work properly.

    Currently, normalization is disabled.
    Update: 2022-01-19: Now you can use data normalization via synthetic dataloader.

    Following Babylon's paper, we would check the performance of FullyNN with integral offsets.
    '''

    def __init__(self, d_hidden, mlp_layers, device):
        super(EHD_backend, self).__init__()
        self.device = device
        self.mlp_layers = mlp_layers

        self.norm = nn.LayerNorm(2)
        self.input_encoder = nn.Linear(2, d_hidden, bias = True, device = device)
        self.seq_encoder = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_hidden, d_hidden, bias = True, device = device),
                    # nn.LayerNorm(d_hidden)
                )
                for _ in range(mlp_layers)
        ])
        self.activate = nn.GELU()

        # We get two marks: Should we remove it or not.
        self.remove_mark = nn.Linear(d_hidden, 2, device = self.device)
        self.normalize = nn.Softmax(dim = -1)


    def forward(self, input_x, input_y):
        '''
        Args:
            events_history: [batch_size, seq_len]
            time_history:   [batch_size, seq_len]
            time_next:      [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            mask:           [batch_size, seq_len]
        '''

        '''
        Prepare the input.
        '''
        combined_input = self.norm(torch.stack([input_x, input_y], dim = -1))  # [batch_size, seq_len_h, 2]
        combined_input = torch.stack([input_x, input_y], dim = -1)             # [batch_size, seq_len_h, 2]
        output = self.input_encoder(combined_input)                            # [batch_size, seq_len_h, d_hidden]
        for layer in self.seq_encoder:
            output = layer(output)                                             # [batch_size, seq_len_h, d_input]
            output = self.activate(output)                                     # [batch_size, seq_len_h, d_input]

        generated_un_probability_masked = self.remove_mark(output)             # [batch_size, seq_len_h, 2]
        # Here we get the probability p(y = 1) and p(y = 0).
        generated_mask_probability = self.normalize(generated_un_probability_masked)
                                                                               # [batch_size, seq_len_h, 2]
        return generated_mask_probability