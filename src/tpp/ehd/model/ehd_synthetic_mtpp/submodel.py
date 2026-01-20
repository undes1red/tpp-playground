import torch.nn as nn
from src.ehd.model.ehd_synthetic_mtpp.transformers import Transformer
from src.ehd.model.ehd_synthetic_mtpp.embedding import DataEmbedding

class EHD_backend(nn.Module):
    '''
    This is our implementation of Omi's paper: Fully Neural Network based Model for General Temporal Point Processes
    Hope it can work properly.

    Currently, normalization is disabled.
    Update: 2022-01-19: Now you can use data normalization via synthetic dataloader.

    Following Babylon's paper, we would check the performance of FullyNN with integral offsets.
    '''

    def __init__(self, num_events, d_input, d_rnn, d_hidden, n_layers_encoder, \
                 n_head, d_qk, n_layers_decoder, d_v, dropout, device):
        super(EHD_backend, self).__init__()
        self.device = device
        self.num_events = num_events

        self.enc_embedding = DataEmbedding(self.num_events + 1, d_input, dropout = dropout, device = self.device)

        self.seq_encoder = Transformer(d_input = d_input, d_rnn = d_rnn, d_hidden = d_hidden, 
                                       n_head = n_head, d_qk = d_qk, n_layers_decoder = n_layers_decoder, d_v = d_v, \
                                       dropout = dropout, device = self.device)

        # We get two marks: Should we remove it or not.
        self.remove_mark = nn.Linear(d_input, 2, device = self.device)
        self.normalize = nn.Softmax(dim = -1)


    def forward(self, input_time, input_mask, mean, std):
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
        input_embs = self.enc_embedding(input_time, input_mask)                # [batch_size, num_of_patches, d_model]
        
        output = self.seq_encoder(input_embs, input_mask)                      # [batch_size, seq_len, d_input]

        generated_un_probability_masked = self.remove_mark(output)             # [batch_size, seq_len, 2]
        # Here we get the probability p(y = 1) and p(y = 0).
        generated_mask_probability = self.normalize(generated_un_probability_masked)
                                                                               # [batch_size, seq_len, 2]
        return generated_mask_probability