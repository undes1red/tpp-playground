import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, pipeline

from src.toolbox.transformer import TransformerLayer

word_list = ['low', 'high', 'increase', 'decrease']

class ReprogramInput(nn.Module):
    '''
    A converter mapping MTPP event embeddings to LLM-understandable embeddings.
    '''
    def __init__(self, n_head, d_lm_embedding, d_hidden, device):
        super(ReprogramInput, self).__init__()
        self.device = device
        self.d_lm_embedding = d_lm_embedding
        
        # Use cross-attention to map event embeddings to token embeddings
        self.attention_to_tokens =  nn.ModuleList([
            TransformerLayer(n_head = n_head, d_input = d_lm_embedding, d_qk = d_lm_embedding, d_v = d_lm_embedding, \
                             device = self.device, d_hidden = d_hidden)
        ])
    

    def forward(self, events_embedding, word_embeddings):
        events_embedding = torch.nn.functional.pad(events_embedding, (0, self.d_lm_embedding - events_embedding.shape[-1]))
                                                                               # [..., batch_size, seq_len, d_lm_model]
        for module_layer in self.attention_to_tokens:
            events_embedding, _ = module_layer(q = events_embedding, k = word_embeddings, v = word_embeddings)
                                                                               # [..., batch_size, seq_len, d_lm_embedding]
        
        return events_embedding


class ReprogramOutput(nn.Module):
    '''
    A converter mapping output from the LLM token domain to the MTPP event domain.
    '''
    def __init__(self, num_events, d_lm_embedding, d_model, device):
        super(ReprogramOutput, self).__init__()
        self.device = device
        
        self.mark = nn.Sequential(
            nn.Linear(d_lm_embedding, d_model, device = self.device),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_events, device = self.device),
            nn.Softmax(dim = -1)
        )

        self.time = nn.Sequential(
            nn.Linear(d_lm_embedding, d_model, device = self.device),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_events, device = self.device),
            nn.Softplus()
        )
    
    
    def forward(self, input):
        mark_pred = self.mark(input)                                           # [..., num_events]
        time_pred = self.time(input)                                           # [..., num_events]
        
        return mark_pred, time_pred