import torch.nn as nn
import torch

from src.toolbox.metrics import L1_distance_across_events
from src.toolbox.subsequent_mask import get_subsequent_mask

from transformers import AutoConfig
from einops import rearrange, repeat, reduce, pack, unpack
from scipy.stats import spearmanr

from src.LH.model.llmtpp.transformers_module import lm_module_location
from src.LH.model.llmtpp.embedding import DataEmbedding
from src.LH.model.llmtpp.rnn import RNN_layers


class LLMTPP(nn.Module):
    def __init__(self, num_events, llm_class_name, full_llm_name, d_model, \
                 d_embedding, lm_layers, lh_length, dropout, device):
        super(LLMTPP, self).__init__()
        self.device = device
        self.num_events = num_events
        self.lh_length = lh_length

        # How many layers in the LM are trainable?
        self.lm_layers = lm_layers
        self.d_model = d_model
        self.d_embedding = d_embedding

        self.enc_embedding = DataEmbedding(self.num_events + 1, d_embedding, d_model, dropout = dropout, device = self.device)

        self.lm = lm_module_location.get(llm_class_name)
        if self.lm is None:
            raise Exception('Language model not recorded in dict lm_module_location.')
        self.config = AutoConfig.from_pretrained(full_llm_name)
        self.d_lm_embedding = self.config.n_embd
        self.retrieved_lm = self.lm.from_pretrained(full_llm_name, output_attentions = True, \
                                                    output_hidden_states = True, device_map = self.device)
        self.retrieved_lm.h = self.retrieved_lm.h[:self.lm_layers]
        
        # We only train the parameters in FFN and LayerNorm
        for _, (name, param) in enumerate(self.retrieved_lm.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.rnn = RNN_layers(d_model = self.d_lm_embedding, d_rnn = self.d_lm_embedding, device = self.device)

        self.mark = nn.Sequential(
            nn.Linear(self.d_lm_embedding, d_model, device = self.device),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.num_events * self.lh_length, device = self.device)
        )

        self.time = nn.Sequential(
            nn.Linear(self.d_lm_embedding, d_model, device = self.device),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.num_events * self.lh_length, device = self.device),
            nn.Softplus()
        )


    def forward(self, mode, *args, **kwargs):
        task_mapper = {
            'train': self.model_forward,
            'evaluate': self.model_forward
        }

        return task_mapper[mode](*args, **kwargs)
    

    def model_forward(self, events_history, time_history, mask_history, mean, std):
        time_history = (time_history - mean) / std

        input_embs = self.enc_embedding(events_history, time_history, mask_history)
                                                                               # [batch_size, seq_len, d_model]
        
        input_embs = torch.nn.functional.pad(input_embs, (0, self.d_lm_embedding - input_embs.shape[-1]))
                                                                               # [batch_size, seq_len, d_lm_model]
        outputs = self.retrieved_lm(inputs_embeds = input_embs).last_hidden_state
                                                                               # [batch_size, seq_len, d_lm_model]
        seq_representation = self.rnn(outputs)                                 # [batch_size, d_lm_model]
        
        mark_dist = self.mark(seq_representation)                              # [batch_size, num_events * lh_length]
        mark_dist = rearrange(mark_dist, '... (ne lhl) -> ... lhl ne', lhl = self.lh_length)
                                                                               # [batch_size, lh_length, num_events]
        mark_dist = torch.nn.functional.softmax(mark_dist, dim = -1)           # [batch_size, lh_length, num_events]
        
        pred_time = self.time(seq_representation)                              # [batch_size, num_events * lh_length]
        pred_time = rearrange(pred_time, '... (ne lhl) -> ... lhl ne', lhl = self.lh_length)
                                                                               # [batch_size, lh_length, num_events]
        pred_time = (pred_time - (mean / std)) * std + mean                    # [batch_size, lh_length, num_events]

        return pred_time, mark_dist