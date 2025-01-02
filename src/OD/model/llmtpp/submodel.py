import torch.nn as nn
import torch

from src.toolbox.metrics import L1_distance_across_events
from src.toolbox.subsequent_mask import get_subsequent_mask

from transformers import AutoConfig
from einops import rearrange, repeat, reduce, pack, unpack
from scipy.stats import spearmanr

from src.OD.model.llmtpp.transformers_module import lm_module_location
from src.OD.model.llmtpp.embedding import DataEmbedding


class LLMTPP(nn.Module):
    def __init__(self, num_events, llm_class_name, full_llm_name, d_model, \
                 d_embedding, lm_layers, device, dropout):
        super(LLMTPP, self).__init__()
        self.device = device
        self.num_events = num_events

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

        self.missing_score = nn.Sequential(
            nn.Linear(self.d_lm_embedding, d_model, device = self.device),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1, device = self.device)
        )


    def forward(self, mode, *args, **kwargs):
        task_mapper = {
            'train': self.model_forward,
            'evaluate': self.model_forward
        }

        return task_mapper[mode](*args, **kwargs)
    

    def model_forward(self, input_time, input_events, input_seq_mask, mean, std):
        input_time = (input_time - mean) / std                                 # [batch_size, sample_size, seq_len, d_model]

        input_embs = self.enc_embedding(input_events, input_time, input_seq_mask)
                                                                               # [batch_size, sample_size, seq_len, d_model]
        
        input_embs = torch.nn.functional.pad(input_embs, (0, self.d_lm_embedding - input_embs.shape[-1]))
                                                                               # [batch_size, sample_size, seq_len, d_lm_model]
        
        batch_size = input_embs.shape[0]
        # The LM does not like the input having four dimensions. Here we reshape the input tensor to make the LM happy.
        input_embs = rearrange(input_embs, 'b s ... -> (b s) ...')             # [batch_size * sample_size, seq_len, d_lm_model]
        outputs = self.retrieved_lm(inputs_embeds = input_embs).last_hidden_state
                                                                               # [batch_size * sample_size, seq_len, d_lm_model]
        outputs = rearrange(outputs, '(b s) ... -> b s ...', b = batch_size)   # [batch_size, sample_size, seq_len, d_lm_model]
        mark_dist = self.missing_score(outputs)                                # [batch_size, sample_size, seq_len, 1]
        missing_score = torch.nn.functional.sigmoid(mark_dist).squeeze(dim = -1)
                                                                               # [batch_size, sample_size, num_events]

        return missing_score