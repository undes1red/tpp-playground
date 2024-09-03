import torch.nn as nn
import torch
import numpy as np
import transformers
import re
import math

from src.toolbox.metrics import L1_distance_across_events
from src.toolbox.subsequent_mask import get_subsequent_mask

from einops import rearrange, repeat, reduce, pack, unpack
from scipy.stats import spearmanr
from src.TPP.model.llmtpp_patch_pred.transformers_module import lm_module_location
from src.TPP.model.llmtpp_patch_pred.embedding import DataEmbedding


class LLMTPP(nn.Module):
    def __init__(self, llm_class_name, full_llm_name, patch_size, d_model, \
                 d_embedding, num_events, lm_layers, d_lm_embedding, device, dropout):
        super(LLMTPP, self).__init__()
        self.device = device

        self.num_events = num_events
        self.patch_size = patch_size
        # How many layers in the LM are trainable?
        self.lm_layers = lm_layers
        self.d_model = d_model
        self.d_embedding = d_embedding
        self.d_lm_embedding = d_lm_embedding

        self.enc_embedding = DataEmbedding(self.num_events + 1, d_embedding, d_model, self.patch_size, dropout = dropout, device = self.device)

        self.lm = lm_module_location.get(llm_class_name)
        if self.lm is None:
            raise Exception('Language model not recorded in dict lm_module_location.')
        
        self.retrieved_lm = self.lm.from_pretrained(full_llm_name, output_attentions = True, \
                                                    output_hidden_states = True, device_map = self.device)
        self.retrieved_lm.h = self.retrieved_lm.h[:self.lm_layers]
        
        # We only train the parameters in FFN and LayerNorm
        for _, (name, param) in enumerate(self.retrieved_lm.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.mark = nn.Sequential(
            nn.Linear(d_lm_embedding, d_model, device = self.device),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.patch_size * self.num_events, device = self.device)
        )

        self.time = nn.Sequential(
            nn.Linear(d_lm_embedding, d_model, device = self.device),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.patch_size * self.num_events, device = self.device),
            nn.Softplus()
        )


    def patchify(self, events_history, time_history, mask_history):
        # Split the input sequence into several patches.
        seq_len = events_history.shape[-1]
        free_events_size = seq_len % self.patch_size
        num_of_patches = int(seq_len / self.patch_size) + (1 if free_events_size > 0 else 0)
        p1d = (0, (self.patch_size - free_events_size) % self.patch_size)

        events_history = torch.nn.functional.pad(events_history, p1d, 'constant', 0)
                                                                               # [batch_size, num_of_patches * self.patch_size]
        time_history = torch.nn.functional.pad(time_history, p1d, 'constant', 0)
                                                                               # [batch_size, num_of_patches * self.patch_size]
        mask_history = torch.nn.functional.pad(mask_history, p1d, 'constant', 0)
                                                                               # [batch_size, num_of_patches * self.patch_size]
        
        events_history = rearrange(events_history, 'b (np ps) -> b np ps', np = num_of_patches)
                                                                               # [batch_size, num_of_patches, self.patch_size]
        time_history = rearrange(time_history, 'b (np ps) -> b np ps', np = num_of_patches)
                                                                               # [batch_size, num_of_patches, self.patch_size]
        mask_history = rearrange(mask_history, 'b (np ps) -> b np ps', np = num_of_patches)
                                                                               # [batch_size, num_of_patches, self.patch_size]
        
        return events_history, time_history, mask_history


    def forward(self, mode, *args, **kwargs):
        task_mapper = {
            'train': self.model_forward,
            'evaluate': self.model_forward
        }

        return task_mapper[mode](*args, **kwargs)
    

    def model_forward(self, events_history, time_history, mask_history, mean, std):
        time_history = (time_history - mean) / std

        patched_events_history, patched_time_history, patched_mask_history = self.patchify(events_history, time_history, mask_history)
                                                                               # [batch_size, num_of_patches, patch_size]
        input_embs = self.enc_embedding(patched_events_history, patched_time_history, patched_mask_history)
                                                                               # [batch_size, num_of_patches, d_model]

        input_embs = torch.nn.functional.pad(input_embs, (0, self.d_lm_embedding - input_embs.shape[-1]))
                                                                               # [batch_size, num_of_patches, d_lm_model]
        outputs = self.retrieved_lm(inputs_embeds = input_embs).last_hidden_state
                                                                               # [batch_size, num_of_patches, d_lm_model]
        mark_dist = self.mark(outputs)                                         # [batch_size, seq_len, patch_size * num_events]
        mark_dist = rearrange(mark_dist, '... (ps ne) -> ... ps ne', ps = self.patch_size)
                                                                               # [batch_size, seq_len, patch_size, num_events]
        mark_dist = torch.nn.functional.softmax(mark_dist, dim = -1)           # [batch_size, seq_len, patch_size, num_events]
        pred_time = self.time(outputs)                                         # [batch_size, seq_len, patch_size * num_events]
        pred_time = rearrange(pred_time, '... (ps ne) -> ... ps ne', ps = self.patch_size)
                                                                               # [batch_size, seq_len, patch_size, num_events]
        pred_time = (pred_time - (mean / std)) * std + mean

        return pred_time, mark_dist