import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
import numpy as np

from src.toolbox.transformer import TransformerLayer
from src.toolbox.position_embedding import PositionalEmbedding
from src.toolbox.subsequent_mask import get_causal_mask


class NotePredictor(nn.Module):
    def __init__(self, n_layer, n_head, d_input, d_qk, d_v, device, d_hidden, dropout = 0.1):
        super(NotePredictor, self).__init__()
        self.device = device

        self.note_transformer = nn.ModuleList(
            [TransformerLayer(n_head = n_head, d_input = d_input, d_qk = d_qk, d_v = d_v, \
                              d_hidden = d_hidden, dropout = 0.1, device = self.device) for _ in range(n_layer)]
        )

        # position vector, used for temporal encoding
        self.position_emb = PositionalEmbedding(d_input, max_len = 4096, device = self.device)

        self.predict_head = nn.Linear(d_input, d_input, device = self.device)
    

    def forward(self, input_tensor, non_pad_mask):
        # prepare attention masks
        # self_attn_mask is where we cannot look, i.e., the future and the padding
        seq_len = input_tensor.shape[-2]
        self_attn_mask_subseq = get_causal_mask(seq_len, device = self.device)
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask_keypad = rearrange(non_pad_mask, 'b s -> b () s')       # [batch_size, seq_len, seq_len]
        self_attn_mask = self_attn_mask_keypad & self_attn_mask_subseq         # [batch_size, seq_len, seq_len]
        
        position_emb = self.position_emb(input_tensor, length_idx = -2)        # [batch_size, seq_len, d_input]
        
        output = input_tensor + position_emb                                   # [batch_size, seq_len, d_input]
        for layer in self.note_transformer:
            output, _ = layer(output, self_attn_mask = self_attn_mask, non_pad_mask = non_pad_mask)
                                                                               # [batch_size, seq_len, d_input]
        output = self.predict_head(output)                                     # [batch_size, seq_len, d_input]

        return output