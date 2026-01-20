import torch.nn as nn
import torch

from src.toolbox.metrics import L1_distance_across_events
from src.toolbox.subsequent_mask import get_subsequent_mask

from transformers import AutoConfig

from einops import rearrange, repeat, reduce, pack, unpack
from scipy.stats import spearmanr

from src.tpp.tpp_models.llmtpp_repro.transformers_module import lm_module_location
from src.tpp.tpp_models.llmtpp_repro.embedding import DataEmbedding
from src.tpp.tpp_models.llmtpp_repro.reprogramming_functions import ReprogramInput, ReprogramOutput


class LLMTPP(nn.Module):
    def __init__(self, num_events, llm_class_name, full_llm_name, d_model, d_embedding, \
                 device, repro_input_layer, dropout, number_of_prototype):
        super(LLMTPP, self).__init__()
        self.device = device
        self.num_events = num_events

        # How many layers in the LM are trainable?
        self.d_model = d_model
        self.d_embedding = d_embedding

        self.enc_embedding = DataEmbedding(self.num_events + 1, d_embedding, d_model, dropout = dropout, device = self.device)
        
        # Properties of the used LLM.
        # We load these features before creaing the sequence -> token converter as the dimension of its output must match the LLM.
        self.lm = lm_module_location.get(llm_class_name)
        if self.lm is None:
            raise Exception('Language model not recorded in dict lm_module_location.')
        self.config = AutoConfig.from_pretrained(full_llm_name)
        self.d_lm_embedding = self.config.hidden_size
        
        # The frozen LLM.
        self.retrieved_lm = self.lm.from_pretrained(full_llm_name, output_hidden_states = True, 
                                                    device_map = self.device)
        for param in self.retrieved_lm.parameters():
            param.requires_grad = False
            
        self.word_embeddings = self.retrieved_lm.get_input_embeddings().weight.float()
                                                                               # [vocab_size, d_lm_embedding]
        self.vocab_size = self.word_embeddings.shape[0]
        self.squeezed_token_embedding = nn.Linear(self.vocab_size, number_of_prototype)
        
        # A converter from sequence embedding to LLM's text embedding.
        self.time_seq_to_token_emb = ReprogramInput(n_head = 3, d_lm_embedding = self.d_lm_embedding, \
                                                    d_hidden = 3 * self.d_lm_embedding, \
                                                    repro_input_layer = repro_input_layer, device = self.device)
        
        # A converter from the LLM output to that in the MTPP domain.
        self.reprogram_output = ReprogramOutput(num_events = self.num_events, d_lm_embedding = self.d_lm_embedding, \
                                                d_model = self.d_model, device = self.device)


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
        
        squeezed_token_emb = self.squeezed_token_embedding(self.word_embeddings.T).T.unsqueeze(dim = 0)
                                                                               # [1, squeezed_vocab_size, d_model]
        converted_emb = self.time_seq_to_token_emb(input_embs, squeezed_token_emb)
                                                                               # [batch_size, seq_len, d_lm_model]
        outputs = self.retrieved_lm(inputs_embeds = converted_emb).last_hidden_state
                                                                               # [batch_size, seq_len, d_lm_model]
        
        mark_dist, pred_time = self.reprogram_output(outputs)                  # [batch_size, seq_len, num_events] + [batch_size, seq_len, num_events]
        pred_time = (pred_time - (mean / std)) * std + mean

        return pred_time, mark_dist