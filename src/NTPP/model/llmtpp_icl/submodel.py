import torch.nn as nn
import torch

from transformers import AutoConfig
from einops import rearrange, repeat, reduce, pack, unpack

from src.toolbox.metrics import L1_distance_across_events
from src.toolbox.subsequent_mask import get_subsequent_mask
from src.toolbox.functional.kl_divergence import kl_divergence
from src.toolbox.misc import list_to_string

from src.TPP.model.llmtpp_icl.tool_functions import SeqRetriever, Text2Text


# This module accepts the batched input.
# However, both module in LLMTPP can only handle .
# They are expected to be very slow.
class LLMTPP(nn.Module):
    def __init__(self, device, num_events, t_0, T, \
                 dataset, database, top_k, \
                 encode_llm_model, embedding_model, ollama_url, 
                 continuation_model_name):
        super(LLMTPP, self).__init__()
        self.device = device
        self.num_events = num_events
        self.start_time = t_0
        self.summary_length = 200
        
        # Part 1: sequence retriever. Here we load the vector database for sequence retrieval.
        # It will return several sequence for building the prompt.
        self.seq_retrieve = SeqRetriever(num_events = num_events, t_0 = t_0, T = T, \
                                         dataset = dataset, database = database, top_k = top_k, \
                                         encode_llm_model = encode_llm_model, embedding_model = embedding_model, \
                                         summary_length = self.summary_length, ollama_url = ollama_url, device = device)
        
        # Part 2: Blackbox LLM.
        # Here the black-box LLM will take in the prompt and return the result.
        # The result should be in a specific format so our code can parse it.
        # We will regenerate the output if it is in the wrong format until data in true format is returned.
        self.text_to_text = Text2Text(num_events = num_events, dataset = dataset, \
                                      continuation_model_name = continuation_model_name, device = device, ollama_url = ollama_url)
 

    def forward(self, mode, *args, **kwargs):
        task_mapper = {
            'train': self.model_forward,
            'evaluate': self.model_forward
        }

        return task_mapper[mode](*args, **kwargs)
    
    
    def model_forward(self, events_history, time_history, mask_history, \
                            events_next, time_next, mask_next, \
                            mean, std):
        
        history_seq_len = mask_next.cumsum(dim = -1)                           # [batch_size, seq_len]
        packed_data = zip(events_history, time_history, events_next, time_next, mask_next, history_seq_len)
        
        real_events = []
        real_time = []
        
        predicted_events = []
        predicted_time = []
        
        for events_history_per_batch, time_history_per_batch, events_next_per_batch, time_next_per_batch, mask_next_per_seq, history_seq_len_per_batch \
            in packed_data:
            
            real_events_history_per_batch = []
            real_events_per_batch = []
            
            real_time_history_per_batch = []
            real_time_per_batch = []
            
            input_filled_into_template = []
            
            predicted_events_per_batch = []
            predicted_time_per_batch = []
            
            for history_length in history_seq_len_per_batch:
                selected_events_history = events_history_per_batch[:history_length]
                selected_time_history = time_history_per_batch[:history_length]
                selected_events_next = events_next_per_batch[history_length - 1]
                selected_time_next = time_next_per_batch[history_length - 1]
                
                real_events_history_per_batch.append(selected_events_history.tolist())
                real_events_per_batch.append(selected_events_next.tolist())
                real_time_history_per_batch.append(selected_time_history.tolist())
                real_time_per_batch.append(selected_time_next.tolist())
                
                # transfer inputted events history from absolute time to relative timestamp.
                relative_time_history = torch.diff(selected_time_history, prepend = torch.tensor([self.start_time,], device = self.device))
        
                string_time_history = list_to_string(relative_time_history.tolist())
                string_events_history = list_to_string(selected_events_history.tolist())
                input_filled_into_template.append({'time_seq': string_time_history, 'mark_seq': string_events_history, 'summary_length': self.summary_length})
            
            selected_mark_seqs, selected_time_seqs = self.seq_retrieve(input_filled_into_template)
                                                                               # [seq_len, topk, ...]

            # The format of result: [{'event': #predicted_mark, 'time': #predicted_time},]
            results = self.text_to_text(selected_mark_seqs, selected_time_seqs, real_events_history_per_batch, real_time_history_per_batch)

            real_events.append(real_events_per_batch)
            real_time.append(real_time_per_batch)
            
            for result in results:
                predicted_events_per_batch.append(result['event'])
                predicted_time_per_batch.append(result['time'])
                
            predicted_events.append(predicted_events_per_batch)
            predicted_time.append(predicted_time_per_batch)
        
        return real_events, real_time, predicted_events, predicted_time