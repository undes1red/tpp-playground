import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel

from einops import rearrange

from src.toolbox.transformer import TransformerLayer
from src.toolbox.subsequent_mask import get_subsequent_mask

# from src.toolbox.llms import OllamaToken2Token
from src.toolbox.llms.huggingface import LangChainEmbedding, LangChainToken2Token
from src.toolbox.misc import list_to_string

# from src.TPP.model.llmtpp_icl.description_gen_prompt import get_prompt as get_prompt_desc
from src.TPP.model.llmtpp_icl.description_gen_prompt import prompt_dict as desc_prompt_dict
from src.TPP.model.llmtpp_icl.event_predict_prompt import prompt_dict as pred_prompt_dict

# Module 1: seq_to_tokens
class SeqRetriever(nn.Module):
    def __init__(self, device, num_events, t_0, T, dataset, database, top_k, encode_llm_model, embedding_model, 
                 summary_length, ollama_url):
        super(SeqRetriever, self).__init__()
        self.device = device
        self.dataset = dataset
        self.num_events = num_events
        self.database = self.reform(database)
        self.top_k = top_k
        self.encode_llm_model = encode_llm_model
        self.embedding_model = embedding_model
        self.start_time = t_0
        # self.end_time not used.
        self.end_time = T
        
        # self.ollama_client = OllamaToken2Token(ollama_url, device = self.device)
        self.sequence_expresser = LangChainToken2Token(encode_llm_model, device = self.device, 
                                                       prompt_template = desc_prompt_dict[self.dataset],
                                                       batch_size = 16, 
                                                       pipeline_kwargs = {'max_new_tokens': summary_length})
        self.embedder = LangChainEmbedding(embedding_model, device = self.device)


    def reform(self, database):
        self.representations = torch.tensor(database['representation'], device = self.device)
                                                                               # [dataset_size, vector_size]
        self.time_seq = database['time_seq']
        self.event = database['event']


    def forward(self, input_filled_into_template):
        # prompt_desc = get_prompt_desc(self.dataset, time_seq = string_time_history, mark_seq = string_events_history)
        descriptions = self.sequence_expresser(input_filled_into_template)     # [seq_len, ...]
        
        descriptions_embedding = []
        for description in descriptions:
            descriptions_embedding.append(torch.tensor(self.embedder(description), device = self.device))
                                                                               # [embedding_size]
        descriptions_embedding = torch.stack(descriptions_embedding, dim = 0)  # [seq_len, ...]
        similarity = nn.functional.cosine_similarity(x1 = descriptions_embedding.unsqueeze(dim = -2), \
                                                     x2 = self.representations, dim = -1)
                                                                               # [seq_len, dataset_size]
        _, topk_indexes = torch.topk(similarity, self.top_k, dim = -1)         # [seq_len, top_k] * 2
        
        selected_time_seqs = []
        selected_mark_seqs = []
        for topk_index in topk_indexes:
            selected_time_seqs_per_batch = []
            selected_mark_seqs_per_batch = []
            for topk_index_per_seq in topk_index:
                selected_time_seqs_per_batch.append(self.time_seq[topk_index_per_seq])
                selected_mark_seqs_per_batch.append(self.event[topk_index_per_seq])
            selected_time_seqs.append(selected_time_seqs_per_batch)
            selected_mark_seqs.append(selected_mark_seqs_per_batch)
        
        return selected_mark_seqs, selected_time_seqs
        
    
    def get_token_score(self, events_history, time_history, mask_history):
        seq_len = events_history.shape[-1]
        self_attn_mask_subseq = get_subsequent_mask(seq_len, device = self.device)
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask_keypad = rearrange(mask_history, 'b s -> b () s')       # [batch_size, seq_len, seq_len]
        self_attn_mask = self_attn_mask_keypad & self_attn_mask_subseq         # [batch_size, seq_len, seq_len]
        
        representation = self.enc_embedding(events_history, time_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        for layer in self.translator_transformer_part:
            representation, _ = layer(representation, self_attn_mask = self_attn_mask, non_pad_mask = mask_history)
                                                                               # [batch_size, seq_len, d_input]
        score = self.translator_linear_part(representation)                    # [batch_size, seq_len, token_list_length]
        
        return score


# Module 2: tokens_to_tokens
class Text2Text(nn.Module):
    def __init__(self, num_events, dataset, continuation_model_name, device, ollama_url):
        super(Text2Text, self).__init__()
        self.continuation_model_name = continuation_model_name
        self.dataset = dataset
        self.device = device
        self.num_events = num_events
        self.batch_size = 4
        
        self.t2t = LangChainToken2Token(self.continuation_model_name, device = self.device, 
                                        prompt_template = pred_prompt_dict[self.dataset], 
                                        batch_size = self.batch_size,
                                        pipeline_kwargs = {'max_new_tokens': 50},
                                        model_kwargs = {'temperature': 0.6},
                                        token_kwargs = {'model_max_length': 8192, 'truncation': True})


    def forward(self, selected_mark_seqs, selected_time_seqs, mark_seq_question, time_seq_question):
        input_filled_into_template = []
        for selected_mark_seqs_per_seq, selected_time_seqs_per_seq, mark_seq_question_per_seq, time_seq_question_per_seq in \
            zip(selected_mark_seqs, selected_time_seqs, mark_seq_question, time_seq_question):
            
            string_sequences = []
            string_sequences.append(list_to_string(*selected_mark_seqs_per_seq))
            string_sequences.append(list_to_string(*selected_time_seqs_per_seq))
            string_sequences.extend(list_to_string(time_seq_question_per_seq, mark_seq_question_per_seq))
                
            input_filled_into_template.append(
                {
                    'reference_seqs': "\n".join([f"Available sequence {idx + 1}:\n  Time: {time_seq}\n  Mark: {mark_seq}\n" for idx, (time_seq, mark_seq) in enumerate(zip(string_sequences[0], string_sequences[1]))]),
                    'time_seq_question': string_sequences[2],
                    'mark_seq_question': string_sequences[3]
                }
            )
        
        results = []
        for i in range(0, len(input_filled_into_template), self.batch_size):
            while True:
                try:
                    responses = self.t2t(input_filled_into_template[i:i+self.batch_size])
                                                                                   # [batch_size]
                    for response in responses:
                        time_string = re.search(r'Time of the next event:\s*\d*\.\d*', response).group()
                        time = float(time_string.split(':')[1])
                        mark_string = re.search(r'Mark of the next event:\s*\d*', response).group()
                        mark = int(mark_string.split(':')[1])
                        
                        if mark >= self.num_events:
                            mark = 0
                        
                        results.append({'time': time, 'event': mark})
                    break
                except Exception as e:
                    print('parse failed! Retrying....')
                    continue
        
        return results