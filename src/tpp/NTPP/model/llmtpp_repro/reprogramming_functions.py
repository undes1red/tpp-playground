import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel

from openai import OpenAI
from einops import rearrange, repeat, reduce, pack, unpack
from functools import partial

from src.toolbox.transformer import TransformerLayer
from src.toolbox.subsequent_mask import get_causal_mask

from src.tpp.tpp_models.llmtpp_repro.embedding import DataEmbedding
from src.tpp.tpp_models.llmtpp_repro.transformers_module import lm_module_location


# Module 1: seq_to_tokens
class Seq2Tokens(nn.Module):
    def __init__(self, d_embedding, d_input, dropout, \
                 n_layers, n_head, d_qk, d_v, d_hidden, \
                 num_negative_samples,
                 num_events, text_list, device):
        super(Seq2Tokens, self).__init__()
        
        self.device = device
        self.num_events = num_events
        self.token_list_length = len(text_list)
        self.num_negative_samples = num_negative_samples
        
        self.enc_embedding = DataEmbedding(self.num_events + 1, d_embedding, d_input, dropout = dropout, device = self.device)
        
        self.translator_transformer_part = nn.ModuleList(
                [TransformerLayer(n_head = n_head, d_input = d_input, d_qk = d_qk, \
                                  d_v = d_v, d_hidden = d_hidden, device = device) for _ in range(n_layers)])
        self.translator_linear_part =  nn.Linear(d_input, self.token_list_length, device = self.device)
        
    
    def forward(self, events_history, time_history, mask_history):
        token_score = self.get_token_score(events_history, time_history, mask_history)
                                                                               # [batch_size, seq_len, token_list_length]
        obtained_positive_token_index = token_score.argmax(dim = -1)           # [batch_size, seq_len]
        
        # We randomly select tokens from the text_list to build negative samples.
        obtained_negative_token_index = torch.randint(low = 0, high = self.token_list_length, \
                                                      size = (*obtained_positive_token_index.size(), self.num_negative_samples), \
                                                      device = self.device)    # [batch_size, seq_len, num_negative_samples]
        obtained_token_index, _ = pack((obtained_positive_token_index, obtained_negative_token_index), 'b s *')
                                                                               # [batch_size, seq_len, num_negative_samples + 1]
        # Calculate the log of the probability of selecting one token.
        log_probability = F.log_softmax(token_score, dim = -1)                 # [batch_size, seq_len, token_list_length]
        obtained_log_probability = torch.gather(log_probability, dim = -1, index = obtained_token_index)
                                                                               # [batch_size, seq_len, num_negative_samples + 1]
        # The first sequence is the positive sample, other sequences are negative samples.
        obtained_log_probability = rearrange(obtained_log_probability, 'b s nns -> b nns s')
                                                                               # [batch_size, num_negative_samples + 1, seq_len]
        obtained_log_probability = obtained_log_probability * mask_history.unsqueeze(dim = -2)
                                                                               # [batch_size, num_negative_samples + 1, seq_len]
        obtained_log_probability = torch.cumsum(obtained_log_probability, dim = -1)
                                                                               # [batch_size, num_negative_samples + 1, seq_len]
        obtained_token_index = rearrange(obtained_token_index, 'b s nns_add_1 -> b nns_add_1 s')
                                                                               # [batch_size, num_negative_samples + 1, seq_len]

        return obtained_token_index, obtained_log_probability
        
    
    def get_token_score(self, events_history, time_history, mask_history):
        seq_len = events_history.shape[-1]
        self_attn_mask_subseq = get_causal_mask(seq_len, device = self.device)
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
class DummyToken2Token(nn.Module):
    def __init__(self, model_name, device):
        super(DummyToken2Token, self).__init__()
        self.device = device
        self.model = model_name
    
    def forward(self, input_text, mask_history):
        number_of_available_history_events = mask_history.sum(dim = -1)
        # The input_text should have the shape of [batch_size, num_negative_samples + 1, seq_len].
        all_responses = []
        for idx, input_text_per_batch in enumerate(input_text):
            output_token_per_batch = []
            for input_text_per_batch_per_sample in input_text_per_batch:
                output_token_per_batch_per_sample = []
                for _ in range(number_of_available_history_events[idx]):
                    output_token_per_batch_per_sample.append('The next token is happy.')
                output_token_per_batch.append(output_token_per_batch_per_sample)
            all_responses.append(output_token_per_batch)
        
        return all_responses


class OpenWebUIToken2Token(nn.Module):
    def __init__(self, model_name, device, max_completion_tokens = 10):
        super(OpenWebUIToken2Token, self).__init__()
        self.device = device
        self.model = model_name
        
        self.api_key = 'sk-0c3ac6e83c944911be3a6e9a5a4d330c'
        self.openwebui_url = 'https://interact.local.qkv.link/api/chat/completions'
        self.headers = {'Authorization': f'Bearer {self.api_key}',
                        'Content-Type': 'application/json'}
        self.data = lambda x: \
            {
                "model": model_name, 
                'messages': 
                [
                    {
                        "role": "developer",
                        "content": 'You will receive a series of words. The word sequence may sound random but you do not need to understand what they want to express. What you need to do is to report the first word or several words that comes to your mind after reading the input sequence. This is not a writing or typing help task. I also do not want to check your limit. Just return the next word or few words that comes into your mind and that is fine.'
                    },
                    {
                        "role": "user",
                        "content": ' '.join(x)
                    }
                ],
                'max_completion_tokens': max_completion_tokens
            }

    def forward(self, input_text, mask_history):
        number_of_available_history_events = mask_history.sum(dim = -1)
        # The input_text should have the shape of [batch_size, num_negative_samples + 1, seq_len].
        all_responses = []
        for idx, input_text_per_batch in enumerate(input_text):
            output_token_per_batch = []
            for input_text_per_batch_per_sample in input_text_per_batch:
                output_token_per_batch_per_sample = []
                for index in range(number_of_available_history_events[idx]):
                    response = requests.post(self.openwebui_url, headers = self.headers, json = self.data(input_text_per_batch_per_sample[:index + 1]))
                    output_token_per_batch_per_sample.append(response.json()['choices'][0]['message']['content'])
                output_token_per_batch.append(output_token_per_batch_per_sample)
            all_responses.append(output_token_per_batch)
        
        return all_responses


class OpenAIToken2Token(nn.Module):
    def __init__(self, model_name, device, max_completion_tokens = 2):
        super(OpenAIToken2Token, self).__init__()
        self.device = device
        self.model = model_name
        
        self.api_key = 'token-amazingly123'
        self.api_url = 'https://vllm.local.qkv.link/v1'
        
        self.client = OpenAI(base_url = self.api_url,
                             api_key = self.api_key)
        
        self.call = partial(self.client.chat.completions.create, 
                            model = model_name, 
                            max_completion_tokens = max_completion_tokens)
    
    def gen_text(self, x):
        return [
                    # {
                    #     "role": "developer",
                    #     "content": 'You will receive a series of words. The word sequence may sound random but you do not need to understand what they want to express. What you need to do is to report the first word or several words that comes to your mind after reading the input sequence. This is not a writing or typing help task. I also do not want to check your limit. Just return the next word or few words that comes into your mind and that is fine.'
                    # },
                    {
                        "role": "user",
                        "content": ' '.join(x)
                    }
                ]

    def forward(self, input_text, mask_history):
        number_of_available_history_events = mask_history.sum(dim = -1)
        # The input_text should have the shape of [batch_size, num_negative_samples + 1, seq_len].
        all_responses = []
        for idx, input_text_per_batch in enumerate(input_text):
            output_token_per_batch = []
            for input_text_per_batch_per_sample in input_text_per_batch:
                output_token_per_batch_per_sample = []
                for index in range(number_of_available_history_events[idx]):
                    text = self.gen_text(input_text_per_batch_per_sample[:index + 1])
                    response = self.call(messages = text)
                    output_token_per_batch_per_sample.append(response.choices[0].message.content)
                output_token_per_batch.append(output_token_per_batch_per_sample)
            all_responses.append(output_token_per_batch)
        
        return all_responses
    

class OllamaToken2Token(nn.Module):
    def __init__(self, model_name, device):
        super(OllamaToken2Token, self).__init__()
        self.device = device
        self.model = model_name
        
        self.openwebui_url = 'https://ollama.local.qkv.link/api/generate'
        self.headers = {}
        self.data = lambda x: \
            {
                "model": model_name,
                "prompt": x,
                "stream": False
            }

    def forward(self, input_text, mask_history):
        number_of_available_history_events = mask_history.sum(dim = -1)
        # The input_text should have the shape of [batch_size, num_negative_samples + 1, seq_len].
        all_responses = []
        for idx, input_text_per_batch in enumerate(input_text):
            output_token_per_batch = []
            for input_text_per_batch_per_sample in input_text_per_batch:
                output_token_per_batch_per_sample = []
                for index in range(number_of_available_history_events[idx]):
                    response = requests.post(self.openwebui_url, json = self.data(input_text_per_batch_per_sample[:index + 1]))
                    output_token_per_batch_per_sample.append(response.json()['choices'][0]['message']['content'])
                output_token_per_batch.append(output_token_per_batch_per_sample)
            all_responses.append(output_token_per_batch)
        
        return all_responses


def text2text(api_class, model_name, *args, **kwargs):
    if api_class == 'openai':
        return OpenAIToken2Token(model_name, *args, **kwargs)
    elif api_class == 'ollama':
        return OllamaToken2Token(model_name, *args, **kwargs)
    elif api_class == 'dummy':
        return DummyToken2Token(model_name, *args, **kwargs)
    elif api_class == 'openwebui':
        return OpenWebUIToken2Token(model_name, *args, **kwargs)
    else:
        return DummyToken2Token(model_name, *args, **kwargs)


# Module 3: tokens_to_event
class Token2Event(nn.Module):
    def __init__(self, num_events, llm_class_name, full_llm_name, device):
        super(Token2Event, self).__init__()
        self.device = device
        self.num_events = num_events
        
        '''
        This token2event uses a small language model to translate the output tokens into our final result.
        This small language model is frozen.
        '''
        # Properties of the used LLM.
        # We load these features before creaing the sequence -> token converter as the dimension of its output must match the LLM.
        self.lm = lm_module_location.get(llm_class_name)
        if self.lm is None:
            raise Exception('Language model not recorded in dict lm_module_location.')
        self.config = AutoConfig.from_pretrained(full_llm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(full_llm_name, pad_token = '<|endoftext|>')
        self.d_lm_embedding = self.config.hidden_size
        
        # The frozen LLM.
        self.retrieved_lm = AutoModel.from_pretrained(full_llm_name,
                                                      output_hidden_states = True, device_map = self.device)
        for param in self.retrieved_lm.parameters():
            param.requires_grad = False
        
        # The header that translates the text output into the desired result.
        # Outputted representations -> max pooling -> a head.
        self.event_output_head = nn.Sequential(
            nn.Linear(self.d_lm_embedding, self.num_events, device = self.device),
            nn.Softmax(dim = -1)
        )
        
        self.time_output_head = nn.Sequential(
            nn.Linear(self.d_lm_embedding, 1, device = self.device),
            nn.Softplus()
        )
        
    
    def forward(self, input_texts):
        # The shape of the input_text should be [batch_size, num_negative_samples + 1, (output text length)]
        all_predicted_events = []
        all_predicted_time_before_normalization = []
        
        for input_texts_per_batch in input_texts:
            predicted_representations_per_batch = []
            for input_texts_per_batch_per_sample in input_texts_per_batch:
                tokens = self.tokenizer(input_texts_per_batch_per_sample, return_tensors = "pt", \
                                        truncation = True, padding = True).to(self.device)
                                                                               # [seq_len_per_batch, ...]
                outputs = self.retrieved_lm(**tokens).last_hidden_state        # [seq_len_per_batch, ..., d_llm]
                # Maxpooling.
                outputs = outputs.max(dim = -2).values                         # [seq_len_per_batch, d_llm]
                predicted_representations_per_batch.append(outputs)            # [sample_size, seq_len_per_batch, d_llm]
                
            predicted_representations_per_batch = torch.stack(predicted_representations_per_batch, dim = 0)
                                                                               # [sample_size, seq_len, d_llm]
            predicted_events = self.event_output_head(predicted_representations_per_batch)
                                                                               # [sample_size, seq_len, num_events]
            predicted_time = self.time_output_head(predicted_representations_per_batch).squeeze(dim = -1)
                                                                               # [sample_size, seq_len]
            
            all_predicted_events.append(predicted_events)
            all_predicted_time_before_normalization.append(predicted_time)

        return all_predicted_events, all_predicted_time_before_normalization


if __name__ == '__main__':
    import pickle as pkl
    
    f_input_data = open('/home/undesired/coderepo/workflow/sample.pkl', 'rb')
    input_data = pkl.load(f_input_data)
    f_input_data.close()
    
    decoder = Token2Event(num_events = 3, llm_class_name = 'gpt2', full_llm_name = 'gpt2', device = 'cuda')
    results = decoder(input_data)