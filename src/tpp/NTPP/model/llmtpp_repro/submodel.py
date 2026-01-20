import torch.nn as nn
import torch

from transformers import AutoConfig
from einops import rearrange, repeat, reduce, pack, unpack

from src.toolbox.metrics import L1_distance_across_events
from src.toolbox.subsequent_mask import get_subsequent_mask
from src.toolbox.functional.kl_divergence import kl_divergence

from src.tpp.tpp_models.llmtpp_repro.reprogramming_functions import Seq2Tokens, text2text, Token2Event

# The token list.
from src.tpp.tpp_models.llmtpp_repro.text_list import text_list


class LLMTPP(nn.Module):
    def __init__(self, device, num_events, d_embedding, d_input, dropout, \
                 n_layers, n_head, d_qk, d_v, d_hidden, num_negative_samples, \
                 api_class, blackbox_llm_name, \
                 decode_llm_class_name, decode_full_llm_name):
        super(LLMTPP, self).__init__()
        self.device = device
        self.num_events = num_events
        
        # Part 1: sequence to texts
        self.seq_to_texts = Seq2Tokens(d_embedding, d_input, dropout, \
                                       n_layers, n_head, d_qk, d_v, d_hidden, \
                                       num_negative_samples,
                                       num_events, text_list, device)
        
        # Part 2: Blackbox LLM.
        # We will test our model with openwebui.
        # Then use other proprietary models.
        self.text_to_text = text2text(api_class, blackbox_llm_name, device = self.device)
        
        # Part 3: texts to result.
        # Here we may fine-tune a small LLM, such as a normal gpt2 model.
        self.token_to_event = Token2Event(num_events, decode_llm_class_name, decode_full_llm_name, device = self.device)
        
        # Loss function for calculating posterior.
        self.lambda_t = 1.0
        self.lambda_m = 1.0
        

    def forward(self, mode, *args, **kwargs):
        task_mapper = {
            'train': self.model_forward,
            'evaluate': self.model_forward
        }

        return task_mapper[mode](*args, **kwargs)
    
    
    def model_forward(self, events_history, time_history, mask_history, \
                            events_next, time_next, mask_next, \
                            mean, std):
        
        all_predicted_time, all_predicted_events, all_p_posterior_prior_with_pad = \
            self.get_time_and_mark_prediction(events_history, time_history, mask_history, mean, std)
                                                                               # batch_size * [sample_size, seq_len] + batch_size * [sample_size, seq_len, num_events] + batch_size * [sample_size, seq_len]
        all_p_posterior, time_and_mark_prediction_loss = \
            self.obtain_posterior_of_t2(all_predicted_time, all_predicted_events, events_next, time_next, mask_next)
                                                                               # batch_size * [sample_size, seq_len]
        # Loss 1: the prior should match the posterior
        all_p_prior = []
        for idx, all_p_posterior_prior_with_pad_per_batch in enumerate(all_p_posterior_prior_with_pad):
            all_p_prior.append(all_p_posterior_prior_with_pad_per_batch[:, :mask_next[idx].sum()])
        kl_div = self.get_kl_div_between_tk1_and_tk2(all_p_prior, all_p_posterior)
        
        # Loss 2: self.token_to_event should translate the obtained token to the next event.
        training_loss = 0
        for time_and_mark_prediction_loss_per_batch in time_and_mark_prediction_loss:
            training_loss += time_and_mark_prediction_loss_per_batch[0].sum()
        
        return kl_div, training_loss
        
    
    def get_kl_div_between_tk1_and_tk2(self, distribution_of_tk1, distribution_of_tk2):
        # Caution. The KL divergence between tk1 and tk2 is for training the Seq2Tokens module, so the connection 
        # between distribution_of_tk2 and self.token_to_event should be removed.
        kl_div_sum = 0
        for (distribution_of_tk1_per_batch, distribution_of_tk2_per_batch) in zip(distribution_of_tk1, distribution_of_tk2):
            distribution_of_tk2_per_batch = distribution_of_tk2_per_batch.detach()
            kl_div = kl_divergence(distribution_of_tk2_per_batch, distribution_of_tk1_per_batch, dim = -2, loss = True)
                                                                               # [seq_len]
            kl_div_sum = kl_div.sum()
        
        return kl_div_sum


    def get_time_and_mark_prediction(self, events_history, time_history, mask_history, mean, std):
        time_history = (time_history - mean) / std
        
        # Obtain positive and negative samples.
        # [..., 0] is the positive sample, while [..., 1:] are negative samples.
        obtained_token_index, obtained_log_probability \
            = self.seq_to_texts(events_history, time_history, mask_history)    # [batch_size, num_negative_samples + 1, seq_len], [batch_size, num_negative_samples + 1]
        
        generated_texts = []
        # Generate the text list based on obtained token index.
        for obtained_token_index_per_batch in obtained_token_index:
            generated_texts_per_batch = []
            for obtained_token_index_per_seq_per_batch in obtained_token_index_per_batch:
                generated_text = []
                for index in obtained_token_index_per_seq_per_batch:
                    generated_text.append(text_list[index])
                generated_texts_per_batch.append(generated_text)
            generated_texts.append(generated_texts_per_batch)
                                                                               # [batch_size, num_negative_samples + 1, seq_len]
        
        output_tokens = self.text_to_text(generated_texts, mask_history)       # [batch_size, num_negative_samples + 1, ...]
        all_predicted_events, all_predicted_time_before_normalization = self.token_to_event(output_tokens)
                                                                               # batch_size * [sample_size, seq_len, num_events] + batch_size * [sample_size, seq_len]
        all_predicted_time = []
        for item in all_predicted_time_before_normalization:
            all_predicted_time.append((item - mean / std)  * std + mean)       # batch_size * [sample_size, seq_len]

        return all_predicted_time, all_predicted_events, obtained_log_probability


    def loss_func(self, all_predicted_time, all_predicted_events, events_next, time_next, mask_next):
        # The shape of all_predicted_time should be batch_size * [sample_size, seq_len]
        # The shape of all_predicted_events should be batch_size * [samples_size, seq_len, num_events]
        # The shape of events_next should be [batch_size, padded_seq_len]
        # The shape of time_next should be [batch_size, padded_seq_len]. time_next should be normalized.
        # The shape of mask_next should be [batch_size, padded_seq_len].
        # time loss.
        batch_size = len(all_predicted_time)
        sample_num_positive_negative = all_predicted_time[0].shape[0]
        
        loss_val = []
        for batch_index in range(batch_size):
            picked_predicted_time = all_predicted_time[batch_index]            # [sample_size, seq_len]
            picked_predicted_events = all_predicted_events[batch_index]        # [sample_size, seq_len, num_events]
            picked_events_next = events_next[batch_index:batch_index + 1]      # [1, padded_seq_len]
            picked_time_next = time_next[batch_index:batch_index + 1]          # [1, padded_seq_len]
            picked_mask_next = mask_next[batch_index:batch_index + 1]          # [1, padded_seq_len]
            
            picked_events_next = picked_events_next[:, :picked_mask_next.sum()]# [1, seq_len]
            picked_events_next = repeat(picked_events_next, '() s -> b s', b = sample_num_positive_negative)
                                                                               # [sample_size, seq_len]
            picked_time_next = picked_time_next[:, :picked_mask_next.sum()]    # [1, seq_len]
            picked_predicted_time = picked_predicted_time[:, :picked_mask_next.sum()]
                                                                               # [sample_size, seq_len]
            picked_predicted_events = picked_predicted_events[:, :picked_mask_next.sum()]
                                                                               # [sample_size, seq_len]
            # Time loss.
            time_difference = torch.abs(picked_predicted_time - picked_time_next)
                                                                               # [sample_size, seq_len]
            # Event loss.
            event_difference = nn.functional.cross_entropy(input = rearrange(picked_predicted_events, '... sl ne -> ... ne sl'), \
                                                           target = picked_events_next, \
                                                           reduction = 'none') # [sample_size, seq_len]
            
            overall_difference = self.lambda_m * event_difference + self.lambda_t * time_difference
                                                                               # [sample_size, seq_len]
            loss_val.append(overall_difference)
        
        return loss_val


    def obtain_posterior_of_t2(self, all_predicted_time, all_predicted_events, \
                               events_next, time_next, mask_next):
        # Imitating the posterior probability in REPLUG, we build the posterior probability of T_2 conditioned on y based on
        # the accuracy of the event prediction, which is measured by the loss function as shown below:
        # L(t, e) = \lambda_t * |t - t_i| + \lambda_m * cross_entropy_loss(e, e_i)
        # where t and e are predicted time and mark, respectively. The posterior is a softmax over the loss of positive and negative
        # samples, which is:
        # p(T_2|y) = \frac{\exp(L(t, e))}{\sum_{T^{\prime}}{\exp(L(t, e))}}
        # In this function, we calculate p(T_2|y).
        
        loss_val = self.loss_func(all_predicted_time, all_predicted_events, \
                                  events_next, time_next, mask_next)           # batch_size * [sample_size, seq_len]
        all_p_posterior = []
        for loss_val_per_batch in loss_val:
            p_posterior = nn.functional.log_softmax(loss_val_per_batch, dim = 0)
                                                                               # [sample_size, seq_len]
            all_p_posterior.append(p_posterior)
        
        return all_p_posterior, loss_val