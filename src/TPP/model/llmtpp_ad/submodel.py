import torch.nn as nn
import torch
import numpy as np
import transformers
import re
import math

from einops import rearrange, repeat, reduce, pack, unpack
from scipy.stats import spearmanr
from src.TPP.model.utils import L1_distance_across_events
from src.TPP.model.llmtpp_ad.transformers_module import lm_module_location
from src.TPP.model.llmtpp_ad.embedding import DataEmbedding


class LLMTPP(nn.Module):
    def __init__(self, llm_class_name, full_llm_name, patch_size, d_model, \
                 d_embedding, num_events, lm_layers, d_lm_embedding, device, dropout):
        super(LLMTPP, self).__init__()
        self.device = device

        self.patch_size = patch_size
        # How many layers in the LM are trainable?
        self.lm_layers = lm_layers
        self.d_model = d_model
        self.d_embedding = d_embedding
        self.d_lm_embedding = d_lm_embedding

        self.enc_embedding = DataEmbedding(d_embedding, d_model, self.patch_size, dropout = dropout, device = self.device)

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

        self.time = nn.Sequential(
            nn.Linear(d_lm_embedding, d_model, device = self.device),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.patch_size, device = self.device),
            nn.Softplus()
        )


    def patchify(self, time_history, mask_history):
        # Split the input sequence into several patches.
        seq_len = time_history.shape[-1]
        free_events_size = seq_len % self.patch_size
        num_of_patches = int(seq_len / self.patch_size) + (1 if free_events_size > 0 else 0)
        p1d = (0, (self.patch_size - free_events_size) % self.patch_size)

        time_history = torch.nn.functional.pad(time_history, p1d, 'constant', 0)
                                                                               # [batch_size, num_of_patches * self.patch_size]
        mask_history = torch.nn.functional.pad(mask_history, p1d, 'constant', 0)
                                                                               # [batch_size, num_of_patches * self.patch_size]
        
        time_history = rearrange(time_history, 'b (np ps) -> b np ps', np = num_of_patches)
                                                                               # [batch_size, num_of_patches, self.patch_size]
        mask_history = rearrange(mask_history, 'b (np ps) -> b np ps', np = num_of_patches)
                                                                               # [batch_size, num_of_patches, self.patch_size]
        
        return time_history, mask_history


    def forward(self, mode, *args, **kwargs):
        task_mapper = {
            'train': self.model_forward,
            'evaluate': self.model_forward
        }

        return task_mapper[mode](*args, **kwargs)
    

    def model_forward(self, input_time, input_mask, mean, var):
        input_time = (input_time - mean) / var                                 # [batch_size, seq_len]
        input_time, input_mask = self.patchify(input_time, input_mask)
                                                                               # [batch_size, num_of_patches, patch_size]
        input_embs = self.enc_embedding(input_time, input_mask)                # [batch_size, num_of_patches, d_model]

        input_embs = torch.nn.functional.pad(input_embs, (0, self.d_lm_embedding - input_embs.shape[-1]))
                                                                               # [batch_size, num_of_patches, d_lm_model]
        outputs = self.retrieved_lm(inputs_embeds = input_embs).last_hidden_state
                                                                               # [batch_size, num_of_patches, d_lm_model]
        time = self.time(outputs)                                              # [batch_size, seq_len, patch_size]

        return time


    def train_procedure(self, time_history, time_prediction, event_history, \
                        event_prediction, mask_history, mask_prediction, mean, var):
        '''
        Args:
            events_history: [batch_size, seq_len] or [batch_size, seq_len, d_history] if custom_events_history = True
            time_history:   [batch_size, seq_len]
        '''

        emb_event_history = self.enc_embedding(event_history)
        event_embeddings = self.mark_embedding(input_events)                   # [batch_size, seq_len, hidden_size]
        time_embeddings = self.compute_temporal_embedding(input_time)          # [batch_size, seq_len, hidden_size]

        # Predict event
        event_prob = []

        for event_per_seq, time_per_seq in zip(input_events, input_time):
            transferred_event_time_per_seq = [(each_event, each_time) for (each_event, each_time) in zip(event_per_seq, time_per_seq)]
            et_mark_sequence_strings = []
            sub_sequence_history_string, sub_sequence_target_string = self.seq_to_str(transferred_event_time_per_seq)
            et_mark_sequence_strings.append(self.long_horizon_prompt.format(sub_sequence_history_string))

            pred_length = max(len(sub_sequence_history_string), len(sub_sequence_target_string))
            
            # Preprocess
            splited_prediction_task = []
            remaining_sample_rate = self.num_return_sequences
            while remaining_sample_rate > 0:
                splited_prediction_task.append(self.num_return_sequences_per_run)
                remaining_sample_rate -= self.num_return_sequences_per_run
            splited_prediction_task[-1] += remaining_sample_rate
            
            predicted_events = None
            for num_return_sequences_per_run in splited_prediction_task:
                predicted_events_per_batch = self.pipeline(et_mark_sequence_strings,
                                                           max_length = self.target_length_cap + len(self.long_horizon_prompt) + pred_length,
                                                           num_return_sequences = num_return_sequences_per_run,
                                                           eos_token_id = self.pipeline.tokenizer.eos_token_id,
                                                           pad_token_id = self.pipeline.tokenizer.eos_token_id,
                                                           **self.generating_parameters)                    
                if predicted_events is None:
                    predicted_events = predicted_events_per_batch
                else:
                    predicted_events = list(map(lambda lst1, lst2: lst1 + lst2, predicted_events, predicted_events_per_batch))
                    
            parsed_predicted_events = self.parse_event(predicted_events)

            def predicted_event_distribution(lst):
                return list(map(lambda n: lst.count(n) / len(lst), range(self.num_events)))
            
            event_prob_per_seq = []
            for item in parsed_predicted_events:
                event_prob_per_seq.append(predicted_event_distribution(item))
            
            event_prob.append(event_prob_per_seq)

        print('Event prediction completed.')


        return torch.argmax(torch.tensor(predicted_event_prob_per_seq), dim = -1).tolist(), \
               time_prediction, time_prediction_with_real_events


    def forward_time_event(self, events_history, time_history, time_next):
        '''
        Args:
            events_history: [batch_size, seq_len] or [batch_size, seq_len, d_history] if custom_events_history = True
            time_history:   [batch_size, seq_len]
            time_next:      [..., batch_size, seq_len, num_events]
            mask:           [batch_size, seq_len]
        '''

        '''
        Turn the history into sequences. We might need a prompt.
        '''
        events_history, time_history = events_history.tolist(), time_history.tolist()

        # Predict time
        time_pred = []
        for event_per_seq, time_per_seq in zip(events_history, time_history):
            transferred_event_time_per_seq = [(each_event, each_time) for (each_event, each_time) in zip(event_per_seq, time_per_seq)]
            te_time_sequence_strings = []
            for idx in range(1, len(transferred_event_time_per_seq)):
                sub_sequence = transferred_event_time_per_seq[0:idx]
                sub_sequence_string = self.seq_to_str(sub_sequence)
                te_time_sequence_strings.append(self.te_time_prompt.format(sub_sequence_string))
            
            # Preprocess
            splited_prediction_task = []
            remaining_sample_rate = self.num_return_sequences
            while remaining_sample_rate > 0:
                splited_prediction_task.append(self.num_return_sequences_per_run)
                remaining_sample_rate -= self.num_return_sequences_per_run
            splited_prediction_task[-1] += remaining_sample_rate
            
            print(len(te_time_sequence_strings))
            predicted_times = None
            for num_return_sequences_per_run in splited_prediction_task:
                predicted_time_per_batch = self.pipeline(te_time_sequence_strings,
                                                           max_length = self.history_length + len(self.et_mark_prompt) + 20,
                                                           do_sample = True,
                                                           temperature = 0.7,
                                                           top_p = 0.95,
                                                           top_k = 40,
                                                           num_return_sequences = num_return_sequences_per_run,
                                                           repetition_penalty = 1.1,
                                                           eos_token_id = self.pipeline.tokenizer.eos_token_id,
                                                           pad_token_id = self.pipeline.tokenizer.eos_token_id)
                if predicted_times is None:
                    predicted_times = predicted_time_per_batch
                else:
                    predicted_times = list(map(lambda lst1, lst2: lst1 + lst2, predicted_times, predicted_time_per_batch))
                    
            parsed_predicted_times = self.parse_time(predicted_times)

            def predicted_time(lst):
                return torch.tensor(lst).mean(dim = -1).tolist()
            
            time_prob_per_seq = []
            for item in parsed_predicted_times:
                time_prob_per_seq.append(predicted_time(item))
            
            time_pred.append(time_prob_per_seq)
        
        print('Time prediction completed.')
        
        # Predict time based on history and predicted events.
        event_prediction = []
        event_prediction_with_real_time = []
        for event_per_seq, time_per_seq, time_pred_per_seq, time_next_per_seq \
            in zip(events_history, time_history, time_pred, time_next):
            transferred_event_time_per_seq = \
                [(each_event, each_time) for (each_event, each_time) in zip(event_per_seq, time_per_seq)]
            
            te_event_sequence_strings = []
            te_event_sequence_strings_with_real_events = []
            for idx in range(1, len(transferred_event_time_per_seq)):
                sub_sequence = transferred_event_time_per_seq[0:idx]
                sub_sequence_string = self.seq_to_str(sub_sequence)
                te_event_sequence_strings.append(self.et_time_prompt.format(sub_sequence_string, time_pred_per_seq[idx - 1]))
                te_event_sequence_strings_with_real_events.append(self.et_time_prompt.format(sub_sequence_string, time_next_per_seq[idx - 1]))

            # Preprocess
            splited_prediction_task = []
            remaining_sample_rate = self.num_return_sequences
            while remaining_sample_rate > 0:
                splited_prediction_task.append(self.num_return_sequences_per_run)
                remaining_sample_rate -= self.num_return_sequences_per_run
            splited_prediction_task[-1] += remaining_sample_rate
            
            predicted_event = None
            predicted_event_with_real_events = None
            for num_return_sequences_per_run in splited_prediction_task:
                predicted_event_per_batch = self.pipeline(te_event_sequence_strings,
                                                         max_length = self.history_length + len(self.et_time_prompt) + 20,
                                                         do_sample = True,
                                                         temperature = 0.7,
                                                         top_p = 0.95,
                                                         top_k = 40,
                                                         num_return_sequences = num_return_sequences_per_run,
                                                         repetition_penalty = 1.1,
                                                         eos_token_id = self.pipeline.tokenizer.eos_token_id,
                                                         pad_token_id = self.pipeline.tokenizer.eos_token_id)
                predicted_event_with_real_time_per_batch = self.pipeline(te_event_sequence_strings_with_real_events,
                                                                         max_length = self.history_length + len(self.et_time_prompt) + 20,
                                                                         do_sample = True,
                                                                         temperature = 0.7,
                                                                         top_p = 0.95,
                                                                         top_k = 40,
                                                                         num_return_sequences = num_return_sequences_per_run,
                                                                         repetition_penalty = 1.1,
                                                                         eos_token_id = self.pipeline.tokenizer.eos_token_id,
                                                                         pad_token_id = self.pipeline.tokenizer.eos_token_id)
                if predicted_event is None:
                    predicted_event = predicted_event_per_batch
                    predicted_event_with_real_events = predicted_event_with_real_time_per_batch
                else:
                    predicted_event = list(map(lambda lst1, lst2: lst1 + lst2, predicted_event, predicted_event_per_batch))
                    predicted_event_with_real_events = list(map(lambda lst1, lst2: lst1 + lst2, predicted_event_with_real_events, predicted_event_with_real_time_per_batch))
                    
            parsed_predicted_event = self.parse_event(predicted_event)
            parsed_predicted_event_with_real_times = self.parse_event(predicted_event_with_real_events)

            def predicted_event_distribution(lst):
                return list(map(lambda n: lst.count(n) / len(lst), range(self.num_events)))
            
            pred_event_per_seq = []
            pred_event_per_seq_with_real_times = []
            for item1, item2 in zip(parsed_predicted_event, parsed_predicted_event_with_real_times):
                pred_event_per_seq.append(predicted_event_distribution(item1))
                pred_event_per_seq_with_real_times.append(predicted_event_distribution(item2))
            
            event_prediction.append(pred_event_per_seq)
            event_prediction_with_real_time.append(pred_event_per_seq_with_real_times)

        return torch.argmax(torch.tensor(event_prediction), dim = -1).tolist(), \
               torch.argmax(torch.tensor(event_prediction_with_real_time), dim = -1).tolist(), \
               time_pred


    def sample(self, sampled_events_history, sampled_time_history, tau, mean, var):
        '''
        Args:
            events_history: [number_of_sampled_sequences, sampled_seq_len]
            time_history:   [number_of_sampled_sequences, sampled_seq_len]
            tau:            [number_of_sampled_sequences, 1, num_events] if we need events else [batch_size, 1]
            mask:           [number_of_sampled_sequences, sampled_seq_len]
        '''

        '''
        Obtain historical embeddings.
        '''
        sampled_time_history = (sampled_time_history - mean) / var             # [number_of_sampled_sequences, sampled_seq_len]

        sampled_events_embeddings = self.events(sampled_events_history)        # [number_of_sampled_sequences, sampled_seq_len, d_history]
        sampled_history, sampled_history_ps = pack([sampled_events_embeddings, sampled_time_history], 'b s *')
                                                                               # [number_of_sampled_sequences, sampled_seq_len, d_history + 1]

        # Reshape hidden output for full connection layers.
        _, (sampled_history_embedding, _) = self.his_encoder(sampled_history)  # [1, number_of_sampled_sequences, d_history]
        sampled_history_embedding = rearrange(sampled_history_embedding, 'l nss dh -> nss l dh')
                                                                               # [number_of_sampled_sequences, 1, d_history]
        sampled_history_embedding = repeat(sampled_history_embedding, 'b s dh -> b s ne dh', ne = self.num_events)
                                                                               # [number_of_sampled_sequences, 1, num_events, d_history]
        sampled_history_embedding = self.history_mapper(sampled_history_embedding)
                                                                               # [number_of_sampled_sequences, 1, num_events, d_intensity]
        '''
        Obtain timestamp embeddings.
        '''
        tau = (tau - mean) / var                                               # [number_of_sampled_sequences, 1, num_events]
        time_next_zero = torch.ones_like(tau) * (-mean / var)                  # [number_of_sampled_sequences, 1, num_events]

        time_embedding = tau.unsqueeze(dim = -1) * self.nonneg_activation(self.weight_for_t)
                                                                               # [number_of_sampled_sequences, 1, num_events, d_intensity]
        time_zero_embedding = time_next_zero.unsqueeze(dim = -1) * self.nonneg_activation(self.weight_for_t)
                                                                               # [number_of_sampled_sequences, 1, num_events, d_intensity]
        
        time_embedding = self.time_mapper(time_embedding)                      # [number_of_sampled_sequences, 1, num_events, d_intensity]
        time_zero_embedding = self.time_mapper(time_zero_embedding)            # [number_of_sampled_sequences, 1, num_events, d_intensity]
        
        output = time_embedding + sampled_history_embedding                    # [number_of_sampled_sequences, 1, num_events, d_intensity]
        output_zero = time_zero_embedding + sampled_history_embedding          # [number_of_sampled_sequences, 1, num_events, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [number_of_sampled_sequences, 1, num_events, d_intensity]
            output = self.layer_activation(output)                             # [number_of_sampled_sequences, 1, num_events, d_intensity]

            output_zero = layer(output_zero)                                   # [number_of_sampled_sequences, 1, num_events, d_intensity]
            output_zero = self.layer_activation(output_zero)                   # [number_of_sampled_sequences, 1, num_events, d_intensity]

        probability_integral_from_t_to_inf = self.nonneg_integral(-self.aggregate(output))
                                                                               # [number_of_sampled_sequences, 1, num_events, 1]
        probability_integral_from_tl_to_inf = self.nonneg_integral(-self.aggregate(output_zero))
                                                                               # [number_of_sampled_sequences, 1, num_events, 1]

        probability_integral_from_t_to_inf = rearrange(probability_integral_from_t_to_inf, '... 1 -> ...')
                                                                               # [number_of_sampled_sequences, 1, num_events]
        probability_integral_from_tl_to_inf = reduce(probability_integral_from_tl_to_inf, '... ne 1 -> ... ()', 'sum')
                                                                               # [number_of_sampled_sequences, 1, 1]

        return probability_integral_from_t_to_inf / (probability_integral_from_tl_to_inf + self.epsilon)


    def probability(self, events_history, time_history, time_next, resolution, mean, var):
        '''
        Intensity integral & intensity function prober. Perhaps, we can support intensity integral as well.
        Args:
        events_history:[batch_size, seq_len]
        time_history:  [batch_size, seq_len]
        time_next:     [batch_size, seq_len]
        resolution:    int
        '''

        '''
        History embeddings
        '''
        time_history = (time_history - mean) / var                             # [batch_size, seq_len]

        events_embeddings = self.events(events_history)                        # [batch_size, seq_len, d_history]
        history, history_ps = pack([events_embeddings, time_history], 'b s *') # [batch_size, seq_len, d_history + 1]

        hidden_history, (_, _) = self.his_encoder(history)                     # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)                   # [batch_size, seq_len, d_intensity]

        hidden_history = repeat(hidden_history, 'b s di -> b s r ne di', r = resolution, ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]

        '''
        Expanded time embedding 
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_multiplier * time_next.unsqueeze(dim = -1) # [batch_size, seq_len, resolution]
        time_expand = original_time_expand.clone()                             # [batch_size, seq_len, resolution]
        time_expand = repeat(time_expand, 'b s r -> b s r ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events]

        time_expand.requires_grad = True
        time_expand_norm = (time_expand - mean) / var                          # [batch_size, seq_len, resolution, num_events]

        emb_time_expand = time_expand_norm.unsqueeze(dim = -1) * self.nonneg_activation(self.weight_for_t)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]

        emb_time_expand = self.time_mapper(emb_time_expand)                    # [batch_size, seq_len, resolution, num_events, d_intensity]
        output = emb_time_expand + hidden_history                              # [batch_size, seq_len, resolution, num_events, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity]
            output = self.layer_activation(output)                             # [batch_size, seq_len, resolution, num_events, d_intensity]

        expand_integral = self.nonneg_integral(-self.aggregate(output))        # [batch_size, seq_len, resolution, num_events, 1]
        
        integral_from_zero_to_inf = expand_integral[:, :, 0, :, :].detach()    # [batch_size, seq_len, num_events, 1]
        integral_sum = reduce(integral_from_zero_to_inf, 'b s ne 1 -> b s 1 1 1', 'sum')
                                                                               # [batch_size, seq_len, 1, 1, 1]
        expand_integral = expand_integral / (integral_sum + self.epsilon)      # [batch_size, seq_len, resolution, num_events, 1]

        expand_probability = - torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events]
        time_expand.requires_grad = False

        expand_probability = expand_probability.detach()                       # [batch_size, seq_len, resolution, num_events]

        '''
        Restore the original timestamp
        '''
        batch_size, seq_len = events_history.shape[0], events_history.shape[1]
        dummy_inception = torch.zeros((batch_size, seq_len, 1), device = self.device)
        timestamp, timestamp_ps = pack(
            [dummy_inception, original_time_expand.diff(dim = -1)],
            'b s *')                                                           # [batch_size, seq_len, resolution]

        return expand_probability, timestamp


    def get_event_embedding(self, input_event):
        return self.events(input_event)                                        # [batch_size, seq_len, d_history]


    def model_probe_function(self, events_history, time_history, time_next, resolution, mean, var, mask):
        '''
        We use this function to dive into the fullynn and find the reason of abrupt gradient drop around 0
        Args:
        time_history: [batch_size, seq_len]
        time_next:    [batch_size, seq_len]
        resolution:   int
        '''

        '''
        History embeddings
        '''
        time_history = (time_history - mean) / var                             # [batch_size, seq_len]

        events_embeddings = self.events(events_history)                        # [batch_size, seq_len, d_history]
        history, history_ps = pack([events_embeddings, time_history], 'b s *') # [batch_size, seq_len, d_history + 1]

        hidden_history, (_, _) = self.his_encoder(history)                     # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)                   # [batch_size, seq_len, d_intensity]

        hidden_history = repeat(hidden_history, 'b s di -> b s r ne di', r = resolution, ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]

        '''
        Expanded time embedding 
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_multiplier * rearrange(time_next, '... -> ... 1')
                                                                               # [batch_size, seq_len, resolution]
        time_expand = original_time_expand.clone()                             # [batch_size, seq_len, resolution]
        time_expand = repeat(original_time_expand, 'b s r -> b s r ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events]
        
        time_expand.requires_grad = True      
        time_expand_norm = (time_expand - mean) / var                          # [batch_size, seq_len, resolution, num_events]

        emb_time_expand = time_expand_norm.unsqueeze(dim = -1) * self.nonneg_activation(self.weight_for_t)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]

        emb_time_expand = self.time_mapper(emb_time_expand)                    # [batch_size, seq_len, resolution, num_events, d_intensity]
        output = emb_time_expand + hidden_history                              # [batch_size, seq_len, resolution, num_events, d_intensity]


        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity]
            output = self.layer_activation(output)                             # [batch_size, seq_len, resolution, num_events, d_intensity]

        expand_integral = self.nonneg_activation(-self.aggregate(output))      # [batch_size, seq_len, resolution, num_events, 1]
        expand_integral = expand_integral.squeeze(dim = -1)                    # [batch_size, seq_len, resolution, num_events]

        integral_from_zero_to_inf = expand_integral[:, :, 0, :].detach()       # [batch_size, seq_len, num_events]
        integral_sum = reduce(integral_from_zero_to_inf, 'b s ne -> b s ()', 'sum')
                                                                               # [batch_size, seq_len, 1]
        integral_sum = rearrange(integral_sum, 'b s 1 -> b s 1 1')             # [batch_size, seq_len, 1, 1]
        expand_integral = expand_integral / (integral_sum + self.epsilon)      # [batch_size, seq_len, resolution, num_events]


        # Gradient 1: Integral -> time
        events_probability_at_each_interpolated_timestamp = - torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
            retain_graph=True
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events]
                
        time_expand.requires_grad = False

        # Timestamp part
        batch_size, seq_len = hidden_history.shape[0], hidden_history.shape[1]
        zero_inception = torch.zeros((batch_size, seq_len, 1), device = self.device)
        timestamp, timstamp_ps = pack(
            [zero_inception, original_time_expand.diff(dim = -1)],
            'b s *')                                                           # [batch_size, seq_len, resolution]
        timestamp = rearrange(timestamp, 'b s r -> b (s r)')                   # [batch_size, seq_len * resolution]

        '''
        The data dict is defined here.
        This dict should pack all data required by plot().
        '''
        data = {}
        data['expand_probability_for_each_event'] = events_probability_at_each_interpolated_timestamp
                                                                               # [batch_size, seq_len, resolution, num_events]

        probability_for_each_event = \
            rearrange(events_probability_at_each_interpolated_timestamp.detach().cpu(), 'b s r ne -> b (s r) ne')
                                                                           # [batch_size, seq_len * resolution, num_events]
        
        spearman_matrix = []
        pearson_matrix = []
        L1_matrix = []
        for _, (expand_probability_per_seq, mask_per_seq, time_next_per_seq) in \
                                              enumerate(zip(probability_for_each_event, mask, time_next)):
            seq_len = mask_per_seq.sum()

            # rho: spearman coefficient
            if self.num_events == 1:
                spearman_matrix_per_seq = np.array([[1.,],])
            else:
                spearman_matrix_per_seq = spearmanr(expand_probability_per_seq[:seq_len * resolution])[0]
                if self.num_events == 2:
                    spearman_matrix_per_seq = np.array([[1, spearman_matrix_per_seq], [spearman_matrix_per_seq, 1]])

            # r: pearson coefficient
            pearson_matrix_per_seq = np.corrcoef(expand_probability_per_seq[:seq_len * resolution], rowvar = False)
            if self.num_events == 1:
                pearson_matrix_per_seq = rearrange(np.array(pearson_matrix_per_seq), ' -> () ()')
            
            # L^1 metric
            L1_matrix_per_seq = L1_distance_across_events(expand_probability_per_seq[:seq_len * resolution], 
                                            resolution = resolution, num_events = self.num_events,
                                            time_next = time_next_per_seq[:seq_len])

            spearman_matrix.append(spearman_matrix_per_seq)
            pearson_matrix.append(pearson_matrix_per_seq)
            L1_matrix.append(L1_matrix_per_seq)

        data['spearman_matrix'] = spearman_matrix
        data['pearson_matrix'] = pearson_matrix
        data['L1_matrix'] = L1_matrix

        return data, timestamp