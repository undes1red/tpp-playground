import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange, pack

from src.toolbox.misc import check_tensor, move_from_tensor_to_ndarray, easy_model_load, conditional_compile_class_method
from src.toolbox.integration import approximate_integration

from src.tpp.tpp_models.utils import median_prediction, predict_event

from src.MDI.model.nhps.ctlstm import CTLSTM_barebone


class NHPS(nn.Module):
    def __init__(self, opt, device, num_events, history_module_name, d_mark_embedding, d_input, d_hidden, \
                 history_encoder_layers, dropout, integration_sample_rate, mark_missing_probability, config_loaded_model):
        super(NHPS, self).__init__()
        self.device = device
        self.compile_or_not = opt.compile
        self.num_events = num_events
        self.d_input = d_input
        self.mark_missing_probability = mark_missing_probability
        
        self.left_to_right_mtpp_model = easy_model_load(root_path = opt.root_path, device = self.device,
                                                        **config_loaded_model)

        self.right_to_left_mtpp_model \
            = CTLSTM_barebone(device = device, num_events = self.num_events, history_module_name = history_module_name, \
                              d_mark_embedding = d_mark_embedding, d_input = d_input, d_hidden = d_hidden, \
                              history_encoder_layers = history_encoder_layers, dropout = dropout, \
                              integration_sample_rate = integration_sample_rate)
        
        self.backward_history_remap = nn.Linear(d_input, d_input, device = self.device)

        # This layer translates decayed hidden states into intensity function values.
        self.intensity_layer = nn.Sequential(
            nn.Linear(d_input, self.num_events, bias = True, device = self.device),
            nn.Softplus(beta = 1.)
        )
    
    
    def divide_history_and_next(self, input):
        input_history, input_next = input[..., :-1].clone(), input[..., 1:].clone()
        return input_history, input_next
    

    def remove_dummy_event_from_mask(self, mask):
        '''
        Remove the probability of the dummy event by mask.
        '''
        mask_without_dummy = torch.zeros_like(mask)                            # [batch_size, seq_len - 1]
        for idx, mask_per_seq in enumerate(mask):
            dummy_index = mask_per_seq.sum() - 1
            mask_without_dummy_per_seq = copy.deepcopy(mask_per_seq.detach())
            mask_without_dummy_per_seq[dummy_index] = 0
            mask_without_dummy[idx] = mask_without_dummy_per_seq
        
        return mask_without_dummy
    
    
    def forward(self, forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq):
        complete_time, complete_events, complete_mask = forward_complete_data
        backward_complete_time, backward_complete_events, backward_complete_mask = backward_complete_data
        
        time_history, time_next = self.divide_history_and_next(complete_time)  # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(complete_events)
                                                                               # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(complete_mask)  # [batch_size, seq_len] * 2
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len] * 2

        backward_time_history, backward_time_next = self.divide_history_and_next(backward_complete_time)
                                                                               # [batch_size, seq_len] * 2
        log_q_z_con_x_on_all_samples = []
        packed_data = zip(padded_obs_data, padded_backward_obs_event_seq)
        for idx, ((obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, missing_mask_for_one_seq, _,),
                  (backward_obs_time_for_one_seq, backward_obs_events_for_one_seq, backward_obs_mask_for_one_seq, backward_missing_mask_for_one_seq, _)) in enumerate(packed_data):

            missing_mask_history_for_one_seq, missing_mask_next_for_one_seq \
                = self.divide_history_and_next(missing_mask_for_one_seq)       # [sample_num, seq_len] * 2
            backward_missing_mask_history_for_one_seq, backward_missing_mask_next_for_one_seq \
                = self.divide_history_and_next(backward_missing_mask_for_one_seq)
                                                                               # [sample_num, seq_len] * 2

            which_history_the_next_event_should_know = missing_mask_history_for_one_seq.cumsum(dim = -1) - 1
                                                                               # [sample_num, seq_len]
            backward_which_history_the_next_event_should_know = backward_missing_mask_history_for_one_seq.cumsum(dim = -1) - 1
                                                                               # [sample_num, seq_len]
            repeated_which_history_the_next_event_should_know = repeat(which_history_the_next_event_should_know, '... -> ... di', di = self.d_input)
                                                                               # [sample_num, seq_len, d_input]
            backward_repeated_which_history_the_next_event_should_know = repeat(backward_which_history_the_next_event_should_know, '... -> ... di', di = self.d_input)
                                                                               # [sample_num, seq_len, d_input]

            obs_time_history, obs_time_next = self.divide_history_and_next(obs_time_for_one_seq)
                                                                               # [sample_num, sample_length] * 2
            obs_events_history, obs_events_next = self.divide_history_and_next(obs_events_for_one_seq)
                                                                               # [sample_num, sample_length] * 2
            
            backward_obs_time_history, backward_obs_time_next = self.divide_history_and_next(backward_obs_time_for_one_seq)
                                                                               # [sample_num, sample_length] * 2
            backward_obs_events_history, backward_obs_events_next = self.divide_history_and_next(backward_obs_events_for_one_seq)
                                                                               # [sample_num, sample_length] * 2

            forward_history_state = self.left_to_right_mtpp_model.model.nhps_get_history_state(obs_time_history.float(), obs_events_history)
                                                                               # [sample_num, sample_length, d_input]
            backward_history_state = self.right_to_left_mtpp_model(backward_obs_time_history.float(), backward_obs_events_history)
                                                                               # [sample_num, sample_length, d_input]

            aligned_forward_history_state = torch.gather(forward_history_state, -2, repeated_which_history_the_next_event_should_know)
                                                                               # [sample_num, seq_len, d_input]
            cum_obs_time_history = obs_time_history.cumsum(dim = -1)           # [sample_num, sample_length]
            aligned_cum_obs_time_history = torch.gather(cum_obs_time_history, -1, which_history_the_next_event_should_know)
                                                                               # [sample_num, seq_len]
            aligned_forward_time_next = time_next[idx:idx + 1, :aligned_cum_obs_time_history.shape[-1]].cumsum(dim = -1) - aligned_cum_obs_time_history
                                                                               # [sample_num, seq_len]

            aligned_backward_history_state = torch.gather(backward_history_state, -2, backward_repeated_which_history_the_next_event_should_know)
                                                                               # [sample_num, seq_len, d_input]
            cum_backward_obs_time_history = backward_obs_time_history.cumsum(dim = -1)
                                                                               # [sample_num, sample_length]
            aligned_backward_cum_obs_time_history = torch.gather(cum_backward_obs_time_history, -1, backward_which_history_the_next_event_should_know)
                                                                               # [sample_num, seq_len]
            
            forward_where_to_start = time_history[idx:idx+1, :aligned_cum_obs_time_history.shape[-1]].cumsum(dim = -1) - aligned_cum_obs_time_history
            
            backward_where_to_start = backward_time_history[idx:idx+1, :aligned_cum_obs_time_history.shape[-1]].cumsum(dim = -1) - aligned_backward_cum_obs_time_history
                                                                               # [sample_num, seq_len]
            forward_where_to_start = (forward_where_to_start * (~missing_mask_history_for_one_seq)).float()
            backward_where_to_start = (backward_where_to_start * (~backward_missing_mask_history_for_one_seq)).float()
            
            forward_hidden_state_at_t = \
                self.left_to_right_mtpp_model.model.nhps_get_decayed_state(
                    aligned_forward_history_state,
                    aligned_forward_time_next.float()
                )                                                              # [sample_num, seq_len, d_input]
            backward_hidden_state_at_t = \
                self.right_to_left_mtpp_model.get_decayed_state(
                    aligned_backward_history_state,
                    backward_where_to_start
                )                                                              # [sample_num, seq_len, d_input]
            backward_hidden_state_at_t = backward_hidden_state_at_t.flip(dims = (-2,))
                                                                               # [sample_num, seq_len, d_input]
            mixed_hidden_state_at_t = forward_hidden_state_at_t + self.backward_history_remap(backward_hidden_state_at_t)
                                                                               # [sample_num, seq_len, d_input]
            intensity_q = self.intensity_layer(mixed_hidden_state_at_t)        # [sample_num, seq_len, num_events]
            
            expanded_forward_hidden_state_at_t, expanded_time = \
                self.left_to_right_mtpp_model.model.nhps_get_decayed_state_of_a_interval(
                    aligned_forward_history_state, 
                    forward_where_to_start, 
                    time_next[idx:idx+1, :missing_mask_next_for_one_seq.shape[-1]]
                )                                                              # [sample_num, seq_len, integration_sample_rate, d_input]
            expanded_backward_hidden_state_at_t, _ = \
                self.right_to_left_mtpp_model.get_decayed_state_of_a_interval(
                    aligned_backward_history_state,
                    backward_where_to_start,
                    backward_time_next[idx:idx+1, :missing_mask_next_for_one_seq.shape[-1]]
                )                                                              # [sample_num, seq_len, integration_sample_rate, d_input]
            expanded_backward_hidden_state_at_t = expanded_backward_hidden_state_at_t.flip(dims = (-3, -2))
                                                                               # [sample_num, seq_len, integration_sample_rate, d_input]

            mixed_expanded_hidden_state_at_t = expanded_forward_hidden_state_at_t + self.backward_history_remap(expanded_backward_hidden_state_at_t)
                                                                               # [sample_num, seq_len, integration_sample_rate, d_input]
            expanded_intensity_q = self.intensity_layer(mixed_expanded_hidden_state_at_t)
                                                                               # [sample_num, seq_len, integration_sample_rate, d_input]
            integral_of_intensity_q = approximate_integration(expanded_intensity_q, expanded_time, dim = -2, only_integral = True)
                                                                               # [sample_num, seq_len, num_events]
            
            obs_events_mask_for_one_seq = F.one_hot((events_next * mask_next_without_dummy)[idx:idx + 1, :missing_mask_next_for_one_seq.shape[-1]], num_classes = self.num_events)
                                                                               # [1, seq_len, num_events]
            log_q_z_and_x = torch.log((intensity_q * obs_events_mask_for_one_seq).sum(dim = -1) + 1e-20) - integral_of_intensity_q.sum(dim = -1)
                                                                               # [sample_num, seq_len]
            log_q_z_con_x = log_q_z_and_x * (~missing_mask_next_for_one_seq).int()
                                                                               # [sample_num, seq_len]
            log_q_z_con_x = log_q_z_con_x.sum(dim = -1) / (~missing_mask_next_for_one_seq).sum(dim = -1)
                                                                               # [sample_num]
            log_q_z_con_x_on_all_samples.append(log_q_z_con_x)
        
        log_q_z_con_x_on_all_samples = torch.stack(log_q_z_con_x_on_all_samples, dim = 0).mean(dim = -1)
                                                                               # [batch_size]
        return log_q_z_con_x_on_all_samples
    
    
    @conditional_compile_class_method
    def autoregressive_sampling_by_its_for_nhpf(self, events_history, time_history, \
                                                its_lower_bound, mean, std):
        def bisect_target(taus, probability_threshold):
            '''
            Args:
            1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
            2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
            3. mask: the padding mask introduced by the dataloader. shape: [batch_size, seq_len + 1]
            '''
            expanded_integral_all_events, _, = \
                self.left_to_right_mtpp_model.model.sample_for_tm(time_history, taus, events_history)
                                                                               # [1, num_events]
            expanded_integral = expanded_integral_all_events.sum(dim = -1)     # [1]

            return expanded_integral + torch.log(1 - probability_threshold)

        probability_threshold = torch.zeros((1,), device = self.device)        # [1]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound)     # [1]
        sampled_time = median_prediction(50, 1e-4, bisect_target, probability_threshold)
                                                                               # [1]
        integral_all_events, intensity_all_events = \
            self.left_to_right_mtpp_model.model.sample_for_tm(time_history, sampled_time, events_history)
                                                                               # [1, num_events]
        probability = intensity_all_events * torch.exp(-integral_all_events.sum(dim = -1))
                                                                               # [1, num_events]
        sampled_marks = predict_event(intensity_all_events, sample = True)     # [1]
        sampled_marks_mask = F.one_hot(sampled_marks, num_classes = self.num_events)
                                                                               # [1, num_events]
        probability = (probability * sampled_marks_mask).sum()                 # [1]

        return sampled_time, sampled_marks, probability, probability_threshold


    def imputing_by_nhpf(self, obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, num_imputed_seq, mean, std):
        weights = []
        imputed_sequences_events = []
        imputed_sequences_time = []
        number_of_events = obs_mask_for_one_seq.sum()
        
        for _ in range(num_imputed_seq):
            idx_on_obs_events = 1
            
            events_history = torch.tensor([self.num_events, ], dtype = torch.int32, device = self.device)
            time_history = torch.tensor([0.0, ], dtype = torch.float32, device = self.device)
            weight = 1.0
            aggregate_time = 0.0
            
            while True:
                if idx_on_obs_events >= number_of_events:
                    break
                
                sampled_time, sampled_marks, _, _ = \
                    self.autoregressive_sampling_by_its_for_nhpf(
                        events_history = events_history, time_history = time_history, \
                        its_lower_bound = 0.0, mean = mean, std = std
                    )                                                          # [1,] * 4
                
                if aggregate_time + sampled_time < obs_time_for_one_seq[idx_on_obs_events]:
                    events_history, _ = pack((events_history, sampled_marks), 'b *')
                                                                                # [1, seq_len + 1]
                    time_history, _ = pack((time_history, sampled_time), 'b *')
                                                                                # [1, seq_len + 1]
                    aggregate_time += sampled_time
                else:
                    picked_obs_event = obs_events_for_one_seq[idx_on_obs_events:idx_on_obs_events+1]
                    picked_obs_time = obs_time_for_one_seq[idx_on_obs_events:idx_on_obs_events+1]
                    
                    events_history, _ = pack((events_history, picked_obs_event), 'b *')
                                                                                # [1, seq_len + 1]
                    time_history, _ = pack((time_history, picked_obs_time.float() - aggregate_time), 'b *')
                                                                                # [1, seq_len + 1]
                    idx_on_obs_events += 1
                    aggregate_time = 0.0
            
            weights.append(weight)
            imputed_sequences_events.append(events_history)
            imputed_sequences_time.append(time_history)
        
        return weights, (imputed_sequences_events, imputed_sequences_time)
    
    
    @conditional_compile_class_method
    def autoregressive_sampling_by_its_for_nhps(self, picked_forward_history_state, picked_backward_history_state, \
                                                sample_start_time, max_interval_length):
        # Add fake dimensions to all history states so we can use
        # nhps_get_decayed_state_of_a_interval() and get_decayed_state_of_a_interval().
        picked_forward_history_state = rearrange(picked_forward_history_state, 'di -> () () di')
                                                                               # [batch_size, seq_len, di]
        picked_backward_history_state = rearrange(picked_backward_history_state, 'di -> () () di')
                                                                               # [batch_size, seq_len, di]
        sample_start_time = rearrange(sample_start_time, '() -> () ()')        # [batch_size, seq_len]
        max_interval_length = rearrange(max_interval_length, '() -> () ()')    # [batch_size, seq_len]
        
        assert max_interval_length > 0, "Why negative max_interval_length?"
        
        def bisect_target(taus, probability_threshold):
            '''
            Args:
            1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
            2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
            3. mask: the padding mask introduced by the dataloader. shape: [batch_size, seq_len + 1]
            '''
            expanded_forward_hidden_state_at_t, expanded_time = \
                self.left_to_right_mtpp_model.model.nhps_get_decayed_state_of_a_interval(
                    picked_forward_history_state,
                    sample_start_time,
                    taus
                )                                                              # [batch_size, seq_len, integration_sample_rate, d_input]
            expanded_backward_hidden_state_at_t, _ = \
                self.right_to_left_mtpp_model.get_decayed_state_of_a_interval(
                    picked_backward_history_state,
                    max_interval_length - taus,
                    taus
                )                                                              # [batch_size, seq_len, integration_sample_rate, d_input]
            expanded_backward_hidden_state_at_t = expanded_backward_hidden_state_at_t.flip(dims = (-3, -2))
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]
            mixed_expanded_hidden_state_at_t = expanded_forward_hidden_state_at_t + self.backward_history_remap(expanded_backward_hidden_state_at_t)
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]
            expanded_intensity_q = self.intensity_layer(mixed_expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]
            integral_of_intensity_q = approximate_integration(expanded_intensity_q, expanded_time, dim = -2, only_integral = True).sum(dim = -1)
                                                                               # [batch_size, seq_len, num_events]
            return integral_of_intensity_q + torch.log(1 - probability_threshold)

        probability_threshold = torch.zeros((1,), device = self.device)        # [1]
        torch.nn.init.uniform_(probability_threshold)                          # [1]
        
        # Step 1: Will we find a solution in the given time interval?
        # Yes: find the solution as a imputed event.
        # No: there is no event between the current and the next observed event. 
        expanded_forward_hidden_state_at_t, expanded_time = \
            self.left_to_right_mtpp_model.model.nhps_get_decayed_state_of_a_interval(
                picked_forward_history_state,
                sample_start_time,
                max_interval_length
            )                                                                  # [batch_size, seq_len, integration_sample_rate, d_input]
        expanded_backward_hidden_state_at_t, _ = \
            self.right_to_left_mtpp_model.get_decayed_state_of_a_interval(
                picked_backward_history_state,
                sample_start_time,
                max_interval_length
            )                                                                  # [batch_size, seq_len, integration_sample_rate, d_input]
        expanded_backward_hidden_state_at_t = expanded_backward_hidden_state_at_t.flip(dims = (-3, -2))
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]
        mixed_expanded_hidden_state_at_t = expanded_forward_hidden_state_at_t + self.backward_history_remap(expanded_backward_hidden_state_at_t)
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]
        expanded_intensity_q = self.intensity_layer(mixed_expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]
        integral_of_intensity_q = approximate_integration(expanded_intensity_q, expanded_time, dim = -2, only_integral = True).sum(dim = -1)
                                                                               # [batch_size, seq_len]
        '''
        are_we_get_legit_sample = (integral_of_intensity_q + torch.log(1 - probability_threshold) >= 0).item()
        
        if not are_we_get_legit_sample:
            # Next event is an observed event.
            expanded_intensity_p = self.left_to_right_mtpp_model.model.nhps_get_intensity(expanded_forward_hidden_state_at_t)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
            integral_of_intensity_p = approximate_integration(expanded_intensity_p, expanded_time, dim = -2, only_integral = True).sum(dim = -1)
                                                                               # [batch_size, seq_len]
            selected_intensity_p = expanded_intensity_p[:, :, -1]              # [batch_size, seq_len, num_events]
            probability_p = selected_intensity_p * torch.exp(-integral_of_intensity_p).squeeze()
                                                                               # [num_events]
        
            return are_we_get_legit_sample, probability_p
        '''
        
        # Next event is a missing event.
        sampled_time = median_prediction(50, 1e-4, bisect_target, probability_threshold, r_val = max_interval_length)
                                                                               # [1]
        expanded_forward_hidden_state_at_t, expanded_time = \
            self.left_to_right_mtpp_model.model.nhps_get_decayed_state_of_a_interval(
                picked_forward_history_state,
                sample_start_time,
                sampled_time
            )                                                                  # [batch_size, seq_len, integration_sample_rate, d_input]
        expanded_backward_hidden_state_at_t, _ = \
            self.right_to_left_mtpp_model.get_decayed_state_of_a_interval(
                picked_backward_history_state,
                max_interval_length - sampled_time,
                sampled_time
            )                                                                  # [batch_size, seq_len, integration_sample_rate, d_input]
        expanded_backward_hidden_state_at_t = expanded_backward_hidden_state_at_t.flip(dims = (-3, -2))
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]
        mixed_expanded_hidden_state_at_t = expanded_forward_hidden_state_at_t + self.backward_history_remap(expanded_backward_hidden_state_at_t)
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]
        expanded_intensity_q = self.intensity_layer(mixed_expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
        integral_of_intensity_q = approximate_integration(expanded_intensity_q, expanded_time, dim = -2, only_integral = True).sum(dim = -1)
                                                                               # [batch_size, seq_len]
        selected_intensity_q = expanded_intensity_q[:, :, -1]                  # [batch_size, seq_len, num_events]
        probability_q = selected_intensity_q * torch.exp(-integral_of_intensity_q)
                                                                               # [batch_size, seq_len, num_events]
        sampled_marks = predict_event(selected_intensity_q, sample = True)     # [1]
        sampled_marks_mask = F.one_hot(sampled_marks, num_classes = self.num_events)
                                                                               # [1, num_events]
        probability_q = (probability_q * sampled_marks_mask).sum()             # [1]
        
        expanded_intensity_p = \
            self.left_to_right_mtpp_model.model.nhps_get_intensity(
                expanded_forward_hidden_state_at_t
            )                                                                  # [batch_size, seq_len, integration_sample_rate, num_events]
        integral_of_intensity_p = approximate_integration(expanded_intensity_p, expanded_time, dim = -2, only_integral = True).sum(dim = -1)
                                                                               # [batch_size, seq_len]
        selected_intensity_p = expanded_intensity_p[:, :, -1]                  # [batch_size, seq_len, num_events]
        probability_p = selected_intensity_p * torch.exp(-integral_of_intensity_p)
                                                                               # [batch_size, seq_len, num_events]
        probability_p = (probability_p * sampled_marks_mask).sum()             # [1]

        # Reshape the output.
        sampled_time = sampled_time.squeeze(dim = 0)                           # [1]
        sampled_marks = sampled_marks.squeeze(dim = 0)                         # [1]
        
        return sampled_time, sampled_marks, probability_q, probability_p


    def imputing_by_nhps(self, obs_time_for_one_seq, obs_events_for_one_seq, obs_missing_mask_for_one_seq, \
                         backward_obs_time_for_one_seq, backward_obs_events_for_one_seq, backward_obs_missing_mask_for_one_seq, \
                         num_imputed_seq, mean, std):
        missing_mask_history_for_one_seq, missing_mask_next_for_one_seq \
            = self.divide_history_and_next(obs_missing_mask_for_one_seq)       # [full_seq_len] * 2
        backward_missing_mask_history_for_one_seq, backward_missing_mask_next_for_one_seq \
            = self.divide_history_and_next(backward_obs_missing_mask_for_one_seq)
                                                                               # [full_seq_len] * 2
            
        which_history_the_next_event_should_know = missing_mask_history_for_one_seq.cumsum(dim = -1) - 1
                                                                               # [full_seq_len]
        backward_which_history_the_next_event_should_know = backward_missing_mask_history_for_one_seq.cumsum(dim = -1) - 1
                                                                               # [full_seq_len]
        repeated_which_history_the_next_event_should_know = repeat(which_history_the_next_event_should_know, '... -> ... di', di = self.d_input)
                                                                               # [full_seq_len, d_input]
        backward_repeated_which_history_the_next_event_should_know = repeat(backward_which_history_the_next_event_should_know, '... -> ... di', di = self.d_input)
                                                                               # [full_seq_len, d_input]
        
        obs_time_history, obs_time_next = self.divide_history_and_next(obs_time_for_one_seq)
                                                                               # [sample_length] * 2
        obs_events_history, obs_events_next = self.divide_history_and_next(obs_events_for_one_seq)
                                                                               # [sample_length] * 2
        obs_seq_len = obs_time_next.shape[-1]
        
        backward_obs_time_history, backward_obs_time_next = self.divide_history_and_next(backward_obs_time_for_one_seq)
                                                                               # [sample_length] * 2
        backward_obs_events_history, backward_obs_events_next = self.divide_history_and_next(backward_obs_events_for_one_seq)
                                                                               # [sample_length] * 2

        backward_history_state = self.right_to_left_mtpp_model(backward_obs_time_history.float(), backward_obs_events_history)
                                                                               # [sample_length, d_input]
        obtained_imputed_seqs_weight = [1.0,] * num_imputed_seq
        obtained_imputed_seqs_time = [torch.tensor([obs_time_history[0]], device = self.device, dtype = torch.float32) for _ in range(num_imputed_seq)]
        obtained_imputed_seqs_events = [torch.tensor([obs_events_history[0]], device = self.device, dtype = torch.int64) for _ in range(num_imputed_seq)]
        
        for idx in range(obs_seq_len):
            picked_backward_history_state = backward_history_state[-idx]       # [d_input]
            max_interval_length = obs_time_next[idx:idx+1]
            sample_start_time = torch.zeros(1, device = self.device)
            aggregate_time = 0
            
            for sample_idx in range(num_imputed_seq):
                while True:
                    hidden_state_at_sampled_time \
                        = self.left_to_right_mtpp_model.model.nhps_get_history_state(obtained_imputed_seqs_time[sample_idx].float(), obtained_imputed_seqs_events[sample_idx])
                    hidden_state_at_sampled_time = hidden_state_at_sampled_time[-1]

                    data = self.autoregressive_sampling_by_its_for_nhps(hidden_state_at_sampled_time, picked_backward_history_state, \
                                                                          sample_start_time, max_interval_length - aggregate_time)
                                                                                   # [1, 1]
                    sampled_time, sampled_marks, probability_q, probability_p = data
                    new_aggregate_time = aggregate_time + sampled_time
                    
                    if new_aggregate_time < obs_time_next[idx:idx+1]:
                        # find a missing event.
                        aggregate_time = new_aggregate_time
                        obtained_imputed_seqs_time[sample_idx] = torch.cat((obtained_imputed_seqs_time[sample_idx], sampled_time))
                        obtained_imputed_seqs_events[sample_idx] = torch.cat((obtained_imputed_seqs_events[sample_idx], sampled_marks))
                        obtained_imputed_seqs_weight[sample_idx] = obtained_imputed_seqs_weight[sample_idx] * probability_p / probability_q * self.mark_missing_probability[sampled_marks]
                    else:
                        # sampling failed, no missing event observed.
                        obtained_imputed_seqs_time[sample_idx] = torch.cat((obtained_imputed_seqs_time[sample_idx], obs_time_next[idx:idx+1] - aggregate_time))
                        obtained_imputed_seqs_events[sample_idx] = torch.cat((obtained_imputed_seqs_events[sample_idx], obs_events_next[idx:idx+1]))
                        if obs_events_next[idx] != self.num_events:
                            # We hit the end of the sequence when obs_events_next[idx] == self.num_events. The common practice outputs the probability that no event occurs between [t_n, T].
                            # But here we need the value of the probability density function, which is undefined, so in this case we do nothing when we hit the end of the sequence.
                            obtained_imputed_seqs_weight[sample_idx] = obtained_imputed_seqs_weight[sample_idx] * probability_p * (1 - self.mark_missing_probability[obs_events_next[idx]])
                        
                        # Restore the history state.
                        aggregate_time = 0
                        break
        
        '''
        for _ in range(num_imputed_seq):
            idx_on_obs_events = 1
            
            events_history = torch.tensor([[self.num_events], ], dtype = torch.int32, device = self.device)
            time_history = torch.tensor([[0.0], ], dtype = torch.float32, device = self.device)
            
            backward_events_history = torch.tensor([[self.num_events], ], dtype = torch.int32, device = self.device)
            backward_time_history = torch.tensor([[0.0], ], dtype = torch.float32, device = self.device)

            weight = 1.0
            aggregate_time = 0.0
            
            while True:
                if idx_on_obs_events >= number_of_events:
                    break
                
                sampled_time, sampled_marks, _, _ = \
                    self.autoregressive_sampling_by_its_for_nhps(
                        events_history = events_history, time_history = time_history, \
                        backward_events_history = backward_events_history, backward_time_history = backward_time_history, \
                        its_lower_bound = 0.0, mean = mean, std = std
                    )                                                          # [1,] * 4
                
                if aggregate_time + sampled_time < obs_time_for_one_seq[idx_on_obs_events]:
                    events_history, _ = pack((events_history, sampled_marks), 'b *')
                                                                                # [1, seq_len + 1]
                    time_history, _ = pack((time_history, sampled_time), 'b *')
                                                                                # [1, seq_len + 1]
                    aggregate_time += sampled_time
                else:
                    picked_obs_event = obs_events_for_one_seq[idx_on_obs_events:idx_on_obs_events+1]
                    picked_obs_time = obs_time_for_one_seq[idx_on_obs_events:idx_on_obs_events+1]
                    
                    events_history, _ = pack((events_history, picked_obs_event), 'b *')
                                                                                # [1, seq_len + 1]
                    time_history, _ = pack((time_history, picked_obs_time.float() - aggregate_time), 'b *')
                                                                                # [1, seq_len + 1]
                    idx_on_obs_events += 1
                    aggregate_time = 0.0
            
            weights.append(weight)
            events_history, time_history = move_from_tensor_to_ndarray(events_history, time_history)
            imputed_sequences.append([events_history, time_history])
        '''
        
        return obtained_imputed_seqs_weight, (obtained_imputed_seqs_events, obtained_imputed_seqs_time)
    
    
    def get_nhpf_probability(self, forward_complete_data, padded_obs_data):
        complete_time, complete_events, complete_mask = forward_complete_data

        time_history, time_next = self.divide_history_and_next(complete_time)  # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(complete_events)
                                                                               # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(complete_mask)  # [batch_size, seq_len] * 2
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len] * 2

        integral_all_events, intensity_all_events \
            = self.left_to_right_mtpp_model.model(time_history.float(), time_next.float(), events_history)
                                                                               # 2 * [batch_size, seq_len, num_events]
        
        complete_events_mask = F.one_hot(events_next * mask_next_without_dummy, num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        log_p_x_z = torch.log((intensity_all_events * complete_events_mask).sum(dim = -1) + 1e-20) - integral_all_events.sum(dim = -1)
                                                                               # [batch_size, seq_len]
        log_p_x_z = log_p_x_z * mask_next_without_dummy                        # [batch_size, seq_len]
        
        log_p_x_z_missing = []
        for (obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, missing_mask_for_one_seq, _,) in padded_obs_data:
            _, missing_mask_next_for_one_seq = self.divide_history_and_next(missing_mask_for_one_seq)
                                                                               # [sample_num, seq_len] * 2
            log_p_x_z_missing.append((log_p_x_z * ~missing_mask_next_for_one_seq).sum(dim = -1) / (~missing_mask_next_for_one_seq).sum(dim = -1))
                                                                               # [*, sample_num]
        log_p_x_z_missing = torch.stack(log_p_x_z_missing, dim = 0).mean(dim = -1)
                                                                               # [batch_size]
        return log_p_x_z_missing