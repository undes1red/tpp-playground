import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import pack, rearrange, repeat, reduce
from src.tpp.tpp_models.dstpp.gdst import GaussianDiffusion_ST
from src.tpp.tpp_models.dstpp.tst import ST_Diffusion
from src.tpp.tpp_models.dstpp.encoders import Transformer_ST
from src.tpp.tpp_models.dstpp.metrics import time_metric_func, mark_metric_func


class DSTPP(nn.Module):
    '''
    Main DSTPP module.
    '''
    def __init__(self, dim_marks, d_enc_emb, d_rnn, d_inner, n_layers, n_head, d_k, d_v, dropout, \
                 loss_type, timesteps, samplingsteps, objective, beta_schedule, padding_index, device):
        super(DSTPP, self).__init__()
        self.device = device
        self.dim_marks = dim_marks
        
        model = ST_Diffusion(device = device, dim = 1 + self.dim_marks, condition = True, cond_dim = d_enc_emb)
    
        self.diffusion = GaussianDiffusion_ST(model, loss_type = loss_type, dim_events = 1 + self.dim_marks, \
                                              timesteps = timesteps, sampling_timesteps = samplingsteps, objective = objective, \
                                              beta_schedule = beta_schedule).to(self.device)
    
        self.transformer = Transformer_ST(device = device, d_model = d_enc_emb, d_rnn = d_rnn, d_inner = d_inner, \
                                          n_layers = n_layers, n_head = n_head, d_k = d_k, d_v = d_v, dropout = dropout, \
                                          loc_dim = self.dim_marks)
    

    def mask_separate_history_and_next(self, mask):
        '''
        This function is a little bit different from the divide_history_and_next() in other MTPP models.

        This function knows where padding events are and the separation operation ignores them.

        a b c d e [pad] [pad] [pad]
                   |
                   |
                   |
                   \/
        History: a b c d [pad] [pad] [pad] 
        Next:    b c d e [pad] [pad] [pad]
        '''
        batch_size, _ = mask.shape
        event_length = mask.sum(dim = -1, keepdim = True)                      # [batch_size, 1]
        blank_history = torch.zeros(batch_size, 1, dtype = torch.long, device = self.device)
                                                                               # [batch_size, 1]
        mask_history, mask_next = mask.clone(), mask.clone()                   # 2 * [batch_size, seq_len]
        mask_history = mask_history.scatter(dim = 1, index = event_length - 1, src = blank_history)
                                                                               # [batch_size, seq_len]
        mask_next[:, 0] = 0                                                    # [batch_size, seq_len]

        return mask_history, mask_next


    def normalize_time_and_mark(self, mark, time, mean_and_std_mark, mean_and_std_time):
        mean_event, std_event = mean_and_std_mark
        mean_time, std_time = mean_and_std_time
        einop = f'dim -> {"() " * (len(mark.shape) - 1)}dim'

        normed_mark = (mark - rearrange(mean_event, einop)) / (rearrange(std_event, einop))
                                                                               # [batch_size, seq_len, dim_marks]
        normed_time = (time - mean_time) / std_time                            # [batch_size, seq_len]

        return normed_mark, normed_time

    
    def unnormalize_time_and_mark(self, normalized_mark, normalized_time, mean_and_std_mark, mean_and_std_time):
        mean_event, std_event = mean_and_std_mark
        mean_time, std_time = mean_and_std_time
        einop = f'dim -> {"() " * (len(normalized_mark.shape) - 1)}dim'

        original_mark = (normalized_mark * (rearrange(std_event, einop))) + rearrange(mean_event, einop)
                                                                               # [batch_size, seq_len, dim_marks]
        original_time = (normalized_time * std_time) + mean_time               # [batch_size, seq_len]

        return original_mark, original_time
    

    def evaluate_time(self, sampled_time, relative_time_sequence, mask_next, time_metric):
        metric_val_per_time = time_metric_func(time_metric, sampled_time, relative_time_sequence.unsqueeze(dim = 0))
                                                                               # [sample_rate, batch_size, seq_len]
        metric_val_per_sample = reduce(metric_val_per_time * mask_next.unsqueeze(dim = 0), 'sr ... -> sr', 'sum')
                                                                               # [sample_rate]
        return metric_val_per_sample
    

    def evaluate_mark(self, sampled_mark, mark_sequence, mask_next, mark_metric):
        metric_val_per_mark = mark_metric_func(mark_metric, sampled_mark, mark_sequence.unsqueeze(dim = 0))
                                                                               # [sample_rate, batch_size, seq_len]
        metric_val_per_sample = reduce(metric_val_per_mark * mask_next.unsqueeze(dim = 0), 'sr ... -> sr', 'sum')
                                                                               # [sample_rate]
        return metric_val_per_sample


    def forward(self, input_events, absolute_time_sequence, relative_time_sequence, mask, \
                mean_and_std_events, mean_and_std_time):
        normed_input_events, normed_relative_time \
            = self.normalize_time_and_mark(input_events, relative_time_sequence, mean_and_std_events, mean_and_std_time)
                                                                               # [batch_size, seq_len, dim_marks] + # [batch_size, seq_len]

        seq_enc_out = self.transformer(normed_input_events, absolute_time_sequence, mask)
                                                                               # [batch_size, seq_len, 3 * d_model]
        
        combined_event_sequence, _ = pack((normed_relative_time, normed_input_events), 'b s *')
                                                                               # [batch_size, seq_len, dim_marks + 1]

        mask_history, mask_next = self.mask_separate_history_and_next(mask)    # [batch_size, seq_len]

        selected_his_event_representation = seq_enc_out[mask_history == 1]     # [..., d_model]
        selected_combined_event_sequence = combined_event_sequence[mask_next == 1]
                                                                               # [..., dim_marks + 1]
        # Fit the dimension requirement of the diffusion model.
        selected_his_event_representation = selected_his_event_representation.unsqueeze(dim = -2)
                                                                               # [..., 1, d_model]
        selected_combined_event_sequence = selected_combined_event_sequence.unsqueeze(dim = -2)
                                                                               # [..., 1, dim_marks + 1]

        loss_mean = self.diffusion(selected_combined_event_sequence, selected_his_event_representation)
        nll_batch_sum, nll_temporal_sum, nll_spatial_sum \
            = self.diffusion.NLL_cal(selected_combined_event_sequence, selected_his_event_representation)
        
        the_number_of_events = mask_next.sum().item()

        return loss_mean, nll_batch_sum, nll_temporal_sum, nll_spatial_sum, the_number_of_events


    def sample(self, input_events, absolute_time_sequence, relative_time_sequence, mask, \
                 mean_and_std_events, mean_and_std_time, sample_rate):
        normed_input_events, _ \
            = self.normalize_time_and_mark(input_events, relative_time_sequence, mean_and_std_events, mean_and_std_time)
                                                                               # [batch_size, seq_len, dim_marks] + # [batch_size, seq_len]
        seq_enc_out = self.transformer(normed_input_events, absolute_time_sequence, mask)
                                                                               # [batch_size, seq_len, 3 * d_model]
        mask_history, mask_next = self.mask_separate_history_and_next(mask)    # [batch_size, seq_len]
        selected_his_event_representation = seq_enc_out[mask_history == 1]     # [..., d_model]

        # Fit the dimension requirement of the diffusion model.
        selected_his_event_representation = selected_his_event_representation.unsqueeze(dim = -2)
                                                                               # [..., 1, d_model]
        sampled_unbatched_seq = []
        for _ in range(sample_rate):
            sampled_unbatched_seq.append(self.diffusion.sample(batch_size = selected_his_event_representation.shape[0], cond = selected_his_event_representation))
                                                                               # [..., 1, dim_marks + 1]
        sampled_unbatched_seq = torch.cat(sampled_unbatched_seq, dim = 0)      # [sample_rate * ..., 1, dim_marks + 1]
        sampled_normed_time = sampled_unbatched_seq[..., 0]                    # [sample_rate * ..., 1]
        sampled_normed_mark = sampled_unbatched_seq[..., 1:]                   # [sample_rate * ..., 1, dim_marks]

        sampled_unbatched_mark, sampled_unbatched_time = \
            self.unnormalize_time_and_mark(sampled_normed_mark, sampled_normed_time, mean_and_std_events, mean_and_std_time)
                                                                               # [sample_rate * ..., 1, dim_marks] + [sample_rate * ..., 1]
        
        sampled_mark = torch.zeros(sample_rate, *input_events.shape, device = self.device)
                                                                               # [sample_rate, batch_size, seq_len, dim_mark]
        sampled_time = torch.zeros(sample_rate, *relative_time_sequence.shape, device = self.device)
                                                                               # [sample_rate, batch_size, seq_len]
        
        sampled_mark[repeat(mask_next, '... -> sr ...', sr = sample_rate) == 1] = sampled_unbatched_mark.squeeze(dim = -2)
                                                                               # [sample_rate, batch_size, seq_len, dim_mark]
        sampled_time[repeat(mask_next, '... -> sr ...', sr = sample_rate) == 1] = sampled_unbatched_time.squeeze(dim = -1)
                                                                               # [sample_rate, batch_size, seq_len]

        return sampled_mark, sampled_time, mask_next


    def evaluate(self, input_events, absolute_time_sequence, relative_time_sequence, mask, \
                 mean_and_std_events, mean_and_std_time, sample_rate = 50, time_metric = 'mae', mark_metric = 'euclid'):
        sampled_mark, sampled_time, mask_next \
              = self.sample(input_events, absolute_time_sequence, relative_time_sequence, mask, \
                            mean_and_std_events, mean_and_std_time, sample_rate = sample_rate)
                                                                               # [sample_rate, batch_size, seq_len, dim_marks] + [sample_rate, batch_size, seq_len]
        # Evaluate on mark prediction.
        metric_on_mark = self.evaluate_mark(sampled_mark, input_events, mask_next, mark_metric).mean()
        # Evaluate on time prediction.
        metric_on_time = self.evaluate_time(sampled_time, relative_time_sequence, mask_next, time_metric).mean()

        return metric_on_mark, metric_on_time