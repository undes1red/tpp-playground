import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
import numpy as np
from scipy.stats import spearmanr
import math

from src.TPP.model.utils import L1_distance_across_events
from src.TPP.model.attnhp.transformers import EncoderLayer, MultiHeadAttention


class ATTNHP(nn.Module):
    def __init__(self, device, num_events, d_input, d_time, n_layers, n_head, dropout, integration_sample_rate,
                 use_norm, sharing_param_layer):
        super(ATTNHP, self).__init__()
        self.num_events = num_events
        self.device = device
        self.integration_sample_rate = integration_sample_rate

        self.d_input = d_input
        self.d_time = d_time

        self.div_term = torch.exp(torch.arange(0, d_time, 2, device = self.device) * -(math.log(10000.0) / d_time)).reshape(1, 1, -1)
        # here num_types already includes [PAD], [BOS], [EOS]
        self.Emb = nn.Embedding(self.num_events + 1, d_input, padding_idx = self.num_events, device = self.device)
        self.n_layers = n_layers
        self.n_head = n_head
        self.sharing_param_layer = sharing_param_layer
        if not sharing_param_layer:
            self.heads = []
            for i in range(n_head):
                self.heads.append(
                    nn.ModuleList(
                        [
                            EncoderLayer(d_model = d_input + d_time,
                                         self_attn = MultiHeadAttention(1, d_input + d_time, d_input, dropout, output_linear = False, device = self.device),
                                         use_residual = False,
                                         dropout = dropout, 
                                         device = self.device)
                            for _ in range(n_layers)
                        ]
                    )
                )
            self.heads = nn.ModuleList(self.heads)
        else:
            self.heads = []
            for i in range(n_head):
                self.heads.append(
                    nn.ModuleList(
                        [
                            EncoderLayer(d_model = d_input + d_time,
                                         self_attn = MultiHeadAttention(1, d_input + d_time, d_input, dropout, output_linear = False, device = self.device),
                                         use_residual = False,
                                         dropout = dropout,
                                         device = self.device
                        )
                            for _ in range(0)
                        ]
                    )
                )
            self.heads = nn.ModuleList(self.heads)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(d_input, device = self.device)
        self.inten_linear = nn.Linear(d_input * n_head, self.num_events, device = self.device)
        self.softplus = nn.Softplus()
        self.eps = torch.finfo(torch.float32).eps
        # self.add_bos = dataset.add_bos
        self.add_bos = True


    def compute_temporal_embedding(self, time):
        pe = torch.zeros(*(time.shape), self.d_time, device = self.device)
        _time = time.unsqueeze(-1)
        pe[..., 0::2] = torch.sin(_time * self.div_term)
        pe[..., 1::2] = torch.cos(_time * self.div_term)
        # pe = pe * non_pad_mask.unsqueeze(-1)
        return pe


    def forward_pass(self, init_cur_layer_, tem_enc, tem_enc_layer, enc_input, combined_mask, batch_non_pad_mask = None):
        cur_layers = []
        seq_len = enc_input.size(-2)
        for head_i in range(self.n_head):
            cur_layer_ = init_cur_layer_                                       # [batch_size, seq_len, d_input]
            for layer_i in range(self.n_layers):
                layer_ = torch.cat([cur_layer_, tem_enc_layer], dim = -1)      # [...batch_size, seq_len, d_input + d_time]
                _combined_input = torch.cat([enc_input, layer_], dim = -2)     # [...batch_size, 2 * seq_len, d_input + d_time]
                if self.sharing_param_layer:
                    enc_layer = self.heads[head_i][0]
                else:
                    enc_layer = self.heads[head_i][layer_i]
                enc_output = enc_layer(
                    _combined_input,
                    combined_mask
                )                                                              # [...batch_size, 2 * seq_len, d_input + d_time]
                if batch_non_pad_mask is not None:
                    _cur_layer_ = enc_output[:, seq_len:, :] * (batch_non_pad_mask.unsqueeze(-1))
                else:
                    _cur_layer_ = enc_output[:, seq_len:, :]

                # add residual connection
                cur_layer_ = torch.tanh(_cur_layer_) + cur_layer_              # [...batch_size, seq_len, d_input]
                enc_input = torch.cat([enc_output[:, :seq_len, :], tem_enc], dim = -1)
                                                                               # [...batch_size, seq_len, d_input + d_time]
                # non-residual connection
                # cur_layer_ = torch.tanh(_cur_layer_)

                # enc_output *= _combined_non_pad_mask.unsqueeze(-1)
                # layer_ = torch.tanh(enc_output[:, enc_input.size(1):, :])
                if self.use_norm:
                    cur_layer_ = self.norm(cur_layer_)                         # [...batch_size, seq_len, d_input]
            cur_layers.append(cur_layer_)                                      # n_head * [...batch_size, seq_len, d_input]
        cur_layer_ = torch.cat(cur_layers, dim = -1)                           # [...batch_size, seq_len, n_head * d_input]

        return cur_layer_
    

    def add_decorative_dimensions(self, input, tensor_start_with_decorative_dimensions):
        '''
        tensor_start_with_decorative_dimensions: must looks like a tensor with shape [..., batch_size, seq_len]
        '''
        number_of_additional_dimensions = range(len(tensor_start_with_decorative_dimensions.shape) - 2)
        einop = '... -> ' + ' '.join([f'a_{i}' for i in number_of_additional_dimensions]) + ' ...'
        dimension_dict = {f'a_{i}': value for i, value in zip(number_of_additional_dimensions, tensor_start_with_decorative_dimensions.shape)}
        output = repeat(input, einop, **dimension_dict)                        # [..., batch_size, seq_len, d_input]
        
        # Make it compatible with other attnhp codes.
        output = rearrange(output, '... sl di -> (...) sl di')                 # [...batch_size, seq_len, d_input]

        return output


    def hidden_rep_encoder(self, event_seqs, time_seqs, batch_non_pad_mask, attention_mask, extra_times = None):
        tem_enc = self.compute_temporal_embedding(time_seqs)                   # [batch_size, seq_len, d_time]
        tem_enc *= batch_non_pad_mask.unsqueeze(-1)                            # [batch_size, seq_len, d_time]
        enc_input = torch.tanh(self.Emb(event_seqs))                           # [batch_size, seq_len, d_input]
        init_cur_layer_ = torch.zeros_like(enc_input)                          # [batch_size, seq_len, d_input]

        layer_mask = (torch.eye(attention_mask.size(1), device = self.device) < 1).unsqueeze(0).expand_as(attention_mask)
                                                                               # [batch_size, seq_len, seq_len]
        if extra_times is None:
            tem_enc_layer = tem_enc                                            # [batch_size, seq_len, d_time]
        else:
            tem_enc_layer = self.compute_temporal_embedding(extra_times)       # [..., batch_size, seq_len, d_time]
            tem_enc_layer *= batch_non_pad_mask.unsqueeze(-1)                  # [..., batch_size, seq_len, d_time]
            tem_enc_layer = rearrange(tem_enc_layer, '... sl di -> (...) sl di')
                                                                               # [...batch_size, seq_len, d_input]

        _combined_mask = torch.cat([attention_mask, layer_mask], dim=-1)       # [batch_size, seq_len, 2 * seq_len]
        contextual_mask = torch.cat([attention_mask, torch.ones_like(layer_mask)], dim=-1)
                                                                               # [batch_size, seq_len, 2 * seq_len]
        _combined_mask = torch.cat([contextual_mask, _combined_mask], dim=1)   # [batch_size, 2 * seq_len, 2 * seq_len]
        enc_input = torch.cat([enc_input, tem_enc], dim=-1)                    # [batch_size, seq_len, d_input + d_time]

        enc_input = self.add_decorative_dimensions(enc_input, extra_times)     # [...batch_size, seq_len, d_input]
        init_cur_layer_ = self.add_decorative_dimensions(init_cur_layer_, extra_times)
                                                                               # [...batch_size, seq_len, d_input]
        _combined_mask = self.add_decorative_dimensions(_combined_mask, extra_times)
                                                                               # [...batch_size, 2 * seq_len, 2 * seq_len]
        tem_enc = self.add_decorative_dimensions(tem_enc, extra_times)         # [...batch_size, seq_len, d_time]
        extended_batch_non_pad_mask = self.add_decorative_dimensions(batch_non_pad_mask, extra_times)
                                                                               # [..., batch_size, seq_len]
        extended_batch_non_pad_mask = rearrange(extended_batch_non_pad_mask, '... sl -> (...) sl')
                                                                               # [...batch_size, seq_len]
        cur_layer_ = self.forward_pass(init_cur_layer_, tem_enc, tem_enc_layer, enc_input, _combined_mask, extended_batch_non_pad_mask)
                                                                               # [batch_size, seq_len, d_input]

        return cur_layer_


    def forward(self, absolute_time_history, absolute_time_next, relative_time_history, relative_time_next, \
                events_history, mask_history, mask_next):
        '''
        time_seq: absolute timestamps
        time_delta_seq: relative timestamps, seemingly not used in this model.
        event_seq: event mark sequences
        batch_non_pad_mask: mask out all padding events
        attention_mask: attention masks for self-attention module
        type_mask: one hot vector of event sequence to select proper intensity values from enc_inten.

        Rosetta:
        event_seq[:, :-1] <-> events_history
        event_seq[:, 1:] <-> events_next

        time_seq[:, :-1] <-> absolute_time_history
        time_seq[:, 1:] <-> absolute_time_next

        time_delta_seq[:, :-1] <-> relative_time_history
        time_delta_seq[:, 1:] <-> relative_time_next

        batch_non_pad_mask[:, :-1] <-> mask_history
        batch_non_pad_mask[:, 1:] <-> mask_next

        attention_mask <-> an upper triangle matrix.
        It seems that the original code uses attention mask[:, 1:, :-1] rather than the full mask
        because they assume that bos always presents.
        
        If the length of a sequence is 5 (without bos and eos dummy events)
        attention_mask:
          [1, 1, 1, 1, 1]
          [0, 1, 1, 1, 1]
          [0, 0, 1, 1, 1]
          [0, 0, 0, 1, 1]
          [0, 0, 0, 0, 1]
        attention_mask[:, 1:, :-1](We directly generate this tensor and name it the new "attention_mask".):
          [0, 1, 1, 1]
          [0, 0, 1, 1]
          [0, 0, 0, 1]
          [0, 0, 0, 0]
        '''

        '''
        original batch
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = batch
        '''
        #0. preprocessing
        batch_size_with_decoration = absolute_time_next.shape[:-1]
        attention_mask = torch.ones(*(absolute_time_history.shape), absolute_time_history.shape[-1], device = self.device)
                                                                               # [batch_size, seq_len, seq_len]
        attention_mask = torch.triu(attention_mask, diagonal = 1)              # [batch_size, seq_len, seq_len]

        # 1. compute event-loglik
        enc_out = self.hidden_rep_encoder(events_history, absolute_time_history, \
                                          mask_next, attention_mask, absolute_time_next)
                                                                               # [...batch_size, seq_len, d_input * n_head]
        event_lambdas = self.softplus(self.inten_linear(enc_out))              # [...batch_size, seq_len, num_events]
        event_lambdas = event_lambdas.view(*(absolute_time_next.shape), self.num_events)
                                                                               # [..., batch_size, seq_len, num_events]
        # original: 1->1, 2->2
        # event_lambdas = torch.sum(enc_inten * type_mask, dim = 2) + self.eps # [batch_size, seq_len]
        # now: 1->2, 2->3
        # event_lambdas = enc_inten + self.eps
        # in case event_lambdas == 0
        # event_lambdas.masked_fill_(~batch_non_pad_mask, 1.0)
        # event_lambdas.masked_fill_(~batch_non_pad_mask[:, 1:], 1.0)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample times
        # 2.2 compute intensities at sampled times
        # due to GPU memory limitation, we may not be able to compute all intensities at all sampled times,
        # step gives the batch size w.r.t how many sampled times we should process at each batch
        # note from sishun: no need to change this value.
        # Another proof of why the numerical integration estimation could be bad.

        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device = self.device)
                                                                               # [integration_sample_rate]
        einop = f'... -> ...{" ()" * len(relative_time_next.shape)}'
        time_multiplier = rearrange(time_multiplier, einop)                    # [integration_sample_rate, batch_size, seq_len]
        diff_time = relative_time_next * mask_next                             # [..., batch_size, seq_len]
        temp_time = diff_time.unsqueeze(dim = 0) * time_multiplier             # [integration_sample_rate, ..., batch_size, seq_len]
        temp_time = rearrange(temp_time, 'isr ... bs sl -> isr (... bs) sl')   # [integration_sample_rate, ...batch_size, seq_len]

        # The original ANHP always assumes that self.add_bos = True.
        # Thus we safely remove the codes when self.add_bos = False
        # why non_pad_mask start from 1?
        # think about a simple case: [e] [e] [pad] (non_pad_mask: 1 1 0)
        # you want to compute the first interval only, so if you use non_pad_mask[:, :-1] (1, 1),
        # you will compute both the first and the second intervals!
        repeated_absolute_time_history = self.add_decorative_dimensions(absolute_time_history, absolute_time_next)
                                                                               # [..., batch_size, seq_len]
        repeated_absolute_time_history = rearrange(repeated_absolute_time_history, '... bs sl -> (... bs) sl')
                                                                               # [...batch_size, seq_len]
        temp_time += repeated_absolute_time_history.unsqueeze(0)               # [integration_sample_rate, ...batch_size, seq_len]
        # for interval computation, we will never use the last event -- that is why we have -1 in
        # event_seq, time_seq, attention_mask
        einop = '... bs sl -> (... bs) sl'
        reshaped_events_history = rearrange(self.add_decorative_dimensions(events_history, absolute_time_next), einop)
                                                                               # [...batch_size, seq_len]
        reshaped_absolute_time_history = rearrange(self.add_decorative_dimensions(absolute_time_history, absolute_time_next), einop)
                                                                               # [...batch_size, seq_len]
        reshaped_mask_next = rearrange(self.add_decorative_dimensions(mask_next, absolute_time_next), einop)
                                                                               # [...batch_size, seq_len]
        reshaped_attention_mask = self.add_decorative_dimensions(attention_mask, absolute_time_next)
                                                                               # [...batch_size, seq_len, seq_len]
        all_lambda = self._compute_intensities_fast(reshaped_events_history, reshaped_absolute_time_history, \
                                                    reshaped_mask_next, reshaped_attention_mask,temp_time)
                                                                               # [integration_sample_rate, ...batch_size, seq_len, num_events]
        
        # 2.3 compute the empirical expectation of the summation
        all_lambda = all_lambda.sum(dim = 0) / self.integration_sample_rate    # [...batch_size, seq_len, num_events]
        all_lambda = all_lambda.view(*diff_time.shape, self.num_events)        # [..., batch_size, seq_len, num_events]
        integral_lambda = all_lambda * diff_time.unsqueeze(dim = -1)           # [batch_size, seq_len, num_events]

        return integral_lambda, event_lambdas


    def _compute_intensities_fast(self, event_seq, time_seq, batch_non_pad_mask, attention_mask, temp_time, step = 20):
        # fast version, can only use in log-likelihood computation
        # assume we will sample the same number of times in each interval of the event_seqs
        all_lambda = []
        batch_size = event_seq.size(0)
        seq_len = event_seq.size(1)
        num_samples = temp_time.size(0)
        for i in range(0, num_samples, step):
            _extra_time = temp_time[i: i + step, :, :]
            _step = _extra_time.size(0)
            _extra_time = _extra_time.reshape(_step * batch_size, -1)
            _types = event_seq.expand(_step, -1, -1).reshape(_step * batch_size, -1)
            _times = time_seq.expand(_step, -1, -1).reshape(_step * batch_size, -1)
            _batch_non_pad_mask = batch_non_pad_mask.unsqueeze(0).expand(_step, -1, -1).reshape(_step * batch_size, -1)
            _attn_mask = attention_mask.unsqueeze(0).expand(_step, -1, -1, -1).reshape(_step * batch_size, seq_len,
                                                                                       seq_len)
            _enc_output = self.hidden_rep_encoder(_types, _times, _batch_non_pad_mask, _attn_mask, _extra_time)
            all_lambda.append(self.softplus(self.inten_linear(_enc_output)).reshape(_step, batch_size, seq_len, -1))
        all_lambda = torch.cat(all_lambda, dim = 0)
        return all_lambda


    def compute_intensities_at_sampled_times(self, event_seq, time_seq, sampled_times):
        # Assumption: all the sampled times are distributed [time_seq[...,-1], next_event_time]
        # used for thinning algorithm
        num_batches = event_seq.size(0)
        seq_len = event_seq.size(1)
        assert num_batches == 1, "Currently, no support for batch mode (what is a good way to do batching in thinning?)"
        if num_batches == 1 and num_batches < sampled_times.size(0):
            _sample_size = sampled_times.size(0)
            # multiple sampled_times
            event_seq = event_seq.unsqueeze(0).expand(_sample_size, num_batches, seq_len).reshape(_sample_size, seq_len)
            time_seq = time_seq.unsqueeze(0).expand(_sample_size, num_batches, seq_len).reshape(_sample_size, seq_len)
            num_batches = event_seq.size(0)
        assert (time_seq[:, -1:] <= sampled_times).all(), "sampled times must occur not earlier than last events!"
        num_samples = sampled_times.size(1)

        # 1. prepare input embeddings for "history"
        tem_enc = self.compute_temporal_embedding(time_seq)
        enc_input = torch.tanh(self.Emb(event_seq))
        init_cur_layer_ = torch.zeros((sampled_times.size(0), sampled_times.size(1), enc_input.size(-1))).to(
            sampled_times.device)
        enc_input = torch.cat([enc_input, tem_enc], dim=-1)
        tem_layer_ = self.compute_temporal_embedding(sampled_times)

        # 2. prepare attention mask
        attention_mask = torch.ones((num_batches, seq_len + num_samples, seq_len + num_samples)).to(event_seq.device)
        attention_mask[:, :seq_len, :seq_len] = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0).cuda()
        # by default, regard all_sampled times to be equal to the last_event_time
        # recall that we use 1 for "not attending", 0 for "attending"
        # t_i == sampled_t
        attention_mask[:, seq_len:, :seq_len - 1] = 0
        # t_i < sampled_t
        attention_mask[:, seq_len:, seq_len - 1][time_seq[:, -1:] < sampled_times] = 0
        attention_mask[:, seq_len:, seq_len:] = (torch.eye(num_samples) < 1).unsqueeze(0).to(event_seq.device)
        cur_layer_ = self.forward_pass(init_cur_layer_, tem_enc, tem_layer_, enc_input, attention_mask)

        sampled_intensities = self.softplus(self.inten_linear(cur_layer_))

        return sampled_intensities

'''
    def state_decay(self, mu, eta, gamma, duration_t, num_dimension_prior_batch):
        \'''
        mu, eta, gamma: shape: [batch_size, seq_len, d_hidden]
        dutation_t:     shape: [batch_size, seq_len, (integration_sample_rate, num_events)]
        \'''
        assert len(duration_t.shape) - 2 - num_dimension_prior_batch >= 0, "Too few dimensions in duration_t!"

        # add additional dimension to mu, eta, and gamma.
        mu = rearrange(mu, f'... d_i -> {"() " * num_dimension_prior_batch}... {"() " * (len(duration_t.shape) - 2 - num_dimension_prior_batch)}d_i')
                                                                               # [..., batch_size, seq_len, (integration_sample_rate, num_events), d_input]
        eta = rearrange(eta, f'... d_i -> {"() " * num_dimension_prior_batch}... {"() " * (len(duration_t.shape) - 2 - num_dimension_prior_batch)}d_i')
                                                                               # [..., batch_size, seq_len, (integration_sample_rate, num_events), d_input]
        gamma = rearrange(gamma, f'... d_i -> {"() " * num_dimension_prior_batch}... {"() " * (len(duration_t.shape) - 2)}d_i')
                                                                               # [..., batch_size, seq_len, (integration_sample_rate, num_events), d_input]

        duration_t = duration_t.unsqueeze(dim = -1)                            # [..., batch_size, seq_len, (integration_sample_rate, num_events), 1]
        cell_t = torch.tanh(mu + (eta - mu) * torch.exp(-gamma * duration_t))  # [..., batch_size, seq_len, (integration_sample_rate, num_events), d_input]
        
        return cell_t


    def integration_estimator(self, expanded_intensity_value, expanded_time, integration_sample_rate):
        # tensor check
        assert expanded_intensity_value.shape[-2:] == (integration_sample_rate, self.num_events)
        assert expanded_time.shape[-1] == integration_sample_rate
        
        expanded_intensity_value_1 = expanded_intensity_value[..., :-1, :]     # [..., integration_sample_rate - 1, num_events]
        expanded_intensity_value_2 = expanded_intensity_value[..., 1:, :]      # [..., integration_sample_rate - 1, num_events]
        timestamp_for_integral = expanded_time.diff(dim = -1)                  # [..., integration_sample_rate - 1]

        # \int_{a}{b}{f(x)dx} = \sum_{i = 0}^{N - 2}{f(\frac{(b - a)i}{N - 1}) * \frac{(b - a)}{N - 1}}
        integral_of_all_events_1 = (expanded_intensity_value_1 * timestamp_for_integral.unsqueeze(dim = -1)).cumsum(dim = -2)
                                                                               # [..., integration_sample_rate - 1, num_events]
        # \int_{a}{b}{f(x)dx} = \sum_{i = 0}^{N - 2}{f(\frac{(b - a)(i + 1)}{N - 1}) * \frac{(b - a)}{N - 1}}
        integral_of_all_events_2 = (expanded_intensity_value_2 * timestamp_for_integral.unsqueeze(dim = -1)).cumsum(dim = -2)
                                                                               # [..., integration_sample_rate - 1, num_events]
        # Effectively increase the precision.
        integral_of_all_events = (integral_of_all_events_1 + integral_of_all_events_2) / 2
                                                                               # [..., integration_sample_rate - 1, num_events]
        
        # Prepend 0 to integral_of_all_events because \int_{t_l}^{t_l}{\lambda^*(\tau)d\tau} = 0
        # We have to check the shape.
        if len(integral_of_all_events.shape) == 5:
            integral_of_all_events, integral_of_all_events_ps = pack(
                (torch.zeros(*(integral_of_all_events).shape[:-2], 1, self.num_events, device = self.device), integral_of_all_events), 'b s ne1 * ne'
            )                                                                  # [..., integration_sample_rate, num_events]
        elif len(integral_of_all_events.shape) == 4:
            integral_of_all_events, integral_of_all_events_ps = pack(
                (torch.zeros(*(integral_of_all_events).shape[:-2], 1, self.num_events, device = self.device), integral_of_all_events), 'b s * ne'
            )                                                                  # [..., integration_sample_rate, num_events]
        
        return integral_of_all_events


    def integration_probability_estimator(self, expanded_probability_value, expanded_time, integration_sample_rate):
        # tensor check
        assert expanded_probability_value.shape[-2:] == (self.num_events, integration_sample_rate)
        assert expanded_time.shape[-1] == integration_sample_rate
        
        expanded_probability_value_1 = expanded_probability_value[..., :-1]    # [..., integration_sample_rate - 1]
        expanded_probability_value_2 = expanded_probability_value[..., 1:]     # [..., integration_sample_rate - 1]
        timestamp_for_integral = expanded_time.diff(dim = -1)                  # [..., integration_sample_rate - 1]

        # \int_{a}{b}{f(x)dx} = \sum_{i = 0}^{N - 2}{f(\frac{(b - a)i}{N - 1}) * \frac{(b - a)}{N - 1}}
        integral_of_all_events_1 = (expanded_probability_value_1 * timestamp_for_integral).cumsum(dim = -1)
                                                                               # [..., integration_sample_rate - 1]
        # \int_{a}{b}{f(x)dx} = \sum_{i = 0}^{N - 2}{f(\frac{(b - a)(i + 1)}{N - 1}) * \frac{(b - a)}{N - 1}}
        integral_of_all_events_2 = (expanded_probability_value_2 * timestamp_for_integral).cumsum(dim = -1)
                                                                               # [..., integration_sample_rate - 1]
        # Effectively increase the precision.
        integral_of_all_events = (integral_of_all_events_1 + integral_of_all_events_2) / 2
                                                                               # [..., integration_sample_rate - 1]
        
        # Prepend 0 to integral_of_all_events because \int_{t_l}^{t_l}{\lambda^*(\tau)d\tau} = 0
        # We have to check the shape.
        integral_of_all_events, integral_of_all_events_ps = pack(
            (torch.zeros(*(integral_of_all_events).shape[:-1], 1, device = self.device), integral_of_all_events), 'b s ne *'
        )                                                                      # [..., integration_sample_rate]

        return integral_of_all_events


    def forward(self, time_history, time_next, events_history, mask_history, custom_events_history = False, num_dimension_prior_batch = 0):
        history = self.history_encoder(time_history, events_history, mask_history, custom_events_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = time_next, num_dimension_prior_batch = num_dimension_prior_batch)
                                                                               # [..., batch_size, seq_len, d_input]
        # calculate the intensity.
        intensity_all_events = self.intensity_layer(hidden_state_at_t)         # [..., batch_size, seq_len, num_events]
        # calculate the integral
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [..., batch_size, seq_len, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time, num_dimension_prior_batch = num_dimension_prior_batch)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]
        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]

        integral_all_events = self.integration_estimator(expanded_intensity_all_events, \
                                                         expanded_time, self.integration_sample_rate)[:, :, -1, :]
                                                                               # [batch_size, seq_len, num_events]

        return integral_all_events, intensity_all_events


    def get_event_embedding(self, input_event):
        return self.history_encoder.get_event_embedding(input_event)           # [batch_size, seq_len, d_history]


    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, mask_history, integration_sample_rate):
        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time)
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]

        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]

        expanded_integral_all_events = self.integration_estimator(expanded_intensity_all_events, \
                                                                  expanded_time, integration_sample_rate)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]

        # Obtain timestamp
        timestamp, timestamp_ps = pack(
            (torch.zeros_like(time_next), expanded_time.diff(dim = -1)),
            'b s *'
        )                                                                      # [batch_size, seq_len, integration_sample_rate]

        return expanded_integral_all_events, expanded_intensity_all_events, timestamp


    def integral_intensity_time_next_3d(self, events_history, time_history, time_next, mask_history, integration_sample_rate):
        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, num_events, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time)
                                                                               # [batch_size, seq_len, num_events, integration_sample_rate, d_input]

        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, num_events, integration_sample_rate, num_events]

        expanded_integral_all_events = self.integration_estimator(expanded_intensity_all_events, expanded_time, integration_sample_rate)
                                                                               # [batch_size, seq_len, num_events, integration_sample_rate, num_events]

        # Obtain timestamp
        timestamp, timestamp_ps = pack(
            (torch.zeros_like(time_next), expanded_time.diff(dim = -1)),
            'b s ne *'
        )                                                                      # [batch_size, seq_len, num_events, integration_sample_rate]

        return expanded_integral_all_events, expanded_intensity_all_events, timestamp


    def model_probe_function(self, events_history, time_history, time_next, mask_history, mask_next, integration_sample_rate):
        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time)
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]

        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]

        expanded_integral_all_events = self.integration_estimator(expanded_intensity_all_events, expanded_time, integration_sample_rate)
                                                                               # [batch_size, seq_len, num_events, integration_sample_rate, num_events]

        # Obtain timestamp
        timestamp, timestamp_ps = pack(
            (torch.zeros_like(time_next), expanded_time.diff(dim = -1)),
            'b s *'
        )                                                                      # [batch_size, seq_len, integration_sample_rate]
        
        # construct the plot dict
        data = {}
        data['expand_intensity_for_each_event'] = expanded_intensity_all_events# [batch_size, seq_len, integration_sample_rate, num_events]
        data['expand_integral_for_each_event'] = expanded_integral_all_events  # [batch_size, seq_len, integration_sample_rate, num_events]

        # THP always assumes that the event information is present.
        # So model_probe_function() always provides spearman, pearson coefficient and L1 distance.

        expand_intensity = rearrange(expanded_intensity_all_events.detach().cpu(), 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * integration_sample_rate, num_event]
        expand_integral = rearrange(expanded_integral_all_events.detach().cpu(), 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * integration_sample_rate, num_event]
            
        spearman_matrix = []
        pearson_matrix = []
        L1_matrix = []
        for idx, (expand_intensity_per_seq, expand_integral_per_seq, mask_per_seq, time_next_per_seq) \
            in enumerate(zip(expand_intensity, expand_integral, mask_next, time_next)):
            seq_len = mask_per_seq.sum()

            probability_distribution = expand_intensity_per_seq * torch.exp(-expand_integral_per_seq)
            # rho: spearman coefficient
            spearman_matrix_per_seq = spearmanr(probability_distribution[:seq_len * integration_sample_rate])[0]
            if self.num_events == 2:
                spearman_matrix_per_seq = np.array([[1, spearman_matrix_per_seq], [spearman_matrix_per_seq, 1]])

            # r: pearson coefficient
            pearson_matrix_per_seq = np.corrcoef(probability_distribution[:seq_len * integration_sample_rate], rowvar = False)
            # L^1 metric
            L1_matrix_per_seq = L1_distance_across_events(probability_distribution[:seq_len * integration_sample_rate], 
                                            resolution = integration_sample_rate, num_events = self.num_events,
                                            time_next = time_next_per_seq[:seq_len])
            spearman_matrix.append(spearman_matrix_per_seq)
            pearson_matrix.append(pearson_matrix_per_seq)
            L1_matrix.append(L1_matrix_per_seq)

        data['spearman_matrix'] = spearman_matrix
        data['pearson_matrix'] = pearson_matrix
        data['L1_matrix'] = L1_matrix
        
        return data, timestamp
'''