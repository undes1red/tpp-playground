import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
import numpy as np
from scipy.stats import spearmanr
import math

from src.toolbox.integration import approximate_integration
from src.TPP.model.utils import L1_distance_across_events, move_from_tensor_to_ndarray
from src.TPP.model.odetpp.model_factory import build_ode_model


class ODETPP(nn.Module):
    def __init__(self, device, num_events, hdim, cond_dim, hidden_dims, actfn, separate, \
                 tol, otreg_strength, ode_method, integration_sample_rate):
        super(ODETPP, self).__init__()
        self.num_events = num_events
        self.device = device
        self.integration_sample_rate = max(1, integration_sample_rate - 1)
        self.hidden_dims = hidden_dims
        self.ode_method = ode_method

        # event embedding layer.
        self.events_embedding = nn.Embedding(num_events + 1, cond_dim, padding_idx = num_events, device = device)

        # build the ode model.
        self.hidden_state_dynamics, self.ode_net \
            = build_ode_model(hdim = hdim, cond_dim = cond_dim, hidden_dims = hidden_dims, \
                              actfn = actfn, separate = separate, tol = tol, otreg_strength = otreg_strength)
        # RNN start state.
        self._init_state = nn.Parameter(torch.randn(hidden_dims[0]) / math.sqrt(hidden_dims[0]))

    
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
        gamma = rearrange(gamma, f'... d_i -> {"() " * num_dimension_prior_batch}... {"() " * (len(duration_t.shape) - 2 - num_dimension_prior_batch)}d_i')
                                                                               # [..., batch_size, seq_len, (integration_sample_rate, num_events), d_input]

        duration_t = duration_t.unsqueeze(dim = -1)                            # [..., batch_size, seq_len, (integration_sample_rate, num_events), 1]
        cell_t = torch.tanh(mu + (eta - mu) * torch.exp(-gamma * duration_t))  # [..., batch_size, seq_len, (integration_sample_rate, num_events), d_input]

        return cell_t
    '''

    def forward(self, task_name, *args, **kwargs):
        return getattr(self, task_name)(*args, **kwargs)


    def get_intensity_integral_train(self, time_next, events_next, mask_next, t0, t1, nlinspace = 1):
        batch_size, seq_len = time_next.shape[-2:]

        # NeuralODE accepts cumulative timestamps.
        time_next = time_next.cumsum(dim = -1)                                 # [batch_size, seq_len]
        time_next = time_next * mask_next                                      # [batch_size, seq_len]
        emb_events = self.events_embedding(events_next)                        # [batch_size, seq_len, hidden_dims]
        intensity_integral, intensity, _ = self.integrate_lambda(input_time = time_next, emb_events = emb_events, \
                                                                 input_mask = mask_next, t0 = t0, t1 = t1, nlinspace = nlinspace)
                                                                               # [batch_size, seq_len + 1] + [batch_size, seq_len]
        intensity_integral = intensity_integral.diff(dim = -1, prepend = torch.zeros(batch_size, 1, device = self.device))
                                                                               # [batch_size, seq_len + 1]
        intensity_integral, intensity_integral_escape = intensity_integral.split((seq_len, 1), dim = -1)
                                                                               # [batch_size, seq_len], [batch_size, 1]

        return intensity_integral, intensity_integral_escape, intensity


    def get_intensity(self, state):
        return self.ode_net.func.get_intensity(state)


    def integrate_lambda(self, input_time, emb_events, input_mask, t0, t1, nlinspace = 1):
        """
        Args:
            event_times: (N, T)
            spatial_location: (N, T, D) -> replace it with event embedding should work.
            input_mask: (N, T)
            t0: (N,) or (1,)
            t1: (N,) or (1,)
        """
        batch_size, seq_len = input_time.shape

        input_mask = input_mask.bool()                                         # [batch_size, seq_len]

        init_state = repeat(self._init_state, 'dim -> b dim', b = batch_size)  # [batch_size, hidden_dims[0]]
        state = (
            torch.zeros(batch_size, device = self.device),
            init_state,
        )                                                                      # ([batch_size] + [batch_size, hidden_dims[0]])

        t0 = torch.tensor(t0, dtype = torch.float32, device = self.device)
        t0 = repeat(t0, '... -> b ...', b = batch_size)                        # [batch_size]

        self.ode_net.nfe = 0
        intensity = []
        intensity_integral = []
        prejump_hidden_states = []
        for i in range(seq_len):
            # Set t1 = t0 if the input is masked out at time t1.
            t1_i = torch.where(input_mask[:, i], input_time[:, i], t0)         # [batch_size]
            state_traj = self.ode_net.integrate(t0, t1_i, state, nlinspace = nlinspace, method = self.ode_method)
                                                                               # ([1 + nlinspace, batch_size] + [1 + nlinspace, batch_size, hidden_dims[0]])
            integrals = state_traj[0]                                          # [1 + nlinspace, batch_size]
            hiddens = state_traj[1]                                            # [1 + nlinspace, batch_size, hidden_dims[0]]
            if i > 0:
                hiddens = hiddens[1:]
            # set hidden states to zero if input is masked out at the next time step.
            hiddens = torch.where(input_mask[:, i].reshape(1, -1, 1).expand_as(hiddens), hiddens, torch.zeros_like(hiddens))
                                                                               # [1 + nlinspace, batch_size, hidden_dims[0]]
            prejump_hidden_states.append(hiddens)
            intensity_integral.append(integrals[-1])                           # [batch_size]

            time_interval_end_state = tuple(s[-1] for s in state_traj)         # ([batch_size] + [batch_size, hidden_dims[0]])
            intensity_integral_t0_t_i, tpp_state = time_interval_end_state     # [batch_size] + [batch_size, hidden_dims[0]]
            intensity.append(self.get_intensity(tpp_state).squeeze())          # [batch_size]

            if i < seq_len - 1 or t1 is not None:
                cond = emb_events[:, i] if emb_events is not None else None    # [batch_size, hidden_dims[0]]
                # tpp hidden state with the latest observed event.
                updated_tpp_state = self.hidden_state_dynamics.update_state(input_time[:, i], tpp_state, cond = cond)
                                                                               # [batch_size, hidden_dims[0]]
                tpp_state = torch.where(input_mask[:, i].reshape(-1, 1).expand_as(tpp_state), updated_tpp_state, tpp_state)
                                                                               # [batch_size, hidden_dims[0]]
                state = (intensity_integral_t0_t_i, tpp_state)

            # Track t0 as the last valid event time.
            t0 = torch.where(input_mask[:, i], input_time[:, i], t0)

        if t1 is not None:
            # Integrate from last time sample to t1.
            t1 = torch.tensor(t1, dtype = torch.float32, device = self.device)
            t1 = repeat(t1, '... -> b ...', b = batch_size)                    # [batch_size]
            state_traj = self.ode_net.integrate(t0, t1, state, nlinspace = nlinspace, method = self.ode_method)
                                                                               # ([1 + nlinspace, batch_size] + [1 + nlinspace, batch_size, hidden_dims[0]])
            hiddens = state_traj[1][1:]                                        # [nlinspace, batch_size, hidden_dims[0]]
            prejump_hidden_states.append(hiddens)

            state = tuple(s[-1] for s in state_traj)                           # ([batch_size] + [batch_size, hidden_dims[0]])

        intensity_integral_escape, _ = state                                   # [batch_size]
        intensity_integral.append(intensity_integral_escape)

        intensity_integral = torch.stack(intensity_integral, dim = -1)         # [batch_size, seq_len]
        intensity = torch.stack(intensity, dim = -1)                           # [batch_size, seq_len]
        prejump_hidden_states = torch.cat(prejump_hidden_states, dim = 0).transpose(0, 1)
                                                                               # [batch_size, (seq_len + 2) * nlinspace, hidden_dims[0]]
        return intensity_integral, intensity, prejump_hidden_states


    def integrate_lambda_from_t_l_to_t(self, input_time, tau, events_next, input_mask, t0, t1, nlinspace = 1):
        """
        Args:
            event_times: (N, T)
            spatial_location: (N, T, D) -> replace it with event embedding should work.
            input_mask: (N, T)
            t0: (N,) or (1,)
            t1: (N,) or (1,)
        """
        batch_size, seq_len = input_time.shape[-2:]

        input_mask = input_mask.bool()                                         # [batch_size, seq_len]
        emb_events = self.events_embedding(events_next)                        # [batch_size, seq_len, hidden_dims]

        init_state = repeat(self._init_state, 'dim -> b dim', b = batch_size)  # [batch_size, hidden_dims[0]]
        state = (
            torch.zeros(batch_size, device = self.device),
            init_state,
        )                                                                      # ([batch_size] + [batch_size, hidden_dims[0]])

        t0 = torch.tensor(t0, dtype = torch.float32, device = self.device)
        t0 = repeat(t0, '... -> b ...', b = batch_size)                        # [batch_size]

        self.ode_net.nfe = 0
        intensity = []
        intensity_integral = []
        prejump_hidden_states = []
        for i in range(seq_len):
            # Set t1 = t0 if the input is masked out at time t1.
            t1_i = torch.where(input_mask[:, i], input_time[:, i], t0)         # [batch_size]
            state_traj = self.ode_net.integrate(t0, t1_i, state, nlinspace = nlinspace, method = self.ode_method)
                                                                               # ([1 + nlinspace, batch_size] + [1 + nlinspace, batch_size, hidden_dims[0]])
            integrals = state_traj[0]                                          # [1 + nlinspace, batch_size]
            hiddens = state_traj[1]                                            # [1 + nlinspace, batch_size, hidden_dims[0]]
            if i > 0:
                hiddens = hiddens[1:]
            # set hidden states to zero if input is masked out at the next time step.
            hiddens = torch.where(input_mask[:, i].reshape(1, -1, 1).expand_as(hiddens), hiddens, torch.zeros_like(hiddens))
                                                                               # [1 + nlinspace, batch_size, hidden_dims[0]]
            prejump_hidden_states.append(hiddens)
            intensity_integral.append(integrals[-1])                           # [batch_size]

            time_interval_end_state = tuple(s[-1] for s in state_traj)         # ([batch_size] + [batch_size, hidden_dims[0]])
            intensity_integral_t0_t_i, tpp_state = time_interval_end_state     # [batch_size] + [batch_size, hidden_dims[0]]
            intensity.append(self.get_intensity(tpp_state).squeeze())          # [batch_size]

            if i < seq_len - 1 or t1 is not None:
                cond = emb_events[:, i] if emb_events is not None else None    # [batch_size, hidden_dims[0]]
                # tpp hidden state with the latest observed event.
                updated_tpp_state = self.hidden_state_dynamics.update_state(input_time[:, i], tpp_state, cond = cond)
                                                                               # [batch_size, hidden_dims[0]]
                tpp_state = torch.where(input_mask[:, i].reshape(-1, 1).expand_as(tpp_state), updated_tpp_state, tpp_state)
                                                                               # [batch_size, hidden_dims[0]]
                state = (intensity_integral_t0_t_i, tpp_state)

            # Track t0 as the last valid event time.
            t0 = torch.where(input_mask[:, i], input_time[:, i], t0)

        if t1 is not None:
            # Integrate from last time sample to t1.
            t1 = torch.tensor(t1, dtype = torch.float32, device = self.device)
            t1 = repeat(t1, '... -> b ...', b = batch_size)                    # [batch_size]
            state_traj = self.ode_net.integrate(t0, t1, state, nlinspace = nlinspace, method = self.ode_method)
                                                                               # ([1 + nlinspace, batch_size] + [1 + nlinspace, batch_size, hidden_dims[0]])
            hiddens = state_traj[1][1:]                                        # [nlinspace, batch_size, hidden_dims[0]]
            prejump_hidden_states.append(hiddens)

            state = tuple(s[-1] for s in state_traj)                           # ([batch_size] + [batch_size, hidden_dims[0]])

        intensity_integral_escape, _ = state                                   # [batch_size]
        intensity_integral.append(intensity_integral_escape)

        intensity_integral = torch.stack(intensity_integral, dim = -1)         # [batch_size, seq_len]
        intensity = torch.stack(intensity, dim = -1)                           # [batch_size, seq_len]
        prejump_hidden_states = torch.cat(prejump_hidden_states, dim = 0).transpose(0, 1)
                                                                               # [batch_size, (seq_len + 2) * nlinspace, hidden_dims[0]]
        return intensity_integral, intensity, prejump_hidden_states
    
    '''
    def forward(self, time_next, events_next, mask_next):
        \'''
        We only need the representation of the first event.
        As the embedding layer assigns zero embedding to padding events, we can safely initiate the current state as a zero vector.
        \'''
        batch_size = time_next.shape[0]

        emb_events_next = self.events_embedding(events_next)                   # [batch_size, seq_len + 1, d_mark_embedding]
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate)   # [resolution]

        hidden_states_at_start_of_interval = []
        hidden_states_at_end_of_interval = []
        current_state = torch.zeros(batch_size, self.d_hidden, device = self.device)
                                                                               # [batch_size, d_hidden]

        for emb_events, time_gap in zip(torch.unbind(emb_events_next, dim = -2), torch.unbind(time_next, dim = -1)):
            # hidden state at the start of an interval.
            hidden_states_at_start_of_interval.append(current_state)
            # hidden state at the end of an interval, but right before the next event.
            current_state = odeint(self.state_transfer_model, current_state, time_gap)
                                                                               # [batch_size, d_hidden]
            hidden_states_at_end_of_interval.append(current_state)
            # hidden state right after the next event happened.
            current_state = current_state + emb_events                         # [batch_size, d_hidden]

            # calculate the decayed hidden states at all time steps.
            expanded_time_next = time_gap.unsqueeze(dim = -1) * time_multiplier
                                                                               # [batch_size, seq_len + 1, resolution]

        # hidden state at $t_{i-1}^{+}$.
        hidden_states_at_start_of_interval = torch.cat(hidden_states_at_start_of_interval, dim = -2)
                                                                               # [batch_size, seq_len + 1, d_hidden]
        # hidden state at $t_i^{-}$.
        hidden_states_at_end_of_interval = torch.cat(hidden_states_at_end_of_interval, dim = -2)
                                                                               # [batch_size, seq_len + 1, d_hidden]
        expanded_hidden_states = odeint(self.state_transfer_model, \
                                        hidden_states_at_start_of_interval.unsqueeze(dim = -2), \
                                        expanded_time_next.unsqueeze(dim = -1))# [batch_size, seq_len + 1, resolution, d_hidden]


        return intensity, intensity_integral
    '''

    '''
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
        
        integral_all_events = approximate_integration(expanded_intensity_all_events, \
                                                      expanded_time, dim = -2, only_integral = True)
                                                                               # [..., batch_size, seq_len, num_events]
        
        return integral_all_events, intensity_all_events
    '''


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
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time, num_dimension_prior_batch = 0)
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]

        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
        expanded_integral_all_events = approximate_integration(expanded_intensity_all_events, expanded_time, dim = -2)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
        
        return expanded_integral_all_events, expanded_intensity_all_events, expanded_time


    def integral_intensity_time_next_3d(self, events_history, time_history, time_next, mask_history, integration_sample_rate, num_dimension_prior_batch = 0):
        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
                                                                               # [integration_sample_rate]
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [..., batch_size, seq_len, num_events, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time, num_dimension_prior_batch = num_dimension_prior_batch)
                                                                               # [..., batch_size, seq_len, num_events, integration_sample_rate, d_input]
        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [..., batch_size, seq_len, num_events, integration_sample_rate, num_events]
        expanded_integral_all_events = approximate_integration(expanded_intensity_all_events, \
                                                                   expanded_time, dim = -2)
                                                                               # [..., batch_size, seq_len, num_events, integration_sample_rate, num_events]

        return expanded_integral_all_events, expanded_intensity_all_events, expanded_time


    def model_probe_function(self, events_history, time_history, time_next, mask_history, mask_next, integration_sample_rate):
        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time, num_dimension_prior_batch = 0)
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]

        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
        expanded_integral_all_events = approximate_integration(expanded_intensity_all_events, expanded_time, dim = -2)
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

        expand_intensity = rearrange(expanded_intensity_all_events, 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * integration_sample_rate, num_event]
        expand_integral = rearrange(expanded_integral_all_events, 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * integration_sample_rate, num_event]
            
        spearman_matrix = []
        pearson_matrix = []
        L1_matrix = []
        for idx, (expand_intensity_per_seq, expand_integral_per_seq, mask_per_seq, time_next_per_seq) \
            in enumerate(zip(expand_intensity, expand_integral, mask_next, time_next)):
            seq_len = mask_per_seq.sum()
            probability_distribution = expand_intensity_per_seq * torch.exp(-expand_integral_per_seq)
            probability_distribution = move_from_tensor_to_ndarray(probability_distribution)

            # rho: spearman coefficient
            if self.num_events == 1:
                spearman_matrix_per_seq = np.array([[1.,],])
            else:
                spearman_matrix_per_seq = spearmanr(probability_distribution[:seq_len * integration_sample_rate])[0]
                if self.num_events == 2:
                    spearman_matrix_per_seq = np.array([[1, spearman_matrix_per_seq], [spearman_matrix_per_seq, 1]])

            # r: pearson coefficient
            pearson_matrix_per_seq = np.corrcoef(probability_distribution[:seq_len * integration_sample_rate], rowvar = False)
            if self.num_events == 1:
                pearson_matrix_per_seq = rearrange(np.array(pearson_matrix_per_seq), ' -> () ()')
            
            # L^1 metric
            L1_matrix_per_seq = L1_distance_across_events(probability_distribution[:seq_len * integration_sample_rate], 
                                                          time_next = time_next_per_seq[:seq_len], has_flatten = True)
            spearman_matrix.append(spearman_matrix_per_seq)
            pearson_matrix.append(pearson_matrix_per_seq)
            L1_matrix.append(L1_matrix_per_seq)

        data['spearman_matrix'] = spearman_matrix
        data['pearson_matrix'] = pearson_matrix
        data['L1_matrix'] = L1_matrix
        
        return data, timestamp