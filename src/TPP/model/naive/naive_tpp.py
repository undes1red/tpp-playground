import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, reduce, rearrange, pack, unpack


class Poisson(nn.Module):
    def __init__(self, num_events, device):
        super(Poisson, self).__init__()

        self.device = device
        self.num_events = num_events

        '''
        \\the learned intensity function
        '''
        self.intensities = nn.Parameter(torch.ones(num_events, device = self.device))
        torch.nn.init.normal_(self.intensities)


    def forward(self, events_history, time_history, time_next):
        # Get the integral.
        integral = time_next.unsqueeze(dim = -1) * F.softplus(self.intensities)# [..., batch_size, seq_len, num_events]

        # Get the intensity
        intensity = torch.ones_like(integral) * F.softplus(self.intensities)   # [..., batch_size, seq_len, num_events]
        
        return integral, intensity


    def forward_time_next_2d(self, events_history, time_history, time_next, integration_sample_rate):
        # confirm that the last dimension is integration_sample_rate
        assert time_next.shape[-1] == integration_sample_rate

        # Get the integral.
        integral = time_next.unsqueeze(dim = -1) * F.softplus(self.intensities)# [..., batch_size, seq_len, integration_sample_rate, num_events]

        # Get the intensity
        intensity = torch.ones_like(integral) * F.softplus(self.intensities)   # [..., batch_size, seq_len, integration_sample_rate, num_events]
        
        return integral, intensity
    

    def forward_time_next_3d(self, events_history, time_history, time_next, integration_sample_rate):
        # confirm that the last dimension is integration_sample_rate
        assert time_next.shape[-1] == integration_sample_rate

        # Get the integral.
        integral = time_next.unsqueeze(dim = -1) * F.softplus(self.intensities)# [..., batch_size, seq_len, num_events, integration_sample_rate, num_events]

        # Get the intensity
        intensity = torch.ones_like(integral) * F.softplus(self.intensities)   # [..., batch_size, seq_len, num_events, integration_sample_rate, num_events]
        
        return integral, intensity


class Hawkes(nn.Module):
    def __init__(self, num_events, device):
        '''
        Hawkes process:
        \\lambda(m, t) = \\mu + \\sum_{e = (m_i, t_i) \\in \\history}{a_{m_i, m} * b_m * exp(-b_m(t - t_i))}.
        \\Lambda(t) = \\mu * (t - t_l) + \\sum_{e = (m_i, t_i) \\in \\history}{a_{m_i, m} - a_{m_i, m} * exp(-b_m(t - t_l))}. When t = t_l, \\Lambda(t) = 0.
        '''
        super(Hawkes, self).__init__()
        self.device = device
        self.num_events = num_events

        # \\mu
        self.base_intensity = nn.Parameter(torch.zeros(num_events, device = self.device))

        # \\alpha matrix
        # \\alpha_{ij} is the effect of event i to event j.
        self.transition_matrix = nn.Parameter(torch.ones(num_events, num_events, device = self.device))

        # \\beta
        self.time_scaling_factors = nn.Parameter(torch.ones(num_events, device = self.device))

        # Parameter initialization.
        torch.nn.init.normal_(self.base_intensity)
        torch.nn.init.xavier_normal_(self.transition_matrix)
        torch.nn.init.normal_(self.time_scaling_factors)
        

    def forward(self, events_history, time_history, time_next):
        batch_size, seq_len = events_history.shape

        '''
        Forward function of the hawkes process.
        '''
        base_intensity = F.softplus(self.base_intensity)
        transition_matrix = F.softplus(self.transition_matrix).T
        time_scaling_factors = F.softplus(self.time_scaling_factors)

        # calculating t_i - t_j.
        absolute_history_offset = torch.cumsum(time_history, dim = -1)         # [batch_size, seq_len]
        einop = f'... -> {"() " * (len(absolute_history_offset.shape) - 2)}...'
        absolute_history_offset = rearrange(absolute_history_offset, einop)    # [..., batch_size, seq_len]
        absolute_next_time = absolute_history_offset + time_next               # [..., batch_size, seq_len]
        A = absolute_history_offset.unsqueeze(dim = -2)                        # [..., batch_size, 1, seq_len]
        B = absolute_next_time.unsqueeze(dim = -1)                             # [..., batch_size, seq_len, 1]
        time_interval_matrix = (B - A).unsqueeze(dim = -2)                     # [..., batch_size, seq_len, 1, seq_len]

        # Additionally, we need the time_interval matrix across events in events_history for calculating exp(-b_m(t_l - t_i)).
        # This is mandatory as the time_next can be anytime.
        absolute_history_offset = torch.cumsum(time_history, dim = -1)         # [batch_size, seq_len]
        time_interval_matrix_from_t_l_to_t_i = absolute_history_offset.unsqueeze(dim = -1) - absolute_history_offset.unsqueeze(dim = -2)
                                                                               # [batch_size, seq_len, seq_len]
        einop = f'b s1 s2 -> {"() " * (len(time_interval_matrix.shape) - 4)}b s1 () s2'
        time_interval_matrix_from_t_l_to_t_i = rearrange(time_interval_matrix_from_t_l_to_t_i, einop)
                                                                               # [..., batch_size, seq_len, 1, seq_len]

        # We replace all negative values in time_interal_matrix with a fixed value as some of them might introduce infinity to exp_b_m_t.
        # This action is safe because these negative values are t_i - t_j with i < j, while intensity and integral calculation only counts t_i - t_j with i >= j.
        time_interval_matrix = time_interval_matrix.clamp(min = -1)            # [..., batch_size, seq_len, 1, seq_len]
        time_interval_matrix_from_t_l_to_t_i = time_interval_matrix_from_t_l_to_t_i.clamp(min = -1)
                                                                               # [..., batch_size, seq_len, 1, seq_len]


        event_history_without_dummy_start = events_history[..., 1:]            # [batch_size, seq_len - 1]
        if event_history_without_dummy_start.numel() == 0:
            return self.base_intensity * time_next.unsqueeze(dim = -1), self.base_intensity * torch.ones_like(time_next.unsqueeze(dim = -1))
                                                                               # [..., batch_size, seq_len, num_events] * 2

        # gathering \\alpha_{m_i, m}
        gather_index = repeat(event_history_without_dummy_start, 'b s -> b ne s', ne = self.num_events)
                                                                               # [batch_size, num_events, seq_len - 1]
        alpha = torch.gather(repeat(transition_matrix, '... -> b ...', b = batch_size), -1, gather_index)
                                                                               # [batch_size, num_events, seq_len - 1]
        
        alpha = torch.concat([torch.zeros(*alpha.shape[:-1], 1, device = self.device), alpha], dim = -1).unsqueeze(dim = -3)
                                                                               # [batch_size, 1, num_events, seq_len]
        # exp(-b_m(t_i - t_j))
        exp_b_m_t = torch.exp(-time_scaling_factors.unsqueeze(dim = -1) * time_interval_matrix)
                                                                               # [..., batch_size, seq_len, num_events, seq_len]
        # exp(-b_m(t_l - t_i))
        exp_b_m_t_l = torch.exp(-time_scaling_factors.unsqueeze(dim = -1) * time_interval_matrix_from_t_l_to_t_i)
                                                                               # [..., batch_size, seq_len, num_events, seq_len]

        # calculating the intensity function.
        # \\lambda(m, t) = \\mu + \sum_{e = (m_i, t_i) \\in \\history}{a_{m_i, m} * b_m * exp(-b_m(t - t_i))}.
        history_influence = alpha * time_scaling_factors.unsqueeze(dim = -1) * exp_b_m_t
                                                                               # [..., batch_size, seq_len, num_events, seq_len]
        history_influence_mask = torch.triu(torch.ones(seq_len, seq_len, device = self.device), diagonal = 0).T.unsqueeze(dim = -2)
                                                                               # [seq_len, 1, seq_len]
        history_part = (history_influence * history_influence_mask).sum(dim = -1)
                                                                               # [..., batch_size, seq_len, num_events]
        
        einop = f'... -> {"() " * (len(history_part.shape) - 1)}...'
        base_intensity = rearrange(base_intensity, einop)                      # [..., batch_size, seq_len, num_events]
        intensity = base_intensity + history_part                              # [..., batch_size, seq_len, num_events]

        # calculating the integral of intensity function.
        # \\Lambda(t) = \\mu * (t - t_l) + \\sum_{e = (m_i, t_i) \\in \\history}{a_{m_i, m} * (exp(-b_m(t_l - t_i)) - exp(-b_m(t - t_i)))}. When t = t_l, \\Lambda(t) = 0.
        interval = exp_b_m_t_l - exp_b_m_t                                     # [..., batch_size, seq_len, num_events, seq_len]
        interval = (interval * alpha) * history_influence_mask                 # [..., batch_size, seq_len, num_events, seq_len]

        interval = interval.sum(dim = -1)                                      # [..., batch_size, seq_len, num_events]
        base_intensity_integral = time_next.unsqueeze(dim = -1) * base_intensity
                                                                               # [..., batch_size, seq_len, num_events]
        integral = interval + base_intensity_integral                          # [..., batch_size, seq_len, num_events]

        return integral, intensity


    def forward_time_next_2d(self, events_history, time_history, time_next, integration_sample_rate):
        # confirm that the last dimension is integration_sample_rate
        assert time_next.shape[-1] == integration_sample_rate
        # shape of time_next: [batch_size, seq_len, integration_sample_rate]

        batch_size, seq_len = events_history.shape

        '''
        Forward function of the hawkes process.
        '''
        base_intensity = F.softplus(self.base_intensity)
        transition_matrix = F.softplus(self.transition_matrix).T
        time_scaling_factors = F.softplus(self.time_scaling_factors)

        # calculating t_i - t_j.
        absolute_history_time = torch.cumsum(time_history, dim = -1)           # [batch_size, seq_len]
        time_next_offset_to_abs_time, _ = pack((torch.zeros(batch_size, 1, device = self.device), time_next[:, :-1, -1]), 'b *')
                                                                               # [batch_size, seq_len]
        time_next_offset_to_abs_time = time_next_offset_to_abs_time.cumsum(dim = -1).unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, 1]
        absolute_next_time = time_next_offset_to_abs_time + time_next          # [batch_size, seq_len, integration_sample_rate]
        A = rearrange(absolute_history_time, 'b s -> b () () s')               # [batch_size, 1, 1, seq_len]
        B = absolute_next_time.unsqueeze(dim = -1)                             # [batch_size, seq_len, integration_sample_rate, 1]
        time_interval_matrix = (B - A).unsqueeze(dim = -2)                     # [batch_size, seq_len, integration_sample_rate, 1, seq_len]
        # We replace all negative values in time_interal_matrix with a fixed value as some of them might introduce infinity to exp_b_m_t.
        # This action is safe because these negative values are t_i - t_j with i < j, while intensity and integral calculation only counts t_i - t_j with i >= j.
        time_interval_matrix = time_interval_matrix.clamp(min = -1)            # [batch_size, seq_len, integration_sample_rate, 1, seq_len]

        event_history_without_dummy_start = events_history[..., 1:]            # [batch_size, seq_len - 1]
        if event_history_without_dummy_start.numel() == 0:
            return self.base_intensity * time_next.unsqueeze(dim = -1), self.base_intensity * torch.ones_like(time_next.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events] * 2

        # gathering \\alpha_{m_i, m}
        gather_index = repeat(event_history_without_dummy_start, 'b s -> b ne s', ne = self.num_events)
                                                                               # [batch_size, num_events, seq_len - 1]
        alpha = torch.gather(repeat(transition_matrix, '... -> b ...', b = batch_size), -1, gather_index)
                                                                               # [batch_size, num_events, seq_len - 1]
        alpha = torch.concat([torch.zeros(*alpha.shape[:-1], 1, device = self.device), alpha], dim = -1)
                                                                               # [batch_size, num_events, seq_len]
        alpha = rearrange(alpha, 'b ne sl -> b () () ne sl')                   # [batch_size, 1, 1, num_events, seq_len]
        # exp(-b_m(t_i - t_j))
        exp_b_m_t = torch.exp(-time_scaling_factors.unsqueeze(dim = -1) * time_interval_matrix)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events, seq_len]
        # calculating the intensity function.
        # \\lambda(m, t) = \\mu + \sum_{e = (m_i, t_i) \\in \\history}{a_{m_i, m} * b_m * exp(-b_m(t - t_i))}.
        history_influence = alpha * time_scaling_factors.unsqueeze(dim = -1) * exp_b_m_t
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events, seq_len]
        history_influence_mask = torch.triu(torch.ones(seq_len, seq_len, device = self.device), diagonal = 0).T
                                                                               # [seq_len, seq_len]
        history_influence_mask = rearrange(history_influence_mask, 'sl1 sl2 -> sl1 () () sl2')
                                                                               # [seq_len, 1, 1, seq_len]
        history_part = (history_influence * history_influence_mask).sum(dim = -1)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
        
        einop = f'... -> {"() " * (len(history_part.shape) - 1)}...'
        base_intensity = rearrange(base_intensity, einop)                      # [batch_size, seq_len, integration_sample_rate, num_events]
        intensity = base_intensity + history_part                              # [batch_size, seq_len, integration_sample_rate, num_events]

        # calculating the integral of intensity function.
        # \\Lambda(t) = \\mu * (t - t_l) + \\sum_{e = (m_i, t_i) \\in \\history}{a_{m_i, m} * (exp(-b_m(t_l - t_i)) - exp(-b_m(t - t_i)))}.
        # When t = t_l, \\Lambda(t) = 0.
        interval = exp_b_m_t[..., 0:1, :, :] - exp_b_m_t                       # [batch_size, seq_len, integration_sample_rate, num_events, seq_len]
        interval = (interval * alpha) * history_influence_mask                 # [batch_size, seq_len, integration_sample_rate, num_events, seq_len]

        interval = interval.sum(dim = -1)                                      # [batch_size, seq_len, integration_sample_rate, num_events]
        base_intensity_integral = time_next.unsqueeze(dim = -1) * base_intensity
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
        integral = interval + base_intensity_integral                          # [batch_size, seq_len, integration_sample_rate, num_events]

        return integral, intensity
    

    def forward_time_next_3d(self, events_history, time_history, time_next, integration_sample_rate):
        # confirm that the last dimension is integration_sample_rate
        assert time_next.shape[-1] == integration_sample_rate
        # shape of time_next: [batch_size, seq_len, num_events, integration_sample_rate]

        batch_size, seq_len = events_history.shape

        '''
        Forward function of the hawkes process.
        '''
        base_intensity = F.softplus(self.base_intensity)
        transition_matrix = F.softplus(self.transition_matrix).T
        time_scaling_factors = F.softplus(self.time_scaling_factors)

        # calculating t_i - t_j.
        absolute_history_time = torch.cumsum(time_history, dim = -1)           # [batch_size, seq_len]
        time_next_offset_to_abs_time, _ = pack((torch.zeros(batch_size, 1, self.num_events, device = self.device), time_next[:, :-1, :, -1]), 'b * ne')
                                                                               # [batch_size, seq_len, num_events]
        time_next_offset_to_abs_time = time_next_offset_to_abs_time.cumsum(dim = -2).unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, num_events, 1]
        absolute_next_time = time_next_offset_to_abs_time + time_next          # [batch_size, seq_len, num_events, integration_sample_rate]
        A = rearrange(absolute_history_time, 'b s -> b () () () s')            # [batch_size, 1, 1, 1, seq_len]
        B = absolute_next_time.unsqueeze(dim = -1)                             # [batch_size, seq_len, num_events, integration_sample_rate, 1]
        time_interval_matrix = (B - A).unsqueeze(dim = -2)                     # [batch_size, seq_len, num_events, integration_sample_rate, 1, seq_len]
        # We replace all negative values in time_interal_matrix with a fixed value as some of them might introduce infinity to exp_b_m_t.
        # This action is safe because these negative values are t_i - t_j with i < j, while intensity and integral calculation only counts t_i - t_j with i >= j.
        time_interval_matrix = time_interval_matrix.clamp(min = -1)            # [batch_size, seq_len, num_events, integration_sample_rate, 1, seq_len]

        event_history_without_dummy_start = events_history[..., 1:]            # [batch_size, seq_len - 1]
        if event_history_without_dummy_start.numel() == 0:
            return self.base_intensity * time_next.unsqueeze(dim = -1), self.base_intensity * torch.ones_like(time_next.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, num_events, integration_sample_rate, num_events] * 2

        # gathering \\alpha_{m_i, m}
        gather_index = repeat(event_history_without_dummy_start, 'b s -> b ne s', ne = self.num_events)
                                                                               # [batch_size, num_events, seq_len - 1]
        alpha = torch.gather(repeat(transition_matrix, '... -> b ...', b = batch_size), -1, gather_index)
                                                                               # [batch_size, num_events, seq_len - 1]
        alpha = torch.concat([torch.zeros(*alpha.shape[:-1], 1, device = self.device), alpha], dim = -1)
                                                                               # [batch_size, num_events, seq_len]
        alpha = rearrange(alpha, 'b ne sl -> b () () () ne sl')                # [batch_size, 1, 1, num_events, seq_len]
        # exp(-b_m(t_i - t_j))
        exp_b_m_t = torch.exp(-time_scaling_factors.unsqueeze(dim = -1) * time_interval_matrix)
                                                                               # [batch_size, seq_len, num_events, integration_sample_rate, num_events, seq_len]
        # calculating the intensity function.
        # \\lambda(m, t) = \\mu + \sum_{e = (m_i, t_i) \\in \\history}{a_{m_i, m} * b_m * exp(-b_m(t - t_i))}.
        history_influence = alpha * time_scaling_factors.unsqueeze(dim = -1) * exp_b_m_t
                                                                               # [batch_size, seq_len, num_events, integration_sample_rate, num_events, seq_len]
        history_influence_mask = torch.triu(torch.ones(seq_len, seq_len, device = self.device), diagonal = 0).T
                                                                               # [seq_len, seq_len]
        history_influence_mask = rearrange(history_influence_mask, 'sl1 sl2 -> sl1 () () () sl2')
                                                                               # [seq_len, 1, 1, 1, seq_len]
        history_part = (history_influence * history_influence_mask).sum(dim = -1)
                                                                               # [batch_size, seq_len, num_events, integration_sample_rate, num_events]
        
        einop = f'... -> {"() " * (len(history_part.shape) - 1)}...'
        base_intensity = rearrange(base_intensity, einop)                      # [batch_size, seq_len, num_events, integration_sample_rate, num_events]
        intensity = base_intensity + history_part                              # [batch_size, seq_len, num_events, integration_sample_rate, num_events]

        # calculating the integral of intensity function.
        # \\Lambda(t) = \\mu * (t - t_l) + \\sum_{e = (m_i, t_i) \\in \\history}{a_{m_i, m} * (exp(-b_m(t_l - t_i)) - exp(-b_m(t - t_i)))}.
        # When t = t_l, \\Lambda(t) = 0.
        interval = exp_b_m_t[..., 0:1, :, :] - exp_b_m_t                       # [batch_size, seq_len, num_events, integration_sample_rate, num_events, seq_len]
        interval = (interval * alpha) * history_influence_mask                 # [batch_size, seq_len, num_events, integration_sample_rate, num_events, seq_len]

        interval = interval.sum(dim = -1)                                      # [batch_size, seq_len, num_events, integration_sample_rate, num_events]
        base_intensity_integral = time_next.unsqueeze(dim = -1) * base_intensity
                                                                               # [batch_size, seq_len, num_events, integration_sample_rate, num_events]
        integral = interval + base_intensity_integral                          # [batch_size, seq_len, num_events, integration_sample_rate, num_events]

        return integral, intensity