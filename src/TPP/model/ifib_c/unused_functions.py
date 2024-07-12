'''
    def mean_absolute_error_per_event_worker(self, events_history, events_next, time_history, time_next, p_m, mask_next, mean, var):
        \'''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        \'''
        def evaluate_per_event(taus):
            # Train k FullyNN models for k different event types.
            if self.event_toggle:
                taus = repeat(taus, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            taus.requires_grad = True
            # \int_{t}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
            probability_integral_from_t_to_infinite = self.model(events_history, time_history, taus, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
    
            if self.event_toggle:
                events_next_index = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
                probability_integral_from_t_to_infinite = probability_integral_from_t_to_infinite * events_next_index
                                                                               # [batch_size, seq_len, num_events]
                probability_integral_from_t_to_infinite = reduce(probability_integral_from_t_to_infinite, 'b s ne -> b s', 'sum')
                                                                               # [batch_size, seq_len]

            probability_integral_from_zero_to_t = p_m - probability_integral_from_t_to_infinite

            return probability_integral_from_zero_to_t

        def bisect_target(taus):
            p_mt = evaluate_per_event(taus)                                    # [batch_size, seq_len]
            p_t_m = p_mt / p_m                                                 # [batch_size, seq_len]
            p_gap = p_t_m - self.probability_threshold                         # [batch_size, seq_len]

            return p_gap
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2

        max_ = min(1e6, mean + 10 * var)
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = max_*torch.ones_like(time_history, dtype = torch.float32)          # [batch_size, seq_len]

        tau_pred = median_prediction(l, r)
        gap = (tau_pred - time_next) * mask_next
        gap = torch.abs(gap)

        return gap
'''

'''
        p_m_predicted = reduce(probability_integral_from_zero_to_infinite * predict_index_one_hot, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        p_m_real = reduce(probability_integral_from_zero_to_infinite * events_next_one_hot, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predict_index, time_history, time_next,
                                                                               p_m_predicted, mask_next, mean, var)
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, time_next, 
                                                                  p_m_real, mask_next, mean, var)
'''

    '''
    def sample_one_event_from_model_event_time(self, number_of_sampled_sequences, events_history_for_sampling, \
                                               time_history_for_sampling, mean, std, mark_mask = None, output_p_m = False):
        \'''
        events_history_for_sampling: [batch_size, seq_len](batch_size is number_of_sampled_sequences when sample_event_time() calls this function.)
        time_history_for_sampling: [batch_size, seq_len]
        \'''
        def bisect_target_sample(taus, sample_input, integral_from_zero_to_inf):
            probability_integral_from_t_to_inf_for_sample = self.model.sample(events_history_for_sampling, time_history_for_sampling, taus, mean, std)
                                                                               # [...,batch_size, num_events]
            probability_integral_from_t_to_inf_for_sample = probability_integral_from_t_to_inf_for_sample.detach()
                                                                               # [..., number_of_sampled_sequences, batch_size, num_events]
            # P_m(t) = \int_{0}^{t}{p(t|m, \mathcal{H})}
            probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf_for_sample
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
            probability_integral = probability_integral / integral_from_zero_to_inf
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
            return probability_integral - sample_input


        batch_size, _ = events_history_for_sampling.shape
        time_next_zero = torch.zeros(number_of_sampled_sequences, batch_size, device = self.device)
                                                                               # [number_of_sampled_sequences, batch_size]
        time_next_zero = repeat(time_next_zero, 'nss b -> nss b ne', ne = self.num_events)
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
        integral_from_zero_to_inf = self.model.sample(events_history_for_sampling, time_history_for_sampling, time_next_zero, mean = mean, std = std)
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
        distribution_of_marks = torch.distributions.categorical.Categorical(integral_from_zero_to_inf)
        sampled_marks = distribution_of_marks.sample()                         # [number_of_sampled_sequences, batch_size]
        sampled_marks = sampled_marks.to(self.device)                          # [number_of_sampled_sequences, batch_size]

        sampled_threshold = torch.zeros((number_of_sampled_sequences, batch_size, self.num_events), device = self.device)
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
        torch.nn.init.uniform_(sampled_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
        tau_sampled = median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                        bisect_target_sample, sampled_threshold, integral_from_zero_to_inf)
        
        if mark_mask is not None:
            \'''
            Return all sampled timestamps of the selected marks if mark_mask is not None.
            \'''
            einop = f'... -> {"() " * (len(tau_sampled.shape) - len(mark_mask.shape))} ...'
            tau_mask = rearrange(mark_mask, einop)                             # [number_of_sampled_sequences, batch_size, num_events]
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
            tau_sampled = tau_sampled * tau_mask                               # [number_of_sampled_sequences, batch_size]
        else:
            tau_mask = torch.nn.functional.one_hot(sampled_marks, num_classes = self.num_events)
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
            tau_sampled = (tau_sampled * tau_mask).sum(dim = -1)               # [number_of_sampled_sequences, batch_size]

        if output_p_m:
            return tau_sampled, sampled_marks, integral_from_zero_to_inf
        else:
            return tau_sampled, sampled_marks
    '''

    '''
    def sample_one_event_from_model_time_event(self, number_of_sampled_sequences, events_history_for_sampling, time_history_for_sampling, mean, std):
        def bisect_target_sample(taus, sample_input, integral_from_zero_to_inf):
            taus = repeat(taus, '... -> ... ne', ne = self.num_events)         # [number_of_sampled_sequences, 1, num_events]
            probability_integral_from_t_to_inf_for_sample = self.model.sample(events_history_for_sampling, time_history_for_sampling, taus, mean, std)
                                                                               # [number_of_sampled_sequences, 1, num_events]
            probability_integral_from_t_to_inf_for_sample = probability_integral_from_t_to_inf_for_sample.detach()
                                                                               # [number_of_sampled_sequences, 1, num_events]
            # P_m(t) = \int_{0}^{t}{p(t|m, \mathcal{H})}
            probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf_for_sample
                                                                               # [number_of_sampled_sequences, 1, num_events]
            probability_integral = reduce(probability_integral, '... ne -> ...', 'sum')
                                                                               # [number_of_sampled_sequences, 1]
            return probability_integral - sample_input


        sampled_threshold = torch.zeros((number_of_sampled_sequences, 1), device = self.device)
                                                                               # [number_of_sampled_sequences, 1]
        torch.nn.init.uniform_(sampled_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [number_of_sampled_sequences, 1]
        time_next_zero = torch.zeros(number_of_sampled_sequences, 1, device = self.device)
                                                                               # [number_of_sampled_sequences, 1]
        time_next_zero = repeat(time_next_zero, 'b s -> b s ne', ne = self.num_events)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        integral_from_zero_to_inf = self.model.sample(events_history_for_sampling, time_history_for_sampling, time_next_zero, mean = mean, std = std)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        tau_sampled = median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                        bisect_target_sample, sampled_threshold, integral_from_zero_to_inf)
                                                                               # [number_of_sampled_sequences, 1]
        repeated_tau_sampled = repeat(tau_sampled, 'b s -> b s ne', ne = self.num_events)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        repeated_tau_sampled.requires_grad = True
        integral_from_sampled_time_to_inf = self.model('default_forward', events_history_for_sampling, time_history_for_sampling, repeated_tau_sampled, mean = mean, std = std)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        probability_for_each_event_at_pred_time = - torch.autograd.grad(
            outputs = integral_from_sampled_time_to_inf,
            inputs = repeated_tau_sampled,
            grad_outputs = torch.ones_like(integral_from_sampled_time_to_inf)
        )[0]                                                                   # [number_of_sampled_sequences, 1, num_events]
        repeated_tau_sampled.requires_grad = False

        distribution_of_marks = torch.distributions.categorical.Categorical(probability_for_each_event_at_pred_time)
        sampled_marks = distribution_of_marks.sample()                         # [number_of_sampled_sequences, 1]
        sampled_marks = sampled_marks.to(self.device)                          # [number_of_sampled_sequences, 1]

        return tau_sampled, sampled_marks
    '''