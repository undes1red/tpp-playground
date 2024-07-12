'''
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predicted_events, \
                                                                               time_history, time_next, mask_history, \
                                                                               mask_next, probability, resolution, \
                                                                               mean, var, max_)
                                                                               # [batch_size, seq_len]
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, \
                                                                  time_history, time_next, mask_history, \
                                                                  mask_next, probability, resolution, \
                                                                  mean, var, max_)
                                                                               # [batch_size, seq_len]
'''

'''

    def mean_absolute_error_per_event_worker(self, events_history, events_next, 
        time_history, time_next, mask_history, mask_next, probability_integral, resolution, mean, var, max_val):
        \'''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        \'''
        def evaluate_per_event(taus):
            expanded_integral_all_events, expanded_intensity_all_events, timestamp = \
                self.model.integral_intensity_time_next_2d(events_history, time_history, taus, mask_history, \
                                                           resolution, mean, var)
                                                                               # 2 * [batch_size, seq_len, resolution, num_events] + [batch_size, seq_len, resolution]
            events_next_mask = F.one_hot(events_next.long(), num_classes = self.num_events).unsqueeze(dim = -2)
                                                                               # [batch_size, seq_len, 1, num_events]
            expanded_intensity_selected_event = (expanded_intensity_all_events * events_next_mask).sum(dim = -1)
                                                                               # [batch_size, seq_len, resolution]
            expanded_integral_selected_event = expanded_integral_all_events.sum(dim = -1)
                                                                               # [batch_size, seq_len, resolution]
            expanded_probability_selected_event = expanded_intensity_selected_event * torch.exp(-expanded_integral_selected_event)
                                                                               # [batch_size, seq_len, resolution]

            expanded_probability_selected_event_monte_carlo = expanded_probability_selected_event[:, :, :-1]
                                                                               # [batch_size, seq_len, resolution - 1]
            timestamp_monte_carlo = timestamp[:, :, 1:]                        # [batch_size, seq_len, resolution - 1]
            probability = (expanded_probability_selected_event_monte_carlo * timestamp_monte_carlo).sum(dim = -1)
                                                                               # [batch_size, seq_len]
            return probability

        def bisect_target(taus):
            events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            p_xt = evaluate_per_event(taus)                                    # [batch_size, seq_len]
            p_x = torch.sum(probability_integral * events_next_one_hot, dim = -1)
                                                                               # [batch_size, seq_len]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len]
            p_gap = p_t_x - self.probability_threshold                         # [batch_size, seq_len]

            return p_gap
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = max_val*torch.ones_like(time_history, dtype = torch.float32)       # [batch_size, seq_len]
        tau_pred = median_prediction(l, r)
        gap = (tau_pred - time_next) * mask_next
        gap = torch.abs(gap)

        return gap
'''

'''
    @torch.no_grad()
    def mean_absolute_error(self, time_history, time_next, events_history, mask_history, mask_next):
        \'''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        Update: 2022-09-23
        Add event-wise MAE support.
        \'''
        sample_rate_list = step_split(self.sample_rate, self.mae_step)

        def bisect_target(taus, probability_threshold):
            \'''
            MTPP loss function
            \'''
            integral_all_events, _ = self.model(time_history, taus, events_history, mask_history)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            gap = integral_all_events.sum(dim = -1) + torch.log(1 - probability_threshold)
                                                                               # [sample_rate, batch_size, seq_len]
            return gap

        tau_pred = []
        for sub_sample_rate in sample_rate_list:
            probability_threshold = torch.zeros((sub_sample_rate, *time_next.shape), device = self.device)
                                                                               # [sample_rate, batch_size, seq_len]
            torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [sample_rate, batch_size, seq_len]
            tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                              bisect_target, probability_threshold))
                                                                               # [sample_rate, batch_size, seq_len]
        tau_pred = torch.cat(tau_pred, dim = 0)                                # [sample_rate, batch_size, seq_len]
        tau_pred = tau_pred.mean(dim = 0)                                      # [batch_size, seq_len]
        mae = torch.abs(tau_pred - time_next) * mask_next                      # [batch_size, seq_len]
        
        return mae, tau_pred
    '''

'''
    @torch.no_grad()
    def prediction_with_all_event_types(self, events_history, time_history, p_x, resolution, mask_history, mean, var, inf_val, return_mean):
        \'''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        \'''
        # Preprocess
        sample_rate_list = step_split(self.sample_rate, self.mae_e_step)

        def evaluate_all_event(taus):
            expanded_integral_across_events, expanded_intensity_across_events, timestamp = \
                self.model.integral_intensity_time_next_3d(events_history, time_history, taus, mask_history, resolution, mean, var)
                                                                               # 2 * [sample_rate, batch_size, seq_len, num_events, resolution, num_events] + [sample_rate, batch_size, seq_len, num_events, resolution]
            expanded_integral_sum_across_events = expanded_integral_across_events.sum(dim = -1)
                                                                               # [sample_rate, batch_size, seq_len, num_events, resolution]
            intensity_event_mask = torch.diag(torch.ones(self.num_events, device = self.device))
                                                                               # [num_events, num_events]
            intensity_event_mask = rearrange(intensity_event_mask, f'ne ne1 -> {"() " * (len(expanded_intensity_across_events.shape) - 3)}ne () ne1')
                                                                               # [sample_rate, batch_size, seq_len, num_events, resolution, num_events]
            expanded_intensity_per_event = (expanded_intensity_across_events * intensity_event_mask).sum(dim = -1)
                                                                               # [sample_rate, batch_size, seq_len, num_events, resolution]
            expanded_probability_per_event = expanded_intensity_per_event * torch.exp(-expanded_integral_sum_across_events)
                                                                               # [sample_rate, batch_size, seq_len, num_events, resolution]
            probability = approximate_integration(expanded_probability_per_event, timestamp, dim = -1, only_integral = True)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            return probability

        def bisect_target(taus, probability_threshold):
            p_xt = evaluate_all_event(taus)                                    # [sample_rate, batch_size, seq_len, num_events]
            p_t_x = p_xt / p_x                                                 # [sample_rate, batch_size, seq_len, num_events]
            p_gap = p_t_x - probability_threshold                              # [sample_rate, batch_size, seq_len, num_events]

            return p_gap
        
        tau_pred = []
        batch_size, seq_len = time_history.shape
        p_x = p_x.unsqueeze(dim = 0)                                           # [1, batch_size, seq_len, num_events]

        for sub_sample_rate in sample_rate_list:
            probability_threshold = torch.zeros((sub_sample_rate, batch_size, seq_len, self.num_events), device = self.device)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                              bisect_target, probability_threshold, r_val = inf_val))
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        tau_pred = torch.cat(tau_pred, dim = 0)                                # [sample_rate, batch_size, seq_len, num_events]
        if return_mean:
            tau_pred = tau_pred.mean(dim = 0)                                  # [batch_size, seq_len, num_events]

        return tau_pred
    '''