'''

   def mean_absolute_error_per_event(self, input_time, input_events, mask, mean, var, fast):
        if self.original_mark_generation:
            raise Exception('Original RMTPP model is in fact a TPP model with a dedicated event prediction module, so pe-MAE does not function here.')
        
        time_history, time_next = self.divide_history_and_next(input_time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        events_history, events_next = self.divide_history_and_next(input_events, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask, unsqueeze = False)   # [batch_size, seq_len]

        if mean == 0 and var == 1:
            \'''
            This dataset does not apply normalisation, so we need to calculate the mean and variance here.
            \'''
            mean = input_time.mean()
            var = input_time.var()
        
        # Use a relatively large number as the positive infinity.
        max_ = min(1e6, mean + 10 * var)
        resolution = min(int(max_ * 100), 50000)
        time_infinite = torch.ones_like(time_next, device = self.device) * max_# [batch_size, seq_len, 1]

        # First, we find the integral and intensity function that RMTPP estimates.
        # This part is only available when original_mark_generation is false as the original RMTPP model
        # is in fact a TPP model.
        intensity, integral, timestamp = \
                self.submodel.intensity_integral(events_history, time_history, time_infinite, resolution, mean, var, sum = False)
                                                                               # 2 * [batch_size, seq_len * resolution, num_events] + [batch_size, seq_len * resolution]
        intensity = rearrange(intensity, 'b (s r) ne -> b s r ne', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        integral = rearrange(integral, 'b (s r) ne -> b s r ne', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        timestamp = rearrange(timestamp, 'b (s r) -> b s r 1', r = resolution) # [batch_size, seq_len, resolution, 1]
        probability_dist = intensity * torch.exp(-integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        # After investigation, sometimes we could get nan when both intensity and integral are inf
        # Based on TPP's definition, the true value should be 0.
        probability_dist = torch.nan_to_num(probability_dist, nan = 0.0)       # [batch_size, seq_len, resolution, num_events]
        
        cumulated_probability = probability_dist[:, :, :-1, :] * timestamp[:, :, 1:, :] / var
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = cumulated_probability.sum(dim = -2)                      # [batch_size, seq_len, num_events]
        probability_integral_sum = probability.sum(dim = -1)                   # [batch_size, seq_len]
        predicted_events = torch.argmax(probability, dim = -1)                 # [batch_size, seq_len]

        # F1 value and top_k_acc are only avaliable when batch_size = 1
        f1 = f1_score(y_true = events_next.squeeze().detach().cpu(),
                      y_pred = predicted_events.squeeze().detach().cpu(), average = 'macro')

        # Only available when batch_size = 1
        top_k_acc = []
        if not fast:
            if self.num_events > 2:
                for k in range(1, self.num_events + 1):
                    top_k_acc.append(
                        top_k_accuracy_score(y_true = events_next.squeeze().detach().cpu(),
                                             y_score = probability.reshape(-1, self.num_events).detach().cpu(),
                                             k = k,
                                             labels = np.arange(self.num_events))
                    )
            else:
                top_k_acc.append(
                    accuracy_score(
                        y_true = events_next.squeeze().detach().cpu(),
                        y_pred = predicted_events.squeeze().detach().cpu()
                    )
                )
                top_k_acc.append(1.0)
        
        if mean == 0:
            resolution = max(min(int(input_time.mean().item() * 200), 1000), 1)
        else:
            resolution = max(min(int(mean * 200), 1000), 1)

        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history,
                                                                  probability, resolution, mask_next, mean, var, max_)
                                                                               # [batch_size, seq_len, num_events]
        
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predicted_events, time_history, time_next,
                                                                               probability, resolution, mask_next, max_, mean, var)
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, time_next, 
                                                                  probability, resolution, mask_next, max_, mean, var)

        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict) / mask_next.sum()
        mae_per_event_avg = torch.sum(mae_per_event) / mask_next.sum()
        
        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, (mae_per_event_pure_predict_avg.item(), mae_per_event_avg.item()), \
               (mae_per_event_pure_predict, mae_per_event)


    def evaluate_per_event(self, events_history, time_history, events_mask, tau, resolution, mean, var):
        intensity, integral, timestamp = \
                        self.model.integral_intensity_time_next_2d(events_history, time_history, tau, resolution, mean, var)
                                                                               # 2 * [batch_size, seq_len * resolution, num_events] + [batch_size, seq_len * resolution]
        probability_dist = intensity * torch.exp(-integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len * resolution, num_events]
        probability_dist = rearrange(probability_dist, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        timestamp = rearrange(timestamp, 'b (s r) -> b s r 1', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        cumulative_probability = probability_dist[:, :, :-1, :] * timestamp[:, :, 1:, :] / var
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
        cumulative_probability = cumulative_probability.sum(dim = -2)          # [batch_size, seq_len, num_events]
        cumulative_probability = cumulative_probability * events_mask          # [batch_size, seq_len, num_events]
        probability = cumulative_probability.sum(dim = -1)                     # [batch_size, seq_len]

        return probability


    def mean_absolute_error_per_event_worker(self, events_history, events_next, 
        time_history, time_next, probability_integral, resolution, mask, max_val, mean, var):
        \'''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        \'''
        def bisect_target(taus):
            events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            p_xt = self.evaluate_per_event(events_history, time_history, events_next_one_hot, taus, resolution, mean, var)
                                                                               # [batch_size, seq_len]
            p_x = torch.sum(probability_integral * events_next_one_hot, dim = -1)
                                                                               # [batch_size, seq_len]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len]
            p_gap = p_t_x - self.probability_threshold                         # [batch_size, seq_len]

            return p_gap.unsqueeze(dim = -1)
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len, 1]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len, 1]
        tau_pred = median_prediction(l, r)
        gap = (tau_pred - time_next).squeeze(-1) * mask
        gap = torch.abs(gap)

        return gap


'''