'''
def divide_history_and_future(self, input_time, input_events, input_mask):
        \'''
        TODO: This function needs an overhaul to handle real-world datasets.
        I don't want to generate too much data from one sequence for memory and training speed concerns.
        Maybe at most around 50 generated data sequence from one original data sequence. 
        \'''
        max_subseqence = 50
        subsequence_length = self.seq_len_h + self.seq_len_x
        the_number_of_events = input_mask.sum(dim = -1)                        # [batch_size]
        the_number_of_subseq = the_number_of_events - subsequence_length       # [batch_size]
        the_number_of_subseq = torch.clamp(the_number_of_subseq, max = max_subseqence)
                                                                               # [batch_size]
        # some sequences are so short that we simply remove them.
        short_sequences = the_number_of_subseq <= - int(0.5 * self.seq_len_h)
        the_number_of_subseq[the_number_of_subseq <= 0] = 1
        the_number_of_subseq[short_sequences] = 0

        gen_time_history, gen_time_next, gen_events_history, gen_events_next, gen_mask_history, gen_mask_next = [], [], [], [], [], []
        for idx, (the_number_of_subseq_i, the_number_of_events_i) in enumerate(zip(the_number_of_subseq, the_number_of_events)):
            for start_idx in range(the_number_of_subseq_i):
                history_start_idx = max(the_number_of_events_i - self.seq_len_x - self.seq_len_h - start_idx, 0)
                history_end_idx = history_start_idx + self.seq_len_h
                future_end_idx = history_end_idx + self.seq_len_x

                input_time_history, input_time_next \
                    = input_time[idx, history_start_idx:history_end_idx].detach(), \
                      input_time[idx, history_end_idx:future_end_idx].detach()
                                                                               # [?] + [self.seq_len_x]
                input_events_history, input_events_next \
                    = input_events[idx, history_start_idx:history_end_idx].detach(), \
                      input_events[idx, history_end_idx:future_end_idx].detach()
                                                                               # [?] + [self.seq_len_x]
                input_mask_history, input_mask_next \
                    = input_mask[idx, history_start_idx:history_end_idx].detach(), \
                      input_mask[idx, history_end_idx:future_end_idx].detach()
                                                                               # [?] + [self.seq_len_x]
                
                if input_time_history.shape[0] < self.seq_len_h:
                    input_time_history = F.pad(input_time_history, (0, self.seq_len_h - input_time_history.shape[0]), 'constant', 0)
                    input_events_history = F.pad(input_events_history, (0, self.seq_len_h - input_events_history.shape[0]), 'constant', 0)
                    input_mask_history = F.pad(input_mask_history, (0, self.seq_len_h - input_mask_history.shape[0]), 'constant', 0)


                gen_time_history.append(input_time_history)
                gen_time_next.append(input_time_next)
                gen_events_history.append(input_events_history)
                gen_events_next.append(input_events_next)
                gen_mask_history.append(input_mask_history)
                gen_mask_next.append(input_mask_next)
            
        # stack the data
        if len(gen_time_history) == 0:
            \'''
            No available sequence available.
            \'''
            return (None, None), (None, None), (None, None)

        gen_time_history = torch.stack(gen_time_history, dim = 0)              # [new_batch_size, self.seq_len_h]
        gen_events_history = torch.stack(gen_events_history, dim = 0)          # [new_batch_size, self.seq_len_h]
        gen_mask_history = torch.stack(gen_mask_history, dim = 0)              # [new_batch_size, self.seq_len_h]
        gen_time_next = torch.stack(gen_time_next, dim = 0)                    # [new_batch_size, self.seq_len_x]
        gen_events_next = torch.stack(gen_events_next, dim = 0)                # [new_batch_size, self.seq_len_x]
        gen_mask_next = torch.stack(gen_mask_next, dim = 0)                    # [new_batch_size, self.seq_len_x]
        
        return (gen_time_history, gen_time_next), (gen_events_history, gen_events_next), (gen_mask_history, gen_mask_next)

        '''

'''
        # \'''
        # Dataset preparation.
        # \'''
        history_ends_at = mask_history.sum(dim = -1)                           # [batch_size]
        new_input_time, new_input_events, new_input_mask, new_filter_mask = [], [], [], []
        for idx, history_ends_at_i in enumerate(history_ends_at):
            new_input_time.append(torch.cat((time_history[idx, :history_ends_at_i], time_future[idx], time_history[idx, history_ends_at_i:])))
            new_input_events.append(torch.cat((events_history[idx, :history_ends_at_i], events_future[idx], events_history[idx, history_ends_at_i:])))
            new_input_mask.append(torch.cat((mask_history[idx, :history_ends_at_i], mask_future[idx], mask_history[idx, history_ends_at_i:])))
            new_filter_mask.append(torch.cat((history_mask[idx, :history_ends_at_i, :], future_mask[idx], history_mask[idx, history_ends_at_i:, :])))

        new_input_time = torch.stack(new_input_time, dim = 0)                  # [batch_size, seq_len]
        new_input_events = torch.stack(new_input_events, dim = 0)              # [batch_size, seq_len]
        new_input_mask = torch.stack(new_input_mask, dim = 0)                  # [batch_size, seq_len]
        new_filter_mask = torch.stack(new_filter_mask, dim = 0)                # [batch_size, seq_len]
'''

'''
filter_v1

    def filter(self, input_time, input_events, events_embeddings, input_mask, filter_mask, evaluate = False):
        '''
        Please be careful: the mean and var should come from the training dataset!
        '''
        assert filter_mask is not None, "You want to filter the existing history following the filter mask, but filter mask is unavailable!"
        assert torch.is_tensor(filter_mask), "The filter mask has to be a pytorch tensor!"
        if not evaluate:
            assert filter_mask.requires_grad, "The filter mask must be differentiable!"

        '''
        Dealing with time.
        We select the time whose history[:, :, 0] == 1(meaning this event will remain).
        '''
        filter_mask_for_nominated = filter_mask[..., 0]                        # [samples_for_l_p, batch_size, seq_len]

        '''
        Why this works?
        We generate the history_mark with Gumbel-softmax trick with zero temperature.
        That enforce the possible values of history_mark is either 1 or 0, although the data type is float.
        We use discrete_history_mask_for_nominated for data selection after we multiply history_mask_for_nominated
        with the input sequence data to introduce the gradient of mask to the selected data sequence.
        Caveat: We convert the float tensor history_mask_for_nominated to LongTensor because we ensure this tensor only contains
        0 and 1. DO NOT do this if your float tensor contains non-integers!
        '''
        discrete_filter_mask_for_nominated = filter_mask[..., 0].detach().int()# [samples_for_l_p, batch_size, seq_len]
        the_number_of_remained_event_each_batch = discrete_filter_mask_for_nominated.sum(dim = -1)
                                                                               # [samples_for_l_p, batch_size]
        # if (the_number_of_remained_event_each_batch < 0).any():
        #     print(filter_mask[the_number_of_remained_event_each_batch < 0, :, 0])
        #     print(discrete_filter_mask_for_nominated[the_number_of_remained_event_each_batch < 0])
        #     print(the_number_of_remained_event_each_batch)
        
        length_padded_to = the_number_of_remained_event_each_batch.max(dim = -1)[0]
                                                                               # [samples_for_l_p]
        
        packed_data = zip(filter_mask_for_nominated, discrete_filter_mask_for_nominated, the_number_of_remained_event_each_batch, length_padded_to)
        
        padded_filtered_time, padded_filtered_events, padded_filtered_event_embeddings, padded_filtered_masks \
        = [], [], [], []

        for filter_mask_for_nominated_per_sample, discrete_filter_mask_for_nominated_per_sample, \
            the_number_of_remained_event_each_batch_per_sample, length_padded_to_per_sample in packed_data:
            '''
            Preparing for time.
            '''
            masked_input_time = input_time.cumsum(dim = -1)                        # [batch_size, seq_len]
            masked_nominated_time = masked_input_time * filter_mask_for_nominated_per_sample
                                                                                   # [batch_size, seq_len]
            selected_time = masked_nominated_time[discrete_filter_mask_for_nominated_per_sample == 1]
                                                                                   # (the_number_of_remained_event_each_batch_per_sample.sum())
            
            '''
            Preparing for discrete events.
            Caution: The differentiable mask directly applies to the embedding of events. PyTorch's embedding layer only accepts
            LongTensor, and converting the event tensor from Float to Long will absolutely lose gradients.
            '''
            selected_input_events = input_events[discrete_filter_mask_for_nominated_per_sample == 1]
                                                                                   # (the_number_of_remained_event_each_batch_per_sample.sum())
    
            '''
            Preparing for events embeddings.
            '''
            nominated_events_embeddings = events_embeddings * filter_mask_for_nominated_per_sample.unsqueeze(dim = -1)
                                                                                   # [batch_size, seq_len_h, d_history]
            selected_embeddings = nominated_events_embeddings[discrete_filter_mask_for_nominated_per_sample == 1]
                                                                                   # [(the_number_of_remained_event_each_batch_per_sample.sum()), d_history]

            '''
            Preparing for mask.
            '''
            selected_input_mask = input_mask[discrete_filter_mask_for_nominated_per_sample == 1]
                                                                                   # (the_number_of_remained_event_each_batch_per_sample.sum())
    
            # Reshape the selected_time into a new batch.
            idx_new_seq_start = 0
            padded_times_per_sample, padded_filtered_events_per_sample, \
            padded_filtered_event_embeddings_per_sample, padded_filtered_masks_per_sample \
            = [], [], [], []
            
            for num_remained_event in the_number_of_remained_event_each_batch_per_sample:
                assert num_remained_event >= 0, "Negative num_remained_event!"
                '''
                Dealing with input_times.
                '''
                remained_times = F.pad(
                    selected_time[idx_new_seq_start:idx_new_seq_start + num_remained_event], 
                    (1, length_padded_to_per_sample - num_remained_event), 'constant', 0
                )                                                                  # [length_padded_to + 1]
                padded_times_per_sample.append(remained_times)                     # [(n_loops) * length_padded_to + 1]
            
                '''
                Dealing with input_events.
                '''
                remained_events = F.pad(selected_input_events[idx_new_seq_start:idx_new_seq_start + num_remained_event], 
                                           (0, length_padded_to_per_sample - num_remained_event), 'constant', 0)
                                                                                   # [length_padded_to]
                padded_filtered_events_per_sample.append(remained_events)          # [(n_loops), length_padded_to]

                '''
                Dealing with mark embeddings.
                We can not directly apply the method above to the mark sequence because marks are discrete.
                However, the embedding vectors are continuous, where our trick should work.
                '''
                remained_events_embeddings = F.pad(selected_embeddings[idx_new_seq_start:idx_new_seq_start + num_remained_event, :], 
                                                      (0, 0, 0, length_padded_to_per_sample - num_remained_event), 'constant', 0)
                                                                                   # [length_padded_to, d_history]
                padded_filtered_event_embeddings_per_sample.append(remained_events_embeddings)
                                                                                   # [(n_loops), length_padded_to, d_history]
                
                '''
                Why we don't need the gradient from the mask?
                Reasons:
                Mask vectors for data are always LongTensors only featuring 0 and 1. That means we can not save any gradient in input_mask.
                Thus, we only select the mask using discrete_history_mask_for_nominated.
                '''
                remained_masks = F.pad(selected_input_mask[idx_new_seq_start:idx_new_seq_start + num_remained_event], 
                                           (0, length_padded_to_per_sample - num_remained_event), 'constant', 0)
                                                                                   # [length_padded_to]
                padded_filtered_masks_per_sample.append(remained_masks)            # [(n_loops), length_padded_to]
                
                idx_new_seq_start += num_remained_event
            
            padded_times_per_sample = torch.stack(padded_times_per_sample, dim = 0)# [batch_size, length_padded_to]
            padded_filtered_events_per_sample = torch.stack(padded_filtered_events_per_sample, dim = 0)
                                                                                   # [batch_size, length_padded_to]
            padded_filtered_event_embeddings_per_sample = torch.stack(padded_filtered_event_embeddings_per_sample, dim = 0)
                                                                                   # [batch_size, length_padded_to, d_history]
            padded_filtered_masks_per_sample = torch.stack(padded_filtered_masks_per_sample, dim = 0)
                                                                                   # [batch_size, length_padded_to]
            padded_filtered_time_per_sample = padded_times_per_sample.diff(dim = -1)
                                                                                   # [batch_size, length_padded_to]
            
            padded_filtered_time.append(padded_filtered_time_per_sample)           # [batch_size, length_padded_to]
            padded_filtered_events.append(padded_filtered_events_per_sample)       # [batch_size, length_padded_to]
            padded_filtered_event_embeddings.append(padded_filtered_event_embeddings_per_sample)
                                                                                   # [batch_size, length_padded_to, d_history]
            padded_filtered_masks.append(padded_filtered_masks_per_sample)         # [batch_size, length_padded_to]


        return padded_filtered_time, padded_filtered_events, padded_filtered_event_embeddings, padded_filtered_masks
'''