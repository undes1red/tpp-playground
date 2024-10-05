# Temporary name: Neural Event Selector
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce, pack, unpack
from src.ehd.model.ehd_perplexity.nes.utils import regenerate_batch

from src.toolbox.misc import check_tensor


class NES(nn.Module):
    def __init__(self, num_of_samples_mask, length_of_x, length_of_h, \
                 epsilon, tau, device):
        super(NES, self).__init__()
        self.num_of_samples_mask = num_of_samples_mask
        self.length_of_x = length_of_x
        self.length_of_h = length_of_h 

        self.epsilon = epsilon
        self.tau = tau
        self.device = device


    def forward(self, probability_model, model_args, mask_history, number_of_events, \
                discrete_inputs, continuous_inputs, evaluate = False, output_probability_instead_of_mask = False):
        '''
        By default, we treat the events with mark 1, which means input_probability[1] > input_probability[0] as
        essential events.
        '''
        if evaluate:
            num_of_samples_mask = 1
        else:
            num_of_samples_mask = self.num_of_samples_mask

        '''
        Call the Neural probability model for the probability p(y|x, H).
        Now, we treat the first event in the history as a special event. This event will simultaneously appear in
        the essential set and the noise set.
        '''
        # Here, mask = 1: important. Removing them would cause counterfactual results.
        #       mask = 0: noises or unrelated events. Keeping them makes no benefit for modeling the future.
        input_probability = probability_model(*model_args)                     # [batch_size, length_of_h + 1]
        check_tensor(input_probability)
        batch_size = input_probability.shape[-3]

        '''
        Tell the average gap between p(y = 1|x, H) and p(y = 0|x, H). Bigger probability gap means the model
        is more certain about the result.
        '''
        gap_between_p_1_and_p_0 = input_probability[:, :, 1] - input_probability[:, :, 0]
                                                                               # [batch_size, length_of_h + 1]
        gap_sum = torch.abs(gap_between_p_1_and_p_0.detach() * mask_history).sum()
        gap_mean = gap_sum / number_of_events

        repeated_input_probability = repeat(input_probability, '... -> n ...', n = num_of_samples_mask)
                                                                               # [1, batch_size, length_of_h + 1, 2]
        if evaluate:
            history_mask = F.one_hot(torch.argmax(repeated_input_probability, dim = -1), num_classes = 2)
                                                                               # [1, batch_size, seq_len_h + 1, 2]
        else:
            history_mask = F.gumbel_softmax(torch.log(repeated_input_probability + self.epsilon), tau = 1.0, hard = True, dim = -1)
                                                                               # [samples_for_l_p, batch_size, seq_len_h + 1, 2]
        check_tensor(history_mask)
        future_mask = torch.ones((batch_size, self.length_of_x, 2), device = self.device)
                                                                               # [batch_size, length_of_x + 1, 2]
        future_mask = repeat(future_mask, '... -> n ...', n = num_of_samples_mask)
                                                                               # [num_of_samples_mask, batch_size, length_of_x + 1, 2]
        '''
        L_n, optimize the length of essential events.
        '''
        selected_mask = history_mask[..., 1:, 1]
        L_n = torch.linalg.norm(selected_mask.float(), ord = 1, dim = -1) / self.length_of_h
                                                                               # [num_of_samples_mask, batch_size]
        L_n = L_n.mean()

        filter_mask = torch.cat([history_mask, future_mask], dim = -2)         # [num_of_samples_mask, batch_size, length_of_h + length_of_x + ?, 2]
        filter_mask[:, :, 0] = 1

        padded_distilled_discrete_inputs, padded_distilled_continuous_inputs, \
        padded_left_discrete_inputs, padded_left_continuous_inputs \
            = self.filter(discrete_inputs, continuous_inputs, filter_mask = filter_mask, \
                          evaluate = evaluate)                                 # [num_of_samples_mask, batch_size, length_of_h + length_of_x + ?] * 2 + [num_of_samples_mask, batch_size, length_of_h + length_of_x + 2, d_history] + [num_of_samples_mask, batch_size, length_of_h + length_of_x + 2]
        
        return (L_n, gap_mean), input_probability if output_probability_instead_of_mask else history_mask , \
               padded_distilled_discrete_inputs, padded_distilled_continuous_inputs, \
               padded_left_discrete_inputs, padded_left_continuous_inputs


    def filter(self, discrete_inputs, continuous_inputs, filter_mask, evaluate = False):
        '''
        Now, filter() should provide \\mathcal{H}_{s,o,t_l} and \\mathcal{H}_{r,o,t_l} when evaluate = True.
        filter still only provides \\mathcal{H}_{r,o,t_l} when evaluate = False.
        '''
        '''
        Please be careful: the mean and var should come from the training dataset!
        '''
        assert filter_mask is not None, "You want to filter the existing history following the filter mask, but filter mask is unavailable!"
        assert torch.is_tensor(filter_mask), "The filter mask has to be a pytorch tensor!"
        if not evaluate:
            assert filter_mask.requires_grad, "The filter mask must be differentiable!"
        num_of_samples_mask = filter_mask.shape[0]

        '''
        Why this works?
        We generate the history_mark with Gumbel-softmax trick with zero temperature.
        That enforce the possible values of history_mark is either 1 or 0, although the data type is float so gradient calculation makes sense.
        We use discrete_filter_mask_for_distilled_events to select data after we multiply filter_mask_for_distilled_events
        with the input sequence to attach the gradient of mask to the selected data sequence.
        Caveat: We convert the float tensor history_mask_for_nominated to LongTensor because we ensure this tensor only contains
        0 and 1. DO NOT do this if your float tensor contains non-integers!
        '''
        select_value_for_gradient_attachment = torch.argmax(filter_mask, dim = -1, keepdim = True)
                                                                               # [num_of_samples_mask, batch_size, seq_len, 2]
        one_tensor_for_gradient_attachment = torch.take_along_dim(filter_mask, select_value_for_gradient_attachment, dim = -1).squeeze(dim = -1)
                                                                               # [num_of_samples_mask, batch_size, seq_len]
        discrete_filter_mask_for_left_events = filter_mask[..., 0].detach().int()
                                                                               # [num_of_samples_mask, batch_size, seq_len]
        discrete_filter_mask_for_distilled_events = filter_mask[..., 1].detach().int()
                                                                               # [num_of_samples_mask, batch_size, seq_len]
        the_number_of_distilled_event = discrete_filter_mask_for_distilled_events.sum(dim = -1).flatten().tolist()
                                                                               # [num_of_samples_mask, batch_size]
        the_number_of_left_event = discrete_filter_mask_for_left_events.sum(dim = -1).flatten().tolist()
                                                                               # [num_of_samples_mask, batch_size]
        def repeat_n_times(x):
            return repeat(x, '... -> n ...', n = num_of_samples_mask)          # [num_of_samples_mask, batch_size, seq_len]
        
        discrete_inputs = map(repeat_n_times, discrete_inputs)                 # [num_of_samples_mask, batch_size, seq_len] * m
        continuous_inputs = map(repeat_n_times, continuous_inputs)             # [num_of_samples_mask, batch_size, seq_len] * n
        
        def gradient_attachment(x):
            if len(x.shape) == len(one_tensor_for_gradient_attachment.shape):
                return x * one_tensor_for_gradient_attachment                  # [num_of_samples_mask, batch_size, seq_len]
            elif len(x.shape) > len(one_tensor_for_gradient_attachment.shape):
                einop = f'... -> ... {"() " * (len(x.shape) - len(one_tensor_for_gradient_attachment.shape))}'
                return x * rearrange(one_tensor_for_gradient_attachment, einop)# [num_of_samples_mask, batch_size, seq_len, ...]
            else:
                raise Exception('Mask has more dimension than the input tensor, which is unexpected.')
            
        def select_events_by_mask(x):
            if len(x.shape) >= len(discrete_filter_mask_for_distilled_events.shape):
                min_shape_len = min(len(x.shape), len(discrete_filter_mask_for_distilled_events.shape))
                                                                               # [num_of_samples_mask, batch_size, seq_len, ...]
                assert x.shape[:min_shape_len] == discrete_filter_mask_for_distilled_events.shape[:min_shape_len], "Dimension of input and mask tensor mismatches, selection can not continue."
                return x[discrete_filter_mask_for_distilled_events == 1], x[discrete_filter_mask_for_left_events == 1]
            else:
                raise Exception('Mask has more dimension than the input tensor, which is unexpected.')

        # select the remained events from the original input.
        continuous_inputs = map(gradient_attachment, continuous_inputs)        # [num_of_samples_mask, batch_size, seq_len, (...)] * n
        selected_discrete_inputs = list(map(select_events_by_mask, discrete_inputs)) # [(...) * 2] * m
        selected_continuous_inputs = list(map(select_events_by_mask, continuous_inputs))
                                                                               # [(...) * 2] * n
        padded_distilled_discrete_inputs = []
        padded_left_discrete_inputs = []
        for selected_discrete_input in selected_discrete_inputs:
            padded_distilled_discrete_input, padded_left_discrete_input \
                = regenerate_batch(selected_discrete_input, (the_number_of_distilled_event, the_number_of_left_event), num_of_samples_mask)
            padded_distilled_discrete_inputs.append(padded_distilled_discrete_input)
            padded_left_discrete_inputs.append(padded_left_discrete_input)

        padded_distilled_continuous_inputs = []
        padded_left_continuous_inputs = []
        for selected_continuous_input in selected_continuous_inputs:
            padded_distilled_continuous_input, padded_left_continuous_input \
                = regenerate_batch(selected_continuous_input, (the_number_of_distilled_event, the_number_of_left_event), num_of_samples_mask)
            padded_distilled_continuous_inputs.append(padded_distilled_continuous_input)
            padded_left_continuous_inputs.append(padded_left_continuous_input)

        return padded_distilled_discrete_inputs, padded_distilled_continuous_inputs, \
               padded_left_discrete_inputs, padded_left_continuous_inputs