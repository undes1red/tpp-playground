import torch
import torch.nn.functional as F
from einops import repeat, rearrange

from src.toolbox.misc import check_tensor


def generate_masks(input_probability, seq_len_x, evaluate = False):
    if evaluate:
        history_mask = F.one_hot(torch.argmax(input_probability, dim = -1), num_classes = 2)
                                                                           # [samples_for_l_p, batch_size, seq_len_h + 1, 2]
    else:
        history_mask = F.gumbel_softmax(torch.log(input_probability + 1e-20), tau = 1.0, hard = True, dim = -1)
                                                                           # [samples_for_l_p, batch_size, seq_len_h + 1, 2]
        check_tensor(history_mask)
    
    future_mask = torch.ones((*history_mask.shape[:-2], seq_len_x, 2), device = history_mask.device)
                                                                           # [num_of_samples_mask, batch_size, length_of_x + 1, 2]

    filter_mask = torch.cat([history_mask, future_mask], dim = -2)         # [num_of_samples_mask, batch_size, length_of_h + length_of_x + ?, 2]
    filter_mask[:, :, 0] = 1

    return history_mask, filter_mask


def generate_masks_from_zero_one_vector(input_seq, seq_len_x):
    history_mask = F.one_hot(input_seq, num_classes = 2)                   # [samples_for_l_p, batch_size, seq_len_h + 1, 2]

    future_mask = torch.ones((*history_mask.shape[:-2], seq_len_x, 2), device = history_mask.device)
                                                                           # [num_of_samples_mask, batch_size, length_of_x + 1, 2]

    filter_mask = torch.cat([history_mask, future_mask], dim = -2)         # [num_of_samples_mask, batch_size, length_of_h + length_of_x + ?, 2]
    filter_mask[:, :, 0] = 1

    return history_mask, filter_mask


def filter(discrete_inputs, continuous_inputs, filter_mask, evaluate = False):
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
    selected_discrete_inputs = map(select_events_by_mask, discrete_inputs) # [(...) * 2] * m
    selected_continuous_inputs = map(select_events_by_mask, continuous_inputs)
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


def regenerate_batch(input_seqs, the_number_of_events, num_of_samples_mask):
    distilled_features, left_features = input_seqs
    the_number_of_distilled_event, the_number_of_left_event = the_number_of_events

    distilled_features = distilled_features.split(the_number_of_distilled_event, dim = 0)
                                                                               # (num_of_samples_mask * batch_size) * (*)
    left_features = left_features.split(the_number_of_left_event, dim = 0)     # (num_of_samples_mask * batch_size) * (*)
    output_padded_distilled_seqs = []
    output_padded_left_seqs = []
    
    batch_size = len(distilled_features) // num_of_samples_mask
    for num_of_batch in range(num_of_samples_mask):
        output_padded_distilled_seqs.append(torch.nn.utils.rnn.pad_sequence(
            distilled_features[batch_size*num_of_batch:batch_size*(num_of_batch + 1)], batch_first = True))
                                                                               # [batch_size, padded_seq_len]
        output_padded_left_seqs.append(torch.nn.utils.rnn.pad_sequence(
            left_features[batch_size*num_of_batch:batch_size*(num_of_batch + 1)], batch_first = True))
                                                                               # [batch_size, padded_seq_len]
    return output_padded_distilled_seqs, output_padded_left_seqs