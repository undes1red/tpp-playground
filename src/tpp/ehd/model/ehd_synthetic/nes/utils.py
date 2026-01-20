import torch


def check_tensor(x, positive = True, inf = True, nan = True):
    '''
    Ensure that the input tensor does not contain: negative numbers, inf, and nan.
    
    Args:
    * x  type: torch.tensor shape: any shape
         the input tensor.

    Outputs:
      No outputs available.
    '''
    if positive:
        assert (x < 0).any() == False, 'Negative numbers detected!'

    if inf:
        assert torch.isfinite(x).all() == True, 'inf detected in input!'

    if nan:
        assert torch.isnan(x).any() == False, 'Nan detected in input!'


def regenerate_batch(input_seq, the_number_of_remained_event):
    output_padded_seqs = torch.tensor_split(input_seq, the_number_of_remained_event.flatten().cumsum(dim = -1).cpu(), dim = 0)[:-1]
                                                                               # (num_of_samples_mask * batch_size) * (*)
    output_padded_seqs = torch.nn.utils.rnn.pad_sequence(output_padded_seqs, batch_first = True, padding_value = -100)
                                                                               # [num_of_samples_mask * batch_size, padded_seq_len]
    return output_padded_seqs