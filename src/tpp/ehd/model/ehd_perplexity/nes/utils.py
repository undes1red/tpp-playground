import torch


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