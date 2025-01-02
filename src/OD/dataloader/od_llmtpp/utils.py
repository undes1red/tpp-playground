import numpy as np
import copy


def convert_missing_mask_to_gap_mask(missing_mask):
    # input shape: [num_samples, seq_len]
    
    masks = []
    for missing_mask_per_seq in missing_mask:
        current_in_missing = False
        mask_current_seq = []
        for item in missing_mask_per_seq[1:]:
            if item == 1 and not current_in_missing:
                mask_current_seq.append(1)
            elif item == 1 and current_in_missing:
                current_in_missing = False
            elif item == 0 and not current_in_missing:
                mask_current_seq.append(0)
                current_in_missing = True
            else:
                continue
        
        masks.append(mask_current_seq)
    
    return masks


def remove_the_last_event_from_mask(mask):
    '''
    Remove the probability of the dummy event by mask.
    '''
    mask_without_last = np.zeros_like(mask)                                    # [batch_size, seq_len - 1]
    for idx, mask_per_seq in enumerate(mask):
        last_event_index = mask_per_seq.sum() - 1
        mask_without_last_per_seq = copy.deepcopy(mask_per_seq)
        mask_without_last_per_seq[last_event_index] = 0
        mask_without_last[idx] = mask_without_last_per_seq
    
    return mask_without_last