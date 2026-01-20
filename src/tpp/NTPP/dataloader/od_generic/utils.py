import numpy as np


def sample_particles(event_seq, num_of_missing_sample, missing_probability, eps = 1e-20):
    assert event_seq is not None
    seq_len = event_seq.size
    rng = np.random.default_rng()
    
    random_vals = rng.random((num_of_missing_sample, seq_len))                 # [num_of_missing_sample, seq_len]
    threshold = missing_probability[event_seq]                                 # [num_of_missing_sample, seq_len]
    mask = random_vals > threshold                                             # [num_of_missing_sample, seq_len]
    censor_probs = (1 - 2 * threshold) * mask + threshold                      # [num_of_missing_sample, seq_len]
    censor_probs[censor_probs < eps] = eps                                     # [num_of_missing_sample, seq_len]
    log_censor_prob = np.log(censor_probs).sum(axis=1)                         # [num_of_missing_sample, seq_len]
    
    return mask, log_censor_prob