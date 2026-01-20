def dataset_translator(input):
    if input in ['hawkes_1_v2', 'hawkes_2_v2', 'poisson_v2', 'self_correct_v2', 'stationary_renewal_v2']:
        return 'syn'
    else:
        return input