import numpy as np
import pandas as pd

def transform_autoregression(data_input, history_seq, num_events):
    history_time = []
    history_event = []
    result = []
    score = []
    event = []
    intensity = []
    arg_size = 0

    size = data_input.shape[0]
    for index in range(size):
        original_data_size = data_input.iloc[index].time_seq.size
        time = data_input.iloc[index].time_seq
        L = data_input.iloc[index].score
        event_seq = data_input.iloc[index].event
        intensity_seq = data_input.iloc[index].intensity
        try:
            assert (time < 0).any() == False
        except:
            raise ValueError("Non-monotonic increase time input detected.")

        padded_time = np.concatenate(([0.] * (history_seq - 1), time))
        padded_event_seq = np.concatenate(([num_events + 1] * (history_seq - 1), event_seq))
        arg_history_time = []
        arg_history_event = []

        for i in range(0, original_data_size - 1):
            arg_history_time.append(padded_time[i:i+history_seq].tolist())
            arg_history_event.append(padded_event_seq[i:i+history_seq].tolist())

        history_time.append(arg_history_time)
        history_event.append(arg_history_event)
        result.append(time[1:].tolist())
        score.append(L[1:].tolist())
        event.append(event_seq[1:].tolist())
        intensity.append(intensity_seq[1:].tolist())
        
    return pd.DataFrame.from_dict({'history_time': history_time, 'history_event': history_event, 'result': result, \
                                   'score': score, 'event': event, 'intensity': intensity})