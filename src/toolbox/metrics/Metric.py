import math

from src.toolbox.list_operation import list_mul

class Metric():
    '''
    A Metric handler.
    1. metric_number: How many metric do you have?
    2. smaller_is_better: If model performance is better with lower metric value, you should set it to true. Otherwise, it is false.
    If smaller_is_better is set, its length must match argument 'metric_number'.
    '''
    def __init__(self, metric_number, smaller_is_better = None):
        self.metric_number = metric_number
        self.map = {True: 1, False: -1}
        self.best_metric = [math.inf] * self.metric_number
        if smaller_is_better is None:
            self.mask = [1] * self.metric_number
        else:
            assert len(smaller_is_better) == self.metric_number
            self.mask = [self.map[item] for item in smaller_is_better]
    

    def compare(self, input_metric):
        assert len(input_metric) == len(self.mask)
        tmp = list_mul(input_metric, self.mask)
        output = True

        for input_number, recorded in zip(tmp, self.best_metric):
            if input_number > recorded:
                output = False
                break
        
        if output:
            self.best_metric = input_metric
        
        return output
    
    
    def show(self):
        return self.best_metric