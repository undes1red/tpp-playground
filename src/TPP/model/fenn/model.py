import torch
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
from einops import rearrange, repeat, reduce
import numpy as np

from src.TPP.model.fenn.submodel import FENN
from src.TPP.model.utils import BasicModule
from src.TPP.model.fenn.utils import *
from src.TPP.model.fenn.plot import *



class FENNModel(BasicModule):
    '''
    The original FullyNN model with dedicated marker prediction module.
    As the time distribution p^*(t) and p^*(m) are independent, only function time_event_prediction() works while
    event_time_prediction() is unimplemented.

    This FullyNN employs RNN modules as the history encoder. One should use TFullyNN if they want to evaluate FullyNN's performance
    against Transformer history encoders.
    '''
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 history_module_layers,
                 mlp_layers,
                 nonlinear,
                 probability_threshold,
                 num_events,
                 device,
                 history_module = 'LSTM',
                 event_toggle = False,
                 zero_shift = False):
        super(FENNModel, self).__init__()
        self.device = device
        self.probability_threshold = probability_threshold
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.zero_shift_factor = 1e-12

        self.model = FENN(d_history = d_history, d_intensity = d_intensity, num_events = num_events,
                          dropout = dropout, history_module = history_module, history_module_layers = history_module_layers,
                          mlp_layers = mlp_layers, nonlinear = nonlinear, event_toggle = event_toggle, 
                          zero_shift = zero_shift, device = device)


    def divide_history_and_next(self, input):
        '''
        Extract the history and prediction sequences from the original output

        Args:
        * input  type: torch.tensor shape: [batch_size, seq_len + 1]
                 The input tensor.
        
        Outputs:
        * input_history  type: torch.tensor shape: [batch_size, seq_len]
                         The history sequence extracted from the original input.
        * input_next     type: torch.tensor shape: [batch_size, seq_len]
                         The history sequence extracted from the original input.
        '''

        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next


    def forward(self, input_time, input_events, mask, mean, var, evaluate):
        '''
        The entrance of the FullyNN wrapper.
        
        Args:
        * input_time    type: torch.tensor shape: [batch_size, seq_len + 1]
                        The original time sequence. We should extract the history and target sequence from it
                        by divide_history_and_next().
        * input_events  type: torch.tensor shape: [batch_size, seq_len + 1]
                        The original event sequence. We should extract the history and target sequence from it
                        by divide_history_and_next().
        * mask          type: torch.tensor shape: [batch_size, seq_len + 1]
                        We use mask to mask out unneeded outputs.
        * mean          type: float shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        * var           type: float shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        * evaluate      type: bool shape: N/A
                        perform a model training step when evaluate == False
                        perform a model evaluate step when evaluate == True
        
        Outputs:
        Refers to train() and evaluate()'s documentation for detailed information.

        '''
        return self.evaluate_procedure(input_time, input_events, mask, mean, var) if evaluate \
            else self.train_procedure(input_time, input_events, mask, mean, var)


    def train_procedure(self, input_time, input_events, mask, mean, var):
        '''
        The forwardpropagation function of the FullyNNModel, the wrapper of FullyNN with lots of useful
        utilities.
        
        Outputs:
        * time_loss             type: torch.tensor shape: [1]
                                The value of NLL loss: L = -log \frac{\partial \Lambda^*(m, t)}{\partial t} + \Lambda^*(m, t)
        * events_loss           type: torch.tensor shape: [1]
                                The value of the event loss: L = -log \frac{\lambda^*(m, t)}{\sum_{n \in M}{\lambda^*(n, t)}}
        * the_number_of_events  type: int shape: N/A
                                The number of legit predicted events.
        '''

        self.train()

        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        '''
        preparing for multi-event training when needed
        '''
        if self.event_toggle:
            time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
        time_next.requires_grad = True
        integral_for_each_event = self.model(events_history, time_history, time_next, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
        '''
        Obtains intensity values.
        '''
        intensity_for_each_event = torch.autograd.grad(
            outputs = integral_for_each_event,
            inputs = time_next,
            grad_outputs = torch.ones_like(integral_for_each_event),
            create_graph = True,
        )[0]
        check_tensor(intensity_for_each_event)                                 # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
        assert intensity_for_each_event.shape == integral_for_each_event.shape
        time_next.requires_grad = False

        '''
        Calculate the event loss, macro-F1, and other possible metrics measuring event prediction accuracy.
        This part is only available when event_toggle = True
        '''
        events_loss = torch.tensor(0., dtype = torch.float32)
        if self.event_toggle:
            probability_for_each_event = torch.log(intensity_for_each_event + self.zero_shift_factor)
                                                                               # [batch_size, seq_len, num_events]
            events_probability = torch.nn.functional.softmax(probability_for_each_event, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
            events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                                      events_next.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
            events_loss = events_loss * mask_next                              # [batch_size, seq_len]
            events_loss = events_loss.sum()

        '''
        Calculate the NLL loss of p^*(m, t).
        L = -log \frac{\partial \Lambda^*(m, t)}{\partial t} + \Lambda^*(m, t)
        '''
        time_loss = self.nll_loss(intensity = intensity_for_each_event, events_next = events_next, \
                                  intensity_integral = integral_for_each_event, mask_next = mask_next)
        the_number_of_events = mask_next.sum().item()

        return time_loss, events_loss, the_number_of_events


    def evaluate_procedure(self, input_time, input_events, mask, mean, var):
        '''
        The forwardpropagation function of the FullyNNModel, the wrapper of FullyNN with lots of useful
        utilities, with evaluation enabled.

        Outputs:
        * time_loss             type: torch.tensor shape: [1]
                                The value of NLL loss: L = -log \frac{\partial \Lambda^*(m, t)}{\partial t} + \Lambda^*(m, t)
        * events_loss           type: torch.tensor shape: [1]
                                The value of the event loss: L = -log \frac{\lambda^*(m, t)}{\sum_{n \in M}{\lambda^*(n, t)}}
        * the_number_of_events  type: int shape: N/A
                                The number of legit predicted events.
        '''

        self.eval()

        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]
        the_number_of_events = mask_next.sum().item()

        mae, pred_time = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                                  time_next = time_next, mask_next = mask_next, mean = mean, var = var)
                                                                               # 2 * [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_events

        if self.event_toggle:
            pred_time = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
            time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]

        '''
        preparing for multi-event training when needed
        '''
        pred_time.requires_grad = True
        time_next.requires_grad = True
        integral_for_each_event_from_tl_to_pred_time = self.model(events_history, time_history, pred_time, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
        integral_for_each_event_from_tl_to_time_next = self.model(events_history, time_history, time_next, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]

        '''
        Obtains intensity values.
        '''
        intensity_for_each_event_from_tl_to_pred_time = torch.autograd.grad(
            outputs = integral_for_each_event_from_tl_to_pred_time,
            inputs = pred_time,
            grad_outputs = torch.ones_like(integral_for_each_event_from_tl_to_pred_time),
        )[0]                                                                   # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
        intensity_for_each_event_from_tl_to_time_next = torch.autograd.grad(
            outputs = integral_for_each_event_from_tl_to_time_next,
            inputs = time_next,
            grad_outputs = torch.ones_like(integral_for_each_event_from_tl_to_time_next),
        )[0]                                                                   # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
        pred_time.requires_grad = False
        time_next.requires_grad = False
        check_tensor(intensity_for_each_event_from_tl_to_pred_time)            # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
        check_tensor(intensity_for_each_event_from_tl_to_time_next)            # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
        assert intensity_for_each_event_from_tl_to_pred_time.shape == integral_for_each_event_from_tl_to_pred_time.shape
        assert intensity_for_each_event_from_tl_to_time_next.shape == integral_for_each_event_from_tl_to_time_next.shape

        '''
        Calculate the event loss, macro-F1, and other possible metrics measuring event prediction accuracy.
        This part is only available when event_toggle = True
        '''
        events_loss = torch.tensor(0., dtype = torch.float32)
        f1 = 0
        if self.event_toggle:
            probability_for_each_event = torch.log(intensity_for_each_event_from_tl_to_pred_time + self.zero_shift_factor)
                                                                               # [batch_size, seq_len, num_events]
            events_probability = torch.nn.functional.softmax(probability_for_each_event, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
            events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                                      events_next.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
            events_loss = events_loss * mask_next                              # [batch_size, seq_len]
            events_loss = events_loss.sum()

            events_pred_index, events_true = \
                move_from_tensor_to_ndarray(torch.argmax(events_probability, dim = -1)[mask_next == 1], \
                                            events_next[mask_next == 1])
            f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')

        '''
        Calculate the NLL loss of p^*(m, t).
        L = -log \frac{\partial \Lambda^*(m, t)}{\partial t} + \Lambda^*(m, t)
        '''
        time_loss = self.nll_loss(intensity = intensity_for_each_event_from_tl_to_time_next, events_next = events_next, \
                                  intensity_integral = integral_for_each_event_from_tl_to_time_next, mask_next = mask_next)

        return time_loss, events_loss, mae, f1, the_number_of_events


    def nll_loss(self, intensity, intensity_integral, events_next, mask_next):
        '''
        This function calculates the NLL loss at each legit event in events_next.
    
        Args:
        * intensity           type: torch.tensor shape: [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
                              intensity values at $ t_i $
        * intensity_integral  type: torch.tensor shape: [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
                              intensity integral from $ t_{i - 1} $ to $ t_{i} $
        * events_next:        type: torch.tensor shape: [batch_size, seq_len]
                              The mark of the events that we need to predict.
        * mask_next:          type: torch.tensor shape: [batch_size, seq_len]
                              Needed mask to mask out unneeded loss values.
        
        Outputs:
        * loss                type: torch.tensor shape: [1]
                              the sum of NLL loss on all event.
        '''

        if self.event_toggle:
            intensity_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            log_intensity = torch.log(intensity + self.zero_shift_factor) * intensity_mask
            log_intensity = reduce(log_intensity, '... ne -> ...', 'sum')      # [batch_size, seq_len]
            intensity_integral = reduce(intensity_integral, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
            nll_p = -log_intensity + intensity_integral                        # [batch_size, seq_len]
        else:
            log_intensity = torch.log(intensity + self.zero_shift_factor)      # [batch_size, seq_len]
            nll_p = -log_intensity + intensity_integral                        # [batch_size, seq_len]
    
        loss = nll_p * mask_next
        # loss = torch.clamp(loss, max = 15)
        loss = torch.sum(loss)

        return loss


    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, var):
        '''

        '''
        self.eval()

        mae, pred_time = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                                  time_next = time_next, mask_next = mask_next, mean = mean, var = var)
                                                                               # 2 * [batch_size, seq_len]

        if self.event_toggle:
            pred_time = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
        '''
        preparing for multi-event training when needed
        '''
        pred_time.requires_grad = True
        integral_for_each_event = self.model(events_history, time_history, pred_time, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
        '''
        Obtains intensity values.
        '''
        intensity_for_each_event = torch.autograd.grad(
            outputs = integral_for_each_event,
            inputs = pred_time,
            grad_outputs = torch.ones_like(integral_for_each_event),
        )[0]
        pred_time.requires_grad = False
        assert intensity_for_each_event.shape == integral_for_each_event.shape
        check_tensor(intensity_for_each_event)                                 # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]

        '''
        Calculate the event loss, macro-F1, and other possible metrics measuring event prediction accuracy.
        This part is only available when event_toggle = True
        '''
        f1 = 0
        if self.event_toggle:
            probability_for_each_event = torch.log(intensity_for_each_event + self.zero_shift_factor)
                                                                               # [batch_size, seq_len, num_events]
            events_probability = torch.nn.functional.softmax(probability_for_each_event, dim = -1)
                                                                               # [batch_size, seq_len, num_events]

            events_pred_index, events_true = \
                move_from_tensor_to_ndarray(torch.argmax(events_probability, dim = -1)[mask_next == 1], \
                                            events_next[mask_next == 1])
            f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')

        return mae, f1


    def mean_absolute_error(self, events_history, time_history, time_next, mask_next, mean, var):
        '''
        MAE evaluation module.

        Args:
        * events_history  type: torch.tensor shape: [batch_size, seq_len]
                          Historical event sequences. Commonly, this sequence is a slice of 
                          the original event sequence from 0 to seq_len - 1(included). 
        * time_history    type: torch.tensor shape: [batch_size, seq_len]
                          Historical time sequences. Similar to events_history, we always generate
                          this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * time_next       type: torch.tensor shape: [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
                          When the next event actually happens. 
        * mask_next       type: torch.tensor shape: [batch_size, seq_len]
                          Needed mask to mask out unneeded loss values.
        * mean            type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * var             type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        Outputs:
        * mae             type: torch.tensor shape: [batch_size, seq_len]
                          MAE(Mean Absolute Error) between predicted time and ground truth.
        * tau_pred        type: torch.tensor shape: [batch_size, seq_len]
                          Time predicted by the sum of all intensity functions $ \lambda^*(m, t) $ over $ m $.
        '''

        def get_sum_of_integral(taus):
            '''
            Retrieve the sum of all $ \Lambda^*(m, t) $ over all $ m $ at $ \tau $.

            Outputs:
            * integral    type: torch.tensor shape: [batch_size, seq_len]
                          $ \sum_{n \in M}{\Lambda^*(n, \tau)} $
            '''

            if self.event_toggle:
                taus = repeat(taus, 'b s -> b s ne', ne = self.num_events)     # [batch_size, seq_len, num_events]
            integral = self.model(events_history, time_history, taus, mean, var)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
            if self.event_toggle:
                integral = integral.sum(dim = -1)                              # [batch_size, seq_len]
            
            return integral

        def bisect_target(taus):
            return get_sum_of_integral(taus) + \
                   torch.log(1 - torch.tensor(self.probability_threshold, device = self.device))
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len]
        tau_pred = median_prediction(l, r)                                     # [batch_size, seq_len]
        gap = (tau_pred - time_next) * mask_next                               # [batch_size, seq_len]
        mae = torch.abs(gap)                                                   # [batch_size, seq_len]

        return mae, tau_pred


    def mean_absolute_error_e(self, events_history, events_next, time_history, time_next, mask_next, mean, var):
        '''
        MAE-E evaluation module.

        Args:
        * events_history  type: torch.tensor shape: [batch_size, seq_len]
                          Historical event sequences. Commonly, this sequence is a slice of 
                          the original event sequence from 0 to seq_len - 1(included).
        * events_next     type: torch.tensor shape: [batch_size, seq_len]
                          The mark of the events that we need to predict.
        * time_history    type: torch.tensor shape: [batch_size, seq_len]
                          Historical time sequences. Similar to events_history, we always generate
                          this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * time_next       type: torch.tensor shape: [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
                          When the next event actually happens. 
        * mask_next       type: torch.tensor shape: [batch_size, seq_len]
                          Needed mask to mask out unneeded loss values.
        * mean            type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * var             type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        Outputs:
        * mae             type: torch.tensor shape: [batch_size, seq_len]
                          MAE(Mean Absolute Error) between predicted time and ground truth.
        * tau_pred        type: torch.tensor shape: [batch_size, seq_len]
                          Time predicted by the sum of all intensity functions $ \lambda^*(m, t) $ over $ m $.
        '''

        self.eval()

        '''
        Set the memory limit
        '''
        memory_ceiling = 1e7

        '''
        set a relatively large number as the infinity and decide resolution based on this large value and
        the memory_ceiling.
        '''
        if mean == 0 and var == 1:
            max_ = time_next.mean() + 10 * time_next.var()
        else:
            max_ = mean + 10 * var
        
        max_ = min(1e6, max_)
        time_next_inf = torch.ones_like(time_history, device = self.device) * max_
                                                                               # [batch_size, seq_len]
        resolution = max(int(max_ // 0.005), 100)

        _, seq_len = events_next.shape
        if seq_len * resolution * self.num_events > memory_ceiling:
            resolution = int(memory_ceiling // (seq_len * self.num_events))

        '''
        Step 1: obtain p^*(m) = \int_{t_l}^{+infty}{p(m, t)\dt}
        '''
        expand_integral_to_inf, expand_intensity_to_inf, time_interval \
                = self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, resolution, mean, var)
                                                                               # [batch_size, seq_len, resolution, num_events]

        '''
        Step 2: provide event predictions
        '''        
        expand_probability_per_event = expand_intensity_to_inf * torch.exp(-expand_integral_to_inf.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        expand_probability_per_event_for_monte_carlo = expand_probability_per_event[:, :, :-1, :]
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
        time_interval_used_for_monte_carlo = time_interval[:, :, 1:].unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution - 1, 1]
        probability_integral = expand_probability_per_event_for_monte_carlo * time_interval_used_for_monte_carlo
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
        p_m = reduce(probability_integral, 'b s r ne -> b s ne', 'sum')        # [batch_size, seq_len, num_events]
        probability_integral_sum = reduce(p_m, 'b s ne -> b s', 'sum')         # [batch_size, seq_len]
        predict_index = torch.argmax(p_m, dim = -1)                            # [batch_size, seq_len]

        '''
        Step 3: calculate macro-F1 and top-K accuracy
        '''
        f1 = []
        top_k_acc = []
        for (events_next_per_seq, p_m_per_seq) in zip(events_next, p_m):
            f1.append(f1_score(y_true = events_next_per_seq.detach().cpu(),
                               y_pred = torch.argmax(p_m_per_seq, dim = -1).detach().cpu(), average = 'macro'))
            
            top_k_acc_single_event_seq = []
            if self.num_events > 2:
                for k in range(1, self.num_events):
                    top_k_acc_single_event_seq.append(
                        top_k_accuracy_score(y_true = events_next_per_seq.detach().cpu(),
                                             y_score = p_m_per_seq.detach().cpu(),
                                             k = k,
                                             labels = np.arange(self.num_events))
                    )
            else:
                top_k_acc_single_event_seq.append(
                    accuracy_score(
                        y_true = events_next_per_seq.detach().cpu(),
                        y_pred = p_m_per_seq.detach().cpu()
                    )
                )
            top_k_acc.append(top_k_acc_single_event_seq)

        predict_index_one_hot_mask = torch.nn.functional.one_hot(predict_index.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        '''
        Step 4: get the time prediction for all, predicted, and real events.
        '''
        if mean == 0:
            resolution = max(min(int(time_next.mean().item() // 0.005), 500), 10)
        else:
            resolution = max(min(int(mean // 0.005), 500), 10)

        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history, p_m, \
                                                                  resolution, mean, var, max_)
                                                                               # [batch_size, seq_len, num_events]
        mae_per_event_with_predict_index = torch.abs(((tau_pred_all_event * predict_index_one_hot_mask).sum(dim = -1)) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
        mae_per_event_with_event_next = torch.abs(((tau_pred_all_event * events_next_one_hot_mask).sum(dim = -1)) - time_next) * mask_next
                                                                               # [batch_size, seq_len]

        mae_per_event_with_predict_index_avg = torch.sum(mae_per_event_with_predict_index, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_with_event_next_avg = torch.sum(mae_per_event_with_event_next, dim = -1) / mask_next.sum(dim = -1)

        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, \
               (mae_per_event_with_predict_index_avg, mae_per_event_with_event_next_avg), \
               (mae_per_event_with_predict_index, mae_per_event_with_event_next)


    def prediction_with_all_event_types(self, events_history, time_history, p_m, resolution, mean, var, max_val):
        '''
        The time prediction of every marker whose probability is not 0.

        Still, this function is currently buggy.

        Args:
        * events_history  type: torch.tensor shape: [batch_size, seq_len]
                          Historical event sequences. Commonly, this sequence is a slice of 
                          the original event sequence from 0 to seq_len - 1(included). 
        * time_history    type: torch.tensor shape: [batch_size, seq_len]
                          Historical time sequences. Similar to events_history, we always generate
                          this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * p_m             type: torch.tensor shape: [batch_size, seq_len]
                          the value of p(m) with given markers.
        * resolution      type: int shape: N/A
                          How many values do we need in each time interval [t_{i}, t_{i + 1}].
        * mask_next       type: torch.tensor shape: [batch_size, seq_len]
                          Needed mask to mask out unneeded loss values.
        * mean            type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * var             type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * max_val         type: float shape: N/A
                          The upper bound used in the bisect method.
        Outputs:
        * tau_pred        type: torch.tensor shape: [batch_size, seq_len]
                          Time predicted by the sum of all intensity functions $ \lambda^*(m, t) $ over $ m $.
        '''
        def evaluate_all_event(taus):
            '''
            placeholder
            '''
            # Train k FullyNN models for k different event types.
            integral_all_events, intensity_all_events, time_interval \
                    = self.model.integral_intensity_time_next_3d(events_history, time_history, taus, resolution, mean, var)
                                                                               # 2 * [batch_size, seq_len, resolution, num_events, num_events] + [batch_size, seq_len, resolution, num_events]
            event_mask = torch.diag(torch.ones(self.num_events, device = self.device))
                                                                               # [num_events, num_events]
            event_mask = repeat(event_mask, 'ne ne1 -> 1 1 1 ne ne1')          # [batch_size, seq_len, resolution, num_events, num_events]
            intensity_all_events = reduce(intensity_all_events * event_mask, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len, resolution, num_events]
            integral_all_events = reduce(integral_all_events, 'b s r ne ne1 -> b s r ne', 'sum')
                                                                               # [batch_size, seq_len, resolution, num_events]
            
            p_dist = intensity_all_events * torch.exp(-integral_all_events)    # [batch_size, seq_len, resolution, num_events]
            
            p_dist_for_monte_carlo = p_dist[:, :, :-1, :]                      # [batch_size, seq_len, resolution - 1, num_events]
            time_interval_for_monte_carlo = time_interval[:, :, 1:, :]         # [batch_size, seq_len, resolution - 1, num_events]
            probability = reduce(p_dist_for_monte_carlo * time_interval_for_monte_carlo, 'b s r ne -> b s ne', 'sum')
                                                                               # [batch_size, seq_len, num_events]
            return probability

        def bisect_target(taus):
            p_mt = evaluate_all_event(taus)                                    # [batch_size, seq_len, num_events]
            p_t_m = p_mt / p_m                                                 # [batch_size, seq_len, num_events]
            p_gap = p_t_m - self.probability_threshold                         # [batch_size, seq_len, num_events]

            return p_gap
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        r = max_val*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        tau_pred = median_prediction(l, r)                                     # [batch_size, seq_len, num_events]

        return tau_pred


    def plot(self, minibatch, opt):
        plot_type_to_functions = {
            'intensity': self.intensity,
            'integral': self.integral,
            'probability': self.probability,
            'debug': self.debug
        }
    
        return plot_type_to_functions[opt.plot_type](minibatch, opt)


    def extract_plot_data(self, minibatch):
        '''
        This function extracts input_time, input_events, input_intensity, mask, mean, and var from the minibatch.

        Args:
        * minibatch  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                     data structure: [[input_time, input_events, score, mask], (mean, var)]
        
        Outputs:
        * input_time    type: torch.tensor shape: [batch_size, seq_len + 1]
                        Raw event timestamp sequence.
        * input_events  type: torch.tensor shape: [batch_size, seq_len + 1]
                        Raw event marks sequence.
        * mask          type: torch.tensor shape: [batch_size, seq_len + 1]
                        Raw mask sequence.
        * mean          type: int shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        * var           type: int shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        '''
        input_time, input_events, _, mask, input_intensity = minibatch[0]
        mean, var = minibatch[1]

        return input_time, input_events, input_intensity, mask, mean, var


    def intensity(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]
        
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_intensity': expand_intensity,
            'input_intensity': input_intensity
            }
        plots = plot_intensity(data, timestamp, opt)
        
        return plots


    def integral(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]
        
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_integral': expand_integral,
            'input_intensity': input_intensity
            }
        plots = plot_integral(data, timestamp, opt)
        return plots


    def probability(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]

        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape
        expand_probability = expand_intensity * torch.exp(-expand_integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_probability': expand_probability,
            'input_intensity': input_intensity
            }
        plots = plot_probability(data, timestamp, opt)
        return plots


    def debug(self, input_data, opt):
        '''
        Args:
        time: [batch_size(always 1), seq_len + 1]
              The original dataset records. 
        resolution: int
              How many interpretive numbers we have between an event interval?
        '''
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1 = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                             time_next, mask_history, mask_next, mean, var)
                                                                               # [batch_size, seq_len]
        
        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_next, mean, var)

        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, opt.resolution, mean, var, mask_next)

        '''
        Append additional info into the data dict.
        '''
        data['events_next'] = events_next
        data['time_next'] = time_next
        data['mask_next'] = mask_next
        data['f1_after_time_pred'] = f1_1
        data['f1_before_time_pred'] = f1_2
        data['top_k'] = top_k
        data['probability_sum'] = probability_sum
        data['tau_pred_all_event'] = tau_pred_all_event
        data['mae_before_event'] = mae
        data['maes_after_event_avg'] = maes_avg
        data['maes_after_event'] = maes

        plots = plot_debug(data, timestamp, opt)

        return plots


    '''
    All static methods
    '''
    def train_step(model, minibatch, device):
        ''' 
        Epoch operation in training phase.
        The input minibatch comprise time sequences.

        Args:
            minibatch: [batch_size, seq_len]
                       contains [time_seq, event_seq, score, mask]
        '''
    
        [time_seq, event_seq, score, mask], (mean, var) = minibatch
        time_loss, events_loss, the_number_of_events = model(         
                input_time = time_seq, input_events = event_seq, mask = mask, \
                mean = mean, var = var, evaluate = False
        )

        time_loss.backward()
    
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return [time_loss, fact, events_loss]
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        [time_seq, event_seq, score, mask], (mean, var) = minibatch
        time_loss, events_loss, mae, f1, the_number_of_events = model(
                input_time = time_seq, input_events = event_seq,
                mask = mask, mean = mean, var = var, evaluate = True
        )
    
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return [time_loss, fact, events_loss, mae, f1]

    def postprocess(input, procedure):
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss, events loss]
            '''
            return [input[0], input[0] - input[1], input[2]]
        
        def test_postprocess(input):
            '''
            Evaluation process
            [absolute loss, relative loss, events loss, mae value]
            '''
            return [input[0], input[0] - input[1], input[2], input[3], input[4]]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    
    def log_print_format(input, procedure):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['events_loss'] = input[2]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f', \
                                         'events_loss': ':6.5f'}
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['events_loss'] = input[2]
            format_dict['mae'] = input[3]
            format_dict['f1_value'] = input[4]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f',
                                         'events_loss': ':6.5f', 'mae': ':2.8f', 'f1_value': ':2.8f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 5
    
    logfile_format = {'step': '', 'absolute loss': ':6.5f', 'relative loss': ':6.5f', 'events loss': ':6.5f', 'mae': ':2.8f', 'f1_value': ':2.8f'}

    def logfile_print_format(input):
        if len(input) == 3:
            format_dict = {}
            format_dict['absolute loss'] = input[0]
            format_dict['relative loss'] = input[1]
            format_dict['events loss'] = input[2]
            format_dict['mae'] = 0
            format_dict['f1_value'] = 0
        else:
            format_dict = {}
            format_dict['absolute loss'] = input[0]
            format_dict['relative loss'] = input[1]
            format_dict['events loss'] = input[2]
            format_dict['mae'] = input[3]
            format_dict['f1_value'] = input[4]
        return format_dict
    
    def choose_metric(evaluation_report, test_report):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [test_report[0],]
    
    metric_number = 1 # metric number is the length of the output of choose_metric