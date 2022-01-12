from ..utils import BasicModule
from .rmtpp import RMTPPModule

import torch

# We use syn dataloader for this model.
class RMTPP(BasicModule):
    def __init__(self, device, input_size, hidden_size, num_layers, dropout, num_events, output_size, increase):
        super(RMTPP, self).__init__()
        self.device = device
        self.increase = increase
        self.num_events = num_events

        self.submodel = RMTPPModule(input_size, hidden_size, num_layers, dropout, num_events, output_size, increase = increase, device = device)

    def forward(self, event, time):
        event_history, _ = self.divide_history_and_next(event, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]

        intensity, integral, mark, expectation, constant = self.submodel(event_history, time_history, time_next)

        return intensity, integral, mark, expectation, constant

    def divide_history_and_next(self, input, unsqueeze):
        history, next = input.clone()[:, :-1], input.clone()[:, 1:]
        if unsqueeze:
            history = history.unsqueeze(-1)                                    # [batch_size, seq_len, 1] or [batch_size, seq_len]
            next = next.unsqueeze(-1)                                          # [batch_size, seq_len, 1] or [batch_size, seq_len]
        return history, next

    def loss_f(self, intensity, integral, mark, expectation, time, event, mask):
        # temporal point process loss
        # intensity shape: [batch, seq_length]
        # so does tensor mask.
        _, event_next = self.divide_history_and_next(event, unsqueeze = False) # [batch_size, seq_len]
        _, time_next = self.divide_history_and_next(time, unsqueeze = True)    # [batch_size, seq_len, 1]
        _, mask_next = self.divide_history_and_next(mask, unsqueeze = False)   # [batch_size, seq_len]
        mask_next = mask_next.to(self.device)
        event_next = event_next.to(self.device)
        time_next = time_next.to(self.device)

        time_loss = -torch.log(intensity) + integral                           # [batch_size, seq_len, 1]
        time_loss_value = time_loss.clamp(min = -15, max = 15).squeeze(-1) * mask_next
        time_loss_value = time_loss_value.sum()
        # expectation loss
        expectation = torch.gather(expectation, -1, event_next.long().unsqueeze(-1))
                                                                               # [batch_size, seq_len, 1]
        expectation_loss = torch.nn.functional.mse_loss(expectation, time_next, reduction = 'none')
                                                                               # [batch_size, seq_len, 1]
        expectation_loss = expectation_loss.clamp(min = -15, max = 15).squeeze(-1) * mask_next
        expectation_loss = expectation_loss.sum()
    
        # mark loss
        # Event shape: [batch, seq_length]
        # mark:        [batch, seq_length, num_mark]
        if self.num_events > 1:
            event_loss = torch.nn.functional.cross_entropy(input = mark.transpose(1, 2), \
                                                           target = event_next.to(self.device).long(), \
                                                           reduction = 'none') # [batch_size, seq_len]
            event_loss_value = event_loss.clamp(min = -15, max = 15) * mask_next
            event_loss_value = event_loss_value.sum()
        else:
            event_loss_value = torch.tensor(0., device = self.device)

        return time_loss_value, expectation_loss, event_loss_value, mask_next.sum()

    def function_prober(self, data, resolution):
        time, event, _, _, _ = data                                               # 2 * [batch_size, seq_len + 1]
        event_history, _ = self.divide_history_and_next(event, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        intensity, integral, timestamp = self.submodel.intensity_integral(event_history, time_history, time_next, resolution)
                                                                               # 3 * [batch_size, seq_len * resolution]
        
        return integral, intensity, timestamp
    

    def train_step(model, minibatch, device):
        model.train()
        
        time, event, score, mask = minibatch                                   # 4 * [batch_size, seq_len + 1]
        intensity, integral, mark, expectation, constant = model(event, time)
        time_loss, expectation_loss, event_loss, the_number_of_events\
                                       = model.module.loss_f(intensity, integral,\
                                                             mark, expectation,\
                                                             time, event, mask)
        loss = time_loss + event_loss
        loss.backward()

        event_loss = event_loss.item() / the_number_of_events
        expectation_loss = expectation_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        constant_norm = torch.linalg.norm(constant).detach().item() / the_number_of_events
        time_loss_item = time_loss.item() / the_number_of_events

        return time_loss_item, fact, expectation_loss, event_loss, constant_norm

    def evaluation_step(model, minibatch, device):
        model.eval()

        time, event, score, mask = minibatch                                   # 4 * [batch_size, seq_len + 1]
        intensity, integral, mark, expectation, constant = model(event, time)
        time_loss, expectation_loss, event_loss, the_number_of_events\
                                       = model.module.loss_f(intensity, integral,\
                                                             mark, expectation,\
                                                             time, event, mask)

        fact = score.sum().item() / the_number_of_events
        time_loss_item = time_loss.item() / the_number_of_events
        expectation_loss = expectation_loss.item() / the_number_of_events
        event_loss = event_loss.item() / the_number_of_events
        constant_norm = torch.linalg.norm(constant).detach().item() / the_number_of_events

        return time_loss_item, fact, expectation_loss, event_loss, constant_norm

    def postprocess(input):
        return [input[0], input[0] - input[1], input[2], input[3], input[4]]

    def log_print_format(input):
        format_dict = {}
        format_dict['absolute_loss'] = input[0]
        format_dict['relative_loss'] = input[1]
        format_dict['expectation_loss'] = input[2]
        format_dict['event_loss'] = input[3]
        format_dict['constant_norm'] = input[4]
        format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'expectation_loss': ':8.5f', 'event_loss': ':8.5f', 'constant_norm': ':8.5f'}
        return format_dict
    
    format_dict_length = 5
    
    logfile_format = {'step': '', 'absolute loss': ':8.5f', 'relative loss': ':8.5f'}

    def logfile_print_format(input):
        format_dict = {}
        format_dict['absolute loss'] = input[0]
        format_dict['relative loss'] = input[1]
        return format_dict
    
    def choose_metric(evaluation_report, test_report):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        return [abs(evaluation_report[1]), abs(test_report[1])]
    
    metric_number = 2 # metric number is the length of the output of choose_metric