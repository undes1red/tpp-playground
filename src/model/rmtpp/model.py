from ..utils import BasicModule
from .rmtpp import Model

import torch

# We use IFL dataloader for this model
class RMTPP(BasicModule):
    def __init__(self, device, input_size, hidden_size, num_layers, dropout, num_events, output_size):
        super(RMTPP, self).__init__()
        self.device = device
        self.submodel = Model(input_size, hidden_size, num_layers, dropout, num_events, output_size).to(self.device)

    def forward(self, event, time):
        intensity, integral, mark = self.submodel(event, time)

        return intensity, integral, mark

    def train_step(model, minibatch, device):
        model.train()
        
        event, time, mask = minibatch[0]
        event = event[:, :-1]
        time = time[:, :-1]
        intensity, integral, mark = model(event, time)
        time_loss, event_loss = loss_f(intensity, integral, mark, event.to(device))
        loss = time_loss + event_loss
        loss.backward()

        fact = minibatch[1].sum().item()
        time_loss_item = time_loss.item()

        return time_loss_item, fact


    def evaluation_step(model, minibatch, device):
        model.eval()

        event, time, mask, = minibatch[0]
        event = event[:, :-1]
        time = time[:, :-1]
        intensity, integral, mark = model(event, time)
        time_loss, event_loss = loss_f(intensity, integral, mark, event.to(device))

        fact = minibatch[1].sum().item()
        time_loss_item = time_loss.item()

        return time_loss_item, fact

    def postprocess(input):
        return [input[0], input[0] - input[1]]

    def log_print_format(input):
        format_dict = {}
        format_dict['absolute_loss'] = input[0]
        format_dict['relative_loss'] = input[1]
        format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f'}
        return format_dict
    
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
        return [torch.abs(evaluation_report[-1]).item(), torch.abs(test_report[-1]).item()]
    
    metric_number = 2 # metric number is the length of the output of choose_metric


def loss_f(intensity, integral, mark, event):
    # temporal point process loss
    # intensity shape: [batch, seq_length]
    # so does tensor mask.
    time_loss = -torch.log(intensity) + integral
    time_loss_value = (time_loss).clamp(min = -15, max = 15).sum()
    # mark loss
    # Event shape: [batch, seq_length]
    # mark:        [batch, seq_length, num_mark]
    event_loss = torch.nn.functional.cross_entropy(input = mark.transpose(1, 2), target = event.long(), reduction = 'none')
    event_loss_value = (event_loss).clamp(min = -15, max = 15).sum()

    return time_loss_value, event_loss_value