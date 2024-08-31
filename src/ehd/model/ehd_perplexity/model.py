import torch, copy, os, importlib, math
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce, pack

from src.toolbox.misc import read_yaml, move_from_tensor_to_ndarray, check_tensor, get_logger
from src.utils import print_args

from src.ehd.model.ehd_perplexity.submodel import EHD_backend
from src.ehd.model.utils import BasicModule

from src.ehd.model.ehd_perplexity.plot import * 
from src.ehd.model.ehd_perplexity.nes.nes import NES
from src.ehd.utils import suffix

logger = get_logger(__name__)


class EHD(BasicModule):
    '''
    The EHD module.
    This module takes data and trained MTPP model, such as FullyNN, FENN, IFIB-C, etc.
    '''
    def __init__(self,
                 d_input, d_rnn, d_hidden,
                 n_layers_encoder, n_layers_decoder,
                 n_head, d_qk, d_v, dropout, perplexity_gap, loss_balancer, epsilon, 
                 opt, device, additional_model, samples_for_l_p = 32, training = True):
        super(EHD, self).__init__()
        self.device = device
        self.opt = opt
        # The probability gap is the ratio between p(x, H_{o, t_l} - H) and p(x, H_{o, t_l}).
        self.perplexity_gap = math.log(perplexity_gap)
        self.loss_balancer = loss_balancer
        self.samples_for_l_p = samples_for_l_p

        '''
        Load the trained TPP model checkpoint.
        '''
        self.opt.__dict__.update(additional_model)
        opt.abs_mtpp_model_config = os.path.join(opt.root_path, 'config', opt.used_procedure, opt.used_model_name, opt.used_model_config)
        opt.used_model_config = os.path.basename(opt.abs_mtpp_model_config)
        opt.abs_used_dataloader_config = os.path.join(opt.root_path, 'config', opt.procedure, opt.model_name, opt.used_dataloader_config) if opt.used_dataloader_config else None
        opt.used_dataloader_config = os.path.basename(opt.abs_used_dataloader_config) if opt.abs_used_dataloader_config else None
        model_hyperparameters = suffix(opt, 'used_model_name', 'used_lr', 'used_training_batch_size', 'used_n_training_steps', 'used_dataloader_config', 'used_model_config')
        folder_suffix = 'model_' + model_hyperparameters
        opt.mtpp_checkpoint_dir = os.path.join(opt.root_path, 'model', opt.used_procedure, opt.base_dataset_name, folder_suffix)
        self.load_model(training = training)

        '''
        Preparing the EHD model-agnostic part.
        '''
        self.epsilon = epsilon
        self.num_events = opt.info_dict['num_events']
        self.start_time = opt.info_dict['t_0']
        self.end_time = opt.info_dict['T']
        self.seq_len_x = opt.info_dict['length_of_x']
        self.seq_len_h = opt.info_dict['length_of_h']

        self.model = EHD_backend(num_events = self.num_events, seq_len_x = self.seq_len_x, seq_len_h = self.seq_len_h,
                                 d_input = d_input, d_rnn = d_rnn, d_hidden = d_hidden, n_layers_encoder = n_layers_encoder, 
                                 n_layers_decoder = n_layers_decoder, n_head = n_head, d_qk = d_qk, d_v = d_v, 
                                 dropout = dropout, device = device)

        self.nes_module = NES(num_of_samples_mask = self.samples_for_l_p, \
                              length_of_x = self.seq_len_x, last_event_in_x_is_dummy = True, \
                              length_of_h = self.seq_len_h, first_event_in_his_is_dummy = True, \
                              epsilon = self.epsilon, tau = 0.1, device = self.device)


    def load_model(self, training):
        # load the model from src.<used_procedure_name>.model
        mtpp_class = importlib.import_module('.' + self.opt.used_model_name, package = f'src.{self.opt.used_procedure}.model')
        model_class = mtpp_class.get_model()
        
        # Find the model_param of the loaded MTPP model
        model_param = read_yaml(self.opt.abs_mtpp_model_config)
        self.mtpp_model = model_class(device = self.opt.device, info_dict = self.opt.info_dict,
            **model_param
        )
        further_info = f'Now we load trained {self.opt.used_model_name} for p(x, H).' if training else f'Now we load the EHD checkpoint. MTPP backend: {self.opt.used_model_name}.'
        logger.info(f'MTPP Model created. {further_info}.')

        # Find and load the MTPP checkpoint.
        # We only need the MTPP checkpoint during training. The stored EHD model already contains the trained MTPP model, so
        # no need to load it again.
        if training:
            model_raw = torch.load(os.path.join(self.opt.mtpp_checkpoint_dir, 'checkpoint.chkpt'), map_location=self.device)
            model_state_dict = model_raw['model']
            self.mtpp_model.load_state_dict(model_state_dict)
    
            # Freeze the MTPP model.
            self.mtpp_model.requires_grad_(requires_grad = False)
    
            # Report the details of the loaded MTPP model.
            model_info = read_yaml(os.path.join(self.opt.mtpp_checkpoint_dir, 'model_card.yml'))
            logger.info(f'Recorded information of the trained MTPP model.')
            # logger.info(print_args(model_info, heading = 'Model Properties:'))
            logger.info(print_args(self.opt))
            total_trainable_params = sum(p.numel() for p in self.mtpp_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.mtpp_model.parameters())
            logger.info(f'MTPP Model restore completed. The number of trainable parameters in this model is {total_trainable_params} out of {total_params}.')


    def divide_history_and_future(self, input_time, input_events, input_mask):
        '''
        TODO: This function needs an overhaul to handle real-world datasets.
        I don't want to generate too much data from one sequence for memory and training speed concerns.
        Maybe at most around 50 generated data sequence from one original data sequence. 
        '''
        gen_time_history, gen_time_next = input_time[:, :self.seq_len_h + 1], input_time[:, self.seq_len_h + 1:]
        gen_events_history, gen_events_next = input_events[:, :self.seq_len_h + 1], input_events[:, self.seq_len_h + 1:]
        gen_mask_history, gen_mask_next = input_mask[:, :self.seq_len_h + 1], input_mask[:, self.seq_len_h + 1:]
        
        return (gen_time_history, gen_time_next), (gen_events_history, gen_events_next), (gen_mask_history, gen_mask_next)


    def divide_history_and_next(self, input):
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next


    def forward(self, task_name, *args, **kwargs):
        '''
        The entrance of the IFIB-C wrapper.
        
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
        task_mapper = {
            'train': self.train_procedure,
            'evaluate': self.evaluate_procedure,
            'lsp_and_lrp': self.get_lsp_and_lrp,
            'lsp_and_lrp_trend': self.lsp_and_lrp_trend,
            'graph': self.plot
        }

        return task_mapper[task_name](*args, **kwargs)
    

    def remove_dummy_event_from_mask(self, mask):
        '''
        Remove the probability of the dummy event by mask.
        '''
        mask_without_dummy = torch.zeros_like(mask)                            # [batch_size, seq_len - 1]
        for idx, mask_per_seq in enumerate(mask):
            dummy_index = mask_per_seq.sum() - 1
            mask_without_dummy_per_seq = copy.deepcopy(mask_per_seq.detach())
            mask_without_dummy_per_seq[dummy_index] = 0
            mask_without_dummy[idx] = mask_without_dummy_per_seq
        
        return mask_without_dummy


    def train_procedure(self, input_time, input_events, input_mask, mean, var):
        assert (input_mask == 1).all()
        (time_history, time_future), (events_history, events_future), (mask_history, mask_future) \
            = self.divide_history_and_future(input_time, input_events, input_mask)
                                                                               # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3
        number_of_events = mask_history.sum().item()
        events_embeddings = self.mtpp_model('ehd_event_emb', input_events)     # [batch_size, seq_len, d_history]
        cum_input_time = input_time.cumsum(dim = -1)                           # [batch_size, seq_len]
        batch_size = input_time.shape[0]

        (L_c, L_g), _, (padded_filtered_events, padded_filtered_masks), (padded_filtered_times, padded_filtered_event_embeddings) \
            = self.nes_module(self.model, (time_history, time_future, events_history, events_future, mask_history, mask_future, mean, var), \
                              mask_history, number_of_events, \
                              discrete_inputs = (input_events, input_mask), \
                              continuous_inputs = (cum_input_time, events_embeddings))
                                                                               # [samples_for_l_p * batch_size, *]

        padded_filtered_times = torch.diff(padded_filtered_times, prepend = torch.zeros(batch_size * self.samples_for_l_p, 1, device = self.device))
                                                                               # [samples_for_l_p * batch_size, *]
        # Loss 3 for asking the model to find the most important events.
        # rebuild the original history for H_{o,t_l} - H_{s,o,t_l} based on history_mask.
        # You should be really careful to implement this part for not accidentally dropping any gradients.
        log_p_h_o_t_l_x_o_mean = self.mtpp_model('ehd_perplexity', input_time, input_events, events_embeddings, input_mask, self.seq_len_x, mean, var)
                                                                               # [batch_size]
        log_p_h_r_o_t_l_x_o_mean = self.mtpp_model('ehd_perplexity', padded_filtered_times, padded_filtered_events,
                                                    padded_filtered_event_embeddings, padded_filtered_masks, 
                                                    self.seq_len_x, mean, var) # [samples_for_l_p * batch_size]

        L_p = F.relu(
            repeat(log_p_h_o_t_l_x_o_mean, 'b -> s b', s = self.samples_for_l_p).flatten() - log_p_h_r_o_t_l_x_o_mean - self.perplexity_gap
            ).mean()
        # Loss = self.loss_balancer * L_c + L_p
        Loss = self.loss_balancer * L_c

        return Loss, L_c, L_p, L_g


    def evaluate_procedure(self, input_time, input_events, input_mask, mean, var, percentage = True):
        '''
        Since we removed all sequence shorter than seq_len_x + seq_len_h.
        We do not need to worry about the input_mask anymore.
        '''
        assert (input_mask == 1).all()
        (time_history, time_future), (events_history, events_future), (mask_history, mask_future) \
            = self.divide_history_and_future(input_time, input_events, input_mask)
                                                                               # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3
        number_of_events = mask_history.sum().item()

        events_embeddings = self.mtpp_model('ehd_event_emb', input_events)     # [batch_size, seq_len, d_history]
        cum_input_time = input_time.cumsum(dim = -1)                           # [batch_size, seq_len]
        batch_size = input_time.shape[0]

        (L_c, L_g), history_mask, (padded_filtered_events, padded_filtered_masks), (padded_filtered_times, padded_filtered_event_embeddings) \
            = self.nes_module(self.model, (time_history, time_future, events_history, events_future, mask_history, mask_future, mean, var), \
                              mask_history, number_of_events, \
                              discrete_inputs = (input_events, input_mask), \
                              continuous_inputs = (cum_input_time, events_embeddings), evaluate = True)
        padded_filtered_times = torch.diff(padded_filtered_times, dim = -1, prepend = torch.zeros(batch_size, 1, device = self.device))

        _, history_mask, \
           (padded_selected_filtered_events, padded_selected_filtered_masks), \
           (padded_selected_filtered_times, padded_selected_filtered_event_embeddings) \
            = self.nes_module(self.model, (time_history, time_future, events_history, events_future, mask_history, mask_future, mean, var), \
                              mask_history, number_of_events, \
                              discrete_inputs = (input_events, input_mask), \
                              continuous_inputs = (cum_input_time, events_embeddings), evaluate = True, pick_essential_events = True)
        padded_selected_filtered_times = torch.diff(padded_selected_filtered_times, dim = -1, prepend = torch.zeros(batch_size, 1, device = self.device))

        # Loss 3 for asking the model to find the most important events.
        # rebuild the original history for H_{o,t_l} - H_{s,o,t_l} based on history_mask.
        # You should be really careful to implement this part for not accidentally dropping any gradients.
        log_p_h_o_t_l_x_o_mean = self.mtpp_model('ehd_perplexity', input_time, input_events, events_embeddings, input_mask, self.seq_len_x, mean, var)
                                                                               # [batch_size]
        log_p_h_r_o_t_l_x_o_mean = self.mtpp_model('ehd_perplexity', padded_filtered_times, padded_filtered_events,
                                                   padded_filtered_event_embeddings, padded_filtered_masks, 
                                                   self.seq_len_x, mean, var)  # [batch_size]
        log_p_h_s_o_t_l_x_o_mean = self.mtpp_model('ehd_perplexity', padded_selected_filtered_times, padded_selected_filtered_events,
                                                   padded_selected_filtered_event_embeddings, padded_selected_filtered_masks, 
                                                   self.seq_len_x, mean, var)  # [batch_size]

        L_p = F.relu(log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_r_o_t_l_x_o_mean - self.perplexity_gap).mean()

        # Loss = self.loss_balancer * L_c + L_p
        Loss = L_p

        '''
        Evaluation part.
        '''
        # part 1: How many percents of events are left?
        discrete_remained_mask = history_mask[..., 0].detach().int()
        the_number_of_remained_events = discrete_remained_mask.sum(dim = -1)
        # the_number_of_total_events = mask_history.sum(dim = -1)

        if percentage:
            percentage_remained_events = the_number_of_remained_events.float().mean()
            L_rp = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_r_o_t_l_x_o_mean).mean()
            L_sp = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_s_o_t_l_x_o_mean).mean()
        else:
            percentage_remained_events = the_number_of_remained_events
            L_rp = log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_r_o_t_l_x_o_mean
            L_sp = log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_s_o_t_l_x_o_mean


        return Loss, L_c, L_p, L_g, percentage_remained_events, L_sp, L_rp


    def filter(self, input_time, input_events, events_embeddings, input_mask, filter_mask, evaluate = False, output_removed_events = False):
        '''
        Now, filter() should provide \\mathcal{H}_{s,o,t_l} and \\mathcal{H}_{r,o,t_l} when evaluate = True.
        filter still only provide \\mathcal{H}_{r,o,t_l} when evaluate = False.
        '''
        '''
        Please be careful: the mean and var should come from the training dataset!
        '''
        assert filter_mask is not None, "You want to filter the existing history following the filter mask, but filter mask is unavailable!"
        assert torch.is_tensor(filter_mask), "The filter mask has to be a pytorch tensor!"
        if not evaluate:
            assert filter_mask.requires_grad, "The filter mask must be differentiable!"
        samples_for_l_p, batch_size = filter_mask.shape[0], filter_mask.shape[1]

        '''
        Dealing with time.
        We select the time whose history[:, :, 0] == 1(meaning this event will remain).
        '''
        filter_mask_for_nominated = filter_mask[..., 1 if output_removed_events else 0]
                                                                               # [samples_for_l_p, batch_size, seq_len]

        '''
        Why this works?
        We generate the history_mark with Gumbel-softmax trick with zero temperature.
        That enforce the possible values of history_mark is either 1 or 0, although the data type is float.
        We use discrete_history_mask_for_nominated for data selection after we multiply history_mask_for_nominated
        with the input sequence data to introduce the gradient of mask to the selected data sequence.
        Caveat: We convert the float tensor history_mask_for_nominated to LongTensor because we ensure this tensor only contains
        0 and 1. DO NOT do this if your float tensor contains non-integers!
        '''
        discrete_filter_mask_for_nominated = filter_mask[..., 1 if output_removed_events else 0].detach().int()
                                                                               # [samples_for_l_p, batch_size, seq_len]
        the_number_of_remained_event = discrete_filter_mask_for_nominated.sum(dim = -1)
                                                                               # [samples_for_l_p, batch_size]
                
        repeated_input_time = repeat(input_time, '... -> n ...', n = samples_for_l_p)
                                                                               # [samples_for_l_p, batch_size, seq_len]
        repeated_input_events = repeat(input_events, '... -> n ...', n = samples_for_l_p)
                                                                               # [samples_for_l_p, batch_size, seq_len]
        repeated_events_embeddings = repeat(events_embeddings, '... -> n ...', n = samples_for_l_p)
                                                                               # [samples_for_l_p, batch_size, seq_len, d_history]
        repeated_input_mask = repeat(input_mask, '... -> n ...', n = samples_for_l_p)
                                                                               # [samples_for_l_p, batch_size, seq_len]
        
        repeated_cumsum_time = repeated_input_time.cumsum(dim = -1)            # [samples_for_l_p, batch_size, seq_len]
        
        # select the remained events from the original input.
        selected_time = repeated_cumsum_time * filter_mask_for_nominated       # [samples_for_l_p, batch_size, seq_len]
        selected_time = selected_time[discrete_filter_mask_for_nominated == 1] # [...]
        selected_input_events = repeated_input_events[discrete_filter_mask_for_nominated == 1]
                                                                               # [...]
        selected_events_embeddings = repeated_events_embeddings * filter_mask_for_nominated.unsqueeze(dim = -1)
                                                                               # [samples_for_l_p, batch_size, seq_len, d_history]
        selected_events_embeddings = selected_events_embeddings[discrete_filter_mask_for_nominated == 1]
                                                                               # [..., d_history]
        selected_input_mask = repeated_input_mask[discrete_filter_mask_for_nominated == 1]
                                                                               # [...]
        
        data_start_index = 0
        all_reshaped_time, all_reshaped_input_events, all_reshaped_events_embeddings, all_reshaped_input_mask \
            = [], [], [], []
        for the_number_of_remained_event_per_batch in the_number_of_remained_event:
            '''
            Padding the selected timestamps.
            '''
            reshaped_time, reshaped_input_events, reshaped_events_embeddings, reshaped_input_mask \
                = [], [], [], []
            for the_number_of_remained_event_per_batch_per_seq in the_number_of_remained_event_per_batch:
                reshaped_time.append(selected_time[data_start_index:data_start_index + the_number_of_remained_event_per_batch_per_seq])
                reshaped_input_events.append(selected_input_events[data_start_index:data_start_index + the_number_of_remained_event_per_batch_per_seq])
                reshaped_events_embeddings.append(selected_events_embeddings[data_start_index:data_start_index + the_number_of_remained_event_per_batch_per_seq, :])
                reshaped_input_mask.append(selected_input_mask[data_start_index:data_start_index + the_number_of_remained_event_per_batch_per_seq])

                data_start_index += the_number_of_remained_event_per_batch_per_seq
                        
            padded_reshaped_time = torch.nn.utils.rnn.pad_sequence(reshaped_time, batch_first = True)
                                                                               # [batch_size, padded_seq_len]
            padded_input_events = torch.nn.utils.rnn.pad_sequence(reshaped_input_events, batch_first = True)
                                                                               # [batch_size, padded_seq_len]
            padded_events_embeddings = torch.nn.utils.rnn.pad_sequence(reshaped_events_embeddings, batch_first = True)
                                                                               # [batch_size, padded_seq_len, d_history]
            padded_input_mask = torch.nn.utils.rnn.pad_sequence(reshaped_input_mask, batch_first = True)
                                                                               # [batch_size, padded_seq_len]
            
            padded_reshaped_time = padded_reshaped_time.diff(dim = -1, prepend = torch.zeros(batch_size, 1, device = self.device))
                                                                               # [batch_size, padded_seq_len]
            all_reshaped_time.append(padded_reshaped_time)
            all_reshaped_input_events.append(padded_input_events)
            all_reshaped_events_embeddings.append(padded_events_embeddings)
            all_reshaped_input_mask.append(padded_input_mask)

            del reshaped_time, reshaped_input_events, reshaped_events_embeddings, reshaped_input_mask
            del padded_reshaped_time, padded_input_events, padded_events_embeddings, padded_input_mask
        
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_reshaped_time, all_reshaped_input_events, all_reshaped_events_embeddings, all_reshaped_input_mask


    def plot(self, minibatch, opt):
        plot_type_to_functions = {
            'removed_events': self.removed_events
        }
    
        return plot_type_to_functions[opt.plot_type](minibatch, opt)


    def extract_plot_data(self, minibatch):
        '''
        This function extracts input_time, input_events, input_intensity, mask, mean, and var from the minibatch.
        Caution: dataloader won't add the end dummy event during evaluation!

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
        input_time, input_events, _, mask = minibatch[0]
        mean, var = minibatch[1]

        return input_time, input_events, mask, mean, var


    def removed_events(self, input_data, opt):
        '''
        Since we removed all sequence shorter than seq_len_x + seq_len_h.
        We do not need to worry about the input_mask anymore.
        '''

        '''
        Extract data from the input minibatch.
        '''
        input_time, input_events, input_mask, mean, var = self.extract_plot_data(input_data)

        assert (input_mask == 1).all()
        (time_history, time_future), (events_history, events_future), (mask_history, mask_future) \
            = self.divide_history_and_future(input_time, input_events, input_mask)
                                                                               # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3

        # Here, mask = 1: important. Removing them would cause counterfactual results.
        #       mask = 0: noises or unrelated events. Keeping them makes no benefit for modeling the future.
        generated_mask_probability = self.model(time_history, time_future, events_history, events_future, \
                                                mask_history, mask_future, mean, var)
                                                                               # [batch_size, seq_len_h + 1, 2]
        check_tensor(generated_mask_probability)

        # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.u
        history_mask = F.one_hot(torch.argmax(generated_mask_probability, dim = -1), num_classes = 2)
                                                                               # [batch_size, seq_len_h + 1, 2]
        history_mask[:, 0] = 1
        check_tensor(history_mask)
        
        future_mask = torch.ones(*mask_future.shape, 2, device = self.device)  # [batch_size, seq_len_x + 1, 2]

        filter_mask, _ = pack((history_mask, future_mask), 'b * m')            # [batch_size, seq_len_h + seq_len_x + 2, 2]
        filter_mask = repeat(filter_mask, 'b l m -> n b l m', n = 1)           # [1, batch_size, seq_len_h + seq_len_x + 2, 2]

        events_embeddings = self.mtpp_model('ehd_event_emb', input_events)     # [batch_size, seq_len, d_history]
        padded_filtered_time, padded_filtered_events, padded_filtered_event_embeddings, padded_filtered_masks \
            = self.filter(input_time = input_time, input_events = input_events, events_embeddings = events_embeddings, \
                          input_mask = input_mask, filter_mask = filter_mask, evaluate = True)
                                                                               # [1, batch_size, seq_len_h + seq_len_x + 2] * 2 + [samples_for_l_p, batch_size, seq_len_h + seq_len_x + 2, d_history] + [samples_for_l_p, batch_size, seq_len_h + seq_len_x + 2]

        padded_filtered_removed_time, padded_filtered_removed_events, padded_filtered_event_removed_embeddings, padded_filtered_removed_masks \
            = self.filter(input_time = input_time, input_events = input_events, events_embeddings = events_embeddings, \
                          input_mask = input_mask, filter_mask = filter_mask, evaluate = True, output_removed_events = True)
                                                                               # [1, batch_size, seq_len_h + seq_len_x + 2] * 2 + [samples_for_l_p, batch_size, seq_len_h + seq_len_x + 2, d_history] + [samples_for_l_p, batch_size, seq_len_h + seq_len_x + 2]

        # Loss 3 for asking the model to find the most important events.
        # rebuild the original history for H_{o,t_l} - H_{s,o,t_l} based on history_mask.
        # You should be really careful to implement this part for not accidentally dropping any gradients.
        log_p_h_o_t_l_x_o_mean = self.mtpp_model('ehd_perplexity', input_time, input_events, events_embeddings, input_mask, \
                                                 self.seq_len_x, mean, var)
                                                                               # [batch_size]
        
        log_p_h_r_o_t_l_x_o_mean = []
        for padded_filtered_time_per_sample, padded_filtered_events_per_sample, \
            padded_filtered_event_embeddings_per_sample, padded_filtered_masks_per_sample in \
            zip(padded_filtered_time, padded_filtered_events, padded_filtered_event_embeddings, padded_filtered_masks):
            log_p_h_r_o_t_l_x_o_mean.append(self.mtpp_model('ehd_perplexity', padded_filtered_time_per_sample, padded_filtered_events_per_sample,
                                                       padded_filtered_event_embeddings_per_sample, padded_filtered_masks_per_sample, 
                                                       self.seq_len_x, mean, var))
                                                                               # [batch_size]
        log_p_h_r_o_t_l_x_o_mean = torch.stack(log_p_h_r_o_t_l_x_o_mean, dim = 0)
                                                                               # [1, batch_size]

        L_rp = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_r_o_t_l_x_o_mean).mean().item()

        '''
        Evaluation part.
        '''
        # part 1: How many percents of events are left?
        discrete_remained_mask = history_mask[..., 0].detach().int()
        the_number_of_remained_events = discrete_remained_mask.sum(dim = -1)
        the_number_of_total_events = mask_history.sum(dim = -1)
        percentage_remained_events = (the_number_of_remained_events / the_number_of_total_events).mean().item()


        # part 2: What is the value of log_p_h_s_o_t_l_x_o_mean?
        log_p_h_s_o_t_l_x_o_mean = []
        for padded_filtered_removed_time_per_sample, padded_filtered_removed_events_per_sample, \
            padded_filtered_removed_event_embeddings_per_sample, padded_filtered_removed_masks_per_sample in \
            zip(padded_filtered_removed_time, padded_filtered_removed_events, padded_filtered_event_removed_embeddings, padded_filtered_removed_masks):
            log_p_h_s_o_t_l_x_o_mean.append(self.mtpp_model('ehd_perplexity', padded_filtered_removed_time_per_sample, padded_filtered_removed_events_per_sample,
                                                       padded_filtered_removed_event_embeddings_per_sample, padded_filtered_removed_masks_per_sample, 
                                                       self.seq_len_x, mean, var))
                                                                               # [batch_size]
        log_p_h_s_o_t_l_x_o_mean = torch.stack(log_p_h_s_o_t_l_x_o_mean, dim = 0)
                                                                               # [1, batch_size]

        L_sp = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_s_o_t_l_x_o_mean).mean().item()


        # Comparison with random removal.
        # we i.i.d. sample the mask multiple times to eliminate serendipity.
        number_of_sampled_sequence = 16
        input_time_random = repeat(input_time, 'b ... -> (nss b) ...', nss = number_of_sampled_sequence)
                                                                               # 
        input_events_random = repeat(input_events, 'b ... -> (nss b) ...', nss = number_of_sampled_sequence)
        events_embeddings_random = repeat(events_embeddings, 'b ... -> (nss b) ...', nss = number_of_sampled_sequence)
        input_mask_random = repeat(input_mask, 'b ... -> (nss b) ...', nss = number_of_sampled_sequence)

        for the_number_of_remained_events_per_seq in the_number_of_remained_events:
            rand_mat = torch.rand(number_of_sampled_sequence, self.seq_len_h, device = self.device)
                                                                               # [number_of_sampled_sequence, seq_len_h]
            k_th_quant = torch.topk(rand_mat, the_number_of_remained_events_per_seq - 1, largest = False)[0][:,-1:]
                                                                               # [number_of_sampled_sequence, 1]
            if the_number_of_remained_events_per_seq == 1:
                mask = torch.ones_like(rand_mat, device = self.device).long()  # [number_of_sampled_sequence, seq_len_h]
            else:
                mask = (rand_mat > k_th_quant).long()                          # [number_of_sampled_sequence, seq_len_h]
            generated_mask_probability_random = F.one_hot(mask, num_classes = 2)
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(generated_mask_probability_random)

            # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.
            history_mask_random = F.one_hot(torch.argmax(generated_mask_probability_random, dim = -1), num_classes = 2)
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
            history_mask_random, _ = pack((torch.ones(number_of_sampled_sequence, 1, 2, device = self.device), history_mask_random), 'nss * m')
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(history_mask_random)
            
            future_mask_random = torch.ones(number_of_sampled_sequence, self.seq_len_x + 1, 2, device = self.device)
                                                                               # [number_of_sampled_sequence, seq_len_x + 1, 2]
    
            filter_mask_random, _ = pack((history_mask_random, future_mask_random), 'nss * m')
                                                                               # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
            filter_mask_random = repeat(filter_mask_random, 'b l m -> n b l m', n = 1)
                                                                               # [1, number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
    
            events_embeddings = self.mtpp_model('ehd_event_emb', input_events) # [number_of_sampled_sequence, seq_len, d_history]
            padded_filtered_time_random, padded_filtered_events_random, padded_filtered_event_embeddings_random, padded_filtered_masks_random \
                = self.filter(input_time = input_time_random, input_events = input_events_random, events_embeddings = events_embeddings_random, \
                              input_mask = input_mask_random, filter_mask = filter_mask_random, evaluate = True)
                                                                               # [1, number_of_sampled_sequence, seq_len_h + seq_len_x + 2] * 2 + [samples_for_l_p, number_of_sampled_sequence, seq_len_h + seq_len_x + 2, d_history] + [samples_for_l_p, number_of_sampled_sequence, seq_len_h + seq_len_x + 2]
    
            padded_filtered_removed_time_random, padded_filtered_removed_events_random, padded_filtered_event_removed_embeddings_random, padded_filtered_removed_masks_random \
                = self.filter(input_time = input_time_random, input_events = input_events_random, events_embeddings = events_embeddings_random, \
                              input_mask = input_mask_random, filter_mask = filter_mask_random, evaluate = True, output_removed_events = True)
                                                                               # [1, number_of_sampled_sequence, seq_len_h + seq_len_x + 2] * 2 + [samples_for_l_p, number_of_sampled_sequence, seq_len_h + seq_len_x + 2, d_history] + [samples_for_l_p, number_of_sampled_sequence, seq_len_h + seq_len_x + 2]
    
            # Loss 3 for asking the model to find the most important events.
            # rebuild the original history for H_{o,t_l} - H_{s,o,t_l} based on history_mask.
            # You should be really careful to implement this part for not accidentally dropping any gradients.
            log_p_h_o_t_l_x_o_mean = self.mtpp_model('ehd_perplexity', input_time, input_events, events_embeddings, input_mask, self.seq_len_x, mean, var)
                                                                               # [number_of_sampled_sequence]
            
            log_p_h_r_o_t_l_x_o_mean_random = []
            for padded_filtered_time_per_sample, padded_filtered_events_per_sample, \
                padded_filtered_event_embeddings_per_sample, padded_filtered_masks_per_sample in \
                zip(padded_filtered_time_random, padded_filtered_events_random, padded_filtered_event_embeddings_random, padded_filtered_masks_random):
                log_p_h_r_o_t_l_x_o_mean_random.append(self.mtpp_model('ehd_perplexity', padded_filtered_time_per_sample, padded_filtered_events_per_sample,
                                                           padded_filtered_event_embeddings_per_sample, padded_filtered_masks_per_sample, 
                                                           self.seq_len_x, mean, var))
                                                                               # [number_of_sampled_sequence]
            log_p_h_r_o_t_l_x_o_mean_random = torch.stack(log_p_h_r_o_t_l_x_o_mean_random, dim = 0)
                                                                               # [1, number_of_sampled_sequence]
    
            L_rp_r = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_r_o_t_l_x_o_mean_random).mean().item()
    
            '''
            Evaluation part of EHD_random
            '''
            # part 1: What is the value of log_p_h_s_o_t_l_x_o_mean?
            log_p_h_s_o_t_l_x_o_mean_random = []
            for padded_filtered_removed_time_per_sample, padded_filtered_removed_events_per_sample, \
                padded_filtered_removed_event_embeddings_per_sample, padded_filtered_removed_masks_per_sample in \
                zip(padded_filtered_removed_time_random, padded_filtered_removed_events_random, padded_filtered_event_removed_embeddings_random, padded_filtered_removed_masks_random):
                log_p_h_s_o_t_l_x_o_mean_random.append(self.mtpp_model('ehd_perplexity', padded_filtered_removed_time_per_sample, padded_filtered_removed_events_per_sample,
                                                           padded_filtered_removed_event_embeddings_per_sample, padded_filtered_removed_masks_per_sample, 
                                                           self.seq_len_x, mean, var))
                                                                               # [number_of_sampled_sequence]
            log_p_h_s_o_t_l_x_o_mean_random = torch.stack(log_p_h_s_o_t_l_x_o_mean_random, dim = 0)
                                                                               # [1, number_of_sampled_sequence]
    
            L_sp_r = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_s_o_t_l_x_o_mean_random).mean().item()


        data = {
            'percentage_remained_events': percentage_remained_events,
            'generated_mask_probability': generated_mask_probability.detach().cpu(),
            'L_sp': L_sp,
            'L_sp_r': L_sp_r,
            'L_rp': L_rp,
            'L_rp_r': L_rp_r,
            'events_history': events_history.detach().cpu(),
            'events_future': events_future.detach().cpu(),
            'time_history': time_history.detach().cpu(),
            'time_future': time_future.detach().cpu(),
            'filter_mask': filter_mask.squeeze(dim = 0).detach().cpu()
        }

        plots = plot_removed_events(data, opt)


        return plots
    

    def get_lsp_and_lrp(self, input_data, opt, fast = False):
        '''
        Since we removed all sequence shorter than seq_len_x + seq_len_h.
        We do not need to worry about the input_mask anymore.

        Update: we merge ehd_random here because ehd_random should remove the same or close amount of
        event from the original sequence. Setting up a dedicated ehd_random module won't work because the
        number of removed event varies with the sequence length.
        '''
        import time

        '''
        Extract data from the input minibatch.
        '''

        input_time, input_events, input_mask, mean, var = self.extract_plot_data(input_data)

        assert (input_mask == 1).all()
        (time_history, time_future), (events_history, events_future), (mask_history, mask_future) \
            = self.divide_history_and_future(input_time, input_events, input_mask)
                                                                               # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3
        if fast:
            start = time.time()

            # Here, mask = 1: important. Removing them would cause counterfactual results.
            #       mask = 0: noises or unrelated events. Keeping them makes no benefit for modeling the future.
            generated_mask_probability = self.model(time_history, time_future, events_history, events_future, \
                                                    mask_history, mask_future, mean, var)
                                                                               # [batch_size, seq_len_h + 1, 2]
            check_tensor(generated_mask_probability)

            # Loss 3 for asking the model to find the best sequence to remove.
            # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.
            history_mask = F.one_hot(torch.argmax(generated_mask_probability, dim = -1), num_classes = 2)
                                                                               # [batch_size, seq_len_h + 1, 2]
            history_mask[:, 0] = 1
            check_tensor(history_mask)

            future_mask = torch.ones(*mask_future.shape, 2, device = self.device)
                                                                               # [batch_size, seq_len_x + 1, 2]

            filter_mask, _ = pack((history_mask, future_mask), 'b * m')        # [batch_size, seq_len_h + seq_len_x + 2, 2]
            filter_mask = repeat(filter_mask, 'b l m -> n b l m', n = 1)       # [1, batch_size, seq_len_h + seq_len_x + 2, 2]

            events_embeddings = self.mtpp_model('ehd_event_emb', input_events) # [batch_size, seq_len, d_history]
            padded_filtered_time, padded_filtered_events, padded_filtered_event_embeddings, padded_filtered_masks \
                = self.filter(input_time = input_time, input_events = input_events, events_embeddings = events_embeddings, \
                              input_mask = input_mask, filter_mask = filter_mask, evaluate = True)
                                                                               # [1, batch_size, seq_len_h + seq_len_x + 2] * 2 + [samples_for_l_p, batch_size, seq_len_h + seq_len_x + 2, d_history] + [samples_for_l_p, batch_size, seq_len_h + seq_len_x + 2]

            padded_filtered_removed_time, padded_filtered_removed_events, padded_filtered_event_removed_embeddings, padded_filtered_removed_masks \
                = self.filter(input_time = input_time, input_events = input_events, events_embeddings = events_embeddings, \
                              input_mask = input_mask, filter_mask = filter_mask, evaluate = True, output_removed_events = True)
                                                                               # [1, batch_size, seq_len_h + seq_len_x + 2] * 2 + [samples_for_l_p, batch_size, seq_len_h + seq_len_x + 2, d_history] + [samples_for_l_p, batch_size, seq_len_h + seq_len_x + 2]

            # Loss 3 for asking the model to find the most important events.
            # rebuild the original history for H_{o,t_l} - H_{s,o,t_l} based on history_mask.
            # You should be really careful to implement this part for not accidentally dropping any gradients.
            log_p_h_o_t_l_x_o_mean = self.mtpp_model('ehd_perplexity', input_time, input_events, events_embeddings, input_mask, self.seq_len_x, mean, var)
                                                                               # [batch_size]
        
            log_p_h_r_o_t_l_x_o_mean = []
            for padded_filtered_time_per_sample, padded_filtered_events_per_sample, \
                padded_filtered_event_embeddings_per_sample, padded_filtered_masks_per_sample in \
                zip(padded_filtered_time, padded_filtered_events, padded_filtered_event_embeddings, padded_filtered_masks):
                log_p_h_r_o_t_l_x_o_mean.append(self.mtpp_model('ehd_perplexity', padded_filtered_time_per_sample, padded_filtered_events_per_sample,
                                                           padded_filtered_event_embeddings_per_sample, padded_filtered_masks_per_sample, 
                                                           self.seq_len_x, mean, var))
                                                                               # [batch_size]
            log_p_h_r_o_t_l_x_o_mean = torch.stack(log_p_h_r_o_t_l_x_o_mean, dim = 0)
                                                                               # [1, batch_size]

            '''
            Evaluation part.
            '''
            # part 1: How many percents of events are left?
            discrete_remained_mask = history_mask[..., 0].detach().int()
            the_number_of_remained_events = discrete_remained_mask.sum(dim = -1)
            the_number_of_total_events = mask_history.sum(dim = -1)
    
            # part 2: What is the value of log_p_h_s_o_t_l_x_o_mean?
            log_p_h_s_o_t_l_x_o_mean = []
            for padded_filtered_removed_time_per_sample, padded_filtered_removed_events_per_sample, \
                padded_filtered_removed_event_embeddings_per_sample, padded_filtered_removed_masks_per_sample in \
                zip(padded_filtered_removed_time, padded_filtered_removed_events, padded_filtered_event_removed_embeddings, padded_filtered_removed_masks):
                log_p_h_s_o_t_l_x_o_mean.append(self.mtpp_model('ehd_perplexity', padded_filtered_removed_time_per_sample, padded_filtered_removed_events_per_sample,
                                                           padded_filtered_removed_event_embeddings_per_sample, padded_filtered_removed_masks_per_sample, 
                                                           self.seq_len_x, mean, var))
                                                                               # [batch_size]
            log_p_h_s_o_t_l_x_o_mean = torch.stack(log_p_h_s_o_t_l_x_o_mean, dim = 0)
                                                                               # [1, batch_size]

            L_rp = log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_r_o_t_l_x_o_mean
            L_sp = log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_s_o_t_l_x_o_mean

            end = time.time()
            time_ehd_mtpp = end - start
            
            percentage_remained_events = the_number_of_remained_events.float().mean().item()
            L_sp = L_sp.item()
            L_rp = L_rp.item()
            the_number_of_total_events = mask_history.sum(dim = -1)
            # Comparison with random removal.
            # we i.i.d. sample the mask multiple times to eliminate serendipity.

            start = time.time()
            number_of_sampled_sequence = 16
            for the_number_of_remained_events_per_seq, the_number_of_historical_events_per_seq \
                in zip(the_number_of_remained_events, mask_history.sum(dim = -1)):
                # baseline 1, random removal.
                rand_mat = torch.rand(number_of_sampled_sequence, self.seq_len_h, device = self.device)
                                                                               # [number_of_sampled_sequence, seq_len_h]
                k_th_quant = torch.topk(rand_mat, the_number_of_remained_events_per_seq - 1, largest = False)[0][:,-1:]
                                                                               # [number_of_sampled_sequence, 1]
                if the_number_of_remained_events_per_seq == 1:
                    mask = torch.ones_like(rand_mat, device = self.device).long()
                                                                               # [number_of_sampled_sequence, seq_len_h]
                else:
                    mask = (rand_mat > k_th_quant).long()                      # [number_of_sampled_sequence, seq_len_h]
                generated_mask_probability_random = F.one_hot(mask, num_classes = 2)
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
                check_tensor(generated_mask_probability_random)

                # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.
                history_mask_random = F.one_hot(torch.argmax(generated_mask_probability_random, dim = -1), num_classes = 2)
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
                history_mask_random, _ = pack((torch.ones(number_of_sampled_sequence, 1, 2, device = self.device), history_mask_random), 'nss * m')
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
                check_tensor(history_mask_random)
            
                future_mask_random = torch.ones(number_of_sampled_sequence, self.seq_len_x + 1, 2, device = self.device)
                                                                               # [number_of_sampled_sequence, seq_len_x + 1, 2]
        
                filter_mask_random, _ = pack((history_mask_random, future_mask_random), 'nss * m')
                                                                               # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
                filter_mask_random = repeat(filter_mask_random, 'n l m -> n b l m', b = 1)
                                                                               # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]

                L_sp_r, L_rp_r = self.get_metric_values(input_events, input_time, input_mask, filter_mask_random, mean, var)
    
                mask = torch.zeros_like(mask_history, device = self.device) * mask_history
                                                                               # [batch_size, seq_len_h + 1]
            end = time.time()
            time_baseline_1_given_percentage = end - start
            time_baseline_1_given_percentage_to_ehd = time_baseline_1_given_percentage / time_ehd_mtpp

            return percentage_remained_events, L_sp, L_sp_r, L_rp, L_rp_r, time_baseline_1_given_percentage_to_ehd, \
                   history_mask.tolist(), time_history.tolist(), time_future.tolist(), \
                   events_history.tolist(), events_future.tolist()

        '''
        Evaluation part.
        '''
        # part 1: How many percents of events are left?

        start = time.time()
        percentage_remained_events, L_sp, L_rp = self.evaluate_procedure(input_time, input_events, input_mask, mean, var, percentage = False)[-3:]
        end = time.time()
        time_ehd_mtpp = end - start
        the_number_of_remained_events = percentage_remained_events

        percentage_remained_events = percentage_remained_events.float().mean().item()
        L_sp = L_sp.item()
        L_rp = L_rp.item()
        the_number_of_total_events = mask_history.sum(dim = -1)

        # Comparison with random removal.
        # we i.i.d. sample the mask multiple times to eliminate serendipity.
        start = time.time()
        number_of_sampled_sequence = 16
        for the_number_of_remained_events_per_seq, the_number_of_historical_events_per_seq \
            in zip(the_number_of_remained_events, mask_history.sum(dim = -1)):
            # baseline 1, random removal.
            rand_mat = torch.rand(number_of_sampled_sequence, self.seq_len_h, device = self.device)
                                                                               # [number_of_sampled_sequence, seq_len_h]
            k_th_quant = torch.topk(rand_mat, the_number_of_remained_events_per_seq - 1, largest = False)[0][:,-1:]
                                                                               # [number_of_sampled_sequence, 1]
            if the_number_of_remained_events_per_seq == 1:
                mask = torch.ones_like(rand_mat, device = self.device).long()  # [number_of_sampled_sequence, seq_len_h]
            else:
                mask = (rand_mat > k_th_quant).long()                          # [number_of_sampled_sequence, seq_len_h]
            generated_mask_probability_random = F.one_hot(mask, num_classes = 2)
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(generated_mask_probability_random)

            # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.
            history_mask_random = F.one_hot(torch.argmax(generated_mask_probability_random, dim = -1), num_classes = 2)
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
            history_mask_random, _ = pack((torch.ones(number_of_sampled_sequence, 1, 2, device = self.device), history_mask_random), 'nss * m')
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(history_mask_random)
            
            future_mask_random = torch.ones(number_of_sampled_sequence, self.seq_len_x + 1, 2, device = self.device)
                                                                               # [number_of_sampled_sequence, seq_len_x + 1, 2]
    
            filter_mask_random, _ = pack((history_mask_random, future_mask_random), 'nss * m')
                                                                               # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
            filter_mask_random = repeat(filter_mask_random, 'n l m -> n b l m', b = 1)
                                                                               # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]

            L_sp_r, L_rp_r = self.get_metric_values(input_events, input_time, input_mask, filter_mask_random, mean, var)

            mask = torch.zeros_like(mask_history, device = self.device) * mask_history
                                                                               # [batch_size, seq_len_h + 1]
        end = time.time()
        time_baseline_1_given_percentage = end - start
        time_baseline_1_given_percentage_to_ehd = time_baseline_1_given_percentage / time_ehd_mtpp
        L_sp_g1, L_rp_g1 = 0, 0

        start = time.time()
        mask = torch.zeros_like(mask_history) * mask_history                   # [batch_size, seq_len_h + 1]
        for the_number_of_remained_events_per_seq, the_number_of_historical_events_per_seq \
            in zip(the_number_of_remained_events, mask_history.sum(dim = -1)):
            # The first event is always included.
            mask[:, 0] = 1                                                     # [batch_size, seq_len_h + 1]
            full_mask = torch.ones_like(mask_history)                          # [batch_size, seq_len_h + 1]
            seq_len_h_1 = full_mask.sum(dim = -1)                              # [batch_size]
            masked_events = 1

            initial_history_mask = F.one_hot(mask.long(), num_classes = 2)     # [batch_size, seq_len_h, 2]
            check_tensor(initial_history_mask)
            future_mask = torch.ones(initial_history_mask.shape[0], self.seq_len_x + 1, 2, device = self.device)
                                                                               # [batch_size, seq_len_x + 1, 2]
            initial_filter_mask, _ = pack((initial_history_mask, future_mask), 'b * m')
                                                                               # [batch_size, seq_len_h + seq_len_x + 2, 2]
            initial_filter_mask = rearrange(initial_filter_mask, 'b l m -> () b l m')
                                                                               # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]
            selected_L_sp_given_events, selected_L_rp_given_events = \
                self.get_metric_values(input_events, input_time, input_mask, initial_filter_mask, mean, var)

            while masked_events < the_number_of_historical_events_per_seq - the_number_of_remained_events_per_seq:
                number_of_masked_events = mask.sum(dim = -1)                   # [batch_size]
                generated_mask = repeat(mask, 'b s -> nss b s', nss = seq_len_h_1)
                                                                               # [seq_len_h + 1, batch_size, seq_len_h + 1]
                added_mask = torch.diag_embed(torch.ones(seq_len_h_1, device = self.device)).unsqueeze(dim = 1)
                                                                               # [seq_len_h + 1, 1, seq_len_h + 1]
                generated_mask = generated_mask.long() | added_mask.long()     # [seq_len_h + 1, batch_size, seq_len_h + 1]
                generated_mask = generated_mask[generated_mask.sum(dim = -1) != number_of_masked_events]
                                                                               # [seq_len_h + 1 - number_of_masked_events, seq_len_h + 1]
    
                history_mask_random = F.one_hot(generated_mask, num_classes = 2)
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
                check_tensor(history_mask_random)
                    
                future_mask_random = torch.ones(generated_mask.shape[0], self.seq_len_x + 1, 2, device = self.device)
                                                                               # [number_of_sampled_sequence, seq_len_x + 1, 2]
            
                filter_mask_random, _ = pack((history_mask_random, future_mask_random), 'nss * m')
                                                                               # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
                filter_mask_random = repeat(filter_mask_random, 'n l m -> n b l m', b = 1)
                                                                               # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]
        
                L_sp_r_d, L_rp_r_d = self.get_metric_values(input_events, input_time, input_mask, filter_mask_random, mean, var, return_mean = False)
                                                                               # [number_of_sampled_sequence]
                    
                selected_index = torch.argmin(L_rp_r_d)
                selected_L_sp_given_events = L_sp_r_d[selected_index].item()
                selected_L_rp_given_events = L_rp_r_d[selected_index].item()
                selected_mask = generated_mask[selected_index]                 # [batch_size, seq_len_h + seq_len_x + 2, 2]
    
                mask = selected_mask.unsqueeze(dim = 0)                        # [batch_size, seq_len_h + seq_len_x + 2]
                masked_events += 1

            greedy_remained_events = (the_number_of_total_events - mask.sum(dim = 1)).item()

        end = time.time()
        time_greedy_given_percentage = end - start

        # detect where the random removal's performance could reach EHD's performance?
        # Caution: only works when batch_size = 1
        start = time.time()
        random_remained_events = copy.deepcopy(the_number_of_total_events)
        while True:
            if (random_remained_events == 0).any():
                break

            for the_number_of_remained_events_per_seq in random_remained_events:
                # baseline 1, random removal.
                rand_mat = torch.rand(number_of_sampled_sequence, self.seq_len_h, device = self.device)
                                                                               # [number_of_sampled_sequence, seq_len_h]
                k_th_quant = torch.topk(rand_mat, the_number_of_remained_events_per_seq - 1, largest = False)[0][:,-1:]
                                                                               # [number_of_sampled_sequence, 1]
                if the_number_of_remained_events_per_seq == 1:
                    mask = torch.ones_like(rand_mat, device = self.device).long()
                else:
                    mask = (rand_mat > k_th_quant).long()                      # [number_of_sampled_sequence, seq_len_h]
                generated_mask_probability_random = F.one_hot(mask, num_classes = 2)
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
                check_tensor(generated_mask_probability_random)
    
                # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.
                history_mask_random = F.one_hot(torch.argmax(generated_mask_probability_random, dim = -1), num_classes = 2)
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
                history_mask_random, _ = pack((torch.ones(number_of_sampled_sequence, 1, 2, device = self.device), history_mask_random), 'nss * m')
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
                check_tensor(history_mask_random)
                
                future_mask_random = torch.ones(number_of_sampled_sequence, self.seq_len_x + 1, 2, device = self.device)
                                                                               # [number_of_sampled_sequence, seq_len_x + 1, 2]
        
                filter_mask_random, _ = pack((history_mask_random, future_mask_random), 'nss * m')
                                                                               # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
                filter_mask_random = repeat(filter_mask_random, 'n l m -> n b l m', b = 1)
                                                                               # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]
    
                L_sp_r_d, L_rp_r_d = self.get_metric_values(input_events, input_time, input_mask, filter_mask_random, mean, var)
            
            if L_rp_r_d < L_rp and L_sp_r_d > L_sp:
                break
            else:
                random_remained_events = random_remained_events - 1
        end = time.time()
        time_baseline_1 = end - start
        random_remained_events = random_remained_events.item()

        # baseline 2, greedy: remove historical events until it meets EHD's performance.
        # Each time we remove the event which perform highest reduction to the probability.
        # Only works when batch_size = 1
        start = time.time()
        mask = torch.zeros_like(mask_history, device = self.device) * mask_history
                                                                               # [batch_size, seq_len_h + 1]
        # The first event is always included.
        mask[:, 0] = 1                                                         # [batch_size, seq_len_h + 1]
        full_mask = torch.ones_like(mask_history, device = self.device)        # [batch_size, seq_len_h + 1]
        seq_len_h_1 = full_mask.sum(dim = -1)                                  # [batch_size]

        while True:
            # Break when the mask equals to the full_mask, meaning that no event is left.
            if (mask == full_mask).all():
                break

            number_of_masked_events = mask.sum(dim = -1)                       # [batch_size]
            generated_mask = repeat(mask, 'b s -> nss b s', nss = seq_len_h_1) # [seq_len_h + 1, batch_size, seq_len_h + 1]
            added_mask = torch.diag_embed(torch.ones(seq_len_h_1, device = self.device)).unsqueeze(dim = 1)
                                                                               # [seq_len_h + 1, 1, seq_len_h + 1]
            generated_mask = generated_mask.long() | added_mask.long()         # [seq_len_h + 1, batch_size, seq_len_h + 1]
            generated_mask = generated_mask[generated_mask.sum(dim = -1) != number_of_masked_events]
                                                                               # [seq_len_h + 1 - number_of_masked_events, seq_len_h + 1]

            history_mask_random = F.one_hot(generated_mask, num_classes = 2)   # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(history_mask_random)
                
            future_mask_random = torch.ones(generated_mask.shape[0], self.seq_len_x + 1, 2, device = self.device)
                                                                               # [number_of_sampled_sequence, seq_len_x + 1, 2]
        
            filter_mask_random, _ = pack((history_mask_random, future_mask_random), 'nss * m')
                                                                               # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
            filter_mask_random = repeat(filter_mask_random, 'n l m -> n b l m', b = 1)
                                                                               # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]
    
            L_sp_r_d, L_rp_r_d = self.get_metric_values(input_events, input_time, input_mask, filter_mask_random, mean, var, return_mean = False)
                                                                               # [number_of_sampled_sequence]
                
            selected_index = torch.argmin(L_rp_r_d)
            selected_L_sp = L_sp_r_d[selected_index]
            selected_L_rp = L_rp_r_d[selected_index]
            selected_mask = generated_mask[selected_index]                     # [batch_size, seq_len_h + seq_len_x + 2, 2]

            if selected_L_rp < L_rp and selected_L_sp > L_sp:
                break
            else:
                mask = selected_mask.unsqueeze(dim = 0)                        # [batch_size, seq_len_h + seq_len_x + 2]
    
        greedy_remained_events = (the_number_of_total_events - mask.sum(dim = 1)).item()
        end = time.time()
        time_baseline_2 = end - start

        # time evaluation
        time_greedy_given_percentage_to_ehd = time_greedy_given_percentage / time_ehd_mtpp
        time_baseline_1_given_percentage_to_ehd = time_baseline_1_given_percentage / time_ehd_mtpp
        time_baseline_1_to_ehd = time_baseline_1 / time_ehd_mtpp
        time_baseline_2_to_ehd = time_baseline_2 / time_ehd_mtpp

        return percentage_remained_events, random_remained_events, greedy_remained_events, \
               L_sp, L_sp_r, L_sp_g1, selected_L_sp_given_events, L_rp, L_rp_r, L_rp_g1, selected_L_rp_given_events, \
               time_baseline_1_given_percentage_to_ehd, time_baseline_1_to_ehd, time_baseline_2_to_ehd, time_greedy_given_percentage_to_ehd


    def lsp_and_lrp_trend(self, input_data, opt):
        '''
        So this function verifies the assumption 1.
        '''
        input_time, input_events, input_mask, mean, var = self.extract_plot_data(input_data)

        assert (input_mask == 1).all()
        (time_history, time_future), (events_history, events_future), (mask_history, mask_future) \
            = self.divide_history_and_future(input_time, input_events, input_mask)
                                                                               # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3

        the_number_of_remained_events = range(1, self.seq_len_h + 1)
        number_of_sampled_sequence = 32
        L_sp_rs = []
        L_rp_rs = []
        for the_number_of_remained_events_per_seq in the_number_of_remained_events:
            # baseline 1, random removal.
            rand_mat = torch.rand(number_of_sampled_sequence, self.seq_len_h, device = self.device)
                                                                               # [number_of_sampled_sequence, seq_len_h]
            k_th_quant = torch.topk(rand_mat, the_number_of_remained_events_per_seq - 1, largest = False)[0][:,-1:]
                                                                               # [number_of_sampled_sequence, 1]
            if the_number_of_remained_events_per_seq == 1:
                mask = torch.ones_like(rand_mat, device = self.device).long()  # [number_of_sampled_sequence, seq_len_h]
            else:
                mask = (rand_mat > k_th_quant).long()                          # [number_of_sampled_sequence, seq_len_h]
            generated_mask_probability_random = F.one_hot(mask, num_classes = 2)
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(generated_mask_probability_random)

            # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.
            history_mask_random = F.one_hot(torch.argmax(generated_mask_probability_random, dim = -1), num_classes = 2)
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
            history_mask_random, _ = pack((torch.ones(number_of_sampled_sequence, 1, 2, device = self.device), history_mask_random), 'nss * m')
                                                                               # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(history_mask_random)
            
            future_mask_random = torch.ones(number_of_sampled_sequence, self.seq_len_x + 1, 2, device = self.device)
                                                                               # [number_of_sampled_sequence, seq_len_x + 1, 2]
    
            filter_mask_random, _ = pack((history_mask_random, future_mask_random), 'nss * m')
                                                                               # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
            filter_mask_random = repeat(filter_mask_random, 'n l m -> n b l m', b = 1)
                                                                               # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]

            L_sp_r, L_rp_r = self.get_metric_values(input_events, input_time, input_mask, filter_mask_random, mean, var)
            L_sp_rs.append(L_sp_r)
            L_rp_rs.append(L_rp_r)

            mask = torch.zeros_like(mask_history, device = self.device) * mask_history
                                                                               # [batch_size, seq_len_h + 1]
        
        def get_ratio(input_list):
            tmp = np.array(input_list)
            return (tmp - tmp.min()) / (tmp.max() - tmp.min())

        L_rp_rs_ratio = get_ratio(L_rp_rs)
        L_sp_rs_ratio = get_ratio(L_sp_rs)
        return L_rp_rs_ratio, L_sp_rs_ratio


    def get_metric_values(self, input_events, input_time, input_mask, filter_mask, mean, var, return_mean = True):
        '''
        Calculate L_rp and L_sp based on the given filter_mask.
        '''

        events_embeddings = self.mtpp_model('ehd_event_emb', input_events)     # [batch_size, seq_len, d_history]

        padded_filtered_time, padded_filtered_events, padded_filtered_event_embeddings, padded_filtered_masks \
            = self.filter(input_time = input_time, input_events = input_events, events_embeddings = events_embeddings, \
                          input_mask = input_mask, filter_mask = filter_mask, evaluate = True)
                                                                               # [1, batch_size, seq_len_h + seq_len_x + 2] * 2 + [samples_for_l_p, batch_size, seq_len_h + seq_len_x + 2, d_history] + [samples_for_l_p, batch_size, seq_len_h + seq_len_x + 2]

        padded_filtered_removed_time, padded_filtered_removed_events, padded_filtered_event_removed_embeddings, padded_filtered_removed_masks \
            = self.filter(input_time = input_time, input_events = input_events, events_embeddings = events_embeddings, \
                          input_mask = input_mask, filter_mask = filter_mask, evaluate = True, output_removed_events = True)
                                                                               # [1, batch_size, seq_len_h + seq_len_x + 2] * 2 + [samples_for_l_p, batch_size, seq_len_h + seq_len_x + 2, d_history] + [samples_for_l_p, batch_size, seq_len_h + seq_len_x + 2]

        # Loss 3 for asking the model to find the most important events.
        # rebuild the original history for H_{o,t_l} - H_{s,o,t_l} based on history_mask.
        # You should be really careful to implement this part for not accidentally dropping any gradients.
        log_p_h_o_t_l_x_o_mean = self.mtpp_model('ehd_perplexity', input_time, input_events, events_embeddings, input_mask, self.seq_len_x, mean, var)
                                                                               # [batch_size]
        
        log_p_h_r_o_t_l_x_o_mean = []
        for padded_filtered_time_per_sample, padded_filtered_events_per_sample, \
            padded_filtered_event_embeddings_per_sample, padded_filtered_masks_per_sample in \
            zip(padded_filtered_time, padded_filtered_events, padded_filtered_event_embeddings, padded_filtered_masks):
            log_p_h_r_o_t_l_x_o_mean.append(self.mtpp_model('ehd_perplexity', padded_filtered_time_per_sample, padded_filtered_events_per_sample,
                                                       padded_filtered_event_embeddings_per_sample, padded_filtered_masks_per_sample, 
                                                       self.seq_len_x, mean, var))
                                                                               # [batch_size]
        log_p_h_r_o_t_l_x_o_mean = torch.stack(log_p_h_r_o_t_l_x_o_mean, dim = 0)
                                                                               # [1, batch_size]

        if return_mean:
            L_rp = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_r_o_t_l_x_o_mean).mean().item()
        else:
            L_rp = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_r_o_t_l_x_o_mean)

        # part 2: What is the value of log_p_h_s_o_t_l_x_o_mean?
        log_p_h_s_o_t_l_x_o_mean = []
        for padded_filtered_removed_time_per_sample, padded_filtered_removed_events_per_sample, \
            padded_filtered_removed_event_embeddings_per_sample, padded_filtered_removed_masks_per_sample in \
            zip(padded_filtered_removed_time, padded_filtered_removed_events, padded_filtered_event_removed_embeddings, padded_filtered_removed_masks):
            log_p_h_s_o_t_l_x_o_mean.append(self.mtpp_model('ehd_perplexity', padded_filtered_removed_time_per_sample, padded_filtered_removed_events_per_sample,
                                                       padded_filtered_removed_event_embeddings_per_sample, padded_filtered_removed_masks_per_sample, 
                                                       self.seq_len_x, mean, var))
                                                                               # [batch_size]
        log_p_h_s_o_t_l_x_o_mean = torch.stack(log_p_h_s_o_t_l_x_o_mean, dim = 0)
                                                                               # [1, batch_size]

        if return_mean:
            L_sp = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_s_o_t_l_x_o_mean).mean().item()
        else:
            L_sp = log_p_h_o_t_l_x_o_mean.unsqueeze(dim = 0) - log_p_h_s_o_t_l_x_o_mean

        return L_sp, L_rp


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
    
        model.train()
        [time_seq, event_seq, score, mask], (mean, var) = minibatch
        loss, L_c, L_p, L_g = model(         
                task_name = 'train', input_time = time_seq, input_events = event_seq, \
                input_mask = mask, mean = mean, var = var
        )
        
        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        
        loss = loss.item()
        L_c = L_c.item()
        L_p = L_p.item()
        L_g = L_g.item()

        return loss, L_c, L_p, L_g
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        [time_seq, event_seq, score, mask], (mean, var) = minibatch
        loss, L_c, L_p, L_g, percentage_remained_events, L_sp, L_rp = model(
                task_name = 'evaluate', input_time = time_seq, input_events = event_seq, 
                input_mask = mask, mean = mean, var = var
        )
    
        loss = loss.item()
        L_c = L_c.item()
        L_p = L_p.item()
        L_g = L_g.item()
        percentage_remained_events = percentage_remained_events.item()
        L_sp = L_sp.item()
        L_rp = L_rp.item()

        return loss, L_c, L_p, L_g, percentage_remained_events, L_sp, L_rp


    def postprocess(input, procedure):
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss, events loss]
            '''
            return [input[0], input[1], input[2], input[3]]
        
        def test_postprocess(input):
            '''
            Evaluation process
            [absolute loss, relative loss, events loss, mae value]
            '''
            return [input[0], input[1], input[2], input[3], input[4], input[5], input[6]]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    

    def log_print_format(input, procedure):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['Loss'] = input[0]
            format_dict['L_c'] = input[1]
            format_dict['L_p'] = input[2]
            format_dict['L_g'] = input[3]
            # format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f', \
            #                              'events_loss': ':6.5f'}
            format_dict['num_format'] = {'Loss': ':6.5f', 'L_c': ':6.5f', 'L_p': ':6.5f', 'L_g': ':6.5f'}
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['Loss'] = input[0]
            format_dict['L_c'] = input[1]
            format_dict['L_p'] = input[2]
            format_dict['L_g'] = input[3]
            format_dict['percentage_remained_events'] = input[4]
            format_dict['L_sp'] = input[5]
            format_dict['L_rp'] = input[6]
            # format_dict['f1_pred_at_time_next'] = input[4]
            # format_dict['mae'] = input[5]
            # format_dict['f1_pred_at_pred_time'] = input[6]
            format_dict['num_format'] = {'Loss': ':6.5f', 'L_c': ':6.5f', 'L_p': ':6.5f', 'L_g': ':6.5f', 
                                         'percentage_remained_events': ':6.5f', 'L_sp': ':6.5f', 'L_rp': ':6.5f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 7
    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['Loss'], 
                test_report_format_dict['Loss']], \
               ['evaluation_Loss', 'test_Loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric