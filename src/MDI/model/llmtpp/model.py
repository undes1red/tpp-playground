import torch, copy
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
from sklearn.metrics import f1_score

from src.toolbox.misc import check_tensor, move_from_tensor_to_ndarray, easy_model_load
from src.toolbox.integration import approximate_integration
from src.toolbox.metrics import otd

from src.MDI.model.basic_tpp_model import memory_ceiling, BasicModel, its_lower_bound, its_upper_bound
from src.MDI.model.nhps.plot import *
from src.toolbox.misc import pack_one_value_to_dict
from src.MDI.model.nhps.submodel import NHPS
from src.MDI.model.utils import *
from src.MDI.model.nhps.decoder import ConsensusDecoder


class NHPSWrapper(BasicModel):
    def __init__(self, opt, device, event_del_costs, d_input = 64, history_module_name = 'LSTM', history_encoder_layers = 1, \
                 d_mark_embedding = 64, d_hidden = 256, dropout = 0.1, epsilon = 1e-20, mae_step = 8, mae_e_step = 8, \
                 integration_sample_rate = 100, survival_loss_during_training = True, config_loaded_model = {}):
        super(NHPSWrapper, self).__init__()
        self.device = device
        self.num_events = opt.info_dict['num_events']
        self.start_time = opt.info_dict['t_0']
        self.end_time = opt.info_dict['T']
        self.mark_missing_probability = opt.dataloader_config_dict['missing_probability']
        self.integration_sample_rate = integration_sample_rate
        self.event_del_costs = event_del_costs
        self.epsilon = epsilon
        self.survival_loss_during_training = survival_loss_during_training
        self.sample_time_rate = 32
        self.mae_step = mae_step
        self.mae_e_step = mae_e_step
        self.bisect_early_stop_threshold = 1e-4
        self.max_step = 50
        
        self.nhps = NHPS(opt, device, self.num_events, history_module_name, d_mark_embedding, d_input, d_hidden, \
                         history_encoder_layers, dropout, integration_sample_rate, self.mark_missing_probability, config_loaded_model)
    

    def divide_history_and_next(self, input):
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next


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
    

    def forward(self, task_name, *args, **kwargs):
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
        * std           type: float shape: N/A
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
            'get_imputed_seq_by_nhpf': self.get_imputed_seq_by_nhpf,
            'get_imputed_seq_by_nhps': self.get_imputed_seq_by_nhps,
            
            'probability_nhps_nhpf': self.probability_nhps_nhpf,
            'otd_nhpf': self.otd_nhpf,
            'otd_nhps': self.otd_nhps,
        }

        return task_mapper[task_name](*args, **kwargs)


    '''
    Functions for model training.
    '''
    def train_procedure(self, forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, mean, std):
        '''
        Check if events data is present.
        Now, we assume that no event data is available.
        Args:
        1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
        2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
        3. mask: filter out the padding events in the event batches. shape: [batch_size, seq_len + 1]
        '''

        log_q_z_con_x_on_all_samples \
            = self.nhps(forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq)
                                                                                 # 2 * [batch_size, seq_len, num_events]
        
        loss = -log_q_z_con_x_on_all_samples.mean()
        check_tensor(loss, positive = False)

        return loss


    '''
    Functions for model evaluation
    '''
    @torch.no_grad()
    def evaluate_procedure(self, forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, mean, std):
        '''
        Check if events data is present.
        Now, we assume that no event data is available.
        Args:
        1. time_seq: the sequence containing events' timestamps.
           shape: [batch_size, seq_len]
        2. events: the sequence containing information about events.
           shape: [batch_size, seq_len]
        3. mask: filter out the padding events in the event batches.
           shape: [batch_size, seq_len]
        4. padded_obs_data: sequence with some events missing.
           shape: [batch_size, [[sample_num, seq_len]]] * 3
           tags:  obs_time_seq, obs_event_seq, obs_mask_seq
        5. padded_backward_obs_event_seq: sequence with some events missing, backward.
           shape: [batch_size, [[sample_num, seq_len]]] * 3
           tags:  backward_obs_time_seq, backward_obs_event_seq, backward_obs_mask_seq
        '''
        log_q_z_con_x_on_all_samples \
            = self.nhps(forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq)
                                                                                 # 2 * [batch_size, seq_len, num_events]
        loss = -log_q_z_con_x_on_all_samples.mean()
        check_tensor(loss, positive = False)

        return loss
    

    @torch.no_grad()
    def imputing_by_nhpf(self, obs_time_seq, obs_event_seq, obs_mask_seq, 
                         samples_to_calc_otd, imputed_retries_num, mean, std):
        # Considering the length of imputed sequences is not known before the imputation, here nhps.imputing_one_seq_by_nhpf sample one z from x per loop.
        # This approach is expected to be slower but easier to implement.
        # Next, a consensus decoder merges all imputed sequences into a single prediction with the lowest Beyes risk.
        packed_data = zip(obs_time_seq, obs_event_seq, obs_mask_seq)
        
        weights = []
        imputed_seqs = []
        for idx, (obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq) in enumerate(packed_data):
            if idx >= samples_to_calc_otd:
                break
            # Extremely slow!
            weights_for_one_seq, imputed_seqs_for_one_seq = \
                    self.nhps.imputing_by_nhpf(obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, \
                                               imputed_retries_num, mean, std) # [imputed_retries_num] + imputed_retries_num * [time, event]
            weights.append(weights_for_one_seq)
            imputed_seqs.append(imputed_seqs_for_one_seq)
        
        return weights, imputed_seqs
    
    
    @torch.no_grad()
    def get_imputed_seq_by_nhpf(self, input_data, opt):
        forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, (mean, std) \
            = input_data
        complete_time, complete_events, complete_mask = move_from_tensor_to_ndarray(*forward_complete_data)
        
        for obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, missing_mask_for_one_seq, _ in padded_obs_data:
            weights, imputed_seqs \
                = self.imputing_by_nhpf(obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, \
                                        samples_to_calc_otd = 1, imputed_retries_num = 2, mean = mean, std = std)
                                                                               # [num_samples, len(self.event_del_costs), seq_len]
        return complete_time, complete_events, complete_mask, weights, imputed_seqs


    '''
    @torch.no_grad()
    def imputing_by_nhpf(self, obs_time_seq, obs_event_seq, obs_mask_seq, 
                         samples_to_calc_otd, imputed_retries_num, resample, mean, std):
        # Considering the length of imputed sequences is not known before the imputation, here nhps.imputing_one_seq_by_nhpf sample one z from x per loop.
        # This approach is expected to be slower but easier to implement.
        # Next, a consensus decoder merges all imputed sequences into a single prediction with the lowest Beyes risk.
        packed_data = zip(obs_time_seq, obs_event_seq, obs_mask_seq)
        decoder = ConsensusDecoder(del_cost = self.event_del_costs, n_types = self.num_events)
        
        imputed_sequences_after_decode = []
        for idx, (obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq) in enumerate(packed_data):
            if idx >= samples_to_calc_otd:
                break
            # Extremely slow!
            weights, imputed_seqs = \
                    self.nhps.imputing_by_nhpf(obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, \
                                               imputed_retries_num, mean, std) # [imputed_retries_num] + imputed_retries_num * [time, event]
            
            selected_index = np.argsort(-np.array(weights))[:resample]         # [resample]
            selected_weights = [weights[idx] for idx in selected_index]        # imputed_retries_num * [time, event]
            selected_seqs = [imputed_seqs[idx] for idx in selected_index]      # imputed_retries_num * [time, event]
            # Consensus decoding, merging num_imputed_seq sequences into one prediction according to the weights.
            decoded_sequences = decoder.decode(selected_seqs, selected_weights)# len(self.event_del_costs) * [time, event]
            imputed_sequences_after_decode.append(decoded_sequences)
        
        return imputed_sequences_after_decode
    '''


    @torch.no_grad()
    def imputing_by_nhps(self, 
                         obs_time_seq, obs_event_seq, obs_mask_seq, obs_missing_mask, 
                         backward_obs_time_seq, backward_obs_events_seq, backward_obs_mask_seq, backward_obs_missing_mask,
                         samples_to_calc_otd, imputed_retries_num, mean, std):
        # Considering the length of imputed sequences is not known before the imputation, here nhps.imputing_one_seq_by_nhpf sample one z from x per loop.
        # This approach is expected to be slower but easier to implement.
        # Next, a consensus decoder merges all imputed sequences into a single prediction with the lowest Beyes risk.
        packed_data = zip(obs_time_seq, obs_event_seq, obs_mask_seq, obs_missing_mask, \
                          backward_obs_time_seq, backward_obs_events_seq, backward_obs_mask_seq, backward_obs_missing_mask)
        
        weights = []
        imputed_seqs = []
        for idx, (obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, obs_missing_mask_for_one_seq, \
                  backward_obs_time_for_one_seq, backward_obs_events_for_one_seq, backward_obs_mask_for_one_seq, backward_obs_missing_mask_for_one_seq) in enumerate(packed_data):
            if idx >= samples_to_calc_otd:
                break
            # Extremely slow!
            num_of_events_for_one_seq = obs_mask_for_one_seq.sum(dim = -1)
            obs_time_for_one_seq = obs_time_for_one_seq[:num_of_events_for_one_seq]
            obs_events_for_one_seq = obs_events_for_one_seq[:num_of_events_for_one_seq]
            backward_obs_time_for_one_seq = backward_obs_time_for_one_seq[:num_of_events_for_one_seq]
            backward_obs_events_for_one_seq = backward_obs_events_for_one_seq[:num_of_events_for_one_seq]
            weights_for_one_seq, imputed_seqs_for_one_seq = \
                self.nhps.imputing_by_nhps(obs_time_for_one_seq, obs_events_for_one_seq, obs_missing_mask_for_one_seq, \
                                           backward_obs_time_for_one_seq, backward_obs_events_for_one_seq, backward_obs_missing_mask_for_one_seq, \
                                           imputed_retries_num, mean, std)
                                                                               # num_imputed_seq * [time, event]
            weights.append(weights_for_one_seq)
            imputed_seqs.append(imputed_seqs_for_one_seq)
        
        return weights, imputed_seqs


    @torch.no_grad()
    def get_imputed_seq_by_nhps(self, input_data, opt):
        forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, (mean, std) \
            = input_data
        complete_time, complete_events, complete_mask = move_from_tensor_to_ndarray(*forward_complete_data)
        
        for (obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, missing_mask_for_one_seq, _ ,), \
            (backward_obs_time_for_one_seq, backward_obs_events_for_one_seq, backward_obs_mask_for_one_seq, backward_missing_mask_for_one_seq, _ ,) \
            in zip(padded_obs_data, padded_backward_obs_event_seq):
            weights, imputed_seqs \
                = self.imputing_by_nhps(obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, missing_mask_for_one_seq, \
                                        backward_obs_time_for_one_seq, backward_obs_events_for_one_seq, backward_obs_mask_for_one_seq, backward_missing_mask_for_one_seq, \
                                        samples_to_calc_otd = 1, imputed_retries_num = 4, mean = mean, std = std)
                                                                               # [num_samples, len(self.event_del_costs), seq_len]

        return complete_time, complete_events, complete_mask, weights, imputed_seqs


    @torch.no_grad()
    def otd_nhpf(self, input_data, opt):
        forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, (mean, std) \
            = input_data
        complete_time, complete_events, complete_mask = move_from_tensor_to_ndarray(*forward_complete_data)
        
        otds = []
        for obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, missing_mask_for_one_seq, _ in padded_obs_data:
            imputed_seqs_per_sample_by_nhpf \
                = self.imputing_by_nhpf(obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, \
                                        samples_to_calc_otd = 1, imputed_retries_num = 1, resample = 1, mean = mean, std = std)
                                                                               # [num_samples, len(self.event_del_costs), seq_len]
            otds_per_obs_seq = []
            for imputed_seqs in imputed_seqs_per_sample_by_nhpf:
                otds_per_seq = []
                for cost_idx, (imputed_marks, imputed_times) in enumerate(imputed_seqs):
                    otds_per_seq.append(
                        otd(imputed_marks, imputed_times, complete_events, complete_time.cumsum(axis = -1), \
                            self.num_events + 1, add_remove_event_cost = self.event_del_costs[cost_idx], move_event_cost = 1.0, average = 'macro'))
                
                otds_per_obs_seq.append(otds_per_seq)
            otds.append(np.array(otds_per_obs_seq).mean(axis = 0))             # [len(self.event_del_costs)]
        
        otds = np.stack(otds, axis = -1)                                       # [batch_size, len(self.event_del_costs)]
        return otds


    @torch.no_grad()
    def otd_nhps(self, input_data, opt):
        forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, (mean, std) \
            = input_data
        complete_time, complete_events, complete_mask = move_from_tensor_to_ndarray(*forward_complete_data)
        
        otds = []
        for obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, missing_mask_for_one_seq, _, \
            backward_obs_time_for_one_seq, backward_obs_event_for_one_seq, backward_obs_mask_for_one_seq, backward_missing_mask_for_one_seq, _ \
            in zip(padded_obs_data, padded_backward_obs_event_seq):
            imputed_seqs_per_sample_by_nhpf \
                = self.imputing_by_nhps(obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, \
                                        backward_obs_time_for_one_seq, backward_obs_event_for_one_seq, backward_obs_mask_for_one_seq, 
                                        samples_to_calc_otd = 1, imputed_retries_num = 1, resample = 1, mean = mean, std = std)
                                                                               # [num_samples, len(self.event_del_costs), seq_len]
            otds_per_obs_seq = []
            for imputed_seqs in imputed_seqs_per_sample_by_nhpf:
                otds_per_seq = []
                for cost_idx, (imputed_marks, imputed_times) in enumerate(imputed_seqs):
                    otds_per_seq.append(
                        otd(imputed_marks, imputed_times, complete_events, complete_time.cumsum(axis = -1), \
                            self.num_events + 1, add_remove_event_cost = self.event_del_costs[cost_idx], move_event_cost = 1.0, average = 'macro'))
                
                otds_per_obs_seq.append(otds_per_seq)
            otds.append(np.array(otds_per_obs_seq).mean(axis = 0))             # [len(self.event_del_costs)]
        
        otds = np.stack(otds, axis = -1)                                       # [batch_size, len(self.event_del_costs)]
        return otds
    
    
    @torch.no_grad()
    def probability_nhps_nhpf(self, input_data, opt):
        forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, (mean, std) \
            = input_data
        
        # p(z|x). (Well, maybe not).
        log_p_z_x = self.nhps.get_nhpf_probability(forward_complete_data, padded_obs_data)
        
        # q(z|x)
        log_q_z_con_x_on_all_samples = -self.evaluate_procedure(forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, mean, std)
        
        log_p_z_x, log_q_z_con_x_on_all_samples = move_from_tensor_to_ndarray(log_p_z_x, log_q_z_con_x_on_all_samples)
        
        return log_p_z_x, log_q_z_con_x_on_all_samples


    '''
    Static methods
    '''
    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
        model.train()

        '''
        Maybe need another function to extract data from minibatches.
        Currently, we don't acquire any prediction loss to assist the model training.  
        '''
        forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, (mean, std) \
            = minibatch                                                        # [batch_size, seq_len] + [batch_size, sample_number, obs_seq_len] * 2

        loss = model('train', forward_complete_data = forward_complete_data, backward_complete_data = backward_complete_data,        
                     padded_obs_data = padded_obs_data, padded_backward_obs_event_seq = padded_backward_obs_event_seq, \
                     mean = mean, std = std)

        loss.backward()
        
        return loss.item()
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
        model.eval()
        
        forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, (mean, std) \
            = minibatch                                                        # [batch_size, seq_len] + [batch_size, sample_number, obs_seq_len] * 2
        loss = model('evaluate', forward_complete_data = forward_complete_data, backward_complete_data = backward_complete_data,        
                     padded_obs_data = padded_obs_data, padded_backward_obs_event_seq = padded_backward_obs_event_seq, \
                     mean = mean, std = std)
        
        return loss.item()


    def postprocess(input, procedure):
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss, events loss]
            '''
            return [input[0],]
        
        def test_postprocess(input):
            '''
            Evaluation process
            [absolute loss, relative loss, events loss, mae value]
            '''
            return [input[0],]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    
    
    def log_print_format(input, procedure):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['loss'] = pack_one_value_to_dict(input[0])
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['loss'] = pack_one_value_to_dict(input[0])
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))


    format_dict_length = 6

    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['loss'], 
                test_report_format_dict['loss']], \
               ['evaluation_loss', 'test_loss']

    metric_number = 2 # metric number is the length of the output of choose_metric