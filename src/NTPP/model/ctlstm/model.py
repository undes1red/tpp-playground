import torch, copy, os
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, top_k_accuracy_score
from datetime import datetime, timedelta

from src.toolbox.misc import check_tensor, move_from_tensor_to_ndarray, move_from_tensor_to_list, pack_one_value_to_dict, argument_check, predict_event, reverse_dict_key_val
from src.toolbox.integration import approximate_integration
from src.toolbox.metrics import L1_distance_between_two_funcs
# from src.toolbox.llms import CustomOpenAIforVLLM, VLLMOfflineInference, extract_content, remove_thinking, create_messages

from src.NTPP.model.basic_tpp_model import memory_ceiling, BasicModel
from src.NTPP.model.ctlstm.plot import *
from src.NTPP.model.ctlstm.submodel import CTLSTM
from src.NTPP.model.ctlstm.sample import sample_time, sample_time_event, sample_event_time
from src.NTPP.model.utils import decide_resolution_inf_and_resolution_between_events, get_f1_and_top_k_acc_in_mae_e


class CTLSTMWrapper(BasicModel):
    '''
    continuous-time LSTM, the backbone of the Neural Hawkes Process, proposed by Mei et al. at NeurIPS 2017.
    '''
    def __init__(self, opt, device, d_input = 64, history_module_name = 'LSTM', history_encoder_layers = 1, \
                 d_mark_embedding = 64, d_hidden = 256, dropout = 0.1, epsilon = 1e-20, 
                 sample_rate = 32, mae_step = 8, mae_e_step = 8, \
                 integration_sample_rate = 100, survival_loss_during_training = True, \
                 mtpp_note_global_known = False, mtpp_note_is_true_note = False, mtpp_includes_note_embedding = True, \
                 mtpp_not_conditioned_on_note = False, sample_embedding_pred = True, sample_size = 3):
        '''
        This function creates a CTLSTM model.
        
        ### Args
            * ```int``` d_mark_embedding
              The dimension of the mark embeddings.
            * ```str``` history_module_name
              Which RNN model do we use to encode the history? Default is LSTM. We don't recommend to change it to something else.
            * ```int``` d_hidden
              The dimension of the history representation.
            * ```float``` dropout
              Dropout rate for the history encoder. Only works when history_encoder_layers > 1.
            * ```int``` history_encoder_layers
              How many layer of RNN our model will have?
            * ```int``` d_input
              The dimension of the cumulative hazard function network.
            * ```namespace``` opt
              Model arguments.
            * ```torch.device``` device
              Running models on GPU or CPU?
            * ```float``` epsilon
              Shiftting the calculated intensity function and probability distribution by a little bit so that ```torch.log()``` won't fail.
            * ```int``` sample_rate
              This tells how many time samples from the time distribution are needed for one time prediction.
            * ```int``` mae_step
              This parameter controls how many samples are generated in one shot when sampling from p(t).
            * ```int``` mae_e_step
              This parameter controls how many samples are generated in one shot when sampling from all p(t|m)s at the same time.
              mae_step and mae_e_step are useful when you cannot get sample_rate time samples from time distributions because of insufficient GPU memory.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint events for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```bool``` survival_loss_during_training
              When true, the training loss includes the integral between the last observed event to the end time T. Most of time this argument should be true.
            * ```bool``` llm_contrastive
              The training loss will include the contrastive loss if true.
            * ```str``` url_path
              The url to access the LLM.
            * ```str``` api_key
              The API key used for identification during API calling.
            * ```str``` llm_model
              The name of the used LLM.
        '''
        super(CTLSTMWrapper, self).__init__()
        self.device = device
        self.compile_or_not = opt.compile
        self.num_events = opt.info_dict['num_events']
        self.start_time = opt.info_dict['t_0']
        self.end_time = opt.info_dict['T']
        self.note_embedding_size = opt.info_dict['embedding_size']
        self.rosetta_original_to_mark = {**opt.info_dict['rosetta'], ' ': self.num_events}
        self.rosetta_mark_to_original = reverse_dict_key_val(self.rosetta_original_to_mark)
        self.integration_sample_rate = integration_sample_rate
        self.epsilon = epsilon
        self.survival_loss_during_training = survival_loss_during_training
        self.sample_rate = sample_rate
        self.mae_step = mae_step
        self.mae_e_step = mae_e_step
        self.bisect_early_stop_threshold = 1e-4
        self.max_step = 50
        self.d_mark_embedding = d_mark_embedding
        # Please take care that in the default settings CTLSTM handles continuous-time event stream without notes.
        # You should set mtpp_includes_note_embedding = True to inject note embeddings into the CTLSTM.
        self.mtpp_includes_note_embedding = mtpp_includes_note_embedding
        self.mtpp_note_global_known = mtpp_note_global_known
        # For LLM datasets whose MTPP notes are generated by LLM, not the true note.
        self.mtpp_note_is_true_note = mtpp_note_is_true_note if mtpp_note_global_known else False
        self.mtpp_not_conditioned_on_note = mtpp_not_conditioned_on_note
        
        self.model = CTLSTM(device = device, num_events = self.num_events, history_module_name = history_module_name, \
                            d_mark_embedding = d_mark_embedding, d_input = d_input, d_hidden = d_hidden, \
                            history_encoder_layers = history_encoder_layers, dropout = dropout, \
                            integration_sample_rate = integration_sample_rate, mtpp_includes_note_embedding = self.mtpp_includes_note_embedding, \
                            note_embedding_size = self.note_embedding_size, mtpp_note_global_known = self.mtpp_note_global_known, 
                            sample_embedding_pred = sample_embedding_pred, sample_size = sample_size)


    def divide_history_and_next(self, input):
        '''
        Extract the history and prediction sequences from the input sequence.
        
        ### Args
            * ```torch.tensor``` input
              shape: [batch_size, seq_len + 1]
              The input sequence.
        
        ### Outputs
            * ```torch.tensor``` input_history
              shape: [batch_size, seq_len]
              The history sequence extracted from the original input.
            * ```torch.tensor``` input_next
              shape: [batch_size, seq_len]
              The history sequence extracted from the original input.
        '''
        if torch.is_tensor(input):
            input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        elif isinstance(input, np.ndarray):
            input_history, input_next = input[:, :-1].copy(), input[:, 1:].copy()
        return input_history, input_next


    def remove_dummy_event_from_mask(self, mask):
        '''
        Remove the probability of the dummy event from the mask.

        ### Args
            * ```torch.tensor``` mask
              shape: [batch_size, seq_len]
              The input mask tensor.
        
        ### Outputs
            * ```torch.tensor``` mask_without_dummy
              shape: [batch_size, seq_len]
              The output mask tensor with the last unmask event in each sequence removed.
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
        The entrance of the CTLSTM.
        
        ### Args
            * ```str``` task_name
              The name of the executed task.
        '''
        task_mapper = {
            'train': self.train_procedure,
            'evaluate': self.evaluate_procedure,
            'spearman_and_l1': self.get_spearman_and_l1,
            'mae_and_f1': self.get_mae_and_f1,
            'mae_e_and_f1': self.get_mae_e_and_f1,

            # experiment 1: real event classification
            'llm_max_token_length': self.llm_max_token_length,
            'llm_mtpp_classification': self.llm_mtpp_classification,
            
            # experiment 2: How lucky should you be to consistently get samples better than expectation?
            # Perhaps we do not need to discuss mark if the probability of consistent good time samples is negligible.
            'probability_of_sampling_better_than_expectation': self.probability_of_sampling_better_than_expectation,
            
            # experiment 3: Will LLM give events closer to the real event relatively higher probability?
            'will_llm_assign_higher_probability_to_better_events': self.will_llm_assign_higher_probability_to_better_events,

            # figure drawing funtions
            'intensity': self.figure_intensity,
            'integral': self.figure_integral,
            'probability': self.figure_probability,
            'debug': self.figure_debug,
            
            # For CPPOD, should be used with the od_generic dataloader.
            'cppod_evaluation': self.cppod_evaluation,
            'cppod_commission_evaluation': self.cppod_commission_evaluation
        }

        return task_mapper[task_name](*args, **kwargs)


    def train_procedure(self, time, original_time, events, original_events, note, note_embedding, mask, mean, std, step):
        '''
        CTLSTM's forwardpropagation function for training.
        
        ### Args
            * ```torch.tensor``` time
              shape: ```[batch_size, seq_len + 1]```
              Time sequence for training.
            * ```torch.tensor``` events
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy events.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` loss
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event (the dummy event at end time T included).
            * ```torch.tensor``` log_likeli_loss_without_dummy
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event (the dummy event at end time T excluded).
            * ```torch.tensor``` marker_loss_without_dummy
              shape: ```[1]```
              The sum of the event loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real event.
            * ```int``` the_number_of_events
              The number of legit events.
        '''
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len] * 2
        original_time_history, original_time_next = self.divide_history_and_next(original_time)
                                                                               # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len] * 2
        note_history, note_next = self.divide_history_and_next(note)           # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len] * 2
        note_embedding_history, note_embedding_next = None, None
        if self.mtpp_includes_note_embedding:
            note_embedding_history, note_embedding_next = self.divide_history_and_next(note_embedding)
                                                                               # [batch_size, seq_len, dim_note_embedding] * 2
        
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        event_next_without_dummy = (mask_next_without_dummy * events_next).long()
                                                                               # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()
        
        if self.mtpp_includes_note_embedding:
            integral_all_events, intensity_all_events, predicted_note_embedding \
                = self.model(time_history, time_next, events_history, mask_history, \
                             note_embedding_history = note_embedding_history, output_pred_note_embedding = True, note_embedding_next = note_embedding_next)
                                                                               # 2 * [batch_size, seq_len, num_events]
        else:
            integral_all_events, intensity_all_events \
                = self.model(time_history, time_next, events_history, mask_history, \
                             note_embedding_history = note_embedding_history, output_pred_note_embedding = False, note_embedding_next = note_embedding_next)
                                                                               # 2 * [batch_size, seq_len, num_events]          

        # MSE loss between the predicted note embedding and the real note embedding.
        mse_loss = torch.tensor(0.0, device = self.device)
        if self.mtpp_includes_note_embedding and not self.mtpp_not_conditioned_on_note:
            mse_loss = -F.cosine_similarity(predicted_note_embedding, note_embedding_history[..., :self.d_mark_embedding], dim = -1)
            mse_loss = mse_loss * mask_next_without_dummy                      # [batch_size, seq_len]
            mse_loss = mse_loss.sum()

        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        log_likeli_loss_without_dummy, marker_loss_without_dummy = self.loss_function(
             integral_all_events = integral_all_events, intensity_all_events = intensity_all_events, \
             events_next = event_next_without_dummy, mask_next = mask_next_without_dummy
        )

        # loss_kl = torch.tensor(0.0, device = self.device)
        # average_probability_sum = torch.tensor(0.0, device = self.device)
        
        # if self.llm_contrastive and step % 1 == 0:
        #     loss_kl, average_probability_sum = self.likelihood_loss(note_history, note_next, events_history, time_history, original_time_history, \
        #                                                             event_next_without_dummy, time_next, mask_next_without_dummy, mean, std, \
        #                                                             sparse_rate = self.sparse_rate, note_embedding_history = note_embedding_history)

        loss_survival = 0
        if self.survival_loss_during_training:
            # survival_loss = \\int_{t_n}^{T}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
            dummy_event_index = mask_next.sum(dim = -1) - 1                    # [batch_size]
            integral_survival = integral_all_events.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
            loss_survival = integral_survival.sum()

        loss = log_likeli_loss_without_dummy + loss_survival + mse_loss

        return loss, log_likeli_loss_without_dummy, mse_loss, marker_loss_without_dummy, the_number_of_events


    @torch.inference_mode()
    def evaluate_procedure(self, time, original_time, events, original_events, note, note_embedding, mask, mean, std):
        '''
        CTLSTM's forwardpropagation function for evaluation.
        
        ### Args
            * ```torch.tensor``` time
              shape: ```[batch_size, seq_len + 1]```
              Time sequencalculatesce for training.
            * ```torch.tensor``` events
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy events.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` log_likeli_loss_time_next_without_dummy
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event.
            * ```torch.tensor``` loss_survival
              shape: ```[1]```
              The sum of the integration \\Lambda^*(m, t) from the last observed event to the end time T.
            * ```torch.tensor``` marker_loss_time_next_without_dummy
              shape: ```[1]```
              The sum of the event loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real event.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
            * ```int``` the_number_of_events
              The number of legit events.
        '''
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        note_embedding_history, note_embedding_next = None, None
        if self.mtpp_includes_note_embedding:
            note_embedding_history, note_embedding_next = self.divide_history_and_next(note_embedding)
                                                                               # [batch_size, seq_len, dim_note_embedding] * 2

        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        event_next_without_dummy = (mask_next_without_dummy * events_next).long()
                                                                               # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        mae, f1, _ = self.mean_absolute_error_and_f1(time_history = time_history, time_next = time_next, \
                                                     events_history = events_history, events_next = events_next, \
                                                     mask_history = mask_history, mask_next = mask_next_without_dummy, \
                                                     mean = mean, std = std, \
                                                     note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next)
                                                                               # [batch_size, seq_len] * 2
        mae = mae.sum().item() / the_number_of_events
        
        if self.mtpp_includes_note_embedding:
            integral_all_events_time_next, intensity_all_events_time_next, predicted_note_embedding, \
                = self.model(time_history, time_next, events_history, mask_history, \
                             note_embedding_history = note_embedding_history, output_pred_note_embedding = True, 
                             note_embedding_next = note_embedding_next)        # 2 * [batch_size, seq_len, num_events]
        else:
            integral_all_events_time_next, intensity_all_events_time_next, \
                = self.model(time_history, time_next, events_history, mask_history, \
                             note_embedding_history = note_embedding_history, output_pred_note_embedding = False, 
                             note_embedding_next = note_embedding_next)        # 2 * [batch_size, seq_len, num_events]
                                                                               
        # MSE loss between the predicted note embedding and the real note embedding.
        mse_loss = torch.tensor(0.0, device = self.device)
        if self.mtpp_includes_note_embedding:
            mse_loss = F.cosine_similarity(predicted_note_embedding, note_embedding_next[..., :self.d_mark_embedding], dim = -1)
            mse_loss = mse_loss * mask_next_without_dummy                      # [batch_size, seq_len]
            mse_loss = mse_loss.sum()

        # NLL loss and event loss at time_next
        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        log_likeli_loss_time_next_without_dummy, marker_loss_time_next_without_dummy = self.loss_function(
             integral_all_events = integral_all_events_time_next, intensity_all_events = intensity_all_events_time_next, \
             events_next = event_next_without_dummy, mask_next = mask_next_without_dummy
        )
        # Survival probability: \\int_{t_N}^{T}{\\sum_{k}\\lambda_k^(\\tau)d\\tau}
        dummy_event_index = mask_next.sum(dim = -1) - 1                        # [batch_size]
        integral_survival = integral_all_events_time_next.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
        loss_survival = integral_survival.mean()

        return log_likeli_loss_time_next_without_dummy + integral_survival.sum(), loss_survival, marker_loss_time_next_without_dummy, \
               mse_loss, mae, f1, the_number_of_events


    def loss_function(self, integral_all_events, intensity_all_events, events_next, mask_next):
        '''
        This function computes the NLL loss at each legit event in events_next.
    
        ### Args
            * ```torch.tensor``` intensity_all_events
              shape: ```[batch_size, seq_len, num_events]```
              intensity values at t_i.
            * ```torch.tensor``` integral_all_events
              shape: ```[batch_size, seq_len, num_events]```
              intensity integral from t_{i - 1} to t_{i} (t_0 = 0).
            * ```torch.tensor``` events_next
              shape: ```[batch_size, seq_len]```
              The mark of the events that we need to predict.
            * ```torch.tensor``` mask_next
              shape: ```[batch_size, seq_len]```
              Needed mask to mask out unneeded loss values.
        
        ### Outputs
            * ```torch.tensor``` mtpp_loss, 
              shape: ```[1]```
              The sum of NLL loss on all event.
            * ```torch.tensor``` events_loss
              shape: ```[1]```
              The sum of the event loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real event.
        '''
        type_mask = F.one_hot(events_next, num_classes = self.num_events)      # [batch_size, seq_len, num_events]

        # MTPP loss function
        selected_intensity = (intensity_all_events * type_mask).sum(dim = -1)# [batch_size, seq_len]
        # selected_intensity = intensity_all_events.sum(dim = -1)                # [batch_size, seq_len]
        log_intensity = torch.log(selected_intensity + self.epsilon)           # [batch_size, seq_len]
        nll = -log_intensity + integral_all_events.sum(dim = -1)               # [batch_size, seq_len]
    
        mtpp_loss = torch.sum(nll * mask_next)

        # Event loss function. Only for evaluation, do not use this loss as a part of the training loss.
        events_prediction_probability = torch.log(intensity_all_events + self.epsilon)
                                                                               # [batch_size, seq_len, num_events]
        events_prediction_probability = F.softmax(events_prediction_probability, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
        reshaped_events_prediction_probability = rearrange(events_prediction_probability, 'b s ne -> b ne s')
                                                                               # [batch_size, num_events, seq_len]
        events_loss = F.cross_entropy(input = reshaped_events_prediction_probability, target = events_next, reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss = (events_loss * mask_next).sum()

        return mtpp_loss, events_loss


    def likelihood_loss(self, note_history, note_next, events_history, time_history, original_time_history, 
                        events_next, time_next, mask_next, mean, std, sparse_rate, note_embedding_history = None):
        # We use sparse rate to speed up training.
        # During training, we randomly remove events from the LLM augmentation process with probability sparse_rate.
        # This process should increase the LLM augmentation process, the main bottleneck of the training process, by 1 / (1 - sparse_rate) - 1 times.
        # For instance, sparse_rate = 0.5 should boost the LLM augmentation processing speed by 1 time.

        # step 1: get samples from the existing distribution.
        sampled_time = self.sample_time(sampling_approach = 'its', task = 'tm',
                                        events_history = events_history, time_history = time_history,
                                        number_of_total_samples = self.llm_contrast_sample_num, 
                                        step = self.llm_contrast_sample_step, 
                                        mean = mean, std = std, note_embedding_history = note_embedding_history)
                                                                               # [llm_contrast_sample_num, batch_size, seq_len]
        
        sampled_time = sampled_time.clone()                                    # [llm_contrast_sample_num, batch_size, seq_len]
        intensity_integral_all_events, intensity_all_events \
            = self.model(time_history, sampled_time, events_history, num_dimension_prior_batch = 1, note_embedding_history = note_embedding_history)
                                                                               # [llm_contrast_sample_num, batch_size, seq_len, num_events]
        mark_distribution = intensity_all_events / intensity_all_events.sum(dim = -1, keepdim = True)
                                                                               # [llm_contrast_sample_num, batch_size, seq_len, num_events]
        sampled_events = predict_event(mark_distribution, sample = True)       # [llm_contrast_sample_num, batch_size, seq_len]
        # merge the real next event with the pred_time
        # [real_pred_time, (sampled_times)]
        sampled_time, _ = pack((time_next, sampled_time), '* b s')             # [llm_contrast_sample_num + 1, batch_size, seq_len]
        sampled_events, _ = pack((events_next, sampled_events), '* b s')       # [llm_contrast_sample_num + 1, batch_size, seq_len]
        
        intensity_integral, intensity = self.model(time_history, time_next, events_history, num_dimension_prior_batch = 0, note_embedding_history = note_embedding_history)
                                                                               # [batch_size, seq_len, num_events]
        all_intensity, _ = pack((intensity, intensity_all_events), '* b s ne') # [llm_contrast_sample_num + 1, batch_size, seq_len, num_events]
        all_intensity_integral, _ = pack((intensity_integral, intensity_integral_all_events), '* b s ne')
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len, num_events]
        log_distribution_all = torch.log(all_intensity + self.epsilon) - all_intensity_integral.sum(dim = -1, keepdim = True)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len, num_events]
        sampled_events_one_hot = F.one_hot(sampled_events, num_classes = self.num_events)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len, num_events]
        log_selected_distribution = (log_distribution_all * sampled_events_one_hot).sum(dim = -1)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len]
        # Get score from the LLM.
        log_probs_by_llm, sparse_mask = self.get_score_from_llm(note_history, note_next, events_history, time_history, original_time_history, \
                                                                sampled_time, sampled_events, mask_next, sparse_rate = sparse_rate)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len] + [batch_size, seq_len]
        '''
        Version 1: KL divergence. It does not work.
        '''
        '''
        log_event_distribution_from_mtpp_model = F.log_softmax(log_selected_distribution, dim = 0)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len]
        # KL divengence loss.
        log_event_distribution_from_mtpp_model = rearrange(log_event_distribution_from_mtpp_model, 'lsn b s -> b s lsn')
                                                                               # [batch_size, seq_len, llm_contrast_sample_num + 1]
        log_probs_by_llm = rearrange(log_probs_by_llm, 'lsn b s -> b s lsn')   # [batch_size, seq_len, llm_contrast_sample_num + 1]
        kl_div = log_event_distribution_from_mtpp_model - log_probs_by_llm     # [batch_size, seq_len, llm_contrast_sample_num + 1]
        # kl_div = F.kl_div(input = log_probs_by_llm, target = log_event_distribution_from_mtpp_model, \
        #                   reduction = 'none', log_target = True)               # [batch_size, seq_len, llm_contrast_sample_num + 1]
        kl_div = log_event_distribution_from_mtpp_model - log_probs_by_llm     # [batch_size, seq_len, llm_contrast_sample_num + 1]
        # kl_div = F.kl_div(input = log_probs_by_llm, target = log_event_distribution_from_mtpp_model, \
        #                   reduction = 'none', log_target = True)               # [batch_size, seq_len, llm_contrast_sample_num + 1]
        
        kl_div = (kl_div.mean(dim = -1) * sparse_mask).sum()
        kl_div = (kl_div.mean(dim = -1) * sparse_mask).sum()
        
        # Avoid divided-by-0 exception if sparse_mask.sum() == 0
        if sparse_mask.sum() > 0:
            kl_div = kl_div / sparse_mask.sum().item()
        '''

        '''
        Version 2: ranking.
        If the real next event is promoted by the LLM, we use the negative log-likelihood, otherwise, we apply negative log-likelihood on the real next event and 
        the expected event.
        event_ranking_by_llm = torch.argsort(log_probs_by_llm, dim = 0)        # [llm_contrast_sample_num + 1, batch_size, seq_len]
        event_ranking_first_event_by_llm = event_ranking_by_llm == 0           # [llm_contrast_sample_num + 1, batch_size, seq_len]
        log_selected_distribution_of_first_event_selected_by_llm = (-log_selected_distribution * event_ranking_first_event_by_llm).sum(dim = 0) * sparse_mask
                                                                               # [batch_size, seq_len]
        kl_div = log_selected_distribution_of_first_event_selected_by_llm.sum() / sparse_mask.sum()
        '''
        
        '''
        Version 3: direct KL divergence
        The NLL will be weighted by the score from the LLM. We apply high weight on higher ranked samples with high LLM scores amd 
        low weight to lower ranked samples with low LLM scores.

        # KL divengence loss.
        log_event_distribution_from_mtpp_model = rearrange(log_selected_distribution, 'lsn b s -> b s lsn')
                                                                               # [batch_size, seq_len, llm_contrast_sample_num + 1]
        log_probs_by_llm = rearrange(log_probs_by_llm, 'lsn b s -> b s lsn')   # [batch_size, seq_len, llm_contrast_sample_num + 1]
        kl_div = F.kl_div(input = log_probs_by_llm, target = log_event_distribution_from_mtpp_model, \
                          reduction = 'none', log_target = True)               # [batch_size, seq_len, llm_contrast_sample_num + 1]
        
        kl_div = (kl_div.sum(dim = -1) * sparse_mask).sum()
        
        # Avoid divided-by-0 exception if sparse_mask.sum() == 0
        if sparse_mask.sum() > 0:
            kl_div = kl_div / sparse_mask.sum().item()
        '''

        '''
        Version 4: full ranking.
        If the real next event is promoted by the LLM, we use the negative log-likelihood, otherwise, we apply negative log-likelihood on the real next event and 
        the expected event.
        '''
        '''
        event_ranking_by_llm = torch.argsort(log_probs_by_llm, dim = 0)        # [llm_contrast_sample_num + 1, batch_size, seq_len]
        event_ranking_first_event_by_llm = event_ranking_by_llm == 0           # [llm_contrast_sample_num + 1, batch_size, seq_len]
        log_selected_distribution_of_first_event_selected_by_llm = (-log_selected_distribution * event_ranking_first_event_by_llm).sum(dim = 0) * sparse_mask
                                                                               # [batch_size, seq_len]
        kl_div = log_selected_distribution_of_first_event_selected_by_llm.sum() / sparse_mask.sum()
        '''

        '''
        Version 5: extended ranking.
        If the real next event is promoted by the LLM, we use the negative log-likelihood, otherwise, 
        we apply negative log-likelihood on the real next event and all sampled events ranked higher than the real event.
        '''
        '''
        event_ranking_by_llm = torch.argsort(log_probs_by_llm, dim = 0)        # [llm_contrast_sample_num + 1, batch_size, seq_len]
        event_ranking_of_real_events_by_llm = event_ranking_by_llm[0, ...]     # [batch_size, seq_len]
        event_ranking_first_event_by_llm = event_ranking_by_llm == repeat(event_ranking_of_real_events_by_llm, '... -> f ...', f = self.llm_contrast_sample_num + 1)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len]
        event_ranking_event_higher_than_realevents_by_llm = event_ranking_by_llm < repeat(event_ranking_of_real_events_by_llm, '... -> f ...', f = self.llm_contrast_sample_num + 1)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len]
        event_ranking_event_lower_than_realevents_by_llm = event_ranking_by_llm > repeat(event_ranking_of_real_events_by_llm, '... -> f ...', f = self.llm_contrast_sample_num + 1)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len]
        # the probability of real events.
        log_selected_distribution_of_first_event_selected_by_llm = (-log_selected_distribution * event_ranking_first_event_by_llm).sum(dim = 0) * sparse_mask
                                                                               # [batch_size, seq_len]
        # the probability of events whose ranking higher than real events.
        log_selected_distribution_of_event_higher_than_realevents_selected_by_llm = (-log_selected_distribution * event_ranking_event_higher_than_realevents_by_llm).sum(dim = 0) * sparse_mask
                                                                               # [batch_size, seq_len]
        # the probability of events whose ranking lower than real events.
        log_selected_distribution_of_event_lower_than_realevents_selected_by_llm = (-log_selected_distribution * event_ranking_event_lower_than_realevents_by_llm).sum(dim = 0) * sparse_mask
                                                                               # [batch_size, seq_len]
        
        kl_div = (log_selected_distribution_of_first_event_selected_by_llm.sum() + \
                  2.0 * log_selected_distribution_of_event_higher_than_realevents_selected_by_llm.sum() + \
                  2.0 * log_selected_distribution_of_event_lower_than_realevents_selected_by_llm.sum()) / sparse_mask.sum()
        
        '''
        Version 6: ranking-weighted NLL loss.
        If the real next event is promoted by the LLM, we use the negative log-likelihood, otherwise, 
        we apply negative log-likelihood on the real next event and all sampled events ranked higher than the real event.
        
        loss_weight_by_llm = F.softmax(log_probs_by_llm, dim = 0)              # [llm_contrast_sample_num + 1, batch_size, seq_len]
        kl_div = (-log_selected_distribution * loss_weight_by_llm).sum(dim = 0) * sparse_mask
                                                                               # [batch_size, seq_len]
        event_ranking_by_llm = torch.argsort(log_probs_by_llm, dim = 0)        # [llm_contrast_sample_num + 1, batch_size, seq_len]
        event_ranking_of_real_events_by_llm = event_ranking_by_llm[0, ...]     # [batch_size, seq_len]
        event_ranking_event_higher_than_realevents_by_llm = event_ranking_by_llm < repeat(event_ranking_of_real_events_by_llm, '... -> f ...', f = self.llm_contrast_sample_num + 1)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len]
        kl_div = kl_div.sum() / sparse_mask.sum() 
        '''
        
        # Our loss is expected to increase the sum of p^*(m, t) at sampled points. Is this true?
        masked_distribution_from_mtpp_model = log_selected_distribution.exp().sum(dim = 0) * mask_next
                                                                               # [batch_size, seq_len]
        average_probability_sum = masked_distribution_from_mtpp_model.sum() / mask_next.sum()
        
        return kl_div, average_probability_sum
      
    
    def get_score_from_llm(self, note_history, note_next, events_history, time_history, original_time_history, \
                           sampled_time, sampled_events, mask_next, sparse_rate = 0.0, \
                           llm_contrast_sample_num = None, llm_visible_history_length = None, llm_event_template = None, llm_next_event_template = None, \
                           get_max_length = False):
        llm_contrast_sample_num = self.llm_contrast_sample_num if llm_contrast_sample_num is None else llm_contrast_sample_num
        llm_visible_history_length = self.llm_visible_history_length if llm_visible_history_length is None else llm_visible_history_length
        llm_event_template = self.llm_event_template if llm_event_template is None else llm_event_template
        llm_next_event_template = self.llm_next_event_template if llm_next_event_template is None else llm_next_event_template
        batch_size, seq_len = events_history.shape
        
        # generate the sparse mask
        # The shape of the sampled_time: [batch_size, seq_len]
        dist = torch.distributions.uniform.Uniform(torch.zeros(*time_history.shape), torch.ones(*time_history.shape))
        samples = dist.sample()
        # sparse mask: 1: LLM augmented 0: Skipped.
        # If skipped, the LLM score will be 1 / (llm_contrast_sample_num + 1).
        # The downstream tasks should use the sparse_mask to remove skipped items from the training loss.
        sparse_mask = (samples > sparse_rate).to(self.device)                  # [batch_size, seq_len]
        sparse_mask = sparse_mask * mask_next                                  # [batch_size, seq_len]
        
        if get_max_length:
            max_lengths = []
        else:
            probs_by_llm = []
        for batch_index in range(batch_size):
            selected_note_history = note_history[batch_index]                  # [seq_len]
            selected_note_next = note_next[batch_index]                        # [seq_len]
            selected_events = events_history[batch_index]                      # [seq_len]
            selected_original_time = original_time_history[batch_index]        # [seq_len]
            selected_sparse_mask = sparse_mask[batch_index]                    # [seq_len]
            max_available_length = mask_next[batch_index].sum()
            
            assembled_requests = []
            beacons = []
            
            for seq_index in range(max_available_length):
                # 0: sparse_mask == 0: skip this event.
                # 1: sparse_mask == 1: continue.
                if selected_sparse_mask[seq_index] == 0:
                    continue
                
                llm_history_start_index = max(seq_index + 1 - llm_visible_history_length, 0)
                llm_history_end_index = seq_index + 1
                selected_note_history_events = selected_note_history[llm_history_start_index:llm_history_end_index]
                selected_note_next_event = selected_note_next[seq_index]
                selected_events_history = selected_events[llm_history_start_index:llm_history_end_index]
                selected_original_time_history = selected_original_time[llm_history_start_index:llm_history_end_index]
                
                selected_sampled_time = sampled_time[..., batch_index, seq_index]
                                                                               # [llm_contrast_sample_num + 1]
                selected_sampled_events = sampled_events[..., batch_index, seq_index]
                                                                               # [llm_contrast_sample_num + 1]
                
                selected_latest_original_time_history = datetime.strptime(selected_original_time_history[-1], '%y-%m-%d %H:%M:%S')
                selected_sampled_time = [(selected_latest_original_time_history + timedelta(days = delta.item())).strftime('%y-%m-%d %H:%M:%S') for delta in selected_sampled_time]
                
                selected_events_history, selected_sampled_time, selected_sampled_events = \
                    move_from_tensor_to_list(
                      selected_events_history, selected_sampled_time, selected_sampled_events
                    )

                history_sequence = 'History:\n' + \
                  ' '.join([llm_event_template.format(selected_note_history_, self.rosetta_mark_to_original[selected_events_history_], selected_original_time_history_) \
                            for selected_note_history_, selected_events_history_, selected_original_time_history_ in \
                            zip(selected_note_history_events, selected_events_history, selected_original_time_history)])
                sample_candidates_static = \
                [
                  self.llm.tokenize(llm_next_event_template[0].format(selected_note_next_event)) \
                          for selected_sampled_events_, selected_sampled_time_ in \
                          zip(selected_sampled_events, selected_sampled_time)
                ]
                sample_candidates_dynamic = \
                [
                  self.llm.tokenize(llm_next_event_template[1].format(self.rosetta_mark_to_original[selected_sampled_events_], selected_sampled_time_)) \
                          for selected_sampled_events_, selected_sampled_time_ in \
                          zip(selected_sampled_events, selected_sampled_time)
                ]
                sample_candidates = [static + dynamic for static, dynamic in zip(sample_candidates_static, sample_candidates_dynamic)]
                
                tokenized_history = self.llm.tokenize(history_sequence)
                assembled_request = [self.llm_prompt + tokenized_history + sample_candidate for sample_candidate in sample_candidates]
                                                                               # [llm_contrast_sample_num + 1]
                beacon = [-len(sample_candidate) for sample_candidate in sample_candidates_dynamic]
                                                                               # [llm_contrast_sample_num + 1]
                assembled_requests.extend(assembled_request)
                beacons.extend(beacon)
            
            assert len(assembled_requests) == len(beacons) == selected_sparse_mask.sum().item() * (llm_contrast_sample_num + 1)
            if get_max_length:
                max_lengths.append(max(*[len(seq) for seq in assembled_requests]))
            else:
                if self.llm_request_mode == 'online':
                    outputs = self.llm.completions(assembled_requests, n_threads = 16, max_tokens = 0, temperature = 0.0, logprobs = 0, echo = True)
                                                                               # [max_available_length * (llm_contrast_sample_num + 1)]
                    probs_per_batch = []
                    for idx, output in enumerate(outputs):
                        recorded_logprobs = output.choices[0].logprobs.token_logprobs[beacons[idx]:]
                        log_prob = np.mean(recorded_logprobs)
                        probs_per_batch.append(torch.tensor(log_prob, device = self.device))
                elif self.llm_request_mode == 'offline':
                    outputs = self.llm.completions(assembled_requests, n_threads = 16, max_tokens = 1, temperature = 0.0, prompt_logprobs = 1)
                                                                               # [max_available_length * (llm_contrast_sample_num + 1)]
                    probs_per_batch = []
                    for idx, output in enumerate(outputs):
                        raw_recorded_logprobs = output.prompt_logprobs[beacons[idx]:]
                        
                        recorded_logprobs = []
                        for item in raw_recorded_logprobs:
                            recorded_logprobs.append(list(item.values())[0].logprob)
                        
                        log_prob = np.mean(recorded_logprobs)
                        probs_per_batch.append(torch.tensor(log_prob, device = self.device))
                
                # Avoid condition that probs_per_batch is empty.
                if len(probs_per_batch) > 0:
                    probs_per_batch = torch.stack(probs_per_batch).reshape(-1, llm_contrast_sample_num + 1)
                                                                               # [selected_item, llm_contrast_sample_num + 1]
                processed_recorded_logprobs = []
                event_index = 0
                for item in selected_sparse_mask:
                    # item is 1: insert the corresponding LLM score.
                    if item == 1:
                        processed_recorded_logprobs.append(probs_per_batch[event_index])
                        event_index += 1
                    # item is 0: insert the placeholder. 
                    elif item == 0:
                        processed_recorded_logprobs.append(torch.ones(llm_contrast_sample_num + 1, device = self.device).log())
                
                assert len(processed_recorded_logprobs) == seq_len
                probs_per_batch = torch.stack(processed_recorded_logprobs, dim = -1)
                                                                               # [llm_contrast_sample_num + 1, seq_len]
                # probs_per_batch = rearrange(probs_per_batch, '(sl lcs) -> sl lcs', lcs = llm_contrast_sample_num + 1).T
                                                                               # [llm_contrast_sample_num + 1, max_available_length]
                # probs_per_batch = F.pad(probs_per_batch, (0, seq_len - max_available_length, 0, 0), mode = 'constant', value = 0)
                                                                               # [llm_contrast_sample_num + 1, seq_len]
                probs_by_llm.append(probs_per_batch)
        
        if get_max_length:
            return [max(*max_lengths), ] if len(max_lengths) > 1 else max_lengths
        else:
            probs_by_llm = torch.stack(probs_by_llm, axis = -2)                # [llm_contrast_sample_num + 1, batch_size, seq_len]
            log_probs_by_llm = F.log_softmax(probs_by_llm, dim = 0)            # [llm_contrast_sample_num + 1, batch_size, seq_len]
            # log_probs_by_llm = probs_by_llm                                  # [llm_contrast_sample_num + 1, batch_size, seq_len]

            return log_probs_by_llm, sparse_mask
    
    
    sample_time = sample_time
    sample_time_event = sample_time_event
    sample_event_time = sample_event_time


    @torch.inference_mode()
    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, 
                                   mask_history, mask_next, mean, std, opt = None, note_embedding_history = None, note_embedding_next = None, 
                                   output_pred_note_embedding_mse = False):
        '''
        Called by evaluate_procedure(), debug() and get_mae_and_f1(), this function computed the MAE and macro-F1 of one minibatch.

        ### Args
            * ```torch.tensor``` events_history
              shape: ```[batch_size, seq_len]```
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              The event history \\mathcal{H}_{t_l}. We use these history info and time history for \\(\\lambda^*(m, t)\\) and \\(\\Lambda^*(m, t)\\).
            * ```torch.tensor``` events_next
              shape: ```[batch_size, seq_len]```
            * ```torch.tensor``` time_next
              shape: ```[batch_size, seq_len]```
            * ```torch.tensor``` mask_next
              shape: ```[batch_size, seq_len]```
              The real-world event sequence. We use events in this sequence to evaluate the predicted events.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` mae
              shape: ```[batch_size, seq_len]```
              Mean Absolute Error(MAE) between predicted times \\(t_p\\) and ground truths \\(t_i\\). MAE = |t_p - t_i|.
            * ```float``` f1
              macro-F1 value between events predicted at \\(t_p\\) and the ground truths.
            * ```torch.tensor``` mark_distribution
              shape: ```[batch_size, seq_len, num_events]```
              The mark distribution at the real time.
        '''
        pred_time = self.sample_time(sampling_approach = 'its', task = 'tm',
                                     events_history = events_history, time_history = time_history, mask_history = mask_history,
                                     number_of_total_samples = self.sample_rate if opt is None else opt.sample_rate, 
                                     step = self.mae_step if opt is None else opt.mae_step, 
                                     mean = mean, std = std, note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next)
                                                                               # [sample_rate, batch_size, seq_len]
        pred_time = pred_time.mean(dim = 0)                                    # [batch_size, seq_len]
        mae = torch.abs(pred_time - time_next) * mask_next                     # [batch_size, seq_len]
        if output_pred_note_embedding_mse:
            _, intensity_all_events, note_pred_emb = self.model(time_history, time_next, events_history, mask_history, \
                                                                note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next, \
                                                                output_pred_note_embedding = output_pred_note_embedding_mse)
                                                                               # [batch_size, seq_len, num_events]
            mse = torch.pow(note_embedding_next[..., :self.d_mark_embedding] - note_pred_emb, 2)
                                                                               # [batch_size, seq_len, note_embedding_size]
            mse = mse.sum(dim = -1)                                            # [batch_size, seq_len]
        else:
            _, intensity_all_events = self.model(time_history, time_next, events_history, mask_history, \
                                                 note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next, \
                                                 output_pred_note_embedding = output_pred_note_embedding_mse)
                                                                               # [batch_size, seq_len, num_events]
            
        mark_distribution = intensity_all_events / intensity_all_events.sum(dim = -1, keepdim = True)
                                                                               # [batch_size, seq_len, num_events]
        predicted_events = torch.argmax(intensity_all_events, dim = -1)[mask_next == 1]
        events_true = events_next[mask_next == 1]
        predicted_events, events_true = move_from_tensor_to_ndarray(predicted_events, events_true)
        f1 = f1_score(y_pred = predicted_events, y_true = events_true, average = 'macro')
        
        if output_pred_note_embedding_mse:
            return mae, f1, mark_distribution, mse
        else:
            return mae, f1, mark_distribution


    @torch.inference_mode()
    def mean_absolute_error_e(self, time_history, time_next, events_history, events_next, 
                              mask_history, mask_next, mean, std, return_mean = True, opt = None, 
                              note_embedding_history = None, note_embedding_next = None, \
                              output_pred_note_embedding_mse = False):
        '''
        Called by debug() and get_mae_e_and_f1(), this function computed the MAE-E and macro-F1 of one minibatch.

        ### Args
            * ```torch.tensor``` events_history
              shape: ```[batch_size, seq_len]```
              Historical event sequences. Commonly, this sequence is a slice of the original event sequence from 0 to seq_len - 1(included).
            * ```torch.tensor``` events_next
              shape: ```[batch_size, seq_len]```
              The mark of the events that we need to predict.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequences. Similar to events_history, we always generate this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
            * ```torch.tensor``` time_next
              shape: ```[batch_size, seq_len, num_events]```
              When the next event actually happens. 
            * ```torch.tensor``` mask_next
              shape: ```[batch_size, seq_len]```
              Needed mask to mask out unneeded loss values.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.
            * ```bool``` return_mean
              If true, we compute the mean of mae_per_event_with_predict_index and mae_per_event_with_event_next on all events in the minibatch.
              If false, we compute the mean of mae_per_event_with_predict_index and mae_per_event_with_event_next per sequence.
            * ```namespace``` opt
              One may bring custom settings into this function through this argument during evaluation. Please refers to
              debug() and get_mae_e_and_f1() for more information about what custom settings are available.

        ### Outputs
            * ```float``` f1
              macro-F1 value between events predicted and the ground truths.
            * ```list``` top_k_acc
              top-1 to top-N accuracy value between events predicted and the ground truths.
            * ```torch.tensor``` probability_integral_sum
              shape: ```[batch_size, seq_len]```
              The sum of p(m) over m.
            * ```torch.tensor``` p_m
              shape: ```[batch_size, seq_len, num_events]```
              The value of p(m) over the different mark m.
            * ```torch.tensor``` tau_pred_all_event
              shape: ```[batch_size, seq_len, num_events]```
              Time predicted by p(t|m) over all marks m.
            * ```torch.tensor``` mae_per_event_with_predict_index_avg
              shape: ```[batch_size]``` if return_mean else ```[1]```
              The average of MAE-E when we pick predicted times using predicted marks.
            * ```torch.tensor``` mae_per_event_with_event_next_avg
              shape: ```[batch_size]``` if return_mean else ```[1]```
              The average of MAE-E when we pick predicted times using real marks.
            * ```torch.tensor``` mae_per_event_with_predict_index
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using predicted marks.
            * ```torch.tensor``` mae_per_event_with_event_next
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using real marks.
        '''
        inf_val, resolution_inf, resolution_between_events \
            = decide_resolution_inf_and_resolution_between_events(time_next, memory_ceiling, self.num_events, mean, std)
        time_next_inf = torch.ones_like(time_history, device = self.device) * inf_val
        
        if output_pred_note_embedding_mse:
            expanded_integral_all_events_to_inf, expanded_intensity_all_events_to_inf, timestamp, note_pred_emb = \
                self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, mask_history, resolution_inf, 
                                                           note_embedding_history = note_embedding_history, \
                                                           output_pred_note_embedding = output_pred_note_embedding_mse, \
                                                           note_embedding_next = note_embedding_next)
                                                                               # 2 * [batch_size, seq_len, resolution_inf, num_events]
            mse = torch.pow(note_embedding_next[..., :self.d_mark_embedding] - note_pred_emb, 2)
                                                                               # [batch_size, seq_len, note_embedding_size]
            mse = mse.sum(dim = -1)                                            # [batch_size, seq_len]
        else:
            expanded_integral_all_events_to_inf, expanded_intensity_all_events_to_inf, timestamp = \
                self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, mask_history, resolution_inf, 
                                                           note_embedding_history = note_embedding_history, \
                                                           output_pred_note_embedding = output_pred_note_embedding_mse, \
                                                           note_embedding_next = note_embedding_next)
                                                                               # 2 * [batch_size, seq_len, resolution_inf, num_events]
                                                                                   
        expanded_integral_sum_over_events_to_inf = expanded_integral_all_events_to_inf.sum(dim = -1, keepdim = True)
                                                                               # [batch_size, seq_len, resolution_inf, 1]
        expanded_probability_inf = expanded_intensity_all_events_to_inf * torch.exp(-expanded_integral_sum_over_events_to_inf)
                                                                               # [batch_size, seq_len, resolution_inf, num_events]
        probability_integral_to_inf = approximate_integration(expanded_probability_inf, timestamp, dim = -2, only_integral = True)
                                                                               # [batch_size, seq_len, num_events]
        probability_integral_sum = probability_integral_to_inf.sum(dim = -1)   # [batch_size, seq_len]
        predicted_events = torch.argmax(probability_integral_to_inf, dim = -1) # [batch_size, seq_len]

        f1, top_k_acc = get_f1_and_top_k_acc_in_mae_e(events_next, probability_integral_to_inf, mask_next, self.num_events)

        tau_pred_all_event = self.sample_time(sampling_approach = 'its', task = 'mt', 
                                              events_history = events_history, time_history = time_history, mask_history = mask_history,
                                              p_m = probability_integral_to_inf, resolution = resolution_between_events,
                                              number_of_total_samples = self.sample_rate if opt is None else opt.sample_rate,
                                              step = self.mae_e_step if opt is None else opt.mae_e_step, 
                                              inf_val = inf_val, 
                                              mean = mean, std = std, 
                                              note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        tau_pred_all_event = tau_pred_all_event.mean(dim = 0)                  # [batch_size, seq_len, num_events]
 
        predicted_event_mask = F.one_hot(predicted_events.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        event_next_mask = F.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]

        if return_mean:
            mae_per_event_with_predict_index = torch.abs((tau_pred_all_event * predicted_event_mask).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
            mae_per_event_with_event_next = torch.abs((tau_pred_all_event * event_next_mask).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
    
            mae_per_event_with_predict_index_avg = torch.sum(mae_per_event_with_predict_index, dim = -1) / mask_next.sum(dim = -1)
            mae_per_event_with_event_next_avg = torch.sum(mae_per_event_with_event_next, dim = -1) / mask_next.sum(dim = -1)
        else:
            mae_per_event_with_predict_index = torch.abs((tau_pred_all_event * predicted_event_mask.unsqueeze(dim = 0)).sum(dim = -1) - time_next) * mask_next.unsqueeze(dim = 0)
                                                                               # [sample_rate, batch_size, seq_len]
            mae_per_event_with_event_next = torch.abs((tau_pred_all_event * event_next_mask.unsqueeze(dim = 0)).sum(dim = -1) - time_next) * mask_next.unsqueeze(dim = 0)
                                                                               # [sample_rate, batch_size, seq_len]
    
            mae_per_event_with_predict_index_avg = torch.sum(mae_per_event_with_predict_index, dim = -1) / mask_next.sum(dim = -1)
                                                                               # [sample_rate, batch_size]
            mae_per_event_with_event_next_avg = torch.sum(mae_per_event_with_event_next, dim = -1) / mask_next.sum(dim = -1)
                                                                               # [sample_rate, batch_size]
            
            # Calculate mean
            mae_per_event_with_predict_index = mae_per_event_with_predict_index.mean(dim = 0)
                                                                               # [batch_size, seq_len]
            mae_per_event_with_event_next = mae_per_event_with_event_next.mean(dim = 0)
                                                                               # [batch_size, seq_len]
            mae_per_event_with_predict_index_avg = mae_per_event_with_predict_index_avg.mean(dim = 0)
                                                                               # [batch_size]
            mae_per_event_with_event_next_avg = mae_per_event_with_event_next_avg.mean(dim = 0)
                                                                               # [batch_size]

        if output_pred_note_embedding_mse:
            return f1, top_k_acc, probability_integral_sum, probability_integral_to_inf, \
                   tau_pred_all_event, mse, (mae_per_event_with_predict_index_avg, mae_per_event_with_event_next_avg), \
                   (mae_per_event_with_predict_index, mae_per_event_with_event_next)
        else:
            return f1, top_k_acc, probability_integral_sum, probability_integral_to_inf, \
                   tau_pred_all_event, (mae_per_event_with_predict_index_avg, mae_per_event_with_event_next_avg), \
                   (mae_per_event_with_predict_index, mae_per_event_with_event_next)


    def extract_plot_data(self, minibatch):
        '''
        This function extracts input_time, input_events, input_intensity, mask, mean, and std from the minibatch.

        ### Args
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
              
        ### Outputs
            * ```torch.tensor``` input_time
              shape: ```[batch_size, seq_len + 1]```
              Raw event timestamp sequence.
            * ```torch.tensor``` input_events
              shape: ```[batch_size, seq_len + 1]```
              Raw event marks sequence.
            * ```torch.tensor``` mask
              shape: ```[batch_size, seq_len + 1]```
              Raw mask sequence.
            * ```int``` mean
              The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide this value if needed.
            * ```int``` std
              The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide this value if needed.
        '''
        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, _, mask, input_intensity = minibatch[0]
        mean, std = minibatch[1]

        return input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, mean, std
    

    @torch.inference_mode()
    def figure_intensity(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots of the intensity function.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        '''
        argument_check(opt, **{'resolution': int})
        
        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, \
        mean, std = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        note_embedding_history, note_embedding_next = None, None
        if self.mtpp_includes_note_embedding:
            note_embedding_history, note_embedding_next = self.divide_history_and_next(input_note_embeddings)
                                                                               # [batch_size, seq_len, dim_note_embedding] * 2

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution, \
                                                       note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]
        
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_intensity': expand_intensity,
            'input_intensity': input_intensity,
            'timestamp': timestamp}
        
        generate_intensity_figure(data, opt)


    @torch.inference_mode()
    def figure_integral(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots of the integral of the intensity function.
        
        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        '''
        argument_check(opt, **{'resolution': int})

        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, \
        mean, std = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        note_embedding_history, note_embedding_next = None, None
        if self.mtpp_includes_note_embedding:
            note_embedding_history, note_embedding_next = self.divide_history_and_next(input_note_embeddings)
                                                                               # [batch_size, seq_len, dim_note_embedding] * 2

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution, \
                                                       note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_integral': expand_integral,
            'input_intensity': input_intensity,
            'timestamp': timestamp}
        
        generate_integral_figure(data, opt)


    @torch.inference_mode()
    def figure_probability(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots of the probability distribution.
        
        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        '''
        argument_check(opt, **{'resolution': int})
        
        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, \
        mean, std = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        note_embedding_history, note_embedding_next = None, None
        if self.mtpp_includes_note_embedding:
            note_embedding_history, note_embedding_next = self.divide_history_and_next(input_note_embeddings)
                                                                               # [batch_size, seq_len, dim_note_embedding] * 2

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution, \
                                                       note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next)
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
            'input_intensity': input_intensity,
            'timestamp': timestamp}
        
        generate_probability_figure(data, opt)


    @torch.inference_mode()
    def figure_debug(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots for deeper insight of intensity functions and other metrics.
        
        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        2. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                  The number of interpolated points counts the start and end point of the interval.
        3. ```int``` mae_step: This parameter controls how many samples are generated in one shot when sampling from p(t).
        4. ```int``` mae_e_step: This parameter controls how many samples are generated in one shot when sampling from all p(t|m)s at the same time.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        '''
        argument_check(opt, **{'resolution': int, 'sample_rate': int, 'mae_step': int, 'mae_e_step': int})

        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, \
        mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        note_embedding_history, note_embedding_next = None, None
        if self.mtpp_includes_note_embedding:
            note_embedding_history, note_embedding_next = self.divide_history_and_next(input_note_embeddings)
                                                                               # [batch_size, seq_len, dim_note_embedding] * 2

        mae, f1_1, _ = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                         time_next, mask_history, mask_next, mean, std, opt, \
                                                         note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next)
                                                                               # [batch_size, seq_len]
                                                                               
        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, mask_history, mask_next, opt.resolution, \
                                                          note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next)
        
        f1_2, top_k, probability_sum, _, tau_pred_all_event, maes_avg, maes \
                = self.mean_absolute_error_e(time_history, time_next, events_history, \
                                             events_next, mask_history, mask_next, mean, std, opt = opt, \
                                             note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next, \
                                             output_pred_note_embedding_mse = False)

        # Append additional info into the data dict.
        data.update({
            'events_next': events_next,
            'time_next': time_next,
            'mask_next': mask_next,
            'f1_after_time_pred': f1_1,
            'mae_before_event': mae,
            'f1_before_time_pred': f1_2,
            'top_k': top_k,
            'probability_sum': probability_sum,
            'tau_pred_all_event': tau_pred_all_event,
            'maes_after_event_avg': maes_avg,
            'maes_after_event': maes,
            'timestamp': timestamp
        })

        generate_debug_figure(data, opt)


    # Evaluation over the entire dataset.
    @torch.inference_mode()
    def get_spearman_and_l1(self, input_data, opt):
        '''
        Used by evaluator to calculate the average gap between the predicted and real distribution using L1 distance and spearman coefficient.
        
        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        
        ### Outputs:
            * ```float``` spearman
              The spearman coefficient between the predicted and real distribution.
            * ```float``` l1
              The l1 distance between the predicted and real distribution.
        '''
        argument_check(opt, **{'resolution', int})
        
        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, \
        mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]

        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape
        expand_probability = expand_intensity * torch.exp(-expand_integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        expand_probability = expand_probability.sum(dim = -1)                  # [batch_size, seq_len, resolution]
        true_probability = expand_true_probability(time_next, input_intensity, opt)
                                                                               # [batch_size, seq_len, resolution] or batch_size * None
        
        expand_probability, true_probability, timestamp = move_from_tensor_to_ndarray(expand_probability, true_probability, timestamp)
        zipped_data = zip(expand_probability, true_probability, timestamp, mask_next)

        spearman = 0
        l1 = 0
        for expand_probability_per_seq, true_probability_per_seq, timestamp_per_seq, mask_next_per_seq in zipped_data:
            seq_len = mask_next_per_seq.sum()

            spearman_per_seq = \
                spearmanr(expand_probability_per_seq[:seq_len, :].flatten(), true_probability_per_seq[:seq_len, :].flatten())[0]

            l1_per_seq = L1_distance_between_two_funcs(x = true_probability_per_seq[:seq_len, :], y = expand_probability_per_seq[:seq_len, :], \
                                                       timestamp = timestamp_per_seq)
            spearman += spearman_per_seq
            l1 += l1_per_seq

        batch_size = mask_next.shape[0]
        spearman /= batch_size
        l1 /= batch_size

        return spearman, l1
    

    @torch.inference_mode()
    def get_mae_and_f1(self, input_data, opt):
        '''
        Used by evaluator to evaluate the performance of predicted time from p(t) and mark from p(m|t).
        
        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                  The number of interpolated points counts the start and end point of the interval.
        2. ```int``` mae_step: This parameter controls how many samples are generated in one shot when sampling from p(t).
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        
        ### Outputs:
            * ```np.ndarray``` mae
              shape: ```[batch_size, seq_len]```
              The MAE value, which is the time gap between each predicted and real event.
            * ```float``` f1_1
              The f1 value shows the accuracy of the predicted marks.
            * ```np.ndarray``` p_m
              shape: ```[batch_size, seq_len]```
              Predicted mark distribution at when an event is observed.
            * ```np.ndarray``` events_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed events.
        '''
        argument_check(opt, **{'sample_rate': int, 'mae_step': int})
        
        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, \
        mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        note_embedding_history, note_embedding_next = None, None
        if self.mtpp_includes_note_embedding:
            note_embedding_history, note_embedding_next = self.divide_history_and_next(input_note_embeddings)
                                                                               # [batch_size, seq_len, dim_note_embedding] * 2
        
        mse = 0
        if self.mtpp_includes_note_embedding:
            mae, f1_1, p_m, mse = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                                  time_next, mask_history, mask_next, mean, std, opt, \
                                                                  note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next, \
                                                                  output_pred_note_embedding_mse = True)
                                                                                   # [batch_size, seq_len]
        else:
            mae, f1_1, p_m = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                             time_next, mask_history, mask_next, mean, std, opt, \
                                                             note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next)
                                                                                   # [batch_size, seq_len]
        
        mae, events_next, p_m, mse = move_from_tensor_to_ndarray(mae, events_next, p_m, mse)

        return mae, f1_1, p_m, mse, events_next

    
    @torch.inference_mode()
    def get_mae_e_and_f1(self, input_data, opt):
        '''
        Used by evaluator to evaluate the performance of predicted time from p(m) and mark from p(t|m).
        
        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                  The number of interpolated points counts the start and end point of the interval.
        2. ```int``` mae_e_step: This parameter controls how many samples are generated in one shot when sampling from p(t|m).
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` maes
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using real marks.
            * ```float``` f1_2
              The f1 value shows the accuracy of the predicted marks.
            * ```np.ndarray``` probability_sum
              shape: ```[batch_size, seq_len]```
              The sum of calculated p(m) over all marks.
            * ```np.adarray``` p_m
              shape: ```[batch_size, seq_len, num_events]```
              The value of calculated p(m).
            * ```np.ndarray``` tau_pred_all_event
              shape: ```[batch_size, seq_len, num_events]```
              The predicted time for each mark using p(t|m).
            * ```np.ndarray``` time_next
              shape: ```[batch_size, seq_len]```
              Real time of observed events.
            * ```np.ndarray``` events_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed events.
        '''
        argument_check(opt, **{'sample_rate': int, 'mae_e_step': int})
        
        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, \
        mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        note_embedding_history, note_embedding_next = None, None
        if self.mtpp_includes_note_embedding:
            note_embedding_history, note_embedding_next = self.divide_history_and_next(input_note_embeddings)
                                                                               # [batch_size, seq_len, dim_note_embedding] * 2

        mse = 0
        if self.mtpp_includes_note_embedding:
            f1_2, top_k, probability_sum, p_m, tau_pred_all_event, mse, maes_avg, maes \
                = self.mean_absolute_error_e(time_history, time_next, events_history, \
                                             events_next, mask_history, mask_next, mean, std, opt = opt, \
                                             note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next, \
                                             output_pred_note_embedding_mse = True)
        else:
            f1_2, top_k, probability_sum, p_m, tau_pred_all_event, maes_avg, maes \
                = self.mean_absolute_error_e(time_history, time_next, events_history, \
                                             events_next, mask_history, mask_next, mean, std, opt = opt, \
                                             note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next, \
                                             output_pred_note_embedding_mse = False)
        
        _, maes, probability_sum, p_m, tau_pred_all_event, time_next, events_next, mse \
            = move_from_tensor_to_ndarray(*maes, probability_sum, p_m, tau_pred_all_event, time_next, events_next, mse)

        return maes, f1_2, probability_sum, p_m, tau_pred_all_event, mse, time_next, events_next


    @torch.inference_mode()
    def llm_max_token_length(self, input_data, opt):
        '''
        This function is used to confirm that:
            1. the history note has sufficient info for deciding which next event makes more sense (Verified in NTPP).
            2. the LLM can use the history info in the note to decide which next event makes more sense (Verified with and without notes).
        
        This function is placed here to verify the second point
        
        You should declare the following arguments in your config file:
        1. ```int``` llm_sample_rate: This parameter controls how many samples are sampled from p(t).
        2. ```int``` llm_sample_step: This parameter controls how many samples are sampled in one shot from p(t).
        3. ```str``` llm_prompt: system prompt.
        
        # The following arguments are provided in llm.yml.
        4. ```str``` url_path: the url linked to the LLM service.
        5. ```str``` api_key: API key for user authentication.
        6. ```str``` llm_model: Name of the used LLM.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` maes
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using real marks.
            * ```float``` f1_2
              The f1 value shows the accuracy of the predicted marks.
            * ```np.ndarray``` probability_sum
              shape: ```[batch_size, seq_len]```
              The sum of calculated p(m) over all marks.
            * ```np.adarray``` p_m
              shape: ```[batch_size, seq_len, num_events]```
              The value of calculated p(m).
            * ```np.ndarray``` tau_pred_all_event
              shape: ```[batch_size, seq_len, num_events]```
              The predicted time for each mark using p(t|m).
            * ```np.ndarray``` time_next
              shape: ```[batch_size, seq_len]```
              Real time of observed events.
            * ```np.ndarray``` events_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed events.
        '''
        argument_check(opt, llm_request_mode = str)
        if opt.llm_request_mode == 'online':
            argument_check(opt, **{'llm_sample_rate': int, 'llm_sample_step': int, 'llm_visible_history_length': int, \
                                   'url_path': str, 'api_key': str, 'llm_model': str, 'llm_prompt': str, \
                                   'llm_event_template': str, 'llm_next_event_template': list})
            if not hasattr(self, 'url_path'):
                self.url_path = opt.url_path
                self.api_key = opt.api_key
                self.llm_model = opt.llm_model
                self.llm = CustomOpenAIforVLLM(self.url_path, self.llm_model, device = self.device, api_key = self.api_key)
                self.llm_prompt = self.llm.tokenize(opt.llm_prompt)
        else:
            argument_check(opt, **{'llm_sample_rate': int, 'llm_sample_step': int, 'llm_visible_history_length': int, \
                                   'llm_model': str, 'llm_prompt': str, 'llm_model_args': dict, 
                                   'llm_event_template': str, 'llm_next_event_template': list})
            if not hasattr(self, 'llm_model'):
                self.llm_model = opt.llm_model
                self.llm = VLLMOfflineInference(self.llm_model, device = self.device, model_args = opt.llm_model_args)
                self.llm_prompt = self.llm.tokenize(opt.llm_prompt)
        
        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, \
        mean, std = self.extract_plot_data(input_data)
        note_history, note_next = self.divide_history_and_next(input_notes)    # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        original_time_history, _ = self.divide_history_and_next(input_original_time)
                                                                               # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        original_events_history, original_events_next = self.divide_history_and_next(input_original_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        
        # step 1: get samples from the existing distribution.
        sampled_time = self.sample_time(sampling_approach = 'its', task = 'tm',
                                        events_history = events_history, time_history = time_history,
                                        number_of_total_samples = opt.llm_sample_rate, 
                                        step = opt.llm_sample_step, mean = mean, std = std)
                                                                               # [llm_contrast_sample_num, batch_size, seq_len]

        _, intensity_all_events \
            = self.model(time_history, sampled_time, events_history, num_dimension_prior_batch = 1)
                                                                               # [llm_contrast_sample_num, batch_size, seq_len, num_events]
        mark_distribution = intensity_all_events / intensity_all_events.sum(dim = -1, keepdim = True)
                                                                               # [llm_contrast_sample_num, batch_size, seq_len, num_events]
        sampled_events = predict_event(mark_distribution, sample = True)       # [llm_contrast_sample_num, batch_size, seq_len]
        # merge the real next event with the pred_time
        # [real_pred_time, (sampled_times)]
        sampled_time, _ = pack((time_next, sampled_time), '* b s')             # [llm_contrast_sample_num + 1, batch_size, seq_len]
        sampled_events, _ = pack((events_next, sampled_events), '* b s')       # [llm_contrast_sample_num + 1, batch_size, seq_len]
        
        max_seq_len = self.get_score_from_llm(note_history, note_next, events_history, time_history, original_time_history, \
                                              sampled_time, sampled_events, mask_next, \
                                              llm_contrast_sample_num = opt.llm_sample_rate, llm_visible_history_length = opt.llm_visible_history_length, \
                                              llm_event_template = opt.llm_event_template, llm_next_event_template = opt.llm_next_event_template, \
                                              get_max_length = True)           # [llm_contrast_sample_num + 1, batch_size, seq_len]        
        return max_seq_len


    @torch.inference_mode()
    def llm_mtpp_classification(self, input_data, opt):
        '''
        This function is used to confirm that:
            1. the history note has sufficient info for deciding which next event makes more sense (Verified in NTPP).
            2. the LLM can use the history info in the note to decide which next event makes more sense (Verified with and without notes).
        
        This function is placed here to verify the second point
        
        You should declare the following arguments in your config file:
        1. ```int``` llm_sample_rate: This parameter controls how many samples are sampled from p(t).
        2. ```int``` llm_sample_step: This parameter controls how many samples are sampled in one shot from p(t).
        3. ```str``` llm_prompt: system prompt.
        
        # The following arguments are provided in llm.yml.
        4. ```str``` url_path: the url linked to the LLM service.
        5. ```str``` api_key: API key for user authentication.
        6. ```str``` llm_model: Name of the used LLM.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` maes
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using real marks.
            * ```float``` f1_2
              The f1 value shows the accuracy of the predicted marks.
            * ```np.ndarray``` probability_sum
              shape: ```[batch_size, seq_len]```
              The sum of calculated p(m) over all marks.
            * ```np.adarray``` p_m
              shape: ```[batch_size, seq_len, num_events]```
              The value of calculated p(m).
            * ```np.ndarray``` tau_pred_all_event
              shape: ```[batch_size, seq_len, num_events]```
              The predicted time for each mark using p(t|m).
            * ```np.ndarray``` time_next
              shape: ```[batch_size, seq_len]```
              Real time of observed events.
            * ```np.ndarray``` events_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed events.
        '''
        argument_check(opt, llm_request_mode = str)
        if opt.llm_request_mode == 'online':
            argument_check(opt, **{'llm_sample_rate': int, 'llm_sample_step': int, 'llm_visible_history_length': int, \
                                   'url_path': str, 'api_key': str, 'llm_model': str, 'llm_prompt': str, \
                                   'llm_event_template': str, 'llm_next_event_template': list})
            if not hasattr(self, 'url_path'):
                self.llm_request_mode = opt.llm_request_mode
                self.url_path = opt.url_path
                self.api_key = opt.api_key
                self.llm_model = opt.llm_model
                self.llm = CustomOpenAIforVLLM(self.url_path, self.llm_model, device = self.device, api_key = self.api_key)
                self.llm_prompt = self.llm.tokenize(opt.llm_prompt)
        else:
            argument_check(opt, **{'llm_sample_rate': int, 'llm_sample_step': int, 'llm_visible_history_length': int, \
                                   'llm_model': str, 'llm_prompt': str, 'llm_model_args': dict, 
                                   'llm_event_template': str, 'llm_next_event_template': list})
            if not hasattr(self, 'llm_model'):
                self.llm_request_mode = opt.llm_request_mode
                self.llm_model = opt.llm_model
                self.llm = VLLMOfflineInference(self.llm_model, device = self.device, model_args = opt.llm_model_args)
                # self.llm = VLLMOfflineInference(self.llm_model, device = self.device, model_args = {**opt.llm_model_args, "enforce_eager": True})
                self.llm_prompt = self.llm.tokenize(opt.llm_prompt)
        
        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, \
        mean, std = self.extract_plot_data(input_data)
        note_history, note_next = self.divide_history_and_next(input_notes)    # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        original_time_history, _ = self.divide_history_and_next(input_original_time)
                                                                               # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        
        note_embedding_history = None
        if self.mtpp_includes_note_embedding:
            note_embedding_history, _ = self.divide_history_and_next(input_note_embeddings)
                                                                               # [batch_size, seq_len, dim_note_embedding] * 2
        
        # step 1: get samples from the existing distribution.
        sampled_time = self.sample_time(sampling_approach = 'its', task = 'tm',
                                        events_history = events_history, time_history = time_history,
                                        number_of_total_samples = opt.llm_sample_rate, 
                                        step = opt.llm_sample_step, mean = mean, std = std, note_embedding_history = note_embedding_history)
                                                                               # [llm_contrast_sample_num, batch_size, seq_len]

        intensity_integral_all_events, intensity_all_events \
            = self.model(time_history, sampled_time, events_history, num_dimension_prior_batch = 1, note_embedding_history = note_embedding_history)
                                                                               # [llm_contrast_sample_num, batch_size, seq_len, num_events]
        mark_distribution = intensity_all_events / intensity_all_events.sum(dim = -1, keepdim = True)
                                                                               # [llm_contrast_sample_num, batch_size, seq_len, num_events]
        sampled_events = predict_event(mark_distribution, sample = True)       # [llm_contrast_sample_num, batch_size, seq_len]
        # merge the real next event with the pred_time
        # [real_pred_time, (sampled_times)]
        sampled_time, _ = pack((time_next, sampled_time), '* b s')             # [llm_contrast_sample_num + 1, batch_size, seq_len]
        sampled_events, _ = pack((events_next, sampled_events), '* b s')       # [llm_contrast_sample_num + 1, batch_size, seq_len]
        
        intensity_integral, intensity = self.model(time_history, time_next, events_history, num_dimension_prior_batch = 0, note_embedding_history = note_embedding_history)
                                                                               # [batch_size, seq_len, num_events]
        all_intensity, _ = pack((intensity, intensity_all_events), '* b s ne') # [llm_contrast_sample_num + 1, batch_size, seq_len, num_events]
        all_intensity_integral, _ = pack((intensity_integral, intensity_integral_all_events), '* b s ne')
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len, num_events]
        log_distribution_all = torch.log(all_intensity + self.epsilon) - all_intensity_integral.sum(dim = -1, keepdim = True)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len, num_events]
        sampled_events_one_hot = F.one_hot(sampled_events, num_classes = self.num_events)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len, num_events]
        log_selected_distribution = (log_distribution_all * sampled_events_one_hot).sum(dim = -1)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len]
        log_event_distribution_from_mtpp_model = F.log_softmax(log_selected_distribution, dim = 0)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len]
        # During inference, the sparse_rate should be 0, so the sparse_mask can be ignored as all events are processed.
        log_probs_by_llm, _ = self.get_score_from_llm(note_history, note_next, events_history, time_history, original_time_history, \
                                                      sampled_time, sampled_events, mask_next, \
                                                      llm_contrast_sample_num = opt.llm_sample_rate, llm_visible_history_length = opt.llm_visible_history_length, \
                                                      llm_event_template = opt.llm_event_template, llm_next_event_template = opt.llm_next_event_template)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len]
        
        # KL divengence loss.
        log_event_distribution_from_mtpp_model_for_kl = rearrange(log_event_distribution_from_mtpp_model, 'lsn b s -> b s lsn')
                                                                               # [batch_size, seq_len, llm_contrast_sample_num + 1]
        log_probs_by_llm_for_kl = rearrange(log_probs_by_llm, 'lsn b s -> b s lsn')
                                                                               # [batch_size, seq_len, llm_contrast_sample_num + 1]
        kl_div = F.kl_div(input = log_event_distribution_from_mtpp_model_for_kl, target = log_probs_by_llm_for_kl, \
                          reduction = 'none', log_target = True)               # [batch_size, seq_len, llm_contrast_sample_num + 1]
        
        kl_div = (kl_div.sum(dim = -1)).mean().item()
        
        # top-K acc.
        pred_which_real_event_by_mtpp_model = log_event_distribution_from_mtpp_model.argmax(dim = 0)
                                                                               # [batch_size, seq_len]
        pred_which_real_event_by_llm = log_probs_by_llm.argmax(dim = 0)        # [batch_size, seq_len]
        
        pred_which_real_event_by_mtpp_model, pred_which_real_event_by_llm, \
        log_event_distribution_from_mtpp_model, log_probs_by_llm = \
            move_from_tensor_to_ndarray(
                pred_which_real_event_by_mtpp_model, pred_which_real_event_by_llm, log_event_distribution_from_mtpp_model, log_probs_by_llm
            )
        
        # Then we calculate the accuracy of MTPP models.
        accuracy_by_mtpp = accuracy_score(y_pred = pred_which_real_event_by_mtpp_model.flatten(), y_true = np.zeros_like(pred_which_real_event_by_mtpp_model.flatten()))
        f1_by_mtpp = f1_score(y_pred = pred_which_real_event_by_mtpp_model.flatten(), \
                              y_true = np.zeros_like(pred_which_real_event_by_mtpp_model.flatten()), average = 'micro')
        top_k_acc_by_mtpp = []
        for k in range(1, opt.llm_sample_rate + 1):
            top_k_acc_by_mtpp.append(
                top_k_accuracy_score(y_true = np.zeros_like(pred_which_real_event_by_mtpp_model.flatten()),
                                     y_score = rearrange(log_event_distribution_from_mtpp_model, 'lcs b s -> (b s) lcs'),
                                     k = k,
                                     labels = np.arange(opt.llm_sample_rate + 1))
            )

        # Then we calculate the accuracy of the LLM.
        accuracy_by_llm = accuracy_score(y_pred = pred_which_real_event_by_llm.flatten(), y_true = np.zeros_like(pred_which_real_event_by_llm.flatten()))
        f1_by_llm = f1_score(y_pred = pred_which_real_event_by_llm.flatten(), \
                             y_true = np.zeros_like(pred_which_real_event_by_llm.flatten()), average = 'micro')
        top_k_acc_by_llm = []
        for k in range(1, opt.llm_sample_rate + 1):
            top_k_acc_by_llm.append(
                top_k_accuracy_score(y_true = np.zeros_like(pred_which_real_event_by_llm.flatten()),
                                     y_score = rearrange(log_probs_by_llm, 'lcs b s -> (b s) lcs'),
                                     k = k,
                                     labels = np.arange(opt.llm_sample_rate + 1))
            )

        return accuracy_by_mtpp, f1_by_mtpp, top_k_acc_by_mtpp, \
               accuracy_by_llm, f1_by_llm, top_k_acc_by_llm, kl_div


    @torch.inference_mode()
    def will_llm_assign_higher_probability_to_better_events(self, input_data, opt):
        '''
        This function is used to confirm that if the LLM will assign higher probability values to events closer to
        the real event
        
        The definition of an event closer to the real event is:
        1. At a given timestamp t, events with the correct mark is considered closer.
        2. At a given true mark m, events closer to the correct time is considered closer.
        
        For all marks, we will sample at the following timestamps (given the time of the real event is t, and MAE is \\Delta):
        t - 2 * \\Delta, t - \\Delta, t - 0.5 * \\Delta, t - 0.25 * \\Delta, t, t + 0.25 * \\Delta, t + 0.5 * \\Delta, t + \\Delta, t + 2 * \\Delta.
        
        You should declare the following arguments in your config file:
        1. ```int``` llm_sample_rate: This parameter controls how many samples are sampled from p(t).
        2. ```int``` llm_sample_step: This parameter controls how many samples are sampled in one shot from p(t).
        3. ```str``` llm_prompt: system prompt.
        
        # The following arguments are provided in llm.yml.
        4. ```str``` url_path: the url linked to the LLM service.
        5. ```str``` api_key: API key for user authentication.
        6. ```str``` llm_model: Name of the used LLM.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` maes
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using real marks.
            * ```float``` f1_2
              The f1 value shows the accuracy of the predicted marks.
            * ```np.ndarray``` probability_sum
              shape: ```[batch_size, seq_len]```
              The sum of calculated p(m) over all marks.
            * ```np.adarray``` p_m
              shape: ```[batch_size, seq_len, num_events]```
              The value of calculated p(m).
            * ```np.ndarray``` tau_pred_all_event
              shape: ```[batch_size, seq_len, num_events]```
              The predicted time for each mark using p(t|m).
            * ```np.ndarray``` time_next
              shape: ```[batch_size, seq_len]```
              Real time of observed events.
            * ```np.ndarray``` events_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed events.
        '''
        argument_check(opt, **{'sample_rate': int, 'mae_step': int, 'resolution': int, 'llm_request_mode': str})
        
        if opt.llm_request_mode == 'online':
            argument_check(opt, **{'llm_sample_rate': int, 'llm_sample_step': int, 'llm_visible_history_length': int, \
                                   'url_path': str, 'api_key': str, 'llm_model': str, 'llm_prompt': str, \
                                   'llm_event_template': str, 'llm_next_event_template': list})
            if not hasattr(self, 'url_path'):
                self.llm_request_mode = opt.llm_request_mode
                self.url_path = opt.url_path
                self.api_key = opt.api_key
                self.llm_model = opt.llm_model
                self.llm = CustomOpenAIforVLLM(self.url_path, self.llm_model, device = self.device, api_key = self.api_key)
                self.llm_prompt = self.llm.tokenize(opt.llm_prompt)
        else:
            argument_check(opt, **{'llm_sample_rate': int, 'llm_sample_step': int, 'llm_visible_history_length': int, \
                                   'llm_model': str, 'llm_prompt': str, 'llm_model_args': dict, 
                                   'llm_event_template': str, 'llm_next_event_template': list})
            if not hasattr(self, 'llm_model'):
                self.llm_request_mode = opt.llm_request_mode
                self.llm_model = opt.llm_model
                self.llm = VLLMOfflineInference(self.llm_model, device = self.device, model_args = opt.llm_model_args)
                # self.llm = VLLMOfflineInference(self.llm_model, device = self.device, model_args = {**opt.llm_model_args, "enforce_eager": True})
                self.llm_prompt = self.llm.tokenize(opt.llm_prompt)
        
        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, \
        mean, std = self.extract_plot_data(input_data)
        note_history, note_next = self.divide_history_and_next(input_notes)    # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        original_time_history, _ = self.divide_history_and_next(input_original_time)
                                                                               # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        note_embedding_history = None
        if self.mtpp_includes_note_embedding:
            note_embedding_history, _ = self.divide_history_and_next(input_note_embeddings)
                                                                               # [batch_size, seq_len, dim_note_embedding] * 2
        
        mae, f1_1, p_m = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                         time_next, mask_next, mean, std, opt, note_embedding_history)
                                                                               # [batch_size, seq_len]
        time_offset = torch.stack([
          -2 * mae, - mae, -0.5 * mae, -0.25 * mae, 0 * mae, 0.25 * mae, 0.5 * mae, mae, 2 * mae
        ] * self.num_events, axis = 0)                                         # [sample_num * num_events, batch_size, seq_len]
        mark_for_llm_probability_probe \
            = torch.tensor([mark for mark in range(self.num_events) for _ in range(9)], device = self.device).unsqueeze(dim = -1).unsqueeze(dim = -1) * torch.ones_like(events_next).unsqueeze(dim = 0)
                                                                               # [sample_num * num_events, batch_size, seq_len]
        time_for_llm_probability_probe = torch.clamp(time_next.unsqueeze(dim = 0) + time_offset, \
                                                     min = torch.tensor(0, device = self.device))
                                                                               # [sample_num * num_events, batch_size, seq_len]

        # During inference, the sparse_rate should be 0, so the sparse_mask can be ignored as all events are processed.
        log_probs_by_llm, _ = self.get_score_from_llm(note_history, note_next, events_history, time_history, original_time_history, \
                                                      time_for_llm_probability_probe, mark_for_llm_probability_probe, mask_next, \
                                                      llm_contrast_sample_num = 9 * self.num_events - 1, llm_visible_history_length = opt.llm_visible_history_length, \
                                                      llm_event_template = opt.llm_event_template, llm_next_event_template = opt.llm_next_event_template)
                                                                               # [sample_num * num_events, batch_size, seq_len]
        log_probs_by_llm = rearrange(log_probs_by_llm, '(ne sn) bs sl -> ne sn bs sl', ne = self.num_events)
                                                                               # [num_events, sample_num, batch_size, seq_len]
        log_probs_by_llm = move_from_tensor_to_ndarray(log_probs_by_llm)       # [num_events, sample_num, batch_size, seq_len]
        
        return log_probs_by_llm


    @torch.inference_mode()
    def probability_of_sampling_better_than_expectation(self, input_data, opt):
        '''
        This function is used to confirm that:
          LAMP thinks samples can be better than expectations. We chcek how lucky a person would be if he always gets samples
          better than the expectation.
        
        This function is placed here to verify the second point
        
        You should declare the following arguments in your config file:
        1. ```int``` llm_sample_rate: This parameter controls how many samples are sampled from p(t).
        2. ```int``` llm_sample_step: This parameter controls how many samples are sampled in one shot from p(t).
        3. ```str``` llm_prompt: system prompt.
        
        # The following arguments are provided in llm.yml.
        4. ```str``` url_path: the url linked to the LLM service.
        5. ```str``` api_key: API key for user authentication.
        6. ```str``` llm_model: Name of the used LLM.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` maes
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using real marks.
            * ```float``` f1_2
              The f1 value shows the accuracy of the predicted marks.
            * ```np.ndarray``` probability_sum
              shape: ```[batch_size, seq_len]```
              The sum of calculated p(m) over all marks.
            * ```np.adarray``` p_m
              shape: ```[batch_size, seq_len, num_events]```
              The value of calculated p(m).
            * ```np.ndarray``` tau_pred_all_event
              shape: ```[batch_size, seq_len, num_events]```
              The predicted time for each mark using p(t|m).
            * ```np.ndarray``` time_next
              shape: ```[batch_size, seq_len]```
              Real time of observed events.
            * ```np.ndarray``` events_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed events.
        '''
        argument_check(opt, **{'sample_rate': int, 'mae_step': int, 'resolution': int})
        
        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, \
        mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1, p_m = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                         time_next, mask_next, mean, std, opt = opt)
                                                                               # [batch_size, seq_len]
        
        time_next_lower_bound = torch.clamp(time_next - mae, min = 0.0)        # [batch_size, seq_len]
        time_next_upper_bound = time_next + mae                                # [batch_size, seq_len]
        
        expanded_integral_all_events_from_lower_to_upper_bound, \
        expanded_intensity_all_events_from_lower_to_upper_bound, timestamp = \
              self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_upper_bound, \
                                                         opt.resolution, time_next_start = time_next_lower_bound)
                                                                               # 2 * [batch_size, seq_len, resolution_inf, num_events]
        expanded_probability_all_events_from_lower_to_upper_bound = expanded_intensity_all_events_from_lower_to_upper_bound * torch.exp(-expanded_integral_all_events_from_lower_to_upper_bound.sum(dim = -1, keepdim = True))
                                                                               # 2 * [batch_size, seq_len, resolution_inf, num_events]
        probability_integral_all_events_from_lower_to_upper_bound \
            = approximate_integration(expanded_probability_all_events_from_lower_to_upper_bound, timestamp, dim = -2, only_integral = True)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot_mask = F.one_hot(events_next, num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        probability_sample_better_than_expectation \
            = (probability_integral_all_events_from_lower_to_upper_bound * events_next_one_hot_mask).sum(dim = -1)
                                                                               # [batch_size, seq_len]
                                                                               
        probability_sample_better_than_expectation = probability_sample_better_than_expectation[mask_next]
        probability_sample_better_than_expectation = move_from_tensor_to_list(probability_sample_better_than_expectation)
        
        return probability_sample_better_than_expectation


    @torch.inference_mode()
    def llm_lamp_inference(self, input_data, opt):
        '''
        This function is used to confirm that:
            1. the history note has sufficient info for deciding which next event makes more sense (Verified in NTPP).
            2. the LLM can use the history info in the note to decide which next event makes more sense (Verified with and without notes).
        
        This function is placed here to verify the second point
        
        You should declare the following arguments in your config file:
        1. ```int``` llm_sample_rate: This parameter controls how many samples are sampled from p(t).
        2. ```int``` llm_sample_step: This parameter controls how many samples are sampled in one shot from p(t).
        3. ```str``` llm_prompt: system prompt.
        
        # The following arguments are provided in llm.yml.
        4. ```str``` url_path: the url linked to the LLM service.
        5. ```str``` api_key: API key for user authentication.
        6. ```str``` llm_model: Name of the used LLM.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` maes
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using real marks.
            * ```float``` f1_2
              The f1 value shows the accuracy of the predicted marks.
            * ```np.ndarray``` probability_sum
              shape: ```[batch_size, seq_len]```
              The sum of calculated p(m) over all marks.
            * ```np.adarray``` p_m
              shape: ```[batch_size, seq_len, num_events]```
              The value of calculated p(m).
            * ```np.ndarray``` tau_pred_all_event
              shape: ```[batch_size, seq_len, num_events]```
              The predicted time for each mark using p(t|m).
            * ```np.ndarray``` time_next
              shape: ```[batch_size, seq_len]```
              Real time of observed events.
            * ```np.ndarray``` events_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed events.
        '''
        argument_check(opt, llm_request_mode = str)
        if opt.llm_request_mode == 'online':
            argument_check(opt, **{'llm_sample_rate': int, 'llm_sample_step': int, 'llm_visible_history_length': int, \
                                   'url_path': str, 'api_key': str, 'llm_model': str, 'llm_prompt': str, \
                                   'llm_event_template': str, 'llm_next_event_template': list})
            if not hasattr(self, 'url_path'):
                self.llm_request_mode = opt.llm_request_mode
                self.url_path = opt.url_path
                self.api_key = opt.api_key
                self.llm_model = opt.llm_model
                self.llm = CustomOpenAIforVLLM(self.url_path, self.llm_model, device = self.device, api_key = self.api_key)
                self.llm_prompt = self.llm.tokenize(opt.llm_prompt)
        else:
            argument_check(opt, **{'llm_sample_rate': int, 'llm_sample_step': int, 'llm_visible_history_length': int, \
                                   'llm_model': str, 'llm_prompt': str, 'llm_model_args': dict, 
                                   'llm_event_template': str, 'llm_next_event_template': list})
            if not hasattr(self, 'llm_model'):
                self.llm_request_mode = opt.llm_request_mode
                self.llm_model = opt.llm_model
                # self.llm = VLLMOfflineInference(self.llm_model, device = self.device, model_args = opt.llm_model_args)
                self.llm = VLLMOfflineInference(self.llm_model, device = self.device, model_args = {**opt.llm_model_args, "enforce_eager": True})

                self.llm_prompt = self.llm.tokenize(opt.llm_prompt)
        
        input_time, input_original_time, input_events, input_original_events, input_notes, input_note_embeddings, input_intensity, mask, \
        mean, std = self.extract_plot_data(input_data)
        note_history, note_next = self.divide_history_and_next(input_notes)    # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        original_time_history, _ = self.divide_history_and_next(input_original_time)
                                                                               # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        
        # step 1: get samples from the existing distribution.
        sampled_time = self.sample_time(sampling_approach = 'its', task = 'tm',
                                        events_history = events_history, time_history = time_history,
                                        number_of_total_samples = opt.llm_sample_rate, 
                                        step = opt.llm_sample_step, mean = mean, std = std)
                                                                               # [llm_contrast_sample_num, batch_size, seq_len]

        intensity_integral_all_events, intensity_all_events \
            = self.model(time_history, sampled_time, events_history, num_dimension_prior_batch = 1)
                                                                               # [llm_contrast_sample_num, batch_size, seq_len, num_events]
        mark_distribution = intensity_all_events / intensity_all_events.sum(dim = -1, keepdim = True)
                                                                               # [llm_contrast_sample_num, batch_size, seq_len, num_events]
        sampled_events = predict_event(mark_distribution, sample = True)       # [llm_contrast_sample_num, batch_size, seq_len]
        # merge the real next event with the pred_time
        # [real_pred_time, (sampled_times)]
        sampled_time, _ = pack((time_next, sampled_time), '* b s')             # [llm_contrast_sample_num + 1, batch_size, seq_len]
        sampled_events, _ = pack((events_next, sampled_events), '* b s')       # [llm_contrast_sample_num + 1, batch_size, seq_len]
        
        intensity_integral, intensity = self.model(time_history, time_next, events_history, num_dimension_prior_batch = 0)
                                                                               # [batch_size, seq_len, num_events]
        all_intensity, _ = pack((intensity, intensity_all_events), '* b s ne') # [llm_contrast_sample_num + 1, batch_size, seq_len, num_events]
        all_intensity_integral, _ = pack((intensity_integral, intensity_integral_all_events), '* b s ne')
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len, num_events]
        distribution_all = all_intensity * torch.exp(-all_intensity_integral)  # [llm_contrast_sample_num + 1, batch_size, seq_len, num_events]
        sampled_events_one_hot = F.one_hot(sampled_events, num_classes = self.num_events)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len, num_events]
        selected_distribution = (distribution_all * sampled_events_one_hot).sum(dim = -1)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len]
        log_event_distribution_from_mtpp_model = F.log_softmax(selected_distribution, dim = 0)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len]
        log_probs_by_llm = self.get_score_from_llm(note_history, note_next, events_history, time_history, original_time_history, \
                                                   sampled_time, sampled_events, mask_next, \
                                                   llm_contrast_sample_num = opt.llm_sample_rate, llm_visible_history_length = opt.llm_visible_history_length, \
                                                   llm_event_template = opt.llm_event_template, llm_next_event_template = opt.llm_next_event_template)
                                                                               # [llm_contrast_sample_num + 1, batch_size, seq_len]
        
        pred_which_real_event_by_mtpp_model = log_event_distribution_from_mtpp_model.argmax(dim = 0)
                                                                               # [batch_size, seq_len]
        pred_which_real_event_by_llm = log_probs_by_llm.argmax(dim = 0)        # [batch_size, seq_len]
        
        pred_which_real_event_by_mtpp_model, pred_which_real_event_by_llm, \
        log_event_distribution_from_mtpp_model, log_probs_by_llm = \
            move_from_tensor_to_ndarray(
                pred_which_real_event_by_mtpp_model, pred_which_real_event_by_llm, log_event_distribution_from_mtpp_model, log_probs_by_llm
            )
        
        # Then we calculate the accuracy of MTPP models.
        accuracy_by_mtpp = accuracy_score(y_pred = pred_which_real_event_by_mtpp_model.flatten(), y_true = np.zeros_like(pred_which_real_event_by_mtpp_model.flatten()))
        f1_by_mtpp = f1_score(y_pred = pred_which_real_event_by_mtpp_model.flatten(), \
                              y_true = np.zeros_like(pred_which_real_event_by_mtpp_model.flatten()), average = 'micro')
        top_k_acc_by_mtpp = []
        for k in range(1, opt.llm_sample_rate + 1):
            top_k_acc_by_mtpp.append(
                top_k_accuracy_score(y_true = np.zeros_like(pred_which_real_event_by_mtpp_model.flatten()),
                                     y_score = rearrange(log_event_distribution_from_mtpp_model, 'lcs b s -> (b s) lcs'),
                                     k = k,
                                     labels = np.arange(opt.llm_sample_rate + 1)).item()
            )

        # Then we calculate the accuracy of the LLM.
        accuracy_by_llm = accuracy_score(y_pred = pred_which_real_event_by_llm.flatten(), y_true = np.zeros_like(pred_which_real_event_by_llm.flatten()))
        f1_by_llm = f1_score(y_pred = pred_which_real_event_by_llm.flatten(), \
                             y_true = np.zeros_like(pred_which_real_event_by_llm.flatten()), average = 'micro')
        top_k_acc_by_llm = []
        for k in range(1, opt.llm_sample_rate + 1):
            top_k_acc_by_llm.append(
                top_k_accuracy_score(y_true = np.zeros_like(pred_which_real_event_by_llm.flatten()),
                                     y_score = rearrange(log_probs_by_llm, 'lcs b s -> (b s) lcs'),
                                     k = k,
                                     labels = np.arange(opt.llm_sample_rate + 1)).item()
            )

        return accuracy_by_mtpp, f1_by_mtpp, top_k_acc_by_mtpp, \
               accuracy_by_llm, f1_by_llm, top_k_acc_by_llm


    def convert_missing_mask_to_gap_mask(self, missing_mask):
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


    def cppod_evaluation(self, input_data, opt):
        '''
        Take care. This function only evaluates the omission outlier.
        Interestingly, the original CPPOD code seems only focusing on omission too as only omission scores are recorded in model.detect_outlier().
        Paired with the od_genetic dataloader.
        '''
        forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, (mean, std) \
            = input_data
        
        roc_result = []
        for obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, missing_mask_for_one_seq, _ in padded_obs_data:
            obs_time_history_for_one_seq, obs_time_next_for_one_seq = self.divide_history_and_next(obs_time_for_one_seq)
                                                                               # [batch_size, seq_len] * 2
            obs_events_history_for_one_seq, obs_events_next_for_one_seq = self.divide_history_and_next(obs_events_for_one_seq)
                                                                               # [batch_size, seq_len] * 2
            obs_mask_history_for_one_seq, obs_mask_next_for_one_seq = self.divide_history_and_next(obs_mask_for_one_seq)
                                                                               # [batch_size, seq_len]
            
            missing_mask_for_one_seq = self.convert_missing_mask_to_gap_mask(missing_mask_for_one_seq)
                                                                               # [num_samples, ...]
            integral_all_events, intensity_all_events \
                = self.model(obs_time_history_for_one_seq.float(), obs_time_next_for_one_seq.float(), obs_events_history_for_one_seq)
                                                                               # [num_samples, seq_len, num_events]
            
            integral_sum = integral_all_events.sum(dim = -1)                   # [num_samples, seq_len]
            intensity_sum = intensity_all_events.sum(dim = -1)                 # [num_samples, seq_len]
            
            all_roauc_area = []
            for integral_sum_per_seq_per_sample, missing_mask_for_one_seq_per_sample in \
                zip(integral_sum, missing_mask_for_one_seq):
                
                sample_len = len(missing_mask_for_one_seq_per_sample)
                selected_integral_sum_per_seq_per_sample = move_from_tensor_to_ndarray(integral_sum_per_seq_per_sample[:sample_len])
                
                roauc_area = roc_auc_score(y_true = np.array(missing_mask_for_one_seq_per_sample) ^ 1, y_score = selected_integral_sum_per_seq_per_sample)
                all_roauc_area.append(roauc_area)
            
            roc_result.append(np.mean(all_roauc_area))
        
        roc_result = np.array(roc_result)
        return roc_result


    def cppod_commission_evaluation(self, input_data, opt):
        (time_seq, events, commission, mask), (mean, std) = input_data
        
        time_history, time_next = self.divide_history_and_next(time_seq)       # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len]
        _, commission_next = self.divide_history_and_next(commission)          # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        time_history = time_history.float()
        time_next = time_next.float()
        
        _, intensity_all_events = self.model(time_history, time_next, events_history)
                                                                               # 2 * [batch_size, seq_len, num_events]
        
        intensity_sum_from_tl_to_time_next = intensity_all_events.sum(dim = -1)
                                                                               # [batch_size, seq_len]
        score = -intensity_sum_from_tl_to_time_next                            # [batch_size, seq_len]
        
        packed_data = zip(score, commission_next, mask_next)
        all_roauc_area = []

        for score_per_seq, commission_next_per_seq, mask_next_per_seq in packed_data:
            available_score = score_per_seq[mask_next_per_seq]
            available_commission_label = commission_next_per_seq[mask_next_per_seq]
            
            available_score, available_commission_label = move_from_tensor_to_ndarray(available_score, available_commission_label)
            roauc_area = roc_auc_score(y_true = available_commission_label, y_score = available_score)
            all_roauc_area.append(roauc_area)
        
        all_roauc_area = np.array(all_roauc_area)
        return all_roauc_area


    def train_step(model, minibatch, device, step):
        '''
        This function unpacks the minibatch, calls the train_procedure() to calculate the loss, and do the backpropagation.

        ### Args
            * ```torch.nn.Module``` model
              The MTPP model that we train.
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```torch.device``` device
              where we train the model.

        ### Outputs:
            * ```float``` time_loss_without_dummy
              The average NLL loss without dummy events, specifically the start and the end event.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` events_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
              The training loss does not and should not include this value.
        '''
        model.train()

        # Maybe need another function to extract data from minibatches.
        # For now, we don't acquire any prediction loss to assist the model training.  
        (time, original_time_seq, events, original_events, note, note_embedding, score, mask), (mean, std) = minibatch
                                                                               # 6 * [batch_size, seq_len + 1] + two floats
        
        # loss, log_likeli_loss_without_dummy, mse_loss, marker_loss_without_dummy, the_number_of_events
        loss, time_loss_without_dummy, mse_loss, events_loss, the_number_of_events \
            = model('train', time, original_time_seq, events, original_events, note, note_embedding, mask, \
                    mean = mean, std = std, step = step)

        loss.backward()
    
        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        mse_loss = mse_loss.item() / the_number_of_events
        
        return time_loss_without_dummy, fact, mse_loss, events_loss
    

    def evaluation_step(model, minibatch, device):
        '''
        This function unpacks the minibatch, calls the evaluation_procedure() to calculate the metrics.

        ### Args
            * ```torch.nn.Module``` model
              The MTPP model that we train.
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```torch.device``` device
              where we train the model.

        ### Outputs:
            * ```float``` time_loss
              The average NLL loss without dummy events, specifically the start and the end event.
            * ```float``` loss_survival
              The average NLL loss of the end event, which is the integral of the intensity function from the last occurred event to the end time.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` events_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
        '''
        model.eval()
        
        (time, original_time_seq, events, original_events, note, note_embedding, score, mask), (mean, std) = minibatch
                                                                               # 6 * [batch_size, seq_len + 1] + two floats
        time_loss, loss_survival, events_loss, mse_loss, mae, f1, the_number_of_events \
            = model('evaluate', time, original_time_seq, events, original_events, note, note_embedding, mask, mean = mean, std = std)

        time_loss = time_loss.item() / the_number_of_events
        loss_survival = loss_survival.item()
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        mse_loss = mse_loss.item() / the_number_of_events
        
        return time_loss, loss_survival, fact, events_loss, mse_loss, mae, f1


    def postprocess(input, procedure):
        '''
        This function makes some modifications to the output of training_step() and evaluation_step().

        ### Args
            * ```list``` input
              The output of either training_step() or evaluation_step().
            * ```str``` procedure
              This string tells the function which function the input comes from.

        ### Outputs:
            * ```list```
              The postprocessed outputs.
        '''
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss, events loss]
            '''
            return [input[0], input[0] - input[1], input[2], input[3]]
        
        def test_postprocess(input):
            '''
            Evaluation process
            [absolute loss, relative loss, events loss, mae value]
            time_loss, loss_survival, fact, events_loss, mse_loss, mae, f1
            '''
            return [input[0], input[1], input[0] - input[2], input[3], input[4], input[5], input[6]]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    
    '''
    The maximum length of the format_dict in different procedures.
    '''
    format_dict_length = 7
    
    
    def log_print_format(input, procedure):
        '''
        This function packs the procedure input into a dict that can be handled by trainer and evaluator for logging.

        ### Args
            * ```list``` input
              The output of either training_step() or evaluation_step().
            * ```str``` procedure
              This string tells the function which function the input comes from.

        ### Outputs:
            * ```dict``` format_dict
              format: {..., <variable name>: {'data': <value>, 'num_format': <num_format>, 'suffix': <suffix>}, ...}
              example: {..., 'memory': {'data': 12.123456, 'num_format': ':2.4f', 'suffix': 'GiB'}, ...}
              The formated results.
        '''
        def train_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = pack_one_value_to_dict(input[0])
            format_dict['relative_loss'] = pack_one_value_to_dict(input[1])
            format_dict['mse_loss'] = pack_one_value_to_dict(input[2])
            format_dict['events_loss'] = pack_one_value_to_dict(input[3])
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['absolute_NLL_loss'] = pack_one_value_to_dict(input[0])
            format_dict['avg_survival_loss'] = pack_one_value_to_dict(input[1])
            format_dict['relative_NLL_loss'] = pack_one_value_to_dict(input[2])
            format_dict['events_loss'] = pack_one_value_to_dict(input[3])
            format_dict['note_pred_mse_loss'] = pack_one_value_to_dict(input[4])
            format_dict['mae'] = pack_one_value_to_dict(input[5], '2.8f')
            format_dict['f1'] = pack_one_value_to_dict(input[6])
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))


    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        This function helps the trainer to pick the best checkpoint based on several metrics.

        ### Args
            * ```dict``` evaluation_report_format_dict
            * ```dict``` test_report_format_dict
              The formated output of training_step() and evaluation_step().

        ### Outputs:
            * ```list```
              The picked metrics used for model select.
            * ```list```
              The name of these metrics.
        '''
        return [evaluation_report_format_dict['mae'], evaluation_report_format_dict['f1']], \
               ['evaluation_MAE', 'evaluation_f1']
    
    '''
    metric number is the length of the output of choose_metric
    '''
    metric_number = 2
    
    '''
    True: The lower the metric is, the better the model is.
    False: The higher the metric is, the better the model is.
    '''
    smaller_is_better = [True, False]