import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from src.fakenews.model.heard.ctlstm import CTLSTMCell
from src.fakenews.model.heard.rdlstm import RDLSTMCell


class HEARD(nn.Module):
    def __init__(self, num_events, device, batch_size, RD_d_hidden, HC_d_hidden, RD_d_input, RD_num_layer, 
                 loss_weight_HC, loss_weight_early_pred, less_weight_shift_count, fcn_dropout, \
                 lstm_dropout):
        super(HEARD, self).__init__()
        self.num_events = num_events
        self.device = device
        self.batch_size = batch_size
        self.RD_d_hidden = RD_d_hidden
        self.HC_d_hidden = HC_d_hidden
        self.RD_d_input = RD_d_input
        self.HC_d_input = 1
        self.resolution = 100

        self.HCLSTM = CTLSTMCell(self.HC_d_hidden, self.batch_size, lstm_dropout, self.device)

        self.EmbeddingRD = nn.Linear(self.RD_d_input, self.RD_d_hidden, bias = False, device = self.device)
        self.EmbeddingHC = nn.Linear(self.HC_d_input, self.HC_d_hidden, device = self.device)
        # nn.init.orthogonal_(self.EmbeddingRD.weight)
        # nn.init.orthogonal_(self.EmbeddingHC.weight)

        self.fcRD = nn.Linear(self.RD_d_hidden, 2, device = self.device)
        self.fcHC = nn.Linear(self.HC_d_hidden, 1, device = self.device)
        self.fcN = nn.Linear(self.HC_d_hidden, 2, device = self.device)
        # nn.init.orthogonal_(self.fcHC.weight)
        # nn.init.orthogonal_(self.fcRD.weight)
        # nn.init.orthogonal_(self.fcN.weight)

        self.dropout = nn.Dropout(fcn_dropout)
        self.sigmoid = nn.Sigmoid()

        self.loss_weight_HC = torch.tensor(loss_weight_HC, device = self.device)
        self.loss_weight_early_pred = torch.tensor(loss_weight_early_pred, device = self.device)
        self.less_weight_shift_count = torch.tensor(less_weight_shift_count, device = self.device)
        
        self.layers = nn.ModuleList()
        for _ in range(RD_num_layer):
            self.layers.append(RDLSTMCell(self.RD_d_hidden, self.batch_size, lstm_dropout, self.device))


    def forward(self, data, if_dp):
        
        h_d_RDs, c_d_RDs = [], []

        # all variables ending with "RDs" refers to the veracity checking model.
        for layer in self.layers:
            layer.init_states()
            h_d_RDs.append(layer.h_d)
            c_d_RDs.append(layer.c_d)
        
        # HCLSTM: counts the number of veracity shifts. Similar to the role of the "RD" suffix, "HC" means the variable are CTLSTM related.
        self.HCLSTM.init_states()

        h_d_HC,c_d_HC,c_bar_HC = self.HCLSTM.h_d,self.HCLSTM.c_d,self.HCLSTM.c_bar
        num_reverse = torch.zeros(self.batch_size,1,dtype=torch.long).to(self.device)

        label_seqs,text_seqs_tensor, \
        time_seqs_tensor, seqs_length,_,_,posts_length,max_post_len,real_lens,post_indexes = self.__data_adapt__(data,if_dp)

        batch_length = time_seqs_tensor.size()[1]
        seqs_length = seqs_length.unsqueeze(1)

        h_list_RD, c_list_RD, pred_list_RD,prob_list_RD,h_list_RD_tmp = [], [], [], [], []
        h_list_HC, c_list_HC, c_bar_list_HC, o_list_HC, delta_list_HC, reverse_list_HC, if_reverse_HC = [], [], [], [], [], [], []

        for t in range(batch_length):                                          # loop pick every embeddings from index 0 to max_seq_len - 1

            text_input = text_seqs_tensor[:,t,:]                               # [batch_size, d_input]
            text_seq_emb = self.EmbeddingRD(text_input)

            h_d_RD =  text_seq_emb
            for li,layer in enumerate(self.layers):
                h_d_RD,(_,c_d_RD) = layer(h_d_RD,h_d_RDs[li],c_d_RDs[li])      # [batch_size, d_hidden]
                h_d_RDs[li],c_d_RDs[li] = h_d_RD,c_d_RD                        # [batch_size, d_hidden] * seq_len

            if if_dp:
                pred_RD = self.fcRD(self.dropout(h_d_RD))                      # [batch_size, 2]
            else:
                pred_RD = self.fcRD(h_d_RD)
            pred_RD_prob = F.softmax(pred_RD, dim=1)
            label_RD = torch.argmax(pred_RD_prob, dim=1)
            h_list_RD.append(h_d_RD)                                           # store the hidden state
            c_list_RD.append(c_d_RD)                                           # store the memory state
            prob_list_RD.append(pred_RD)                                       # store the temporary veracity checking result

            if t == 0:
                if_reverse_t = torch.zeros_like(label_RD).long().unsqueeze(1)  # no reverse as we just start checking
            else:
                if_reverse_t = (label_RD != pred_list_RD[-1]).long().unsqueeze(1)
                                                                               # reverse counting
            pred_list_RD.append(label_RD)
            num_reverse = num_reverse + if_reverse_t
           
            reverse_list_HC.append(num_reverse)                                # prepare for the input of the CTLSTM for veracity shift counting.
            if_reverse_HC.append(if_reverse_t)                                 # also: have the markers been shifted after the current post?
            num_reverse_n = num_reverse * 1.0
            
            reverse_seq_emb = self.EmbeddingHC(num_reverse_n)
            reverse_seq_emb = reverse_seq_emb.squeeze(1)
            c, c_bar_HC, delta_t,o_t = self.HCLSTM(reverse_seq_emb,h_d_HC,c_d_HC,c_bar_HC)
            c_d_HC, h_d_HC = self.HCLSTM.decay(c, c_bar_HC, delta_t, o_t, time_seqs_tensor[:,t])
                                                                               # the hidden state
            h_list_HC.append(c_d_HC)
            c_list_HC.append(c)
            c_bar_list_HC.append(c_bar_HC)
            o_list_HC.append(o_t)
            delta_list_HC.append(delta_t)

        h_seq_HC,c_seq_HC,c_bar_seq_HC,o_seq_HC,delta_seq_HC,reverse_seq_HC,if_reverce_seq_HC\
             = torch.stack(h_list_HC),torch.stack(c_list_HC),torch.stack(c_bar_list_HC),\
                 torch.stack(o_list_HC),torch.stack(delta_list_HC),torch.stack(reverse_list_HC),torch.stack(if_reverse_HC)

        h_seq_RD, c_seq_RD, pred_seq_RD,prob_seq_RD = torch.stack(h_list_RD),torch.stack(c_list_RD),torch.stack(pred_list_RD),torch.stack(prob_list_RD)

        # (stacked hidden state, stacked memory state, stacked prediction sequence(seq_len, batch_size), stacked prediction probability sequence(seq_len, batch_size, 2))
        self.output_RD = (h_seq_RD, c_seq_RD, pred_seq_RD,prob_seq_RD)
        # (stacked CTLSTM hidden state, stacked CTLSTM memory state, stacked CTLSTM base memory state, ..., the cumulative number of veracity shifts, whether the veracity shifts after a given merged tweet.)
        self.output_HC = (h_seq_HC,c_seq_HC,c_bar_seq_HC,o_seq_HC,delta_seq_HC,reverse_seq_HC,if_reverce_seq_HC)

        return self.output_RD,self.output_HC,label_seqs


    def __data_adapt__(self, data,if_dp):
        label_seqs,target_seqs,\
        text_seqs_tensor,\
        post_since_start_seqs_tensor,time_seqs_tensor,\
        last_time_seqs, seqs_length,timstamp_seqs_tensor,posts_length,max_post_len,real_lens,_,_ = data
        
        if not if_dp:
            label_seqs, seqs_length, real_lens= label_seqs.repeat(self.batch_size,),seqs_length.repeat(self.batch_size,),real_lens.repeat(self.batch_size,)
            post_since_start_seqs_tensor = post_since_start_seqs_tensor.repeat(self.batch_size,1)
            if max_post_len:
                text_seqs_tensor = text_seqs_tensor.repeat(self.batch_size,1,1,1)
                posts_length = posts_length.repeat(self.batch_size,1)
            else:
                text_seqs_tensor = text_seqs_tensor.repeat(self.batch_size,1,1)

            time_seqs_tensor, timstamp_seqs_tensor,target_seqs = \
                time_seqs_tensor.repeat(self.batch_size,1),\
                timstamp_seqs_tensor.repeat(self.batch_size,1),target_seqs.repeat(self.batch_size,1)
                
        label_seqs,text_seqs_tensor, time_seqs_tensor,seqs_length,timstamp_seqs_tensor,target_seqs, real_lens,post_since_start_seqs_tensor = \
        label_seqs.to(self.device),text_seqs_tensor.to(self.device),time_seqs_tensor.to(self.device),\
        seqs_length.to(self.device),timstamp_seqs_tensor.to(self.device),target_seqs.to(self.device), real_lens.to(self.device),post_since_start_seqs_tensor.to(self.device)

        return label_seqs,text_seqs_tensor, time_seqs_tensor, seqs_length,timstamp_seqs_tensor,target_seqs,posts_length,max_post_len,real_lens,post_since_start_seqs_tensor


    def __label_convert__(self,label_seqs):
        targets_seqs = torch.zeros((self.batch_size,2),dtype = torch.float)
        for idx,target in enumerate(label_seqs):
            if target.item() == 1:
                targets_seqs[idx,:] = torch.FloatTensor([0,1])
            else:
                targets_seqs[idx,:] = torch.FloatTensor([1,0])
        targets_seqs = targets_seqs.to(self.device)
        return targets_seqs


    def compute_lambda(self,h_seq_HC):

        lambda_k  = F.softplus(self.fcHC(h_seq_HC.transpose(0, 1)))
        return lambda_k


    def compute_sim_lambda(self,timestamps,gates_states):

        c, c_bar_HC, o_t, delta_t= gates_states
        if not timestamps:
            time_diffs = torch.FloatTensor(np.float32(np.array(sorted(
                                np.random.exponential(
                                    scale=1.0,
                                    size=(self.resolution,)))))).to(self.device)
        else:
            diff_time = (timestamps[1] - timestamps[0])
            sample_time = diff_time * \
                        torch.rand([self.resolution], device=self.device).squeeze(0)
            interval_left_shift = (timestamps[0] + 1)
            time_diffs = sample_time / interval_left_shift
        c_p_HC, h_p_HC = self.HCLSTM.decay(c, c_bar_HC, delta_t, o_t, time_diffs,True)
        lambda_pred  = F.softplus(self.fcHC(h_p_HC.transpose(1, -1))).transpose(1, -1).squeeze(0)
        return lambda_pred,time_diffs


    def coumpute_loss_step(self,idx,step,f_step,seq_len,label_seqs,final_N,prob_seq_RD,
                            timstamp_seqs_tensor,
                            N_i,h_step,gates_states):

        c_seq_HC,c_bar_seq_HC,o_seq_HC,delta_seq_HC = gates_states
        c_seq_HC,c_bar_seq_HC,o_seq_HC,delta_seq_HC = c_seq_HC.transpose(0,1),c_bar_seq_HC.transpose(0,1),o_seq_HC.transpose(0,1),delta_seq_HC.transpose(0,1)

        step_loss,term1,term2,term3,term4,term5,log_likelihood_HC= [0.0]*7
        delta_N = np.Inf
        target_RD = label_seqs[idx]
        output_RD = prob_seq_RD[idx,step]

        if step == 0:
            term3 = F.binary_cross_entropy_with_logits(output_RD,target_RD)
        else:
            gates_states_step = (c_seq_HC[idx,step,:],c_bar_seq_HC[idx,step,:],o_seq_HC[idx,step,:],delta_seq_HC[idx,step,:])
            timestamps = timstamp_seqs_tensor[idx, :step+1]

            lambda_inf_step,time_diffs = self.compute_sim_lambda(None,gates_states_step)

            cum_num = (torch.arange(time_diffs.size()[0]+1)[1:]*1.0).to(self.device)
            time_density_term2 = torch.exp((-1.0*torch.cumsum(lambda_inf_step,dim=1) / cum_num[None, :])*time_diffs[None,:])
            time_density = time_density_term2 * lambda_inf_step

            time_pred_step = torch.mean(time_diffs[None, :]*time_density,dim=1)*time_diffs[-1]
            time_true = timestamps[step].to(self.device)
            term1 = torch.sqrt((time_true - time_pred_step).abs()**2)          # the last term of eqn. 7. It seems that time_true misses the time unit conversion.

            delta_N_i = self.fcN(h_step)
            target_N_i = self.__label_convert__(N_i)[0:1,:]
            term2 = F.binary_cross_entropy_with_logits(delta_N_i,target_N_i)   # the first term of eqn. 7. BEARD reqires the CTLSTM to correctly judge whether the veracity would shift when the next post is present.

            term3 = F.binary_cross_entropy_with_logits(output_RD,target_RD)    # L_{i}^{r}.
            
            term4 = -torch.log(1.0-(f_step)/(1.0*seq_len))                     # the last term of eqn. 6.

            delta_N = torch.sum(lambda_inf_step,dim=1)*time_diffs[-1] / self.resolution
            term5 = torch.sqrt((delta_N - final_N).abs()**2)                   # the first term of eqn. 6


        step_loss = self.loss_weight_HC*(term1 + term2) + term3 + self.loss_weight_early_pred*term4 + self.less_weight_shift_count*term5
        return step_loss,delta_N,term1,term2,term3,term4,term5


    def compute_log_likelihood(self,Batch_data,if_dp):

        label_seqs,_,_, seqs_length,timstamp_seqs_tensor,_,posts_length,max_post_len,real_lens,post_indexes = self.__data_adapt__(Batch_data,if_dp)
        target_seqs = self.__label_convert__(label_seqs)
        h_seq_HC,c_seq_HC,c_bar_seq_HC,o_seq_HC,delta_seq_HC,reverse_seq_HC,if_reverse_seq_HC = self.output_HC
        prob_seq_RD = self.output_RD[3].transpose(0,1)
        reverse_seq_HC = reverse_seq_HC.transpose(0,1)
        if_reverse_seq_HC = if_reverse_seq_HC.transpose(0,1)
        h_seq_HC = h_seq_HC.transpose(0,1)
        delta_N_at_stops = []

        stop_points = torch.LongTensor([-1]*seqs_length.size()[0]).to(self.device)
        batch_loss = torch.zeros(seqs_length.size()[0]).to(self.device)
        for idx, seq_len in enumerate(seqs_length):
            seq_loss = []
            seq_delta = []
            delta_N_at_stop = None
            term1s,term2s,term3s,term4s,term5s = [],[],[],[],[]
            stop_points[idx] = seq_len-1
            for step in range(seq_len):
                if step == seq_len-1 and seq_len>1:
                    seq_loss.append(100.0)
                    break
                h_step = h_seq_HC[idx,step,:].unsqueeze(0)

                final_N = (reverse_seq_HC[idx,-1]-reverse_seq_HC[idx,step]).to(self.device)
                if seq_len>1:
                    N_i = (reverse_seq_HC[idx,step+1]-reverse_seq_HC[idx,step]).to(self.device)
                else:
                    N_i = None
                f_seq_len = real_lens[idx]
                f_step = post_indexes.cpu().data[idx,step]*1.0
                loss_step,delta_N,term1,term2,term3,term4,term5 = self.coumpute_loss_step(idx,step,f_step,f_seq_len,target_seqs,final_N,
                                                prob_seq_RD,timstamp_seqs_tensor,N_i,h_step,
                                                (c_seq_HC,c_bar_seq_HC,o_seq_HC,delta_seq_HC))
                seq_delta.append(delta_N)
                seq_loss.append(loss_step)
                term1s.append(term1)
                term2s.append(term2)
                term3s.append(term3)
                term4s.append(term4)
                term5s.append(term5)
                if (delta_N < 1.0):
                    stop_points[idx] = step
                    seq_loss.pop()
                    delta_N_at_stop = 0
                    break

            if delta_N_at_stop != None:
                delta_N_at_stops.append(delta_N_at_stop)
            else:
                delta_N_at_stops.append(1)
            batch_loss[idx] = (sum(seq_loss) / len(seq_loss))+term3s[-1]

        loss_batch = torch.mean(batch_loss)

        stop_preds = torch.LongTensor([-1]*seqs_length.size()[0]).to(self.device)

        for idx, stop_point in enumerate(stop_points):
            if stop_point.item() == -1:
                stop_points[idx] = seqs_length[idx]-1
            prob_stop_RD = prob_seq_RD[idx,stop_points[idx]]
            # prob_stop_RD = prob_seq_RD[idx, 0]
            pred = torch.argmax(self.sigmoid(prob_stop_RD))
            stop_preds[idx] = pred
        

        return loss_batch,stop_points,stop_preds,delta_N_at_stops,seqs_length