import torch
import torch.nn as nn
import torch.nn.functional as F

from .cont_time_cell import CTLSTMCell

class CTLSTM(nn.Module):
    def __init__(self, hidden_dim, event_num, beta, mc_sample_num, device):
        super(CTLSTM, self).__init__()

        # In current dataset, there is no mark information. Currently, this CTLSTM implementation has
        # mark processing but doesn't used.

        # Initial
        # These three vectors refer to initial hidden state(h(t)), initial memory state(c(t)) and initial memory base state(\bar(c)(t)).
        self.hidden_dim = hidden_dim
        self.device = device
        self.beta = beta
        self.mc_sample_num = mc_sample_num
        self.mc_sample_num_eval = max(1.0, mc_sample_num)
        self.eps = torch.finfo(float).eps
    
        self.h_0 = torch.zeros(self.hidden_dim, device = self.device)
        self.c_0 = torch.zeros(self.hidden_dim, device = self.device)
        self.cb_0 = torch.zeros(self.hidden_dim, device = self.device)

        # Intensity part
        self.rnn_cell = CTLSTMCell(hidden_dim, device = self.device)

        # Mark part
        # The definition of a event sequences in CTLSTM:
        # <BOS> <> <> <> <> ... <> <> <EOS>
        # The pad event here is confusing.
        self.event_num = event_num
        self.idx_BOS = self.event_num
        self.idx_EOS = self.event_num + 1
        self.idx_PAD = self.event_num + 2
        self.event_range = torch.arange(0, event_num, device = self.device)

        self.mark = nn.Embedding(self.event_num + 3, self.hidden_dim).to(self.device)
        self.intensity = nn.Linear(self.hidden_dim, 1, bias = False).to(self.device)
    
    def forward(self, event_tensor, dtime_tensor, token_num_tensor, duration_tensor, eval_tag=False): 
        """
		input
			model [object of GNHP] : p_theta or simply p 
			event_tensor [B x T+2] : a batch of event types with length T+2
			dtime_tensor [B x T+2] : a batch of dtimes with length T+2
			token_num_tensor [B] : # of tokens per seq 
			duration_tensor [B] : duration per seq
		"""
        target_tensor, mask_tensor = self.get_target(event_tensor)
		# B x T+1 , starting from 1st actual event 
        all_c_p_actual, all_cb_p_actual, all_d_p_actual, all_o_p_actual, _, _ = \
        self.get_cells_gates_states(event_tensor, dtime_tensor)
		# B x T+1 x D 
        inten_p_actual = self.get_intensities_given_types(
			all_c_p_actual, all_cb_p_actual, all_d_p_actual, all_o_p_actual, 
			target_tensor.unsqueeze(-1), dtime_tensor[:, 1:]
		) # B x (T + 1) x 1
        """
		NOTE : we sometimes use very small mc sample to train for speed-up
		but we need >= 1 for stable eval
		"""
        mc_sample_num = self.mc_sample_num_eval if eval_tag else self.mc_sample_num
        mc_sample_num_tensor = (token_num_tensor.float() * mc_sample_num).long()
        all_c_p_noise, all_cb_p_noise, all_d_p_noise, all_o_p_noise, \
        all_dtime_noise, all_mask_noise = self.get_mc_samples(
			all_c_p_actual, all_cb_p_actual, 
			all_d_p_actual, all_o_p_actual, 
			dtime_tensor[:, 1:], mc_sample_num_tensor, 
			duration_tensor, mask_tensor
		)
		# B x T' (x D)
        inten_p_noise = self.get_intensities_all_types(
			all_c_p_noise, all_cb_p_noise, all_d_p_noise, all_o_p_noise, 
			all_dtime_noise
		)
		# B x T' x K
        log_inten = torch.log(inten_p_actual.sum(-1) + self.eps) * mask_tensor
		# B x (T + 1)
        integral = torch.sum(inten_p_noise, dim=-1) * all_mask_noise
		# B x T'
        actual_counts = torch.sum(all_mask_noise, dim=-1) # B 
        integral = torch.sum(integral, dim=-1) / actual_counts
        integral = duration_tensor * integral # B
        log_likelihood = torch.sum(log_inten) - torch.sum(integral)

        return log_likelihood
        
    def get_target(self, event_tensor): 
        """
		make target variables and masks 
		i.e., set >= event_num to 0, also mask them out 
		"""
        batch_size, T_plus_2 = event_tensor.size()
        mask = torch.ones((batch_size, T_plus_2-1), dtype=torch.float32, device=self.device)
        target_data = event_tensor[:, 1:].clone().detach()
        mask[target_data >= self.event_num] = 0.0
        target_data[target_data >= self.event_num] = 0 # PAD to be 0
        return target_data, mask
    
    def get_cells_gates_states(self, x_event, x_dtime):
        '''
        The expected one sequence input:
        <> <> <> <> ... <> <>
        What our model should do is to add special events into the original sequences when needed.
        <EOS> and <PAD> doesn't update the CTLSTM.
        Additionally, x_event only has identical elements when mark information is not present.
        '''
        batch_size, length= x_event.shape
        
        cell_t_i_minus = self.c_0.unsqueeze(0).expand(batch_size, self.hidden_dim).to(self.device)
        cell_bar_im1 = self.cb_0.unsqueeze(0).expand(batch_size, self.hidden_dim).to(self.device)
        hidden_t_i_minus = self.h_0.unsqueeze(0).expand(batch_size, self.hidden_dim).to(self.device)
        
        all_cell, all_cell_bar = [], []
        all_gate_output, all_gate_decay = [], []
        all_hidden = []
        all_hidden_after_update = []

        # Check nhp.get_cells_gates_states()
        
        for i in range(length - 1):
			# only T+1 events update LSTM 
			# BOS, k1t1, ..., kItI
            emb_i = self.mark(x_event[:, i ])
            dtime_i = x_dtime[:, i + 1 ] # need to carefully check here
            
            cell_i, cell_bar_i, gate_decay_i, gate_output_i = self.rnn_cell(
				emb_i, hidden_t_i_minus, cell_t_i_minus, cell_bar_im1
			)
            _, hidden_t_i_plus = self.rnn_cell.decay(
				cell_i, cell_bar_i, gate_decay_i, gate_output_i,
				torch.zeros(dtime_i.size(), device=self.device)
			)
            cell_t_ip1_minus, hidden_t_ip1_minus = self.rnn_cell.decay(
				cell_i, cell_bar_i, gate_decay_i, gate_output_i,
				dtime_i
			)
            all_cell.append(cell_i)
            all_cell_bar.append(cell_bar_i)
            all_gate_decay.append(gate_decay_i)
            all_gate_output.append(gate_output_i)
            all_hidden.append(hidden_t_ip1_minus)
            all_hidden_after_update.append(hidden_t_i_plus)
            cell_t_i_minus = cell_t_ip1_minus
            cell_bar_im1 = cell_bar_i
            hidden_t_i_minus = hidden_t_ip1_minus

        
        all_cell = torch.stack(all_cell, dim=1)
        all_cell_bar = torch.stack(all_cell_bar, dim=1)
        all_gate_decay = torch.stack(all_gate_decay, dim=1)
        all_gate_output = torch.stack(all_gate_output, dim=1)
        all_hidden = torch.stack(all_hidden, dim=1)
        all_hidden_after_update = torch.stack(all_hidden_after_update, dim=1)
        
        return all_cell, all_cell_bar, all_gate_decay, all_gate_output, \
               all_hidden, all_hidden_after_update

    def get_intensities_given_types(self, 
        all_cell, all_cell_bar, all_gate_decay, all_gate_output, 
        event_tensor, dtime_tensor): 
        """
		get intensities given types 
		e.g., for MLE, compute intensities for the sum term in log-likelihood
		note that these cells, gates, event types and dtimes 
		may be either at the oberservation times when actual events happened 
		or at the sampled times 
		e.g., for MLE, they are sampled times for Monte-Carlo approx 
		"""
        embedding = self.mark(event_tensor)
        # batch_size x T+1 x N (N can be 1)
        _, all_h_t = self.rnn_cell.decay(
			all_cell, all_cell_bar, all_gate_decay, all_gate_output, dtime_tensor)
		# batch_size x T+1 x D
        intensities = F.softplus(
			torch.sum(
				embedding * all_h_t.unsqueeze(-2), dim=-1
			), beta=self.beta
		) # batch_size x T+1 x N (N can be 1)

        return intensities
        
    def get_mc_samples(self, 
        all_cell, all_cell_bar, all_gate_decay, all_gate_output, dtime_tensor, 
        mc_sample_num_tensor, duration_tensor, mask_tensor): 
        """
		for MLE, sample time points for each interval 
		for Monte-Carlo approximation of the integral in log-likelihood
		"""
        """
		input 
			mc_sample_num_tensor [B] : # of MC samples per sequence 
			duration_tensor [B] : duration per sequence 
			mask_tensor [B x T+1] : 1.0/0.0 mask of each token of each sequence
		"""
        all_c_inter, all_cb_inter = [], []
        all_d_inter, all_o_inter = [], []
        all_dtime_inter = []
        all_mask_inter = []

        batch_size, T_plus_1, hidden_dim = all_cell.size()
        """
		draw MC time samles 
		TODO : use randomized rounding when rho * I is not integer !!!
		"""
        mc_max = torch.max(mc_sample_num_tensor)
        mc_max = mc_max if mc_max > 1 else 1
        u = torch.ones(size=[batch_size, mc_max], dtype=torch.float32, device=self.device)
        u, _ = torch.sort(u.uniform_(0.0, 1.0)) # batch_size x mc_max 
        sampled_time = u * duration_tensor.unsqueeze(-1)
        
        last_time = torch.zeros(size=[batch_size], dtype=torch.float32, device=self.device)
        
        for i in range(T_plus_1): 
            """
			starting from the 1st (non-BOS) event 
			find mc samples in this interval
			"""
            dtime_i = dtime_tensor[:, i] # batch_size 
            curr_time = last_time + dtime_i 
            fallin = (sampled_time > last_time.unsqueeze(-1)) \
				& (sampled_time <= curr_time.unsqueeze(-1))
			# 0/1 unit 8 : batch_size x mc_max
            """
			find the min rectangle covering all 1 
			"""
            fallin_idx = fallin.sum(0) > 0.5
            
            mask_inter = fallin[:, fallin_idx].float() 
            mask_inter *= mask_tensor[:, i].unsqueeze(1)
            
            chosen_time = sampled_time[:, fallin_idx]
            sampled_dt = chosen_time - last_time.unsqueeze(-1)
            """
			chosen time may < past time : they are chosen cuz that col is chosen 
			and they will eventually be masked out in the end 
			"""
            sampled_dt[sampled_dt < 0.0] = 0.0
			# batch_size x S (S may be 0)
            _, S = sampled_dt.size()
            
            c_inter = all_cell[:, i, :].unsqueeze(1).expand(batch_size, S, hidden_dim)
            cb_inter = all_cell_bar[:, i, :].unsqueeze(1).expand(batch_size, S, hidden_dim)
            d_inter = all_gate_decay[:, i, :].unsqueeze(1).expand(batch_size, S, hidden_dim)
            o_inter = all_gate_output[:, i, :].unsqueeze(1).expand(batch_size, S, hidden_dim)
            
            last_time = curr_time

            all_c_inter.append(c_inter)
            all_cb_inter.append(cb_inter)
            all_d_inter.append(d_inter)
            all_o_inter.append(o_inter)

            all_dtime_inter.append(sampled_dt)
            all_mask_inter.append(mask_inter)

        all_c_inter = torch.cat(all_c_inter, dim=1)
        all_cb_inter = torch.cat(all_cb_inter, dim=1)
        all_d_inter = torch.cat(all_d_inter, dim=1)
        all_o_inter = torch.cat(all_o_inter, dim=1)
        all_dtime_inter = torch.cat(all_dtime_inter, dim=1)
        all_mask_inter = torch.cat(all_mask_inter, dim=1)
        
        return all_c_inter, all_cb_inter, all_d_inter, all_o_inter, \
               all_dtime_inter, all_mask_inter
    
    def get_intensities_all_types(self, 
		all_cell, all_cell_bar, all_gate_decay, all_gate_output, 
		dtime_tensor): 
        """
		get intensities for all types 
		e.g., compute all intensities for integral term in log-likelihood
		note that these cells, gates, and dtimes 
		may be either at the oberservation times when actual events happened 
		or at the sampled times 
		e.g., for MLE, they are sampled times for Monte-Carlo approx 
		e.g., for NCE, they are sampled times from noise dist q 
		"""
        all_embs = self.get_embeddings_all_types()
		# C x D 
        _, all_h_t = self.rnn_cell.decay(
			all_cell, all_cell_bar, all_gate_decay, all_gate_output, dtime_tensor)
		# batch_size x T+1 x D 
        all_intensities = F.softplus(
			torch.matmul(all_h_t, all_embs.t()), beta=self.beta) 
		# batch_size x T+1 x C
        return all_intensities
    
    def get_embeddings_all_types(self):
        return self.mark(self.event_range)