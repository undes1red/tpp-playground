import torch
import torch.nn as nn

from torch.distributions import Categorical

class RecurrentTPP(nn.Module):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """
    def __init__(
        self,
        num_marks: int,
        device: str,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 32,
        mark_embedding_size: int = 32,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.device = device
        self.num_marks = num_marks
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_size = context_size
        self.mark_embedding_size = mark_embedding_size
        if self.num_marks > 1:
            self.num_features = 1 + self.mark_embedding_size
            self.mark_embedding = nn.Embedding(self.num_marks, self.mark_embedding_size, device = self.device)
            self.mark_linear = nn.Linear(self.context_size, self.num_marks, device = self.device)
        else:
            self.num_features = 1
        self.rnn_type = rnn_type
        self.context_init = nn.Parameter(torch.zeros(context_size, device = self.device))  # initial state of the RNN
        self.rnn = getattr(nn, rnn_type)(input_size=self.num_features, hidden_size=self.context_size, batch_first=True, device = self.device)

    def get_features(self, sequences, mean_and_var) -> torch.Tensor:
        """
        Convert each event in a sequence into a feature vector using normalization.

        Args:
            sequences: [event_tensor, time_tensor, mask_tensor]
            mean_and_var: (mean, var) or None

        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)

        """
        mean, var = mean_and_var if mean_and_var else (0, 1)
        features = torch.log(sequences[1] + 1e-8).unsqueeze(-1)                # [batch_size, seq_len, 1]
        features = (features - mean)/var                                       # [batch_size, seq_len, 1]
        if self.num_marks > 1:
            mark_emb = self.mark_embedding(sequences[0])                       # [batch_size, seq_len, mark_embedding_size]
            features = torch.cat([features, mark_emb], dim=-1)
        return features                                                        # [batch_size, seq_len, mark_embedding_size + 1]
    
    def get_features_prober(self, sequences, mean_and_var, resolution) -> torch.Tensor:
        """
        Convert each event in a sequence into a feature vector using normalization after expand the original data into distribution prober
        batch.

        Args:
            sequences: [event_tensor, time_tensor, mask_tensor]
            mean_and_var: (mean, var) or None
            resolution: how many points does every time interval have?

        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)

        """
        mean, var = mean_and_var
        time = torch.log(sequences[1] + 1e-8).unsqueeze(-1)                    # [batch_size, seq_len, 1]
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        expanded_time = time * time_multiplier                                 # [batch_size, seq_len, resolution]
        expanded_time = (expanded_time - mean)/var                             # [batch_size, seq_len, resolution]
        expanded_time = expanded_time.unsqueeze(-1)                            # [batch_size, seq_len, resolution, 1]

        if self.num_marks > 1:
            event_expander = torch.ones(resolution)                            # [resolution]
            expand_events = sequences[0] * event_expander                      # [batch_size, seq_len, resolution]
            expended_mark_emb = self.mark_embedding(expand_events)             # [batch_size, seq_len, resolution, mark_embedding_size]
            features = torch.cat([expanded_time, expended_mark_emb], dim=-1)   # [batch_size, seq_len, resolution, mark_embedding_size + 1]
        return features                                                        # [batch_size, seq_len, resolution, mark_embedding_size + 1]

    def get_context(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get the context (history) embedding from the sequence of events.

        Args:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)

        Returns:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size) if remove_last == False
                shape (batch_size, seq_len + 1, context_size) if remove_last == True

        """
        context = self.rnn(features)[0]                                        # [batch_size, seq_len + 1, context_size]
        batch_size, seq_len, context_size = context.shape
        context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)
                                                                               # [batch_size, 1, context_size]
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        context = context[:, :-1, :]                                           # [batch_size, seq_len, context_size]
        context = torch.cat([context_init, context], dim=1)                    
        return context

    def get_context_prober(self, features: torch.Tensor, resolution) -> torch.Tensor:
        """
        Get the context (history) embedding from the sequence of events.

        Args:
            features: Feature vector corresponding to each event,
                shape [batch_size, seq_len, num_features]

        Returns:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size) if remove_last == False
                shape (batch_size, seq_len + 1, context_size) if remove_last == True

        """
        context = self.rnn(features)[0]                                        # [batch_size, seq_len + 1, context_size]
        batch_size, seq_len, context_size = context.shape
        context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)
                                                                               # [batch_size, 1, context_size]
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        context = context[:, :-1, :]                                           # [batch_size, seq_len, context_size]
        context = torch.cat([context_init, context], dim=1)                    # [batch_size, seq_len + 1, context_size]
        context = context.unsqueeze(-2).repeat(1, 1, resolution, 1)            # [batch_size, seq_len + 1, resolution, context_size]
        return context

    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raise NotImplementedError()

    def log_prob(self, batch) -> torch.Tensor:
        """Compute log-likelihood for a batch of sequences.

        Args:
            batch: the input minibatch
            [
                [
                    event_tensor,
                    time_tensor,
                    mask_tensor
                ],
                score,
                [
                    mean,
                    var
                ](if self.input_norm_data is True, otherwise it is a None.)
            ]
        Returns:
            log_p: shape (batch_size,)

        """
        # extract features from minibatch, data normalization applies here.
        sequences, _, mean_and_var = batch
        event, time_interval, mask = sequences
        features = self.get_features(sequences, mean_and_var)                  # [batch_size, seq_len, mark_embedding_size + 1]
        # I think this statement is for debugging.
        # if features.isnan().any():
        #     print(batch)
        '''
        RNN is employed to generate context vector. self.get_inter_time_dist will generate the history embedding,
        metadata and sequence embedding from the context representation. These embeddings are the backbone of the
        distribution.
        inter_time_dist is the p(\tau | w, \mu, s) defined in Equation 2.
        '''
        context = self.get_context(features)
        inter_time_dist = self.get_inter_time_dist(context)
        inter_times = time_interval.clamp(1e-10)
        # Using obtained invertible distribution we can obatin the log probability for each inter time.
        log_p = inter_time_dist.log_prob(inter_times)                          # [batch_size, seq_len]

        '''
        Survival probability of the last interval (from t_N to t_end).
        You can comment this section of the code out if you don't want to implement the log_survival_function
        for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
        but the difference shouldn't be significant if you are working with long sequences.
        '''
        last_event_idx = mask.sum(-1, keepdim=True).long()                     # [batch_size, 1]
        log_surv_all = inter_time_dist.log_survival_function(inter_times)
                                                                               # [batch_size, seq_len]
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)
                                                                               # [batch_size]

        if self.num_marks > 1:
            mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1) # [batch_size, seq_len, num_marks]
            mark_dist = Categorical(logits=mark_logits)
            log_p += mark_dist.log_prob(event)                                 # [batch_size, seq_len]
        log_p *= mask                                                          # [batch_size, seq_len]
        the_number_of_events = mask.sum()
        return log_p.sum(-1) + log_surv_last, the_number_of_events

    def log_prob_prober(self, batch, resolution) -> torch.Tensor:
        """Compute log-likelihood for a batch of sequences.
        Args:
            batch: the input minibatch
            [
                [
                    event_tensor,
                    time_tensor,
                    mask_tensor
                ],
                score,
                [
                    mean,
                    var
                ](if self.input_norm_data is True, otherwise it is a None.)
            ]
            resolution: Shows how many interpolative points each time interval has.
        Returns:
            log_p: shape (batch_size,)

        """
        # extract features from minibatch, data normalization applies here.
        sequences, _, mean_and_var, _ = batch
        event, time_interval, mask = sequences
        batch_size, seq_len = time_interval.shape
        seq_len -= 1

        features = self.get_features(sequences, mean_and_var)                  # [batch_size, seq_len + 1, mark_embedding_size + 1]
        # if features.isnan().any():
        #     print(batch)
        # RNN is employed to generate context vector. self.get_inter_time_dist will generate the history embedding,
        # metadata and sequence embedding from the context representation. These embeddings are the backbone of the
        # distribution.
        # inter_time_dist is the p(\tau | w, \mu, s) defined in Equation 2.
        expanded_context = self.get_context_prober(features, resolution = resolution)
                                                                               # [batch_size, seq_len + 1, resolution, context_size]
        inter_time_dist = self.get_inter_time_dist(expanded_context)
        
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        expanded_inter_times = time_interval.unsqueeze(-1) * time_multiplier.clamp(1e-10)
                                                                               # [batch_size, seq_len + 1, resolution]
        # Using obtained invertible distribution we can obatin the log probability for each inter time.
        expanded_log_p = inter_time_dist.log_prob(expanded_inter_times)        # [batch_size, seq_len + 1, resolution]

        # drop probability predictions between the last event and end_time.
        log_p = torch.exp(expanded_log_p[:, :-1, :]).reshape(batch_size, -1)   # [batch_size, seq_len * resolution]

        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_inter_times[:, :-1, :].diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        '''
        # Survival probability of the last interval (from t_N to t_end).
        # You can comment this section of the code out if you don't want to implement the log_survival_function
        # for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
        # but the difference shouldn't be significant if you are working with long sequences.
        '''
        # last_event_idx = mask.sum(-1, keepdim=True).long()                     # [batch_size, 1]
        # log_surv_all = inter_time_dist.log_survival_function(expanded_inter_times)
        #                                                                        # [batch_size, seq_len, resolution]
        # log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx.unsqueeze(-1)).squeeze(-1)
        #                                                                        # [batch_size]

        return log_p, timestamp

    # def sample(self, t_end: float, batch_size: int = 1, context_init: torch.Tensor = None):
    #     """Generate a batch of sequence from the model.
# 
    #     Args:
    #         t_end: Size of the interval on which to simulate the TPP.
    #         batch_size: Number of independent sequences to simulate.
    #         context_init: Context vector for the first event.
    #             Can be used to condition the generator on past events,
    #             shape (context_size,)
# 
    #     Returns;
    #         batch: Batch of sampled sequences. See dpp.data.batch.Batch.
    #     """
    #     if context_init is None:
    #         # Use the default context vector
    #         context_init = self.context_init
    #     else:
    #         # Use the provided context vector
    #         context_init = context_init.view(self.context_size)
    #     next_context = context_init[None, None, :].expand(batch_size, 1, -1)
    #     inter_times = torch.empty(batch_size, 0)
    #     if self.num_marks > 1:
    #         marks = torch.empty(batch_size, 0, dtype=torch.long)
    #     
    #     generated = False
    #     while not generated:
    #         inter_time_dist = self.get_inter_time_dist(next_context)
    #         next_inter_times = inter_time_dist.sample()  # (batch_size, 1)
    #         inter_times = torch.cat([inter_times, next_inter_times], dim=1)  # (batch_size, seq_len)
    # 
    #         # Generate marks, if necessary
    #         if self.num_marks > 1:
    #             mark_logits = torch.log_softmax(self.mark_linear(next_context), dim=-1)  # (batch_size, 1, num_marks)
    #             mark_dist = Categorical(logits=mark_logits)
    #             next_marks = mark_dist.sample()  # (batch_size, 1)
    #             marks = torch.cat([marks, next_marks], dim=1)
    #         else:
    #             marks = None
    #     
    #         with torch.no_grad():
    #             generated = inter_times.sum(-1).min() >= t_end
    #         batch = Batch(inter_times=inter_times, mask=torch.ones_like(inter_times), marks=marks)
    #         features = self.get_features(batch)  # (batch_size, seq_len, num_features)
    #         context = self.get_context(features, remove_last=False)  # (batch_size, seq_len, context_size)
    #         next_context = context[:, [-1], :]  # (batch_size, 1, context_size)
    # 
    #     arrival_times = inter_times.cumsum(-1)  # (batch_size, seq_len)
    #     inter_times = diff(arrival_times.clamp(max=t_end), dim=-1)
    #     mask = (arrival_times <= t_end).float()  # (batch_size, seq_len)
    #     if self.num_marks > 1:
    #         marks = marks * mask  # (batch_size, seq_len)
    #     return Batch(inter_times=inter_times, mask=mask, marks=marks)
