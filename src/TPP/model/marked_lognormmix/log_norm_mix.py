import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack

import torch.distributions as D
from src.TPP.model.marked_lognormmix.distributions import Normal, MixtureSameFamily, TransformedDistribution
from src.TPP.model.marked_lognormmix.utils import clamp_preserve_gradients
from src.TPP.model.marked_lognormmix.recurrent_tpp import RecurrentTPP


class LogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    We model it in the following way (see Appendix D.2 in the paper):

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time
    z = exp(y)

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_scales: Logarithms of scale parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
    """
    def __init__(
        self,
        locs: torch.Tensor,                                                    # \\mu
        log_scales: torch.Tensor,                                              # s without exp()
        log_weights: torch.Tensor,                                             # w
    ):
        mixture_dist = D.Categorical(logits = log_weights)
        component_dist = Normal(loc = locs, scale = log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        transforms = [D.ExpTransform(),]
        super().__init__(GMM, transforms)


    @property
    def mean(self) -> torch.Tensor:
        """
        Compute the expected value of the distribution.

        See https://github.com/shchur/ifl-tpp/issues/3#issuecomment-623720667

        Returns:
            mean: Expected value, shape (batch_size, seq_len)
        """
        a = self.std_log_inter_time
        b = self.mean_log_inter_time
        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.logits
        return (log_weights + a * loc + b + 0.5 * a**2 * variance).logsumexp(-1).exp()


class MarkedLogNormMix(RecurrentTPP):
    """
    RNN-based MTPP model for marked and unmarked event sequences.
    The distribution of the inter-event times given the history is modeled with a LogNormal mixture distribution.
    Original model assumes that the marks are conditionally independent of the inter-event times.
    We try to fix it by setting up multiple LogNorm Distributions, one for each mark.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        num_mix_components: Number of mixture components in the inter-event time distribution.
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """

    def __init__(
        self,
        num_marks: int,
        device: str,
        context_size: int = 32,
        mark_embedding_size: int = 32,
        num_mix_components: int = 16,
        rnn_type: str = "GRU",
    ):
        super().__init__(
            num_marks=num_marks,
            context_size=context_size,
            mark_embedding_size=mark_embedding_size,
            rnn_type=rnn_type,
            device = device
        )
        self.device = device
        self.num_marks = num_marks
        self.num_mix_components = num_mix_components
        self.eps = 1e-20

        self.linear = nn.Linear(2 * self.context_size, 3 * self.num_mix_components, device = self.device)
        self.transform_mark_dist = nn.Linear(self.context_size, self.num_marks, device = self.device)
        self.salt = nn.Parameter(torch.randn((num_marks, context_size), device = self.device))
        nn.init.xavier_uniform_(self.salt)


    def get_inter_time_dist(self, context_history: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        batch_size, seq_len, _  = context_history.shape[:3]

        # Repeat the history representation self.num_marks times. Each group of representation are responsible for
        # one group of Mixture distribution parameters.
        repeated_context_history = repeat(context_history, '... cs -> ... nm cs', nm = self.num_marks)
                                                                               # [batch_size, seq_len + 1, num_marks, context_size]
        self.mark_dist = torch.nn.functional.softmax(self.transform_mark_dist(context_history), dim = -1)
                                                                               # [batch_size, seq_len + 1, num_marks]

        # We have to add some mark-specific salt to context history, otherwise we have to implement a special linear module that applies
        # different linear transformation on each mark dimension.
        salt = repeat(self.salt, f'... -> bs sl {"() " * (len(context_history.shape) - 3)} ...', bs = batch_size, sl = seq_len)
                                                                               # [batch_size, seq_len + 1, num_marks, context_size]
        salty_context_history = torch.cat((repeated_context_history, salt), dim = -1)
                                                                               # [batch_size, seq_len + 1, num_marks, 2 * context_size]

        raw_params = self.linear(salty_context_history)                        # [batch_size, seq_len + 1, num_marks, 3 * num_mix_components]
        # Slice the tensor to get the parameters of the mixture
        locs, log_scales, log_weights = torch.chunk(raw_params, 3, dim = -1)   # 3 * [batch_size, seq_len + 1, num_marks, num_mix_components]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)           # [batch_size, seq_len + 1, num_marks, num_mix_components]
        log_weights = torch.log_softmax(log_weights, dim = -1)                 # [batch_size, seq_len + 1, num_marks, num_mix_components]

        locs = rearrange(locs, '... a b -> a ... b')                           # [num_marks, batch_size, seq_len + 1, num_mix_components]
        log_scales = rearrange(log_scales, '... a b -> a ... b')               # [num_marks, batch_size, seq_len + 1, num_mix_components]
        log_weights = rearrange(log_weights, '... a b -> a ... b')             # [num_marks, batch_size, seq_len + 1, num_mix_components]

        self.distribution_list = []

        for loc_per_mark, log_scales_per_mark, log_weights_per_mark in \
            zip(locs, log_scales, log_weights):
            self.distribution_list.append(LogNormalMixtureDistribution(
                locs = loc_per_mark,
                log_scales = log_scales_per_mark,
                log_weights = log_weights_per_mark
            ))
        
        assert len(self.distribution_list) == self.num_marks
        
        return self
    

    def get_log_prob(self, input_time):
        log_prob = []
        for sub_distribution in self.distribution_list:
            log_prob.append(sub_distribution.log_prob(input_time))             # [batch_size, seq_len + 1, ...]
        
        log_prob = torch.stack(log_prob, dim = -1)                             # [batch_size, seq_len + 1, ..., num_marks]
        log_mark_distribution = torch.log(self.mark_dist + self.eps)           # [batch_size, seq_len + 1, ..., num_marks]
        log_prob += log_mark_distribution                                      # [batch_size, seq_len + 1, ..., num_marks]

        return log_prob


    def get_prob(self, input_time):
        log_prob = []
        for sub_distribution in self.distribution_list:
            log_prob.append(sub_distribution.log_prob(input_time))             # [batch_size, seq_len + 1, ...]
        
        log_prob = torch.stack(log_prob, dim = -1)                             # [batch_size, seq_len + 1, ..., num_marks]
        prob = torch.exp(log_prob)                                             # [batch_size, seq_len + 1, ..., num_marks]
        einop = f'... nm -> ... {"() " * (len(log_prob.shape) - len(self.mark_dist.shape))} nm'
        mark_distribution = rearrange(self.mark_dist, einop)                   # [batch_size, seq_len + 1, ..., num_marks]
        prob = mark_distribution * prob                                        # [batch_size, seq_len + 1, ..., num_marks]

        return prob


    def get_log_survival_function(self, input_time):
        log_prob_survival = []
        for sub_distribution in self.distribution_list:
            log_prob_survival.append(sub_distribution.log_survival_function(input_time))
                                                                               # [batch_size, seq_len + 1]
        
        log_prob_survival = torch.stack(log_prob_survival, dim = -1)           # [batch_size, seq_len + 1, num_marks]
        prob_survival = torch.exp(log_prob_survival)                           # [batch_size, seq_len + 1, num_marks]
        prob_survival = self.mark_dist * prob_survival                         # [batch_size, seq_len + 1, num_marks]
        log_prob_survival = torch.log(prob_survival.sum(dim = -1) + self.eps)  # [batch_size, seq_len + 1]

        return log_prob_survival


    def get_cdf(self, input_time):
        log_cdf = []
        for sub_distribution in self.distribution_list:
            log_cdf.append(sub_distribution.log_cdf(input_time))               # [..., batch_size, seq_len + 1]
        
        log_cdf = torch.stack(log_cdf, dim = -1)                               # [..., batch_size, seq_len + 1, num_marks]
        cdf = torch.exp(log_cdf)                                               # [..., batch_size, seq_len + 1, num_marks]
        needed_einops = f'... -> {"() " * (len(log_cdf.shape) - len(self.mark_dist.shape))}...'
        log_mark_distribution = rearrange(self.mark_dist, needed_einops)       # [..., batch_size, seq_len + 1, num_marks]
        cdf *= log_mark_distribution                                           # [..., batch_size, seq_len + 1, num_marks]

        return cdf


    def get_cdf_3d(self, input_time):
        log_cdf = []
        input_time = rearrange(input_time, '... nm -> nm ...')                 # [num_marks, ..., batch_size, seq_len + 1]

        for sub_distribution, sub_time in zip(self.distribution_list, input_time):
            log_cdf.append(sub_distribution.log_cdf(sub_time))                 # [..., batch_size, seq_len + 1]
        
        log_cdf = torch.stack(log_cdf, dim = -1)                               # [..., batch_size, seq_len + 1, num_marks]
        cdf = torch.exp(log_cdf)                                               # [..., batch_size, seq_len + 1, num_marks]
        needed_einops = f'... -> {"() " * (len(log_cdf.shape) - len(self.mark_dist.shape))}...'
        log_mark_distribution = rearrange(self.mark_dist, needed_einops)       # [..., batch_size, seq_len + 1, num_marks]
        cdf *= log_mark_distribution                                           # [..., batch_size, seq_len + 1, num_marks]

        return cdf


    def get_mark_distribution(self):
        return self.mark_dist