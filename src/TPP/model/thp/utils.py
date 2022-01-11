import math, torch
import torch.nn as nn
import torch.nn.functional as F

def get_non_pad_mask(seq, pad):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(pad).type(torch.float).unsqueeze(-1)                         # [(seq.dim()), 1]


def get_attn_key_pad_mask(seq_k, seq_q, pad):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss