import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUHiddenStateODEFunc(nn.Module):

    def __init__(self, dstate_net, update_net):
        super().__init__()
        self.dstate_net = dstate_net
        self.update_net = update_net

    def forward(self, t, tpp_state):
        return self.dstate_net(t, tpp_state)

    def update_state(self, t, tpp_state, cond=None):
        if cond is None:
            bsz = tpp_state.shape[0]
            cond = torch.zeros(bsz, 0).to(tpp_state)

        return self.update_net(cond, tpp_state)