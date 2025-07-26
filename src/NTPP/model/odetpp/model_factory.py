# Code from https://github.com/facebookresearch/neural_stpp.git.

import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint

from src.TPP.model.odetpp.diffeqnet import construct_diffeqnet
from src.TPP.model.odetpp.split import GRUHiddenStateODEFunc


def build_ode_model(hdim, cond_dim, hidden_dims, actfn, separate, tol, otreg_strength):
    dynamics = []
    for _ in range(separate):
        dstate_net = construct_diffeqnet(hidden_dims[0] // separate, hidden_dims[1:], hidden_dims[0] // separate, \
                                         actfn = actfn, zero_init = True)
        update_net = nn.GRUCell(cond_dim, hidden_dims[0] // separate)
        dynamics.append(GRUHiddenStateODEFunc(dstate_net, update_net))

    hidden_state_dynamics = HiddenStateODEFuncList(*dynamics)

    intensity_net = nn.Sequential(
        nn.Linear(hdim, hdim * 4),
        nn.Softplus(),
        nn.Linear(hdim * 4, 1),
    )
    intensity_odefunc = IntensityODEFunc(hdim, hidden_state_dynamics, intensity_net)
    ode_solver = TimeVariableODE(intensity_odefunc, atol = tol, rtol = tol, energy_regularization = otreg_strength)

    return hidden_state_dynamics, ode_solver


class HiddenStateODEFuncList(nn.Module):
    def __init__(self, *odefuncs):
        super().__init__()
        self.odefuncs = nn.ModuleList(odefuncs)


    def forward(self, t, tpp_state):
        states = torch.split(tpp_state, tpp_state.shape[-1] // len(self.odefuncs), dim = -1)
        ds = []
        for s, func in zip(states, self.odefuncs):
            ds.append(func(t, s))
        return torch.cat(ds, dim = -1)


    def update_state(self, t, tpp_state, cond = None):
        states = torch.split(tpp_state, tpp_state.shape[-1] // len(self.odefuncs), dim = -1)
        upds = []
        for s, func in zip(states, self.odefuncs):
            upds.append(func.update_state(t, s, cond))
        return torch.cat(upds, dim = -1)


class IntensityODEFunc(nn.Module):
    def __init__(self, hdim, dstate_fn, intensity_fn):
        super().__init__()
        self.hdim = hdim
        self.dstate_fn = dstate_fn
        self.intensity_fn = intensity_fn


    def forward(self, t, state):
        Lambda, tpp_state = state
        intensity = self.get_intensity(tpp_state).reshape(-1)
        return intensity, self.dstate_fn(t, tpp_state)


    def get_intensity(self, tpp_state):
        return torch.sigmoid(self.intensity_fn(tpp_state[..., :self.hdim]) - 2.0) * 50


class TimeVariableODE(nn.Module):

    start_time = 0.0
    end_time = 1.0

    def __init__(self, func, atol = 1e-6, rtol = 1e-6, method = "dopri5", energy_regularization = 0.01):
        super().__init__()
        self.func = func
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.energy_regularization = energy_regularization
        self.nfe = 0


    def integrate(self, t0, t1, x0, nlinspace = 1, method = None):
        assert nlinspace > 0
        method = method or self.method

        solution = odeint(
            self,
            (t0, t1, torch.zeros(1).to(x0[0]), *x0),
            torch.linspace(self.start_time, self.end_time, nlinspace + 1).to(t0),
            rtol=self.rtol,
            atol=self.atol,
            method=method,
        )
        _, _, energy, *xs = solution
        reg = energy * self.energy_regularization
        return WrapRegularization.apply(reg, *xs)


    def forward(self, s, state):
        """Solves the same dynamics but uses a dummy variable that always integrates [0, 1]."""
        self.nfe += 1
        t0, t1, _, *x = state

        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0

        with torch.enable_grad():
            x = tuple(x_.requires_grad_(True) for x_ in x)
            dx = self.func(t, x)
            dx = tuple(dx_ * ratio.reshape(-1, *([1] * (dx_.ndim - 1))) for dx_ in dx)

            d_energy = sum(torch.sum(dx_ * dx_) for dx_ in dx) / sum(x_.numel() for x_ in x)

        if not self.training:
            dx = tuple(dx_.detach() for dx_ in dx)

        return tuple([torch.zeros_like(t0), torch.zeros_like(t1), d_energy, *dx])


    def extra_repr(self):
        return f"method={self.method}, atol={self.atol}, rtol={self.rtol}, energy={self.energy_regularization}"


class WrapRegularization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reg, *x):
        ctx.save_for_backward(reg)
        return x


    @staticmethod
    def backward(ctx, *grad_x):
        reg, = ctx.saved_variables
        return (torch.ones_like(reg), *grad_x)