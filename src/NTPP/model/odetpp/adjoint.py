import torch

class NeuralODEAdjoint(torch.autograd.Function):

    def __init__(self, device):
        super(NeuralODEAdjoint, self).__init__()
        self.device = device

    @staticmethod
    def forward(ctx, z_init, delta_t, ode_fn, solver, num_sample_times, *model_parameters):
        """

        Args:
            ctx:
            input: (tensor): [batch_size]
            model:
            solver:
            delta_t (tensor): [batch_size, num_sample_times]

        Returns:

        """

        ctx.ode_fn = ode_fn
        ctx.solver = solver
        ctx.delta_t = delta_t
        ctx.model_parameters = model_parameters
        ctx.num_sample_times = num_sample_times

        total_state = []
        dt_ratio = 1.0 / num_sample_times
        delta_t = delta_t * dt_ratio
        with torch.inference_mode():
            state = z_init
            for _ in range(num_sample_times):
                state = solver(diff_func = ode_fn, dt = delta_t, z0 = state)   # [batch_size, hidden_size]
                total_state.append(state)

        ctx.save_for_backward(state)                                           # [batch_size, num_samples, hidden_size]

        return state

    @staticmethod
    def backward(ctx, grad_z):
        output_state = ctx.saved_tensors[0]  # return a tuple
        ode_fn = ctx.ode_fn
        solver = ctx.solver
        delta_t = ctx.delta_t
        model_parameters = ctx.model_parameters
        num_sample_times = ctx.num_sample_times

        # Dynamics of augmented system to be calculated backwards in time
        def aug_dynamics(aug_states):
            tmp_z = aug_states[0]
            tmp_neg_a = -aug_states[1]

            with torch.set_grad_enabled(True):
                tmp_z = tmp_z.detach().requires_grad_(True)
                func_eval = ode_fn(tmp_z)
                tmp_ds = torch.autograd.grad(
                    (func_eval,), (tmp_z, *model_parameters),
                    grad_outputs=tmp_neg_a,
                    allow_unused=True,
                    retain_graph=True)

            neg_adfdz = tmp_ds[0]
            neg_adfdtheta = [torch.flatten(var) for var in tmp_ds[1:]]

            return [func_eval, neg_adfdz, *neg_adfdtheta]

        dt_ratio = 1.0 / num_sample_times
        delta_t = delta_t * dt_ratio

        with torch.inference_mode():
            # Construct back-state for ode solver
            # reshape variable \theta for batch solving
            init_var_grad = [torch.zeros_like(torch.flatten(var)) for var in model_parameters]

            # [z(t_1), a(t_1), \theta]
            z1 = output_state
            a1 = grad_z
            states = [z1, a1, *init_var_grad]

            for i in range(num_sample_times):
                states = solver(aug_dynamics, -delta_t, states)

            grad_z0 = states[1]

            grad_theta = [torch.reshape(torch.mean(var_grad, dim=0), var.shape) for var, var_grad in
                          zip(model_parameters, states[2:])]

        return (grad_z0, None, None, None, None, *grad_theta)


class NeuralODE(torch.nn.Module):
    def __init__(self, model, solver, num_sample_times, device):
        super().__init__()
        self.model = model
        self.solver = solver
        self.params = [w for w in model.parameters()]
        self.num_sample_times = num_sample_times
        self.device = device

    def forward(self, input_state, delta_time):
        """

        Args:
            input_state: [batch_size, hidden_size]
            return_state:

        Returns:

        """
        output_state = NeuralODEAdjoint.apply(input_state,
                                              delta_time,
                                              self.model,
                                              self.solver,
                                              self.num_sample_times,
                                              *self.params)

        
        return output_state                                                    # [batch_size, num_sample_times, hidden_size]