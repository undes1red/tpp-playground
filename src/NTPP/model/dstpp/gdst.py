import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from einops import rearrange, reduce
from random import random
from functools import partial
from tqdm.auto import tqdm
from src.TPP.model.dstpp.gdst_utils import *


class GaussianDiffusion_ST(nn.Module):
    def __init__(
        self,
        model,
        *,
        dim_events,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l2',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0.,
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.dim_events = dim_events

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)

        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min = posterior_variance[1])))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)


    def _vb_terms_bpd(self, x_start, x_t, t, *, clip_denoised: bool,cond=None):
        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        model_mean, _, model_log_variance, pred_xstart, _ = self.p_mean_variance(x=x_t, t=t, clip_denoised=clip_denoised, cond = cond)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl_all = mean_flat(kl) / np.log(np.e)
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, model_mean, 0.5*model_log_variance) # 之前是0.5 * model_log_variance
        assert decoder_nll.shape == x_start.shape
        decoder_nll_all = mean_flat(decoder_nll) / np.log(np.e)

        kl_temporal = mean_flat(kl[:,:,:1]) / np.log(np.e)
        kl_spatial = mean_flat(kl[:,:,-(self.dim_events-1):]) / np.log(np.e)

        decoder_nll_temporal = mean_flat(decoder_nll[:,:,:1]) / np.log(np.e)
        decoder_nll_spatial = mean_flat(decoder_nll[:,:,-(self.dim_events-1):]) / np.log(np.e)

        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        assert kl_all.shape == decoder_nll_all.shape == t.shape == torch.zeros([x_start.shape[0]]).shape
        output = torch.where(t==0, decoder_nll_all, kl_all)
        output_temporal = torch.where(t==0, decoder_nll_temporal, kl_temporal)
        output_spatial = torch.where(t==0, decoder_nll_spatial, kl_spatial)
        
        return output, output_temporal, output_spatial,pred_xstart


    def predict_start_from_noise(self, x_t, t, noise):

        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False,cond=None):
        model_output = self.model(x, t, x_self_cond,cond=cond)
        attn_weight = self.model.get_attn(x, t, x_self_cond,cond=cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), attn_weight

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True, cond=None, Type=None):
        preds, attn_weight = self.model_predictions(x, t, x_self_cond,cond=cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start, attn_weight 


    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True, cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start, attn_weight = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised, cond=cond)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        
        return pred_img, x_start, attn_weight

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            img, x_start, _ = self.p_sample(img, t, self_cond, cond=cond)
        
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.inference_mode()
    def ddim_sample(self, shape, clip_denoised = True, cond=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised, cond=cond)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.inference_mode()
    def sample(self, batch_size = 16, cond=None):
        dim_events, channels = self.dim_events, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, dim_events),cond=cond)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'Euclid':
            return F.pairwise_distance
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise = None, cond=None):
        s0 = time.time()
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond, _ = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond, cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if self.loss_type in  ['l1','l2']:
            loss = self.loss_fn(model_out, target, reduction = 'none')
        elif self.loss_type == 'Euclid':
            loss = self.loss_fn(model_out, target)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        e0 = time.time()
        return loss.mean()

    def NLL_cal(self, img, cond, noise=None):
        x_start = normalize_to_neg_one_to_one(img)
        b, c, n, device  = *x_start.shape,x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        vb_all, vb_temporal_all, vb_spatial_all = [], [], []

        for tt in list(range(self.num_timesteps))[::-1]:
            t = torch.tensor([tt]).expand(b).long().to(device)
            x = self.q_sample(x_start = x_start, t = t, noise = noise)
            vb, vb_temporal, vb_spatial = self._vb_terms_bpd(x_start,x,t,clip_denoised=True,cond=cond)
            vb_all.append(vb.unsqueeze(dim=1))
            vb_temporal_all.append(vb_temporal.unsqueeze(dim=1))
            vb_spatial_all.append(vb_spatial.unsqueeze(dim=1))

        vb_all = torch.sum(torch.cat(vb_all,dim=-1),dim=-1)
        vb_temporal_all = torch.sum(torch.cat(vb_temporal_all,dim=-1),dim=-1)
        vb_spatial_all = torch.sum(torch.cat(vb_spatial_all,dim=-1),dim=-1)

        prior_bpd = self._prior_bpd(x_start)
        prior_bpd_temporal = self._prior_bpd(x_start[:,:,:1])
        prior_bpd_spatial = self._prior_bpd(x_start[:,:,-(self.dim_events-1):])
        total_bpd = vb_all + prior_bpd
        total_bpd_temporal = vb_temporal_all + prior_bpd_temporal
        total_bpd_spatial = vb_spatial_all + prior_bpd_spatial

        assert vb_all.shape == prior_bpd.shape
        return vb_all.sum().item(), vb_temporal_all.sum().item(), vb_spatial_all.sum().item()


    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(np.e), mean_flat(kl_prior[:,:,:1]) / np.log(np.e), mean_flat(kl_prior[:,:,-(self.dim_events-1):]) / np.log(np.e)

    def NLL_cal(self, x_start, cond, noise=None, clip_denoised=True):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        x_start = normalize_to_neg_one_to_one(x_start)
        device = x_start.device
        batch_size = x_start.shape[0]

        vb_all, vb_temporal_all, vb_spatial_all = [], [], []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t] * batch_size, device=device)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with torch.inference_mode():
                vb, vb_temporal, vb_spatial, _ = self._vb_terms_bpd(
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    cond = cond
                )
            vb_all.append(vb.unsqueeze(dim=1))
            vb_temporal_all.append(vb_temporal.unsqueeze(dim=1))
            vb_spatial_all.append(vb_spatial.unsqueeze(dim=1))

        vb_all = torch.sum(torch.cat(vb_all,dim=-1),dim=-1)
        vb_temporal_all = torch.sum(torch.cat(vb_temporal_all,dim=-1),dim=-1)
        vb_spatial_all = torch.sum(torch.cat(vb_spatial_all,dim=-1),dim=-1)

        prior_bpd_all,  prior_bpd_temporal, prior_bpd_spatial= self._prior_bpd(x_start)

        assert vb_all.shape == prior_bpd_all.shape

        total_bpd = vb_all + prior_bpd_all
        total_bpd_temporal = vb_temporal_all + prior_bpd_temporal
        total_bpd_spatial = vb_spatial_all + prior_bpd_spatial
        return total_bpd.sum().item(), total_bpd_temporal.sum().item(), total_bpd_spatial.sum().item()


    def forward(self, img, cond, *args, **kwargs):
        b, c, n, device, dim_events, = *img.shape, img.device, self.dim_events
        assert n == dim_events, f'seq length must be {dim_events}'
        t = torch.randint(0, self.num_timesteps, (b,), device = device).long()
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, cond=cond,  *args, **kwargs)