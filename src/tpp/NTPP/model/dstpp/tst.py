import torch, math
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, rearrange


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.device = device
        self.dim = dim


    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = self.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ST_Diffusion(nn.Module):
    def __init__(self, device, dim, num_units = 64, self_condition = False, condition = True, cond_dim = 0):
        super(ST_Diffusion, self).__init__()
        self.device = device
        self.channels = 1
        self.self_condition = self_condition
        self.condition = condition

        sinu_pos_emb = SinusoidalPosEmb(num_units, device = self.device)
        fourier_dim = num_units

        time_dim = num_units

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim, device = self.device),
            nn.GELU(),
            nn.Linear(time_dim, time_dim, device = self.device)
        )

        self.linears_spatial = nn.ModuleList(
            [
                nn.Linear(dim - 1, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, num_units, device = self.device),
            ]
        )

        self.linears_temporal = nn.ModuleList(
            [
                nn.Linear(1, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, num_units, device = self.device),
            ]
        )

        self.output_spatial = nn.Sequential(
                nn.Linear(num_units, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, dim-1, device = self.device)
        )

        self.output_temporal = nn.Sequential(
                nn.Linear(num_units, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, 1, device = self.device)
        )

        self.linear_t = nn.Sequential(
                nn.Linear(num_units * 2, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, 2, device = self.device)
        )

        self.linear_s = nn.Sequential(
                nn.Linear(num_units * 2, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, 2, device = self.device)
        )

        self.cond_all = nn.Sequential(
                nn.Linear(cond_dim * 3, num_units, device = self.device),
                nn.ReLU(),
                nn.Linear(num_units, num_units, device = self.device)
        )

        self.cond_temporal = nn.ModuleList(
            [
                nn.Linear(cond_dim, num_units, device = self.device),
                nn.Linear(cond_dim, num_units, device = self.device),
                nn.Linear(cond_dim, num_units, device = self.device)
            ]
        )

        self.cond_spatial = nn.ModuleList(
            [
                nn.Linear(cond_dim, num_units, device = self.device),
                nn.Linear(cond_dim, num_units, device = self.device),
                nn.Linear(cond_dim, num_units, device = self.device)
            ]
        )

        self.cond_joint = nn.ModuleList(
            [
                nn.Linear(cond_dim, num_units, device = self.device),
                nn.Linear(cond_dim, num_units, device = self.device),
                nn.Linear(cond_dim, num_units, device = self.device)
            ]
        )

    def get_attn(self, x, t, x_self_cond = None, cond = None):
        
        cond = self.cond_all(cond)
        t_embedding = self.time_mlp(t).unsqueeze(dim=1)

        
        cond_all = torch.cat((cond,t_embedding),dim=-1)

        alpha_s = F.softmax(self.linear_s(cond_all), dim=-1).squeeze(dim=1)
        alpha_t = F.softmax(self.linear_t(cond_all), dim=-1).squeeze(dim=1)

        return alpha_s, alpha_t


    def forward(self, x, t, x_self_cond = None, cond = None):
        
        x_spatial, x_temporal = x[..., 1:].clone(), x[..., :1].clone()         # [..., 1, dim_marks] + [..., 1, 1]

        cond_temporal, cond_spatial, cond_joint = cond.chunk(3, dim = -1)      # 3 * [..., 1, d_model]
        assert cond_temporal.shape[-1] == cond_spatial.shape[-1] == cond_joint.shape[-1]

        cond = self.cond_all(cond)                                             # [..., 1, num_units]
        t_embedding = self.time_mlp(t).unsqueeze(dim = 1)                      # [..., 1, num_units]
        cond_all = torch.cat((cond, t_embedding),dim = -1)                     # [..., 1, 2 * num_units]

        alpha_s = F.softmax(self.linear_s(cond_all), dim = -1)                 # [..., 1, 2]
        alpha_t = F.softmax(self.linear_t(cond_all), dim = -1)                 # [..., 1, 2]
        einop = '... a b -> ... b a'
        alpha_s = rearrange(alpha_s, einop)                                    # [..., 2, 1]
        alpha_t = rearrange(alpha_t, einop)                                    # [..., 2, 1]

        for idx in range(3):
            #t_embedding = embedding_layer(t).unsqueeze(dim=1)
            x_spatial = self.linears_spatial[2 * idx](x_spatial)
            x_temporal = self.linears_temporal[2 * idx](x_temporal)
            assert x_spatial.shape == t_embedding.shape
            x_spatial += t_embedding
            x_temporal += t_embedding

            cond_joint_emb = self.cond_joint[idx](cond_joint)
            cond_temporal_emb = self.cond_temporal[idx](cond_temporal)
            cond_spatial_emb = self.cond_spatial[idx](cond_spatial)

            x_spatial += cond_joint_emb + cond_spatial_emb
            x_temporal += cond_joint_emb + cond_temporal_emb

            x_spatial = self.linears_spatial[2 * idx + 1](x_spatial)
            x_temporal = self.linears_temporal[2 * idx + 1](x_temporal)

        x_spatial = self.linears_spatial[-1](x_spatial)                        # [..., 1, num_units]
        x_temporal = self.linears_temporal[-1](x_temporal)                     # [..., 1, num_units]

        x_output = torch.cat((x_temporal, x_spatial), dim = -2)                # [..., 1, num_units]

        x_output_t = (x_output * alpha_t).sum(dim = 1, keepdim = True)         # [..., 1, num_units]
        x_output_s = (x_output * alpha_s).sum(dim = 1, keepdim = True)         # [..., 1, num_units]

        pred = torch.cat((self.output_temporal(x_output_t), self.output_spatial(x_output_s)), dim = -1)
                                                                               # [..., 1, dim_marks + 1]
        return pred