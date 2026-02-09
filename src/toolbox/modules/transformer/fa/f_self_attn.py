import torch
import torch.nn as nn
from einops import rearrange

try:
    import flash_attn
    flash_attn_qkvpacked_func = flash_attn.flash_attn_qkvpacked_func
except ImportError:
    flash_attn = None
    flash_attn_qkvpacked_func = None

from src.toolbox.modules.ffn import FFN


class FMHSA(nn.Module):
    def __init__(self, training, n_head, d_input, d_qkv, device, d_hidden, dropout=0.1):
        super().__init__()
        self.training = training
        self.device = device

        self.attn = FMHSALayer(
            training=training, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=self.device, dropout=dropout
        )
        self.ffn = FFN(d_input=d_input, d_hidden=d_hidden, device=self.device, dropout=dropout)

    def forward(self, x, non_pad_mask=None):
        """
        Args:
        1. x: input tensor. shape: [batch_size, seq_len, d_input]
        2. non_pad_mask: mask tensor for used by self attention and to mask out pad items. shape: [batch_size, seq_len]
        Outputs:
        1. output: results of transformer layer. shape: [batch_size, seq_len, d_input]
        """
        output = self.attn(x, mask=non_pad_mask)
        # [..., batch_size, seq_len, d_input]

        output = self.ffn(output)  # [..., batch_size, seq_len, d_input]

        if non_pad_mask is not None:
            output *= rearrange(non_pad_mask, "... -> ... 1")  # [..., batch_size, seq_len, d_input]

        return output


class FMHSALayer(nn.Module):
    def __init__(self, training, n_head, d_input, d_qkv, device, dropout=0.1):
        super().__init__()
        self.training = training
        self.device = device
        if device == "cpu":
            raise ValueError("Flash Attention does not work on CPU.")

        self.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func

        self.d_input = d_input
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.dropout = dropout if self.training else 0

        # Linear: d_input -> d_q, d_k, or d_v
        self.w_qkv = nn.Linear(d_input, self.d_qkv * self.n_head * 3, bias=True, device=self.device)

        # Linear: n_head * d_q, d_k, or d_v -> d_input
        self.fc_attn_output = nn.Linear(self.n_head * d_qkv, self.d_input, bias=True, device=self.device)

        # layer normalization
        self.layer_norm_for_q = nn.RMSNorm(self.d_input, eps=1e-6, device=self.device, dtype=torch.get_default_dtype())

    def forward(self, x, mask=None):
        """
        Args:
        1. x: input tensor. shape: [..., batch_size, seq_len, d_input]
        2. mask: the mask tensor used by self attention. shape: [..., batch_size, seq_len]
        Output:
        1. output: results of transformer layer. shape: [..., batch_size, seq_len, d_input]
        """
        input_shape = x.shape[:-1]

        residual = x
        x = self.layer_norm_for_q(x)  # [..., batch_size, seq_len, d_input]

        if len(x.shape) > 3:
            x = x.flatten(end_dim=-3)
            # [(... * batch_size), seq_len, d_input]

        qkv = self.w_qkv(x)  # [(... * batch_size), seq_len, d_qkv * n_head * 3]
        qkv = rearrange(
            qkv,
            "... seq_len (nqkv nhead dqkv) -> ... seq_len nqkv nhead dqkv",
            nqkv=3,
            nhead=self.n_head,
            dqkv=self.d_qkv,
        )
        # [(... * batch_size), seq_len, 3, n_head, d_qkv]

        fa_output = self.flash_attn_qkvpacked_func(
            qkv, self.dropout, causal=True, deterministic=True
        )  # [..., nheads, d_qkv]
        fa_output = rearrange(fa_output, "...  nh dqkv -> ... (nh dqkv)")
        # [(... * batch_size), seq_len, n_head * d_qkv]

        fa_output = fa_output.view(*input_shape, -1)  # [..., batch_size, seq_len, n_head * d_qkv]
        fa_output = self.fc_attn_output(fa_output)  # [..., batch_size, seq_len, d_input]
        if mask is not None:
            fa_output = fa_output * mask.unsqueeze(dim=-1)  # [..., batch_size, seq_len, d_input]
        fa_output += residual

        return fa_output  # [..., batch_size, seq_len, d_output]


if __name__ == "__main__":
    # Mock flash_attn before it's used in the module if possible,
    # but here it's already imported. We rely on patch.

    torch.set_default_dtype(torch.bfloat16)

    def mock_fa_func(qkv, dropout, causal, deterministic):
        return torch.randn(
            qkv.shape[0], qkv.shape[1], qkv.shape[3], qkv.shape[4], device=qkv.device, dtype=qkv.dtype
        )

    # Apply patch globally for the tests in this module
    # We patch the local reference in the module
    import src.toolbox.modules.transformer.fa.f_self_attn as m
    m.flash_attn_qkvpacked_func = mock_fa_func

    def test_fmhsa_layer_init():
        device = torch.device("cuda")
        n_head = 4
        d_input = 64
        d_qkv = 32

        layer = FMHSALayer(training=True, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.1)

        assert layer.n_head == n_head
        assert layer.d_input == d_input
        assert layer.d_qkv == d_qkv
        assert layer.w_qkv.out_features == d_qkv * n_head * 3

    def test_fmhsa_init():
        device = torch.device("cuda")
        n_head = 4
        d_input = 64
        d_qkv = 32
        d_hidden = 128

        model = FMHSA(
            training=True,
            n_head=n_head,
            d_input=d_input,
            d_qkv=d_qkv,
            device=device,
            d_hidden=d_hidden,
            dropout=0.1,
        )
        assert isinstance(model.attn, FMHSALayer)
        assert model.attn.d_qkv == d_qkv

    def test_fmhsa_layer_forward():
        device = torch.device("cuda")
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 32

        layer = FMHSALayer(training=True, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.0)
        layer.flash_attn_qkvpacked_func = mock_fa_func

        x = torch.randn(batch_size, seq_len, d_input, device=device)
        mask = torch.ones(batch_size, seq_len, device=device)

        output = layer(x, mask=mask)

        assert output.shape == (batch_size, seq_len, d_input)

    def test_fmhsa_forward():
        device = torch.device("cuda")
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 32
        d_hidden = 128

        model = FMHSA(
            training=True,
            n_head=n_head,
            d_input=d_input,
            d_qkv=d_qkv,
            device=device,
            d_hidden=d_hidden,
            dropout=0.1,
        )
        model.attn.flash_attn_qkvpacked_func = mock_fa_func

        x = torch.randn(batch_size, seq_len, d_input, device=device)
        non_pad_mask = torch.ones(batch_size, seq_len, device=device)

        output = model(x, non_pad_mask=non_pad_mask)

        assert output.shape == (batch_size, seq_len, d_input)

    def test_fmhsa_with_masking():
        device = torch.device("cuda")
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 32
        d_hidden = 128

        model = FMHSA(
            training=True,
            n_head=n_head,
            d_input=d_input,
            d_qkv=d_qkv,
            device=device,
            d_hidden=d_hidden,
            dropout=0.1,
        )
        model.attn.flash_attn_qkvpacked_func = mock_fa_func

        x = torch.randn(batch_size, seq_len, d_input, device=device)
        # Mask out the last 5 elements of the second sequence
        non_pad_mask = torch.ones(batch_size, seq_len, device=device)
        non_pad_mask[1, 5:] = 0

        output = model(x, non_pad_mask=non_pad_mask)

        assert output.shape == (batch_size, seq_len, d_input)
        # Check if masked positions are zeroed out
        assert torch.all(output[1, 5:] == 0)

    def test_fmhsa_high_dim():
        device = torch.device("cuda")
        extra_dim = 3
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 32
        d_hidden = 128

        model = FMHSA(
            training=True,
            n_head=n_head,
            d_input=d_input,
            d_qkv=d_qkv,
            device=device,
            d_hidden=d_hidden,
            dropout=0.1,
        )
        model.attn.flash_attn_qkvpacked_func = mock_fa_func

        x = torch.randn(extra_dim, batch_size, seq_len, d_input, device=device)
        non_pad_mask = torch.ones(extra_dim, batch_size, seq_len, device=device)

        output = model(x, non_pad_mask=non_pad_mask)

        assert output.shape == (extra_dim, batch_size, seq_len, d_input)

    def test_fmhsa_layer_high_dim():
        device = torch.device("cuda")
        extra_dim = 2
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 32

        layer = FMHSALayer(training=True, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.0)
        layer.flash_attn_qkvpacked_func = mock_fa_func

        x = torch.randn(extra_dim, batch_size, seq_len, d_input, device=device)
        mask = torch.ones(extra_dim, batch_size, seq_len, device=device)

        output = layer(x, mask=mask)

        assert output.shape == (extra_dim, batch_size, seq_len, d_input)

    print("Running tests...")
    test_fmhsa_layer_init()
    test_fmhsa_init()
    test_fmhsa_layer_forward()
    test_fmhsa_layer_high_dim()
    test_fmhsa_forward()
    test_fmhsa_with_masking()
    test_fmhsa_high_dim()
    print("All tests passed!")
