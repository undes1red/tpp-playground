import torch
import torch.nn as nn
from einops import rearrange
from flash_attn import flash_attn_qkvpacked_func

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
        """
        Template self-attention module with multihead-attention type 2: this module concatenates original outputs and
        compress high-dimensional vectors into d_input
        """
        super().__init__()
        self.training = training
        self.device = device
        if device == "cpu":
            raise ValueError("Flash Attention does not work on CPU.")

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
        self.layer_norm_for_output = nn.RMSNorm(
            self.d_input, eps=1e-6, device=self.device, dtype=torch.get_default_dtype()
        )

        if self.w_qkv.weight.dtype not in [torch.bfloat16, torch.float16]:
            raise ValueError("Flash Attention only supports bfloat16 and float16.")

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
            x = x.flatten(end_dim=-4)
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

        fa_output = flash_attn_qkvpacked_func(
            qkv, self.dropout, causal=True, deterministic=True
        )  # [..., nheads, d_qkv]
        fa_output = rearrange(fa_output, "...  nh dqkv -> ... (nh dqkv)")
        # [(... * batch_size), seq_len, n_head * d_qkv]

        fa_output = fa_output.view(*input_shape, -1)  # [..., batch_size, seq_len, n_head * d_qkv]
        fa_output = self.fc_attn_output(fa_output)  # [..., batch_size, seq_len, d_input]
        fa_output = fa_output * mask.unsqueeze(dim=-1)  # [..., batch_size, seq_len, d_input]
        fa_output += residual

        return self.layer_norm_for_output(fa_output)  # [..., batch_size, seq_len, d_output]


if __name__ == "__main__":
    from unittest.mock import patch

    torch.set_default_dtype(torch.bfloat16)

    def test_fmhsa_layer_init():
        device = torch.device("cuda")
        n_head = 4
        d_input = 64
        d_qkv = 16

        layer = FMHSALayer(training=True, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.1)

        assert layer.n_head == n_head
        assert layer.d_input == d_input
        assert layer.d_qkv == d_qkv
        assert layer.w_qkv.out_features == d_qkv * n_head * 3

    def test_fmhsa_init():
        device = torch.device("cuda")
        n_head = 4
        d_input = 64
        d_qkv = 16
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

    @patch("src.toolbox.modules.transformer.f_self_attn.flash_attn_varlen_qkvpacked_func")
    def test_fmhsa_layer_forward(mock_flash_attn):
        mock_flash_attn.side_effect = lambda qkv, cu_seqlens, max_seqlen, dropout, causal, deterministic: torch.randn(
            qkv.shape[0], qkv.shape[2], qkv.shape[3], device=qkv.device
        )

        device = torch.device("cuda")
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 16

        layer = FMHSALayer(training=True, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.0)

        x = torch.randn(batch_size, seq_len, d_input, device=device)
        mask = torch.ones(batch_size, seq_len, device=device)

        output = layer(x, mask=mask)

        assert output.shape == (batch_size, seq_len, d_input)

    @patch("src.toolbox.modules.transformer.f_self_attn.flash_attn_varlen_qkvpacked_func")
    def test_fmhsa_forward(mock_flash_attn):
        mock_flash_attn.side_effect = lambda qkv, cu_seqlens, max_seqlen, dropout, causal, deterministic: torch.randn(
            qkv.shape[0], qkv.shape[2], qkv.shape[3], device=qkv.device
        )

        device = torch.device("cuda")
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 16
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

        x = torch.randn(batch_size, seq_len, d_input, device=device)
        non_pad_mask = torch.ones(batch_size, seq_len, device=device)

        output = model(x, non_pad_mask=non_pad_mask)

        assert output.shape == (batch_size, seq_len, d_input)

    @patch("src.toolbox.modules.transformer.f_self_attn.flash_attn_varlen_qkvpacked_func")
    def test_fmhsa_with_masking(mock_flash_attn):
        mock_flash_attn.side_effect = lambda qkv, cu_seqlens, max_seqlen, dropout, causal, deterministic: torch.randn(
            qkv.shape[0], qkv.shape[2], qkv.shape[3], device=qkv.device
        )

        device = torch.device("cuda")
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 16
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

        x = torch.randn(batch_size, seq_len, d_input, device=device)
        # Mask out the last 5 elements of the second sequence
        non_pad_mask = torch.ones(batch_size, seq_len, device=device)
        non_pad_mask[1, 5:] = 0

        output = model(x, non_pad_mask=non_pad_mask)

        assert output.shape == (batch_size, seq_len, d_input)
        # Check if masked positions are zeroed out
        assert torch.all(output[1, 5:] == 0)

    @patch("src.toolbox.modules.transformer.f_self_attn.flash_attn_varlen_qkvpacked_func")
    def test_fmhsa_high_dim(mock_flash_attn):
        mock_flash_attn.side_effect = lambda qkv, cu_seqlens, max_seqlen, dropout, causal, deterministic: torch.randn(
            qkv.shape[0], qkv.shape[2], qkv.shape[3], device=qkv.device
        )

        device = torch.device("cuda")
        extra_dim = 3
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 16
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

        x = torch.randn(extra_dim, batch_size, seq_len, d_input, device=device)
        non_pad_mask = torch.ones(extra_dim, batch_size, seq_len, device=device)

        output = model(x, non_pad_mask=non_pad_mask)

        assert output.shape == (extra_dim, batch_size, seq_len, d_input)

    @patch("src.toolbox.modules.transformer.f_self_attn.flash_attn_varlen_qkvpacked_func")
    def test_fmhsa_layer_high_dim(mock_flash_attn):
        mock_flash_attn.side_effect = lambda qkv, cu_seqlens, max_seqlen, dropout, causal, deterministic: torch.randn(
            qkv.shape[0], qkv.shape[2], qkv.shape[3], device=qkv.device
        )

        device = torch.device("cuda")
        extra_dim = 2
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 16

        layer = FMHSALayer(training=True, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.0)

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
