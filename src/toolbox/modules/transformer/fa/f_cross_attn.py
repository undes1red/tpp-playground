import torch
import torch.nn as nn
from einops import rearrange
from flash_attn import flash_attn_func

from src.toolbox.modules.ffn import FFN


class FMHCA(nn.Module):
    def __init__(self, training, n_head, d_input, d_qkv, device, d_hidden, dropout=0.1):
        super().__init__()
        self.training = training
        self.device = device

        self.attn = FMHCALayer(
            training=training, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=self.device, dropout=dropout
        )
        self.ffn = FFN(d_input=d_input, d_hidden=d_hidden, device=self.device, dropout=dropout)

    def forward(self, q, k, v, non_pad_mask=None):
        """
        Args:
        1. x: input tensor. shape: [batch_size, seq_len, d_input]
        2. non_pad_mask: mask tensor for used by self attention and to mask out pad items. shape: [batch_size, seq_len]
        Outputs:
        1. output: results of transformer layer. shape: [batch_size, seq_len, d_input]
        """
        output = self.attn(q, k, v)
        # [..., batch_size, seq_len, d_input]

        output = self.ffn(output)  # [..., batch_size, seq_len, d_input]

        if non_pad_mask is not None:
            output *= rearrange(non_pad_mask, "... -> ... 1")  # [..., batch_size, seq_len, d_input]

        return output


class FMHCALayer(nn.Module):
    def __init__(self, training, n_head, d_input, d_qkv, device, dropout=0.1):
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
        self.w_q = nn.Linear(d_input, self.d_qkv * self.n_head, bias=True, device=self.device)
        self.w_k = nn.Linear(d_input, self.d_qkv * self.n_head, bias=True, device=self.device)
        self.w_v = nn.Linear(d_input, self.d_qkv * self.n_head, bias=True, device=self.device)

        # Linear: n_head * d_q, d_k, or d_v -> d_input
        self.fc_attn_output = nn.Linear(self.n_head * d_qkv, self.d_input, bias=True, device=self.device)

        # layer normalization
        self.layer_norm = nn.RMSNorm(self.d_input, eps=1e-6, device=self.device, dtype=torch.get_default_dtype())

    def forward(self, q, k, v):
        """
        Args:
        1. q, k, v: input tensor. shape: [batch_size, seq_len, d_input]
        2. mask: the mask tensor used by self attention. shape: [batch_size, seq_len]
        Output:
        1. output: results of transformer layer. shape: [batch_size, seq_len, d_input]
        """

        residual = q
        q = self.layer_norm(q)  # [batch_size, seq_len, d_input]
        k = self.layer_norm(k)  # [batch_size, seq_len, d_input]
        v = self.layer_norm(v)  # [batch_size, seq_len, d_input]

        q_ = self.w_q(q)  # [batch_size, seq_len, d_qkv * n_head]
        k_ = self.w_k(k)  # [batch_size, seq_len, d_qkv * n_head]
        v_ = self.w_v(v)  # [batch_size, seq_len, d_qkv * n_head]

        q_ = rearrange(q_, 'b s (head dqkv) -> b s head dqkv', head=self.n_head)
        k_ = rearrange(k_, 'b s (head dqkv) -> b s head dqkv', head=self.n_head)
        v_ = rearrange(v_, 'b s (head dqkv) -> b s head dqkv', head=self.n_head)

        fa_output = flash_attn_func(
            q, k, v, dropout_p=self.dropout, causal=True, deterministic=True
        )  # [batch_size, seq_len, nheads, d_qkv]
        fa_output = rearrange(fa_output, "...  nh dqkv -> ... (nh dqkv)")
        # [batch_size, seq_len, n_head * d_qkv]

        fa_output = self.fc_attn_output(fa_output)  # [batch_size, seq_len, d_input]
        fa_output += residual # [batch_size, seq_len, d_input]

        return fa_output  # [batch_size, seq_len, d_output]


if __name__ == "__main__":
    from unittest.mock import patch

    torch.set_default_dtype(torch.bfloat16)

    def test_fmhca_layer_init():
        device = torch.device("cuda")
        n_head = 4
        d_input = 64
        d_qkv = 16

        layer = FMHCALayer(training=True, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.1)

        assert layer.n_head == n_head
        assert layer.d_input == d_input
        assert layer.d_qkv == d_qkv
        assert layer.w_q.out_features == d_qkv * n_head

    def test_fmhca_init():
        device = torch.device("cuda")
        n_head = 4
        d_input = 64
        d_qkv = 16
        d_hidden = 128

        model = FMHCA(
            training=True,
            n_head=n_head,
            d_input=d_input,
            d_qkv=d_qkv,
            device=device,
            d_hidden=d_hidden,
            dropout=0.1,
        )
        assert isinstance(model.attn, FMHCALayer)
        assert model.attn.d_qkv == d_qkv

    @patch("src.toolbox.modules.transformer.fa.f_cross_attn.flash_attn_func")
    def test_fmhca_layer_forward(mock_flash_attn):
        mock_flash_attn.side_effect = lambda q, k, v, dropout_p, causal, deterministic: torch.randn(
            q.shape[0], q.shape[1], q.shape[2], q.shape[3], device=q.device
        )

        device = torch.device("cuda")
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 16

        layer = FMHCALayer(training=True, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.0)

        q = torch.randn(batch_size, seq_len, d_input, device=device)
        k = torch.randn(batch_size, seq_len, d_input, device=device)
        v = torch.randn(batch_size, seq_len, d_input, device=device)

        output = layer(q, k, v)

        assert output.shape == (batch_size, seq_len, d_input)

    @patch("src.toolbox.modules.transformer.fa.f_cross_attn.flash_attn_func")
    def test_fmhca_forward(mock_flash_attn):
        mock_flash_attn.side_effect = lambda q, k, v, dropout_p, causal, deterministic: torch.randn(
            q.shape[0], q.shape[1], q.shape[2], q.shape[3], device=q.device
        )

        device = torch.device("cuda")
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 16
        d_hidden = 128

        model = FMHCA(
            training=True,
            n_head=n_head,
            d_input=d_input,
            d_qkv=d_qkv,
            device=device,
            d_hidden=d_hidden,
            dropout=0.1,
        )

        q = torch.randn(batch_size, seq_len, d_input, device=device)
        k = torch.randn(batch_size, seq_len, d_input, device=device)
        v = torch.randn(batch_size, seq_len, d_input, device=device)
        non_pad_mask = torch.ones(batch_size, seq_len, device=device)

        output = model(q, k, v, non_pad_mask=non_pad_mask)

        assert output.shape == (batch_size, seq_len, d_input)

    @patch("src.toolbox.modules.transformer.fa.f_cross_attn.flash_attn_func")
    def test_fmhca_with_masking(mock_flash_attn):
        mock_flash_attn.side_effect = lambda q, k, v, dropout_p, causal, deterministic: torch.randn(
            q.shape[0], q.shape[1], q.shape[2], q.shape[3], device=q.device
        )

        device = torch.device("cuda")
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 16
        d_hidden = 128

        model = FMHCA(
            training=True,
            n_head=n_head,
            d_input=d_input,
            d_qkv=d_qkv,
            device=device,
            d_hidden=d_hidden,
            dropout=0.1,
        )

        q = torch.randn(batch_size, seq_len, d_input, device=device)
        k = torch.randn(batch_size, seq_len, d_input, device=device)
        v = torch.randn(batch_size, seq_len, d_input, device=device)
        # Mask out the last 5 elements of the second sequence
        non_pad_mask = torch.ones(batch_size, seq_len, device=device)
        non_pad_mask[1, 5:] = 0

        output = model(q, k, v, non_pad_mask=non_pad_mask)

        assert output.shape == (batch_size, seq_len, d_input)
        # Check if masked positions are zeroed out
        assert torch.all(output[1, 5:] == 0)

    @patch("src.toolbox.modules.transformer.fa.f_cross_attn.flash_attn_func")
    def test_fmhca_high_dim(mock_flash_attn):
        mock_flash_attn.side_effect = lambda q, k, v, dropout_p, causal, deterministic: torch.randn(
            q.shape[0], q.shape[1], q.shape[2], q.shape[3], device=q.device
        )

        device = torch.device("cuda")
        extra_dim = 3
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 16
        d_hidden = 128

        model = FMHCA(
            training=True,
            n_head=n_head,
            d_input=d_input,
            d_qkv=d_qkv,
            device=device,
            d_hidden=d_hidden,
            dropout=0.1,
        )

        q = torch.randn(extra_dim, batch_size, seq_len, d_input, device=device)
        k = torch.randn(extra_dim, batch_size, seq_len, d_input, device=device)
        v = torch.randn(extra_dim, batch_size, seq_len, d_input, device=device)
        non_pad_mask = torch.ones(extra_dim, batch_size, seq_len, device=device)

        output = model(q, k, v, non_pad_mask=non_pad_mask)

        assert output.shape == (extra_dim, batch_size, seq_len, d_input)

    @patch("src.toolbox.modules.transformer.fa.f_cross_attn.flash_attn_func")
    def test_fmhca_layer_high_dim(mock_flash_attn):
        mock_flash_attn.side_effect = lambda q, k, v, dropout_p, causal, deterministic: torch.randn(
            q.shape[0], q.shape[1], q.shape[2], q.shape[3], device=q.device
        )

        device = torch.device("cuda")
        extra_dim = 2
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 16

        layer = FMHCALayer(training=True, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.0)

        q = torch.randn(extra_dim, batch_size, seq_len, d_input, device=device)
        k = torch.randn(extra_dim, batch_size, seq_len, d_input, device=device)
        v = torch.randn(extra_dim, batch_size, seq_len, d_input, device=device)

        output = layer(q, k, v)

        assert output.shape == (extra_dim, batch_size, seq_len, d_input)

    print("Running tests...")
    test_fmhca_layer_init()
    test_fmhca_init()
    test_fmhca_layer_forward()
    test_fmhca_layer_high_dim()
    test_fmhca_forward()
    test_fmhca_with_masking()
    test_fmhca_high_dim()
    print("All tests passed!")
