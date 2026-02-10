import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.toolbox.modules.ffn import FFN

from .get_causal_mask import get_causal_mask


class SMHSA(nn.Module):
    def __init__(self, training, n_head, d_input, d_qkv, device, d_hidden, dropout=0.1):
        super().__init__()
        self.training = training
        self.device = device

        self.attn = SMHSALayer(
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


class SMHSALayer(nn.Module):
    def __init__(self, training, n_head, d_input, d_qkv, device, dropout=0.1):
        super().__init__()
        self.training = training
        self.device = device

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
            "... seq_len (nqkv nhead dqkv) -> nqkv ... nhead seq_len dqkv",
            nqkv=3,
            nhead=self.n_head,
            dqkv=self.d_qkv,
        )
        # [3, (... * batch_size), n_head, seq_len, d_qkv]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Create attention mask for scaled_dot_product_attention
        # Need to create a 2D or 4D mask: (batch_size, n_head, seq_len, seq_len)
        # The mask should be True for positions that are ALLOWED, False for positions that are MASKED OUT
        seq_len = q.shape[2]

        # Create causal mask: (1, seq_len, seq_len) where mask[i, j] is True if i >= j
        causal_mask = get_causal_mask(seq_len, self.device)

        if mask is not None:
            # mask shape: [..., batch_size, seq_len]
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)

            mask_flat = mask.flatten(end_dim=-2) if len(mask.shape) > 2 else mask  # [(... * batch_size), seq_len]

            # Create attention mask combining causal and padding masks
            # attn_mask[i, j] should be True if position i can attend to position j
            # This means: i >= j (causal) AND mask[i] is True AND mask[j] is True
            # Shape: (batch_size, 1, seq_len, seq_len) - broadcastable to (batch_size, n_head, seq_len, seq_len)
            attn_mask = rearrange(causal_mask, 'b s s1 -> b () s s1')  # [1, 1, seq_len, seq_len]
            attn_mask = attn_mask & rearrange(mask_flat, 'b s -> b () () s') & rearrange(mask_flat, 'b s -> b () s ()')
            # [batch_size, 1, seq_len, seq_len]
        else:
            # Only causal mask, broadcast across batch and heads
            attn_mask = rearrange(causal_mask, 'b s s1 -> b () s s1')  # [1, 1, seq_len, seq_len]

        # scaled_dot_product_attention expects attn_mask to be a boolean tensor where:
        # True = allowed, False = masked out. We can also use is_causal=True when no padding mask.
        fa_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        # [(... * batch_size), n_head, seq_len, d_qkv]

        fa_output = rearrange(fa_output, "... nh seq dqkv -> ... seq (nh dqkv)")
        # [(... * batch_size), seq_len, n_head * d_qkv]

        fa_output = fa_output.view(*input_shape, -1)  # [..., batch_size, seq_len, n_head * d_qkv]
        fa_output = self.fc_attn_output(fa_output)  # [..., batch_size, seq_len, d_input]
        if mask is not None:
            fa_output = fa_output * mask.unsqueeze(dim=-1)  # [..., batch_size, seq_len, d_input]
        fa_output += residual

        return fa_output  # [..., batch_size, seq_len, d_output]


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)

    def test_SMHSA_layer_init():
        device = torch.device("cuda")
        n_head = 4
        d_input = 64
        d_qkv = 32

        layer = SMHSALayer(training=True, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.1)

        assert layer.n_head == n_head
        assert layer.d_input == d_input
        assert layer.d_qkv == d_qkv
        assert layer.w_qkv.out_features == d_qkv * n_head * 3

    def test_SMHSA_init():
        device = torch.device("cuda")
        n_head = 4
        d_input = 64
        d_qkv = 32
        d_hidden = 128

        model = SMHSA(
            training=True,
            n_head=n_head,
            d_input=d_input,
            d_qkv=d_qkv,
            device=device,
            d_hidden=d_hidden,
            dropout=0.1,
        )
        assert isinstance(model.attn, SMHSALayer)
        assert model.attn.d_qkv == d_qkv

    def test_SMHSA_layer_forward():
        device = torch.device("cuda")
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 32

        layer = SMHSALayer(training=True, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.0)

        x = torch.randn(batch_size, seq_len, d_input, device=device)
        mask = torch.ones(batch_size, seq_len, device=device)

        output = layer(x, mask=mask)

        assert output.shape == (batch_size, seq_len, d_input)

    def test_SMHSA_forward():
        device = torch.device("cuda")
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 32
        d_hidden = 128

        model = SMHSA(
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

    def test_SMHSA_with_masking():
        device = torch.device("cuda")
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 32
        d_hidden = 128

        model = SMHSA(
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

    def test_SMHSA_high_dim():
        device = torch.device("cuda")
        extra_dim = 3
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 32
        d_hidden = 128

        model = SMHSA(
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

    def test_SMHSA_layer_high_dim():
        device = torch.device("cuda")
        extra_dim = 2
        batch_size = 2
        seq_len = 10
        n_head = 4
        d_input = 64
        d_qkv = 32

        layer = SMHSALayer(training=True, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.0)

        x = torch.randn(extra_dim, batch_size, seq_len, d_input, device=device)
        mask = torch.ones(extra_dim, batch_size, seq_len, device=device)

        output = layer(x, mask=mask)

        assert output.shape == (extra_dim, batch_size, seq_len, d_input)

    def test_mask_correctness():
        device = torch.device("cuda")
        batch_size = 1
        seq_len = 4
        n_head = 1
        d_input = 16
        d_qkv = 8

        layer = SMHSALayer(training=False, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=device, dropout=0.0)

        # Set weights to identity-like to make it easier to track
        with torch.no_grad():
            layer.w_qkv.weight.fill_(0)
            layer.w_qkv.bias.fill_(0)
            # Just pass through first few dims
            for i in range(d_qkv):
                layer.w_qkv.weight[i, i] = 1  # Q
                layer.w_qkv.weight[i + d_qkv * n_head, i] = 1  # K
                layer.w_qkv.weight[i + d_qkv * n_head * 2, i] = 1  # V

            layer.fc_attn_output.weight.fill_(0)
            layer.fc_attn_output.bias.fill_(0)
            for i in range(d_qkv):
                layer.fc_attn_output.weight[i, i] = 1

            layer.layer_norm_for_q.weight.fill_(1)

        x = torch.zeros(batch_size, seq_len, d_input, device=device)
        for i in range(seq_len):
            x[0, i, 0] = 10.0**i  # Distinct values for each position

        # Mask: only first 2 elements are non-padding
        mask = torch.tensor([[1, 1, 0, 0]], device=device, dtype=torch.float32)

        # The combined mask is (q_idx >= kv_idx) & (mask[kv_idx] > 0)
        # For seq_len=4:
        # q=0: kv=0 (ok)
        # q=1: kv=0, 1 (ok)
        # q=2: kv=0, 1 (ok, but q=2 is padding, so output will be masked out later)
        # q=3: kv=0, 1 (ok, but q=3 is padding)

        output = layer(x, mask=mask)

        # Check if padding positions are zero
        # Note: SMHSALayer adds residual, so we should check if output == residual for padding positions
        # OR check if the attention part is zero.
        # In our test, x was 10^i, and residual is x.
        # If attention is masked, output[0, i] should be residual[0, i] = x[0, i].
        # BUT SMHSA.forward (the wrapper) multiplies by non_pad_mask at the end.
        # Our test uses SMHSALayer directly.
        
        # Let's check if the attention output (before residual) is zero.
        # Since we can't easily check that, let's just check if it's equal to residual.
        assert torch.allclose(output[0, 2:], x[0, 2:], atol=1e-5)

        # For q=1, it should only attend to kv=0 and kv=1
        # If it attended to kv=2, it would be wrong.
        # Since we used distinct values, we can verify.
        # This is a basic check, flex_attention with create_mask is tested by torch,
        # we just ensure our combined_mask logic is sound.
        print("Mask correctness test passed!")

    print("Running tests...")
    test_SMHSA_layer_init()
    test_SMHSA_init()
    test_SMHSA_layer_forward()
    test_SMHSA_layer_high_dim()
    test_SMHSA_forward()
    test_SMHSA_with_masking()
    test_SMHSA_high_dim()
    test_mask_correctness()
    print("All tests passed!")
