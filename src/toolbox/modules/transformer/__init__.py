from .self_attn.flex_self_attn import FMHSA
from .self_attn.get_causal_mask import get_causal_mask
from .self_attn.plain_self_attn import TransformerLayer
from .self_attn.scaled_self_attn import SMHSA

__all__ = [
    "FMHSA",
    "FMHCA",
    "SMHSA",
    "TransformerLayer",
    "get_causal_mask",
]


if __name__ == "__main__":
    # Test equivalence between TransformerLayer and FMHSA
    # Run: python -m src.toolbox.modules.transformer or uv run -m src.toolbox.modules.transformer
    from .__main__ import test_transformer_equivalence, test_transformer_with_padding

    print("=" * 70)
    print("TRANSFORMER EQUIVALENCE TEST SUITE")
    print("=" * 70)

    success1 = test_transformer_equivalence()
    success2 = test_transformer_with_padding()

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    if success1 and success2:
        print("✓ All equivalence tests passed!")
        exit(0)
    else:
        print("✗ Some tests failed!")
        exit(1)
