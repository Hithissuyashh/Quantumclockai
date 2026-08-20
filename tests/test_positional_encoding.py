import torch

from ml.models.positional_encoding import PositionalEncoding

encoder = PositionalEncoding(
    d_model=64
)

x = torch.randn(
    32,
    128,
    64
)

y = encoder(x)

print("Input :", x.shape)
print("Output:", y.shape)

assert x.shape == y.shape

print("\n✓ Positional Encoding Works")