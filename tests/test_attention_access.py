import torch

from ml.config import CHECKPOINT_DIR
from ml.models.transformer_model import QuantumTransformer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device : {device}")

# -------------------------------------------------
# Load model
# -------------------------------------------------

model = QuantumTransformer().to(device)

checkpoint = torch.load(
    CHECKPOINT_DIR / "best_model.pth",
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# -------------------------------------------------
# Inspect Transformer architecture
# -------------------------------------------------

print("\nTransformer attention modules")
print("=" * 60)

for name, module in model.named_modules():

    if isinstance(
        module,
        torch.nn.MultiheadAttention
    ):

        print(
            f"{name} | "
            f"embed_dim={module.embed_dim} | "
            f"heads={module.num_heads}"
        )

print("\n[PASS] Attention modules located.")