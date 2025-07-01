import torch
import torch.nn as nn

class ParallelAdapter(nn.Module):
    """Parallel adapter inserted after FFN modules."""
    def __init__(self, embed_dim, adapter_rank=32):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, adapter_rank)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_rank, embed_dim)

    def forward(self, x):
        # x: (B, N, E)
        z = self.down_proj(x)
        z = self.activation(z)
        z = self.up_proj(z)
        return x + z  # residual parallel connection
