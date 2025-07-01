import torch
import torch.nn as nn

class ContrastiveHead(nn.Module):
    """Contrastive learning head for Siamese pretraining."""
    def __init__(self, embed_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # x: (B, E)
        return self.net(x)

class ClassificationHead(nn.Module):
    """Classification head for downstream tasks."""
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, E)
        return self.fc(x)
