import torch
import torch.nn as nn

class PatchEmbed3D(nn.Module):
    """Embed 3D patches from spatio-temporal input."""
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, E, t, h, w)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x

class ViT3D(nn.Module):
    """Spatio-temporal Vision Transformer backbone."""
    def __init__(
        self,
        in_channels=3,
        embed_dim=1024,
        depth=12,
        num_heads=16,
        patch_size=(2,16,16),
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None  # init in forward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B = x.size(0)
        x = self.patch_embed(x)      # (B, N, E)
        N = x.size(1)
        if self.pos_embed is None or self.pos_embed.size(1) != N + 1:
            self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, x.size(2)))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.norm(x[:, 0])     # return CLS embedding
