import torch
import torch.nn as nn

class SpectralTokenPruner(nn.Module):
    """
    Evaluates time-frequency spectrogram tokens and prunes low-information
    silent/background patches before Multi-Head Self-Attention.
    """
    def __init__(self, keep_ratio: float = 0.6):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.keep_ratio >= 1.0:
            return x

        cls_token = x[:, :1, :]       # Always retain [CLS] token: [B, 1, D]
        patch_tokens = x[:, 1:, :]    # Time-frequency patches: [B, N, D]
        B, N, D = patch_tokens.shape
        k = max(1, int(N * self.keep_ratio))

        # Hybrid Spectral Importance Score: L2 Energy + Feature Variance
        l2_energy = torch.norm(patch_tokens, p=2, dim=-1)
        spectral_var = torch.var(patch_tokens, dim=-1)
        importance_score = l2_energy + spectral_var  # [B, N]

        # Keep top-K informative acoustic tokens per clip
        _, topk_indices = torch.topk(importance_score, k=k, dim=1, largest=True, sorted=True)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        pruned_patches = torch.gather(patch_tokens, dim=1, index=topk_indices_expanded)

        return torch.cat([cls_token, pruned_patches], dim=1)  # [B, 1 + k, D]


class AudioSpectrogramTransformer(nn.Module):
    """
    Audio Spectrogram Transformer (AST) with configurable Spectral Token Pruning.
    Input shape: [Batch, 1, 64 (Mel bins), 200 (Time frames)]
    Patch size: (8, 10) -> 8 * 20 = 160 time-frequency patches.
    """
    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        keep_ratio: float = 0.6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(8, 10),
            stride=(8, 10)
        )
        num_patches = (64 // 8) * (200 // 10)  # 160 patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim) * 0.02)

        self.pruner = SpectralTokenPruner(keep_ratio=keep_ratio)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth,enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.pruner(x)

        x = self.transformer(x)
        cls_out = self.norm(x[:, 0])
        return self.head(cls_out)