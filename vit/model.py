import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────
# Vision‑Transformer classifier with the *same public API*
# (forward, compute_model_norm, compute_margin_distribution)
# as the CNNClassifier above.
# ──────────────────────────────────────────────────────────────
class ViTClassifier(nn.Module):
    """
    Minimal Vision Transformer for image classification that mirrors the
    external behaviour of `CNNClassifier`.

    Parameters
    ----------
    image_size : int
        Input resolution.  Assumed square (H = W).
    patch_size : int
        Size (in pixels) of each image patch (also the Conv2d kernel size).
    embed_dim  : int
        Token / channel dimension inside the transformer.
    depth      : int
        Number of Transformer encoder layers.
    num_heads  : int
        Multi‑head‑attention heads in each layer.
    mlp_ratio  : float
        Expansion factor in the feed‑forward block.
    num_classes : int
        Number of output classes.
    dropout, attn_dropout : float
        Dropout applied after attention / MLP.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        num_classes: int = 10,
    ):
        super().__init__()

        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        # 1. Patch‑embedding (conv acts as linear projection)
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 2. CLS token + learned positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # initialise
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    # ---------------------------------------------------------- #
    #  Forward
    # ---------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # (B, C, H, W) → (B, N, D)
        x = self.patch_embed(x)                       # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)              # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, D)
        x = torch.cat((cls, x), dim=1)                # prepend CLS
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(x)                           # Transformer
        x = self.norm(x[:, 0])                        # CLS token
        return self.head(x)                           # logits

    # ---------------------------------------------------------- #
    #  Norm & margin utilities (API compatible with CNNClassifier)
    # ---------------------------------------------------------- #
    @staticmethod
    def _spectral_norm(mat: torch.Tensor) -> torch.Tensor:
        if mat.ndim > 2:
            mat = mat.flatten(1)
        # power iteration (5 steps)
        with torch.no_grad():
            v = torch.randn(mat.size(1), device=mat.device)
            v = v / v.norm()
            for _ in range(5):
                v = torch.mv(mat.t(), torch.mv(mat, v))
                v = v / (v.norm() + 1e-12)
            sigma = torch.mv(mat, v).norm()
        return sigma

    @staticmethod
    def _two_one_norm(mat: torch.Tensor) -> torch.Tensor:
        if mat.ndim > 2:
            mat = mat.flatten(1)
        return torch.norm(mat, p=2, dim=0).sum()

    def compute_model_norm(self) -> torch.Tensor:
        """
        Bartlett–Foster–Telgarsky spectral complexity with reference M_i = 0
        (same formula used in CNNClassifier).
        """
        weights = [p for p in self.parameters() if p.dim() >= 2]

        if not weights:
            return torch.tensor(0.0, device=self.head.weight.device)

        sigmas = torch.stack([self._spectral_norm(w) for w in weights])
        l21s   = torch.stack([self._two_one_norm(w)  for w in weights])

        prod_sigma = sigmas.log().sum().exp()                   # ∏ ||A_i||_σ
        sum_term   = ((l21s / sigmas).pow(2.0 / 3.0)).sum()     # Σ (‖A_i‖₂,₁ / ‖A_i‖_σ)^{2/3}

        return prod_sigma * (sum_term ** 1.5)

    def compute_margin_distribution(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Identical semantics to CNNClassifier.compute_margin_distribution.
        """
        self.eval()
        with torch.no_grad():
            logits = self(inputs)                       # (n, C)
            batch_size = logits.size(0)

            true_scores = logits[torch.arange(batch_size), labels]
            tmp = logits.clone()
            tmp[torch.arange(batch_size), labels] = -float("inf")
            max_other = tmp.max(dim=1)[0]

            raw_margin = true_scores - max_other

            model_norm = self.compute_model_norm()

            X_flat = inputs.view(batch_size, -1)
            frob = torch.norm(X_flat, p=2)
            scale = model_norm * (frob / batch_size + 1e-12)

            return raw_margin / scale

    # ---------------------------------------------------------- #
    #  Helpers
    # ---------------------------------------------------------- #
    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
