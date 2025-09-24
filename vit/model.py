import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- building blocks --------------------------------------------------
class TransformerBlock(nn.Module):
    """
    Single ViT block: pre-LN → MHSA → residual → pre-LN → GELU-MLP → residual
    """
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                           batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


# ---------- ViT classifier ----------------------------------------------------
class ViT(nn.Module):
    """
    Vision Transformer with controllable overall init-scale.
    """
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 num_classes: int = 1000,
                 dropout: float = 0.0,
                 initialization_factor: float = 1.0,  # ← new
                 ):
        super().__init__()
        assert image_size % patch_size == 0, "patch size must divide image size"

        self.initialization_factor = initialization_factor

        # patch stem
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)

        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop   = nn.Dropout(dropout)

        # transformer stack
        mlp_dim = int(embed_dim * mlp_ratio)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm   = nn.LayerNorm(embed_dim)

        # head
        self.head = nn.Linear(embed_dim, num_classes)

        # now do our custom init
        self._init_weights()

    def _init_weights(self):
        # 1) reasonable default scales
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 2) per-module default init, then scale
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming normal on convs
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

                # scale
                m.weight.data.mul_(self.initialization_factor)

            elif isinstance(m, nn.Linear):
                # Xavier uniform on linears
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

                # scale
                m.weight.data.mul_(self.initialization_factor)

            elif isinstance(m, nn.LayerNorm):
                # standard LayerNorm init
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)         # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])

    # ----------------------- complexity (spectral norm) -----------------------
    # ------------------------------------------------------------------ #
    # 64‑bit helpers – used *only* inside compute_model_norm()           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _spectral_norm(w: torch.Tensor) -> torch.Tensor:
        """Largest singular value ‖A‖σ computed in float64 on the same device."""
        w64 = w.detach().to(dtype=torch.float64)           # ➊ promote to fp64
        w2d = w64.view(w64.size(0), -1) if w64.ndim > 2 else w64
        return torch.linalg.svdvals(w2d)[0]                # dtype = float64

    @staticmethod
    def _two_one_norm_transpose(w: torch.Tensor) -> torch.Tensor:
        """(2,1)‑norm of Aᵀ, also in float64."""
        w64 = w.detach().to(dtype=torch.float64)
        w2d = w64.view(w64.size(0), -1) if w64.ndim > 2 else w64
        col_l2 = torch.norm(w2d.T, p=2, dim=0)
        return col_l2.sum()                                # float64

    @torch.no_grad()
    def compute_linear_model_norm(self) -> torch.Tensor:
        """
        Bartlett‑style spectral complexity R_ViT.
        Internally uses float64 to avoid overflow, then casts the *result*
        back to the model’s original dtype (usually float32).
        """
        device = next(self.parameters()).device
        dtype_out = next(self.parameters()).dtype          # fp32 or fp16

        specs, two1s = [], []
        for W in self._all_weight_matrices():
            specs.append(self._spectral_norm(W))
            two1s.append(self._two_one_norm_transpose(W))

        prod_spec = torch.prod(torch.stack(specs))                          # fp64
        correction = sum((t**(2/3)) / (s**(2/3) + 1e-12)
                         for t, s in zip(two1s, specs))                    # fp64
        R = prod_spec * (correction ** 1.5)                                # fp64
        return R.to(dtype=dtype_out)            

    # ------------------------------------------------------------------ #
    #  log‑space   spectral complexity  (never overflows)                #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_model_norm(self) -> torch.Tensor:
        """
        Returns log(R) in natural base; works even when R would overflow.
        """
        logs, terms = [], []              # float64 throughout
        for W in self._all_weight_matrices():
            s  = self._spectral_norm(W)            # float64
            t  = self._two_one_norm_transpose(W)
            logs.append(torch.log(s))              # log s_i
            terms.append((t ** (2/3)) / (s ** (2/3) + 1e-12))

        log_prod = torch.stack(logs).sum()         # Σ log s_i
        log_corr = torch.log( torch.stack(terms).sum() + 1e-12 )
        return log_prod + 1.5 * log_corr           # log R

    def _all_weight_matrices(self):
        """
        Yield *individual* linear operators:
          • patch embedding (Conv2d)
          • Q, K, V, O of every MHSA block
          • two MLP layers per block
          • final head
        """
        # 1) patch embed
        yield self.patch_embed.weight

        # 2) transformer blocks
        for blk in self.blocks:
            # Multi‑head Attention: in_proj_weight is (3D, D) → split
            Wqkv = blk.attn.in_proj_weight   # shape (3*D, D)
            D = Wqkv.size(1)
            for mat in torch.chunk(Wqkv, 3, dim=0):  # Q, K, V
                yield mat
            yield blk.attn.out_proj.weight   # O
            # Feed‑forward
            for layer in blk.mlp:
                if isinstance(layer, nn.Linear):
                    yield layer.weight

        # 3) classification head
        yield self.head.weight

    @torch.no_grad()
    def compute_l1_norm(self) -> torch.Tensor:
        """Log of entry-wise L1 norm over all weight matrices."""
        total = torch.tensor(0., dtype=torch.float64, device=next(self.parameters()).device)
        for W in self._all_weight_matrices():
            total += torch.sum(W.detach().to(torch.float64).abs())
        return torch.log(total + 1e-12)

    @torch.no_grad()
    def compute_frobenius_norm(self) -> torch.Tensor:
        """Log of Frobenius norm over all weight matrices."""
        total_sq = torch.tensor(0., dtype=torch.float64, device=next(self.parameters()).device)
        for W in self._all_weight_matrices():
            total_sq += torch.sum(W.detach().to(torch.float64)**2)
        return 0.5 * torch.log(total_sq + 1e-12)

    @torch.no_grad()
    def compute_group_2_1_norm(self) -> torch.Tensor:
        """Log of (2,1)-group norm (sum of column L2 norms) over all weight matrices."""
        total = torch.tensor(0., dtype=torch.float64, device=next(self.parameters()).device)
        for W in self._all_weight_matrices():
            w2d = W.detach().to(torch.float64).view(W.size(0), -1) if W.ndim > 2 else W.detach().to(torch.float64)
            total += torch.norm(w2d, p=2, dim=0).sum()
        return torch.log(total + 1e-12)

    @torch.no_grad()
    def compute_spectral_norm(self) -> torch.Tensor:
        """Log of product of spectral norms (sum of log singular values)."""
        logs = []
        for W in self._all_weight_matrices():
            s = self._spectral_norm(W)
            logs.append(torch.log(s))
        return torch.stack(logs).sum()


    # ------------------------ margin distribution -----------------------------
    @torch.no_grad()
    def compute_margin_distribution(self, inputs, labels):
        """
        Normalised margins  m̃(x,y) = m(x,y) / [ R * (‖X‖_F / n) ]
        where m(x,y)=f(x)_y − max_{j≠y} f(x)_j.
        """
        self.eval()
        logits = self(inputs)                    # (n, C)
        n = logits.size(0)

        true_scores = logits[torch.arange(n), labels]
        logits_ = logits.clone()
        logits_[torch.arange(n), labels] = float('-inf')
        max_others = logits_.max(dim=1).values
        raw_margin = true_scores - max_others

        R = self.compute_model_norm()
        X_flat = inputs.view(n, -1)
        scale = R * (torch.norm(X_flat, p=2) / n + 1e-12)
        return raw_margin / scale
