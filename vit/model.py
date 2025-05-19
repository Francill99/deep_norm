import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- building blocks --------------------------------------------------
class TransformerBlock(nn.Module):
    """
    Single ViT block: LN → MHSA → residual → LN → GELU‑MLP → residual
    """
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads,
                                          batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Multi‑head self‑attention
        y = self.norm1(x)
        attn, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn                        # first residual
        # Feed‑forward
        y = self.norm2(x)
        x = x + self.mlp(y)                # second residual
        return x

# ---------- ViT classifier ----------------------------------------------------
class ViT(nn.Module):
    """
    Vision Transformer with Bartlett‑style spectral complexity + margin utils.
    """
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, "patch size must divide image size"
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        num_patches = (image_size // patch_size) ** 2

        # class token & positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # transformer encoder
        mlp_dim = int(embed_dim * mlp_ratio)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # classification head
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    # ---------------------------- forward -------------------------------------
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                 # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)        # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])               # class token only

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
