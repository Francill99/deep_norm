import torch
import torch.nn as nn
from typing import List, Type, Tuple, Optional

__all__ = ["BasicBlock", "ResNetClassifier"]

# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Standard 3×3 residual block for ResNet‑18/34.

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    planes : int
        Number of output channels of the first conv layer (second conv will match).
    stride : int, optional (default=1)
        Stride of the *first* convolution (handles down‑sampling).
    norm_layer : Callable[..., nn.Module], optional
        Normalisation layer constructor (default: nn.BatchNorm2d).
    dropout : float, optional (default=0.0)
        Drop prob applied after first ReLU (helps Tiny‑ImageNet).
    """

    expansion: int = 1  # multiplier for planes when computing out_planes

    def __init__(
        self,
        in_planes: int,
        planes: int,
        *,
        stride: int = 1,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        # Projection for the residual path when we change spatial dims / channels
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity  # residual add (1‑Lipschitz)
        return self.relu(out)


# -----------------------------------------------------------------------------
# ResNet for CIFAR‑10/100, Tiny‑ImageNet & MNIST
# -----------------------------------------------------------------------------

class ResNetClassifier(nn.Module):
    """Flexible ResNet classifier with initialization scaling.

    All architectural hyper‑parameters are passed via ``__init__`` so the model
    can be instantiated in sweeps or loaded from a config file.

    Parameters
    ----------
    block : nn.Module class
        Residual block class (e.g. BasicBlock)
    layers : list of int
        Number of blocks in each stage
    widths : list of int
        Number of output channels per stage
    num_classes : int
        Number of output classes
    in_channels : int
        Number of input channels (e.g. 3 for RGB, 1 for grayscale)
    norm_layer : Callable[..., nn.Module]
        Normalization layer constructor
    dropout : float
        Drop probability inside residual blocks
    initialization_factor : float
        Global factor to multiply all weights after default initialization
    """

    def __init__(
        self,
        *,
        block: Type[nn.Module] = BasicBlock,
        layers: List[int] = [2, 2, 2, 2],
        widths: List[int] = [64, 128, 256, 512],
        num_classes: int = 10,
        in_channels: int = 3,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        dropout: float = 0.0,
        initialization_factor: float = 1.0,
    ) -> None:
        super().__init__()
        assert len(layers) == len(widths), "`layers` and `widths` must be equal‑length lists"

        self._init_factor = initialization_factor
        self._block = block
        self._norm_layer = norm_layer

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(widths[0]),
            nn.ReLU(inplace=True),
        )

        self.in_planes = widths[0]
        self.stages = nn.ModuleList()
        strides = [1, 2, 2, 2]
        for num_blocks, planes, stride in zip(layers, widths, strides):
            stage = self._make_stage(block, planes, num_blocks, stride, norm_layer, dropout)
            self.stages.append(stage)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1] * block.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if hasattr(m, 'weight'):
                    m.weight.data.mul_(self._init_factor)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                m.weight.data.mul_(self._init_factor)
                nn.init.zeros_(m.bias)

        # Cache for norm computations
        self._weighted_layers = [self.stem[0]]
        self._weighted_layers += [m for stage in self.stages for m in stage if isinstance(m, nn.Conv2d)]
        self._weighted_layers.append(self.fc)

    def _make_stage(
        self,
        block: Type[nn.Module],
        planes: int,
        blocks: int,
        stride: int,
        norm_layer: Type[nn.Module],
        dropout: float,
    ) -> nn.Sequential:
        layers = [block(self.in_planes, planes, stride=stride, norm_layer=norm_layer, dropout=dropout)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, norm_layer=norm_layer, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
    
    @staticmethod
    def _spectral_norm(weight: torch.Tensor) -> torch.Tensor:
        w = weight.view(weight.size(0), -1) if weight.ndim > 2 else weight
        return torch.linalg.svdvals(w)[0]

    @staticmethod
    def _two_one_norm(weight: torch.Tensor) -> torch.Tensor:
        w = weight.view(weight.size(0), -1) if weight.ndim > 2 else weight
        return torch.norm(w, p=2, dim=0).sum()


    def compute_model_norm(self) -> torch.Tensor:
        device = next(self.parameters()).device
        s_list, t_list = [], []
        for m in self._weighted_layers:
            wt = m.weight.detach()
            s_list.append(self._spectral_norm(wt))
            t_list.append(self._two_one_norm(wt))
        prod_spec = torch.prod(torch.stack(s_list))
        terms = []
        L = len(s_list)
        for i in range(L):
            t_sum = torch.sum(torch.stack(t_list[i+1:])) if i < L-1 else torch.tensor(0.0, device=device)
            s_i = s_list[i].clamp_min(1e-12)
            terms.append((t_sum**(2/3)) / (s_i**(2/3)))
        return prod_spec * (torch.sum(torch.stack(terms))**(3/2))

    def compute_margin_distribution(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            outputs = self(inputs)
            bs = outputs.size(0)
            true_scores = outputs[torch.arange(bs), labels]
            clone = outputs.clone()
            clone[torch.arange(bs), labels] = float('-inf')
            max_wrong = clone.max(1)[0]
            raw = true_scores - max_wrong
            norm = self.compute_model_norm()
            Xf = inputs.view(bs, -1)
            Xn = torch.norm(Xf, p=2)
            return raw / (norm * (Xn/bs + 1e-12))
        
    def compute_l1_norm(self) -> torch.Tensor:
        """Sum of absolute values of all weights."""
        total = torch.tensor(0., device=next(self.parameters()).device)
        for param in self.parameters():
            total += param.abs().sum()
        return total

    def compute_frobenius_norm(self) -> torch.Tensor:
        """Square root of sum of squares of all weights."""
        total_sq = torch.tensor(0., device=next(self.parameters()).device)
        for param in self.parameters():
            total_sq += torch.sum(param**2)
        return torch.sqrt(total_sq)

    def compute_group_2_1_norm(self) -> torch.Tensor:
        """Sum of L2 norms of each column of each weight matrix."""
        total = torch.tensor(0., device=next(self.parameters()).device)
        for m in self._weighted_layers:
            w = m.weight
            mat = w.view(w.size(0), -1) if w.ndim>2 else w
            total += torch.norm(mat, p=2, dim=0).sum()
        return total

    def compute_spectral_norm(self) -> torch.Tensor:
        """Product of largest singular values of each layer."""
        prod = torch.tensor(1., device=next(self.parameters()).device)
        for m in self._weighted_layers:
            w = m.weight
            mat = w.view(w.size(0), -1) if w.ndim>2 else w
            prod *= torch.linalg.svdvals(mat)[0]
        return prod

