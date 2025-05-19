import torch
import torch.nn as nn
from typing import List, Type, Tuple, Optional

__all__ = ["BasicBlock", "ResNetClassifier"]

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

    expansion: int = 1  

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

        out += identity  
        return self.relu(out)


class ResNetClassifier(nn.Module):
    """Flexible ResNet classifier with spectral‑complexity utilities.

    All architectural hyper‑parameters are passed via ``__init__`` so the model
    can be instantiated in sweeps or loaded from a config file.

    Notes
    -----
    * Skip‑connections are treated as identity additions and NOT included in the
      spectral product, following Bartlett et al. (2017).
    * ``compute_model_norm`` reproduces Eq. (1.2) in the reference paper.
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
        input_size: Tuple[int, int, int] = (3, 32, 32),
    ) -> None:
        super().__init__()
        assert len(layers) == len(widths), "`layers` and `widths` must be equal‑length lists"

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
        for idx, (num_blocks, planes, stride) in enumerate(zip(layers, widths, strides)):
            stage = self._make_stage(block, planes, num_blocks, stride, norm_layer, dropout)
            self.stages.append(stage)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    
        self._weighted_layers: List[nn.Module] = [self.stem[0]]
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
        if weight.ndim > 2:
            w = weight.view(weight.size(0), -1)
        else:
            w = weight
        return torch.linalg.svdvals(w)[0]

    @staticmethod
    def _two_one_norm(weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim > 2:
            w = weight.view(weight.size(0), -1)
        else:
            w = weight
        return torch.norm(w, p=2, dim=0).sum()

    def compute_model_norm(self) -> torch.Tensor:
        devices = next(self.parameters()).device
        s_list, t_list = [], []
        for module in self._weighted_layers:
            weight = module.weight.detach()
            s_list.append(self._spectral_norm(weight))
            t_list.append(self._two_one_norm(weight))

        prod_spec = torch.prod(torch.stack(s_list))
        correction_sum = []
        L = len(s_list)
        for i in range(L):
            t_sum = torch.sum(torch.stack(t_list[i + 1 :])) if i < L - 1 else torch.tensor(0.0, device=devices)
            s_i = s_list[i].clamp_min(1e-12)
            correction_sum.append((t_sum ** (2.0 / 3.0)) / (s_i ** (2.0 / 3.0)))
        correction_sum = torch.sum(torch.stack(correction_sum))
        return prod_spec * (correction_sum ** (3.0 / 2.0))


    def compute_margin_distribution(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            outputs = self(inputs)
            batch_size = outputs.size(0)
            true_scores = outputs[torch.arange(batch_size), labels]
            outputs_clone = outputs.clone()
            outputs_clone[torch.arange(batch_size), labels] = -float("inf")
            max_incorrect = outputs_clone.max(dim=1)[0]
            raw_margins = true_scores - max_incorrect

            model_norm = self.compute_model_norm()
            X_flat = inputs.view(batch_size, -1)
            X_norm = torch.norm(X_flat, p=2)
            scaling_factor = model_norm * (X_norm / batch_size + 1e-12)
            return raw_margins / scaling_factor
