import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        p_drop: float = 0.3  # ドロップアウト率
    ) -> None:
        super().__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(in_channels * seq_len, 256),  
            nn.ReLU(),
            nn.Dropout(p_drop),  
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p_drop),  
            nn.Linear(128, in_channels * seq_len),  
            nn.ReLU(),
            nn.Dropout(p_drop),  
        )

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange("b d 1 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X (b, c, t): _description_
        Returns:
            X (b, num_classes): _description_
        """
        b, c, t = X.size()
        X = X.view(b, -1)
        X = self.fc_layers(X)
        X = X.view(b, c, t, 1)
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv2d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv2d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size, padding="same")

        self.batchnorm0 = nn.BatchNorm2d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-1)

        return self.dropout(X)
