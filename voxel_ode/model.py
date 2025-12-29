import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class LinearBaseline(nn.Module):
    """
    Simple linear regression baseline for comparison.

    For grid output: y = Wx + b (per-voxel linear transformation via 1x1x1 conv)
    For scalar output: global avg pool + linear layer

    This serves as a baseline to compare against the more complex VoxelAutoRegressor.
    """

    def __init__(
        self,
        in_channels: int,
        grid_out_channels: int = 3,
        scalar_out_dim: int = 5,
        cond_params_dim: int = 26,
        use_param_broadcast: bool = False,
        **kwargs  # Ignore other args like base_channels, depth, r, use_checkpoint
    ):
        super().__init__()
        self.use_param_broadcast = use_param_broadcast
        self.cond_params_dim = cond_params_dim

        # Grid prediction: simple 1x1x1 conv (per-voxel linear transformation)
        # This is equivalent to y[z,y,x] = W @ x[z,y,x] + b for each voxel
        self.grid_head = nn.Conv3d(in_channels, grid_out_channels, kernel_size=1)

        # Scalar prediction: global pooling + linear
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, scalar_out_dim),
        )

    def forward(self, x_grid, params_scalar=None):
        """
        Args:
            x_grid: [B, C_in, Z, Y, X] - concatenated static + dynamic inputs
            params_scalar: [B, 26] - optional scalar parameters (ignored in baseline)

        Returns:
            grid_out: [B, grid_out_channels, Z, Y, X] - predicted residuals
            scalar_out: [B, 5] - predicted field scalars
        """
        # params_scalar is ignored in the linear baseline
        grid_out = self.grid_head(x_grid)
        scalar_out = self.scalar_head(x_grid)
        return grid_out, scalar_out


class LayerNorm3d(nn.Module):
    """3D LayerNorm using GroupNorm with 1 group."""

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(1, num_channels, eps=eps, affine=True)

    def forward(self, x):
        # x: [B, C, Z, Y, X]
        return self.gn(x)


class ResBlock3d(nn.Module):
    """3D Residual block with two conv layers."""

    def __init__(self, c, dilation=1, use_checkpoint=False):
        super().__init__()
        self.conv1 = nn.Conv3d(c, c, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = LayerNorm3d(c)
        self.act = nn.SiLU(inplace=False)  # Disable inplace for better memory management
        self.conv2 = nn.Conv3d(c, c, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.norm2 = LayerNorm3d(c)
        self.use_checkpoint = use_checkpoint

    def _forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(x + h)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class VoxelAutoRegressor(nn.Module):
    """
    3D CNN for autoregressive prediction of voxel grids with residual (delta) outputs.

    Predicts RESIDUALS (deltas from t to t+1):
    - Delta Pressure (change from t to t+1)
    - Delta Temperature (change from t to t+1)
    - Delta WEPT (change from t to t+1, v2.5 only)
    - Scalar field outputs (5 values) at t+1

    From:
    - Static 3D inputs (geology, wells, etc.) - 8 channels
    - Dynamic grid at t-1 (P, T, WEPT) - 2 or 3 channels
    - Dynamic grid at t (P, T, WEPT) - 2 or 3 channels
    - Optional scalar parameters (26 values)

    Total input channels for v2.5: 8 + 3 + 3 = 14
    Total input channels for v2.4: 8 + 2 + 2 = 12
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        depth: int = 8,
        r: int = 2,
        cond_params_dim: int = 26,
        use_param_broadcast: bool = False,
        grid_out_channels: int = 2,  # Number of residual outputs (2=P,T or 3=P,T,WEPT)
        scalar_out_dim: int = 5,
        use_checkpoint: bool = False,  # Gradient checkpointing to save memory
    ):
        super().__init__()
        k = 2 * r + 1  # receptive field kernel size
        self.use_param_broadcast = use_param_broadcast
        self.cond_params_dim = cond_params_dim

        # Stem: large kernel to capture receptive field
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=k, padding=r, bias=False),
            LayerNorm3d(base_channels),
            nn.SiLU(inplace=False),  # Disable inplace for better memory management
        )

        # Residual trunk
        blocks = []
        for i in range(depth):
            # Vary dilation to extend receptive field
            dilation = 1 #if i < depth // 2 else 2
            blocks.append(ResBlock3d(base_channels, dilation=dilation, use_checkpoint=use_checkpoint))
        self.trunk = nn.Sequential(*blocks)

        # FiLM-like conditioning from scalar parameters
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_params_dim, base_channels),
            nn.SiLU(inplace=False),  # Disable inplace for better memory management
            nn.Linear(base_channels, base_channels),
        )

        # Grid prediction head
        self.grid_head = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm3d(base_channels),
            nn.SiLU(inplace=False),  # Disable inplace for better memory management
            nn.Conv3d(base_channels, grid_out_channels, kernel_size=1),
        )

        # Scalar prediction head (global pooling)
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_channels, base_channels),
            nn.SiLU(inplace=False),  # Disable inplace for better memory management
            nn.Linear(base_channels, scalar_out_dim),
        )

    def forward(self, x_grid, params_scalar=None):
        """
        Args:
            x_grid: [B, C_in, Z, Y, X] - concatenated static + dynamic inputs
            params_scalar: [B, 26] - optional scalar parameters

        Returns:
            grid_out: [B, 2, Z, Y, X] - predicted Pressure & Temperature
            scalar_out: [B, 5] - predicted field scalars
        """
        h = self.stem(x_grid)

        # Apply conditioning from scalar parameters
        if params_scalar is not None:
            bias = self.cond_mlp(params_scalar).view(-1, h.size(1), 1, 1, 1)
            h = h + bias

        h = self.trunk(h)

        grid_out = self.grid_head(h)
        scalar_out = self.scalar_head(h)

        return grid_out, scalar_out
