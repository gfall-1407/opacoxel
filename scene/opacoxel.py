import torch
import torch.nn as nn
from typing import Tuple


class Opacoxel(nn.Module):
    """
    Learnable voxelized opacity field storing logits on grid vertices.
    A(p) = sigmoid(trilinear_interpolate(logits, p))

    Contract:
    - bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
    - resolution: (nx, ny, nz) grid vertices count per axis
    - logit_grid: nn.Parameter[nx, ny, nz]
    Inputs: positions in world coordinates [N, 3]
    Outputs: alpha in [0,1] [N, 1]
    """

    def __init__(self,
                 bounds: Tuple[float, float, float, float, float, float],
                 resolution: Tuple[int, int, int] = (64, 64, 64),
                 device: str = "cuda",
                 empty_logit: float = -4.0) -> None:
        super().__init__()
        self.device = device
        self.bounds = bounds
        self.resolution = resolution

        nx, ny, nz = resolution
        init = torch.full((nx, ny, nz), float(empty_logit), dtype=torch.float32, device=device)
        # Learnable logits at grid vertices
        self.logit_grid = nn.Parameter(init)

    @property
    def voxel_size(self):
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        nx, ny, nz = self.resolution
        return (
            (xmax - xmin) / max(nx - 1, 1),
            (ymax - ymin) / max(ny - 1, 1),
            (zmax - zmin) / max(nz - 1, 1),
        )

    def world_to_grid(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Map world coordinates to grid index space [0, nx-1] etc. (float).
        positions: [N,3] on device
        returns: [N,3] float grid coordinates
        """
        assert positions.shape[-1] == 3
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        nx, ny, nz = self.resolution
        x = (positions[:, 0] - xmin) / max(xmax - xmin, 1e-8) * (nx - 1)
        y = (positions[:, 1] - ymin) / max(ymax - ymin, 1e-8) * (ny - 1)
        z = (positions[:, 2] - zmin) / max(zmax - zmin, 1e-8) * (nz - 1)
        return torch.stack([x, y, z], dim=-1)

    def sample_logits(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Trilinear interpolation of logits at world positions.
        positions: [N,3]
        returns: [N,1] logits
        """
        grid_coords = self.world_to_grid(positions)
        nx, ny, nz = self.resolution

        # Integer and fractional parts
        xyz0 = torch.floor(grid_coords)
        frac = grid_coords - xyz0
        xyz0 = xyz0.long()

        # Clamp base and base+1 to valid range
        max_idx = torch.tensor([nx - 1, ny - 1, nz - 1], device=positions.device)
        xyz0 = torch.clamp(xyz0, min=0, max=max_idx)
        xyz1 = torch.clamp(xyz0 + 1, min=0, max=max_idx)

        x0, y0, z0 = xyz0[:, 0], xyz0[:, 1], xyz0[:, 2]
        x1, y1, z1 = xyz1[:, 0], xyz1[:, 1], xyz1[:, 2]

        # 8 corners
        g = self.logit_grid
        c000 = g[x0, y0, z0]
        c100 = g[x1, y0, z0]
        c010 = g[x0, y1, z0]
        c110 = g[x1, y1, z0]
        c001 = g[x0, y0, z1]
        c101 = g[x1, y0, z1]
        c011 = g[x0, y1, z1]
        c111 = g[x1, y1, z1]

        xd, yd, zd = frac[:, 0], frac[:, 1], frac[:, 2]
        c00 = c000 * (1 - xd) + c100 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        logits = c0 * (1 - zd) + c1 * zd

        return logits.unsqueeze(-1)

    def sample_alpha(self, positions: torch.Tensor) -> torch.Tensor:
        """Return alpha in [0,1] at positions [N,3] -> [N,1]."""
        logits = self.sample_logits(positions)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def initialize_from_point_cloud(self,
                                    points_world: torch.Tensor,
                                    occupied_logit: float = 2.0,
                                    empty_logit: float = -4.0) -> None:
        """
        Initialize grid logits:
        - Fill with empty_logit
        - For each SfM point, set the 8 vertices of its containing cell to occupied_logit
        points_world: [M,3] tensor on self.device
        """
        self.logit_grid.data.fill_(float(empty_logit))
        if points_world.numel() == 0:
            return

        grid_coords = self.world_to_grid(points_world)
        base = torch.floor(grid_coords).long()
        nx, ny, nz = self.resolution
        max_idx = torch.tensor([nx - 1, ny - 1, nz - 1], device=points_world.device)
        base = torch.clamp(base, min=0, max=max_idx)

        # Corner offsets (0/1)^3
        corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                                [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                               device=points_world.device, dtype=torch.long)

        # Loop (kept for clarity and broad compatibility)
        for i in range(base.shape[0]):
            b = base[i]
            idxs = b.unsqueeze(0) + corners
            idxs = torch.clamp(idxs, min=0, max=max_idx)
            for j in range(8):
                x, y, z = idxs[j]
                # use max in case multiple points touch same vertex
                self.logit_grid.data[x, y, z] = max(self.logit_grid.data[x, y, z], float(occupied_logit))
