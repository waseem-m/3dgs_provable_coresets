# gaussian_utils.py ()
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class GaussianPLY:
    """Reference-compatible container for GraphDECO 3DGS PLY fields."""
    means: torch.Tensor            # (N,3) xyz
    scales: torch.Tensor           # (N,3) raw from file (log-scales in official code)
    rotations_wxyz: torch.Tensor   # (N,4) quaternion in WXYZ order (w first)
    opacity_logits: torch.Tensor   # (N,1) logit (not post-sigmoid)
    features_dc: torch.Tensor      # (N,3,1) DC per channel
    features_rest: torch.Tensor    # (N,3,M-1) channel-major rest (R 15, G 15, B 15 when deg=3)
    sh_degree: int                 # scalar degree d

    def subset(self, idx: torch.Tensor) -> "GaussianPLY":
        """
        Return a view/copy with only rows in `idx` (long or bool).
        """
        if idx.dtype == torch.bool:
            idx = torch.nonzero(idx, as_tuple=False).squeeze(1)

        return GaussianPLY(
            means=self.means[idx],
            scales=self.scales[idx],
            rotations_wxyz=self.rotations_wxyz[idx],
            opacity_logits=self.opacity_logits[idx],
            features_dc=self.features_dc[idx],
            features_rest=self.features_rest[idx],
            sh_degree=self.sh_degree,
        )

def scale_colors_inplace(ply, idx: torch.Tensor, factors: torch.Tensor) -> None:
    """
    Multiply SH color (DC + rest) of selected Gaussians by 'factors'.

    - idx: (K,) long or bool indices into N
    - factors: (K,) float >= 0
    Broadcasts over channels/coeffs.
    """
    if idx.dtype == torch.bool:
        idx = idx.nonzero(as_tuple=False).squeeze(1)
    f = factors.view(-1, 1, 1)  # broadcast over (3, M) -> (K,1,1)
    ply.features_dc[idx]   = ply.features_dc[idx]   * f
    ply.features_rest[idx] = ply.features_rest[idx] * f