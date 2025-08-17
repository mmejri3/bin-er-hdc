import math
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class Encoder:
    """
    Nonlinear encoder that maps data to a high-dimensional space using random projections.
    
    Given input vector X ∈ ℝ^f and random basis B ∈ ℝ^(D×f), and phase shift b ∈ [0, 2π]^D,
    the hypervector H ∈ ℝ^D is computed as:

        H_i = cos(X · B_i + b_i) * sin(X · B_i)

    Args:
        features (int): Dimensionality of input data.
        dim (int): Target dimensionality (default: 4000).
    """

    def __init__(self, features: int, dim: int = 4000, encoding_system = "kernel"):
        self.features = features
        self.dim = dim

        # Random basis and base
        self.basis = torch.randn(self.dim, self.features).float()
        self.base = torch.empty(self.dim).uniform_(0.0, 2 * math.pi)

        # Save for reproducibility/debugging
        torch.save(self.basis, 'basis.pt')
        torch.save(self.base, 'bais.pt')

        # Level hypervectors for quantized encoding (optional path)
        self.levels = self._gen_level_hvs(total_levels=100, dim=self.dim)
        
        self.n_levels = 100

        # Random IDs for optional HD operations
        self.IDs = 2 * torch.randint(0, 2, (self.features, self.dim)) - 1
        if encoding_system in {"kernel", "record"}:
            self.encoding_system = encoding_system
    def _gen_level_hvs(self, total_levels: int, dim: int):
        """Generates level-wise hypervectors for quantized encoding."""
        level_hvs = {}
        base_val = -1
        base = np.full(dim, base_val)
        index_vector = range(dim)
        change = dim // 2
        next_level = dim // (2 * total_levels)

        for level in range(total_levels):
            if level == 0:
                to_one = np.random.permutation(index_vector)[:change]
            else:
                to_one = np.random.permutation(index_vector)[:next_level]
            for index in to_one:
                base[index] *= -1
            level_hvs[level] = torch.tensor(copy.deepcopy(base), dtype=torch.float32)
        return level_hvs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the encoder to a batch of input vectors.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, features).
        
        Returns:
            torch.Tensor: Encoded tensor of shape (N, dim).
        """
        if self.encoding_system == "kernel":
            n = x.size(0)
            bsize = math.ceil(0.01 * n)
            h = torch.empty(n, self.dim, device=x.device, dtype=x.dtype)
            temp = torch.empty(bsize, self.dim, device=x.device, dtype=x.dtype)

            for i in range(0, n, bsize):
                torch.matmul(x[i:i + bsize], self.basis.T, out=temp)
                torch.add(temp, self.base, out=h[i:i + bsize])
                h[i:i + bsize].cos_().mul_(temp.sin_())

            return torch.sign(h)

        else:
            p_min = x.min()
            p_max = x.max()
            H = torch.zeros(x.size(0), self.dim)
            for i, in_x in enumerate(x):
                p = in_x
                l = (
                    (self.n_levels - 1) * (p - p_min) / (p_max - p_min)
                ).int()  # This is now a vector of l values

                h = sum(
                    torch.multiply(self.IDs[j], self.levels[l_j.item()])
                    for j, l_j in enumerate(l)
                )

                h = (h - h.min()) / (h.max() - h.min())
                H[i] = torch.sign(2*h-1)  

            return H
    def to(self, *args):
        """
        Moves encoder tensors to the specified device or dtype.

        Args:
            *args: Passed to `tensor.to()`, typically device or dtype.

        Returns:
            Encoder: self (for chaining).
        """
        self.basis = self.basis.to(*args)
        self.base = self.base.to(*args)
        return self

