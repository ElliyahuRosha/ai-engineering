"""Skip-gram model with negative sampling."""

import torch
from torch import nn


class SkipGram(nn.Module):
    """Two embedding tables and a dot product. Nothing else.

    center  : what a word looks like when it is the focus.
    context : what a word looks like when it sits in the neighbourhood.
    """

    def __init__(self, vocab_size: int, dim: int = 100, sparse: bool = True) -> None:
        super().__init__()
        # sparse=True keeps the gradient of each step to the rows that were used.
        # CUDA has no sparse Adagrad kernel, so the GPU path uses dense tables
        # (11k x 100 floats is 4.5 MB -- the saving only matters on CPU).
        self.center = nn.Embedding(vocab_size, dim, sparse=sparse)
        self.context = nn.Embedding(vocab_size, dim, sparse=sparse)

        # PyTorch defaults to N(0, 1), which makes initial dot products ~sqrt(d)
        # and saturates the sigmoid. Word2Vec uses a small uniform box instead.
        bound = 0.5 / dim
        nn.init.uniform_(self.center.weight, -bound, bound)
        nn.init.uniform_(self.context.weight, -bound, bound)

    def forward(self, centers: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        """Score every (center, context) pair.

        centers: (B,)    contexts: (B, K)    returns: (B, K)
        """
        center_vectors = self.center(centers)          # (B, d)
        context_vectors = self.context(contexts)       # (B, K, d)
        return torch.einsum("bd,bkd->bk", center_vectors, context_vectors)
