import math
import torch
from torch import nn
from ff_prior_latent.learning import LatentDenoiser
from .protocol import P


def time_embedding(t):
    freq = torch.exp(torch.arange(16, device=t.device) * (-math.log(10000) / 15))
    phase = t[:, None] * freq[None]
    return torch.cat([phase.sin(), phase.cos()], 1)


class Block(nn.Module):
    """Explicit attention avoids differing train/eval fast-path implementations."""
    def __init__(self, width=104, heads=4, ff=208):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Linear(ff, width))

    def forward(self, x):
        n, tokens, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(n, tokens, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(width // self.heads)).softmax(-1)
        x = x + self.proj((weights @ v).transpose(1, 2).reshape(n, tokens, width))
        return x + self.ff(self.norm2(x))


class SmallTransformer(nn.Module):
    def __init__(self, dimension=102, condition_dim=65):
        super().__init__()
        if dimension != 102 or condition_dim != 65:
            raise ValueError("frozen codec dimensions changed")
        self.calls = 0
        self.input = nn.Linear(17, 104)
        self.position = nn.Parameter(torch.zeros(1, 6, 104))
        nn.init.normal_(self.position, std=.02)
        self.condition = nn.Sequential(nn.Linear(65, 104), nn.SiLU(), nn.Linear(104, 104))
        self.time = nn.Sequential(nn.Linear(32, 104), nn.SiLU(), nn.Linear(104, 104))
        self.blocks = nn.Sequential(Block(), Block())
        self.norm = nn.LayerNorm(104)
        self.output = nn.Linear(104, 17)

    def forward(self, x, t, c):
        self.calls += 1
        h = self.input(x.reshape(-1, 6, 17)) + self.position
        h = h + (self.condition(c) + self.time(time_embedding(t)))[:, None, :]
        return self.output(self.norm(self.blocks(h))).reshape(-1, 102)


def make(kind, seed=None):
    # Different architectures cannot share weights. Initialization does not
    # consume the generator used for matched training examples/noise.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(P["initialization_seed"] if seed is None else seed)
        if kind == "mlp":
            return LatentDenoiser(102, 65)
        if kind == "transformer":
            return SmallTransformer()
        raise ValueError(kind)


def parameter_counts():
    return {k: sum(p.numel() for p in make(k).parameters()) for k in ("mlp", "transformer")}
