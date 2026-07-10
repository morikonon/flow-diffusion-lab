from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class Configuration:

  # train params
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  optimizer: str = "Adam"
  learning_rate: float = 1e-4
  seed: int = 1111

  # diffusion params
  num_timesteps: int = 1000
  beta_start: float = 0.0001
  beta_end: float = 0.02

  # dit params
  patch_size: int = 2
  num_layers: int = 6
  hidden_dim: int = 384
  num_heads: int = 6
  head_dim: int = 64
  timestep_emb_dim: int = 384

  # autoencoder params
  z_channels: int = 4
  codebook_size: int = 8192
  down_channels: List[int] = field(default_factory=lambda: [128, 256, 384])
  mid_channels: List[int] = field(default_factory=lambda: [384])
  att_down: List[bool] = field(default_factory=lambda: [False, False])
  norm_channels: int = 32
  vae_num_heads: int = 4
  num_down_layers: int = 2
  num_mid_layers: int = 2
  num_up_layers: int = 2

config = Configuration()
