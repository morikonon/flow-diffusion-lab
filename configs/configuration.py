from dataclasses import dataclass

@dataclass
class Configuration:
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
  down_channels = [128, 256, 384]
  mid_channels = [384]
  att_down = [False, False]
  norm_channels: int = 32
  num_heads: int = 4
  num_down_layers: int = 2
  num_mid_layers: int = 2
  num_up_layers: int = 2

  # train params
  seed: int = 1111

config = Configuration()