import torch
import torch.nn as nn
import math

# 1. Sinusoidal embeddings for time t
class TimestepEmbedder(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()

		self.hidden_size = hidden_size
	
	def forward(self, t):
		half_dim = self.hidden_size // 2
		emb = math.log(10000) / (half_dim - 1)
		emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
		emb = t[:, None] * t[None, :]
		emb = torch.cat(emb.sin(), emb.cos(), dim=1)
		return emb

# 2. Block DiT with adaLN
class DiTBlock(nn.Module):
	def __init__(self, hidden_size, num_heads):
		super().__init__()

		# Normalization layers
		self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
		self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

		# Attention layers
		self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

		# MLP layer
		self.mlp = nn.Sequential(
			nn.Linear(hidden_size, hidden_size * 4),
			nn.GELU(),
			nn.Linear(4 * hidden_size, hidden_size)
		)

		# Adaptive Layer Norms
		self.adaln_modulation = nn.Linear(hidden_size, 4 * hidden_size)
	
	def forward(self, x, t_emb):
		# emb: [B, hidden_size]
		shift_1, scale_1, shift_2, scale_2 = self.adaln_modulation(t_emb).chunk(4, dim=1)

		# Attention with modulation
		norm_x = self.norm1(x) * (1 + scale_1.unsqueeze(1)) + shift_1.unsqueeze(1)
		attn_out, _ = self.attn(norm_x, norm_x, norm_x)
		x = x + attn_out

		# MLP with modulation
		norm_x2 = self.norm2(x) * (1 + scale_2.unsqueeze(1)) + scale_2.unsqueeze(1)
		x = x + self.mlp(norm_x2)
		return x

# 3. Vision Encoder
class VisionEncoder(nn.Module):
	def __init__(self, h: int = 32, w: int = 32, kernel_size: int = 4, hidden_size: int = 64):
		super().__init__()
		self.pad_size = h / kernel_size
		self.conv = nn.Conv2d(kernel_size=kernel_size, stride=kernel_size)
		self.linear = nn.Linear(self.pad_size, hidden_size)

	def forward(self, x):
		x = self.conv(x)
		x = self.linear(x)
		return x

# 4. Diffusion Transformer for generating image
class DiT(nn.Module):
	def __init__(self, in_channels: int = 3, hidden_size: int = 256, num_heads: int = 6, num_layers: int = 4):
		super().__init__()
		self.vision_encoder = VisionEncoder()
		self.timestembedder = nn.Sequential(
			TimestepEmbedder(hidden_size),
			nn.SiLU(),
			nn.Linear(hidden_size, hidden_size)
		)
		self.blocks = [DiTBlock(hidden_size, num_heads) for _ in range(num_layers)]
		self.out_proj = nn.Linear(hidden_size, in_channels)
	
	def forward(self, x, t):
		x = self.vision_encoder(x)
		t_emb = self.timestembedder(x)
		for block in self.blocks:
			x = block(x, t_emb)
		
		return self.out_proj(x)

