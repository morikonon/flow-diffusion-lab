import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.tools import get_patch_position_embedding, get_timestep_embedding


# Patch Embedding
class PatchEmbedding(nn.Module):
	def __init__(self, image_height, image_width, in_channels, patch_height, patch_width, hidden_size):
		super().__init__()
		self.image_height = image_height
		self.image_width = image_width
		self.in_channels = in_channels
		self.hidden_size = hidden_size
		self.patch_height = patch_height
		self.patch_width = patch_width

		patch_dim = self.in_channels * self.patch_height * self.patch_width
		self.patch_embed = nn.Linear(patch_dim, self.hidden_size)

		nn.init.xavier_uniform_(self.patch_embed.weight)
		nn.init.constant_(self.patch_embed.bias, val=0)

	def forward(self, x):
		grid_size_h = self.image_height // self.patch_height
		grid_size_w = self.image_width // self.patch_width

		out = rearrange(x, "b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)", ph=self.patch_height, pw=self.patch_width)
		out = self.patch_embed(out)

		pos_embed = get_patch_position_embedding(pos_emb_dim=self.hidden_size, grid_size=(grid_size_h, grid_size_w), device=x.device)
		out = out + pos_embed.unsqueeze(0)
		return out


# 1. Sinusoidal embeddings for time t
class TimestepEmbedder(nn.Module):
	def __init__(self, embedding_dim):
		super().__init__()
		self.embedding_dim = embedding_dim

	def forward(self, t):
		return get_timestep_embedding(t, self.embedding_dim)


# 2. Self-attention with a fused qkv projection
class Attention(nn.Module):
	def __init__(self, hidden_size, num_heads, head_dim):
		super().__init__()
		self.n_heads = num_heads
		self.hidden_size = hidden_size
		self.head_dim = head_dim
		self.attn_dim = self.n_heads * self.head_dim

		self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.attn_dim, bias=True)
		self.output_proj = nn.Linear(self.attn_dim, self.hidden_size)

		nn.init.xavier_uniform_(self.qkv_proj.weight)
		nn.init.constant_(self.qkv_proj.bias, 0)
		nn.init.xavier_uniform_(self.output_proj.weight)
		nn.init.constant_(self.output_proj.bias, 0)

	def forward(self, x):
		B, N, C = x.shape
		qkv = self.qkv_proj(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		out = F.scaled_dot_product_attention(q, k, v)

		out = out.transpose(1, 2).reshape(B, N, self.attn_dim)
		return self.output_proj(out)


# 3. Block DiT with adaLN-Zero conditioning
class DiTBlock(nn.Module):
	def __init__(self, hidden_size, num_heads, head_dim):
		super().__init__()
		ff_hidden_dim = 4 * hidden_size

		# Normalization layers
		self.att_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
		self.ff_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

		# Attention layer
		self.attn_block = Attention(hidden_size, num_heads, head_dim)

		# MLP layer
		self.mlp_block = nn.Sequential(
			nn.Linear(hidden_size, ff_hidden_dim),
			nn.GELU(approximate="tanh"),
			nn.Linear(ff_hidden_dim, hidden_size),
		)

		# Adaptive layer norm modulation (shift/scale for both sub-blocks, plus output gates)
		self.adaptive_norm_mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(hidden_size, 6 * hidden_size, bias=True),
		)

		nn.init.xavier_uniform_(self.mlp_block[0].weight)
		nn.init.constant_(self.mlp_block[0].bias, 0)
		nn.init.xavier_uniform_(self.mlp_block[-1].weight)
		nn.init.constant_(self.mlp_block[-1].bias, 0)

		# Zero-init so each block starts as an identity function (adaLN-Zero)
		nn.init.constant_(self.adaptive_norm_mlp[-1].weight, 0)
		nn.init.constant_(self.adaptive_norm_mlp[-1].bias, 0)

	def forward(self, x, condition):
		(pre_attn_shift, pre_attn_scale, post_attn_scale,
		 pre_mlp_shift, pre_mlp_scale, post_mlp_scale) = self.adaptive_norm_mlp(condition).chunk(6, dim=1)

		# Attention with modulation
		attn_norm_out = self.att_norm(x) * (1 + pre_attn_scale.unsqueeze(1)) + pre_attn_shift.unsqueeze(1)
		x = x + post_attn_scale.unsqueeze(1) * self.attn_block(attn_norm_out)

		# MLP with modulation
		mlp_norm_out = self.ff_norm(x) * (1 + pre_mlp_scale.unsqueeze(1)) + pre_mlp_shift.unsqueeze(1)
		x = x + post_mlp_scale.unsqueeze(1) * self.mlp_block(mlp_norm_out)
		return x


# 4. Diffusion Transformer for generating images
class DiT(nn.Module):
	def __init__(self, image_size, in_channels, config):
		super().__init__()
		self.num_layers = config.num_layers
		self.image_height, self.image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
		self.in_channels = in_channels
		self.hidden_size = config.hidden_dim
		self.patch_height = config.patch_size
		self.patch_width = config.patch_size
		self.timestep_emb_dim = config.timestep_emb_dim

		self.grid_h = self.image_height // self.patch_height
		self.grid_w = self.image_width // self.patch_width

		self.patch_embed = PatchEmbedding(
			image_height=self.image_height,
			image_width=self.image_width,
			in_channels=self.in_channels,
			patch_height=self.patch_height,
			patch_width=self.patch_width,
			hidden_size=self.hidden_size,
		)

		self.t_proj = nn.Sequential(
			TimestepEmbedder(self.timestep_emb_dim),
			nn.Linear(self.timestep_emb_dim, self.hidden_size),
			nn.SiLU(),
			nn.Linear(self.hidden_size, self.hidden_size),
		)

		self.blocks = nn.ModuleList([
			DiTBlock(self.hidden_size, config.num_heads, config.head_dim) for _ in range(self.num_layers)
		])

		self.final_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
		self.final_adaln = nn.Sequential(
			nn.SiLU(),
			nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True),
		)
		self.proj_out = nn.Linear(self.hidden_size, self.patch_height * self.patch_width * self.in_channels)

		nn.init.normal_(self.t_proj[1].weight, std=0.02)
		nn.init.normal_(self.t_proj[3].weight, std=0.02)
		nn.init.constant_(self.final_adaln[-1].weight, 0)
		nn.init.constant_(self.final_adaln[-1].bias, 0)
		nn.init.constant_(self.proj_out.weight, 0)
		nn.init.constant_(self.proj_out.bias, 0)

	def forward(self, x, t):
		x = self.patch_embed(x)
		condition = self.t_proj(t)

		for block in self.blocks:
			x = block(x, condition)

		shift, scale = self.final_adaln(condition).chunk(2, dim=1)
		x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
		x = self.proj_out(x)

		x = rearrange(
			x, "b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)",
			nh=self.grid_h, nw=self.grid_w, ph=self.patch_height, pw=self.patch_width, c=self.in_channels,
		)
		return x
