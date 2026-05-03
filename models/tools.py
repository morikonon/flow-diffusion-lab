import torch
import math

def get_patch_position_embedding(pos_emb_dim, grid_size, device):
	grid_size_h, grid_size_w = grid_size
	grid_h = torch.arange(grid_size_h, dtype=torch.float32, device=device)
	grid_w = torch.arange(grid_size_w, dtype=torch.float32, device=device)

	grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
	grid = torch.stack(grid, dim=0)

	grid_h_positions = grid[0].reshape(-1)
	grid_w_positions = grid[1].reshape(-1)

	factor = 10000 ** (torch.arange(0, pos_emb_dim // 4, dtype=torch.float32, device=device) / (pos_emb_dim // 4))

	grid_h_emb = grid_h_positions[:, None] / factor
	grid_w_emb = grid_w_positions[:, None] / factor

	grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=1)
	grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=1)

	pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)
	return pos_emb

def get_timestep_embedding(timestep, embedding_dim):
	half_dim = timestep / 2
	exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timestep.device)
	exponent = exponent / half_dim
	emb = torch.exp(exponent)
	emb = timestep[:, None].float() * emb[None, :]
	emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
	return emb