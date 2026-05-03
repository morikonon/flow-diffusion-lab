import torch
import torch.nn as nn
import torch.nn.functional as F

def flow_match_loss(model, x1):
	B = x1.shape[0]
	x0 = torch.randint_like(x1)

	t = torch.rand((B, ), device=x1.device)
	t_expand = t.view(B, 1, 1, 1)

	xt = t_expand * x1 + (1 - t_expand) * x0
	target = x1 - x0

	pred_v = model(xt, t * 1000)
	loss = F.mse_loss(pred_v, target)
	return loss