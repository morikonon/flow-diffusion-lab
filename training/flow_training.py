import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm

from dataset.load_dataset import load_dataset


# Function to compute flow matching loss
def flow_match_loss(model, x1):
	B = x1.shape[0]
	x0 = torch.randn_like(x1)

	t = torch.rand((B, ), device=x1.device)
	t_expand = t.view(B, 1, 1, 1)

	xt = t_expand * x1 + (1 - t_expand) * x0
	target = x1 - x0

	# prediction of model
	pred_v = model(xt, t * 1000)

	# Compute mse loss
	loss = F.mse_loss(pred_v, target)
	return loss


# Function to training
def training(
	model,
	configs,
	loss_function: str = "flow_matching",
	epochs: int = 10,
	max_steps: int = None,
	checkpoint_dir: str = "checkpoints",
	checkpoint_name: str = "dit",
	subset_size: int = None,
):
	# Set device
	device = configs.device
	use_amp = device == "cuda"
	scaler = GradScaler(device, enabled=use_amp)

	# Set model to training
	model.to(device)
	model.train()

	# Load training dataset
	data_loader = load_dataset(subset_size=subset_size)

	# Create optimizer
	if configs.optimizer.lower() == "adam":
		optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
	else:
		raise ValueError(f"Optimizer '{configs.optimizer}' is not supported!")

	if loss_function.lower() == "flow_matching":
		loss_fn = flow_match_loss
	else:
		raise ValueError(f"Loss function '{loss_function}' is not supported!")

	# Train our model
	global_step = 0
	stop_training = False
	for epoch in tqdm(range(epochs)):
		# collect losses during training
		epoch_loss = 0
		num_steps_this_epoch = 0

		# Iterate images
		for step, (images, labels) in enumerate(tqdm(data_loader, desc=f"{epoch + 1}/{epochs}", leave=False)):
			images = images.to(device)

			# Zero gradient
			optimizer.zero_grad()

			# Here we compute loss
			with autocast(device, enabled=use_amp):
				loss = loss_fn(model, images)

			# Update parameters of our model
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			# Collect losses
			epoch_loss += loss.item()
			num_steps_this_epoch += 1
			global_step += 1

			if max_steps is not None and global_step >= max_steps:
				stop_training = True
				break

		# Simple logging
		print(f"Epoch: {epoch + 1}| mse_loss: {epoch_loss / num_steps_this_epoch}")

		if stop_training:
			break

	# Save a checkpoint at the end of training
	os.makedirs(checkpoint_dir, exist_ok=True)
	checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
	torch.save(model.state_dict(), checkpoint_path)
	print(f"Saved checkpoint to {checkpoint_path}")

	return checkpoint_path
