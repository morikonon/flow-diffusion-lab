import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from dataset import load_dataset

# Function to compute flow matching loss
def flow_match_loss(model, x1):
	B = x1.shape[0]
	x0 = torch.randint_like(x1)

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
def training(model, configs, loss_function: str = "flow_matching", epochs: int = 10):
	scaler = GradScaler()

	# Set device
	device = configs.device

	# Set model to training 
	model.train()

	# Load training dataset
	data_loader = load_dataset()
	
	# Create optimizer
	if (configs.optimizer).lower() == "adam":
		optimizer = optim.Adam(model.parameters(), le=configs.learning_rate)
	else:
		print(f"Your optimizer is not found!")

	if (loss_function).lower() == "flow_matching":
		loss_function = flow_match_loss
	
	# Train our model
	for epoch in tqdm(range(epochs)):
		# collect losses during training
		epoch_loss = 0

		# Iterate images
		for step, (images, labels) in enumerate(tqdm(data_loader), desc=f"{epoch + 1}/ {25}"):
			images = images.to(device)

			# Zero gradient
			optimizer.zero_grad()

			# Here we compute loss
			with autocast():
				loss = loss_function(model, images)
			
			# Update parameters of our molde
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			# Collect losses
			epoch_loss += loss.item()
		
		# Simple logging
		print(f"Epoch: {epoch}| mse_loss: {epoch_loss / len(data_loader)}")