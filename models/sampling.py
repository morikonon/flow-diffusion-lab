import torch


# Euler integration of the learned flow-matching velocity field: x0 ~ N(0, 1) -> x1
@torch.no_grad()
def generate_image(model, num_samples=1, image_size=32, in_channels=3, steps=20, device="cpu", seed=None):
	model.eval()

	generator = torch.Generator(device=device)
	if seed is not None:
		generator.manual_seed(seed)

	x = torch.randn(num_samples, in_channels, image_size, image_size, device=device, generator=generator)
	dt = 1.0 / steps

	for step in range(steps):
		t = torch.full((num_samples,), step * dt, device=device)
		v_pred = model(x, t * 1000)
		x = x + v_pred * dt

	x = (x.clamp(-1, 1) + 1) / 2
	return x
