import os

import gradio as gr
import numpy as np
import torch

from configs.configuration import config
from models.dit import DiT
from models.sampling import generate_image

# Two selectable models. Each maps to an optional checkpoint file; if the
# file doesn't exist yet, the app falls back to a freshly initialized
# (untrained) model so generation still works out of the box.
MODEL_CHOICES = {
	"Model A": "checkpoints/model_a.pt",
	"Model B": "checkpoints/model_b.pt",
}

_model_cache = {}


def get_model(name, image_size, in_channels, device):
	cache_key = (name, image_size, in_channels, device)
	if cache_key in _model_cache:
		return _model_cache[cache_key]

	model = DiT(image_size=image_size, in_channels=in_channels, config=config)

	checkpoint_path = MODEL_CHOICES[name]
	if os.path.exists(checkpoint_path):
		state_dict = torch.load(checkpoint_path, map_location=device)
		model.load_state_dict(state_dict)
	else:
		gr.Warning(f"No checkpoint found at '{checkpoint_path}' — generating with an untrained {name}.")

	model.to(device)
	model.eval()
	_model_cache[cache_key] = model
	return model


def tensor_to_image(x):
	# x: [1, C, H, W] in [0, 1]
	img = x[0].permute(1, 2, 0).cpu().numpy()
	return (img * 255).clip(0, 255).astype(np.uint8)


def run_generate(model_name, steps, seed, image_size, in_channels):
	device = config.device
	image_size = int(image_size)
	in_channels = int(in_channels)
	steps = int(steps)
	seed = int(seed)

	model = get_model(model_name, image_size, in_channels, device)
	image = generate_image(model, num_samples=1, image_size=image_size, in_channels=in_channels, steps=steps, device=device, seed=seed)
	return tensor_to_image(image)


def build_app():
	with gr.Blocks(title="Flow Diffusion Lab") as demo:
		gr.Markdown("# Flow Diffusion Lab\nPick a model and generate an image with the flow-matching DiT sampler.")

		model_name = gr.Dropdown(choices=list(MODEL_CHOICES.keys()), value="Model A", label="Model")

		with gr.Row():
			steps = gr.Slider(minimum=1, maximum=200, value=20, step=1, label="Sampling steps")
			seed = gr.Number(value=0, precision=0, label="Seed")
			image_size = gr.Number(value=32, precision=0, label="Image size")
			in_channels = gr.Number(value=3, precision=0, label="In channels")

		generate_btn = gr.Button("Generate", variant="primary")
		output_image = gr.Image(label="Generated image")

		generate_btn.click(
			run_generate,
			inputs=[model_name, steps, seed, image_size, in_channels],
			outputs=output_image,
		)

	return demo


if __name__ == "__main__":
	app = build_app()
	app.launch()
