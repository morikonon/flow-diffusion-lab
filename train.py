import argparse

import torch

from configs.configuration import config
from models.dit import DiT
from training.flow_training import training


def parse_args():
	parser = argparse.ArgumentParser(description="Train the flow-matching DiT on CIFAR10")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many optimizer steps, regardless of epochs")
	parser.add_argument("--subset-size", type=int, default=None, help="Train on only the first N images of CIFAR10, for quick runs")
	parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
	parser.add_argument("--checkpoint-name", type=str, default="dit")
	parser.add_argument("--image-size", type=int, default=32)
	parser.add_argument("--in-channels", type=int, default=3)
	parser.add_argument("--seed", type=int, default=None, help="Overrides configs.seed if set")
	return parser.parse_args()


def main():
	args = parse_args()

	seed = args.seed if args.seed is not None else config.seed
	torch.manual_seed(seed)

	model = DiT(image_size=args.image_size, in_channels=args.in_channels, config=config)

	checkpoint_path = training(
		model,
		config,
		epochs=args.epochs,
		max_steps=args.max_steps,
		checkpoint_dir=args.checkpoint_dir,
		checkpoint_name=args.checkpoint_name,
		subset_size=args.subset_size,
	)
	print(f"Training complete. Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
	main()
