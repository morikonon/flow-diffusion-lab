import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10


def load_dataset(dataset_name: str = "CIFAR10", batch_size: int = 32, shuffle: bool = True):

	try:

		if dataset_name == "CIFAR10":

			transform = transforms.Compose([
				transforms.ToTensor()
			])

			dataset = CIFAR10(root="./", train=True, shuffle=True, transform=transform)
			data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

			print(f"Dataset: {dataset_name} is loaded| len: {len(data_loader)}")
		else:
			print(f"This {dataset_name} is not available yet!")
	except FileNotFoundError:
		print(f"File was not found!")

